// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZFreeBasisPairsValidation: EXPORT-Z step 2, package EXZ2-P1a (HeckeZFreeBasisPairsValidation.h).

#include "HeckeZFreeBasisPairsValidation.h"

#include "DescentAssembly.h"
#include "DescentHecke.h"
#include "IsoTreeCarrierZ.h"

#include "../IntegralCharpolyCrt.h"
#include "../Model/IsotropyModel.h"
#include "../Model/IsotropyModelTree.h"
#include "../../../ModArithmetic/EngineCheck.h"

#include <sys/stat.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fz = integral_descent::freez;
namespace h3 = integral_descent::hecke3;
using fz::BigInt;
using hecke::Mat22;
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
std::string Cell(uint32_t N, uint32_t n) { return "(" + std::to_string(N) + ", " + std::to_string(n) + ")"; }

bool IsDirectory(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
std::string RealPath(const std::string& p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) == nullptr) return std::string();
    return std::string(buf);
}
bool IsInside(const std::string& inner, const std::string& outer) {
    if (outer.empty() || inner.size() < outer.size()) return false;
    if (inner.compare(0, outer.size(), outer) != 0) return false;
    return inner.size() == outer.size() || inner[outer.size()] == '/';
}

// ---- exact rationals for the independent Gauss-Jordan (tiny cells only) -----------------------------------------------------------
struct Q {
    BigInt num = BigInt(0), den = BigInt(1);
};
Q Norm(BigInt a, BigInt b) {
    if (b.is_neg()) {
        a = -a;
        b = -b;
    }
    const BigInt g = BigInt::gcd(a, b);
    if (!(g == BigInt(1)) && !g.is_zero()) {
        a = a / g;
        b = b / g;
    }
    return Q{a, b};
}
Q Sub(const Q& x, const Q& y) { return Norm(x.num * y.den - y.num * x.den, x.den * y.den); }
Q Mul(const Q& x, const Q& y) { return Norm(x.num * y.num, x.den * y.den); }
Q Div(const Q& x, const Q& y) { return Norm(x.num * y.den, x.den * y.num); }

// rows x cols dense (exact integers) -> RREF over Q; returns the pivot columns
std::vector<uint32_t> RrefQ(std::vector<std::vector<Q>>& A) {
    const size_t R = A.size(), C = R ? A[0].size() : 0;
    std::vector<uint32_t> piv;
    size_t row = 0;
    for (size_t c = 0; c < C && row < R; ++c) {
        size_t p = row;
        while (p < R && A[p][c].num.is_zero()) ++p;
        if (p == R) continue;
        std::swap(A[row], A[p]);
        const Q inv = Div(Q{BigInt(1), BigInt(1)}, A[row][c]);
        for (auto& x : A[row]) x = Mul(x, inv);
        for (size_t i = 0; i < R; ++i) {
            if (i == row || A[i][c].num.is_zero()) continue;
            const Q f = A[i][c];
            for (size_t j = 0; j < C; ++j)
                if (!A[row][j].num.is_zero()) A[i][j] = Sub(A[i][j], Mul(f, A[row][j]));
        }
        piv.push_back(static_cast<uint32_t>(c));
        ++row;
    }
    return piv;
}
// the primitive integer vector of a rational vector with the given coordinate positive
std::vector<BigInt> Primitive(const std::vector<Q>& x, size_t pos_coord) {
    BigInt L(1);
    for (const auto& q : x) L = L / BigInt::gcd(L, q.den) * q.den;
    std::vector<BigInt> v(x.size());
    BigInt g(0);
    for (size_t i = 0; i < x.size(); ++i) {
        v[i] = x[i].num * (L / x[i].den);
        g = BigInt::gcd(g, v[i]);
    }
    if (!g.is_zero())
        for (auto& e : v) e = e / g;
    if (v[pos_coord].is_neg())
        for (auto& e : v) e = -e;
    return v;
}
std::vector<std::vector<BigInt>> DenseOf(const fz::SparseZ& A) {
    std::vector<std::vector<BigInt>> M(A.rows, std::vector<BigInt>(A.cols, BigInt(0)));
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) M[i][A.col[k]] = A.val[k];
    return M;
}

std::vector<uint32_t> RandomCols(std::mt19937_64& rng, size_t len, uint32_t q) {
    std::vector<uint32_t> v(len);
    for (auto& x : v) x = static_cast<uint32_t>(rng() % q);
    return v;
}

// ============================================================================================================================
// (C) the carrier
// ============================================================================================================================
void ArmCarrier() {
    std::cout << "ARM (C) the carrier Iso_T over Z" << std::endl;
    const auto t0 = clk::now();
    const uint64_t c0 = g_checks;
    // RhoNZ against rho_n
    {
        std::mt19937_64 rng(20261106);
        const uint32_t ps[] = {1000003u, fz::kQ0};
        std::vector<Mat22> gs = {Mat22::I(), Mat22::S(), Mat22::ST(), Mat22::T(), Mat22{0, -1, 1, 1}, Mat22{2, 0, 0, 1}, Mat22{1, 0, 0, 2},
                                 Mat22{0, 3, -1, 0}, Mat22{0, 1, 5, 7}, Mat22{3, -1, 7, -2}, Mat22{-4, 1, 11, 2}};
        for (int i = 0; i < 6; ++i) gs.push_back(Mat22{int64_t(rng() % 9) - 4, int64_t(rng() % 9) - 4, int64_t(rng() % 9) - 3, int64_t(rng() % 9) - 4});
        for (uint32_t n : {0u, 1u, 2u, 7u, 30u}) {
            for (const Mat22& g : gs) {
                if (g.a == 0 && g.c == 0) continue;
                const std::vector<BigInt> Z = fz::RhoNZ(g, n);
                for (uint32_t p : ps) {
                    const engine::ModCtx<uint32_t> ctx(p);
                    const engine::DenseBuf<uint32_t> R = integral_model::rho_n_cubic(ctx, g, n);
                    bool same = true;
                    for (uint32_t x = 0; x <= n; ++x)
                        for (uint32_t y = 0; y <= n; ++y) same = same && Z[size_t(x) * (n + 1) + y].mod_u32(p) == R.at(x, y);
                    check(same, "RhoNZ = rho_n mod " + std::to_string(p) + " at n = " + std::to_string(n));
                }
                // the majorant
                const std::vector<BigInt> A = fz::RhoNAbsZ(g, n);
                bool dom = true;
                for (size_t i = 0; i < A.size(); ++i) dom = dom && Z[i].abs() <= A[i];
                check(dom, "|rho_n(g)| <= rho_n(|g|) at n = " + std::to_string(n));
            }
        }
    }
    struct C {
        uint32_t N, n;
    };
    for (const C& cl : std::vector<C>{{1, 10}, {2, 6}, {11, 2}, {5, 12}, {9, 8}}) {
        const fz::IsoTreeCarrierZ car(cl.N, cl.n);
        const std::string tag = Cell(cl.N, cl.n);
        check(car.cells(0) == 2 && car.cells(1) == 3 && car.cells(2) == 3, tag + ": tree ranks 2, 3, 3");
        const integral_model::TreeModelChain tree(0, 3);
        for (uint32_t p : {1000003u, fz::kQ0}) {
            const engine::ModCtx<uint32_t> ctx(p);
            const integral_model::CoefficientModule<uint32_t> M = integral_descent::InducedSymModule(cl.N, cl.n, ctx);
            check(M.dim == car.D(), tag + ": the dense induced module has dim D");
            for (uint32_t q = 0; q < 2; ++q) {
                const engine::DenseBuf<uint32_t> Dq = tree.coboundary(ctx, M, q);
                const fz::SparseZ& S = car.d(q);
                bool same = Dq.rows == S.rows && Dq.cols == S.cols;
                std::vector<uint32_t> nzmap;
                if (same) {
                    std::vector<std::vector<uint32_t>> E(S.rows, std::vector<uint32_t>(S.cols, 0));
                    for (uint32_t i = 0; i < S.rows; ++i)
                        for (uint64_t k = S.rowptr[i]; k < S.rowptr[i + 1]; ++k) E[i][S.col[k]] = S.val[k].mod_u32(p);
                    for (uint32_t i = 0; i < S.rows && same; ++i)
                        for (uint32_t j = 0; j < S.cols && same; ++j) same = E[i][j] == Dq.at(i, j);
                }
                check(same, tag + ": d^" + std::to_string(q) + " over Z mod " + std::to_string(p) + " = TreeModelChain(0)::coboundary");
                // the lane route
                std::mt19937_64 rng(cl.N * 1000 + cl.n + q);
                const uint32_t k = 3;
                const std::vector<uint32_t> X = RandomCols(rng, size_t(S.cols) * k, p);
                std::vector<uint32_t> Y;
                car.d_mod(p, q, X, k, Y);
                bool lane = Y.size() == size_t(S.rows) * k;
                for (uint32_t i = 0; i < S.rows && lane; ++i)
                    for (uint32_t c = 0; c < k; ++c) {
                        uint64_t s = 0;
                        for (uint64_t t = S.rowptr[i]; t < S.rowptr[i + 1]; ++t) s = (s + uint64_t(S.val[t].mod_u32(p)) * X[size_t(S.col[t]) * k + c]) % p;
                        lane = lane && s == Y[size_t(i) * k + c];
                    }
                check(lane, tag + ": d^" + std::to_string(q) + " exact residues = the lane route d_mod");
            }
        }
        // d^1 d^0 = 0 exactly
        {
            const fz::SparseZ& A = car.d(1);
            const fz::SparseZ& B = car.d(0);
            bool zero = true;
            for (uint32_t i = 0; i < A.rows && zero; ++i) {
                std::map<uint32_t, BigInt> row;
                for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
                    const uint32_t mid = A.col[k];
                    for (uint64_t t = B.rowptr[mid]; t < B.rowptr[mid + 1]; ++t) row[B.col[t]] += A.val[k] * B.val[t];
                }
                for (const auto& kv : row) zero = zero && kv.second.is_zero();
            }
            check(zero, tag + ": d^1 d^0 = 0 over Z");
        }
        std::cout << "  CARRIER " << tag << " D=" << car.D() << " dim C^0,1,2=" << car.dim(0) << "," << car.dim(1) << "," << car.dim(2)
                  << " nnz(d0)=" << car.d(0).nnz() << " nnz(d1)=" << car.d(1).nnz() << " bits(d0)=" << car.d(0).max_bits()
                  << " bits(d1)=" << car.d(1).max_bits() << " b1=" << car.b1() << " build=" << std::fixed << std::setprecision(2)
                  << car.build_seconds << "s" << std::endl;
    }
    std::cout << "ARM (C) checks=" << g_checks - c0 << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
}

// ============================================================================================================================
// (T) the Hecke lanes
// ============================================================================================================================
void ArmHecke() {
    std::cout << "ARM (T) the transported operator on the carrier (lanes)" << std::endl;
    const auto t0 = clk::now();
    const uint64_t c0 = g_checks;
    struct C {
        uint32_t N, n, p;
    };
    for (const C& cl : std::vector<C>{{1, 10, 2}, {2, 6, 3}, {11, 2, 2}, {1, 12, 3}}) {
        const std::string tag = Cell(cl.N, cl.n) + " T_" + std::to_string(cl.p);
        const fz::IsoTreeCarrierZ car(cl.N, cl.n);
        const uint32_t q = 1000003u;
        const engine::ModCtx<uint32_t> ctx(q);
        const h3::IsoHeckeN T(cl.N, cl.n, cl.p, ctx, 2, true);
        if (!check(!T.refused(), tag + ": the operator is built (" + T.why() + ")")) continue;
        std::mt19937_64 rng(cl.N * 7919 + cl.n * 31 + cl.p);
        const uint32_t k = 2;
        // terms_Z reduced = the reduced terms (the same apply)
        for (uint32_t deg = 0; deg <= 2; ++deg) {
            check(T.terms_Z(deg).size() == T.terms(deg), tag + ": |terms_Z| = |terms| in degree " + std::to_string(deg));
            const uint32_t W = T.dim(deg);
            const std::vector<uint32_t> X = RandomCols(rng, size_t(W) * k, q);
            std::vector<uint32_t> Y, Z(size_t(W) * k, 0);
            T.apply(deg, X, k, Y);
            const uint32_t d = cl.n + 1;
            for (const auto& t : T.terms_Z(deg)) {
                int64_t mr = t.mul % int64_t(q);
                if (mr < 0) mr += q;
                const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx, t.E, cl.n);
                for (uint32_t x = 0; x < d; ++x)
                    for (uint32_t y = 0; y < d; ++y) {
                        const uint64_t f = uint64_t(mr) * R.at(x, y) % q;
                        for (uint32_t c = 0; c < k; ++c) {
                            uint32_t& z = Z[(size_t(t.rb) * d + x) * k + c];
                            z = static_cast<uint32_t>((z + f * X[(size_t(t.cb) * d + y) * k + c]) % q);
                        }
                    }
            }
            check(Y == Z, tag + ": terms_Z reduced mod q applies as IsoHeckeN::apply in degree " + std::to_string(deg));
        }
        // a cochain map: d^q T_q = T_{q+1} d^q, q = 0, 1
        for (uint32_t deg = 0; deg < 2; ++deg) {
            const std::vector<uint32_t> X = RandomCols(rng, size_t(car.dim(deg)) * k, q);
            std::vector<uint32_t> TX, dTX, dX, TdX;
            car.apply_hecke_mod(T, deg, X, k, TX);
            car.d_mod(q, deg, TX, k, dTX);
            car.d_mod(q, deg, X, k, dX);
            car.apply_hecke_mod(T, deg + 1, dX, k, TdX);
            check(dTX == TdX, tag + ": d^" + std::to_string(deg) + " T~_T = T~_T d^" + std::to_string(deg) + " mod q (a cochain map)");
        }
        // teeth: the BlockSign fault
        if (cl.N == 1 && cl.n == 10) {
            const h3::ScopedD3Fault f(h3::D3Fault::BlockSign);
            const h3::IsoHeckeN Tb(cl.N, cl.n, cl.p, ctx, 2);
            const std::vector<uint32_t> X = RandomCols(rng, size_t(car.dim(1)) * k, q);
            std::vector<uint32_t> TX, dTX, dX, TdX;
            car.apply_hecke_mod(Tb, 1, X, k, TX);
            car.d_mod(q, 1, TX, k, dTX);
            car.d_mod(q, 1, X, k, dX);
            car.apply_hecke_mod(Tb, 2, dX, k, TdX);
            check(dTX != TdX, tag + ": teeth: the BlockSign fault is no cochain map");
        }
        // the majorant dominates |T~_T X| on 0/1 columns (exact values = symmetric residues while below q/2)
        {
            const uint32_t nv = car.dim(1);
            std::vector<uint32_t> X(size_t(nv) * k, 0);
            std::vector<long double> Xa(size_t(nv) * k, 0);
            for (size_t i = 0; i < X.size(); ++i)
                if (rng() % 3 == 0) {
                    X[i] = 1;
                    Xa[i] = 1;
                }
            std::vector<uint32_t> TX;
            std::vector<long double> MX;
            car.apply_hecke_mod(T, 1, X, k, TX);
            car.apply_hecke_majorant(T, Xa, k, MX);
            bool dom = true, small = true;
            for (size_t i = 0; i < TX.size(); ++i) {
                if (MX[i] >= q / 2.0L) {
                    small = false;
                    continue;
                }
                const int64_t v = TX[i] > q / 2 ? int64_t(TX[i]) - q : int64_t(TX[i]);
                dom = dom && std::fabs(static_cast<long double>(v)) <= MX[i];
            }
            check(dom, tag + ": |T~_T X| <= the majorant (Psi^* sum |mul| rho_n(|E|) Phi^*) |X|");
            if (cl.n <= 10) check(small, tag + ": the majorant is below q/2 on this cell (the domination test is complete)");
        }
        std::cout << "  HECKE " << tag << " terms(0,1,2)=" << T.terms(0) << "," << T.terms(1) << "," << T.terms(2) << std::endl;
    }
    std::cout << "ARM (T) checks=" << g_checks - c0 << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
}

// ============================================================================================================================
// (D) the Dixon pairs
// ============================================================================================================================
bool CheckPairShape(const fz::IsoTreeCarrierZ& car, const fz::Pair& P, const std::string& tag) {
    const uint32_t nv = car.dim(1), r = car.b1();
    bool ok = P.B.cols == r && P.Phi.cols == r && P.P.size() == r && P.B.rows == nv && P.Phi.rows == nv;
    if (!ok) return check(false, tag + ": shapes");
    const auto PB = P.B.dense(), PPhi = P.Phi.dense();
    std::vector<uint8_t> inPc(nv, 0), inR0(nv, 0);
    for (uint32_t j : P.Pc) inPc[j] = 1;
    for (uint32_t j : P.R0) inR0[j] = 1;
    bool supp = true, pos = true, prim = true;
    for (uint32_t j = 0; j < r; ++j) {
        BigInt gb(0), gp(0);
        for (uint32_t i = 0; i < nv; ++i) {
            if (!PB[j][i].is_zero() && !inPc[i] && i != P.P[j]) supp = false;
            if (!PPhi[j][i].is_zero() && !inR0[i] && i != P.P[j]) supp = false;
            gb = BigInt::gcd(gb, PB[j][i]);
            gp = BigInt::gcd(gp, PPhi[j][i]);
        }
        pos = pos && PB[j][P.P[j]].sign() > 0 && PPhi[j][P.P[j]].sign() > 0;
        prim = prim && gb == BigInt(1) && gp == BigInt(1);
    }
    check(supp, tag + ": B_s on Pc + {P_j}, Phi_s on R0 + {P_i}");
    check(pos, tag + ": positive at P_j");
    check(prim, tag + ": primitive columns");
    check(fz::ExactCocycles(car, PB), tag + ": d^1 B_s = 0 (plain BigInt)");
    check(fz::ExactKillsD0(car, PPhi), tag + ": Phi_s^T d^0 = 0 (plain BigInt)");
    check(fz::PackedFnv(P.B) == fz::MatrixFnv(PB, nv) && P.fnv_B == fz::MatrixFnv(PB, nv) && P.fnv_Phi == fz::MatrixFnv(PPhi, nv),
          tag + ": PackedFnv = MatrixFnv of the dense columns (the canonical text is layout-free)");
    return supp && pos && prim;
}

void ArmDixon() {
    std::cout << "ARM (D) the Dixon pairs (" << fz::kBasisRule << ")" << std::endl;
    const auto t0 = clk::now();
    const uint64_t c0 = g_checks;
    struct C {
        uint32_t N, n;
        bool oracle;
    };
    for (const C& cl : std::vector<C>{{1, 10, true}, {2, 6, true}, {11, 2, true}, {11, 0, false}, {1, 30, false}, {9, 8, false}}) {
        const std::string tag = Cell(cl.N, cl.n);
        const fz::IsoTreeCarrierZ car(cl.N, cl.n);
        const fz::PairsResult R = fz::BuildPairs(car);
        if (!check(!R.refused, tag + ": the pairs are built (" + R.refusal + ")")) continue;
        check(R.rank_d1 + R.rank_d0 + R.r == car.dim(1), tag + ": rank d^1 + rank d^0 + b_1 = dim C^1");
        check(R.pairs.size() == fz::kPairsK, tag + ": K = 3 pairs");
        const uint32_t nv = car.dim(1), r = car.b1();
        for (const fz::Pair& P : R.pairs) {
            const std::string pt = tag + " s=" + std::to_string(P.s);
            CheckPairShape(car, P, pt);
            {
                // iso_dixon/v3 (alt2): rot(rho_t), reversed for s odd, rho = (0, floor(D/2), floor(D/4))[floor(s/2)]
                const uint32_t D = car.D(), rho[3] = {0u, D / 2, D / 4};
                check(P.rot == rho[P.s / 2] && P.reversed == (P.s % 2 == 1), pt + ": the iso_dixon/v3 order (rho_t, reversal for s odd)");
            }
            const auto PB = P.B.dense(), PPhi = P.Phi.dense();
            const BigInt det = fz::DetBareiss(fz::PairingMatrix(P.Phi, P.B), r);
            check(!det.is_zero(), pt + ": det(Phi_s^T B_s) != 0");
            if (cl.oracle) {
                // the independent exact rational solve: RREF of d^1[:, Pc | P] and of d^0[R0 | P, :]^T over Q
                const auto D1 = DenseOf(car.d(1));
                const auto D0 = DenseOf(car.d(0));
                const size_t k1 = P.Pc.size(), k0 = P.R0.size();
                std::vector<std::vector<Q>> A(D1.size(), std::vector<Q>(k1 + r));
                for (size_t i = 0; i < D1.size(); ++i) {
                    for (size_t t = 0; t < k1; ++t) A[i][t] = Q{D1[i][P.Pc[t]], BigInt(1)};
                    for (uint32_t j = 0; j < r; ++j) A[i][k1 + j] = Q{D1[i][P.P[j]], BigInt(1)};
                }
                const std::vector<uint32_t> piv = RrefQ(A);
                bool first = piv.size() == k1;
                for (size_t t = 0; t < piv.size() && first; ++t) first = piv[t] == t;
                check(first, pt + ": oracle: the columns Pc of d^1 are independent over Q and span the P columns");
                bool same = first;
                for (uint32_t j = 0; j < r && same; ++j) {
                    std::vector<Q> x(nv);
                    x[P.P[j]] = Q{BigInt(1), BigInt(1)};
                    for (size_t t = 0; t < k1; ++t) x[P.Pc[t]] = Norm(-A[t][k1 + j].num, A[t][k1 + j].den);
                    same = Primitive(x, P.P[j]) == PB[j];
                }
                check(same, pt + ": oracle: B_s = the primitive rational solutions (Gauss-Jordan over Q)");
                std::vector<std::vector<Q>> A0(car.dim(0), std::vector<Q>(k0 + r));
                for (size_t c = 0; c < car.dim(0); ++c) {
                    for (size_t t = 0; t < k0; ++t) A0[c][t] = Q{D0[P.R0[t]][c], BigInt(1)};
                    for (uint32_t i = 0; i < r; ++i) A0[c][k0 + i] = Q{D0[P.P[i]][c], BigInt(1)};
                }
                const std::vector<uint32_t> piv0 = RrefQ(A0);
                bool first0 = piv0.size() == k0;
                for (size_t t = 0; t < piv0.size() && first0; ++t) first0 = piv0[t] == t;
                check(first0, pt + ": oracle: the rows R0 of d^0 are independent over Q and span the P rows");
                bool same0 = first0;
                for (uint32_t i = 0; i < r && same0; ++i) {
                    std::vector<Q> x(nv);
                    x[P.P[i]] = Q{BigInt(1), BigInt(1)};
                    for (size_t t = 0; t < k0; ++t) x[P.R0[t]] = Norm(-A0[t][k0 + i].num, A0[t][k0 + i].den);
                    same0 = Primitive(x, P.P[i]) == PPhi[i];
                }
                check(same0, pt + ": oracle: Phi_s = the primitive rational solutions (Gauss-Jordan over Q)");
            }
        }
        // q0-independence: the pairs at the second CrtPrimes prime
        {
            fz::PairsOptions o;
            o.q0 = integral::CrtPrimes(2)[1];
            const fz::PairsResult R2 = fz::BuildPairs(car, o);
            bool same = !R2.refused && R2.pairs.size() == R.pairs.size();
            for (size_t s = 0; s < R.pairs.size() && same; ++s)
                same = R2.pairs[s].fnv_B == R.pairs[s].fnv_B && R2.pairs[s].fnv_Phi == R.pairs[s].fnv_Phi && R2.pairs[s].P == R.pairs[s].P;
            check(same, tag + ": q0-independence: the pairs at q0' = " + std::to_string(o.q0) + " have the same FNVs");
        }
        // teeth
        if (r > 0 && !R.pairs[0].Pc.empty()) {
            auto B = R.pairs[0].B.dense();
            B[0][R.pairs[0].Pc[0]] += BigInt(1);
            check(!fz::ExactCocycles(car, B), tag + ": teeth: b_1 + e_c is no cocycle");
            auto B7 = R.pairs[0].B.dense();
            BigInt g(0);
            for (auto& x : B7[0]) {
                x = x * BigInt(7);
                g = BigInt::gcd(g, x);
            }
            check(fz::ExactCocycles(car, B7) && !(g == BigInt(1)), tag + ": teeth: 7 b_1 is a cocycle but not primitive (the shape check sees it)");
        }
        std::ostringstream o;
        o << "  PAIRS " << tag << " r=" << r << " q0=" << R.q0;
        for (const fz::Pair& P : R.pairs)
            o << " | s=" << P.s << " steps(B,Phi)=(" << P.dixB.steps << "," << P.dixPhi.steps << ") bits(B,Phi)=(" << P.bits_B << ","
              << P.bits_Phi << ") fnvB=0x" << std::hex << P.fnv_B << std::dec;
        std::cout << o.str() << std::endl;
        // the pinned FNVs of the rule on (1, 10) (cross-checked by exz2p1a_check.py on the dump: the same rule run on the same complex)
        if (cl.N == 1 && cl.n == 10) {
            // s = 0: pinned 2026-11-06 under v2 (unchanged by v3: pair 0 is v2's pair 0); s = 1, 2: re-pinned for iso_dixon/v3 by EXZ2-P1b
            // (2026-11-06); exz2p1a_check.py (the model's pair() with the v3 order on the dumped complex) gave the same six values
            const uint64_t pinB[3] = {0xf43eff0900c718dcull, 0x54c48fffb07c6339ull, 0x607dd54be0aa9849ull};
            const uint64_t pinP[3] = {0xe525ff39dc2032b7ull, 0xf506b4d0b2dfaaf2ull, 0xe0795a7d096eb5daull};
            bool same = R.pairs.size() == 3;
            for (int s = 0; s < 3 && same; ++s) same = R.pairs[s].fnv_B == pinB[s] && R.pairs[s].fnv_Phi == pinP[s];
            check(same, tag + ": the pinned FNVs of iso_dixon/v3 (B_s, Phi_s, s = 0, 1, 2)");
        }
    }
    std::cout << "ARM (D) checks=" << g_checks - c0 << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
}

// ============================================================================================================================
// (R) the size cap
// ============================================================================================================================
void ArmRss() {
    std::cout << "ARM (R) the size cap RssCeiling" << std::endl;
    const uint64_t c0 = g_checks;
    const fz::IsoTreeCarrierZ car(1, 10);
    fz::PairsOptions o;
    o.rss_cap_bytes = 1;
    const fz::PairsResult R = fz::BuildPairs(car, o);
    check(R.refused && R.refusal.find("REFUSED [RssCeiling]") == 0 && R.pairs.empty(), "(1, 10): a 1-byte ceiling refuses [RssCeiling]: " + R.refusal);
    check(R.rss_pred > 0 && R.refusal.find(std::to_string(R.rss_pred)) != std::string::npos, "(1, 10): the refusal prints the predicted bytes");
    std::cout << "ARM (R) checks=" << g_checks - c0 << std::endl;
}

// ============================================================================================================================
// (M) measurement and dump
// ============================================================================================================================
struct MCell {
    uint32_t N, n;
    std::vector<uint32_t> ps;
};

void WriteMatrixCols(std::ostream& o, const std::vector<std::vector<BigInt>>& cols, uint32_t rows) {
    // row-major text, rows lines of cols entries
    for (uint32_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols.size(); ++j) o << (j ? " " : "") << cols[j][i].to_string();
        o << "\n";
    }
}
void WriteSparse(std::ostream& o, const fz::SparseZ& A) {
    o << A.rows << " " << A.cols << " " << A.nnz() << "\n";
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) o << i << " " << A.col[k] << " " << A.val[k].to_string() << "\n";
}

// exact H = Phi_0^T T~_T X (r x k) by lanes, lanes from the majorant's bits + 8, one held-out lane; returns bits(H) (or -1 on a mismatch)
int ExactPairingImage(const fz::IsoTreeCarrierZ& car, uint32_t p, const std::vector<std::vector<BigInt>>& Phi0,
                      const std::vector<std::vector<BigInt>>& X, uint32_t bound_bits, std::vector<BigInt>& H, double* lane_seconds,
                      uint32_t* lanes_out) {
    const uint32_t nv = car.dim(1), r = static_cast<uint32_t>(Phi0.size()), k = static_cast<uint32_t>(X.size());
    std::vector<uint32_t> lanes = fz::LanePrimes(uint64_t(bound_bits) + 8 + 2, fz::kQ0, car.N() * p);
    const uint32_t held = fz::LanePrimes(uint64_t(lanes.size() + 1) * 29, fz::kQ0, car.N() * p).back();
    std::vector<uint32_t> all = lanes;
    all.push_back(held);
    std::vector<std::vector<uint32_t>> res(all.size());
    double tl = 0;
    for (size_t l = 0; l < all.size(); ++l) {
        const uint32_t q = all[l];
        const engine::ModCtx<uint32_t> ctx(q);
        const h3::IsoHeckeN T(car.N(), car.n(), p, ctx, 1, true);
        ENGINE_CHECK_MSG(!T.refused(), "ExactPairingImage: the operator is refused");
        std::vector<uint32_t> Xq(size_t(nv) * k), Y;
        for (uint32_t i = 0; i < nv; ++i)
            for (uint32_t c = 0; c < k; ++c) Xq[size_t(i) * k + c] = X[c][i].is_zero() ? 0u : X[c][i].mod_u32(q);
        const auto ta = clk::now();
        car.apply_hecke_mod(T, 1, Xq, k, Y);
        tl += since(ta);
        std::vector<uint32_t> out(size_t(r) * k, 0);
        for (uint32_t a = 0; a < r; ++a) {
            std::vector<uint32_t> ph(nv);
            for (uint32_t i = 0; i < nv; ++i) ph[i] = Phi0[a][i].is_zero() ? 0u : Phi0[a][i].mod_u32(q);
            for (uint32_t c = 0; c < k; ++c) {
                uint64_t s = 0;
                for (uint32_t i = 0; i < nv; ++i)
                    if (ph[i]) s = (s + uint64_t(ph[i]) * Y[size_t(i) * k + c]) % q;
                out[size_t(a) * k + c] = static_cast<uint32_t>(s);
            }
        }
        res[l] = std::move(out);
    }
    if (lane_seconds) *lane_seconds = tl / all.size();
    if (lanes_out) *lanes_out = static_cast<uint32_t>(lanes.size());
    const fz::GarnerCtx G(lanes);
    H.assign(size_t(r) * k, BigInt(0));
    int bits = 0;
    for (size_t e = 0; e < H.size(); ++e) {
        std::vector<uint32_t> v(lanes.size());
        for (size_t l = 0; l < lanes.size(); ++l) v[l] = res[l][e];
        H[e] = fz::GarnerSigned(v, G);
        if (H[e].mod_u32(held) != res.back()[e]) return -1;
        bits = std::max<int>(bits, static_cast<int>(fz::BitsOf(H[e])));
    }
    return bits;
}

// the exact T~_T on C^1 (dense, row-major nv x nv) by lanes (measurement / dump, small cells only)
bool ExactHeckeDense(const fz::IsoTreeCarrierZ& car, uint32_t p, uint32_t bound_bits, std::vector<BigInt>& T) {
    const uint32_t nv = car.dim(1);
    std::vector<uint32_t> lanes = fz::LanePrimes(uint64_t(bound_bits) + 8 + 2, fz::kQ0, car.N() * p);
    const uint32_t held = fz::LanePrimes(uint64_t(lanes.size() + 1) * 29, fz::kQ0, car.N() * p).back();
    std::vector<uint32_t> all = lanes;
    all.push_back(held);
    std::vector<std::vector<uint32_t>> res(all.size());
    for (size_t l = 0; l < all.size(); ++l) {
        const engine::ModCtx<uint32_t> ctx(all[l]);
        const h3::IsoHeckeN Th(car.N(), car.n(), p, ctx, 1, true);
        std::vector<uint32_t> I(size_t(nv) * nv, 0);
        for (uint32_t i = 0; i < nv; ++i) I[size_t(i) * nv + i] = 1;
        car.apply_hecke_mod(Th, 1, I, nv, res[l]);
    }
    const fz::GarnerCtx G(lanes);
    T.assign(size_t(nv) * nv, BigInt(0));
    for (size_t e = 0; e < T.size(); ++e) {
        std::vector<uint32_t> v(lanes.size());
        for (size_t l = 0; l < lanes.size(); ++l) v[l] = res[l][e];
        T[e] = fz::GarnerSigned(v, G);
        if (T[e].mod_u32(held) != res.back()[e]) return false;
    }
    return true;
}

void MeasureCell(const MCell& mc, const std::string& dump_dir) {
    const std::string tag = Cell(mc.N, mc.n);
    const uint64_t rss0 = integral_descent::CurrentRssBytes();
    const auto t0 = clk::now();
    integral_descent::RssSampler samp;
    const fz::IsoTreeCarrierZ car(mc.N, mc.n);
    const double t_car = since(t0);
    const uint32_t nv = car.dim(1), r = car.b1();
    const uint64_t pred = fz::PredictPairsRss(car);                              // the packed / batched layout (P1b)
    const uint64_t pred_p1a = fz::PredictPairsRss(car, fz::kPairsK, false, 0);     // the audit's formula on the P1a layout
    const fz::PairsResult R = fz::BuildPairs(car);
    const uint64_t peak_pairs = samp.stop();
    if (!check(!R.refused, tag + ": measurement pairs built (" + R.refusal + ")")) return;
    for (const fz::Pair& P : R.pairs) CheckPairShape(car, P, tag + " s=" + std::to_string(P.s) + " (M)");
    std::ostringstream o;
    o << "EXZ2_P1A cell=" << mc.N << ":" << mc.n << " D=" << car.D() << " dimC=" << car.dim(0) << "/" << nv << "/" << car.dim(2) << " r=" << r
      << " rank_d1=" << R.rank_d1 << " rank_d0=" << R.rank_d0 << " nnz_d0=" << car.d(0).nnz() << " nnz_d1=" << car.d(1).nnz()
      << " bits_d0=" << car.d(0).max_bits() << " bits_d1=" << car.d(1).max_bits() << " q0=" << R.q0 << std::fixed << std::setprecision(3)
      << " t_carrier=" << t_car << " t_pairs=" << R.seconds << " rss_pred_MiB=" << (pred >> 20) << " rss_pred_p1a_layout_MiB=" << (pred_p1a >> 20) << " rss_meas_growth_MiB="
      << ((peak_pairs > rss0 ? peak_pairs - rss0 : 0) >> 20) << " rss_peak_MiB=" << (peak_pairs >> 20);
    std::cout << o.str() << std::endl;
    std::vector<uint32_t> detbits;
    for (const fz::Pair& P : R.pairs) {
        const auto td = clk::now();
        const BigInt det = fz::DetBareiss(fz::PairingMatrix(P.Phi, P.B), r);
        detbits.push_back(fz::BitsOf(det));
        std::cout << "EXZ2_P1A_PAIR cell=" << mc.N << ":" << mc.n << " s=" << P.s << " rot=" << P.rot << " |Pc|=" << P.Pc.size() << " |R0|=" << P.R0.size()
                  << " steps_B=" << P.dixB.steps << " steps_Phi=" << P.dixPhi.steps << " attempts=" << P.dixB.attempts << "/" << P.dixPhi.attempts
                  << " col_steps=" << P.dixB.col_steps << "/" << P.dixPhi.col_steps << " lanes=" << P.dixB.lanes << "/" << P.dixPhi.lanes << " verify_lanes=" << P.dixB.verify_lanes << "/" << P.dixPhi.verify_lanes
                  << " bits_B=" << P.bits_B << " bits_Phi=" << P.bits_Phi << " bits_den=" << P.bits_denB << "/" << P.bits_denPhi
                  << " det_bits=" << detbits.back() << " fill_d1=" << P.fill_d1 << " tnnz_d1=" << P.tnnz_d1 << " fill_d0=" << P.fill_d0
                  << " tnnz_d0=" << P.tnnz_d0 << " support_Phi=" << P.support_Phi << std::fixed << std::setprecision(3) << " t_elim=" << P.t_elim
                  << " t_dixB=" << P.dixB.seconds << " t_dixPhi=" << P.dixPhi.seconds << " t_det=" << since(td) << std::hex << " fnvB=0x"
                  << P.fnv_B << " fnvPhi=0x" << P.fnv_Phi << std::dec << std::endl;
    }
    // the Hecke side (P2's figures): terms(1), the majorant's bits of |Phi_0|^T M_T |B_s|, the exact H_s = Phi_0^T T~ B_s
    const auto Phi0 = R.pairs[0].Phi.dense();
    std::ofstream dumpH;
    if (!dump_dir.empty()) dumpH.open(dump_dir + "/H.txt");
    for (uint32_t p : mc.ps) {
        if (mc.N % p == 0) continue;
        const auto th = clk::now();
        const engine::ModCtx<uint32_t> ctx(1000003u);
        const h3::IsoHeckeN T(mc.N, mc.n, p, ctx, 1, true);
        const double t_build = since(th);
        if (!check(!T.refused(), tag + " T_" + std::to_string(p) + ": built (" + T.why() + ")")) continue;
        for (const fz::Pair& P : R.pairs) {
            const auto PB = P.B.dense();
            // majorant |Phi_0|^T M_T |B_s|
            std::vector<long double> Xa(size_t(nv) * r, 0), MX;
            for (uint32_t i = 0; i < nv; ++i)
                for (uint32_t j = 0; j < r; ++j) Xa[size_t(i) * r + j] = std::fabs(fz::ToLongDouble(PB[j][i]));
            const auto tm = clk::now();
            car.apply_hecke_majorant(T, Xa, r, MX);
            long double mx = 0;
            for (uint32_t a = 0; a < r; ++a)
                for (uint32_t c = 0; c < r; ++c) {
                    long double s = 0;
                    for (uint32_t i = 0; i < nv; ++i)
                        if (!Phi0[a][i].is_zero()) s += std::fabs(fz::ToLongDouble(Phi0[a][i])) * MX[size_t(i) * r + c];
                    mx = std::max(mx, s);
                }
            const double t_maj = since(tm);
            const uint32_t bbits = mx > 0 ? static_cast<uint32_t>(std::ceil(std::log2(mx))) + 1 : 1;
            std::vector<BigInt> H;
            double tl = 0;
            uint32_t nl = 0;
            const int hbits = ExactPairingImage(car, p, Phi0, PB, bbits, H, &tl, &nl);
            check(hbits >= 0, tag + " T_" + std::to_string(p) + " s=" + std::to_string(P.s) + ": H_s by lanes agrees with the held-out lane");
            std::cout << "EXZ2_P1A_HECKE cell=" << mc.N << ":" << mc.n << " p=" << p << " s=" << P.s << " terms1=" << T.terms(1) << " beta_H_bits=" << bbits
                      << " bits_H=" << hbits << " lanes_H=" << nl << std::fixed << std::setprecision(3) << " t_op_build=" << t_build
                      << " t_majorant=" << t_maj << " t_lane_apply_r_cols=" << tl << std::endl;
            if (dumpH.is_open()) {
                dumpH << "p " << p << " s " << P.s << " " << r << " " << r << "\n";
                for (uint32_t a = 0; a < r; ++a) {
                    for (uint32_t c = 0; c < r; ++c) dumpH << (c ? " " : "") << H[size_t(a) * r + c].to_string();
                    dumpH << "\n";
                }
            }
        }
        // the dense exact T~_T on C^1 for small cells (the Python model's a-priori bound uses |T|)
        if (!dump_dir.empty() && nv <= 420) {
            std::vector<long double> I(size_t(nv) * nv, 0), MI;
            for (uint32_t i = 0; i < nv; ++i) I[size_t(i) * nv + i] = 1;
            car.apply_hecke_majorant(T, I, nv, MI);
            long double mx = 0;
            for (auto v : MI) mx = std::max(mx, v);
            const uint32_t bbits = mx > 0 ? static_cast<uint32_t>(std::ceil(std::log2(mx))) + 1 : 1;
            std::vector<BigInt> Td;
            const bool ok = ExactHeckeDense(car, p, bbits, Td);
            check(ok, tag + " T_" + std::to_string(p) + ": the dense T~_T by lanes agrees with the held-out lane");
            std::ofstream f(dump_dir + "/T_" + std::to_string(p) + ".txt");
            f << nv << " " << nv << "\n";
            for (uint32_t i = 0; i < nv; ++i) {
                for (uint32_t j = 0; j < nv; ++j) f << (j ? " " : "") << Td[size_t(i) * nv + j].to_string();
                f << "\n";
            }
            std::ofstream g(dump_dir + "/Tmaj_" + std::to_string(p) + ".txt");
            g << nv << " " << nv << "\n";
            for (uint32_t i = 0; i < nv; ++i) {
                for (uint32_t j = 0; j < nv; ++j) {
                    const long double v = MI[size_t(i) * nv + j];
                    // an integer upper bound: ceil(v (1 + 2^-40)) + 1 when non-zero
                    std::ostringstream s;
                    if (v == 0) s << 0;
                    else s << std::fixed << std::setprecision(0) << std::ceil(v * (1.0L + std::ldexp(1.0L, -40))) + 1;
                    g << (j ? " " : "") << s.str();
                }
                g << "\n";
            }
        }
    }
    if (!dump_dir.empty()) {
        std::ofstream m(dump_dir + "/meta.txt");
        m << "N " << mc.N << "\nn " << mc.n << "\nD " << car.D() << "\nr " << r << "\nq0 " << R.q0 << "\nK " << R.pairs.size() << "\nrule " << fz::kBasisRule
          << "\nps";
        for (uint32_t p : mc.ps) m << " " << p;
        m << "\n";
        std::ofstream f0(dump_dir + "/d0.txt"), f1(dump_dir + "/d1.txt");
        WriteSparse(f0, car.d(0));
        WriteSparse(f1, car.d(1));
        for (const fz::Pair& P : R.pairs) {
            std::ofstream f(dump_dir + "/pair_" + std::to_string(P.s) + ".txt");
            f << "rot " << P.rot << "\nreversed " << (P.reversed ? 1 : 0) << "\nsteps " << P.dixB.steps << " " << P.dixPhi.steps << "\nfnv " << std::hex << P.fnv_B << " " << P.fnv_Phi << std::dec
              << "\nP";
            for (uint32_t j : P.P) f << " " << j;
            f << "\nPc";
            for (uint32_t j : P.Pc) f << " " << j;
            f << "\nR0";
            for (uint32_t j : P.R0) f << " " << j;
            f << "\nB " << nv << " " << r << "\n";
            WriteMatrixCols(f, P.B.dense(), nv);
            f << "Phi " << nv << " " << r << "\n";
            WriteMatrixCols(f, P.Phi.dense(), nv);
        }
        std::cout << "  DUMP " << tag << " -> " << dump_dir << std::endl;
    }
}

uint32_t ArmMeasure() {
    std::cout << "ARM (M) measurement" << std::endl;
    const uint64_t c0 = g_checks;
    std::vector<MCell> cells = {{1, 10, {2, 3}}, {11, 2, {2, 3}}};
    std::string arm = "default";
    if (const char* e = std::getenv("HECKE_EXZ2_CELLS")) {
        if (!*e) {
            std::cout << "HeckeZFreeBasisPairsValidation: REFUSED: HECKE_EXZ2_CELLS: empty value (unset it for the default)\n";
            return 1;
        }
        cells.clear();
        arm = "cells";
        std::stringstream ss(e);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            unsigned N = 0, n = 0, p1 = 0, p2 = 0;
            const int got = std::sscanf(tok.c_str(), "%u:%u:%u/%u", &N, &n, &p1, &p2);
            if (got < 2 || N == 0 || n % 2 || n > 400) {
                std::cout << "HeckeZFreeBasisPairsValidation: REFUSED: HECKE_EXZ2_CELLS: bad entry '" << tok << "' (N:n[:p1/p2], n even <= 400)\n";
                return 1;
            }
            MCell c{N, n, {}};
            if (got >= 3) c.ps.push_back(p1);
            if (got >= 4) c.ps.push_back(p2);
            if (c.ps.empty()) c.ps = {2, 3};
            cells.push_back(c);
        }
    }
    std::string dump_root;
    if (const char* d = std::getenv("HECKE_DUMP_EXZ2")) {
        const std::string dir(d);
        if (!IsDirectory(dir)) {
            std::cout << "HeckeZFreeBasisPairsValidation: REFUSED: HECKE_DUMP_EXZ2=" << dir << " is not an existing directory\n";
            return 1;
        }
#ifdef OC_SOURCE_DIR
        const std::string real_src = RealPath(OC_SOURCE_DIR);
#else
        const std::string real_src;
#endif
        const std::string real = RealPath(dir);
        if (real.empty() || IsInside(real, real_src)) {
            std::cout << "HeckeZFreeBasisPairsValidation: REFUSED: HECKE_DUMP_EXZ2=" << dir << " lies inside the source tree; dump into scratch only\n";
            return 1;
        }
        dump_root = real;
    }
    std::cout << "EFFECTIVE HeckeZFreeBasisPairsValidation arm=" << arm << " cells=" << cells.size() << " dump=" << (dump_root.empty() ? "off" : "on")
              << std::endl;
    for (const MCell& c : cells) {
        std::string dd;
        if (!dump_root.empty()) {
            dd = dump_root + "/c" + std::to_string(c.N) + "_" + std::to_string(c.n);
            ::mkdir(dd.c_str(), 0755);
        }
        MeasureCell(c, dd);
    }
    std::cout << "ARM (M) checks=" << g_checks - c0 << std::endl;
    return 0;
}

}  // namespace

uint32_t HeckeZFreeBasisPairsValidation() {
    std::cout << "HeckeZFreeBasisPairsValidation: EXPORT-Z step 2 P1a -- the carrier Iso_T over Z and the Dixon pairs (" << fz::kBasisRule << ")"
              << std::endl;
    const auto t0 = clk::now();
    g_fail = 0;
    g_checks = 0;
    ArmCarrier();
    ArmHecke();
    ArmDixon();
    ArmRss();
    const uint32_t refused = ArmMeasure();
    const bool ok = g_fail == 0 && refused == 0;
    std::cout << "HECKEZ_FREEBASIS_PAIRS arms=5 failing=" << g_fail << " checks=" << g_checks << " rule=" << fz::kBasisRule
              << " verdict=" << (ok ? "OK" : "FAIL") << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
    return ok ? 0 : 1;
}
