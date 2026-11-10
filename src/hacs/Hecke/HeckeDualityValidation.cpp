// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 6, WP-1 (phase E7a) -- see HeckeDualityValidation.h.
// [C] = docs/latex/gl2_contract.tex, cited by label.

#include "HeckeDualityValidation.h"
#include "../../ModArithmetic/LegacyConfig.h"   // (S24 CL-L2) RunLegacyArm (SKIP-ARM)

#include "ExactInt.h"
#include "HeckeBoundaryDatum.h"
#include "HeckeBoundaryOperator.h"
#include "HeckeConeAction.h"
#include "HeckeContragredient.h"
#include "HeckeInvolution.h"
#include "HeckeOperatorOnK.h"
#include "HeckePairHomotopy.h"
#include "HeckePairing.h"
#include "../HeckeExt/HeckeIdentityCertificate.h"
#include "../EquivariantMappingCone.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemUtils.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../ModArithmetic/ScalarMod.h"

namespace {

using hecke::Mat22;
using ZMat = std::vector<std::vector<int64_t>>;

struct Tally {
    const char* arm;
    uint64_t checks = 0, failures = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            if (failures <= 10) std::cout << "  [" << arm << "] FAILED: " << what << "\n";
        }
    }
    uint32_t report() const {
        std::cout << "  " << arm << ": " << checks << " checks, " << failures << " failed\n";
        return static_cast<uint32_t>(failures);
    }
};

// ---- exact integer helpers ------------------------------------------------

int64_t add_ck(int64_t a, int64_t b) {
    int64_t r;
    if (__builtin_add_overflow(a, b, &r)) throw std::overflow_error("HeckeDualityValidation: int64 add overflow");
    return r;
}
int64_t mul_ck(int64_t a, int64_t b) {
    int64_t r;
    if (__builtin_mul_overflow(a, b, &r)) throw std::overflow_error("HeckeDualityValidation: int64 mul overflow");
    return r;
}

ZMat zzero(uint32_t r, uint32_t c) { return ZMat(r, std::vector<int64_t>(c, 0)); }
ZMat zid(uint32_t d) { ZMat I = zzero(d, d); for (uint32_t i = 0; i < d; ++i) I[i][i] = 1; return I; }
ZMat zmul(const ZMat& A, const ZMat& B) {
    const uint32_t r = static_cast<uint32_t>(A.size()), k = static_cast<uint32_t>(B.size());
    const uint32_t c = k ? static_cast<uint32_t>(B[0].size()) : 0;
    ZMat C = zzero(r, c);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t t = 0; t < k; ++t)
            for (uint32_t j = 0; j < c; ++j) C[i][j] = add_ck(C[i][j], mul_ck(A[i][t], B[t][j]));
    return C;
}
ZMat ztr(const ZMat& A) {
    const uint32_t r = static_cast<uint32_t>(A.size()), c = r ? static_cast<uint32_t>(A[0].size()) : 0;
    ZMat T = zzero(c, r);
    for (uint32_t i = 0; i < r; ++i) for (uint32_t j = 0; j < c; ++j) T[j][i] = A[i][j];
    return T;
}
ZMat zscale(const ZMat& A, int64_t s) {
    ZMat B = A;
    for (auto& row : B) for (auto& x : row) x = mul_ck(x, s);
    return B;
}
int64_t zpow(int64_t b, uint32_t e) { int64_t r = 1; for (uint32_t i = 0; i < e; ++i) r = mul_ck(r, b); return r; }

// Polynomial product in y (coefficient lists, low -> high).
std::vector<int64_t> pmul(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    std::vector<int64_t> c(a.size() + b.size() - 1, 0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j) c[i + j] = add_ck(c[i + j], mul_ck(a[i], b[j]));
    return c;
}

// rho_n(A) over Z, the closed form [C, eq:rhondef]: column j holds the
// coefficients of (a + c y)^{n-j} (b + d y)^j.  INDEPENDENT of
// ComputeSymmetricPowerMatrixFM (no shared code).
ZMat rho_Z(const Mat22& A, uint32_t n) {
    ZMat R = zzero(n + 1, n + 1);
    for (uint32_t j = 0; j <= n; ++j) {
        std::vector<int64_t> poly{ 1 };
        for (uint32_t s = 0; s < n - j; ++s) poly = pmul(poly, { A.a, A.c });
        for (uint32_t s = 0; s < j; ++s) poly = pmul(poly, { A.b, A.d });
        for (uint32_t t = 0; t <= n; ++t) R[t][j] = (t < poly.size()) ? poly[t] : 0;
    }
    return R;
}
// rho_n^vee(A) := rho_n(adj A)^T over Z ([C, def:contragredient]), with the
// adjugate spelled out HERE (not Mat22::adj, which A1 tests).
ZMat rhodual_Z(const Mat22& A, uint32_t n) {
    const Mat22 adjA{ A.d, -A.b, -A.c, A.a };
    return ztr(rho_Z(adjA, n));
}

uint32_t modz(int64_t x, uint32_t ell) {
    int64_t r = x % static_cast<int64_t>(ell);
    if (r < 0) r += ell;
    return static_cast<uint32_t>(r);
}
FlatMatrix zmod(const ZMat& A, uint32_t ell) {
    const uint32_t r = static_cast<uint32_t>(A.size()), c = r ? static_cast<uint32_t>(A[0].size()) : 0;
    FlatMatrix F = FlatMatrix::CreateZero(r, c, ell);
    for (uint32_t i = 0; i < r; ++i) for (uint32_t j = 0; j < c; ++j) F.set_entry(i, j, modz(A[i][j], ell));
    return F;
}
bool same_flat(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols() || A.get_p() != B.get_p()) return false;
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            if (A.get_entry(i, j) != B.get_entry(i, j)) return false;
    return true;
}
FlatMatrix fm22(const Mat22& g, uint32_t ell) {
    const std::array<uint32_t, 4> m = g.reduced_mod(ell);
    return FlatMatrix(2, 2, ell, std::vector<uint32_t>{ m[0], m[1], m[2], m[3] });
}

// Deterministic seeded matrices: the model generators, diag(1,p), diag(p,1),
// singular and negative-determinant ones, plus an LCG batch with entries in
// [-5, 5].
std::vector<Mat22> seeded_matrices(uint32_t n_random) {
    std::vector<Mat22> v = { Mat22::I(), Mat22::T(), Mat22::S(), Mat22::ST(), Mat22::TS(), Mat22::I().neg(),
                             Mat22{ 1, 0, 0, 2 }, Mat22{ 2, 0, 0, 1 }, Mat22{ 1, 0, 0, 7 }, Mat22{ 3, 1, 0, 1 },
                             Mat22{ 2, 3, 4, 6 }, Mat22{ 0, 0, 0, 0 }, Mat22{ 0, 1, 0, 0 }, Mat22{ 3, 5, 1, 2 },
                             Mat22{ 2, 1, 1, -1 }, Mat22{ -4, 3, 5, 2 } };
    uint64_t s = 0x5eed0e7aULL;
    auto next = [&s]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<int64_t>((s >> 33) % 11) - 5; };
    for (uint32_t k = 0; k < n_random; ++k) v.push_back(Mat22{ next(), next(), next(), next() });
    return v;
}

// ---- A1: the adjugate -------------------------------------------------------
uint32_t arm_adj() {
    Tally t{ "A1 adj [C, eq:adjrules]" };
    const std::vector<Mat22> gs = seeded_matrices(24);
    for (const Mat22& g : gs) {
        const Mat22 a = g.adj();
        const int64_t dt = g.det();
        const Mat22 dI{ dt, 0, 0, dt };
        t.check(g * a == dI, "g adj g = det I");
        t.check(a * g == dI, "adj g g = det I");
        t.check(a.adj() == g, "adj adj g = g");
        if (dt == 1) t.check(a == g.inv_unimodular(), "adj gamma = gamma^{-1} on SL(2,Z)");
        for (const Mat22& h : gs) t.check((g * h).adj() == h.adj() * g.adj(), "adj(gh) = adj(h) adj(g)");
    }
    t.check(Mat22::I().adj() == Mat22::I(), "adj I = I");
    return t.report();
}

// ---- A2: rho_n^vee --------------------------------------------------------
uint32_t arm_rhodual() {
    Tally t{ "A2 rho^vee [C, def:contragredient], [C, prop:pairing](1)" };
    const std::vector<Mat22> gs = seeded_matrices(10);
    const uint32_t ells[] = { 5, 7, 11, 13, 101, 134217757u };
    for (uint32_t n = 0; n <= 6; ++n) {
        for (const Mat22& g : gs) {
            const ZMat Rg = rho_Z(g, n), Dg = rhodual_Z(g, n);
            for (uint32_t ell : ells) {
                const FlatMatrix gf = fm22(g, ell);
                t.check(same_flat(ComputeSymmetricPowerMatrixFM(gf, n), zmod(Rg, ell)),
                        "ComputeSymmetricPowerMatrixFM = exact rho_n mod ell (convention pin)");
                t.check(same_flat(ComputeContragredientPowerMatrixFM(gf, n), zmod(Dg, ell)),
                        "ComputeContragredientPowerMatrixFM = exact rho_n(adj g)^T mod ell");
                t.check(same_flat(ComputeFibreMatrixFM(FibreRepresentation::SymN, gf, n),
                                  ComputeSymmetricPowerMatrixFM(gf, n)),
                        "the SymN fibre builder IS ComputeSymmetricPowerMatrixFM");
            }
            if (g.det() == 1) t.check(Dg == ztr(rho_Z(g.inv_unimodular(), n)), "rho^vee(gamma) = rho(gamma^{-1})^T over Z");
            for (size_t k = 0; k < 8; ++k) {
                const Mat22& h = gs[(k * 5 + 3) % gs.size()];
                t.check(rhodual_Z(g * h, n) == zmul(Dg, rhodual_Z(h, n)), "rho^vee(gh) = rho^vee(g) rho^vee(h) over Z");
            }
        }
        t.check(rhodual_Z(Mat22::I(), n) == zid(n + 1), "rho^vee(I) = I");
    }
    // mod ell, larger n, singular g (det = 0 mod ell) included.
    const uint32_t ns[] = { 10, 24, 41 };
    for (uint32_t ell : { 5u, 7u, 101u }) {
        for (uint32_t n : ns) {
            for (size_t k = 0; k < gs.size(); k += 2) {
                const Mat22& g = gs[k];
                const Mat22& h = gs[(k * 7 + 1) % gs.size()];
                const FlatMatrix lhs = ComputeContragredientPowerMatrixFM(fm22(g * h, ell), n);
                const FlatMatrix rhs = ComputeContragredientPowerMatrixFM(fm22(g, ell), n) *
                                       ComputeContragredientPowerMatrixFM(fm22(h, ell), n);
                t.check(same_flat(lhs, rhs), "rho^vee(gh) = rho^vee(g) rho^vee(h) mod ell");
            }
        }
    }
    return t.report();
}

// ---- A3: the pairing over Z -------------------------------------------------
uint32_t arm_pairing() {
    Tally t{ "A3 pairing [C, prop:pairing](2)-(4)" };
    const std::vector<Mat22> gs = seeded_matrices(10);
    uint64_t s = 0xfeedULL;
    auto next = [&s]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<int64_t>((s >> 33) % 9) - 4; };
    for (uint32_t n = 0; n <= 6; ++n) {
        const uint32_t d = n + 1;
        for (uint32_t ell : { 7u, 13u }) t.check(same_flat(hecke::PairingGram(n, ell), zmod(zid(d), ell)), "Gram = I");
        for (const Mat22& g : gs) {
            const Mat22 adjg{ g.d, -g.b, -g.c, g.a };
            const ZMat Rg = rho_Z(g, n), Dadj = rhodual_Z(adjg, n), Dg = rhodual_Z(g, n);
            const int64_t detn = zpow(g.det(), n);
            for (uint32_t trial = 0; trial < 3; ++trial) {
                ZMat v = zzero(d, 1), phi = zzero(d, 1);
                for (uint32_t j = 0; j < d; ++j) { v[j][0] = next(); phi[j][0] = next(); }
                auto pair = [](const ZMat& a, const ZMat& b) {   // <a, b> = sum_j a_j b_j (Gram = I)
                    int64_t acc = 0;
                    for (size_t j = 0; j < a.size(); ++j) acc = add_ck(acc, mul_ck(a[j][0], b[j][0]));
                    return acc;
                };
                t.check(pair(zmul(Rg, v), phi) == pair(v, zmul(Dadj, phi)),
                        "<rho(g) v, phi> = <v, rho^vee(adj g) phi> (no determinant factor)");
                t.check(pair(zmul(Rg, v), zmul(Dg, phi)) == mul_ck(detn, pair(v, phi)),
                        "<rho(g) v, rho^vee(g) phi> = det(g)^n <v, phi>");
                const FlatMatrix vf = zmod(v, 13), pf = zmod(phi, 13);
                t.check(hecke::EvaluationPairing(vf, pf) == modz(pair(v, phi), 13), "EvaluationPairing mod ell");
            }
        }
    }
    return t.report();
}

// ---- A4: Phi_n --------------------------------------------------------------
// B_n(u_t, u_j) from the PERMUTATION definition of [C, lem:symdual].
int64_t Bn_perm(uint32_t n, uint32_t t, uint32_t j) {
    // u_t = x^{n-t} y^t as the word v_1..v_n (x's first); x = (1,0), y = (0,1).
    std::vector<int> v(n), w(n);
    for (uint32_t s = 0; s < n; ++s) { v[s] = (s < n - t) ? 0 : 1; w[s] = (s < n - j) ? 0 : 1; }
    auto det2 = [](int a, int b) -> int64_t {   // det[e_a | e_b]
        if (a == b) return 0;
        return (a == 0) ? 1 : -1;
    };
    std::vector<uint32_t> pi(n);
    std::iota(pi.begin(), pi.end(), 0u);
    int64_t sum = 0;
    do {
        int64_t prod = 1;
        for (uint32_t s = 0; s < n && prod != 0; ++s) prod *= det2(v[s], w[pi[s]]);
        sum += prod;
    } while (std::next_permutation(pi.begin(), pi.end()));
    return sum;
}
int64_t factorial(uint32_t k) { int64_t f = 1; for (uint32_t i = 2; i <= k; ++i) f = mul_ck(f, i); return f; }

uint32_t arm_phi() {
    Tally t{ "A4 Phi_n [C, lem:symdual]" };
    const std::vector<Mat22> gs = seeded_matrices(12);
    for (uint32_t n = 0; n <= 6; ++n) {
        const uint32_t d = n + 1;
        ZMat G = zzero(d, d);
        for (uint32_t a = 0; a < d; ++a)
            for (uint32_t b = 0; b < d; ++b) {
                G[a][b] = Bn_perm(n, a, b);
                const int64_t want = (a + b == n) ? ((a & 1u) ? -1 : 1) * factorial(a) * factorial(n - a) : 0;
                t.check(G[a][b] == want, "B_n Gram (permutation definition) = eq:BnGram");
            }
        const ZMat Phi = ztr(G);   // Phi_n(v) = B_n(v, .): column t = (B_n(u_t, u_j))_j
        for (uint32_t ell : { 5u, 7u, 13u, 101u })
            t.check(same_flat(hecke::PhiNMatrix(n, ell), zmod(Phi, ell)), "PhiNMatrix = Gram^T mod ell");
        for (const Mat22& g : gs) {
            const ZMat Rg = rho_Z(g, n);
            const Mat22 adjg{ g.d, -g.b, -g.c, g.a };
            t.check(zmul(zmul(ztr(Rg), G), Rg) == zscale(G, zpow(g.det(), n)), "B_n(rho v, rho w) = det^n B_n(v, w)");
            t.check(zmul(ztr(Rg), G) == zmul(G, rho_Z(adjg, n)), "B_n(rho(g) v, w) = B_n(v, rho(adj g) w)");
            t.check(zmul(Phi, Rg) == zmul(rhodual_Z(g, n), Phi), "Phi rho(g) = rho^vee(g) Phi over Z (no twist)");
        }
    }
    // mod ell: the intertwining at larger n (every g, singular included) and the rank criterion.
    for (uint32_t ell : { 5u, 7u, 13u, 101u }) {
        for (uint32_t n : { 4u, 10u, 12u, 22u, 40u }) {
            const FlatMatrix Phi = hecke::PhiNMatrix(n, ell);
            for (const Mat22& g : gs) {
                const FlatMatrix gf = fm22(g, ell);
                t.check(same_flat(Phi * ComputeSymmetricPowerMatrixFM(gf, n), ComputeContragredientPowerMatrixFM(gf, n) * Phi),
                        "Phi rho(g) = rho^vee(g) Phi mod ell");
            }
        }
    }
    for (uint32_t ell : { 2u, 3u, 5u, 7u, 11u, 13u, 31u }) {
        for (uint32_t n = 0; n <= 40; ++n) {
            const bool full = hecke::PhiNMatrix(n, ell).rank() == n + 1;
            t.check(full == (n < ell), "rank(Phi_n (x) F_ell) = n+1 iff n < ell");
            t.check(full == hecke::PhiNIsIsomorphism(n, ell), "PhiNIsIsomorphism agrees with the rank");
        }
    }
    return t.report();
}

// ---- the cells ---------------------------------------------------------------
struct CellSpec { uint32_t n, ell; };
const std::vector<CellSpec>& cells() {
    // n < ell (Phi_n an isomorphism) and n >= ell (not); even n (odd n
    // vanishes on this model, the -I gap).  Includes n = 0 (H^0 != 0) and the
    // Dickson locus (6, 5).
    static const std::vector<CellSpec> c = { { 0, 7 }, { 4, 7 }, { 10, 13 }, { 12, 13 }, { 16, 101 }, { 22, 29 },
                                             { 6, 5 }, { 12, 5 }, { 10, 7 }, { 14, 11 } };
    return c;
}

// ---- A5: the relation list ----------------------------------------------------
uint32_t arm_relations() {
    Tally t{ "A5 D_n relation list [C, def:contragredient]" };
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    for (const CellSpec& c : cells()) {
        const auto M = GenerateSL2ZRelationsWithGroupData(K, c.n, c.ell);
        const auto D = GenerateSL2ZContragredientRelationsWithGroupData(K, c.n, c.ell);
        t.check(M.size() == D.size(), "same number of relations");
        const FlatMatrix Phi = hecke::PhiNMatrix(c.n, c.ell);
        for (size_t r = 0; r < std::min(M.size(), D.size()); ++r) {
            const auto& m = M[r];
            const auto& dd = D[r];
            t.check(m.rel.q_dim == dd.rel.q_dim && m.rel.s_lhs_idx_in_Kq == dd.rel.s_lhs_idx_in_Kq &&
                        m.rel.s_rhs_idx_in_Kq == dd.rel.s_rhs_idx_in_Kq && m.rel.tag == dd.rel.tag && m.g_r == dd.g_r,
                    "relation " + std::to_string(r) + " has the same (q, lhs, rhs, tag, g_r)");
            t.check(same_flat(dd.rel.M, ComputeContragredientPowerMatrixFM(fm22(dd.g_r, c.ell), c.n)), "M_D = rho^vee(g_r)");
            t.check(same_flat(dd.rel.M, ComputeSymmetricPowerMatrixFM(fm22(dd.g_r.inv_unimodular(), c.ell), c.n).transpose()),
                    "M_D = rho(g_r^{-1})^T");
            t.check(same_flat(Phi * m.rel.M, dd.rel.M * Phi), "Phi M_M = M_D Phi (morphism of local systems)");
        }
    }
    return t.report();
}

// I (x) Phi on flat q-cochains (simplex-major, fiber-minor).
FlatMatrix block_diag(const FlatMatrix& Phi, uint32_t N_q) {
    const uint32_t d = Phi.get_rows(), dim = N_q * d;
    FlatMatrix B = FlatMatrix::CreateZero(dim, dim, Phi.get_p());
    for (uint32_t s = 0; s < N_q; ++s)
        for (uint32_t r = 0; r < d; ++r)
            for (uint32_t c = 0; c < d; ++c) {
                const uint32_t v = Phi.get_entry(r, c);
                if (v != 0) B.set_entry(s * d + r, s * d + c, v);
            }
    return B;
}

// ---- A6 + A7 + A9: the pipeline ------------------------------------------------
uint32_t arm_pipeline() {
    Tally t6{ "A6 H^*(D_n)" };
    Tally t7{ "A7 A(D_n)(T_p) vs A(M_n)(T_p), exponent 0" };
    Tally t9{ "A9 golden objects (defined, not emitted)" };
    uint64_t untwisted = 0, neg_plus_fail = 0, neg_minus_fail = 0, nonvacuous = 0, vacuous = 0;
    const uint32_t ps[] = { 2, 3, 5, 7 };
    for (const CellSpec& c : cells()) {
        const hecke::ContragredientCell cell(c.n, c.ell);
        const auto& dM = cell.coh_M().get_H_eq_dimensions();
        const auto& dD = cell.coh_D().get_H_eq_dimensions();
        const auto& P = cell.induced_Phi();
        t6.check(dM.size() == 3 && dD.size() == 3 && P.size() == 3, "three degrees");
        std::ostringstream rec;
        rec << "  E7A_CELL n=" << c.n << " ell=" << c.ell << " phi_iso=" << (hecke::PhiNIsIsomorphism(c.n, c.ell) ? 1 : 0)
            << " rank_phi=" << cell.Phi().rank();
        for (uint32_t q = 0; q < 3 && q < dM.size() && q < dD.size() && q < P.size(); ++q) {
            const uint32_t rk = (P[q].get_rows() == 0 || P[q].get_cols() == 0) ? 0 : P[q].rank();
            rec << " | q=" << q << " dim_H_M=" << dM[q] << " dim_H_D=" << dD[q] << " rank_P=" << rk;
            t6.check(P[q].get_rows() == dD[q] && P[q].get_cols() == dM[q], "P^q is dim_H(D) x dim_H(M)");
            if (c.n < c.ell) {
                t6.check(dD[q] == dM[q], "n < ell: dim H^q(D_n) = dim H^q(M_n)");
                t6.check(rk == dM[q], "n < ell: P^q = H^q(Phi_n) is invertible");
            }
            // (1 (x) Phi) z satisfies every D_n relation, for every H^q(M_n) basis cocycle z.
            const auto& basis = cell.coh_M().get_H_eq_bases_fm()[q];
            for (const FlatMatrix& z : basis) {
                // z is N_q x d_V: row s = z(sigma_s).
                const FlatMatrix y = z * cell.Phi().transpose();   // row s = (Phi z(sigma_s))^T
                for (const auto& rel : cell.relations_D()) {
                    if (rel.q_dim != q) continue;
                    bool ok = true;
                    const uint32_t d = c.n + 1;
                    for (uint32_t r = 0; r < d && ok; ++r) {
                        uint64_t acc = 0;
                        for (uint32_t k = 0; k < d; ++k)
                            acc = (acc + static_cast<uint64_t>(rel.M.get_entry(r, k)) * y.get_entry(rel.s_rhs_idx_in_Kq, k)) % c.ell;
                        ok = (acc == y.get_entry(rel.s_lhs_idx_in_Kq, r));
                    }
                    t6.check(ok, "(1 (x) Phi) z satisfies the D_n relation '" + rel.tag + "'");
                }
            }
        }
        std::cout << rec.str() << "\n";

        std::vector<hecke::ContragredientCell::HeckePair> pairs;
        for (uint32_t p : ps) {
            pairs.push_back(cell.hecke_pair(p));
            const auto& hp = pairs.back();
            const auto& A_M = hp.op_M->get_H_matrices();
            const auto& A_D = hp.op_D->get_H_matrices();
            const auto cert = hp.op_D->certificates();
            t7.check(cert.routes_bit_identical && cert.theta_eq_residual_max == 0, "D_n operator: O12(a) and A-N3 certified");
            t7.check(hp.op_D->representation() == FibreRepresentation::ContragredientN &&
                         hp.op_M->representation() == FibreRepresentation::SymN, "representations");
            const auto& TM = hp.op_M->get_Theta_flat_matrices();
            const auto& TD = hp.op_D->get_Theta_flat_matrices();
            for (uint32_t q = 0; q < 3; ++q) {
                const uint32_t N_q = cell.K().get_num_q_simplices(q);
                const FlatMatrix B = block_diag(cell.Phi(), N_q);
                t7.check(same_flat(TD[q] * B, B * TM[q]),
                         "cochain level: Theta^D (1 (x) Phi) = (1 (x) Phi) Theta^M, p=" + std::to_string(p) + " q=" + std::to_string(q));
                bool nz = false;
                const bool ok0 = hecke::HeckeIntertwines(A_D[q], P[q], A_M[q], 1u, c.ell, &nz);
                t7.check(ok0, "A(D_n) P = P A(M_n) (exponent 0), n=" + std::to_string(c.n) + " ell=" +
                                  std::to_string(c.ell) + " p=" + std::to_string(p) + " q=" + std::to_string(q));
                if (ok0) ++untwisted;
                if (p == c.ell) continue;   // p^{-n} undefined mod ell; U_ell is not a control cell
                const uint32_t pn = engine::scalar::powmod<uint32_t>(p, c.n, c.ell);
                if (pn == 1 || !nz) { ++vacuous; continue; }
                ++nonvacuous;
                const uint32_t pinvn = engine::scalar::powmod<uint32_t>(pn, c.ell - 2, c.ell);   // (p^n)^{-1}, ell prime
                const bool okp = hecke::HeckeIntertwines(A_D[q], P[q], A_M[q], pn, c.ell);
                const bool okm = hecke::HeckeIntertwines(A_D[q], P[q], A_M[q], pinvn, c.ell);
                t7.check(!okp, "negative control p^{+n} must FAIL");
                t7.check(!okm, "negative control p^{-n} must FAIL");
                if (!okp) ++neg_plus_fail;
                if (!okm) ++neg_minus_fail;
            }
        }
        // Both routes of one D_n operator on standard bases (small cells only).
        if (c.n <= 10) {
            const auto& hp = pairs.front();
            for (uint32_t q = 0; q < 3; ++q) {
                const uint32_t dim = cell.K().get_num_q_simplices(q) * (c.n + 1);
                bool ok = true;
                for (uint32_t b = 0; b < dim && ok; ++b) {
                    FlatMatrix e(dim, 1, c.ell);
                    e.set_entry(b, 0, 1);
                    ok = same_flat(hp.op_D->apply_Theta(q, e), hp.op_D->apply_route2(q, e));
                }
                t7.check(ok, "D_n operator: apply_Theta = apply_route2 on the standard basis, q=" + std::to_string(q));
            }
        }
        // A9: the golden objects of this cell.
        std::vector<hecke::DualityHeckeInput> in;
        for (const auto& hp : pairs) in.push_back(hecke::DualityHeckeInput{ hp.p, &hp.op_D->get_H_matrices() });
        const auto objs = hecke::ContragredientGoldenObjects(cell.coh_D(), in);
        t9.check(objs.size() == 3 + 3 * in.size(), "3 dim_H_Dn + 3 charpoly per Hecke prime");
        for (uint32_t q = 0; q < 3 && q < objs.size(); ++q)
            t9.check(objs[q].object == "dim_H_Dn^" + std::to_string(q) && objs[q].tier == 'A' &&
                         objs[q].hash == engine::FnvU64(engine::kFnvOffset64, dD[q]),
                     "dim_H_Dn^q record");
        for (size_t k = 3; k < objs.size(); ++k)
            t9.check(objs[k].tier == 'A' && objs[k].object.rfind("charpoly_A_Dn(T_", 0) == 0 && objs[k].cols == 1,
                     "charpoly_A_Dn(T_p)^q record");
    }
    t7.check(nonvacuous >= 1, "at least one non-vacuous negative-control case");
    std::cout << "  DUALITY_EXPONENT expected=0 source=[C, lem:symdual] derivation: Theta_g carries one fibre matrix "
                 "rho(h) per term (h = delta_i gamma_k in Mat_2(Z)); the D_n operator carries rho^vee(h); "
                 "Phi_n rho_n(h) = rho^vee(h) Phi_n for every h with no det factor => A(D_n) P = p^0 P A(M_n)\n";
    std::cout << "  DUALITY_EXPONENT observed=0 untwisted_holds=" << untwisted << " nonvacuous_controls=" << nonvacuous
              << " p^+n_fail=" << neg_plus_fail << " p^-n_fail=" << neg_minus_fail << " vacuous_skipped=" << vacuous << "\n";
    return t6.report() + t7.report() + t9.report();
}

// ---- A8: the memo stamp and the constructor guard ---------------------------------
uint32_t arm_memo() {
    Tally t{ "A8 memo stamp / representation guard" };
    const hecke::ContragredientCell cell(4, 7);
    const hecke::HeckeCosetTables cosets(2, 1, 7, 4);
    const hecke::HeckeTauTable tau(cell.K(), cell.model(), cosets, 7, 4);
    alignas(hecke::HeckeOperatorOnK) unsigned char buf[sizeof(hecke::HeckeOperatorOnK)];
    auto* opM = new (buf) hecke::HeckeOperatorOnK(&cell.coh_M(), &cosets, &tau);
    const auto idM = opM->rho_memo_identity();
    hecke::HeckeOperatorOnK::RhoMemo memo;
    const uint32_t dim0 = cell.K().get_num_q_simplices(0) * 5;
    FlatMatrix e(dim0, 1, 7);
    e.set_entry(0, 0, 1);
    (void)opM->apply_route2(0, e, memo);   // stamps the memo with the rho_n identity
    std::destroy_at(opM);
    auto* opD = new (buf) hecke::HeckeOperatorOnK(&cell.coh_D(), &cosets, &tau, FibreRepresentation::ContragredientN);
    const auto idD = opD->rho_memo_identity();
    t.check(idM.representation == 0 && idD.representation == 1, "representation field: 0 = rho_n, 1 = rho_n^vee");
    t.check(idM.owner == idD.owner && idM.delta_table == idD.delta_table && idM.ell == idD.ell &&
                idM.n_sym == idD.n_sym && idM.m == idD.m,
            "same address, same coset table, ell, n_sym, m: only the representation differs");
    bool refused = false;
    try { (void)opD->apply_route2(0, e, memo); } catch (const std::logic_error&) { refused = true; }
    t.check(refused, "a rho_n memo is refused by the rho_n^vee operator at the same address");
    hecke::HeckeOperatorOnK::RhoMemo fresh;
    bool fresh_ok = true;
    try { (void)opD->apply_route2(0, e, fresh); } catch (const std::exception&) { fresh_ok = false; }
    t.check(fresh_ok, "a fresh memo works");
    std::destroy_at(opD);
    bool guard = false;
    try {
        hecke::HeckeOperatorOnK bad(&cell.coh_M(), &cosets, &tau, FibreRepresentation::ContragredientN);
    } catch (const std::runtime_error&) { guard = true; }
    t.check(guard, "a contragredient operator on the Sym^n local system is refused");
    // The contragredient operator on the LEGACY arm (ENGINE_BACKEND=legacy's
    // path, forced thread-locally) equals the engine arm entrywise.
    // (S24 CL-L2) ON builds only: an OFF build prints SKIP-ARM.
    if (oc::legacy::RunLegacyArm("HeckeDualityValidation", "contragredient-legacy")) {
#if OC_ENABLE_LEGACY
        hecke::HeckeOperatorOnK eng(&cell.coh_D(), &cosets, &tau, FibreRepresentation::ContragredientN);
        eng.compute();
        hecke::HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath force(true);
        hecke::HeckeOperatorOnK leg(&cell.coh_D(), &cosets, &tau, FibreRepresentation::ContragredientN);
        leg.compute();
        bool same = true;
        for (uint32_t q = 0; q < 3; ++q) {
            same = same && same_flat(eng.get_Theta_flat_matrices()[q], leg.get_Theta_flat_matrices()[q]) &&
                   same_flat(eng.get_Theta_eq_matrices()[q], leg.get_Theta_eq_matrices()[q]);
            const auto& a = eng.get_H_matrices()[q];
            const auto& b = leg.get_H_matrices()[q];
            same = same && a.rows == b.rows && a.cols == b.cols && a.data == b.data;
        }
        t.check(same, "contragredient operator: legacy arm == engine arm (Theta_flat, Theta_eq, A)");
#endif
    }
    return t.report();
}

// =============================================================================
// slot 6, WP-3 (phase E7b): the cohomological pairings, perfectness,
// adjointness, the PD gate for X_3, socle <-> Loewy, the iota-sign.
// =============================================================================

// The E7b cells: WP-1's cells (n < ell and n >= ell, the Dickson locus (6, 5),
// n = 0), plus (10, 691) and the ten Kummer cells of HeckeExtKummerValidation
// ((10, 691) among them).  All have ell >= 5 and n even: [C, cl:pd]'s range.
const std::vector<CellSpec>& pd_cells() {
    static const std::vector<CellSpec> c = {
        { 0, 7 }, { 4, 7 }, { 10, 13 }, { 12, 13 }, { 16, 101 }, { 22, 29 }, { 6, 5 }, { 12, 5 }, { 10, 7 }, { 14, 11 },
        { 10, 691 }, { 14, 3617 }, { 16, 43867 }, { 18, 283 }, { 18, 617 }, { 20, 131 }, { 20, 593 }, { 22, 103 },
        { 22, 2294797 }, { 24, 657931 } };
    return c;
}
// One Hecke prime on one E7b cell: the cone action on M_n and the D_n operator.
struct PdOp {
    uint32_t p = 0;
    hecke::ContragredientCell::HeckePair hp;   // cosets, tau, op_M, op_D
    std::unique_ptr<hecke::HeckeBoundaryTauTable> taub;
    std::unique_ptr<hecke::HeckePairHomotopy> eta;
    std::unique_ptr<hecke::HeckeBoundaryOperator> bd;
    std::unique_ptr<hecke::HeckeConeAction> act;
};

std::unique_ptr<PdOp> build_pd_op(const hecke::ContragredientCell& cell, const hecke::PoincareDualityPairing& pd,
                                  uint32_t p) {
    auto o = std::make_unique<PdOp>();
    o->p = p;
    o->hp = cell.hecke_pair(p);
    o->taub = std::make_unique<hecke::HeckeBoundaryTauTable>(cell.K(), cell.model(), *o->hp.cosets);
    o->eta = std::make_unique<hecke::HeckePairHomotopy>(*o->hp.tau, *o->taub);
    o->bd = std::make_unique<hecke::HeckeBoundaryOperator>(&pd.pair_M().boundary_cohomology(), o->hp.cosets.get(),
                                                           o->taub.get());
    o->bd->compute();
    o->act = std::make_unique<hecke::HeckeConeAction>(&pd.cone(), o->hp.op_M.get(), o->bd.get(), o->eta.get());
    o->act->compute();
    return o;
}

FlatMatrix column_of(const FlatMatrix& B, uint32_t j) {
    FlatMatrix c(B.get_rows(), 1, B.get_p());
    for (uint32_t i = 0; i < B.get_rows(); ++i) c.set_entry(i, 0, B.get_entry(i, j));
    return c;
}

FlatMatrix flat_of_mv(const engine::BlockMat<uint32_t>& B, uint32_t ell) {
    const engine::DenseBuf<uint32_t> D = B.to_flat(engine::Layout::RowMajor);
    FlatMatrix out = FlatMatrix::CreateZero(D.rows, D.cols, ell);
    if (D.rows == 0 || D.cols == 0) return out;
    std::vector<uint32_t> d(static_cast<size_t>(D.rows) * D.cols);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j) d[static_cast<size_t>(i) * D.cols + j] = D.at(i, j);
    return FlatMatrix(D.rows, D.cols, ell, d);
}
FlatMatrix flat_of_bm(const engine::BlockMat<uint32_t>& B, uint32_t ell) {
    const engine::DenseBuf<uint32_t> D = B.to_dense(engine::Layout::RowMajor);
    FlatMatrix out = FlatMatrix::CreateZero(D.rows, D.cols, ell);
    if (D.rows == 0 || D.cols == 0) return out;
    std::vector<uint32_t> d(static_cast<size_t>(D.rows) * D.cols);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j) d[static_cast<size_t>(i) * D.cols + j] = D.at(i, j);
    return FlatMatrix(D.rows, D.cols, ell, d);
}

// Exact product with zero-dimension guards.
FlatMatrix mulz(const FlatMatrix& A, const FlatMatrix& B) {
    const uint32_t ell = A.get_p();
    if (A.get_rows() == 0 || B.get_cols() == 0 || A.get_cols() == 0)
        return FlatMatrix::CreateZero(A.get_rows(), B.get_cols(), ell);
    return A * B;
}
FlatMatrix trz(const FlatMatrix& A) {
    if (A.get_rows() == 0 || A.get_cols() == 0) return FlatMatrix::CreateZero(A.get_cols(), A.get_rows(), A.get_p());
    return A.transpose();
}
bool same_or_empty(const FlatMatrix& A, const FlatMatrix& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols()) return false;
    for (uint32_t i = 0; i < A.get_rows(); ++i)
        for (uint32_t j = 0; j < A.get_cols(); ++j)
            if (A.get_entry(i, j) != B.get_entry(i, j)) return false;
    return true;
}
MatrixT<uint32_t> to_mt(const FlatMatrix& F) {
    MatrixT<uint32_t> M(F.get_rows(), F.get_cols(), F.get_p());
    for (uint32_t i = 0; i < F.get_rows(); ++i)
        for (uint32_t j = 0; j < F.get_cols(); ++j) M.data[i][j] = F.get_entry(i, j);
    return M;
}

// iota_c := iota^* (+) iota^*_bd on the cone's flat layout (bulk part first),
// exactly as HeckeExt/HeckeInvolutionValidation's arm A4 builds it.
FlatMatrix iota_cone_flat(const EquivariantMappingCone& cone, const hecke::HeckeInvolution& ib,
                          const hecke::HeckeInvolution& ibd, uint32_t t, uint32_t ell) {
    const uint32_t nb = cone.bulk_part_dim(t), nd = cone.boundary_part_dim(t), dim = nb + nd;
    std::vector<uint32_t> d(static_cast<size_t>(dim) * dim, 0u);
    if (nb != 0u) {
        const FlatMatrix A = ib.cochain_matrix(t, -1);
        for (uint32_t i = 0; i < nb; ++i)
            for (uint32_t j = 0; j < nb; ++j) d[static_cast<size_t>(i) * dim + j] = A.get_entry(i, j);
    }
    if (nd != 0u) {
        const FlatMatrix B = ibd.cochain_matrix(t - 1u, -1);
        for (uint32_t i = 0; i < nd; ++i)
            for (uint32_t j = 0; j < nd; ++j) d[static_cast<size_t>(nb + i) * dim + (nb + j)] = B.get_entry(i, j);
    }
    if (dim == 0) return FlatMatrix::CreateZero(0, 0, ell);
    return FlatMatrix(dim, dim, ell, d);
}

uint32_t arm_pd() {
    Tally t10{ "A10 the pairings: fundamental chain, descent, two routes [C, cl:pd](1) cochain level" };
    Tally t11{ "A11 perfectness [C, cl:pd](1) -- OPEN in [C]; this is the oracle" };
    Tally t12{ "A12 adjointness <A_c x, y> = <x, A(D_n) y> [C, cl:pd](2): THE PD GATE FOR X_3" };
    Tally t13{ "A13 socle <-> Loewy, HeckeIdentityCertificate item 7 [C, cl:pd](3)" };
    Tally t14{ "A14 the iota-pairing sign (RECORDED) [C, rmk:iotapairing]" };
    Tally t15{ "A15 golden objects (defined, not emitted)" };
    const uint32_t ps[] = { 2, 3, 5, 7 };
    uint64_t gate_ops = 0, gate_degrees = 0, gate_fail = 0, neg_nonvacuous = 0, neg_fail = 0;
    uint64_t perfect_pairs = 0, bd_perfect_pairs = 0, item7_passed = 0, item7_nontrivial = 0, tamper_rejected = 0;
    uint64_t iota_recorded = 0, iota_match = 0, iota_unavailable = 0;
    uint32_t peak_flat = 0;
    for (const CellSpec& cs : pd_cells()) {
        const hecke::ContragredientCell cell(cs.n, cs.ell);
        const uint32_t ell = cs.ell;
        std::unique_ptr<hecke::PoincareDualityPairing> pdp;
        try {
            pdp = std::make_unique<hecke::PoincareDualityPairing>(cell);
        } catch (const std::exception& ex) {
            t10.check(false, "PoincareDualityPairing built at n=" + std::to_string(cs.n) + " ell=" + std::to_string(ell) +
                                 ": " + ex.what());
            continue;
        }
        const hecke::PoincareDualityPairing& pd = *pdp;
        const std::string where = "(n=" + std::to_string(cs.n) + ", ell=" + std::to_string(ell) + ")";
        // ---- A10 ------------------------------------------------------------
        t10.check(pd.chain().qualifying_sign_vectors == 2 && pd.chain().horocycle_coefficient == -1,
                  "the fundamental chain: exactly +-eps satisfy the cycle condition " + where);
        t10.check(pd.certificate().verify() == 0, "PairingCertificate::verify() == 0 " + where);
        const auto& cert = pd.certificate();
        for (uint32_t t = 0; t < 3; ++t) {
            peak_flat = std::max(peak_flat, pd.cone_form(t).get_rows());
            // Second route: evaluate_direct on every (H_c column, H_D column) pair == G.
            const FlatMatrix& G = pd.gram(t);
            bool ok = true;
            for (uint32_t i = 0; i < G.get_rows() && ok; ++i)
                for (uint32_t j = 0; j < G.get_cols() && ok; ++j)
                    ok = pd.evaluate_direct(t, column_of(cert.Hc[t], i), column_of(cert.HD[2 - t], j)) == G.get_entry(i, j);
            t10.check(ok, "cup-product definition == stored form on the stored bases, t=" + std::to_string(t) + " " + where);
        }
        {
            // Descent is NOT vacuous: some (cone cochain, D-coboundary) pair pairs non-trivially.
            bool some = false;
            for (uint32_t t = 0; t < 3 && !some; ++t) {
                const FlatMatrix X = mulz(mulz(trz(flat_of_mv(pd.cone().Ceq_block(t), ell)), pd.cone_form(t)), cert.BD[2 - t]);
                for (uint32_t i = 0; i < X.get_rows() && !some; ++i)
                    for (uint32_t j = 0; j < X.get_cols() && !some; ++j) some = X.get_entry(i, j) != 0;
            }
            t10.check(some || cs.n == 0, "descent is non-vacuous: P(cone cochain, D-coboundary) != 0 somewhere " + where);
        }
        // ---- A11 ------------------------------------------------------------
        std::ostringstream rec;
        rec << "  E7B_CELL n=" << cs.n << " ell=" << ell;
        for (uint32_t t = 0; t < 3; ++t) {
            const uint32_t dc = pd.cone().dim_Hc(t), dd = cell.coh_D().get_H_eq_dimensions()[2 - t];
            rec << " | t=" << t << " dim_Hc=" << dc << " dim_H_D^" << (2 - t) << "=" << dd << " perfect=" << (pd.perfect(t) ? 1 : 0);
            t11.check(dc == dd, "dim H^t_c(M_n) == dim H^{2-t}(D_n), t=" + std::to_string(t) + " " + where);
            t11.check(pd.perfect(t), "FINDING if it fails: H^t_c(M_n) x H^{2-t}(D_n) perfect, t=" + std::to_string(t) + " " + where);
            if (pd.perfect(t)) ++perfect_pairs;
        }
        for (uint32_t s = 0; s < 2; ++s) {
            rec << " | bd s=" << s << " " << pd.boundary_gram(s).get_rows() << "x" << pd.boundary_gram(s).get_cols()
                << " perfect=" << (pd.boundary_perfect(s) ? 1 : 0);
            t11.check(pd.boundary_perfect(s),
                      "FINDING if it fails: H^s(bd; M_n) x H^{1-s}(bd; D_n) perfect, s=" + std::to_string(s) + " " + where);
            if (pd.boundary_perfect(s)) ++bd_perfect_pairs;
        }
        std::cout << rec.str() << "\n";
        // ---- A12 ------------------------------------------------------------
        std::vector<std::unique_ptr<PdOp>> ops;
        std::vector<char> adj_t(3, 1);
        for (uint32_t p : ps) {
            try {
                ops.push_back(build_pd_op(cell, pd, p));
            } catch (const std::exception& ex) {
                t12.check(false, "cone action / D_n operator built, p=" + std::to_string(p) + " " + where + ": " + ex.what());
                adj_t.assign(3, 0);
                continue;
            }
            const PdOp& o = *ops.back();
            const auto& A_D = o.hp.op_D->get_H_matrices();
            ++gate_ops;
            for (uint32_t t = 1; t < 3; ++t) {
                const MatrixT<uint32_t>& Ac = o.act->A_c(t);
                t12.check(Ac.rows == pd.cone().dim_Hc(t), "A_c^{(t)} is indexed by the degree of H^*_c");
                bool nz = false;
                const bool ok = hecke::PairingAdjoint(Ac, pd.gram(t), A_D[2 - t], 1u, ell, &nz);
                ++gate_degrees;
                if (!ok) { ++gate_fail; adj_t[t] = 0; }
                t12.check(ok, "FINDING if it fails: A_c^{(" + std::to_string(t) + ")T} G = G A(D_n)^{(" +
                                  std::to_string(2 - t) + ")}, p=" + std::to_string(p) + " " + where);
                // Negative control: the p^n-twisted reading must FAIL wherever it is non-vacuous.
                if (p % ell == 0 || !nz) continue;
                const uint32_t pn = engine::scalar::powmod<uint32_t>(p, cs.n, ell);
                if (pn == 1) continue;
                ++neg_nonvacuous;
                const bool okn = hecke::PairingAdjoint(Ac, pd.gram(t), A_D[2 - t], pn, ell);
                t12.check(!okn, "negative control: the p^n-twisted adjointness must FAIL, p=" + std::to_string(p) + " " + where);
                if (!okn) ++neg_fail;
            }
        }
        // ---- A13: item 7 ------------------------------------------------------
        {
            hecke_ext::IdentityCertificate ic;
            ic.n = cs.n;
            ic.ell = ell;
            for (const auto& o : ops) {
                if (o->p % ell == 0) continue;   // the good-prime algebra
                ic.pd_ops_V.push_back(o->act->A_c(1));
                ic.pd_ops_W.push_back(o->hp.op_D->get_H_matrices()[1]);
            }
            ic.pd_pairing = to_mt(pd.gram(1));
            const uint32_t code = ic.verify();
            const bool nontriv = pd.gram(1).get_rows() > 0;
            if (nontriv) {
                ++item7_nontrivial;
                t13.check(ic.socle_loewy_pd.status == hecke_ext::IdentityStatus::kPassed,
                          "item 7 passed " + where + " (" + ic.socle_loewy_pd.detail + ")");
                t13.check(code != 6, "item code 6 is WP-9's and never returned");
                if (ic.socle_loewy_pd.status == hecke_ext::IdentityStatus::kPassed) ++item7_passed;
                std::cout << "  E7B_SOCLE_LOEWY n=" << cs.n << " ell=" << ell << " " << ic.socle_loewy_pd.detail << "\n";
                // Tamper: G' = G U (U unipotent upper, fixed off-diagonal 1s) keeps perfectness and must
                // break the identity wherever the Loewy length is >= 2 (the perp moves, the radical does not).
                if (ic.pd_socle_dims.size() >= 3) {
                    hecke_ext::IdentityCertificate bad = ic;
                    const uint32_t d = bad.pd_pairing.cols;
                    MatrixT<uint32_t> U(d, ell);
                    for (uint32_t i = 0; i + 1 < d; ++i) U.data[i][i + 1] = 1;
                    MatrixT<uint32_t> Gp = ic.pd_pairing * U;
                    // also a transposition-like mix so that a triangular U cannot preserve the flag by accident
                    MatrixT<uint32_t> Lw(d, ell);
                    for (uint32_t i = 0; i + 1 < d; ++i) Lw.data[i + 1][i] = 2 % ell;
                    bad.pd_pairing = Gp * Lw;
                    bad.verify();
                    const bool rejected = bad.socle_loewy_pd.status == hecke_ext::IdentityStatus::kFailed;
                    std::cout << "  E7B_SOCLE_LOEWY_TAMPER n=" << cs.n << " ell=" << ell << " rejected=" << (rejected ? 1 : 0) << "\n";
                    if (rejected) ++tamper_rejected;
                }
            } else {
                t13.check(ic.socle_loewy_pd.status == hecke_ext::IdentityStatus::kNotApplicable,
                          "item 7 not applicable on a zero X_3 " + where);
            }
        }
        // ---- A14: the iota-sign (recorded) ---------------------------------------
        try {
            hecke::HeckeInvolution ib(&cell.coh_M(), hecke::BuildIotaSimplicialMapOnModel(cell.K(), cell.model()), "iota^* M_n");
            ib.compute();
            hecke::HeckeInvolution ibd(&pd.pair_M().boundary_cohomology(),
                                       hecke::BuildIotaSimplicialMapFromVertexPermutation(
                                           pd.pair_M().K_bd(), hecke::IotaBoundaryPathVertexPermutation()),
                                       "iota^*_bd M_n");
            ibd.compute();
            hecke::HeckeInvolution iD(&cell.coh_D(), hecke::BuildIotaSimplicialMapOnModel(cell.K(), cell.model()), "iota^* D_n");
            iD.compute();
            std::ostringstream ir;
            ir << "  E7B_IOTA_PAIRING n=" << cs.n << " ell=" << ell << " predicted=-(-1)^n=" << (cs.n % 2 == 0 ? "-1" : "+1");
            for (uint32_t t = 1; t < 3; ++t) {
                const FlatMatrix& G = pd.gram(t);
                if (G.get_rows() == 0) { ir << " t=" << t << ":empty"; continue; }
                const FlatMatrix Ic = mulz(mulz(flat_of_bm(pd.cone().P_block(t), ell), iota_cone_flat(pd.cone(), ib, ibd, t, ell)),
                                           pd.certificate().Hc[t]);
                const FlatMatrix& ID = iD.matrix(2 - t);
                const FlatMatrix lhs = mulz(mulz(trz(Ic), G), ID);
                const bool plus = same_or_empty(lhs, G);
                const bool minus = same_or_empty(lhs, G * (ell - 1));
                const char* eps = plus ? "+1" : minus ? "-1" : "none";
                ir << " t=" << t << ":eps=" << eps;
                ++iota_recorded;
                if (minus && cs.n % 2 == 0) ++iota_match;
                t14.check(plus || minus, "RECORDED: <iota x, iota y> = eps <x, y> with eps = +-1, t=" + std::to_string(t) + " " + where);
            }
            std::cout << ir.str() << "\n";
        } catch (const std::exception& ex) {
            ++iota_unavailable;
            std::cout << "  E7B_IOTA_PAIRING n=" << cs.n << " ell=" << ell << " unavailable: " << ex.what() << "\n";
        }
        // ---- A15 ---------------------------------------------------------------
        const auto objs = hecke::PairingGoldenObjects(pd, adj_t);
        t15.check(objs.size() == 9, "3 x (pairing_perfect, pd_adjoint, pairing_matrix)");
        for (uint32_t t = 0; t < 3 && objs.size() == 9; ++t) {
            t15.check(objs[3 * t].object == "pairing_perfect^" + std::to_string(t) && objs[3 * t].tier == 'A', "pairing_perfect^t record");
            t15.check(objs[3 * t + 1].object == "pd_adjoint^" + std::to_string(t) && objs[3 * t + 1].tier == 'A', "pd_adjoint^t record");
            t15.check(objs[3 * t + 2].object == "pairing_matrix^" + std::to_string(t) && objs[3 * t + 2].tier == 'B' &&
                          objs[3 * t + 2].rows == pd.gram(t).get_rows() && objs[3 * t + 2].cols == pd.gram(t).get_cols(),
                      "pairing_matrix^t record");
        }
    }
    t12.check(neg_nonvacuous >= 1 && neg_fail == neg_nonvacuous, "the p^n-twisted negative controls are non-vacuous and all fail");
    t13.check(item7_nontrivial >= 1 && tamper_rejected >= 1, "item 7 non-vacuous and its tamper arm rejected somewhere");
    std::cout << "  PD_GATE X_3 [C, obl:coneexact](vii) third sentence: ASSERTED at ell >= 5 on " << pd_cells().size()
              << " cells, " << gate_ops << " operators, " << gate_degrees << " (operator, t) pairs, failing=" << gate_fail
              << "; p^n-twisted controls " << neg_fail << "/" << neg_nonvacuous << " rejected\n";
    std::cout << "  PD_PERFECT cone pairs perfect=" << perfect_pairs << "/" << 3 * pd_cells().size()
              << " circle pairs perfect=" << bd_perfect_pairs << "/" << 2 * pd_cells().size()
              << " peak_cone_flat_dim=" << peak_flat << "\n";
    std::cout << "  PD_SOCLE_LOEWY item7 passed=" << item7_passed << "/" << item7_nontrivial << " tamper_rejected=" << tamper_rejected << "\n";
    std::cout << "  PD_IOTA_SIGN recorded=" << iota_recorded << " eps=-(-1)^n on " << iota_match << " unavailable_cells=" << iota_unavailable
              << " (a prediction of [C, rmk:iotapairing], asserted nowhere)\n";
    return t10.report() + t11.report() + t12.report() + t13.report() + t14.report() + t15.report();
}

// =============================================================================
// slot 7, WP-1 (E7 acceptance (b)): COVERAGE of [C, cl:pd](1)-(2) on the E2
// case list at ell >= 5 -- the 31 ext-subgrid cells ([X] Sec. 1.6) and the 23
// M1 cells -- beyond the 20 E7b cells above.  Per cell: the certificate, the
// dimension identity and perfectness of the cone and circle pairings (A11's
// checks) and the PD gate (A12's check) for p in {2, 3, 5, 7}, with the
// p^n-twisted negative control on every non-vacuous (cell, p, t).  A failure
// is a FINDING about [C]'s OPEN claim, never worked around.
// =============================================================================

// The E2 case list at ell >= 5, n even, MINUS the 20 pd_cells() above.
// Standard part: every ext-subgrid cell not in pd_cells() (29) and the M1
// cells with n <= 34 not in pd_cells() (6).
const std::vector<CellSpec>& coverage_cells_standard() {
    static const std::vector<CellSpec> c = {
        // ext subgrid ([X] Sec. 1.6), (10, 13) and (10, 691) are in pd_cells()
        { 2, 5 }, { 2, 13 }, { 2, 691 }, { 2, 10007 }, { 2, 134217757 },
        { 4, 5 }, { 4, 13 }, { 4, 691 }, { 4, 10007 }, { 4, 134217757 },
        { 10, 5 }, { 10, 10007 }, { 10, 134217757 },
        { 16, 5 }, { 16, 13 }, { 16, 691 }, { 16, 10007 }, { 16, 134217757 },
        { 22, 5 }, { 22, 13 }, { 22, 691 }, { 22, 10007 }, { 22, 134217757 },
        { 24, 5 }, { 24, 13 }, { 24, 691 }, { 24, 10007 }, { 24, 134217757 },
        { 32, 5 },
        // M1 (HeckeExtEndToEndValidation's list), n <= 34; (22, 5) is above
        { 8, 7 }, { 12, 11 }, { 34, 5 }, { 26, 7 }, { 32, 131 } };
    return c;
}
// Slow part: the M1 cells with n >= 58 (the cone over D_n at large n).
// (58, 7), (58, 5), (82, 7) always (6.2 + 6.4 + 17.1 s, indicative).
// (160, 691) and (198, 691) cost 129 s and 241 s (indicative, slot-7 WP-1):
// LEFT OUT of the registered run for cost, and reachable, read-only, under
// HECKE_DUALITY_COVERAGE_LARGE=1 (they passed there: 0 gate failures).
std::vector<CellSpec> coverage_cells_slow() {
    std::vector<CellSpec> c = { { 58, 7 }, { 58, 5 }, { 82, 7 } };
    const char* env = std::getenv("HECKE_DUALITY_COVERAGE_LARGE");
    if (env != nullptr && std::string(env) == "1") {
        c.push_back({ 160, 691 });
        c.push_back({ 198, 691 });
    } else {
        std::cout << "  E7_COVERAGE (160, 691), (198, 691): left out for cost (~6 min together); "
                     "set HECKE_DUALITY_COVERAGE_LARGE=1 to include them\n";
    }
    return c;
}

uint32_t arm_pd_coverage(const std::vector<CellSpec>& cells, const char* label) {
    Tally t16{ "A16 COVERAGE cl:pd(1)-(2) on the E2 case list [C, cl:pd] -- OPEN in [C]; FINDING if it fails" };
    const uint32_t ps[] = { 2, 3, 5, 7 };
    uint64_t perfect = 0, bd_perfect = 0, gate_pairs = 0, gate_fail = 0, neg_nonvacuous = 0, neg_fail = 0;
    for (const CellSpec& cs : cells) {
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t ell = cs.ell;
        const std::string where = "(n=" + std::to_string(cs.n) + ", ell=" + std::to_string(ell) + ")";
        try {
            const hecke::ContragredientCell cell(cs.n, ell);
            const hecke::PoincareDualityPairing pd(cell);
            t16.check(pd.certificate().verify() == 0, "PairingCertificate::verify() == 0 " + where);
            std::ostringstream rec;
            rec << "  E7_COVERAGE set=" << label << " n=" << cs.n << " ell=" << ell;
            for (uint32_t t = 0; t < 3; ++t) {
                const uint32_t dc = pd.cone().dim_Hc(t), dd = cell.coh_D().get_H_eq_dimensions()[2 - t];
                t16.check(dc == dd, "dim H^t_c(M_n) == dim H^{2-t}(D_n), t=" + std::to_string(t) + " " + where);
                t16.check(pd.perfect(t), "H^t_c(M_n) x H^{2-t}(D_n) perfect, t=" + std::to_string(t) + " " + where);
                if (pd.perfect(t)) ++perfect;
                rec << " t" << t << "=" << dc << (pd.perfect(t) ? "p" : "NP");
            }
            for (uint32_t s = 0; s < 2; ++s) {
                t16.check(pd.boundary_perfect(s), "H^s(bd; M_n) x H^{1-s}(bd; D_n) perfect, s=" + std::to_string(s) + " " + where);
                if (pd.boundary_perfect(s)) ++bd_perfect;
            }
            uint32_t cell_fail = 0;
            for (uint32_t p : ps) {
                const std::unique_ptr<PdOp> o = build_pd_op(cell, pd, p);
                const auto& A_D = o->hp.op_D->get_H_matrices();
                for (uint32_t t = 1; t < 3; ++t) {
                    bool nz = false;
                    const bool ok = hecke::PairingAdjoint(o->act->A_c(t), pd.gram(t), A_D[2 - t], 1u, ell, &nz);
                    ++gate_pairs;
                    if (!ok) { ++gate_fail; ++cell_fail; }
                    t16.check(ok, "A_c^{(" + std::to_string(t) + ")T} G = G A(D_n)^{(" + std::to_string(2 - t) +
                                      ")}, p=" + std::to_string(p) + " " + where);
                    if (p % ell == 0 || !nz) continue;
                    const uint32_t pn = engine::scalar::powmod<uint32_t>(p, cs.n, ell);
                    if (pn == 1) continue;
                    ++neg_nonvacuous;
                    const bool okn = hecke::PairingAdjoint(o->act->A_c(t), pd.gram(t), A_D[2 - t], pn, ell);
                    t16.check(!okn, "negative control: p^n-twisted adjointness must FAIL, p=" + std::to_string(p) + " " + where);
                    if (!okn) ++neg_fail;
                }
            }
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            rec << " gate_fail=" << cell_fail << " " << std::fixed << std::setprecision(2) << secs << " s";
            std::cout << rec.str() << "\n";
        } catch (const std::exception& ex) {
            t16.check(false, "E7 layer built " + where + ": " + ex.what());
        }
    }
    t16.check(neg_nonvacuous >= 1 && neg_fail == neg_nonvacuous, "the p^n-twisted negative controls are non-vacuous and all fail");
    std::cout << "  PD_COVERAGE set=" << label << " cells=" << cells.size() << " cone pairs perfect=" << perfect << "/"
              << 3 * cells.size() << " circle pairs perfect=" << bd_perfect << "/" << 2 * cells.size()
              << " PD gate (operator, t) pairs=" << gate_pairs << " failing=" << gate_fail << "; p^n-twisted controls "
              << neg_fail << "/" << neg_nonvacuous << " rejected\n";
    std::cout << "EFFECTIVE HeckeDualityCoverageSlowValidation arm=" << label << " cells=" << cells.size() << "\n";
    return t16.report();
}

} // namespace

uint32_t HeckeDualityCoverageSlowValidation() {
    std::cout << "HeckeDualityCoverageSlowValidation (slot-7 WP-1, E7 acceptance (b): cl:pd(1)-(2) on the M1 cells "
                 "with n >= 58)\n";
    uint32_t fails = 0;
    try {
        fails += arm_pd_coverage(coverage_cells_slow(), "m1_large_n");
    } catch (const std::exception& ex) {
        std::cout << "  EXCEPTION: " << ex.what() << "\n";
        ++fails;
    }
    std::cout << "HeckeDualityCoverageSlowValidation: " << (fails == 0 ? "PASS" : "FAIL") << " (" << fails
              << " failed checks)\n";
    return fails;
}

uint32_t HeckeDualityValidation() {
    std::cout << "HeckeDualityValidation (slot-6 WP-1, phase E7a: rho^vee, D_n, Phi_n, the pairing, "
                 "H^*(D_n) and A(D_n)(T_p))\n";
    uint32_t fails = 0;
    try {
        fails += arm_adj();
        fails += arm_rhodual();
        fails += arm_pairing();
        fails += arm_phi();
        fails += arm_relations();
        fails += arm_pipeline();
        fails += arm_memo();
        fails += arm_pd();
        fails += arm_pd_coverage(coverage_cells_standard(), "subgrid_and_m1");
    } catch (const std::exception& ex) {
        std::cout << "  EXCEPTION: " << ex.what() << "\n";
        ++fails;
    }
    std::cout << "HeckeDualityValidation: " << (fails == 0 ? "PASS" : "FAIL") << " (" << fails << " failed checks)\n";
    return fails;
}
