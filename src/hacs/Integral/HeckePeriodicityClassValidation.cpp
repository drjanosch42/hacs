// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 10, WP-13 (Line 2, L2-1b (2)):
// HeckePeriodicityClassValidation.  [C] = docs/latex/gl2_contract.tex, cited
// by label.  See the header for the arms.
#include "HeckePeriodicityClassValidation.h"

#include "Model/IntegralGroups.h"
#include "Model/IsotropyModel.h"
#include "Model/PeriodicityClass.h"

#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/DenseView.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using integral_model::IsotropyModel;
using integral_model::PeriodicityClass;
using U32Ctx = engine::ModCtx<uint32_t>;
using Buf = engine::DenseBuf<uint32_t>;

constexpr uint32_t kGoodPrime = 10007;
constexpr uint32_t kNMax = 20;
constexpr uint32_t kQMax = 5;
const uint32_t kGoodEll[] = {5, 7};

uint32_t g_checks = 0;
uint32_t g_fail = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cerr << "HeckePeriodicityClassValidation: FAIL " << what << "\n";
    }
    return ok;
}

std::string vec_str(const std::vector<uint32_t>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << "]";
    return os.str();
}

std::string ring_name(const U32Ctx& ctx) {
    std::ostringstream os;
    if (ctx.is_field()) os << "F_" << ctx.p;
    else os << "Z/" << ctx.ell << "^" << ctx.k;
    return os.str();
}

// ---- small dense helpers over a ModCtx (field or composite) ---------------
Buf matmul(const U32Ctx& ctx, const Buf& A, const Buf& B) {
    ENGINE_CHECK(A.cols == B.rows);
    Buf C(A.rows, B.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t k = 0; k < A.cols; ++k) {
            const uint32_t a = A.at(i, k);
            if (a == 0) continue;
            for (uint32_t j = 0; j < B.cols; ++j) {
                const uint32_t b = B.at(k, j);
                if (b != 0) C.at(i, j) = ctx.add(C.at(i, j), ctx.mul(a, b));
            }
        }
    return C;
}

Buf transpose(const Buf& A) {
    Buf T(A.cols, A.rows);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) T.at(j, i) = A.at(i, j);
    return T;
}

Buf stack(const std::vector<const Buf*>& parts, uint32_t cols) {
    uint32_t rows = 0;
    for (const Buf* p : parts) {
        ENGINE_CHECK(p->rows == 0 || p->cols == cols);
        rows += p->rows;
    }
    Buf S(rows, cols);
    uint32_t r0 = 0;
    for (const Buf* p : parts) {
        for (uint32_t i = 0; i < p->rows; ++i)
            for (uint32_t j = 0; j < cols; ++j) S.at(r0 + i, j) = p->at(i, j);
        r0 += p->rows;
    }
    return S;
}

bool is_zero(const Buf& A) {
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.at(i, j) != 0) return false;
    return true;
}

bool equal(const Buf& A, const Buf& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            if (A.at(i, j) != B.at(i, j)) return false;
    return true;
}

// Composition length of the row span (the rank over a field).
uint32_t len_rows(const U32Ctx& ctx, const Buf& A) {
    if (A.rows == 0 || A.cols == 0) return 0;
    if (ctx.is_field()) {
        Buf C = A;
        return engine::elim::rref(ctx, engine::view(C)).rank;
    }
    return engine::elim::length(ctx, engine::cview(A));
}

// Generators of the right kernel {x : A x = 0}, as rows.
Buf kernel_rows(const U32Ctx& ctx, const Buf& A) {
    if (ctx.is_field()) {
        Buf R = A;
        const auto rr = engine::elim::rref(ctx, engine::view(R));
        return engine::elim::kernel_basis(ctx, engine::cview(R), rr.pivots);
    }
    return engine::elim::kernel_module(ctx, engine::cview(A)).H;
}

Buf identity(const U32Ctx& ctx, uint32_t D) {
    Buf I(D, D);
    for (uint32_t i = 0; i < D; ++i) I.at(i, i) = 1 % ctx.p;
    return I;
}

Buf minus(const U32Ctx& ctx, const Buf& A, const Buf& B) {
    Buf C(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = ctx.sub(A.at(i, j), B.at(i, j));
    return C;
}

Buf scale(const U32Ctx& ctx, const Buf& A, uint32_t s) {
    Buf C(A.rows, A.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j) C.at(i, j) = ctx.mul(A.at(i, j), s % ctx.p);
    return C;
}

// ---------------------------------------------------------------------------
// One ring cell: every clause of [C, lem:bulkperiodic](2) at (n, ctx).
// ---------------------------------------------------------------------------
struct CellResult {
    std::vector<uint32_t> H;   // lengths H^q, q = 0 .. 5
    uint32_t img[4] = {0, 0, 0, 0};
    uint32_t ker1 = 0, conn = 0, conn_v = 0;
};

CellResult run_cell(const IsotropyModel& M, const PeriodicityClass& P, const U32Ctx& ctx, uint32_t n,
                    const std::vector<uint32_t>* m0_pred, bool negative_E2, bool negative_conn) {
    CellResult res;
    const uint32_t D = n + 1;
    const std::string where = "(n=" + std::to_string(n) + ", " + ring_name(ctx) + ")";
    std::vector<Buf> delta;
    for (uint32_t q = 0; q <= kQMax; ++q) delta.push_back(M.coboundary(ctx, n, q));
    std::vector<Buf> E;
    for (uint32_t q = 0; q <= 4; ++q) E.push_back(P.matrix(ctx, n, q));
    if (negative_E2) E[2] = scale(ctx, E[2], 2);

    // (C) cochain map: delta^{q+2} E_q^* = E_{q+1}^* delta^q, q = 0 .. 3.
    if (!negative_E2)
        for (uint32_t q = 0; q <= 3; ++q)
            check(equal(matmul(ctx, delta[q + 2], E[q]), matmul(ctx, E[q + 1], delta[q])),
                  "E_" + std::to_string(q) + "^* is not a cochain map " + where);

    // Lengths.
    std::vector<uint32_t> lenim(kQMax + 1, 0);
    for (uint32_t q = 0; q <= kQMax; ++q) lenim[q] = len_rows(ctx, delta[q]);
    res.H.assign(kQMax + 1, 0);
    for (uint32_t q = 0; q <= kQMax; ++q) res.H[q] = M.rank(q) * D * ctx.k - lenim[q] - (q ? lenim[q - 1] : 0);
    if (!negative_E2 && !negative_conn) {
        check(res.H == M.cohomology_lengths(ctx, n), "lengths != IsotropyModel::cohomology_lengths " + where);
        if (m0_pred) check(res.H == *m0_pred, "lengths != M0's groups (UCT) " + where + ": " + vec_str(res.H) +
                                                  " vs " + vec_str(*m0_pred));
    }

    // Image of H^q in H^{q+2} under E_q, q = 1, 2, 3.
    std::vector<Buf> Brows(kQMax + 1);   // generators of B^q as rows
    for (uint32_t q = 1; q <= kQMax; ++q) Brows[q] = transpose(delta[q - 1]);
    for (uint32_t q = 1; q <= 3; ++q) {
        const Buf Z = kernel_rows(ctx, delta[q]);
        const Buf EZ = transpose(matmul(ctx, E[q], transpose(Z)));
        if (!negative_E2) check(is_zero(matmul(ctx, delta[q + 2], transpose(EZ))), "E_q(Z^q) not in Z^{q+2} " + where);
        const Buf S = stack({&Brows[q + 2], &EZ}, M.rank(q + 2) * D);
        res.img[q] = len_rows(ctx, S) - lenim[q + 1];
    }
    if (negative_E2) return res;

    // q = 2, 3: isomorphisms.
    for (uint32_t q = 2; q <= 3; ++q) {
        check(res.H[q] == res.H[q + 2], "H^" + std::to_string(q) + " and H^" + std::to_string(q + 2) +
                                            " differ in length " + where);
        check(res.img[q] == res.H[q] && res.img[q] == res.H[q + 2],
              "E_" + std::to_string(q) + " is not an isomorphism " + where);
    }
    // q = 1: surjective, kernel = partial(N_{-I} V).
    check(res.img[1] == res.H[3], "E_1 : H^1 -> H^3 is not surjective " + where);
    res.ker1 = res.H[1] - res.img[1];
    Buf F = P.connecting_cochains(ctx, n);   // (w_1 D) x D, columns f_v
    if (negative_conn) F = Buf(F.rows, F.cols);
    // (hecke-extensions slot 10, WP-8, m-9): under negative_conn, F is the
    // zero matrix by construction, so `is_zero(delta[1] F)` is unconditionally
    // true and `|| negative_conn` never actually short-circuits anything --
    // a vacuous pass, harmless but misleading.  The N4 arm SKIPS this check
    // instead (it is testing that the harness rejects the zero cochain
    // downstream, not this cocycle identity).
    if (!negative_conn) check(is_zero(matmul(ctx, delta[1], F)), "a connecting cochain is not a cocycle " + where);
    const Buf Ft = transpose(F);
    // partial(N V) inside ker E_1: E_1^* f_v in B^3.
    const Buf EFt = transpose(matmul(ctx, E[1], F));
    check(len_rows(ctx, stack({&Brows[3], &EFt}, M.rank(3) * D)) == lenim[2],
          "partial(N_{-I} V) is not in ker E_1 " + where);
    res.conn = len_rows(ctx, stack({&Brows[1], &Ft}, M.rank(1) * D)) - lenim[0];
    // The same length in V alone: length((N V + V^S + V^U) / (V^S + V^U)).
    {
        const Buf rS = integral_model::rho_n(ctx, hecke::Mat22::S(), n);
        const Buf rU = integral_model::rho_n(ctx, hecke::Mat22::ST(), n);
        const Buf rN = integral_model::rho_n(ctx, hecke::Mat22::I().neg(), n);
        const Buf I = identity(ctx, D);
        const Buf VS = kernel_rows(ctx, minus(ctx, rS, I));
        const Buf VU = kernel_rows(ctx, minus(ctx, rU, I));
        Buf Nm = rN;
        for (uint32_t i = 0; i < D; ++i) Nm.at(i, i) = ctx.add(Nm.at(i, i), 1 % ctx.p);
        const Buf NVrows = transpose(Nm);
        res.conn_v = len_rows(ctx, stack({&VS, &VU, &NVrows}, D)) - len_rows(ctx, stack({&VS, &VU}, D));
    }
    const bool kernel_ok = (res.conn == res.ker1);
    if (negative_conn) return res;
    check(kernel_ok, "ker E_1 != partial(N_{-I} V) " + where + ": ker length " + std::to_string(res.ker1) +
                         ", partial(N V) length " + std::to_string(res.conn));
    check(res.conn == res.conn_v, "partial(N_{-I} V) on the model != its length in V " + where);
    if (n % 2 == 1) check(res.ker1 == 0 && res.conn_v == 0, "odd n: E_1 not injective " + where);
    return res;
}

}  // namespace

uint32_t HeckePeriodicityClassValidation() {
    g_checks = 0;
    g_fail = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckePeriodicityClassValidation: [C, lem:bulkperiodic] on the isotropy-resolution model "
                 "[C, def:isomodel]\n";
    try {
        const IsotropyModel M(kQMax);
        check(M.certify().all(), "IsotropyModel certificate");
        const PeriodicityClass P(M);

        // (A) the construction certificate.
        {
            const auto c = P.certify();
            check(c.e_cocycle, "(A) e is a cocycle");
            check(c.e_bockstein, "(A) 12 e = Phi o d_2");
            check(c.e_restrictions, "(A) res_C e = chi_C at every vertex stabiliser");
            check(c.chi_is_zeta, "(A) chi == zeta_+ on finite-order elements");
            check(c.aug_E0, "(A) aug o E_0 = e");
            check(c.chain_map, "(A) E is a chain map over Z[Gamma]");
            check(c.tree_chain_map, "(A) the tree map is a chain map");
            if (!c.first_failure.empty()) std::cerr << "  first failure: " << c.first_failure << "\n";
            std::ostringstream es;
            for (size_t i = 0; i < P.e().size(); ++i) es << (i ? "," : "") << P.e()[i];
            std::cout << "  (A) e on R_2 (19 generators, cells(2) order) = [" << es.str() << "]\n"
                      << "      restrictions: e(e_{0,sigma,2}) for sigma = v0,v1,v2,v4,v6 = ";
            for (uint32_t r = 0; r < M.reps(0).size(); ++r)
                std::cout << (r ? "," : "") << P.e()[M.cell_index(0, r, 2)] << " mod "
                          << M.stabiliser_order(0, r);
            std::cout << "\n      E_0..E_" << P.top() << " built; chain-map generators checked "
                      << c.chain_terms_checked << "; finite-order elements checked "
                      << c.finite_order_elements_checked << "\n      integer sizes: max |coefficient| "
                      << c.max_abs_coefficient << ", max |matrix entry| " << c.max_abs_entry << ", terms "
                      << c.total_terms << ", distinct group elements " << c.distinct_group_elements
                      << "; Xbulk 1-cycle fillings " << P.x_fillings() << " (max rounds " << P.x_max_rounds()
                      << ", max candidate triangles " << P.x_max_triangles() << ")\n";
        }

        // (F) the filling used by E, on random boundaries z = d x, x in R_{q+1}
        // with coefficients +-gamma, gamma a word of length <= 8 in S, T, T^-1:
        // fill(q, z) must return y with d y = z (checked exactly inside fill;
        // re-checked here).  This exercises the Xbulk 1-cycle filling, which
        // E itself happens not to need at q_max = 5 (reported in (A)).
        {
            uint64_t seed = 0x5eed13u;
            auto rnd = [&](uint32_t m) {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                return static_cast<uint32_t>((seed >> 33) % m);
            };
            const hecke::Mat22 gens[3] = {hecke::Mat22::S(), hecke::Mat22::T(), hecke::Mat22::T().inv_unimodular()};
            const uint64_t before = P.x_fillings();
            uint32_t tested = 0;
            for (uint32_t q = 0; q + 1 <= M.q_top() && q <= 4; ++q)
                for (uint32_t trial = 0; trial < 12; ++trial) {
                    integral_model::Chain x = P.zero_chain(q + 1);
                    const uint32_t terms = 1 + rnd(3);
                    for (uint32_t k = 0; k < terms; ++k) {
                        hecke::Mat22 g = hecke::Mat22::I();
                        const uint32_t len = rnd(9);
                        for (uint32_t i = 0; i < len; ++i) g = g * gens[rnd(3)];
                        x[rnd(M.rank(q + 1))].add(integral_model::key_of(g), rnd(2) ? 1 : -1);
                    }
                    const integral_model::Chain z = P.apply_d(q + 1, x);
                    const integral_model::Chain y = P.fill(q, z);
                    check(P.apply_d(q + 1, y) == z, "(F) fill(q, d x) is not a filling at q = " + std::to_string(q));
                    ++tested;
                }
            // An ORIENTED disc (WP-8, review M-2 / m-2): the all-+1 chain the
            // previous version placed on "every triangle gamma tau_r, gamma a
            // word of length <= 3" gives every triangle the SAME coefficient
            // in its stored orientation, so a face shared by two adjacent
            // ball triangles does NOT cancel in d x -- d x carries +-2 there,
            // every ball triangle is adjacent to supp(d x), and fill_x1()
            // never needs a second round (WP-14 review Sec. 2.4, verified by
            // its probe: 0 interior triangles at up to 930 triangles, all
            // rounds = 1).  Here the region is every gamma in SL(2,Z) with
            // |a|,|b|,|c|,|d| <= 2 (finite, det = 1), one triangle per
            // representative, SIGNED by propagating orientation across shared
            // faces from a seed (rep 0, I): the shared face's two LOCAL signs
            // (the (-1)^i of IsotropyModel::faces(), the same rule
            // PeriodicityClass::fill_x1()'s tri_boundary()/adjacent() use
            // internally) fix the neighbour's global sign as
            // orient[u] = -orient[t] * sign_t * sign_u, which makes d x the
            // region's true topological boundary: an edge shared by two ball
            // triangles cancels, one shared by only one does not.
            {
                using XKey = std::pair<uint32_t, integral_model::GKey>;
                auto canon2 = [](const hecke::Mat22& g) {
                    return std::min(integral_model::key_of(g), integral_model::key_of(g.neg()));
                };
                std::vector<hecke::Mat22> region;
                for (int64_t a = -2; a <= 2; ++a)
                    for (int64_t b = -2; b <= 2; ++b)
                        for (int64_t c = -2; c <= 2; ++c)
                            for (int64_t d = -2; d <= 2; ++d) {
                                const hecke::Mat22 g{a, b, c, d};
                                if (g.det() == 1) region.push_back(g);
                            }
                const uint32_t ntri = static_cast<uint32_t>(M.reps(2).size());
                std::map<XKey, std::vector<std::pair<XKey, int64_t>>> edge_of;   // edge -> [(triangle, local sign)]
                for (const auto& g : region)
                    for (uint32_t r = 0; r < ntri; ++r) {
                        const XKey tk{r, canon2(g)};
                        for (const auto& [sign, ft] : M.faces(2, r)) edge_of[{ft.rep, canon2(g * ft.g)}].push_back({tk, sign});
                    }
                std::map<XKey, int64_t> orient;
                std::vector<XKey> queue{{0u, canon2(hecke::Mat22::I())}};
                orient[queue[0]] = 1;
                for (size_t qi = 0; qi < queue.size(); ++qi) {
                    const XKey t = queue[qi];
                    const hecke::Mat22 G = integral_model::mat_of(t.second);
                    for (const auto& [sign, ft] : M.faces(2, t.first)) {
                        const XKey ek{ft.rep, canon2(G * ft.g)};
                        for (const auto& [u, usign] : edge_of[ek]) {
                            if (u == t || orient.count(u)) continue;
                            orient[u] = -orient.at(t) * sign * usign;
                            queue.push_back(u);
                        }
                    }
                }
                integral_model::Chain x = P.zero_chain(2);
                for (const auto& [tk, o] : orient) x[M.cell_index(2, tk.first, 0)].add(tk.second, o);
                const uint32_t interior = [&] {
                    uint32_t n = 0;
                    for (const auto& kv : edge_of) {
                        std::vector<XKey> present;
                        for (const auto& sharer : kv.second)
                            if (orient.count(sharer.first) && std::find(present.begin(), present.end(), sharer.first) == present.end())
                                present.push_back(sharer.first);
                        if (present.size() >= 2) ++n;
                    }
                    return n;
                }();
                std::cout << "  (F) oriented disc: " << orient.size() << " triangles, " << interior
                          << " interior edges (region |entries| <= 2, seed rep 0 at I)\n";
                check(interior > 0, "(F) the oriented disc has no interior edge (M-2)");
                const integral_model::Chain z = P.apply_d(2, x);
                const integral_model::Chain y = P.fill(1, z);
                check(P.apply_d(2, y) == z, "(F) fill(1, d x) is not a filling on the oriented disc");
                ++tested;
            }
            std::cout << "  (F) fill() on " << tested << " random boundaries, q = 0..4, and one oriented disc: Xbulk 1-cycle fillings "
                      << (P.x_fillings() - before) << " (max rounds " << P.x_max_rounds()
                      << ", max candidate triangles " << P.x_max_triangles() << ")\n";
            check(P.x_fillings() > before, "(F) the Xbulk 1-cycle filling was not exercised");
            check(P.x_max_rounds() >= 2, "(F) the growth path (fill_x1 round >= 2) was not exercised (M-2)");
        }

        // (O) the order of [e].
        {
            const U32Ctx Z4 = U32Ctx::prime_power(2, 2);
            const Buf d1 = M.coboundary(Z4, 0, 1);            // C^1 -> C^2 at n = 0
            const Buf B = transpose(d1);
            Buf v(1, M.rank(2));
            for (uint32_t c = 0; c < M.rank(2); ++c) v.at(0, c) = static_cast<uint32_t>(((2 * P.e()[c]) % 4 + 4) % 4);
            const bool six_e_not_cob = len_rows(Z4, stack({&B, &v}, M.rank(2))) > len_rows(Z4, B);
            check(six_e_not_cob, "(O) 6 e is a coboundary (2 e mod 4 in B^2(Z/4))");
            const U32Ctx F3(3);
            const Buf B3 = transpose(M.coboundary(F3, 0, 1));
            Buf w(1, M.rank(2));
            for (uint32_t c = 0; c < M.rank(2); ++c) w.at(0, c) = static_cast<uint32_t>((P.e()[c] % 3 + 3) % 3);
            const bool four_e_not_cob = len_rows(F3, stack({&B3, &w}, M.rank(2))) > len_rows(F3, B3);
            check(four_e_not_cob, "(O) 4 e is a coboundary (e mod 3 in B^2(F_3))");
            const auto G0 = integral_model::AssembleFromModel(M, 0, {2, 3}, kGoodPrime);
            const auto inv = G0.H[2].invariant_factors();
            check(G0.H[2].free_rank == 0 && inv == std::vector<uint64_t>{12}, "(O) H^2(Gamma; Z) != Z/12 (O2)");
            std::cout << "  (O) [e] has order 12: 12 e = delta^1(Phi) over Z, 6 e not a coboundary (mod 4), "
                         "4 e not (mod 3); H^2(Gamma; Z) = "
                      << G0.H[2].invariant_factor_string() << " (M0/O2), so [e] generates it\n";
        }

        // (P) the periodicity clauses, cell by cell.
        uint32_t cells = 0, cells_nonzero_ker = 0, cells_nonzero_H2 = 0;
        for (uint32_t n = 0; n <= kNMax; ++n) {
            const auto G = integral_model::AssembleFromModel(M, n, {2, 3}, kGoodPrime);
            check(G.all_certified(), "M0 certification at n = " + std::to_string(n));
            std::ostringstream line;
            line << "  (P) n=" << n;
            for (size_t pi = 0; pi < G.primes.size(); ++pi) {
                const uint32_t ell = G.primes[pi].ell;
                const uint32_t K = std::max<uint32_t>(2, G.primes[pi].K_max());
                for (uint32_t k : {1u, K}) {
                    const U32Ctx ctx = (k == 1) ? U32Ctx(ell) : U32Ctx::prime_power(ell, k);
                    const auto pred = integral_model::PredictedLengths(G, ell, k);
                    const CellResult r = run_cell(M, P, ctx, n, &pred, false, false);
                    ++cells;
                    if (r.ker1) ++cells_nonzero_ker;
                    if (r.H[2]) ++cells_nonzero_H2;
                    line << " | " << ring_name(ctx) << " H=" << vec_str(r.H) << " E1:" << r.img[1] << "/ker"
                         << r.ker1 << "=dN" << r.conn;
                }
            }
            for (uint32_t ell : kGoodEll) {
                const U32Ctx ctx(ell);
                const CellResult r = run_cell(M, P, ctx, n, nullptr, false, false);
                ++cells;
                if (r.ker1) ++cells_nonzero_ker;
                check(r.H[2] == 0 && r.H[3] == 0 && r.H[4] == 0 && r.H[5] == 0,
                      "good ell: H^q != 0 for some q >= 2 (Claim amalgam(3)) at n = " + std::to_string(n));
                check(r.ker1 == r.H[1], "good ell: ker E_1 != H^1 at n = " + std::to_string(n));
                line << " | " << ring_name(ctx) << " H1=" << r.H[1] << " ker" << r.ker1 << "=dN" << r.conn;
            }
            std::cout << line.str() << "\n";
        }
        std::cout << "  (P) " << cells << " ring cells; ker E_1 != 0 on " << cells_nonzero_ker
                  << ", H^2 != 0 on " << cells_nonzero_H2 << " (the clauses are non-vacuous)\n";

        // (N) negative controls.
        {
            PeriodicityClass Pb = P;
            Pb.perturb_E_for_negative_control(2, 0, 0);
            const auto cb = Pb.certify();
            check(!cb.chain_map, "(N1) a perturbed E entry passes the chain-map certificate");
            PeriodicityClass Pe = P;
            Pe.perturb_e_for_negative_control(M.cell_index(0, 4, 2));
            const auto ce = Pe.certify();
            check(!(ce.e_cocycle && ce.e_bockstein && ce.e_restrictions && ce.aug_E0),
                  "(N2) a perturbed e passes the class certificate");
            // 2 E_2^* over Z/2^K at a cell where H^2 has 2-torsion: not an isomorphism.
            {
                const uint32_t n = 4;
                const auto G = integral_model::AssembleFromModel(M, n, {2}, kGoodPrime);
                const uint32_t K = std::max<uint32_t>(2, G.primes[0].K_max());
                const U32Ctx ctx = U32Ctx::prime_power(2, K);
                const CellResult r = run_cell(M, P, ctx, n, nullptr, true, false);
                check(r.H[2] > 0 && r.img[2] < r.H[2], "(N3) 2 E_2^* over Z/2^K accepted as an isomorphism");
                std::cout << "  (N3) n=4 over " << ring_name(ctx) << ": 2 E_2^* has image length " << r.img[2]
                          << " < length H^2 = " << r.H[2] << " -- refused\n";
            }
            // The zero cochain in place of partial(N V): refused where ker E_1 != 0.
            {
                const uint32_t n = 2;
                const U32Ctx ctx(5);
                const CellResult r = run_cell(M, P, ctx, n, nullptr, false, true);
                check(r.ker1 > 0 && r.conn != r.ker1, "(N4) the zero cochain accepted as partial(N_{-I} V)");
                std::cout << "  (N4) n=2 over F_5: ker E_1 length " << r.ker1 << " != " << r.conn
                          << " (zero connecting cochains) -- refused\n";
            }
            std::cout << "  (N) negative controls: perturbed E fails the chain map ("
                      << (cb.first_failure.empty() ? "?" : cb.first_failure) << "); perturbed e fails ("
                      << (ce.first_failure.empty() ? "?" : ce.first_failure) << ")\n";
        }
    } catch (const std::exception& ex) {
        check(false, std::string("exception: ") + ex.what());
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckePeriodicityClassValidation: " << g_checks << " checks, " << g_fail << " failing ("
              << secs << " s)\n";
    return g_fail;
}
