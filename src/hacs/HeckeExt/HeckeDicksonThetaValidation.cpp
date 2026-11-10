// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 11, WP-5 -- HeckeDicksonThetaValidation (standard) and
// HeckeDicksonThetaCoverageSlowValidation (slow).  See HeckeDicksonTheta.h.
//
// Arms (standard):
//   (R) the Q_n relation list: same rows / tags / simplices / g_r as Sym^n's,
//       every matrix (ell+1) x (ell+1) and equal to rho_Q(g_r);
//   (W) per grid cell: the fibre-level certificate of the cell (pi L. = 0,
//       rank pi, pi E = I, preservation of L . Sym^n on seeded, singular and
//       relation samples, rho_Q(g) pi = pi rho_m(g), multiplicativity);
//   (N1) the negative seam x^ell y . Sym^n: some sample breaks it, per cell;
//   (N2) a perturbed delta fails the exactness check, per applicable cell;
//   (Hk) HeckeOperatorOnK on Q_n (FibreRepresentation::QuotientL): T_2 and
//        U_ell computed at two cells, the operator's own certificates
//        (two-route identity, relation / cocycle / Theta_eq checks) held.
// Claim arms (after [C] v1.11.0-pre.1, merge 3a0bbbf; plan Sec. 3.0 rule 21):
//   (E)  [C, thm:thetales](1), [C, prop:dicksonH0]: the full certificate --
//        the LES exact at the six spots, H^2 = 0 on the three terms, the
//        alternating sum 0, H^0 spanned by the classes of L^a D^b;
//   (D)  [C, thm:thetales](3)-(4), [C, lem:Qnfunctions](2): dim H^0(Q_n),
//        rank delta, rank / coker theta on H^0, rank theta on H^1, dim H^1(Q_n);
//   (Q0) [C, lem:Qnfunctions](2): 1_n is an invariant class of Q_n.

#include "HeckeDicksonTheta.h"

#include "../Hecke/HeckeSettings.h"
#include "../ValidationRegistry.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

using hecke_ext::DicksonThetaCell;
using hecke_ext::ThetaMat;
using hecke_ext::ThetaTerm;

struct Cell {
    uint32_t n, ell;
};

// The standard grid of the plan (WP-5 scope 4).
std::vector<Cell> StandardGrid() {
    std::vector<Cell> g;
    for (uint32_t ell : { 5u, 7u }) {
        for (uint32_t n = 0; n + ell + 1 <= 72; n += 2) {
            // even n <= 60, plus every n of the semigroup <ell+1, ell^2-ell> with n + ell + 1 <= 72
            const bool semigroup = !hecke_ext::DicksonExponents(n, ell).empty();
            if (n <= 60 || semigroup) g.push_back({ n, ell });
        }
    }
    for (uint32_t ell : { 11u, 13u })
        for (uint32_t n = 0; n <= 40; n += 2) g.push_back({ n, ell });
    return g;
}

std::vector<Cell> SlowGrid() {
    std::vector<Cell> g;
    for (uint32_t ell : { 5u, 7u, 11u, 13u })
        for (uint32_t n = 0; n <= 200; n += 2) g.push_back({ n, ell });
    return g;
}

struct Counter {
    uint64_t checks = 0, failures = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failures;
            std::cout << "    FAIL: " << what << "\n";
        }
    }
};

// Fibre-level codes of DicksonThetaCertificate (1..7).
bool FibreCodeOk(uint32_t code) { return code == 0 || code > 7; }

// (N2) A perturbed delta must fail the exactness check.  Two perturbations:
// if some class e of H^1(Sym^n) has iota(1) e != 0, add e to delta's first
// column (im delta then leaves ker iota(1)); otherwise replace delta by 0.
bool PerturbedDeltaFails(const DicksonThetaCell& c, std::string& kind) {
    const uint32_t h0q = c.h(ThetaTerm::Quot, 0), h1n = c.h(ThetaTerm::SymN, 1);
    ThetaMat d = c.delta(0);
    const ThetaMat& th = c.iota(1);
    int64_t e = -1;
    for (uint32_t j = 0; j < h1n && e < 0; ++j)
        for (uint32_t i = 0; i < th.rows; ++i)
            if (th.at(i, j) != 0) { e = j; break; }
    if (e >= 0) {
        kind = "column";
        d.at(static_cast<uint32_t>(e), 0) = (d.at(static_cast<uint32_t>(e), 0) + 1) % c.ell();
    } else {
        kind = "zero";
        d = ThetaMat(d.rows, d.cols, d.p);
    }
    (void)h0q;
    const std::vector<uint32_t> dims = { c.h(ThetaTerm::SymN, 0), c.h(ThetaTerm::SymM, 0), c.h(ThetaTerm::Quot, 0),
                                         c.h(ThetaTerm::SymN, 1), c.h(ThetaTerm::SymM, 1), c.h(ThetaTerm::Quot, 1) };
    const auto spots = hecke_ext::CheckSequenceExactness(c.iota(0), c.pi(0), d, c.iota(1), c.pi(1), c.delta(1), dims);
    for (const auto& s : spots)
        if (!s.exact()) return true;
    return false;
}

void RunGrid(const std::vector<Cell>& grid, Counter& ct, bool verbose) {
    uint32_t n2_applicable = 0;
    for (const Cell& g : grid) {
        const DicksonThetaCell c(g.n, g.ell);
        const auto& cert = c.certificate();
        if (verbose) std::cout << "  " << c.summary() << "\n";
        ct.check(FibreCodeOk(cert.code), "(W) fibre certificate at " + c.summary());
        ct.check(cert.xelly_first_failure >= 0, "(N1) x^ell y . Sym^n preserved by every sample at " + c.summary());
        // (E) [C, thm:thetales] / [C, prop:dicksonH0]: the full certificate -- exactness at the six spots
        //     (im = ker as subspaces, both inclusions), H^2 = 0 on the three terms, the alternating
        //     dimension sum, and H^0(Sym^n), H^0(Sym^m) spanned by the classes of L^a D^b.
        ct.check(cert.code == 0, "(E) certificate at " + c.summary());
        // (D) [C, lem:Qnfunctions], [C, thm:thetales]: dim H^0(Q_n) = [(ell-1) | (n+2)] and
        //     rank delta = [(ell-1) | (n+2)] - [ell(ell-1) | (n+ell+1)].
        {
            const uint32_t e = g.ell;
            const uint32_t q0 = ((g.n + 2) % (e - 1) == 0) ? 1u : 0u;
            const uint32_t q1 = ((g.n + e + 1) % (e * (e - 1)) == 0) ? 1u : 0u;
            ct.check(c.h(ThetaTerm::Quot, 0) == q0, "(D) dim H^0(Q_n) != [(ell-1)|(n+2)] at " + c.summary());
            const uint32_t rk_delta = cert.code == 0 ? hecke_ext::ThetaRank(c.delta(0)) : 0u;
            ct.check(cert.code == 0 && rk_delta + q1 == q0, "(D) rank delta != closed form at " + c.summary());
            // [C, thm:thetales](4): the remaining closed forms (n even; d_k of [C, cl:dimH1], d_2 = 0).
            if (cert.code == 0) {
                auto dk = [](uint32_t k) -> uint32_t { return (k % 12 == 2) ? k / 12 : k / 12 + 1; };
                const uint32_t rk_theta0 = hecke_ext::ThetaRank(c.iota(0)), rk_theta1 = hecke_ext::ThetaRank(c.iota(1));
                ct.check(rk_theta0 == c.h(ThetaTerm::SymN, 0) && c.h(ThetaTerm::SymM, 0) - rk_theta0 == q1,
                         "(D) rank / coker theta|H^0 != closed form at " + c.summary());
                ct.check(rk_theta1 + rk_delta == c.h(ThetaTerm::SymN, 1),
                         "(D) rank theta|H^1 != h^1(n) - rank delta at " + c.summary());
                const uint32_t h1q = 2 * (dk(g.n + e + 3) - dk(g.n + 2)) + q0;
                ct.check(c.h(ThetaTerm::Quot, 1) == h1q, "(D) dim H^1(Q_n) != closed form at " + c.summary());
            }
            // (Q0) the invariant class 1_n = u_0 - u_{ell-1} + u_{n+ell+1} of [C, lem:Qnfunctions](2),
            //      as a constant 0-cochain on the complement basis of [C, def:Qn].
            if (q0 == 1) {
                const uint32_t dq = e + 1, N0 = c.K().get_num_q_simplices(0);
                std::vector<uint32_t> y(static_cast<size_t>(N0) * dq, 0);
                for (uint32_t s = 0; s < N0; ++s) {
                    y[static_cast<size_t>(s) * dq + 0] = 1;
                    y[static_cast<size_t>(s) * dq + (e - 1)] = e - 1;
                    y[static_cast<size_t>(s) * dq + e] = 1;
                }
                const bool cocycle = c.is_equivariant_cocycle(ThetaTerm::Quot, 0, y);
                ct.check(cocycle, "(Q0) u_0 - u_{ell-1} + u_N is not a 0-cocycle of Q_n at " + c.summary());
                const std::vector<uint32_t> cls = cocycle ? c.class_of(ThetaTerm::Quot, 0, y) : std::vector<uint32_t>();
                ct.check(!cls.empty() && cls[0] != 0, "(Q0) u_0 - u_{ell-1} + u_N has class 0 in H^0(Q_n) at " + c.summary());
            }
        }
        if (cert.code == 0 && c.h(ThetaTerm::Quot, 0) > 0 && c.h(ThetaTerm::SymN, 1) > 0) {
            std::string kind;
            ++n2_applicable;
            ct.check(PerturbedDeltaFails(c, kind), "(N2) perturbed delta (" + kind + ") still exact at " + c.summary());
        }
    }
    std::cout << "  grid: " << grid.size() << " cells, (N2) applicable at " << n2_applicable << "\n";
    ct.check(n2_applicable > 0, "(N2) never applicable on the grid");
}

void RelationListArm(Counter& ct) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    for (const Cell g : { Cell{ 10, 5 }, Cell{ 12, 7 }, Cell{ 0, 11 } }) {
        const auto s = GenerateSL2ZRelationsWithGroupData(K, g.n, g.ell);
        const auto q = GenerateSL2ZQuotientLRelationsWithGroupData(K, g.n, g.ell);
        bool same = s.size() == q.size();
        for (size_t r = 0; same && r < s.size(); ++r) {
            same = s[r].g_r == q[r].g_r && s[r].rel.tag == q[r].rel.tag && s[r].rel.q_dim == q[r].rel.q_dim &&
                   s[r].rel.s_lhs_idx_in_Kq == q[r].rel.s_lhs_idx_in_Kq &&
                   s[r].rel.s_rhs_idx_in_Kq == q[r].rel.s_rhs_idx_in_Kq &&
                   q[r].rel.M.get_rows() == g.ell + 1 && q[r].rel.M.get_cols() == g.ell + 1;
        }
        ct.check(same, "(R) Q_n relation list shape at n=" + std::to_string(g.n) + " ell=" + std::to_string(g.ell));
    }
}

void HeckeArm(Counter& ct) {
    for (const Cell g : { Cell{ 10, 5 }, Cell{ 20, 7 } }) {
        const DicksonThetaCell c(g.n, g.ell);
        for (uint64_t p : { uint64_t{ 2 }, static_cast<uint64_t>(g.ell) }) {
            bool ok = true;
            std::string what;
            try {
                const DicksonThetaCell::HeckeTriple ht = c.hecke_triple(p);
                const auto cb = ht.op_q->certificates();
                ok = cb.routes_bit_identical && cb.theta_eq_residual_max == 0 &&
                     ht.op_q->representation() == FibreRepresentation::QuotientL;
                const auto& A = ht.op_q->get_H_matrices();
                for (uint32_t q = 0; q < 3; ++q)
                    ok = ok && A[q].rows == c.h(ThetaTerm::Quot, q) && A[q].cols == c.h(ThetaTerm::Quot, q);
            } catch (const std::exception& e) {
                ok = false;
                what = e.what();
            }
            ct.check(ok, "(Hk) HeckeOperatorOnK(quotient_L) p=" + std::to_string(p) + " at " + c.summary() + " " + what);
        }
    }
}

// Refusal at ell in {2, 3}: a mathematical precondition ([E] sec:thetaobj (h)),
// so a std::logic_error naming lem:h2torsion (console exit 3), never
// hecke::NotImplementedInThisBuild (exit 5).  Slot-11 WP-17, F-2.
void RefusalArm(Counter& ct) {
    for (uint32_t ell : { 2u, 3u }) {
        bool refused = false;
        try {
            const DicksonThetaCell c(4, ell);
        } catch (const std::logic_error& e) {
            refused = std::string(e.what()).find("lem:h2torsion") != std::string::npos;
        } catch (const std::exception&) {
            refused = false;  // a NotImplementedInThisBuild (runtime_error) is the wrong kind
        }
        ct.check(refused, "(X) DicksonThetaCell refuses ell = " + std::to_string(ell) +
                              " by std::logic_error naming lem:h2torsion");
    }
}

uint32_t Finish(const char* name, const Counter& ct, std::chrono::steady_clock::time_point t0) {
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %llu checks, %llu failing (%.1f s)\n", name, static_cast<unsigned long long>(ct.checks),
                static_cast<unsigned long long>(ct.failures), s);
    return ct.failures == 0 ? 0u : 1u;
}

} // namespace

uint32_t HeckeDicksonThetaValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Counter ct;
    RelationListArm(ct);
    RefusalArm(ct);
    RunGrid(StandardGrid(), ct, false);
    HeckeArm(ct);
    return Finish("HeckeDicksonThetaValidation", ct, t0);
}

uint32_t HeckeDicksonThetaCoverageSlowValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Counter ct;
    RunGrid(SlowGrid(), ct, false);
    return Finish("HeckeDicksonThetaCoverageSlowValidation", ct, t0);
}
