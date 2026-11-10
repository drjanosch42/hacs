// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S4 -- see HeckeTauTableSelfCheckValidation.h.

#include "HeckeTauTableSelfCheckValidation.h"

#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../LocalSystemSL2Z.h"      // CreateSL2ZFundamentalDomainTriangulation
#include "../SimplicialCohomology.h" // SimplicialComplex

#include <fstream>
#include <iostream>
#include <vector>

namespace {

struct GridPoint {
    uint64_t p;
    uint32_t r;
    uint32_t ell;
    uint32_t n_sym;
    bool bad_case; // p == ell (U_ell regime)
};

// One grid point: construct (the ctor certifies and throws on failure),
// re-assert certify() == 0 explicitly, count the stored chains, check the
// certified bound tier. Returns 0 on success and reports into the summary.
uint32_t run_grid_point(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                        const GridPoint& gp, hecke::TauVertexPolicy policy,
                        std::string& solve_mode_out, uint64_t& bound_out) {
    const hecke::HeckeCosetTables cosets(gp.p, gp.r, gp.ell, gp.n_sym);
    const hecke::HeckeTauTable tab(K, model, cosets, gp.ell, gp.n_sym, policy);

    // Because certify() throws in the constructor, reaching this line
    // already implies a certified table; still assert explicitly (DoD).
    if (tab.certify() != 0) return 1;

    // Count the stored chains (= slots): one per (i, q, sigma).
    const uint32_t m = cosets.num_cosets();
    uint32_t chains = 0;
    for (uint32_t q = 0; q <= 2; ++q) {
        const uint32_t count = K.get_num_q_simplices(q);
        for (uint32_t sigma = 0; sigma < count; ++sigma) {
            for (uint32_t i = 1; i <= m; ++i) {
                const auto& terms = tab.tau(i, q, sigma);
                const auto& xi = tab.xi_terms(i, q, sigma);
                if (xi.size() != terms.size()) return 2; // parallel tables
                // Vertex slots are single coefficient-1 chains ((T3)).
                if (q == 0 && (terms.size() != 1 || terms[0].coeff != 1)) return 3;
                ++chains;
            }
        }
    }
    const uint32_t expected = m * (7 + 11 + 5);
    if (chains != expected) return 4;
    if (gp.p == 2 && gp.r == 1 && chains != 69) return 5; // 3 * 23 ([C, sec:example])

    // Backend tier ([C, thm:backend]): 2 B^2 < 2^63, i.e. B <= 2^31 - 1.
    const uint64_t B = tab.certified_bound_B();
    if (B > 2147483647ull) return 6;
    if (tab.certified_bound_B_tab() > B) return 7;

    solve_mode_out = tab.used_mod_ell_solve() ? "mod-ell" : "Z";
    bound_out = B;
    return 0;
}

} // namespace

uint32_t HeckeTauTableSelfCheckValidation() {
    uint32_t failures = 0;
    uint32_t combinations = 0;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        // DoD grid + the bad cases p = ell in {2, 5}.
        std::vector<GridPoint> grid;
        const std::vector<std::pair<uint64_t, uint32_t>> prs = {
            { 2, 1 }, { 3, 1 }, { 5, 1 }, { 7, 1 }, { 2, 2 }
        };
        const std::vector<uint32_t> ells = { 5, 7, 691, 134217757 };
        const std::vector<uint32_t> ns = { 0, 2, 10 };
        for (const auto& pr : prs) {
            for (uint32_t ell : ells) {
                for (uint32_t n : ns) {
                    grid.push_back(GridPoint{ pr.first, pr.second, ell, n, false });
                }
            }
        }
        for (uint32_t n : ns) grid.push_back(GridPoint{ 2, 1, 2, n, true }); // U_2
        for (uint32_t n : ns) grid.push_back(GridPoint{ 5, 1, 5, n, true }); // U_5

        uint64_t max_B = 0;
        uint32_t mod_ell_points = 0;
        for (const auto& gp : grid) {
            ++combinations;
            std::string mode;
            uint64_t B = 0;
            uint32_t code = 100;
            try {
                code = run_grid_point(K, model, gp, hecke::TauVertexPolicy::GeometricNearest,
                                      mode, B);
            } catch (const std::exception& e) {
                std::cout << "HeckeTauTableSelfCheckValidation: EXCEPTION (construction/"
                             "certification aborted) for p=" << gp.p << " r=" << gp.r
                          << " ell=" << gp.ell << " n=" << gp.n_sym << ": " << e.what()
                          << std::endl;
                ++failures;
                continue;
            }
            if (code != 0) {
                std::cout << "HeckeTauTableSelfCheckValidation: FAILED (code " << code
                          << ") for p=" << gp.p << " r=" << gp.r << " ell=" << gp.ell
                          << " n=" << gp.n_sym << (gp.bad_case ? " (bad case p=ell)" : "")
                          << std::endl;
                ++failures;
                continue;
            }
            if (B > max_B) max_B = B;
            if (mode == "mod-ell") ++mod_ell_points;
            // Report the certified bounds once per (p, r) (the chains, and
            // hence the bounds, are (ell, n)-independent when every triangle
            // fill solved over Z -- which the mode string records).
            if ((gp.bad_case && gp.n_sym == 0) || (gp.ell == 5 && gp.n_sym == 0)) {
                std::cout << "HeckeTauTableSelfCheckValidation: p=" << gp.p << " r=" << gp.r
                          << " ell=" << gp.ell << (gp.bad_case ? " (bad case p=ell)" : "")
                          << ": certified_bound_B = " << B << ", triangle solve mode = "
                          << mode << std::endl;
            }
        }
        std::cout << "HeckeTauTableSelfCheckValidation: max certified_bound_B over the grid = "
                  << max_B << " (int64 tier 2 B^2 < 2^63: "
                  << (max_B <= 2147483647ull ? "OK" : "VIOLATED") << "); "
                  << mod_ell_points << "/" << combinations
                  << " grid points used the mod-ell triangle solve." << std::endl;

        // FixedV6 alternate vertex-assignment policy hook (O12(c)): the
        // table must construct AND certify under the alternate policy too.
        // NO cross-tau comparisons are made (in particular none at ell in
        // {2,3}; contract box after [C, lem:tauexists]).
        const std::vector<GridPoint> policy_points = {
            { 2, 1, 5, 2, false }, { 7, 1, 5, 2, false }, { 5, 1, 7, 10, false }
        };
        for (const auto& gp : policy_points) {
            ++combinations;
            std::string mode;
            uint64_t B = 0;
            try {
                const uint32_t code =
                    run_grid_point(K, model, gp, hecke::TauVertexPolicy::FixedV6, mode, B);
                if (code != 0) {
                    std::cout << "HeckeTauTableSelfCheckValidation: FAILED FixedV6 (code "
                              << code << ") for p=" << gp.p << " r=" << gp.r
                              << " ell=" << gp.ell << " n=" << gp.n_sym << std::endl;
                    ++failures;
                }
            } catch (const std::exception& e) {
                std::cout << "HeckeTauTableSelfCheckValidation: EXCEPTION (FixedV6) for p="
                          << gp.p << " r=" << gp.r << " ell=" << gp.ell << " n=" << gp.n_sym
                          << ": " << e.what() << std::endl;
                ++failures;
            }
        }

        // Human-readable p = 2 table ([C, sec:example]: small enough to
        // inspect by hand) -> docs/implementation/tau_table_p2.txt.
        //
        // (slot-14 WP-2, [C3]/§3 item 2 audit) DELIBERATELY does NOT consult
        // OC_SOURCE_DIR here (unlike MatrixRetirementValidation /
        // SmallEllPolicyScopeValidation / ContractCensusValidation, which only
        // READ the tree): this writer's target is a file TRACKED by git, so
        // resolving it via OC_SOURCE_DIR would let an off-checkout run (e.g. a
        // cluster case directory on /scratch, built from a detached clone)
        // write into that clone's tracked tree -- exactly the pattern the
        // repository-root fallback exists to enable for read-only guards, and
        // exactly what must NOT happen for a writer. The relative path
        // "docs/implementation/tau_table_p2.txt" only ever resolves under the
        // process's own working directory, so it opens the tracked file iff
        // the CWD happens to be the repository root (the in-source build's
        // default) and is a plain miss everywhere else -- there is no path by
        // which this call can write into a DIFFERENT checkout than the one the
        // process is running from.
        //
        // When the CWD is not the repository root nothing is written (slot-14
        // WP-16, audit S-5: the former fallback wrote into /tmp); a NOTE says
        // so, and this never affects the validation result either way.
        try {
            const hecke::HeckeCosetTables cosets2(2, 1, 5, 0);
            const hecke::HeckeTauTable tab2(K, model, cosets2, 5, 0,
                                            hecke::TauVertexPolicy::GeometricNearest);
            const std::string body =
                "Stage S4 -- certified tau-table for p = 2 (g = diag(1,2), m = 3),\n"
                "written by HeckeTauTableSelfCheckValidation for human review\n"
                "([C, sec:example]; ell = 5, n_sym = 0 shown -- the chains are\n"
                "ell/n-independent here since every triangle fill solved over Z).\n\n" +
                tab2.to_string();
            std::ofstream out("docs/implementation/tau_table_p2.txt");
            if (out) {
                out << body;
                std::cout << "HeckeTauTableSelfCheckValidation: wrote "
                             "docs/implementation/tau_table_p2.txt" << std::endl;
            } else {
                // (slot-14 WP-16, audit S-5) no off-root write at all: the former
                // fallback wrote a per-pid copy under the system temp directory (/tmp),
                // against the standing no-/tmp rule, and nothing removed it.  The
                // object is tab2.to_string(), printable on demand.
                std::cout << "HeckeTauTableSelfCheckValidation: NOTE -- the working "
                             "directory is not the repository root, so the tracked "
                             "docs/implementation/tau_table_p2.txt was NOT written "
                             "(run from the repository root to regenerate it); nothing "
                             "else was written." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "HeckeTauTableSelfCheckValidation: EXCEPTION while writing the p=2 "
                         "table: " << e.what() << std::endl;
            ++failures;
        }
    } catch (const std::exception& e) {
        std::cout << "HeckeTauTableSelfCheckValidation: EXCEPTION during setup: " << e.what()
                  << std::endl;
        return 1000;
    }

    std::cout << "HeckeTauTableSelfCheckValidation: " << (combinations - failures) << "/"
              << combinations << " grid points passed." << std::endl;
    return failures;
}
