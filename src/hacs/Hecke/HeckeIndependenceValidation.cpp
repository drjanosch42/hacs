// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S8V -- see HeckeIndependenceValidation.h for the full oracle O12
// statement and, for item (d), the precise documentation of what is and is
// not reachable through the public API.

#include "HeckeIndependenceValidation.h"

#include "ExactInt.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Matrix.h"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using hecke::Mat22;

// ===========================================================================
// Fixed-seed xorshift64 PRNG (deterministic across runs and across
// HECKE_PURE_INTEGER=ON/OFF builds -- no time-based or address-based seeding
// anywhere in this file, required for oracle O12(e)'s stdout byte-identity).
// ===========================================================================
struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
};

// A short (2-4 letter) pseudorandom word in {L = [[1,0],[p,1]], T = [[1,1],[0,1]]},
// both generators of Gamma_0(p): L's lower-left entry is p == 0 mod p by
// construction; T's lower-left entry is 0 trivially. Gamma_0(p) is a group
// (closed under multiplication and det 1), so ANY product of these two
// matrices lies in Gamma_0(p); only positive powers are used (no inverse
// bookkeeping needed) -- see the header note for O12(b).
Mat22 RandomGamma0Word(uint64_t p_hecke, XorShift64& rng) {
    const Mat22 L{1, 0, static_cast<int64_t>(p_hecke), 1};
    const Mat22 T = Mat22::T();
    Mat22 result = Mat22::I();
    const uint32_t word_len = 2 + static_cast<uint32_t>(rng.next() % 3); // 2..4 letters
    for (uint32_t k = 0; k < word_len; ++k) {
        const Mat22& gen = ((rng.next() & 1) != 0) ? L : T;
        result = result * gen;
    }
    return result;
}

// ===========================================================================
// Pipeline fixture: builds L = Sym^n(F_ell^2) on the 7-vertex model and
// computes H^*_eq(K,R;L), exactly per the standard pipeline (identical
// construction to HeckeOracleValidation.cpp / HeckeCrossCheckValidation.cpp's
// fixtures).
// ===========================================================================
struct CohomologyFixture {
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;

    CohomologyFixture(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell)
        : lssc(K, MakeRelations(K, n_sym, ell), n_sym + 1, ell), coh(&lssc) {
        coh.compute_cohomology();
    }

private:
    static std::vector<LocalSystemSimplicialComplex::SimplexRelation> MakeRelations(
            const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
            rels.push_back(datum.rel);
        }
        return rels;
    }
};

std::string SettingTag(uint64_t p_hecke, uint32_t ell, uint32_t n_sym) {
    std::ostringstream oss;
    oss << "(p,ell,n)=(" << p_hecke << "," << ell << "," << n_sym << ")";
    return oss.str();
}

// ===========================================================================
// Battery driver: one instance per run, accumulates failures across all
// settings and all checks (a)-(d); one bad case never aborts the rest
// (matching HeckeOracleValidation.cpp's established style).
// ===========================================================================
class IndependenceContext {
public:
    IndependenceContext(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model)
        : K_(K), model_(model) {}

    uint32_t failures() const { return failures_; }

    void RunSetting(uint64_t p_hecke, uint32_t ell, uint32_t n_sym) {
        const std::string tag = SettingTag(p_hecke, ell, n_sym);
        std::cout << "HeckeIndependenceValidation: === setting " << tag << " ===" << std::endl;
        try {
            CohomologyFixture cf(K_, n_sym, ell);

            // The canonical/baseline pipeline: canonical coset reps,
            // GeometricNearest tau policy. compute() itself already asserts
            // the O12(a) route-1-vs-route-2 bit-identity internally (throws
            // otherwise).
            const hecke::HeckeCosetTables std_cosets(p_hecke, /*r=*/1, ell, n_sym);
            const hecke::HeckeTauTable std_tau(K_, model_, std_cosets, ell, n_sym);
            hecke::HeckeOperatorOnK std_op(&cf.coh, &std_cosets, &std_tau);
            std_op.compute();

            CheckA_RouteAndApplyTheta(std_op, tag);
            CheckB_CustomCosetReps(cf, std_cosets, std_op, p_hecke, ell, n_sym, tag);
            CheckC_TauVertexPolicy(cf, std_cosets, std_op, ell, n_sym, tag);
            CheckD_GWeakForm(cf, std_cosets, std_op, p_hecke, ell, n_sym, tag);
        } catch (const std::exception& e) {
            Fail("EXCEPTION building the canonical pipeline at " + tag + ": " + e.what());
        }
    }

private:
    const SimplicialComplex& K_;
    const hecke::SL2ZTessellationTables& model_;
    uint32_t failures_ = 0;

    void Fail(const std::string& msg) {
        ++failures_;
        std::cout << "HeckeIndependenceValidation: FAIL: " << msg << std::endl;
    }
    void Ok(const std::string& msg) {
        std::cout << "HeckeIndependenceValidation: OK: " << msg << std::endl;
    }

    void ReportFlatDiff(const std::string& what, uint32_t q, const FlatMatrix& a,
                        const FlatMatrix& b, const std::string& tag) {
        std::ostringstream oss;
        oss << what << " at " << tag << ", degree q=" << q << ":\n"
            << "  matrix A:\n" << a.to_string()
            << "  matrix B:\n" << b.to_string();
        Fail(oss.str());
    }

    void ReportMatrixDiff(const std::string& what, uint32_t q, const Matrix& a,
                          const Matrix& b, const std::string& tag) {
        std::ostringstream oss;
        oss << what << " at " << tag << ", degree q=" << q << ":\n"
            << "  matrix A:\n" << a.to_string()
            << "  matrix B:\n" << b.to_string();
        Fail(oss.str());
    }

    // -----------------------------------------------------------------------
    // O12(a): the second half of route bit-identity -- apply_Theta(q, e_b)
    // must reproduce column b of get_Theta_flat_matrices()[q] for every
    // standard basis vector e_b, every degree q. (The first half -- fused vs
    // route-2 bit-identity -- is already asserted internally by compute(),
    // which the caller has already run successfully by the time this is
    // called.) Also spot-checks the JSON certificate's
    // "routes_bit_identical":true flag as an independent re-exercise through
    // a different public accessor.
    // -----------------------------------------------------------------------
    void CheckA_RouteAndApplyTheta(const hecke::HeckeOperatorOnK& op, const std::string& tag) {
        try {
            const std::string json = op.to_json();
            if (json.find("\"routes_bit_identical\":true") == std::string::npos) {
                Fail("O12(a): to_json() does not report routes_bit_identical:true at " + tag);
                return;
            }

            const auto& theta_flat = op.get_Theta_flat_matrices();
            for (uint32_t q = 0; q < theta_flat.size(); ++q) {
                const uint32_t dim = theta_flat[q].get_rows();
                if (theta_flat[q].get_cols() != dim) {
                    Fail("O12(a): Theta_flat[" + std::to_string(q) + "] is not square at " + tag);
                    return;
                }
                const uint32_t ell = theta_flat[q].get_p();
                for (uint32_t b = 0; b < dim; ++b) {
                    FlatMatrix e_b(dim, 1, ell);
                    if (dim > 0) e_b.set_entry(b, 0, 1);
                    const FlatMatrix y = op.apply_Theta(q, e_b);
                    for (uint32_t r = 0; r < dim; ++r) {
                        if (y.get_entry(r, 0) != theta_flat[q].get_entry(r, b)) {
                            std::ostringstream oss;
                            oss << "O12(a): apply_Theta(q=" << q << ", e_" << b
                                << ") disagrees with get_Theta_flat_matrices()[" << q
                                << "] at row " << r << " (" << tag << "): apply_Theta gave "
                                << y.get_entry(r, 0) << ", stored column gave "
                                << theta_flat[q].get_entry(r, b);
                            Fail(oss.str());
                            return;
                        }
                    }
                }
            }
            Ok("O12(a): routes_bit_identical:true, and apply_Theta(q,.) reproduces "
               "get_Theta_flat_matrices() column-by-column on the full standard basis, "
               "all q, at " + tag);
        } catch (const std::exception& e) {
            Fail("O12(a): EXCEPTION at " + tag + ": " + e.what());
        }
    }

    // -----------------------------------------------------------------------
    // O12(b): custom coset representatives s_i' = h_i s_i, h_i pseudorandom
    // (fixed seed) in Gamma_0(p). Rebuilds the WHOLE chain (cosets -> tau ->
    // operator) against the custom cosets, per the task's instruction ("If
    // the tau-table ctor requires the cosets it was built with, rebuild the
    // whole chain consistently").
    //
    // IMPORTANT SUBTLETY, discovered empirically and confirmed mathematically
    // (documented at length here because it changes what can be legitimately
    // hard-asserted through the PUBLIC API, not because the pipeline is
    // buggy):
    //
    // [C, thm:welldef](a)'s proof of cochain-level bit-identity under
    // s_i -> h_i s_i is a statement about reusing THE SAME tau (only the
    // representative LABELS change; tau itself, as an abstract map, is
    // untouched). HeckeTauTable's public constructor offers no "relabel an
    // existing tau" hook -- every HeckeTauTable object performs its OWN
    // independent geometric construction (vertex reduction via
    // reduce_to_fundamental_domain + choose_vertex) from whichever
    // HeckeCosetTables it is given. Two independently-constructed tau tables
    // are each certified valid ([C, obl:tau]; the constructor throws
    // otherwise) but are, by [C, thm:welldef](b) / lem:tauexists, only
    // GUARANTEED to be chain-homotopic -- not bit-identical -- since
    // different tau choices are exactly what (b) covers. So "rebuild the
    // whole chain from custom cosets" necessarily exercises (a) and (b)
    // TOGETHER, and the only invariant the contract actually backs for that
    // combination is agreement on COHOMOLOGY (the composition of (a)'s exact
    // invariance with (b)'s homotopy invariance).
    //
    // This was verified directly (not merely argued): (i) with h_i := I for
    // every i (a "custom" rebuild using representatives literally equal to
    // the canonical ones -- exercising the custom-ctor/rebuild code PATH
    // with no actual representative change), Theta_flat reproduces the
    // canonical build bit-for-bit in EVERY grid setting -- proving the
    // custom-reps machinery itself introduces no bug; (ii) with genuinely
    // nontrivial pseudorandom h_i, Theta_flat sometimes differs (observed at
    // 3 of the 4 grid settings for the fixed seeds used here) while A^(q)
    // (cohomology level) agreed at ALL 4 settings without exception -- the
    // signature predicted by (a)+(b) combining to a cohomology-only
    // guarantee, not a code defect surfacing only for "unlucky" h_i.
    //
    // Therefore this check HARD-ASSERTS exactly what the contract backs for
    // this reachable construction:
    //   (1) the h_i = I sanity rebuild reproduces Theta_flat bit-identically
    //       (tests the custom-ctor/rebuild machinery's own correctness and
    //       determinism -- a real bug here, e.g. an ordering slip in
    //       apply_custom_reps, WOULD be a genuine defect);
    //   (2) A^(q) (cohomology level) is bit-identical between the canonical
    //       and the genuinely-nontrivial-h_i custom build, every q -- the
    //       actual mathematical content of [C, thm:welldef](a) as reachable
    //       through this API.
    // It additionally RECORDS (does not hard-fail) whether Theta_flat
    // happened to also agree bit-for-bit under the nontrivial h_i, since
    // that finer identity is a real but NOT contract-mandated bonus here
    // (contrast with a literal "same tau, relabeled s_i" implementation,
    // which this codebase does not expose).
    // -----------------------------------------------------------------------
    void CheckB_CustomCosetReps(CohomologyFixture& cf, const hecke::HeckeCosetTables& std_cosets,
                                const hecke::HeckeOperatorOnK& std_op, uint64_t p_hecke,
                                uint32_t ell, uint32_t n_sym, const std::string& tag) {
        try {
            const uint32_t m = std_cosets.num_cosets();

            // --- (1) Sanity: h_i = I for all i (custom-ctor path, no actual
            // representative change) must reproduce Theta_flat bit-for-bit. ---
            {
                std::vector<Mat22> trivial_s(m);
                for (uint32_t i = 1; i <= m; ++i) trivial_s[i - 1] = std_cosets.s(i);
                const hecke::HeckeCosetTables trivial_cosets(p_hecke, /*r=*/1, ell, n_sym, trivial_s);
                const hecke::HeckeTauTable trivial_tau(K_, model_, trivial_cosets, ell, n_sym);
                hecke::HeckeOperatorOnK trivial_op(&cf.coh, &trivial_cosets, &trivial_tau);
                trivial_op.compute();
                const auto& std_theta = std_op.get_Theta_flat_matrices();
                const auto& trivial_theta = trivial_op.get_Theta_flat_matrices();
                if (std_theta.size() != trivial_theta.size()) {
                    Fail("O12(b) sanity (h_i=I): Theta_flat vector size mismatch at " + tag);
                    return;
                }
                for (uint32_t q = 0; q < std_theta.size(); ++q) {
                    if (!(std_theta[q] == trivial_theta[q])) {
                        ReportFlatDiff("O12(b) SANITY FAILURE (h_i=I, custom-ctor path "
                                       "reproduced with representatives IDENTICAL to canonical): "
                                       "Theta_flat DIFFERS -- this indicates a bug in the "
                                       "custom-reps rebuild machinery itself",
                                       q, std_theta[q], trivial_theta[q], tag);
                        return;
                    }
                }
            }

            // --- (2) Genuinely nontrivial pseudorandom h_i in Gamma_0(p). ---
            std::vector<Mat22> custom_s(m);
            // Seed mixes (p,ell,n) deterministically so different settings
            // exercise different (but reproducible) h_i's.
            XorShift64 rng(0xC0FFEEULL ^ (p_hecke * 1000003ULL) ^
                           (static_cast<uint64_t>(ell) * 97ULL) ^
                           (static_cast<uint64_t>(n_sym) * 131071ULL));
            for (uint32_t i = 1; i <= m; ++i) {
                const Mat22 h_i = RandomGamma0Word(p_hecke, rng);
                custom_s[i - 1] = h_i * std_cosets.s(i);
            }

            const hecke::HeckeCosetTables custom_cosets(p_hecke, /*r=*/1, ell, n_sym, custom_s);
            const hecke::HeckeTauTable custom_tau(K_, model_, custom_cosets, ell, n_sym);
            hecke::HeckeOperatorOnK custom_op(&cf.coh, &custom_cosets, &custom_tau);
            custom_op.compute(); // asserts its OWN internal route bit-identity too

            // HARD ASSERT: cohomology-level A^(q) invariance -- the actual
            // mathematical content of thm:welldef(a) as reachable here.
            const auto& A_std = std_op.get_H_matrices();
            const auto& A_custom = custom_op.get_H_matrices();
            if (A_std.size() != A_custom.size()) {
                Fail("O12(b): get_H_matrices() size mismatch at " + tag);
                return;
            }
            for (uint32_t q = 0; q < A_std.size(); ++q) {
                if (!(A_std[q] == A_custom[q])) {
                    ReportMatrixDiff("O12(b): A^(q) DIFFERS between canonical and custom "
                                     "(h_i s_i, h_i in Gamma_0(p), nontrivial) coset "
                                     "representatives -- GENUINE WELL-DEFINEDNESS FAILURE",
                                     q, A_std[q], A_custom[q], tag);
                    return;
                }
            }

            // RECORDED (not hard-asserted): does the finer cochain-level
            // Theta_flat identity also happen to hold for these nontrivial
            // h_i? Per the header note, this is NOT contract-mandated for a
            // from-scratch tau rebuild -- only recorded for transparency.
            const auto& std_theta = std_op.get_Theta_flat_matrices();
            const auto& custom_theta = custom_op.get_Theta_flat_matrices();
            bool theta_bitidentical = (std_theta.size() == custom_theta.size());
            if (theta_bitidentical) {
                for (uint32_t q = 0; q < std_theta.size(); ++q) {
                    if (!(std_theta[q] == custom_theta[q])) { theta_bitidentical = false; break; }
                }
            }
            Ok("O12(b): sanity (h_i=I) bit-identical; A^(q) bit-identical under genuinely "
               "nontrivial pseudorandom Gamma_0(p) coset representatives, all q, at " + tag +
               ". [recorded, not asserted -- see header note] Theta_flat (cochain level) under "
               "the nontrivial h_i also happened to be bit-identical: " +
               std::string(theta_bitidentical ? "YES" : "NO") +
               " (a from-scratch tau rebuild is only contract-guaranteed to agree with the "
               "canonical build up to chain homotopy, i.e. on cohomology, not bit-for-bit on "
               "cochains -- see [C, thm:welldef](a)-(b)).");
        } catch (const std::exception& e) {
            Fail("O12(b): EXCEPTION at " + tag + ": " + e.what());
        }
    }

    // -----------------------------------------------------------------------
    // O12(c): TauVertexPolicy::FixedV6 and TauVertexPolicy::GeometricNearestLegacy
    // vs the binding GeometricNearest (same canonical cosets). Every grid
    // setting has ell >= 5 (13, 691, 101), so asserted unconditionally, no
    // gating. Only the COHOMOLOGY-level A^{(q)} matrices are compared (the
    // contract only guarantees tau-independence on cohomology,
    // [C, thm:welldef](b); Theta_flat may legitimately differ between tau
    // policies and is deliberately NOT compared here). The Legacy arm is the
    // pre-2026-09-16 canonical tau (ascending-index tie-break + BFS path
    // templates): its equality here is the in-process form of the golden
    // re-baseline's claim that the iota-symmetric tau moved nothing on H^q.
    // -----------------------------------------------------------------------
    void CheckC_TauVertexPolicy(CohomologyFixture& cf, const hecke::HeckeCosetTables& std_cosets,
                                const hecke::HeckeOperatorOnK& std_op, uint32_t ell,
                                uint32_t n_sym, const std::string& tag) {
        const std::pair<hecke::TauVertexPolicy, const char*> arms[2] = {
            { hecke::TauVertexPolicy::FixedV6, "FixedV6" },
            { hecke::TauVertexPolicy::GeometricNearestLegacy, "GeometricNearestLegacy" },
        };
        for (const auto& arm : arms) {
            try {
                const hecke::HeckeTauTable tau_alt(K_, model_, std_cosets, ell, n_sym, arm.first);
                hecke::HeckeOperatorOnK op_alt(&cf.coh, &std_cosets, &tau_alt);
                op_alt.compute();

                const auto& A_std = std_op.get_H_matrices();
                const auto& A_alt = op_alt.get_H_matrices();
                if (A_std.size() != A_alt.size()) {
                    Fail("O12(c): get_H_matrices() size mismatch at " + tag);
                    continue;
                }
                bool same = true;
                for (uint32_t q = 0; q < A_std.size() && same; ++q) {
                    if (!(A_std[q] == A_alt[q])) {
                        ReportMatrixDiff(std::string("O12(c): A^(q) DIFFERS between TauVertexPolicy::") +
                                             arm.second + " and GeometricNearest",
                                         q, A_std[q], A_alt[q], tag);
                        same = false;
                    }
                }
                if (same) {
                    Ok(std::string("O12(c): ") + arm.second +
                       " vs GeometricNearest tau policies give equal A^(q), all q, at " + tag +
                       " (cochain-level Theta not compared, per contract).");
                }
            } catch (const std::exception& e) {
                Fail(std::string("O12(c): EXCEPTION (") + arm.second + ") at " + tag + ": " + e.what());
            }
        }
    }

    // -----------------------------------------------------------------------
    // O12(d), WEAK reachable form (see the header note for the full
    // documentation of why the strong form g1 = gamma1 g gamma2 is NOT
    // reachable through the public API). For two FIXED, nontrivial, distinct
    // elements gamma_A = L = [[1,0],[N,1]] and gamma_B = L^2 = [[1,0],[2N,1]]
    // of Gamma_0(N) (the only choices the custom-reps constructor can accept
    // as a UNIFORM left-translate of every s_i, since s_1 = I forces
    // gamma in Gamma_0(N)), replace EVERY s_i by gamma * s_i and assert
    // A^{(q)} is unchanged, for each of gamma_A, gamma_B independently.
    // -----------------------------------------------------------------------
    void CheckD_GWeakForm(CohomologyFixture& cf, const hecke::HeckeCosetTables& std_cosets,
                          const hecke::HeckeOperatorOnK& std_op, uint64_t p_hecke, uint32_t ell,
                          uint32_t n_sym, const std::string& tag) {
        const int64_t N = static_cast<int64_t>(std_cosets.N());
        const Mat22 gamma_A{1, 0, N, 1};       // L = [[1,0],[N,1]], nontrivial, in Gamma_0(N)
        const Mat22 gamma_B = gamma_A * gamma_A; // L^2 = [[1,0],[2N,1]], distinct from gamma_A

        const std::vector<std::pair<std::string, Mat22>> cases = {
            {"gamma=L", gamma_A}, {"gamma=L^2", gamma_B}
        };
        for (const auto& [label, gamma] : cases) {
            try {
                const uint32_t m = std_cosets.num_cosets();
                std::vector<Mat22> custom_s(m);
                for (uint32_t i = 1; i <= m; ++i) custom_s[i - 1] = gamma * std_cosets.s(i);

                const hecke::HeckeCosetTables custom_cosets(p_hecke, /*r=*/1, ell, n_sym, custom_s);
                const hecke::HeckeTauTable custom_tau(K_, model_, custom_cosets, ell, n_sym);
                hecke::HeckeOperatorOnK custom_op(&cf.coh, &custom_cosets, &custom_tau);
                custom_op.compute();

                const auto& A_std = std_op.get_H_matrices();
                const auto& A_custom = custom_op.get_H_matrices();
                if (A_std.size() != A_custom.size()) {
                    Fail("O12(d) [weak form, " + label + "]: get_H_matrices() size mismatch at " +
                         tag);
                    continue;
                }
                bool ok = true;
                for (uint32_t q = 0; q < A_std.size(); ++q) {
                    if (!(A_std[q] == A_custom[q])) {
                        ReportMatrixDiff("O12(d) [weak form, uniform left-translate " + label +
                                         " of every s_i, gamma in Gamma_0(N)]: A^(q) DIFFERS",
                                         q, A_std[q], A_custom[q], tag);
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    Ok("O12(d) [weak, documented-reachable form, " + label + "]: uniform "
                       "left-translate of every s_i by a single fixed gamma in Gamma_0(N) "
                       "gives equal A^(q), all q, at " + tag);
                }
            } catch (const std::exception& e) {
                Fail("O12(d) [weak form, " + label + "]: EXCEPTION at " + tag + ": " + e.what());
            }
        }
    }
};

} // namespace

uint32_t HeckeIndependenceValidation() {
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        IndependenceContext ctx(K, model);

        // Settings grid, binding (plan stage S8V): (p, ell, n) in
        // {(2,13,10), (3,13,10), (2,691,10), (5,101,2)}.
        const std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> grid = {
            {2, 13, 10}, {3, 13, 10}, {2, 691, 10}, {5, 101, 2}
        };
        for (const auto& [p_hecke, ell, n_sym] : grid) {
            ctx.RunSetting(p_hecke, ell, n_sym);
        }

        std::cout << "HeckeIndependenceValidation: " << ctx.failures()
                  << " failing check(s) across the full O12 battery." << std::endl;
        return ctx.failures();
    } catch (const std::exception& e) {
        std::cout << "HeckeIndependenceValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
