// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 1, WP-2 -- see HeckeIdentityValidation.h.

#include "HeckeIdentityValidation.h"

#include "HeckeDicksonMultiplication.h"
#include "HeckeIdentityCertificate.h"

#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

uint32_t g_failing = 0;

void Fail(const std::string& msg) {
    std::cout << "HeckeIdentityValidation: FAIL -- " << msg << std::endl;
    ++g_failing;
}

// Builds the (ell, n_sym) stack and returns the computed
// EquivariantCohomologyOnK together with the objects it depends on -- all
// owned by the caller for the lifetime of any HeckeOperatorFamily built on
// it (same pattern as HeckeOperatorFamilySmokeValidation.cpp).
struct Stack {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;

    Stack(uint32_t n_sym, uint32_t ell)
        : K(CreateSL2ZFundamentalDomainTriangulation()),
          model(K),
          lssc(K, BuildRelations(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc) {
        coh.compute_cohomology();
    }

    static std::vector<LocalSystemSimplicialComplex::SimplexRelation>
    BuildRelations(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
            rels.push_back(datum.rel);
        }
        return rels;
    }
};

// Builds one certificate at (n, ell) using two good primes p=2, p'=3 (both
// != ell; the two grid points with ell in {2,3} below use p'=5 / p=3
// instead so that p, p', ell stay pairwise distinct), populates it from the
// real pipeline, and returns it (and the dimensions, for the caller's own
// cross-check) by value.
hecke_ext::IdentityCertificate BuildCertificate(uint32_t n_sym, uint32_t ell,
                                               uint64_t p, uint64_t p_prime) {
    Stack stack(n_sym, ell);
    const auto& dims = stack.coh.get_H_eq_dimensions(); // [dim H^0, dim H^1, dim H^2]

    hecke_ext::IdentityCertificate cert;
    cert.n = n_sym;
    cert.ell = ell;
    cert.dim_H0 = dims.size() > 0 ? dims[0] : 0;
    cert.dim_H1 = dims.size() > 1 ? dims[1] : 0;
    cert.dim_H2 = dims.size() > 2 ? dims[2] : 0;
    cert.p = p;
    cert.p_prime = p_prime;

    if (cert.dim_H1 > 0) {
        hecke::HeckeOperatorFamily family(&stack.coh, stack.K, stack.model, ell, n_sym);
        const std::vector<Matrix>& Ap = family.T_p(p);
        const std::vector<Matrix>& Apprime = family.T_p(p_prime);
        cert.A_p = Ap.size() > 1 ? Ap[1] : MatrixT<uint32_t>();
        cert.A_p_prime = Apprime.size() > 1 ? Apprime[1] : MatrixT<uint32_t>();
        if (ell >= 5) {
            // The INDEPENDENT double-coset construction -- see
            // HeckeIdentityCertificate.h: never the recursion's own output.
            const std::vector<Matrix> Ap2 = family.T_p_pow_direct(p, 2);
            cert.A_p2 = Ap2.size() > 1 ? Ap2[1] : MatrixT<uint32_t>();
        }
    }
    return cert;
}

void CheckStatus(const char* label, hecke_ext::IdentityStatus got,
                 hecke_ext::IdentityStatus want, const char* item) {
    if (got != want) {
        Fail(std::string(label) + ": " + item + " status mismatch");
    }
}

using IS = hecke_ext::IdentityStatus;

// Arm 1: real pipeline points, every applicable item must pass.
void RunRealPointsArm() {
    struct Case {
        uint32_t n, ell;
        uint64_t p, p_prime;
        bool es_d2_applicable; // n>0 even, ell>=5
        bool p2_applicable;    // ell>=5
    };
    const std::vector<Case> cases = {
        {2, 5, 2, 3, true, true},    // classical, dim H^0 = 0
        {4, 5, 2, 3, true, true},    // classical
        {6, 5, 2, 3, true, true},    // Dickson-exceptional, dim H^0 != 0
        {12, 5, 2, 3, true, true},   // Dickson-exceptional
        {8, 7, 2, 3, true, true},    // Dickson-exceptional, different ell
        {10, 5, 2, 3, true, true},   // classical, dim H^1 = 3
        {4, 13, 2, 3, true, true},   // classical, larger ell
        {0, 5, 2, 3, false, false},  // n = 0: not applicable
        {1, 5, 2, 3, false, false},  // odd n: not applicable
        {2, 3, 3, 5, false, false},  // ell < 5: es/d2/p2 not applicable
    };

    for (const Case& c : cases) {
        hecke_ext::IdentityCertificate cert =
            BuildCertificate(c.n, c.ell, c.p, c.p_prime);
        const uint32_t code = cert.verify();
        const std::string label =
            "n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell);
        if (code != 0) {
            Fail(label + ": verify() = " + std::to_string(code) + " (" +
                cert.to_string_offset(0) + ")");
        }
        CheckStatus(label.c_str(), cert.eichler_shimura.status,
                   c.es_d2_applicable ? IS::kPassed : IS::kNotApplicable,
                   "eichler_shimura");
        CheckStatus(label.c_str(), cert.d2_vanishes.status,
                   c.es_d2_applicable ? IS::kPassed : IS::kNotApplicable,
                   "d2_vanishes");
        if (cert.dim_H1 > 0) {
            CheckStatus(label.c_str(), cert.commutation.status, IS::kPassed,
                       "commutation");
            CheckStatus(label.c_str(), cert.p2_recursion.status,
                       c.p2_applicable ? IS::kPassed : IS::kNotApplicable,
                       "p2_recursion");
        }
    }
}

// Arm 2: tampering. Built on (n=10, ell=5) (dim H^1 = 3, so plenty of
// entries to corrupt) and (n=6, ell=5) (Dickson-exceptional, dim H^0 = 1).
void RunTamperArm() {
    // --- tamper dim_H1: eichler_shimura must fail first (code 1). ---------
    {
        hecke_ext::IdentityCertificate cert = BuildCertificate(10, 5, 2, 3);
        cert.dim_H1 += 1;
        const uint32_t code = cert.verify();
        if (code != 1) Fail("tamper dim_H1: expected code 1, got " + std::to_string(code));
        if (cert.eichler_shimura.status != IS::kFailed)
            Fail("tamper dim_H1: eichler_shimura should be kFailed");
    }

    // --- tamper A_p_prime: commutation must fail (code 2), eichler_shimura
    //     and d2_vanishes (untouched dims) must still pass. -----------------
    {
        hecke_ext::IdentityCertificate cert = BuildCertificate(10, 5, 2, 3);
        if (cert.A_p_prime.rows == 0) {
            Fail("tamper A_p_prime: certificate has no operator matrix to tamper");
        } else {
            cert.A_p_prime.data[0][cert.A_p_prime.cols > 1 ? 1 : 0] =
                (cert.A_p_prime.data[0][cert.A_p_prime.cols > 1 ? 1 : 0] + 1) % cert.ell;
            const uint32_t code = cert.verify();
            if (code != 2) Fail("tamper A_p_prime: expected code 2, got " + std::to_string(code));
            if (cert.eichler_shimura.status != IS::kPassed)
                Fail("tamper A_p_prime: eichler_shimura should be unaffected");
            if (cert.d2_vanishes.status != IS::kPassed)
                Fail("tamper A_p_prime: d2_vanishes should be unaffected");
            if (cert.commutation.status != IS::kFailed)
                Fail("tamper A_p_prime: commutation should be kFailed");
        }
    }

    // --- tamper A_p2: p2_recursion must fail (code 4), the other three
    //     items (untouched) must still pass. -------------------------------
    {
        hecke_ext::IdentityCertificate cert = BuildCertificate(10, 5, 2, 3);
        if (cert.A_p2.rows == 0) {
            Fail("tamper A_p2: certificate has no T_{p^2} matrix to tamper");
        } else {
            cert.A_p2.data[0][0] = (cert.A_p2.data[0][0] + 1) % cert.ell;
            const uint32_t code = cert.verify();
            if (code != 4) Fail("tamper A_p2: expected code 4, got " + std::to_string(code));
            if (cert.eichler_shimura.status != IS::kPassed)
                Fail("tamper A_p2: eichler_shimura should be unaffected");
            if (cert.commutation.status != IS::kPassed)
                Fail("tamper A_p2: commutation should be unaffected");
            if (cert.d2_vanishes.status != IS::kPassed)
                Fail("tamper A_p2: d2_vanishes should be unaffected");
            if (cert.p2_recursion.status != IS::kFailed)
                Fail("tamper A_p2: p2_recursion should be kFailed");
        }
    }

    // --- tamper dim_H2: d2_vanishes must fail (code 3), on a point where
    //     eichler_shimura and commutation are untouched. --------------------
    {
        hecke_ext::IdentityCertificate cert = BuildCertificate(6, 5, 2, 3);
        cert.dim_H2 += 1;
        const uint32_t code = cert.verify();
        if (code != 3) Fail("tamper dim_H2: expected code 3, got " + std::to_string(code));
        if (cert.eichler_shimura.status != IS::kPassed)
            Fail("tamper dim_H2: eichler_shimura should be unaffected");
        if (cert.d2_vanishes.status != IS::kFailed)
            Fail("tamper dim_H2: d2_vanishes should be kFailed");
    }
}

// ===========================================================================
// Arm 3 (slot 5, WP-8) -- IDENTITY CLASS 5, the Dickson multiplication L*.
// ===========================================================================
//
// WHY THIS ARM'S RECORDED LINES GO TO `std::cerr`.  This validation's STDOUT
// is a FROZEN ARTEFACT: Console/scripts/cli_frozen/
// validation_HeckeIdentityValidation.stdout is the CLI-preservation gate's
// "one validation name" specimen (console C1a, slot-3 WP-7), compared
// BYTE-FOR-BYTE by ConsoleCliPreservationValidation, which runs the child
// with `2>/dev/null`.  Adding a line to stdout here would be a re-baseline of
// a frozen capture that slot-5 WP-10 explicitly promised not to move, for a
// diagnostic that is not what the gate is about.  The recorded lines
// therefore go to stderr, where a terminal and a log still show them; a
// FAILURE still goes to stdout through Fail(), which is correct -- a failing
// suite must NOT be byte-identical to the frozen success capture.  See
// docs/implementation/slot5/WP-8_report.md Sec. 5, finding 5.
//
// Two halves.  (i) The UNIT's own certificate
// (HeckeExt/HeckeDicksonMultiplication): the (n+1) x (n-ell) matrix, its
// entries re-derived by an independent convolution, its injectivity, and the
// det-twisted intertwining rho_n(A) L = det(A) L rho_m(A) checked entrywise on
// 24 seeded A per (n, ell) -- singular A included, because the identity is
// proved in characteristic ell with no determinant hypothesis.  (ii) The
// COHOMOLOGY-level identity, through IdentityCertificate's new item 5:
//
//     A^{(1)}_n(T_p) . L_*  ==  p . L_* . A^{(1)}_m(T_p),
//
// with L_* the map induced by L. as a ONE-ENTRY FiberBlockTable ([X] Sec. 2.2
// Stage 5: "the DicksonMultiplication map as a one-entry FiberBlockTable"),
// which is legitimate precisely because L. IS a morphism of local systems on
// K: the identifications of the model are elements of Gamma = SL_2(Z), of
// determinant 1, where the twist is trivial.  The twist is visible only on the
// HECKE side, where the coset data carries determinant p.
//
// A consequence asserted with it: L_* H^1(M_m) is a T_S-SUBMODULE of X_4 --
// [E, D6]'s `L_image` canonical position -- because the identity makes the
// image A_n(T_p)-stable for every p.
void RunDicksonArm() {
    // (n, ell) with n >= ell + 1, n and m = n - ell - 1 both even (ell odd),
    // and m in a range where H^1(M_m) is non-trivial.
    struct Case { uint32_t n, ell; uint64_t p; };
    const std::vector<Case> cases = {
        {12, 5, 2},    // m = 6:  the (6,5) Dickson-locus cell as the SOURCE
        {22, 5, 2},    // m = 16
        {22, 5, 3},    // the same cell at another prime
        {34, 5, 2},    // m = 28
        {26, 7, 2},    // m = 18
        {20, 7, 3},    // m = 12
        {24, 11, 2},   // m = 12
        {16, 13, 2},   // m = 2
    };

    for (const Case& c : cases) {
        const std::string label =
            "n=" + std::to_string(c.n) + " ell=" + std::to_string(c.ell) +
            " p=" + std::to_string(c.p);
        const uint32_t m = hecke_ext::DicksonSourceDegree(c.n, c.ell);

        // --- (i) the unit's certificate. ----------------------------------
        const hecke_ext::DicksonMultiplicationCertificate dc =
            hecke_ext::CertifyDicksonMultiplication(c.n, c.ell);
        const uint32_t dcode = dc.verify();
        if (dcode != 0) {
            Fail(label + ": DicksonMultiplicationCertificate::verify() = " +
                 std::to_string(dcode) + " (" +
                 hecke_ext::DicksonMultiplicationCertificate::verify_message(dcode) + ")");
        }
        if (dc.rows != c.n + 1 || dc.cols != c.n - c.ell) {
            Fail(label + ": the L. matrix has the wrong shape");
        }
        // Entries in {0, +-1}, exactly two non-zero per column.
        {
            const uint32_t one = 1 % c.ell;
            const uint32_t minus_one = (c.ell - one) % c.ell;
            bool shape_ok = true;
            for (uint32_t j = 0; j < dc.cols && shape_ok; ++j) {
                uint32_t nz = 0;
                for (uint32_t r = 0; r < dc.rows; ++r) {
                    const uint32_t e = dc.L.get_entry(r, j);
                    if (e == 0) continue;
                    ++nz;
                    if (e != one && e != minus_one) shape_ok = false;
                }
                if (nz != 2) shape_ok = false;
            }
            if (!shape_ok) Fail(label + ": L. is not +-1 with two non-zeros per column");
        }

        // --- (ii) the cohomology-level identity. ---------------------------
        Stack target(c.n, c.ell);
        Stack source(m, c.ell);
        const auto& dims_n = target.coh.get_H_eq_dimensions();
        const auto& dims_m = source.coh.get_H_eq_dimensions();
        const uint32_t dim1_n = dims_n.size() > 1 ? dims_n[1] : 0;
        const uint32_t dim1_m = dims_m.size() > 1 ? dims_m[1] : 0;
        if (dim1_n == 0 || dim1_m == 0) {
            std::cerr << "  DICKSON " << label << " m=" << m
                      << " dim H^1(M_n)=" << dim1_n << " dim H^1(M_m)=" << dim1_m
                      << " -- vacuous, skipped" << std::endl;
            continue;
        }

        // The one-entry FiberBlockTable: every simplex of K draws from ITSELF
        // in the source local system, through the constant fibre map L.
        // (Naming, EquivariantCochainComplexMorphismOnK.h: "domain" is the
        // TARGET of the induced H*-map.)
        const FlatMatrix Lmat = hecke_ext::DicksonMultiplicationMatrix(c.n, c.ell);
        const SimplicialComplex& K = target.K;
        FiberBlockTable table;
        table.resize(K.get_dimension() + 1);
        for (uint32_t q = 0; q < table.size(); ++q) {
            const uint32_t N_q = K.get_num_q_simplices(q);
            table[q].resize(N_q);
            for (uint32_t s = 0; s < N_q; ++s) {
                PerSimplexBlock blk;
                blk.s2_idx_in_K2 = s;
                blk.F = Lmat;
                table[q][s].push_back(blk);
            }
        }
        EquivariantCochainComplexMorphismOnK Lmap(&target.coh, &source.coh, table,
                                                  "DicksonMultiplication");
        Lmap.compute_induced_maps_on_cohomology();
        const std::vector<FlatMatrix>& induced = Lmap.get_induced_H_eq_maps();
        if (induced.size() < 2) { Fail(label + ": no induced degree-1 map"); continue; }
        const MatrixT<uint32_t> L_star = induced[1].to_matrix();
        if (L_star.rows != dim1_n || L_star.cols != dim1_m) {
            Fail(label + ": L_* has the wrong shape");
            continue;
        }

        hecke::HeckeOperatorFamily fam_n(&target.coh, target.K, target.model, c.ell, c.n);
        hecke::HeckeOperatorFamily fam_m(&source.coh, source.K, source.model, c.ell, m);
        const std::vector<Matrix>& An = fam_n.T_p(c.p);
        const std::vector<Matrix>& Am = fam_m.T_p(c.p);
        if (An.size() < 2 || Am.size() < 2) { Fail(label + ": missing A^{(1)}(T_p)"); continue; }

        hecke_ext::IdentityCertificate cert;
        cert.n = c.n;
        cert.ell = c.ell;
        cert.dim_H0 = dims_n.size() > 0 ? dims_n[0] : 0;
        cert.dim_H1 = dim1_n;
        cert.dim_H2 = dims_n.size() > 2 ? dims_n[2] : 0;
        cert.p = c.p;
        cert.p_prime = (c.p == 2 ? 3 : 2);
        cert.A_p = An[1];
        {
            const std::vector<Matrix>& Ap2 = fam_n.T_p(cert.p_prime);
            cert.A_p_prime = Ap2.size() > 1 ? Ap2[1] : MatrixT<uint32_t>();
        }
        cert.n_source = m;
        cert.A_p_source = Am[1];
        cert.L_star = L_star;

        const uint32_t code = cert.verify();
        if (code != 0) {
            Fail(label + ": verify() = " + std::to_string(code) + "\n" +
                 cert.to_string_offset(2));
        }
        if (cert.dickson_multiplication.status != IS::kPassed) {
            Fail(label + ": dickson_multiplication is not kPassed (" +
                 cert.dickson_multiplication.detail + ")");
        }

        // The twist is p^1 and NOT p^0: the untwisted identity must FAIL
        // wherever L_* is non-zero and p is not 1 mod ell -- otherwise the
        // check above would be blind to the exponent.
        {
            const uint32_t p_mod = static_cast<uint32_t>(c.p % c.ell);
            const MatrixT<uint32_t> lhs = cert.A_p * cert.L_star;
            const MatrixT<uint32_t> untwisted = cert.L_star * cert.A_p_source;
            if (p_mod != 1 && !untwisted.is_zero() && lhs == untwisted) {
                Fail(label + ": the UNTWISTED identity also holds -- the p-twist is not pinned");
            }
        }

        // L_* H^1(M_m) is a T_S-submodule of X_4: the image is stable under
        // A_n(T_p) for every p in {2, 3, 5, 7} ([E, D6]'s `L_image` slot).
        {
            const uint32_t r_img = MatrixT<uint32_t>(cert.L_star).rank();
            bool stable = true;
            for (uint64_t q : {2ull, 3ull, 5ull, 7ull}) {
                if (q == c.ell) continue;
                const std::vector<Matrix>& A = fam_n.T_p(q);
                if (A.size() < 2) continue;
                MatrixT<uint32_t> both(cert.L_star.rows, 2 * cert.L_star.cols, c.ell);
                const MatrixT<uint32_t> moved = A[1] * cert.L_star;
                for (uint32_t i = 0; i < cert.L_star.rows; ++i) {
                    for (uint32_t j = 0; j < cert.L_star.cols; ++j) {
                        both.data[i][j] = cert.L_star.data[i][j];
                        both.data[i][cert.L_star.cols + j] = moved.data[i][j];
                    }
                }
                if (both.rank() != r_img) stable = false;
            }
            if (!stable) Fail(label + ": L_* H^1(M_m) is not a T_S-submodule of X_4");
            std::cerr << "  DICKSON " << label << " m=" << m
                      << " L=" << dc.rows << "x" << dc.cols
                      << " samples=" << dc.num_samples()
                      << " dim H^1(M_n)=" << dim1_n << " dim H^1(M_m)=" << dim1_m
                      << " rank L_*=" << r_img
                      << " twist=p^1 submodule=" << (stable ? "yes" : "NO") << std::endl;
        }
    }

    // --- the inapplicable edges and a TAMPER. ------------------------------
    {
        // n < ell + 1: no Dickson class, and the unit says so.
        const hecke_ext::DicksonMultiplicationCertificate dc =
            hecke_ext::CertifyDicksonMultiplication(4, 5);
        if (dc.applicable) Fail("(4,5): L. must be inapplicable (n < ell + 1)");
        if (dc.verify() != 0) Fail("(4,5): the inapplicable certificate must verify");
        if (hecke_ext::DicksonMultiplicationMatrix(4, 5).get_rows() != 0) {
            Fail("(4,5): the inapplicable matrix must be empty");
        }
    }
    {
        // A tampered entry must fail clause 3, and a tampered SIGN clause 3
        // as well (it is still checked against the convolution).
        hecke_ext::DicksonMultiplicationCertificate dc =
            hecke_ext::CertifyDicksonMultiplication(12, 5);
        if (dc.verify() != 0) Fail("(12,5): the untampered certificate must verify");
        dc.L.set_entry(0, 0, 1 % dc.ell);
        if (dc.verify() != 3) Fail("(12,5): a planted entry must fail clause 3");
    }
    {
        // A tampered SAMPLE list must fail clause 5 (an entry >= ell).
        hecke_ext::DicksonMultiplicationCertificate dc =
            hecke_ext::CertifyDicksonMultiplication(12, 5);
        dc.samples[0] = dc.ell;
        if (dc.verify() != 5) Fail("(12,5): an out-of-range sample must fail clause 5");
    }
    {
        // A tampered L_* must fail IdentityCertificate clause 5 and leave the
        // other four items untouched.  The identity A_n X = p X A_m is LINEAR
        // in X, so a single-entry perturbation breaks it only when it leaves
        // that solution space; the arm therefore SEARCHES the perturbations
        // and asserts (i) that at least one breaks the identity -- i.e. the
        // clause has real content and is not satisfied by every matrix of that
        // shape -- and (ii) that a breaking one fails EXACTLY clause 5.
        // (22, 5) is used rather than (12, 5) because its L_* is 5 x 3 of full
        // column rank, so the solution space is small.
        Stack target(22, 5), source(16, 5);
        const auto& dims_n = target.coh.get_H_eq_dimensions();
        hecke::HeckeOperatorFamily fam_n(&target.coh, target.K, target.model, 5, 22);
        hecke::HeckeOperatorFamily fam_m(&source.coh, source.K, source.model, 5, 16);
        hecke_ext::IdentityCertificate cert;
        cert.n = 22; cert.ell = 5;
        cert.dim_H0 = dims_n[0]; cert.dim_H1 = dims_n[1]; cert.dim_H2 = dims_n[2];
        cert.p = 2; cert.p_prime = 3;
        cert.A_p = fam_n.T_p(2)[1];
        cert.A_p_prime = fam_n.T_p(3)[1];
        cert.A_p2 = fam_n.T_p_pow_direct(2, 2)[1];
        cert.n_source = 16;
        cert.A_p_source = fam_m.T_p(2)[1];
        const FlatMatrix Lmat = hecke_ext::DicksonMultiplicationMatrix(22, 5);
        FiberBlockTable table;
        table.resize(target.K.get_dimension() + 1);
        for (uint32_t q = 0; q < table.size(); ++q) {
            const uint32_t N_q = target.K.get_num_q_simplices(q);
            table[q].resize(N_q);
            for (uint32_t s = 0; s < N_q; ++s) {
                PerSimplexBlock blk; blk.s2_idx_in_K2 = s; blk.F = Lmat;
                table[q][s].push_back(blk);
            }
        }
        EquivariantCochainComplexMorphismOnK Lmap(&target.coh, &source.coh, table, "L");
        Lmap.compute_induced_maps_on_cohomology();
        const MatrixT<uint32_t> L_clean = Lmap.get_induced_H_eq_maps()[1].to_matrix();
        cert.L_star = L_clean;
        if (cert.verify() != 0) Fail("(22,5): the untampered identity certificate must verify");
        if (cert.dickson_multiplication.status != IS::kPassed) {
            Fail("(22,5): dickson_multiplication must pass before the tamper");
        }
        uint32_t breaking = 0;
        bool checked_one = false;
        for (uint32_t i = 0; i < L_clean.rows; ++i) {
            for (uint32_t j = 0; j < L_clean.cols; ++j) {
                cert.L_star = L_clean;
                cert.L_star.data[i][j] = (cert.L_star.data[i][j] + 1) % 5;
                const uint32_t code = cert.verify();
                if (code == 0) continue;
                ++breaking;
                if (checked_one) continue;
                checked_one = true;
                if (code != 5) {
                    Fail("tamper L_*: expected code 5, got " + std::to_string(code));
                }
                if (cert.eichler_shimura.status != IS::kPassed ||
                    cert.commutation.status != IS::kPassed ||
                    cert.d2_vanishes.status != IS::kPassed ||
                    cert.p2_recursion.status != IS::kPassed) {
                    Fail("tamper L_*: items 1-4 must be unaffected");
                }
                if (cert.dickson_multiplication.status != IS::kFailed) {
                    Fail("tamper L_*: dickson_multiplication should be kFailed");
                }
            }
        }
        if (breaking == 0) {
            Fail("tamper L_*: NO single-entry perturbation breaks the identity -- "
                 "clause 5 would be vacuous at (22, 5)");
        }
        std::cerr << "  DICKSON tamper (22,5): " << breaking << " of "
                  << (L_clean.rows * L_clean.cols)
                  << " single-entry perturbations break A_n L_* = p L_* A_m" << std::endl;
        // A shape mismatch makes the item inapplicable, never `passed`.
        cert.L_star = MatrixT<uint32_t>();
        if (cert.verify() != 0) Fail("tamper L_*: an EMPTY L_* must be inapplicable, not failing");
        if (cert.dickson_multiplication.status != IS::kNotApplicable) {
            Fail("tamper L_*: an empty L_* must leave the item kNotApplicable");
        }
    }
}

} // namespace

uint32_t HeckeIdentityValidation() {
    g_failing = 0;
    std::cout << "================= HeckeIdentityValidation =================" << std::endl;
    RunRealPointsArm();
    RunTamperArm();
    RunDicksonArm();
    std::cout << "HeckeIdentityValidation: " << g_failing << " failing case(s)." << std::endl;
    return g_failing;
}
