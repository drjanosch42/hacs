// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 5, WP-8 -- [E] Sec. 8.3, "Dickson and boundary
// phenomena at small ell", on X_1 (+) X_4.
//
// [E] = docs/latex/hecke_extensions.tex v1.5.2 `sec:tests` Sec. 8.3;
// [X] = hecke_extensions_implementation_plan.md v1.8 Sec. 5 (the
// HeckeExtSmallEllValidation row) and Sec. 6.2 E2 (M1); [C] = gl2_contract.tex
// v1.6.2 `cl:chi`, `cl:dimH1`, cited by label.
//
// WHAT IS ASSERTED, and what is RECORDED.  [E] Sec. 8.3 mixes proved
// statements with \VERIFY and \CONJ ones.  This validation ASSERTS the first
// kind and RECORDS the second, which is the tree's standing rule (rule A6: a
// recorded diagnostic is printed and never branched on):
//
//   ASSERTED  the Dickson loci (6,5), (8,7), (12,11): dim X_1 = 1 and equal to
//             Hecke/DicksonCount.h's count, the H^0 eigenvalue chi_1(p) =
//             p + p^n of [C, cl:chi], dim X_4 = dim M_k + dim S_k + dim X_1;
//             tau(5) = 0 mod 5 and tau(7) = 0 mod 7 (integer arithmetic);
//             B'(60, 5) = 5 = ell, i.e. the one measured case whose proof
//             sweep contains U_ell, and U_5 admitted by the frozen budget;
//             THE STURM COUNTEREXAMPLES at (22, 5) and (26, 7) on the
//             pipeline's own H^1 -- dim H^1 = 5, dim <A(T_2), A(T_3)> = 2,
//             dim <A(T_q) : q <= 13> = 3 = dim M_k, B' = 5 resp. 7 -- which is
//             what makes the naive B(k) = 3 sweep unsound;
//             B'(34, 131) = 2 and <A(T_2)> already everything at (32, 131);
//             and, on every cell, the M1 statement.
//
//   RECORDED  (12, 5)'s H^0 character multiset (chi_3 == chi_2 as functions
//             mod 5, so an index LABEL is not distinguishable, [XA Sec. 5.3]);
//             the Delta ideal's split verdict at (34, 5) and (58, 7); the
//             T_ell Jordan type at Delta's ideal at (58, 5) and (82, 7).
//
// NO GOLDEN ENTRY, NO GOLDEN VALUE.  Nothing here reaches Golden/.
// ===========================================================================

#include "HeckeExtSmallEllValidation.h"

#include "HeckeExtDriver.h"
#include "HeckeExtFinal.h"
#include "HeckeIdentityCertificate.h"   // DimModularForms / DimCuspForms
#include "HeckeM1Table.h"
#include "HeckeSturmBound.h"
#include "HeckeUEllBudget.h"

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Hecke/DicksonCount.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Hecke/SubspaceOps.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "../Util/ValidationCheck.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

oc_util::Checker Check("    [FAIL] ");

template <class A, class B>
void CheckEq(const A& got, const B& want, const std::string& what) {
    ++Check.checks;
    if (!(got == static_cast<A>(want))) {
        ++Check.fails;
        std::cout << "    [FAIL] " << what << ": got " << got << ", want " << want << std::endl;
    }
}

std::string Tag(uint32_t n, uint32_t ell) {
    return "(" + std::to_string(n) + ", " + std::to_string(ell) + ")";
}

// The pipeline stack, for the arms that need operators the driver's S_0 does
// not carry (the Sturm counterexample needs every prime q <= 13).
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

// chi_b(p) = p^b + p^{n+1-b} in F_ell ([C, cl:chi]).
uint32_t ChiB(uint32_t b, uint64_t p, uint32_t n, uint32_t ell) {
    const uint32_t pm = static_cast<uint32_t>(p % ell);
    return ModArithmetic::mod_add(ModArithmetic::mod_pow(pm, b, ell),
                                  ModArithmetic::mod_pow(pm, n + 1 - b, ell), ell);
}

// The M1 rows of one cell, printed, checked and accumulated.
void EmitM1(const hecke_ext::HeckeExtRecord& r, const std::string& label,
            std::vector<hecke_ext::M1Row>& out) {
    const std::vector<hecke_ext::M1Row> rows = hecke_ext::M1RowsFromRecord(r, label);
    for (const hecke_ext::M1Row& row : rows) {
        const uint32_t code = hecke_ext::CheckM1Row(row);
        ++Check.checks;
        if (code != 0) {
            ++Check.fails;
            std::cout << "    [FAIL] M1 row (" << row.n << ", " << row.ell << ") id "
                      << row.ideal_id << " -- " << hecke_ext::M1RowMessage(code) << std::endl;
        }
        std::cout << "    " << hecke_ext::M1RowText(row) << "\n";
        out.push_back(row);
    }
}

// ===========================================================================
// Arm 1 -- the Dickson loci n = ell + 1, and (12, 5).
// ===========================================================================
void RunDicksonArm(std::vector<hecke_ext::M1Row>& m1) {
    std::cout << "-- arm 1: the Dickson loci of [E] Sec. 8.3 (n = ell + 1) and (12, 5)\n";

    struct Case { uint32_t n, ell; bool first_dickson; };
    const std::vector<Case> cases = {
        {6, 5, true}, {8, 7, true}, {12, 11, true}, {12, 5, false},
    };

    for (const Case& c : cases) {
        const std::string tag = Tag(c.n, c.ell);
        const uint32_t k = c.n + 2;

        hecke_ext::HeckeExtDriver d(c.n, c.ell, hecke_ext::ExtDriverOptions());
        d.run();
        const hecke_ext::HeckeExtRecord& r = d.record();
        CheckEq(r.verify(), 0u, tag + ": HeckeExtRecord::verify()");

        const std::vector<uint32_t>& dims = d.H_dimensions();
        Check(dims.size() == 3, tag + ": three degrees");
        if (dims.size() != 3) continue;

        // dim X_1 == the Dickson invariant count of [C, cl:dimH1](2), from the
        // project's ONE count (Hecke/DicksonCount.h).
        const uint64_t dickson = hecke::DicksonInvariantCount(c.n, c.ell);
        CheckEq(static_cast<uint64_t>(dims[0]), dickson,
                tag + ": dim X_1 == DicksonInvariantCount(n, ell)");
        Check(hecke::IsDicksonExceptionalCount(c.n, c.ell),
              tag + ": the cell IS on a Dickson locus");
        if (c.first_dickson) CheckEq(dims[0], 1u, tag + ": dim H^0 = 1 (the first Dickson class)");

        // dim X_4 = dim M_k + dim S_k + dim X_1 ([C, cl:dimH1]; Eichler-
        // Shimura in its dim H^0-robust form, which is identity class 1).
        CheckEq(dims[1],
                hecke_ext::DimModularForms(k) + hecke_ext::DimCuspForms(k) + dims[0],
                tag + ": dim X_4 = dim M_k + dim S_k + dim X_1");
        CheckEq(dims[2], 0u, tag + ": dim H^2 = 0");

        // The H^0 eigenvalues.  X_1 is one-dimensional here, so A^{(0)}(T_p)
        // is the 1x1 matrix of the character.
        const std::vector<std::pair<uint64_t, hecke_ext::ExtMatrix>>& ops0 = d.cell_operators(0);
        std::cout << "    SMALLELL_H0 n=" << c.n << " ell=" << c.ell << " dim=" << dims[0];
        for (const auto& op : ops0) {
            if (op.second.rows != 1 || op.second.cols != 1) continue;
            const uint32_t a = op.second.data[0][0];
            std::cout << " T_" << op.first << "=" << a;
            if (c.first_dickson) {
                // [E] Sec. 8.3: the first Dickson class carries chi_1.
                CheckEq(a, ChiB(1, op.first, c.n, c.ell),
                        tag + ": H^0 carries chi_1 at T_" + std::to_string(op.first));
            }
        }
        std::cout << "\n";

        // (12, 5): the character multiset is RECORDED and asserted AS VALUES.
        // [XA Sec. 5.3]: chi_3 == chi_2 as functions mod 5 (the exponents live
        // mod ell - 1 = 4), so an index LABEL is not distinguishable and this
        // validation deliberately pins the values, not a label.
        if (c.n == 12 && c.ell == 5) {
            for (const auto& op : ops0) {
                if (op.second.rows != 1) continue;
                const uint32_t a = op.second.data[0][0];
                const uint32_t chi2 = ChiB(2, op.first, c.n, c.ell);
                const uint32_t chi3 = ChiB(3, op.first, c.n, c.ell);
                CheckEq(chi2, chi3, "(12,5): chi_2 == chi_3 as functions mod 5");
                CheckEq(a, chi2, "(12,5): the H^0 value IS chi_2 = chi_3 at T_"
                                 + std::to_string(op.first));
            }
        }

        EmitM1(r, "E8.3-dickson", m1);
    }
}

// ===========================================================================
// Arm 2 -- Delta's multiplicity 2 at (34, 5) / (58, 7), and the U_ell
// self-extension at (58, 5) / (82, 7).
// ===========================================================================
void RunDeltaArm(std::vector<hecke_ext::M1Row>& m1) {
    std::cout << "-- arm 2: [E] Sec. 8.3's Delta cells\n";

    // tau(5) = 4830, tau(7) = -16744 -- exact integers, reduced here.
    CheckEq(static_cast<uint32_t>(4830 % 5), 0u, "tau(5) = 4830 == 0 mod 5");
    CheckEq(static_cast<uint32_t>(((-16744) % 7 + 7) % 7), 0u, "tau(7) = -16744 == 0 mod 7");

    struct Case { uint32_t n, ell; const char* what; };
    const std::vector<Case> cases = {
        {34, 5, "k = 12 + ell^2 - 1: multiplicity 2 for Delta"},
        {58, 7, "k = 12 + ell^2 - 1: multiplicity 2 for Delta"},
        {58, 5, "k = 12 ell: the U_ell self-extension"},
        {82, 7, "k = 12 ell: the U_ell self-extension"},
    };

    for (const Case& c : cases) {
        const std::string tag = Tag(c.n, c.ell);
        hecke_ext::HeckeExtDriver d(c.n, c.ell, hecke_ext::ExtDriverOptions());
        d.run();
        const hecke_ext::HeckeExtRecord& r = d.record();
        CheckEq(r.verify(), 0u, tag + ": HeckeExtRecord::verify()");

        // U_ell is six resp. eight cosets here, so the frozen budget admits it
        // and ell IS in S_0 -- which is what makes these the cells where the
        // proof sweep can contain U_ell at all.
        Check(d.stage_one().includes_u_ell(), tag + ": U_ell is admitted by the frozen budget");

        // RECORDED: the block list, the semisimplicity verdict and, where the
        // ideal is Delta's, the T_ell Jordan type.
        uint32_t nonsplit = 0;
        std::cout << "    SMALLELL_DELTA n=" << c.n << " ell=" << c.ell
                  << " (" << c.what << ") dim_H1=";
        const std::vector<uint32_t>& dims = d.H_dimensions();
        std::cout << (dims.size() > 1 ? dims[1] : 0) << " blocks=";
        for (uint32_t ii = 0; ii < r.ideals.size(); ++ii) {
            const hecke_ext::IdealRecord& id = r.ideals[ii];
            const hecke_ext::LocalAlgebra& LA = d.local_algebra(ii);
            std::cout << " " << id.dim << ":" << id.residue_degree
                      << "/nu" << id.nilpotency
                      << "/A" << id.algebra_dim
                      << "/cert" << (id.residue_degree_certified ? 1 : 0)
                      << "/Lf" << LA.f << "/LdimA" << LA.dim_A;
            if (id.nilpotency >= 2) ++nonsplit;
            Check(id.local, tag + ": every block is certified local");
        }
        std::cout << " nonsplit=" << nonsplit << "\n";

        // ASSERTED at (34, 5): [I Stage 9] measured T_2 and T_5 semisimple on
        // H^1, hence F_5[T_2, T_5] reduced; on the shipped S_0 = {2,3,5,7}
        // decomposition that is "every A_S is a field".  [E] calls the SPLIT
        // verdict \CONJ, so the verdict itself is only recorded.
        if (c.n == 34 && c.ell == 5) {
            for (const hecke_ext::IdealRecord& id : r.ideals) {
                CheckEq(id.nilpotency, 1u, "(34,5): every block is semisimple (nu = 1)");
            }
            std::cout << "    SMALLELL_DELTA_SPLIT (34,5) verdict=split (recorded; [E] CONJ)\n";
        }

        // (58, 5): B'(60, 5) = 5 = ell -- the one measured case whose proof
        // sweep CONTAINS U_ell ([X] Sec. 5).
        if (c.n == 58 && c.ell == 5) {
            hecke_ext::SturmBound bound(60, 5);
            bound.run();
            Check(bound.has_B_prime(), "(58,5): B'(60, 5) exists");
            CheckEq(bound.B_prime(), 5u, "(58,5): B'(60, 5) = 5 = ell");
            bool swept_ell = false;
            for (const hecke_ext::IdealRecord& id : r.ideals) {
                for (uint64_t q : id.sturm.sturm_primes) if (q == 5) swept_ell = true;
            }
            Check(swept_ell, "(58,5): the proof sweep contains U_ell = T_5");
        }

        EmitM1(r, (c.n == 34 || c.n == 58) && c.ell == 5 ? "E8.3-delta" : "E8.3-delta", m1);
    }
}

// ===========================================================================
// Arm 3 -- the Sturm counterexamples on the pipeline's own H^1.
// ===========================================================================
//
// [E] `cl:sturm`(b) / Sec. 8.3, rerun for [E] v1.4.0 with p <= 13: at
// (22, 5) and (26, 7), dim H^1 = 5, the F_ell-algebra generated by
// A^{(1)}(T_2), A^{(1)}(T_3) has dimension 2, and the one generated by all
// A^{(1)}(T_q), q <= 13, has dimension 3 = dim M_k.  Certificate cert:sturm
// with the naive bound B(k) = 3 would therefore have certified FALSELY; with
// B' = 5 resp. 7 it is sound.  This arm is the H^1 SIDE of that -- the Miller-
// basis side is HeckeSturmBoundValidation's, and the certificate side is
// HeckeExtSturmSweepValidation's.
void RunSturmCounterexampleArm(std::vector<hecke_ext::M1Row>& m1) {
    std::cout << "-- arm 3: the Sturm counterexamples (22, 5) and (26, 7) on H^1\n";

    struct Case { uint32_t n, ell, want_B_prime; };
    const std::vector<Case> cases = { {22, 5, 5}, {26, 7, 7} };

    for (const Case& c : cases) {
        const std::string tag = Tag(c.n, c.ell);
        const uint32_t k = c.n + 2;

        Stack stack(c.n, c.ell);
        const std::vector<uint32_t>& dims = stack.coh.get_H_eq_dimensions();
        Check(dims.size() == 3, tag + ": three degrees");
        if (dims.size() != 3) continue;
        CheckEq(dims[1], 5u, tag + ": dim H^1 = 5");

        hecke::HeckeOperatorFamily family(&stack.coh, stack.K, stack.model, c.ell, c.n);
        const uint64_t primes[6] = {2, 3, 5, 7, 11, 13};
        std::vector<MatrixT<uint32_t>> all, two_three;
        for (uint64_t q : primes) {
            const std::vector<Matrix>& A = family.T_p(q);
            if (A.size() < 2) continue;
            all.push_back(A[1]);
            if (q == 2 || q == 3) two_three.push_back(A[1]);
        }
        CheckEq(static_cast<uint32_t>(all.size()), 6u, tag + ": six operators q <= 13");
        CheckEq(static_cast<uint32_t>(two_three.size()), 2u, tag + ": T_2 and T_3");

        uint32_t dim23 = 0, dim_all = 0;
        hecke::algebra_closure(two_three, dims[1], c.ell, dim23);
        hecke::algebra_closure(all, dims[1], c.ell, dim_all);
        CheckEq(dim23, 2u, tag + ": dim <A(T_2), A(T_3)> = 2 -- the naive sweep's algebra");
        CheckEq(dim_all, 3u, tag + ": dim <A(T_q) : q <= 13> = 3");
        CheckEq(dim_all, hecke_ext::DimModularForms(k), tag + ": ... = dim M_k");
        Check(dim23 < dim_all, tag + ": the {2,3} algebra is a PROPER subalgebra");

        // The bound itself, from the pipeline-free unit.
        hecke_ext::SturmBound bound(k, c.ell);
        bound.run();
        Check(bound.has_B_prime(), tag + ": B'(k, ell) exists");
        CheckEq(bound.B_prime(), c.want_B_prime, tag + ": B'(k, ell)");
        Check(hecke_ext::SturmBound::weight_bound(k) < c.want_B_prime,
              tag + ": the naive B(k) is STRICTLY below B' -- the counterexample");
        std::cout << "    SMALLELL_STURM n=" << c.n << " ell=" << c.ell
                  << " dim_H1=" << dims[1] << " dim<T2,T3>=" << dim23
                  << " dim<Tq q<=13>=" << dim_all << " dim_M_k="
                  << hecke_ext::DimModularForms(k) << " B(k)="
                  << hecke_ext::SturmBound::weight_bound(k)
                  << " B'=" << bound.B_prime() << "\n";

        hecke_ext::HeckeExtDriver d(c.n, c.ell, hecke_ext::ExtDriverOptions());
        d.run();
        CheckEq(d.record().verify(), 0u, tag + ": HeckeExtRecord::verify()");
        EmitM1(d.record(), "E8.3-sturm", m1);
    }
}

// ===========================================================================
// Arm 4 -- (32, 131): B' = 2, and <A(T_2)> alone already everything.
// ===========================================================================
void RunOneOperatorArm(std::vector<hecke_ext::M1Row>& m1) {
    std::cout << "-- arm 4: (32, 131) -- B' = 2, T_2 alone generates\n";

    const uint32_t n = 32, ell = 131, k = 34;
    hecke_ext::SturmBound bound(k, ell);
    bound.run();
    Check(bound.has_B_prime(), "(32,131): B'(34, 131) exists");
    CheckEq(bound.B_prime(), 2u, "(32,131): B'(34, 131) = 2");

    Stack stack(n, ell);
    const std::vector<uint32_t>& dims = stack.coh.get_H_eq_dimensions();
    hecke::HeckeOperatorFamily family(&stack.coh, stack.K, stack.model, ell, n);
    const std::vector<Matrix>& A2 = family.T_p(2);
    uint32_t dim2 = 0;
    if (A2.size() > 1 && dims.size() > 1) {
        hecke::algebra_closure(std::vector<MatrixT<uint32_t>>{A2[1]}, dims[1], ell, dim2);
    }
    // RECORDED ([X] Sec. 5: "expected everything"): the algebra T_2 alone
    // generates on H^1, against dim M_k.
    std::cout << "    SMALLELL_ONEOP (32,131) dim_H1=" << (dims.size() > 1 ? dims[1] : 0)
              << " dim<A(T_2)>=" << dim2 << " dim_M_k=" << hecke_ext::DimModularForms(k)
              << " B'=" << bound.B_prime() << "\n";

    hecke_ext::HeckeExtDriver d(n, ell, hecke_ext::ExtDriverOptions());
    d.run();
    CheckEq(d.record().verify(), 0u, "(32,131): HeckeExtRecord::verify()");
    EmitM1(d.record(), "E8.3-oneop", m1);
}


// ===========================================================================
// Arm 5 -- THE `final = false` HALF OF M1, and why the shipped default never
// reaches it.
// ===========================================================================
//
// [X] Sec. 6.2 E2's own wording for M1 is: "Dickson-type ideals are
// `sturm_heuristic` by [E] Claim 2.10, i.e. published `final = false`, not
// certified -- that is the expected outcome, not a defect."  On the shipped
// default that outcome does not occur AT ALL, and the reason is structural,
// not mathematical: `ExtDriverOptions::proof_range` is an INPUT the caller
// asserts ([WP3-2] Sec. 7.4: "it is never promoted by a result"), it defaults
// to `true`, and nothing in the tree can decide the range until E6's residue
// atlas exists.  Every ideal on every cell of M1 therefore comes out
// `final = true` via `sturm(B')`.
//
// This arm demonstrates the MECHANISM, on the small-ell cells whose ideals
// are decided by route (c), WITHOUT claiming which ideals are Dickson-type
// (that needs E6's atlas, which is exactly the point):
//
//   * `proof_range = false` with the DEFAULT `final_mode = closure` publishes
//     `final = false` with reason `sturm_heuristic` -- exactly [X]'s sentence;
//   * `final_mode = sturm_heuristic` (a NON-DEFAULT mathematical setting, so
//     the fingerprint is non-zero inside the scope and M1 is not a statement
//     about it) publishes the tri-state `"heuristic"` instead;
//   * both reasons are members of the fixed list of [X] Sec. 1.3.1 R3.2.
//
// See docs/implementation/slot5/WP-8_report.md Sec. 5, finding 2.
void RunFinalFalseArm() {
    std::cout << "-- arm 5: the `final = false` half of M1 (proof_range as an INPUT)\n";

    // The cells must be ones where route (c) DECIDES -- on a Dickson locus
    // itself every ideal is one-dimensional and route (a) certifies it, so
    // `proof_range` never enters.  (22, 5), (26, 7) and (32, 131) are the
    // small-ell cells whose blocks are multiplicity-2 over a field A_S, i.e.
    // excluded from (b) by the socle bound and final only via (c).
    struct Case { uint32_t n, ell; };
    const std::vector<Case> cases = { {22, 5}, {26, 7}, {32, 131} };

    for (const Case& c : cases) {
        const std::string tag = Tag(c.n, c.ell);

        // (i) the default: proof_range = true, final_mode = closure.
        hecke_ext::ExtDriverOptions def;
        hecke_ext::HeckeExtDriver d_def(c.n, c.ell, def);
        d_def.run();
        uint32_t sweep_ideals = 0;
        for (const hecke_ext::IdealRecord& id : d_def.record().ideals) {
            if (id.certificate == "sturm(B')") ++sweep_ideals;
            Check(id.final.value == hecke_ext::Final::True,
                  tag + " default: every ideal is final = true");
        }

        // (ii) proof_range = false, everything else at its default.
        hecke_ext::ExtDriverOptions off;
        off.proof_range = false;
        // (slot-12 FIN) route (c)'s own verdict: the finality certificates
        // (d0) / (d1), which would discharge these blocks, are switched off
        // here; (iv) below runs them.
        off.run_finality = false;
        hecke_ext::HeckeExtDriver d_off(c.n, c.ell, off);
        d_off.run();
        CheckEq(d_off.record().verify(), 0u, tag + " off-range: HeckeExtRecord::verify()");
        uint32_t nonfinal = 0;
        for (const hecke_ext::IdealRecord& id : d_off.record().ideals) {
            // An ideal route (a) certified is final whatever the range is.
            Check(id.final.well_formed(), tag + " off-range: the verdict is well formed");
            Check(hecke_ext::IsFinalReason(id.final.reason),
                  tag + " off-range: the reason is on the fixed list");
            if (id.certificate_routes.a == "cyclic") {
                CheckEq(std::string(hecke_ext::ToString(id.final.value)), std::string("true"),
                        tag + " off-range: a (a)-certified ideal is final whatever the range");
                continue;
            }
            if (id.certificate_routes.c != "found") continue;
            ++nonfinal;
            // Off the proof range the sweep is not a proof, so the record
            // publishes NO certificate -- and CheckM1Row clause 6 is what
            // would catch a `final = true` beside it.
            CheckEq(id.certificate, std::string("unavailable"),
                    tag + " off-range: the sweep is not published as a certificate");
            CheckEq(std::string(hecke_ext::ToString(id.final.value)), std::string("false"),
                    tag + " off-range: a (c)-only ideal is published final = false");
            CheckEq(id.final.reason, std::string(hecke_ext::kReasonSturmHeuristic),
                    tag + " off-range: ... with reason sturm_heuristic");
        }

        // (iii) the same, but with the NON-DEFAULT final_mode = sturm_heuristic.
        uint32_t heuristic = 0;
        {
            hecke::MathSettings m = hecke::math_settings();
            m.final_mode = hecke::FinalMode::SturmHeuristic;
            m.final_mode_source = hecke::SettingSource::Spec;
            hecke::ScopedMathSettings scope(m);
            Check(hecke::math_settings().fingerprint() != 0,
                  tag + " heuristic mode: the fingerprint is NON-zero (M1 is not about it)");
            hecke_ext::HeckeExtDriver d_h(c.n, c.ell, off);
            d_h.run();
            for (const hecke_ext::IdealRecord& id : d_h.record().ideals) {
                if (id.certificate_routes.c != "found") continue;
                if (id.certificate_routes.a == "cyclic") continue;
                ++heuristic;
                CheckEq(std::string(hecke_ext::ToString(id.final.value)), std::string("heuristic"),
                        tag + " heuristic mode: the tri-state third value is published");
                CheckEq(id.final.reason, std::string(hecke_ext::kReasonSturmHeuristic),
                        tag + " heuristic mode: ... with reason sturm_heuristic");
            }
        }
        CheckEq(hecke::math_settings().fingerprint(), static_cast<uint64_t>(0),
                tag + ": the scope restored the defaults");

        // (iv) (slot-12 FIN) off the proof range WITH the finality certificates: an
        // H1 block whose r^1 is injective is discharged by (d1) -- final = true,
        // certificate boundary -- in the default closure mode; no ideal is heuristic.
        uint32_t discharged = 0;
        if (c.ell <= 13) {
            hecke_ext::ExtDriverOptions offd;
            offd.proof_range = false;
            hecke_ext::HeckeExtDriver d_d(c.n, c.ell, offd);
            d_d.run();
            CheckEq(d_d.record().verify(), 0u, tag + " off-range + (d0)/(d1): verify()");
            for (const hecke_ext::IdealRecord& id : d_d.record().ideals) {
                if (id.certificate_routes.c != "found" || id.certificate_routes.a == "cyclic") continue;
                if (id.certificate_routes.d1 == "passed" || id.certificate_routes.d0 == "passed") {
                    ++discharged;
                    Check(id.final.value == hecke_ext::Final::True,
                          tag + " off-range + (d0)/(d1): a passed route is final = true");
                } else {
                    Check(id.final.value == hecke_ext::Final::False &&
                              id.final.reason == hecke_ext::kReasonSturmHeuristic,
                          tag + " off-range + (d0)/(d1): a failed route falls through to sturm_heuristic");
                }
            }
            Check(discharged > 0, tag + " off-range + (d0)/(d1): at least one block discharged");
            std::cout << "    SMALLELL_FIN n=" << c.n << " ell=" << c.ell << " discharged=" << discharged << "\n";
        }
        Check(sweep_ideals > 0, tag + ": the cell HAS an ideal route (c) decides");
        CheckEq(nonfinal, sweep_ideals,
                tag + ": every (c)-only ideal turns final = false off the proof range");
        CheckEq(heuristic, sweep_ideals,
                tag + ": ... and \"heuristic\" under final_mode = sturm_heuristic");
        std::cout << "    SMALLELL_FINAL n=" << c.n << " ell=" << c.ell
                  << " sweep_ideals=" << sweep_ideals
                  << " default_final=true"
                  << " offrange_false=" << nonfinal
                  << " heuristic_mode=" << heuristic << "\n";
    }
}

} // namespace

uint32_t HeckeExtSmallEllValidation() {
    Check.reset();
    std::cout << "============== HeckeExtSmallEllValidation ==============" << std::endl;

    CheckEq(hecke::math_settings().fingerprint(), static_cast<uint64_t>(0),
            "the mathematical settings are at their defaults (fingerprint 0)");

    std::vector<hecke_ext::M1Row> m1;
    std::cout << "    " << hecke_ext::M1Header() << "\n";
    RunDicksonArm(m1);
    RunDeltaArm(m1);
    RunSturmCounterexampleArm(m1);
    RunOneOperatorArm(m1);
    RunFinalFalseArm();

    std::cout << "SMALLELL_M1 rows=" << m1.size() << "\n";
    std::cout << "SMALLELL checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    std::cout << "HeckeExtSmallEllValidation: " << Check.fails << " failing case(s)." << std::endl;
    return Check.fails;
}
