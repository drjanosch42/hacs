// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// (hecke-extensions slot 9, WP-13) HeckeExtAlgebraInvariantsValidation -- see
// the header for the arm list.
// ===========================================================================

#include "HeckeExtAlgebraInvariantsValidation.h"

#include "HeckeAlgebraInvariants.h"
#include "HeckeDicksonMultiplication.h"
#include "HeckeExtBindingSupport.h"
#include "HeckeExtDriver.h"
#include "HeckeExtRecord.h"
#include "HeckeLocalAlgebra.h"
#include "HeckeModuleStructure.h"

#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/SubspaceOps.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Util/ValidationCheck.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace hecke_ext;

namespace {

oc_util::Checker Check("  *** FAIL [alginv] ");

std::string Tag(uint32_t n, uint32_t ell) {
    return "(" + std::to_string(n) + "," + std::to_string(ell) + ")";
}

ExtMatrix Mat(uint32_t ell, const std::vector<std::vector<uint32_t>>& rows) {
    ExtMatrix m(static_cast<uint32_t>(rows.size()), static_cast<uint32_t>(rows[0].size()), ell);
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < rows[i].size(); ++j) m.data[i][j] = rows[i][j] % ell;
    return m;
}

// The canonical row set of the span of `elts` (D x D matrices).
std::vector<std::vector<uint32_t>> SpanRows(const std::vector<ExtMatrix>& elts, uint32_t D, uint32_t ell) {
    hecke::RrefRowSpan s(D * D, ell);
    for (const ExtMatrix& e : elts) s.insert(hecke::VecOfMatrix(e));
    return s.rows();
}

// ---------------------------------------------------------------------------
// (A) against the certified route on real ideals.
// ---------------------------------------------------------------------------
void ArmA() {
    std::cout << "-- (A) the certificate-free route vs HeckeLocalAlgebra / HeckeModuleStructure\n";
    const std::pair<uint32_t, uint32_t> cells[] = {{10, 691}, {22, 5}, {58, 7}, {82, 7}, {12, 13}, {16, 11}};
    uint32_t ideals = 0, nonreduced = 0;
    for (const auto& c : cells) {
        ExtDriverOptions o;
        o.run_commutant = false;
        o.run_sturm = false;
        HeckeExtDriver d(c.first, c.second, o);
        d.run();
        const HeckeExtRecord& rec = d.record();
        for (uint32_t i = 0; i < rec.ideals.size(); ++i) {
            const LocalAlgebra& L = d.local_algebra(i);
            if (!L.ok) continue;
            ++ideals;
            const std::string tag = Tag(c.first, c.second) + " ideal " + std::to_string(i);
            const AlgebraInvariants A = ComputeAlgebraInvariants(L.t, L.ell);
            Check(A.dim_A == L.dim_A, tag + ": dim A");
            Check(A.basis.size() == L.basis.size(), tag + ": basis size");
            bool same_basis = A.basis.size() == L.basis.size();
            for (size_t k = 0; same_basis && k < A.basis.size(); ++k) same_basis = A.basis[k] == L.basis[k];
            Check(same_basis, tag + ": the algebra basis entrywise (both canonical RREF)");
            Check(A.local, tag + ": local (Berlekamp count 1)");
            Check(A.components == 1, tag + ": one maximal ideal");
            Check(A.residue_degree == L.f, tag + ": f");
            Check(A.dim_rad == L.dim_m_S, tag + ": dim rad == dim m_S");
            const std::vector<ExtMatrix> m_S = L.m_powers.empty() ? std::vector<ExtMatrix>() : L.m_powers[0];
            Check(SpanRows(A.radical, L.D, L.ell) == SpanRows(m_S, L.D, L.ell), tag + ": rad == m_S as spaces");
            Check(A.nilpotency_index == L.nu, tag + ": nilpotency index == nu");
            Check(A.hilbert == L.hilbert, tag + ": Hilbert function");
            Check(A.dim_socle == L.dim_socle, tag + ": dim soc");
            Check(SpanRows(A.socle, L.D, L.ell) == SpanRows(L.socle, L.D, L.ell), tag + ": soc as spaces");
            Check(A.gorenstein == L.gorenstein, tag + ": Gorenstein");
            Check(A.dim_tangent == L.embdim * L.f, tag + ": dim_F m/m^2 == embdim * f");
            if (L.nu > 1) ++nonreduced;
            // Jordan-Chevalley: the nilpotent part IS n_p = t_p - P_p(w_hat).
            for (size_t k = 0; k < L.t.size(); ++k) {
                const JordanChevalley jc = JordanChevalleySplit(L.t[k]);
                Check(jc.nilpotent == L.n[k], tag + ": nilpotent_part(t_" + std::to_string(L.primes[k]) + ") == n_p");
            }
            // The Loewy / socle series on the whole block == ModuleStructure's.
            const IdealRecord& id = rec.ideals[i];
            Check(!id.modules.empty(), tag + ": has a module structure");
            if (id.modules.empty()) continue;
            const ModuleStructure& M = id.modules.front().structure;
            const ExtMatrix I(L.D, L.ell);
            const ModuleSeries lo = LoewySeries(A, I);
            const ModuleSeries so = SocleSeries(A, I);
            Check(lo.dims == M.m_dims, tag + ": Loewy series == m_dims");
            Check(so.dims == M.socle_dims, tag + ": socle series == socle_dims");
        }
    }
    Check(ideals >= 10, "(A) at least ten certified-local ideals compared");
    Check(nonreduced >= 2, "(A) at least two with a nonzero radical");
    std::cout << "    " << ideals << " ideals compared, " << nonreduced << " non-reduced\n";
}

// ---------------------------------------------------------------------------
// (B) constructed algebras.
// ---------------------------------------------------------------------------
void ArmB() {
    std::cout << "-- (B) constructed algebras\n";
    const uint32_t ell = 7;
    {
        // F[x]/(x^3) on its regular representation: the shift.
        const ExtMatrix x = Mat(ell, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}});
        const AlgebraInvariants A = ComputeAlgebraInvariants({x}, ell);
        Check(A.dim_A == 3 && A.local && A.residue_degree == 1, "F[x]/(x^3): dim 3, local, f = 1");
        Check(A.nilpotency_index == 3, "F[x]/(x^3): nu = 3");
        Check(A.hilbert == std::vector<uint32_t>({1, 1, 1}), "F[x]/(x^3): Hilbert (1, 1, 1)");
        Check(A.dim_tangent == 1 && A.dim_socle == 1 && A.gorenstein, "F[x]/(x^3): embdim 1, Gorenstein");
        const ModuleSeries lo = LoewySeries(A, ExtMatrix(3, ell));
        Check(lo.dims == std::vector<uint32_t>({3, 2, 1, 0}), "F[x]/(x^3): Loewy series (3, 2, 1, 0)");
        const ModuleSeries so = SocleSeries(A, ExtMatrix(3, ell));
        Check(so.dims == std::vector<uint32_t>({0, 1, 2, 3}), "F[x]/(x^3): socle series (0, 1, 2, 3)");
    }
    {
        // F[x, y]/(x, y)^2 on F^3 = <1, x, y>: x e1 = e2, y e1 = e3.
        const ExtMatrix x = Mat(ell, {{0, 0, 0}, {1, 0, 0}, {0, 0, 0}});
        const ExtMatrix y = Mat(ell, {{0, 0, 0}, {0, 0, 0}, {1, 0, 0}});
        const AlgebraInvariants A = ComputeAlgebraInvariants({x, y}, ell);
        Check(A.dim_A == 3 && A.local, "F[x,y]/(x,y)^2: dim 3, local");
        Check(A.nilpotency_index == 2 && A.dim_tangent == 2, "F[x,y]/(x,y)^2: nu = 2, embdim 2");
        Check(A.dim_socle == 2 && !A.gorenstein, "F[x,y]/(x,y)^2: socle dim 2, NOT Gorenstein");
        Check(A.hilbert == std::vector<uint32_t>({1, 2}), "F[x,y]/(x,y)^2: Hilbert (1, 2)");
    }
    {
        // Split semisimple: diag(1, 2, 2) -- two maximal ideals, reduced.
        const ExtMatrix t = Mat(ell, {{1, 0, 0}, {0, 2, 0}, {0, 0, 2}});
        const AlgebraInvariants A = ComputeAlgebraInvariants({t}, ell);
        Check(A.dim_A == 2 && A.dim_rad == 0 && A.components == 2 && !A.local,
              "diag(1,2,2): dim 2, reduced, two maximal ideals, not local");
        Check(A.nilpotency_index == 1 && A.residue_degree == 0, "diag(1,2,2): nu = 1, f undefined (0)");
    }
    {
        // A residue field of degree 2: J = [[0,-1],[1,0]], J^2 = -1, and
        // x^2 + 1 is irreducible over F_7.
        const ExtMatrix J = Mat(ell, {{0, 6}, {1, 0}});
        const AlgebraInvariants A = ComputeAlgebraInvariants({J}, ell);
        Check(A.local && A.residue_degree == 2 && A.dim_rad == 0, "F_49: local, f = 2, reduced");
        Check(A.gorenstein && A.hilbert == std::vector<uint32_t>({1}), "F_49: Gorenstein, Hilbert (1)");
    }
    {
        // THE GALOIS-CONJUGATE TRAP.  A = diag(J, J), B = diag(J, -J): each
        // has minimal polynomial x^2 + 1 (irreducible), they commute, and
        // F[A, B] = F_49 x F_49 -- NOT local.
        const ExtMatrix A = Mat(ell, {{0, 6, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 6}, {0, 0, 1, 0}});
        const ExtMatrix B = Mat(ell, {{0, 6, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 6, 0}});
        Check(ComputeAlgebraInvariants({A}, ell).local, "trap: F[A] alone is local (a field)");
        Check(ComputeAlgebraInvariants({B}, ell).local, "trap: F[B] alone is local (a field)");
        const AlgebraInvariants AB = ComputeAlgebraInvariants({A, B}, ell);
        Check(!AB.local && AB.components == 2 && AB.dim_A == 4,
              "trap: F[A, B] = F_49 x F_49, two maximal ideals, not local");
    }
}

// ---------------------------------------------------------------------------
// (C) Jordan-Chevalley and the refusals.
// ---------------------------------------------------------------------------
void ArmC() {
    std::cout << "-- (C) Jordan-Chevalley, refusals\n";
    const uint32_t ell = 5;
    // s0 = diag(2, 2, 3), n0 = E_{12}: [s0, n0] = 0, so the split of s0 + n0
    // is (s0, n0) by uniqueness.
    const ExtMatrix s0 = Mat(ell, {{2, 0, 0}, {0, 2, 0}, {0, 0, 3}});
    const ExtMatrix n0 = Mat(ell, {{0, 1, 0}, {0, 0, 0}, {0, 0, 0}});
    // Conjugate by an invertible P so the test is not on a diagonal form.
    const ExtMatrix P = Mat(ell, {{1, 2, 0}, {0, 1, 3}, {1, 0, 1}});
    ExtMatrix Pc(P);
    const ExtMatrix Pinv = Pc.inverse();
    const ExtMatrix t = P * (s0 + n0) * Pinv;
    const JordanChevalley jc = JordanChevalleySplit(t);
    Check(jc.semisimple == P * s0 * Pinv, "JC: s == P s0 P^-1");
    Check(jc.nilpotent == P * n0 * Pinv, "JC: n == P n0 P^-1");
    // An irreducible quadratic factor: L = 2.
    const ExtMatrix J = Mat(ell, {{0, 3}, {1, 0}});   // x^2 - 3: 3 is a non-square mod 5
    const JordanChevalley jj = JordanChevalleySplit(J);
    Check(jj.lcm_degree == 2 && jj.nilpotent.is_zero(), "JC: a semisimple matrix with an F_25 eigenvalue has n = 0");

    bool threw = false;
    try {
        (void)ComputeAlgebraInvariants({Mat(ell, {{0, 1}, {0, 0}}), Mat(ell, {{1, 0}, {0, 2}})}, ell);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "a non-commuting family is refused");
    threw = false;
    try {
        const AlgebraInvariants A = ComputeAlgebraInvariants({Mat(ell, {{0, 1}, {0, 0}})}, ell);
        (void)LoewySeries(A, Mat(ell, {{0}, {1}}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "a subspace that is not A-stable is refused");
}

// ---------------------------------------------------------------------------
// (D) HeckeExtBindingSupport at (12, 5).
// ---------------------------------------------------------------------------
void ArmD() {
    std::cout << "-- (D) the binding-support builders at (12, 5)\n";
    const uint32_t n = 12, ell = 5;
    SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    hecke::SL2ZTessellationTables model(K);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& d : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(d.rel);
    LocalSystemSimplicialComplex lssc(K, std::move(rels), n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();

    hecke::SL2ZBoundaryPair pair(&coh, K);
    std::unique_ptr<EquivariantMappingCone> cone = pair.make_cone();
    const std::unique_ptr<ConeActionAtPrime> act = BuildConeActionAtPrime(pair, *cone, model, 2);
    Check(act->act->certificate().verify() == 0, "(12,5): the cone action at p = 2 certifies");
    Check(act->eta->certify() == 0, "(12,5): eta certifies over Z");
    const std::unique_ptr<BoundaryActionAtPrime> bd = BuildBoundaryActionAtPrime(pair, model, 3);
    Check(bd->bd->num_degrees() == 2, "(12,5): T_3^bd on two boundary degrees");

    const std::unique_ptr<hecke::HeckeInvolution> inv = BuildProductionInvolution(coh, K, model, n, ell, "validation");
    Check(inv->certificate().verify() == 0, "(12,5): the production involution certifies");
    const std::unique_ptr<HeckeSignSplit> split = BuildSignSplit(coh, K, model, *inv, n, ell, 1, {2, 3, 7});
    Check(split != nullptr && split->certificate().verify() == 0, "(12,5): the degree-1 sign split certifies");

    // L_* from (6, 5): m = 12 - 5 - 1 = 6.
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels_m;
    for (const auto& d : GenerateSL2ZRelationsWithGroupData(K, 6, ell)) rels_m.push_back(d.rel);
    LocalSystemSimplicialComplex lssc_m(K, std::move(rels_m), 7, ell);
    EquivariantCohomologyOnK coh_m(&lssc_m);
    coh_m.compute_cohomology();
    const std::vector<FlatMatrix> Lstar = DicksonInducedMaps(coh, coh_m, K, n, ell);
    Check(Lstar.size() == coh.get_H_eq_dimensions().size(), "(12,5): L_* in every degree");
    if (Lstar.size() > 1) {
        Check(Lstar[1].get_rows() == coh.get_H_eq_dimensions()[1] &&
                  Lstar[1].get_cols() == coh_m.get_H_eq_dimensions()[1],
              "(12,5): L_*^1 has shape dim H^1(M_12) x dim H^1(M_6)");
    }

    std::vector<bool> want(kIdentityItems + 1, true);
    const IdentityAssembly as = AssembleIdentityCertificate(coh, K, model, n, ell, 2, 3, want);
    Check(as.code == 0, "(12,5): IdentityCertificate::verify() == 0\n" + as.cert.to_string_offset(4));
    uint32_t k = 0, unavailable = 0;
    for (const auto& it : IdentityItems(as.cert)) {
        ++k;
        if (!as.supplied[k]) ++unavailable;
        Check(it.second->status != IdentityStatus::kFailed, "(12,5): item " + std::to_string(k) + " did not fail");
    }
    Check(k == kIdentityItems, "the certificate defines kIdentityItems items");
    Check(unavailable == 0, "(12,5): every item's inputs were assembled");
    Check(as.cert.residue_atlas.status == IdentityStatus::kPassed, "(12,5): item 6 (residue atlas) passed");
    Check(as.cert.socle_loewy_pd.status == IdentityStatus::kPassed, "(12,5): item 7 (socle <-> Loewy) passed");
}

} // namespace

uint32_t HeckeExtAlgebraInvariantsValidation(void) {
    Check.reset();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "--- HeckeExtAlgebraInvariantsValidation (slot-9 WP-13, console C5b) ---\n";
    try {
        ArmA();
        ArmB();
        ArmC();
        ArmD();
    } catch (const std::exception& e) {
        Check(false, std::string("unexpected exception: ") + e.what());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  HeckeExtAlgebraInvariantsValidation " << (Check.fails == 0 ? "PASSED" : "FAILED") << " ("
              << Check.checks << " checks, " << Check.fails << " failing, " << s << " s)\n";
    return Check.fails;
}
