// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E2 (slot 6, WP-2; OQ11) -- HeckePresentationValidation.
// See HeckePresentationValidation.h for the arm list and what is deferred.
//
// CONSTRUCTED TRUTHS.  The constructed arms hand-build the module matrices of
// small local algebras whose presentations are known in closed form (the
// shapes of HeckeExtAlgebraValidation), push them through the SHIPPED
// decomposition and hecke_ext::ComputeLocalAlgebra, and compare
// ComputePresentation with the closed form: the counts in every basis
// (conjugated by a seeded U as well), and -- in the stored basis, where the
// least-index generators are known -- the KNOWN relation tuples themselves,
// which must vanish on the generators, number min_relations, and generate the
// relation module over A_S.  Nothing is compared with another run of the same
// code except the determinism check.
//
// The (10, 691) arms go through HeckeExtDriver with
// `with_presentation = true` (the reach path of slot 6: no settings key), and
// run the same cell a second time with the option off, asserting that the
// unsplit and the two per-sign record texts are byte-identical.
//
// assert() is the harness idiom of this repository, but every check below is
// counted and reported and the return value is the number of failures.

#include "HeckePresentationValidation.h"

#include "HeckeExtDriver.h"
#include "HeckeExtReport.h"
#include "HeckeLocalAlgebra.h"
#include "HeckePresentation.h"

#include "../Golden/GoldenHarness.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/SubspaceOps.h"

#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "../Util/ValidationCheck.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using hecke::FpPoly;
using hecke::SplitMix64;
using hecke_ext::ExtMatrix;
using hecke_ext::LocalAlgebra;
using hecke_ext::Presentation;

oc_util::Checker Check("  [FAIL] ");

template <class T>
void CheckEq(const T& got, const T& want, const std::string& what) {
    ++Check.checks;
    if (!(got == want)) {
        ++Check.fails;
        std::cout << "  [FAIL] " << what << ": got " << got << ", want " << want << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Small matrix helpers (validation-local).
// ---------------------------------------------------------------------------

ExtMatrix Zero(uint32_t r, uint32_t c, uint32_t ell) { return ExtMatrix(r, c, ell); }
ExtMatrix Id(uint32_t d, uint32_t ell) { return ExtMatrix(d, ell); }

ExtMatrix Scalar(uint32_t d, uint32_t a, uint32_t ell) {
    ExtMatrix M(d, d, ell);
    for (uint32_t i = 0; i < d; ++i) M.data[i][i] = a % ell;
    return M;
}

// E_{ij}: one 1 at (i, j).
ExtMatrix Unit(uint32_t d, uint32_t i, uint32_t j, uint32_t ell) {
    ExtMatrix M(d, d, ell);
    M.data[i][j] = 1u;
    return M;
}

ExtMatrix Kron(const ExtMatrix& A, const ExtMatrix& B, uint32_t ell) {
    ExtMatrix M(A.rows * B.rows, A.cols * B.cols, ell);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t j = 0; j < A.cols; ++j)
            for (uint32_t k = 0; k < B.rows; ++k)
                for (uint32_t l = 0; l < B.cols; ++l)
                    M.data[i * B.rows + k][j * B.cols + l] =
                            ModArithmetic::mod_mul(A.data[i][j], B.data[k][l], ell);
    return M;
}


ExtMatrix SeededInvertible(uint32_t D, uint32_t ell, uint64_t seed) {
    SplitMix64 rng(seed);
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        ExtMatrix U(D, D, ell);
        for (uint32_t i = 0; i < D; ++i)
            for (uint32_t j = 0; j < D; ++j) U.data[i][j] = rng.next_mod(ell);
        if (U.is_invertible()) return U;
    }
    return ExtMatrix(D, ell);
}

// The least r >= 2 with X^2 - r irreducible over F_ell.
FpPoly IrreducibleQuadratic(uint32_t ell, uint32_t* r_out) {
    for (uint32_t r = 2; r < ell + 2; ++r) {
        const uint32_t rr = r % ell;
        if (rr == 0) continue;
        const FpPoly cand(ell, {ModArithmetic::mod_sub(0u, rr, ell), 0u, 1u});
        if (hecke::CertifyIrreducible(cand) == 0) {
            *r_out = rr;
            return cand;
        }
    }
    *r_out = 0;
    return FpPoly(ell);
}

// ---------------------------------------------------------------------------
// One constructed case.
// ---------------------------------------------------------------------------

using Tuple = std::vector<ExtMatrix>;   // (a_1, ..., a_t), each a D x D element of A_S

struct Case {
    std::string name;
    uint32_t ell = 0;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    // Closed form.
    uint32_t D = 0, f = 1, dim_A = 0, t = 0, dim_R = 0, rels = 0;
    std::vector<uint32_t> generators;   // in the stored basis (least-index lift)
    std::vector<Tuple> known;           // known minimal relations, stored basis
};

// `B_out` (optional) receives the block's basis columns: ComputeLocalAlgebra
// works in the BLOCK's coordinates, t_p|_V = B^{-1} t_p B, so every closed-form
// object stated in the case's own coordinates is transported by B.
bool BuildLocal(const Case& c, bool conjugate, LocalAlgebra& L, ExtMatrix* B_out = nullptr) {
    std::vector<std::pair<uint64_t, ExtMatrix>> ops = c.ops;
    const uint32_t D = ops[0].second.rows;
    if (conjugate) {
        bool moved = false;
        for (uint32_t attempt = 0; attempt < 32u && !moved; ++attempt) {
            const ExtMatrix U = SeededInvertible(D, c.ell, 0x9E5E27A710000001ull + 977u * attempt + D);
            ExtMatrix Uinv = U;
            Uinv = Uinv.inverse();
            std::vector<std::pair<uint64_t, ExtMatrix>> cand = ops;
            for (auto& op : cand) op.second = Uinv * (op.second * U);
            for (size_t j = 0; j < cand.size() && !moved; ++j) {
                if (!(cand[j].second == ops[j].second)) moved = true;
            }
            if (moved) ops = std::move(cand);
        }
        Check(moved, c.name + ": a non-centralising conjugation was found");
        if (!moved) return false;
    }
    hecke::HeckeModuleDecomposition dec(ops, c.ell);
    dec.compute();
    if (dec.blocks().size() != 1u) {
        Check(false, c.name + ": expected one primary block");
        return false;
    }
    L = hecke_ext::ComputeLocalAlgebra(dec.blocks()[0], ops, c.ell);
    if (B_out) *B_out = dec.blocks()[0].basis_columns;
    Check(L.ok, c.name + ": block certified local");
    return L.ok;
}

// Coordinates of an element of A_S in the stored RREF basis.
std::vector<uint32_t> Coords(const LocalAlgebra& L, const ExtMatrix& X) {
    return hecke::AlgebraCoords(L.coords, hecke::RrefPivots(L.coords), X, L.ell,
                                "HeckePresentationValidation");
}

std::vector<uint32_t> TupleCoords(const LocalAlgebra& L, const Tuple& k) {
    std::vector<uint32_t> out;
    for (const ExtMatrix& a : k) {
        const std::vector<uint32_t> c = Coords(L, a);
        out.insert(out.end(), c.begin(), c.end());
    }
    return out;
}

// sum_i a_i w_i for column vectors w_i (D x 1).
std::vector<uint32_t> Evaluate(const Tuple& k, const std::vector<ExtMatrix>& w,
                               uint32_t D, uint32_t ell) {
    std::vector<uint32_t> v(D, 0u);
    for (size_t i = 0; i < k.size(); ++i) {
        const ExtMatrix aw = k[i] * w[i];
        for (uint32_t r = 0; r < D; ++r) v[r] = ModArithmetic::mod_add(v[r], aw.data[r][0], ell);
    }
    return v;
}

bool IsZero(const std::vector<uint32_t>& v) {
    for (uint32_t x : v) if (x) return false;
    return true;
}

void CheckCase(const Case& c, bool conjugate) {
    const std::string tag =
            c.name + " (ell=" + std::to_string(c.ell) + (conjugate ? ", conjugated)" : ")");
    LocalAlgebra L;
    ExtMatrix B;
    if (!BuildLocal(c, conjugate, L, &B)) return;
    const Presentation P = hecke_ext::ComputePresentation(L);

    Check(P.available, tag + ": available");
    CheckEq(P.verify(), 0u, tag + ": verify()");
    CheckEq(P.D, c.D, tag + ": D");
    CheckEq(P.f, c.f, tag + ": f");
    CheckEq(P.dim_A, c.dim_A, tag + ": dim A_S");
    CheckEq(P.min_generators, c.t, tag + ": min_generators");
    CheckEq(P.min_generators, L.min_generators, tag + ": t == LocalAlgebra::min_generators");
    CheckEq(P.dim_relation_module, c.dim_R, tag + ": dim_F R");
    CheckEq(P.min_relations, c.rels, tag + ": min_relations");
    CheckEq(P.is_free(), c.rels == 0u, tag + ": is_free()");
    // Nakayama's two counts, from the invariants alone.
    CheckEq(P.min_generators * P.dim_A, P.D + P.dim_relation_module,
            tag + ": t * dim A_S == D + dim R (exactness of 0 -> R -> A^t -> V -> 0)");

    // Every relation lies in m_S A_S^t: a minimal presentation has no unit
    // coefficient (Nakayama), so each component is in m_S.
    {
        hecke::RrefRowSpan mspan(P.dim_A, c.ell);
        for (const ExtMatrix& m : L.cyclicity.m_basis) mspan.insert(Coords(L, m));
        bool all_in = true;
        for (const std::vector<uint32_t>& r : P.relations) {
            for (uint32_t i = 0; i < P.min_generators; ++i) {
                std::vector<uint32_t> comp(r.begin() + i * P.dim_A, r.begin() + (i + 1) * P.dim_A);
                if (!mspan.contains(comp)) all_in = false;
            }
        }
        Check(all_in, tag + ": every relation lies in m_S A_S^t");
    }

    // Determinism.
    {
        LocalAlgebra L2;
        if (BuildLocal(c, conjugate, L2)) {
            Check(hecke_ext::ComputePresentation(L2).to_json() == P.to_json(),
                  tag + ": to_json() is deterministic");
        }
    }

    if (conjugate) return;

    // --- the known presentation, "up to the stored basis" ([X] Sec. 5) -----
    // The closed form is stated in the case's own coordinates; the local
    // algebra lives in the block's, so generators and relation entries are
    // transported by the block basis B: w' = B^{-1} w, a' = B^{-1} a B.  The
    // KNOWN presentation is then checked as a presentation in its own right
    // -- its generators generate V over A_S, its relations vanish, generate
    // the whole relation module and are minimal -- and its counts must be
    // the computed object's.
    std::vector<ExtMatrix> w;
    for (uint32_t g : c.generators) {
        ExtMatrix e(c.D, 1u, c.ell);
        e.data[g][0] = 1u;
        w.push_back(hecke::SolveColumnsExact(B, e, "HeckePresentationValidation(w)"));
    }
    CheckEq(static_cast<uint32_t>(w.size()), P.min_generators, tag + ": known generator count");
    {
        hecke::RrefRowSpan span(c.D, c.ell);
        for (const ExtMatrix& wi : w) {
            for (const ExtMatrix& b : L.basis) {
                const ExtMatrix bw = b * wi;
                std::vector<uint32_t> v(c.D, 0u);
                for (uint32_t r = 0; r < c.D; ++r) v[r] = bw.data[r][0];
                span.insert(std::move(v));
            }
        }
        CheckEq(span.dim(), c.D, tag + ": the known generators generate V over A_S");
    }
    CheckEq(static_cast<uint32_t>(c.known.size()), P.min_relations, tag + ": known relation count");
    const uint32_t width = static_cast<uint32_t>(w.size()) * P.dim_A;
    hecke::RrefRowSpan known_span(width, c.ell);
    hecke::RrefRowSpan m_known(width, c.ell);
    for (size_t k = 0; k < c.known.size(); ++k) {
        Tuple kt;
        for (const ExtMatrix& a : c.known[k]) {
            kt.push_back(hecke::SolveColumnsExact(B, a * B, "HeckePresentationValidation(a)"));
        }
        Check(IsZero(Evaluate(kt, w, c.D, c.ell)),
              tag + ": known relation " + std::to_string(k) + " vanishes on the known generators");
        for (const ExtMatrix& b : L.basis) {
            Tuple bk;
            for (const ExtMatrix& a : kt) bk.push_back(b * a);
            known_span.insert(TupleCoords(L, bk));
        }
        for (const ExtMatrix& m : L.cyclicity.m_basis) {
            Tuple mk;
            for (const ExtMatrix& a : kt) mk.push_back(m * a);
            m_known.insert(TupleCoords(L, mk));
        }
    }
    CheckEq(known_span.dim(), P.dim_relation_module,
            tag + ": the known relations generate the whole relation module over A_S");
    CheckEq((known_span.dim() - m_known.dim()), P.min_relations * P.f,
            tag + ": the known relations are minimal (dim R / m_S R)");
    std::cout << "    " << tag << ": " << P.to_json() << "\n";
}

std::vector<Case> BuildCases(uint32_t ell) {
    std::vector<Case> cases;
    const uint32_t a = 3u % ell, b = 5u % ell;

    // (1) A^dual for A = F[x,y]/(x,y)^2 -- two generators x*, y*, three
    // relations x g1 = y g2, y g1 = 0, x g2 = 0 ([X] Sec. 2.2 Stage 5).
    {
        Case c;
        c.name = "A^dual, A = F[x,y]/(x,y)^2";
        c.ell = ell;
        // Basis {1*, x*, y*}; x acts by the transpose of the regular action.
        const ExtMatrix x = Unit(3, 0, 1, ell), y = Unit(3, 0, 2, ell);
        c.ops = {{2, Scalar(3, a, ell) + x}, {3, Scalar(3, b, ell) + y}};
        c.D = 3; c.f = 1; c.dim_A = 3; c.t = 2; c.dim_R = 3; c.rels = 3;
        c.generators = {1, 2};
        const ExtMatrix z = Zero(3, 3, ell);
        c.known = {Tuple{x, z - y}, Tuple{y, z}, Tuple{z, x}};
        cases.push_back(c);
    }
    // (2) the cyclic F[x]/(x^3) -- free of rank one.
    {
        Case c;
        c.name = "F[x]/(x^3) (cyclic)";
        c.ell = ell;
        const ExtMatrix N = Unit(3, 1, 0, ell) + Unit(3, 2, 1, ell);
        c.ops = {{2, Scalar(3, a, ell) + N}};
        c.D = 3; c.f = 1; c.dim_A = 3; c.t = 1; c.dim_R = 0; c.rels = 0;
        c.generators = {0};
        cases.push_back(c);
    }
    // (3) A + A over F[x]/(x^3) -- free of rank two.
    {
        Case c;
        c.name = "F[x]/(x^3) acting on A + A (free, rank 2)";
        c.ell = ell;
        const ExtMatrix N = Unit(3, 1, 0, ell) + Unit(3, 2, 1, ell);
        c.ops = {{2, Kron(Id(2, ell), Scalar(3, a, ell) + N, ell)}};
        c.D = 6; c.f = 1; c.dim_A = 3; c.t = 2; c.dim_R = 0; c.rels = 0;
        c.generators = {0, 3};
        cases.push_back(c);
    }
    // (4) A + A/m over F[eps]/eps^2 -- the (10, 691) shape: two generators,
    // one relation eps . g2 = 0.
    {
        Case c;
        c.name = "F[eps]/eps^2 acting on A + A/m (the (10,691) shape)";
        c.ell = ell;
        const ExtMatrix eps = Unit(3, 1, 0, ell);
        c.ops = {{2, Scalar(3, a, ell) + eps}};
        c.D = 3; c.f = 1; c.dim_A = 2; c.t = 2; c.dim_R = 1; c.rels = 1;
        c.generators = {0, 2};
        c.known = {Tuple{Zero(3, 3, ell), eps}};
        cases.push_back(c);
    }
    // (5) f = 2: F_{ell^2}[eps]/eps^2 acting on A + A/m, ONE operator
    // t = alpha + eps.  Over K the shape of (4): two generators and ONE
    // relation, whose F_ell-count is f = 2 -- the reading of [X]'s
    // "dim_F(ker / m_S ker)" over the residue field.
    {
        uint32_t r = 0;
        (void)IrreducibleQuadratic(ell, &r);
        if (r != 0) {
            Case c;
            c.name = "F_{ell^2}[eps]/eps^2 acting on A + A/m (f = 2)";
            c.ell = ell;
            ExtMatrix C(2, 2, ell);   // companion of X^2 - r
            C.data[0][1] = r;
            C.data[1][0] = 1u;
            const ExtMatrix eps3 = Unit(3, 1, 0, ell);
            const ExtMatrix eps = Kron(Id(2, ell), eps3, ell);
            c.ops = {{2, Kron(C, Id(3, ell), ell) + eps}};
            c.D = 6; c.f = 2; c.dim_A = 4; c.t = 2; c.dim_R = 2; c.rels = 1;
            // V = K (x) W with W's basis (1, eps, e); index k*3 + i.  m_S V is
            // spanned by the eps-images e_1, e_4; the least-index lift keeps
            // e_0 (whose A_S-orbit is K (x) A, i.e. e_0, e_1, e_3, e_4) and
            // then e_2.
            c.generators = {0, 2};
            c.known = {Tuple{Zero(6, 6, ell), eps}};
            cases.push_back(c);
        }
    }
    // (6) the field F_{ell^2} (m_S = 0): free of rank one.
    {
        uint32_t r = 0;
        (void)IrreducibleQuadratic(ell, &r);
        if (r != 0) {
            Case c;
            c.name = "F_{ell^2} (a field)";
            c.ell = ell;
            ExtMatrix C(2, 2, ell);
            C.data[0][1] = r;
            C.data[1][0] = 1u;
            c.ops = {{2, C}};
            c.D = 2; c.f = 2; c.dim_A = 2; c.t = 1; c.dim_R = 0; c.rels = 0;
            c.generators = {0};
            cases.push_back(c);
        }
    }
    return cases;
}

// ---------------------------------------------------------------------------
// Tampering.
// ---------------------------------------------------------------------------

void CheckTampering(uint32_t ell) {
    const std::string tag = "tamper (ell=" + std::to_string(ell) + ")";
    const std::vector<Case> cases = BuildCases(ell);
    LocalAlgebra L;
    if (!BuildLocal(cases[0], false, L)) return;
    const Presentation P0 = hecke_ext::ComputePresentation(L);
    CheckEq(P0.verify(), 0u, tag + ": the untampered object verifies");

    auto expect = [&](const Presentation& P, uint32_t code, const std::string& what) {
        CheckEq(P.verify(), code, tag + ": " + what + " -> code " + std::to_string(code) + " (" +
                                          Presentation::verify_message(code) + ")");
    };

    { Presentation P = P0; P.available = false; expect(P, 2, "availability cleared"); }
    {
        Presentation P = P0;
        P.algebra.a_basis[0].data[0][0] = ModArithmetic::mod_add(P.algebra.a_basis[0].data[0][0], 1u, ell);
        expect(P, 1, "an A_S basis entry perturbed");
    }
    {
        Presentation P = P0;
        P.algebra.t[0].data[0][0] = ModArithmetic::mod_add(P.algebra.t[0].data[0][0], 1u, ell);
        expect(P, 1, "a stored t_p perturbed");
    }
    { Presentation P = P0; P.min_generators -= 1; expect(P, 2, "min_generators decremented"); }
    {
        // A generating but non-least lift: {0, 2} spans nothing new at 0
        // (e_0 = 1* lies in m_S V), so the re-derived lift disagrees.
        Presentation P = P0;
        P.generator_indices = {0, 2};
        P.generators = ExtMatrix(P.D, 2, ell);
        P.generators.data[0][0] = 1u;
        P.generators.data[2][1] = 1u;
        expect(P, 4, "a non-least generator lift");
    }
    {
        // Perturb a relation coordinate whose evaluation column is nonzero.
        Presentation P = P0;
        bool done = false;
        for (uint32_t i = 0; i < P.min_generators && !done; ++i) {
            for (uint32_t j = 0; j < P.dim_A && !done; ++j) {
                const ExtMatrix& bj = P.algebra.a_basis[j];
                bool nonzero = false;
                for (uint32_t r = 0; r < P.D; ++r) if (bj.data[r][P.generator_indices[i]]) nonzero = true;
                if (!nonzero) continue;
                uint32_t& x = P.relations[0][i * P.dim_A + j];
                x = ModArithmetic::mod_add(x, 1u, ell);
                done = true;
            }
        }
        Check(done, tag + ": found a perturbable relation coordinate");
        expect(P, 6, "a relation coordinate perturbed");
    }
    {
        Presentation P = P0;
        P.relations.pop_back();
        P.min_relations -= 1;
        expect(P, 7, "a relation dropped (count kept consistent)");
    }
    { Presentation P = P0; P.dim_relation_module += 1; expect(P, 7, "dim R incremented"); }
    {
        Presentation P = P0;
        P.relations.push_back(P.relations[0]);
        P.min_relations += 1;
        expect(P, 8, "a redundant relation added (not minimal)");
    }
    {
        Presentation P = P0;
        P.relations[0][0] = ell;   // out of range
        expect(P, 2, "a relation entry >= ell");
    }
    {
        // A non-local algebra gives an unavailable presentation.
        LocalAlgebra bad;
        const Presentation P = hecke_ext::ComputePresentation(bad);
        Check(!P.available, tag + ": a non-local block gives available = false");
        expect(P, 2, "the unavailable object");
    }
}

// ---------------------------------------------------------------------------
// (10, 691) through the driver.
// ---------------------------------------------------------------------------

void CheckRamanujanCell() {
    std::cout << "-- (10, 691) through HeckeExtDriver (with_presentation)\n";
    const uint32_t n = 10, ell = 691;

    hecke_ext::ExtDriverOptions on;
    on.with_sign_split = true;
    on.with_presentation = true;
    hecke_ext::HeckeExtDriver d(n, ell, on);
    d.run();
    Check(d.has_presentations(), "(10,691): presentations built");
    const hecke_ext::HeckeExtRecord& r = d.record();
    CheckEq(static_cast<uint32_t>(r.ideals.size()), 1u, "(10,691): exactly one maximal ideal");
    if (r.ideals.empty()) return;

    // --- the unsplit H^1_m: t = 2 and its one minimal relation.
    const Presentation& U = d.presentation(0);
    Check(U.available, "(10,691) H^1_m: available");
    CheckEq(U.verify(), 0u, "(10,691) H^1_m: verify()");
    CheckEq(U.D, 3u, "(10,691) H^1_m: D");
    CheckEq(U.dim_A, 2u, "(10,691) H^1_m: dim A_S (F[eps]/eps^2)");
    CheckEq(U.f, 1u, "(10,691) H^1_m: f");
    CheckEq(U.min_generators, 2u, "(10,691) H^1_m: t = 2");
    CheckEq(U.dim_relation_module, 1u, "(10,691) H^1_m: dim_F R = 1");
    CheckEq(U.min_relations, 1u, "(10,691) H^1_m: exactly one minimal relation");
    Check(U.to_json() == hecke_ext::ComputePresentation(d.local_algebra(0)).to_json(),
          "(10,691) H^1_m: the driver's object == the free function on its local algebra");
    std::cout << "    H^1_m " << U.to_json() << "\n";

    // --- H^1_+: free of rank one over T_m.
    Check(d.has_sign_records(), "(10,691): per-sign records built");
    if (d.has_sign_records()) {
        CheckEq(d.num_sign_presentations(true), d.sign_record(true).ideals.size(),
                "(10,691): one presentation per plus ideal");
        CheckEq(d.num_sign_presentations(false), d.sign_record(false).ideals.size(),
                "(10,691): one presentation per minus ideal");
        CheckEq(d.num_sign_presentations(true), static_cast<size_t>(1),
                "(10,691): H^1_+ carries one maximal ideal");
        if (d.num_sign_presentations(true) == 1) {
            const Presentation& Pp = d.sign_presentation(true, 0);
            Check(Pp.available, "(10,691) H^1_+: available");
            CheckEq(Pp.verify(), 0u, "(10,691) H^1_+: verify()");
            CheckEq(Pp.min_generators, 1u, "(10,691) H^1_+: one generator");
            CheckEq(Pp.min_relations, 0u, "(10,691) H^1_+: no relation");
            Check(Pp.is_free(), "(10,691) H^1_+: free");
            CheckEq(Pp.D, Pp.dim_A, "(10,691) H^1_+: rank one (dim V == dim T_m)");
            std::cout << "    H^1_+ " << Pp.to_json() << "\n";
        }
        for (size_t i = 0; i < d.num_sign_presentations(false); ++i) {
            const Presentation& Pm = d.sign_presentation(false, static_cast<uint32_t>(i));
            CheckEq(Pm.verify(), 0u, "(10,691) H^1_-: verify()");
            std::cout << "    H^1_- " << Pm.to_json() << "\n";
        }
    }

    // --- the option does not leak into the record.
    hecke_ext::ExtDriverOptions off = on;
    off.with_presentation = false;
    hecke_ext::HeckeExtDriver d0(n, ell, off);
    d0.run();
    Check(!d0.has_presentations(), "(10,691): nothing built with the option off");
    Check(hecke_ext::BuildHeckeExtReportJson(d.record()) ==
                  hecke_ext::BuildHeckeExtReportJson(d0.record()),
          "(10,691): the record text is byte-identical with with_presentation on/off");
    if (d.has_sign_records() && d0.has_sign_records()) {
        for (int side = 0; side < 2; ++side) {
            Check(hecke_ext::BuildHeckeExtReportJson(d.sign_record(side == 0)) ==
                          hecke_ext::BuildHeckeExtReportJson(d0.sign_record(side == 0)),
                  std::string("(10,691): the ") + (side == 0 ? "plus" : "minus") +
                          " record text is byte-identical with with_presentation on/off");
        }
    }
    Check(BuildHeckeExtReportJson(d.record()).find("\"presentation\":{\"generator") ==
                  std::string::npos,
          "(10,691): the OPTION alone attaches no `presentation` object (only the key does)");
    std::cout << "    presentation_s = " << std::fixed << std::setprecision(6)
              << d.timings().presentation_s << " s\n";
}

size_t CountOccurrences(const std::string& s, const std::string& needle) {
    size_t n = 0;
    for (size_t at = s.find(needle); at != std::string::npos; at = s.find(needle, at + 1)) ++n;
    return n;
}

// Removes every `,"presentation":{...}` member (the ideal-level object) from a
// whitespace-free record text, by bracket matching.
std::string StripPresentationObjects(const std::string& s) {
    static const std::string kNeedle = ",\"presentation\":{\"generator_basis\"";
    std::string out = s;
    size_t at;
    while ((at = out.find(kNeedle)) != std::string::npos) {
        size_t i = at + std::string(",\"presentation\":").size();
        int depth = 0;
        bool in_str = false;
        for (; i < out.size(); ++i) {
            const char c = out[i];
            if (in_str) { if (c == '\\') ++i; else if (c == '"') in_str = false; continue; }
            if (c == '"') in_str = true;
            else if (c == '{' || c == '[') ++depth;
            else if (c == '}' || c == ']') { if (--depth == 0) { ++i; break; } }
        }
        out.erase(at, i - at);
    }
    return out;
}

// The top-level `settings` member of a whitespace-free record text.
std::string SettingsMember(const std::string& s) {
    static const std::string kKey = ",\"settings\":";
    const size_t at = s.find(kKey);
    if (at == std::string::npos) return std::string();
    const size_t b = at + kKey.size();
    int depth = 0;
    for (size_t i = b; i < s.size(); ++i) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}' && --depth == 0) return s.substr(b, i + 1 - b);
    }
    return std::string();
}

// ---------------------------------------------------------------------------
// (hecke-extensions slot 7, WP-2; [X] Sec. 8 stage 1 (2), [X] Sec. 5's key
// arm) THE `presentation` SETTINGS KEY.
//   (k1) registered, default `off`, all-defaults fingerprint still 0;
//   (k2) ABSENT under the default, and the default record's golden text
//        (the harness's path-scoped settings recipe) names neither new key --
//        the registration moved no golden input (the golden gate is the
//        proof; this is its in-process witness);
//   (k3) under `presentation = on`: one object per ideal of the record and of
//        both per-sign records, each {"generator_basis":"block",
//        "relation_basis":"algebra_rref","fragment":<hecke_presentation/v1>}
//        with the fragment the driver's own presentation, verbatim;
//   (k4) NOT A HASH INPUT: the golden ideal / certify hashes and the record's
//        `hashes` are unchanged by turning it on, and the on-text with the
//        objects stripped and the setting put back IS the off-text;
//   (k5) verify() code 24 refuses an object under `presentation = off`.
// ---------------------------------------------------------------------------
void CheckPresentationKey() {
    std::cout << "-- the `presentation` settings key (slot-7 WP-2, [X] Sec. 5 key arm)\n";
    const uint32_t n = 10, ell = 691;

    // (k1)
    Check(hecke::MathSettings::known_key("presentation"), "(k1) `presentation` is registered");
    Check(hecke::MathSettings::defaults().presentation == hecke::PresentationMode::Off,
          "(k1) its default is `off`");
    CheckEq(hecke::MathSettings::defaults().fingerprint(), uint64_t(0),
            "(k1) the all-defaults fingerprint is still 0");

    hecke_ext::ExtDriverOptions opts;
    opts.with_sign_split = true;

    // The default run (the ambient settings are the defaults).
    Check(hecke::math_settings().is_default(), "(k2) the ambient settings are the defaults");
    hecke_ext::HeckeExtDriver d0(n, ell, opts);
    d0.run();
    const std::string off_text = hecke_ext::BuildHeckeExtReportJson(d0.record());
    CheckEq(CountOccurrences(off_text, "\"presentation\":{\"generator_basis\""), size_t(0),
            "(k2) no `presentation` object under the default");
    const std::string golden_text = GoldenRedactJsonKeys(
            GoldenReduceSettingsToFingerprint(GoldenNormalizeJsonWhitespace(off_text)),
            {"commit"});
    // (slot-7 WP-3, stage 2 item 7) The `predictions` SETTING is still seen
    // only through the fingerprint (asserted just below); what the golden
    // text names since item 7 is its CONSUMER's record keys, `predictions_mode`
    // and `predictions` ([E] sec:schema (d)) -- not the settings entry.
    Check(golden_text.find("presentation") == std::string::npos &&
                  golden_text.find("\"predictions\":\"annotate\"") == std::string::npos &&
                  golden_text.find("\"predictions_mode\":\"annotate\"") != std::string::npos,
          "(k2) the default record's golden text names neither new settings entry "
          "(only item 7's record keys)");
    Check(golden_text.find("\"settings\":{\"fingerprint\":\"0000000000000000\"}") !=
                  std::string::npos,
          "(k2) and sees `settings` as the all-defaults fingerprint only");

    // (k3) presentation = on, stamped as a console would.
    hecke::MathSettings on_settings;
    Check(on_settings.set("presentation", "on", hecke::SettingSource::Console),
          "(k3) set(presentation = on)");
    CheckEq(on_settings.canonical_text(), std::string("presentation = on\n"),
            "(k3) the canonical text");
    Check(on_settings.fingerprint() != 0, "(k3) a non-default presentation fingerprints");
    std::string on_text, on_plus, on_minus;
    {
        const hecke::ScopedMathSettings guard(on_settings);
        hecke_ext::HeckeExtDriver d(n, ell, opts);
        d.run();
        Check(d.has_presentations(), "(k3) the key builds the presentations");
        const hecke_ext::HeckeExtRecord& r = d.record();
        CheckEq(r.verify(), 0u, "(k3) the record verifies with its presentation objects");
        for (size_t i = 0; i < r.ideals.size(); ++i) {
            Check(r.ideals[i].presentation_present &&
                          r.ideals[i].presentation_fragment ==
                                  d.presentation(static_cast<uint32_t>(i)).to_json(),
                  "(k3) ideal " + std::to_string(i) + " carries the driver's presentation");
        }
        on_text = hecke_ext::BuildHeckeExtReportJson(r);
        CheckEq(CountOccurrences(on_text,
                                 "\"presentation\":{\"generator_basis\":\"block\","
                                 "\"relation_basis\":\"algebra_rref\",\"fragment\":{\"schema\":"
                                 "\"hecke_presentation/v1\""),
                r.ideals.size(), "(k3) one wrapped object per ideal, in the [E] encoding");
        Check(d.has_sign_records(), "(k3) per-sign records built");
        if (d.has_sign_records()) {
            for (int side = 0; side < 2; ++side) {
                const bool plus = (side == 0);
                const hecke_ext::HeckeExtRecord& sr = d.sign_record(plus);
                CheckEq(sr.verify(), 0u, "(k3) a per-sign record verifies");
                for (size_t i = 0; i < sr.ideals.size(); ++i) {
                    Check(sr.ideals[i].presentation_present &&
                                  sr.ideals[i].presentation_fragment ==
                                          d.sign_presentation(plus, static_cast<uint32_t>(i))
                                                  .to_json(),
                          std::string("(k3) ") + (plus ? "plus" : "minus") + " ideal " +
                                  std::to_string(i) + " carries its per-sign presentation");
                }
                (plus ? on_plus : on_minus) = hecke_ext::BuildHeckeExtReportJson(sr);
            }
        }

        // (k4) not a hash input.
        CheckEq(GoldenHashExtIdeals(r), GoldenHashExtIdeals(d0.record()),
                "(k4) the golden ext_ideals hash is unchanged by turning it on");
        CheckEq(GoldenHashExtCertify(r), GoldenHashExtCertify(d0.record()),
                "(k4) the golden ext_certify hash is unchanged by turning it on");
        Check(r.hashes == d0.record().hashes, "(k4) the record's `hashes` are unchanged");
    }
    Check(hecke::math_settings().is_default(), "(k3) the scoped setting is restored");
    {
        std::string stripped = StripPresentationObjects(on_text);
        const std::string on_settings_json = SettingsMember(stripped);
        const std::string off_settings_json = SettingsMember(off_text);
        Check(!on_settings_json.empty() && on_settings_json != off_settings_json,
              "(k4) the settings block records the non-default key");
        const size_t at = stripped.find(on_settings_json);
        if (at != std::string::npos) stripped.replace(at, on_settings_json.size(), off_settings_json);
        Check(stripped == off_text,
              "(k4) on-text minus its presentation objects, with the setting put back, IS the "
              "default text");
        if (d0.has_sign_records() && !on_plus.empty()) {
            for (int side = 0; side < 2; ++side) {
                const std::string def = hecke_ext::BuildHeckeExtReportJson(d0.sign_record(side == 0));
                std::string s = StripPresentationObjects(side == 0 ? on_plus : on_minus);
                const std::string sj = SettingsMember(s);
                const size_t p = s.find(sj);
                if (p != std::string::npos) s.replace(p, sj.size(), SettingsMember(def));
                Check(s == def, std::string("(k4) the same for the ") +
                                        (side == 0 ? "plus" : "minus") + " record");
            }
        }
    }

    // (k5) the record refuses an object the setting does not ask for.
    {
        hecke_ext::HeckeExtRecord t = d0.record();
        if (!t.ideals.empty()) {
            t.ideals[0].presentation_present = true;
            t.ideals[0].presentation_fragment = "{}";
            CheckEq(t.verify(), 24u, "(k5) verify() refuses a presentation under `off`");
        }
    }
}

} // namespace

uint32_t HeckePresentationValidation(void) {
    std::cout << "=== HeckePresentationValidation (hecke-extensions E2 / slot-6 WP-2, OQ11) ==="
              << std::endl;
    std::cout << "    generators and relations of X_{i,m} over T_m, on request ([X] Sec. 2.2"
              << std::endl
              << "    Stage 5); ExtDriverOptions::with_presentation computes, and since slot-7"
              << std::endl
              << "    WP-2 the settings key `presentation = on` also attaches the record object"
              << std::endl
              << "    (key arm: absent by default, not a hash input)." << std::endl;

    Check.reset();

    for (uint32_t ell : {5u, 13u, 691u, 134217757u}) {
        for (const Case& c : BuildCases(ell)) {
            CheckCase(c, false);
            CheckCase(c, true);
        }
        CheckTampering(ell);
    }
    CheckRamanujanCell();
    CheckPresentationKey();

    std::cout << "    checks " << Check.checks << ", failing " << Check.fails << std::endl;
    if (Check.fails == 0) {
        std::cout << "=== HeckePresentationValidation PASSED ===" << std::endl;
    } else {
        std::cout << "=== HeckePresentationValidation FAILED ===" << std::endl;
    }
    return Check.fails;
}
