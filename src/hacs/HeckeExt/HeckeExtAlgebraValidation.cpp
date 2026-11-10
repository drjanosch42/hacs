// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E1a (slot 2, WP-4 + WP-5) -- HeckeExtAlgebraValidation.
//
// CONSTRUCTED TRUTHS.  Every arm below builds a local commutative F_ell-algebra
// A whose invariants are known in closed form -- A is a MONOMIAL algebra
// F_{ell^f}[x_1..x_v]/(a monomial ideal), so dim A, the ideal filtration, the
// Hilbert function, the socle and the embedding dimension are read off the
// monomial set -- represents V as A (or A + A) by the regular representation,
// hands the resulting matrices to the SHIPPED decomposition
// (hecke::HeckeModuleDecomposition) and then to hecke_ext::ComputeLocalAlgebra,
// and compares every invariant with the closed form.  Nothing here is compared
// against another run of the same code.
//
// The whole grid is run twice, the second time with every operator conjugated
// by a seeded invertible U, because every invariant of A_S is a conjugation
// invariant and a basis-dependent one would show up immediately.
//
// The two traps of [E] `cl:resfield` / [XA B2] have their own arms:
//   * f = 2, A = F_{ell^2}[eps]/eps^2, ONE operator t = alpha + eps:
//     embdim must be 1 (the K-span test), not 2 (an F_ell-span would count one
//     K-direction as f), and `active` must be {t} -- the witness's own
//     direction must not be lost;
//   * the same case computed with P(w) in place of P(w_hat) gives n = 0, i.e.
//     the lost direction, which is asserted explicitly as a negative control.
//
// WP-5 (E1a-2) ADDED the commutant arms of [X] Sec. 5's row, at the end of
// this file: the socle bound `dim C >= dim A_S + f*t*s - dim_F soc A_S`
// against an INDEPENDENT naive dim C on every case (all have D <= 24);
// `excluded_by_socle_bound` on every Gorenstein non-cyclic case (the
// (10, 691) shape A + A/m: dim C = 5, bound 5); the positive case
// V = A^dual for A = F[x,y]/(x,y)^2, where (b) passes and (a) fails;
// structured == naive == the closed form wherever all three run; the R3
// `CommutantState` arms (interrupt at EVERY reachable budget, resume,
// entrywise-identical state and certificate, a lower budget refused, the
// budget history); a `final = false` emission arm; and a twelve-way tamper
// arm.  The REAL-BLOCK arms of that row (Sturm membership on A^{(1)}(T_q) at
// (10, 691), the joint X_1 + X_4 decomposition) are E1b's and are still
// deliberately absent.
//
// assert() is the harness idiom of this repository (the assertion policy,
// EngineCheck.h: bare assert survives only in *Validation.cpp) -- but every
// check below is counted and reported, and the return value is the number of
// failures, so a Release build reports rather than aborts.

#include "HeckeExtAlgebraValidation.h"

#include "HeckeGenerationCertificates.h"
#include "HeckeLocalAlgebra.h"

#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/SubspaceOps.h"

#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using hecke::FpPoly;
using hecke::SplitMix64;
using hecke_ext::CommutantCertificate;
using hecke_ext::CommutantOutcome;
using hecke_ext::CommutantRoute;
using hecke_ext::CyclicityCertificate;
using hecke_ext::ExtMatrix;
using hecke_ext::LocalAlgebra;

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
// The monomial-algebra construction.
// ---------------------------------------------------------------------------
//
//   A = F_{ell^f} ⊗ F_ell[x_1..x_nv] / (the monomials NOT in `mons`)
//
// `mons` is the F_ell-basis of monomials, ordered by (total degree, then
// lexicographically), so mons[0] is the unit.  The F_ell-basis of A is
// { alpha^i * mons[m] : 0 <= i < f }, indexed as m*f + i, where alpha is a root
// of the irreducible `pi` of degree f.  Multiplication of monomials is the sum
// of exponent vectors, ZERO when the result is not in `mons` -- which is what
// makes the ideal monomial and every invariant combinatorial.
struct MonomialAlgebra {
    uint32_t ell = 0;
    uint32_t f = 1;
    FpPoly   pi;                                   // irreducible of degree f
    uint32_t nv = 1;
    std::vector<std::vector<uint32_t>> mons;       // exponent vectors
    std::vector<std::vector<uint32_t>> alpha_pow;  // alpha^k as f coefficients, k < 2f-1

    uint32_t dim() const { return static_cast<uint32_t>(mons.size()) * f; }

    void build_alpha_powers() {
        alpha_pow.clear();
        const uint32_t top = (f >= 1) ? (2u * f - 1u) : 1u;
        for (uint32_t k = 0; k < top; ++k) {
            const FpPoly red = FpPoly::mod(FpPoly::pow(FpPoly::X(ell), k), pi);
            std::vector<uint32_t> c(f, 0u);
            for (int64_t d = 0; d <= red.degree(); ++d) {
                c[static_cast<size_t>(d)] = red.coeffs[static_cast<size_t>(d)];
            }
            alpha_pow.push_back(std::move(c));
        }
    }

    int64_t monomial_index(const std::vector<uint32_t>& e) const {
        for (size_t i = 0; i < mons.size(); ++i) {
            if (mons[i] == e) return static_cast<int64_t>(i);
        }
        return -1;
    }

    // The matrix by which the element with the given coefficients acts on the
    // CYCLIC MODULE A / (the monomials outside `keep`) -- `keep` must be an
    // order ideal of `mons` (closed under division), so that its complement
    // spans an ideal of A and the quotient is a module with basis
    // { alpha^i * keep[m] }.  `keep == mons` is the regular representation.
    // `dual` returns the transpose, i.e. the action on the F_ell-DUAL module
    // (a.phi)(x) = phi(a x) -- WP-5's positive commutant case V = A^dual.
    ExtMatrix module_matrix(const std::vector<uint32_t>& coeffs,
                            const std::vector<std::vector<uint32_t>>& keep,
                            bool dual) const {
        const uint32_t d = static_cast<uint32_t>(keep.size()) * f;
        const uint32_t dA = dim();
        ExtMatrix M(d, d, ell);
        for (uint32_t b = 0; b < dA; ++b) {          // the element's basis term
            if (coeffs[b] == 0) continue;
            const uint32_t m0 = b / f, i0 = b % f;
            for (uint32_t col = 0; col < d; ++col) { // the argument's basis term
                const uint32_t m1 = col / f, i1 = col % f;
                std::vector<uint32_t> e = mons[m0];
                for (uint32_t v = 0; v < nv; ++v) e[v] += keep[m1][v];
                int64_t m2 = -1;
                for (size_t i = 0; i < keep.size(); ++i) {
                    if (keep[i] == e) { m2 = static_cast<int64_t>(i); break; }
                }
                if (m2 < 0) continue;                // the product is 0 in the module
                const std::vector<uint32_t>& ap = alpha_pow[i0 + i1];
                for (uint32_t r = 0; r < f; ++r) {
                    if (ap[r] == 0) continue;
                    const uint32_t add =
                        ModArithmetic::mod_mul(coeffs[b], ap[r], ell);
                    const uint32_t row = static_cast<uint32_t>(m2) * f + r;
                    M.data[row][col] = ModArithmetic::mod_add(M.data[row][col], add, ell);
                }
            }
        }
        if (!dual) return M;
        ExtMatrix Mt = M;
        return Mt.transpose();
    }

    // The matrix of multiplication by the element with the given coefficients
    // over the F_ell-basis of A.
    ExtMatrix multiplication_matrix(const std::vector<uint32_t>& coeffs) const {
        return module_matrix(coeffs, mons, false);
    }
};

// The two least r >= 2 with X^2 - r irreducible over F_ell, deterministically.
void TwoIrreducibleQuadratics(uint32_t ell, FpPoly& pi1, FpPoly& pi2) {
    std::vector<FpPoly> found;
    for (uint32_t r = 2; r < ell + 2 && found.size() < 2; ++r) {
        const uint32_t rr = r % ell;
        if (rr == 0) continue;
        const FpPoly cand(ell, {ModArithmetic::mod_sub(0u, rr, ell), 0u, 1u});
        if (hecke::CertifyIrreducible(cand) == 0) found.push_back(cand);
    }
    pi1 = found.at(0);
    pi2 = found.at(1);
}

ExtMatrix BlockDiagCopies(const ExtMatrix& M, uint32_t copies) {
    const uint32_t d = M.rows;
    ExtMatrix out(d * copies, d * copies, M.p);
    for (uint32_t c = 0; c < copies; ++c) {
        for (uint32_t i = 0; i < d; ++i) {
            for (uint32_t j = 0; j < d; ++j) out.data[c * d + i][c * d + j] = M.data[i][j];
        }
    }
    return out;
}

// A deterministic invertible D x D matrix over F_ell.
ExtMatrix SeededInvertible(uint32_t D, uint32_t ell, uint64_t seed) {
    SplitMix64 rng(seed);
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        ExtMatrix U(D, D, ell);
        for (uint32_t i = 0; i < D; ++i) {
            for (uint32_t j = 0; j < D; ++j) U.data[i][j] = rng.next_mod(ell);
        }
        if (U.is_invertible()) return U;
    }
    // Fall back to the identity rather than loop for ever (ell = 2, tiny D).
    return ExtMatrix(D, ell);
}

// ---------------------------------------------------------------------------
// One constructed case.
// ---------------------------------------------------------------------------

// One direct summand of the module V: the cyclic module A / (the monomials
// outside `keep`), optionally dualised.  An empty `keep` means the whole of
// `mons`, i.e. the regular representation A itself.
struct Summand {
    std::vector<std::vector<uint32_t>> keep;
    bool dual = false;
};

struct Case {
    std::string name;
    MonomialAlgebra A;
    std::vector<std::pair<uint64_t, std::vector<uint32_t>>> ops;  // (prime, coefficients)
    uint32_t copies = 1;          // V = A^copies, unless `summands` is non-empty
    std::vector<Summand> summands;  // WP-5: V = (+)_i A/I_i, some possibly dualised

    // The closed-form truth.
    uint32_t exp_D = 0, exp_f = 0, exp_dim_A = 0, exp_dim_m = 0, exp_nu = 0;
    uint32_t exp_dim_soc = 0, exp_embdim = 0, exp_min_gen = 0;
    bool exp_gorenstein = false, exp_cyclic = false;
    std::vector<uint32_t> exp_hilbert;
    std::vector<uint8_t>  exp_active;
    std::vector<uint64_t> exp_minimal_active;

    // WP-5: the closed-form commutant truth.  `exp_dim_C` is read off the
    // module decomposition of V (End_A(A) = A; End_A(A + A) = M_2(A);
    // End_A(A^dual) = A; End_A(A + A/m) is computed Hom-by-Hom), NEVER by
    // running the shipped route.  `exp_s_soc` is dim_F soc V in units of f.
    uint32_t exp_s_soc = 0, exp_dim_C = 0, exp_socle_bound = 0;
    bool exp_admits = false;
    CommutantOutcome exp_outcome = CommutantOutcome::NotNeeded;  // the UNFORCED route
};

std::vector<std::pair<uint64_t, ExtMatrix>> OperatorMatrices(const Case& c) {
    std::vector<std::pair<uint64_t, ExtMatrix>> out;
    for (const auto& op : c.ops) {
        if (c.summands.empty()) {
            out.emplace_back(op.first,
                             BlockDiagCopies(c.A.multiplication_matrix(op.second), c.copies));
            continue;
        }
        std::vector<ExtMatrix> parts;
        uint32_t d = 0;
        for (const Summand& sm : c.summands) {
            parts.push_back(c.A.module_matrix(op.second,
                                              sm.keep.empty() ? c.A.mons : sm.keep, sm.dual));
            d += parts.back().rows;
        }
        ExtMatrix M(d, d, c.A.ell);
        uint32_t off = 0;
        for (const ExtMatrix& P : parts) {
            for (uint32_t i = 0; i < P.rows; ++i) {
                for (uint32_t j = 0; j < P.cols; ++j) M.data[off + i][off + j] = P.data[i][j];
            }
            off += P.rows;
        }
        out.emplace_back(op.first, std::move(M));
    }
    return out;
}

// dim of the A_S-submodule generated by one column vector, computed here (not
// by the certificate) -- the WP's "cyclicity against an explicit A_S v closure".
uint32_t SpanOfModule(const std::vector<ExtMatrix>& a_basis,
                      const std::vector<uint32_t>& v, uint32_t D, uint32_t ell) {
    hecke::RrefRowSpan span(D, ell);
    for (const ExtMatrix& b : a_basis) {
        std::vector<uint32_t> w(D, 0u);
        for (uint32_t i = 0; i < D; ++i) {
            uint32_t acc = 0;
            for (uint32_t j = 0; j < D; ++j) {
                acc = ModArithmetic::mod_add(
                    acc, ModArithmetic::mod_mul(b.data[i][j], v[j], ell), ell);
            }
            w[i] = acc;
        }
        span.insert(std::move(w));
    }
    return span.dim();
}

// Builds the decomposition and runs Stage 3 + 4(a).  Returns false (and
// reports) when the family does not decompose into exactly one block.
bool RunCase(const Case& c, bool conjugate, LocalAlgebra& out,
             std::vector<std::pair<uint64_t, ExtMatrix>>& ops_out) {
    const uint32_t ell = c.A.ell;
    ops_out = OperatorMatrices(c);
    const uint32_t D = ops_out[0].second.rows;
    if (conjugate) {
        // A U that happens to lie in F_ell[t_p] centralises the family and the
        // arm would compare the case with itself (it does, at ell = 5, D = 2,
        // with probability 1/(ell^2 - ell)).  So the seeded search advances
        // until the conjugation actually moves a matrix; the sequence of seeds
        // is fixed, so the chosen U is still a deterministic function of
        // (D, ell).
        bool moved = false;
        for (uint32_t attempt = 0; attempt < 32u && !moved; ++attempt) {
            const ExtMatrix U =
                SeededInvertible(D, ell, 0x51A7B10C4A1E0001ull + 1000u * attempt + D);
            if (U == ExtMatrix(D, ell)) continue;   // degenerated to the identity
            ExtMatrix Uinv = U;
            Uinv = Uinv.inverse();
            std::vector<std::pair<uint64_t, ExtMatrix>> candidate = ops_out;
            for (auto& op : candidate) op.second = Uinv * (op.second * U);
            for (size_t j = 0; j < candidate.size() && !moved; ++j) {
                if (!(candidate[j].second == ops_out[j].second)) moved = true;
            }
            if (moved) ops_out = std::move(candidate);
        }
        Check(moved, c.name + ": a non-centralising conjugating U was found");
        if (!moved) return false;
    }
    hecke::HeckeModuleDecomposition dec(ops_out, ell);
    dec.compute();
    if (dec.blocks().size() != 1u) {
        Check(false, c.name + ": expected one primary block, got " +
                         std::to_string(dec.blocks().size()));
        return false;
    }
    out = hecke_ext::ComputeLocalAlgebra(dec.blocks()[0], ops_out, ell);
    return true;
}

void CheckCase(const Case& c, bool conjugate) {
    const std::string tag =
        c.name + " (ell=" + std::to_string(c.A.ell) + (conjugate ? ", conjugated)" : ")");
    LocalAlgebra L;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    if (!RunCase(c, conjugate, L, ops)) return;

    Check(L.ok, tag + ": block certified local");
    if (!L.ok) return;

    const uint32_t ell = c.A.ell;
    CheckEq(L.D, c.exp_D, tag + ": D");
    CheckEq(L.f, c.exp_f, tag + ": f");
    CheckEq(L.dim_A, c.exp_dim_A, tag + ": dim A_S");
    CheckEq(L.dim_m_S, c.exp_dim_m, tag + ": dim m_S");
    CheckEq(L.nu, c.exp_nu, tag + ": nu");
    Check(L.hilbert == c.exp_hilbert, tag + ": Hilbert function");
    CheckEq(L.dim_socle, c.exp_dim_soc, tag + ": dim soc A_S");
    CheckEq(L.gorenstein, c.exp_gorenstein, tag + ": gorenstein");
    CheckEq(L.embdim, c.exp_embdim, tag + ": embdim (K-span test)");
    Check(L.active == c.exp_active, tag + ": tangent table `active` over K");
    Check(L.minimal_active == c.exp_minimal_active, tag + ": minimal active subset");
    CheckEq(L.min_generators, c.exp_min_gen, tag + ": min_generators");
    CheckEq(L.cyclicity.cyclic, c.exp_cyclic, tag + ": certificate (a) verdict");
    CheckEq(L.cyclicity.dim_quotient, c.exp_min_gen * c.exp_f, tag + ": dim V/m_S V");

    // Structural invariants that hold independently of the closed form.
    CheckEq(L.dim_A - L.dim_m_S, L.f, tag + ": A_S/m_S is the residue field");
    CheckEq(static_cast<uint32_t>(L.hilbert.size()), L.nu, tag + ": Hilbert length == nu");
    {
        uint32_t sum = 0;
        for (uint32_t h : L.hilbert) sum += h;
        CheckEq(sum * L.f, L.dim_A, tag + ": the Hilbert layers sum to dim A_S");
    }
    Check(hecke::EvalPolyAtMatrix(L.witness_poly, L.w_hat).is_zero(),
          tag + ": pi_w(w_hat) == 0");
    for (size_t j = 0; j < L.n.size(); ++j) {
        Check(hecke::IsNilpotent(L.n[j], L.D), tag + ": n_p nilpotent");
        Check(L.character[j].degree() < static_cast<int64_t>(L.f), tag + ": deg P_p < f");
    }
    // Every socle element annihilates every generator of m_S.
    for (const ExtMatrix& s : L.socle) {
        for (const ExtMatrix& n_p : L.n) {
            Check((s * n_p).is_zero(), tag + ": soc A_S annihilates m_S");
        }
    }
    // Each stored A_S basis element commutes with every t_p (A_S is commutative).
    for (const ExtMatrix& b : L.basis) {
        for (const auto& t : L.t) Check((b * t) == (t * b), tag + ": A_S centralises t_p");
    }

    // --- Certificate (a): verify() on the emitted data. ------------------
    CheckEq(L.cyclicity.verify(), 0u, tag + ": CyclicityCertificate::verify()");

    // --- Cyclicity against an EXPLICIT A_S v closure (not verify()'s). ----
    if (L.cyclicity.cyclic) {
        std::vector<uint32_t> v(L.D, 0u);
        v[L.cyclicity.generator_index] = 1u;
        CheckEq(SpanOfModule(L.basis, v, L.D, ell), L.D, tag + ": A_S v == V on the generator");
    } else {
        bool any = false;
        for (uint32_t j = 0; j < L.D && !any; ++j) {
            std::vector<uint32_t> v(L.D, 0u);
            v[j] = 1u;
            if (SpanOfModule(L.basis, v, L.D, ell) == L.D) any = true;
        }
        SplitMix64 rng(0x5EED0A15E0000001ull);
        for (uint32_t s = 0; s < 64 && !any; ++s) {
            std::vector<uint32_t> v(L.D, 0u);
            for (uint32_t i = 0; i < L.D; ++i) v[i] = rng.next_mod(ell);
            if (SpanOfModule(L.basis, v, L.D, ell) == L.D) any = true;
        }
        Check(!any, tag + ": no vector generates V (the block is genuinely non-cyclic)");
    }

    // --- Determinism: the same inputs give the same JSON. -----------------
    {
        LocalAlgebra again;
        std::vector<std::pair<uint64_t, ExtMatrix>> ops2;
        if (RunCase(c, conjugate, again, ops2)) {
            Check(again.to_json() == L.to_json(), tag + ": to_json() is deterministic");
        }
    }
}

// ---------------------------------------------------------------------------
// The case table.
// ---------------------------------------------------------------------------

std::vector<Case> BuildCases(uint32_t ell) {
    FpPoly pi1(ell), pi2(ell);
    TwoIrreducibleQuadratics(ell, pi1, pi2);
    const FpPoly lin(ell, {0u, 1u});   // X, the degree-1 "irreducible" for f = 1

    const uint32_t a = 3u % ell, b = 5u % ell;
    std::vector<Case> cases;

    auto make_alg = [&](uint32_t f, const FpPoly& pi, uint32_t nv,
                        std::vector<std::vector<uint32_t>> mons) {
        MonomialAlgebra A;
        A.ell = ell;
        A.f = f;
        A.pi = pi;
        A.nv = nv;
        A.mons = std::move(mons);
        A.build_alpha_powers();
        return A;
    };

    // (1) F[eps]/eps^2 -- embdim 1, Gorenstein, nu = 2.
    {
        Case c;
        c.name = "F[eps]/eps^2";
        c.A = make_alg(1, lin, 1, {{0}, {1}});
        c.ops = {{2, {a, 1}}};
        c.exp_D = 2; c.exp_f = 1; c.exp_dim_A = 2; c.exp_dim_m = 1; c.exp_nu = 2;
        c.exp_hilbert = {1, 1}; c.exp_dim_soc = 1; c.exp_gorenstein = true;
        c.exp_embdim = 1; c.exp_min_gen = 1; c.exp_cyclic = true;
        c.exp_active = {1}; c.exp_minimal_active = {2};
        c.exp_s_soc = 1; c.exp_dim_C = 2;   // End_A(A) = A
        c.exp_socle_bound = 2; c.exp_admits = true;
        c.exp_outcome = CommutantOutcome::NotNeeded;
        cases.push_back(c);
    }
    // (2) F[x,y]/(x,y)^2 -- embdim 2, socle dim 2, NOT Gorenstein.
    {
        Case c;
        c.name = "F[x,y]/(x,y)^2";
        c.A = make_alg(1, lin, 2, {{0, 0}, {1, 0}, {0, 1}});
        c.ops = {{2, {a, 1, 0}}, {3, {b, 0, 1}}};
        c.exp_D = 3; c.exp_f = 1; c.exp_dim_A = 3; c.exp_dim_m = 2; c.exp_nu = 2;
        c.exp_hilbert = {1, 2}; c.exp_dim_soc = 2; c.exp_gorenstein = false;
        c.exp_embdim = 2; c.exp_min_gen = 1; c.exp_cyclic = true;
        c.exp_active = {1, 1}; c.exp_minimal_active = {2, 3};
        c.exp_s_soc = 2; c.exp_dim_C = 3;   // End_A(A) = A
        c.exp_socle_bound = 3; c.exp_admits = true;
        c.exp_outcome = CommutantOutcome::NotNeeded;
        cases.push_back(c);
    }
    // (3) F[x]/(x^3) -- nu = 3.
    {
        Case c;
        c.name = "F[x]/(x^3)";
        c.A = make_alg(1, lin, 1, {{0}, {1}, {2}});
        c.ops = {{2, {a, 1, 0}}};
        c.exp_D = 3; c.exp_f = 1; c.exp_dim_A = 3; c.exp_dim_m = 2; c.exp_nu = 3;
        c.exp_hilbert = {1, 1, 1}; c.exp_dim_soc = 1; c.exp_gorenstein = true;
        c.exp_embdim = 1; c.exp_min_gen = 1; c.exp_cyclic = true;
        c.exp_active = {1}; c.exp_minimal_active = {2};
        c.exp_s_soc = 1; c.exp_dim_C = 3;   // End_A(A) = A
        c.exp_socle_bound = 3; c.exp_admits = true;
        c.exp_outcome = CommutantOutcome::NotNeeded;
        cases.push_back(c);
    }
    // (4) F_{ell^2}[eps]/eps^2, ONE operator t = alpha + eps.  THE f = 2 TRAP:
    // embdim must be 1 by the K-span test, and the witness's own direction must
    // survive ([E] cl:resfield, [XA B2]).
    {
        Case c;
        c.name = "F_{ell^2}[eps]/eps^2 (t = alpha + eps)";
        c.A = make_alg(2, pi1, 1, {{0}, {1}});
        c.ops = {{2, {0, 1, 1, 0}}};
        c.exp_D = 4; c.exp_f = 2; c.exp_dim_A = 4; c.exp_dim_m = 2; c.exp_nu = 2;
        c.exp_hilbert = {1, 1}; c.exp_dim_soc = 2; c.exp_gorenstein = true;
        c.exp_embdim = 1; c.exp_min_gen = 1; c.exp_cyclic = true;
        c.exp_active = {1}; c.exp_minimal_active = {2};
        c.exp_s_soc = 1; c.exp_dim_C = 4;   // End_A(A) = A, dim_{F_ell} = 4
        c.exp_socle_bound = 4; c.exp_admits = true;
        c.exp_outcome = CommutantOutcome::NotNeeded;
        cases.push_back(c);
    }
    // (5) F_{ell^2}[x,y]/(x,y)^2 -- f = 2, embdim 2.
    {
        Case c;
        c.name = "F_{ell^2}[x,y]/(x,y)^2";
        c.A = make_alg(2, pi1, 2, {{0, 0}, {1, 0}, {0, 1}});
        c.ops = {{2, {0, 1, 1, 0, 0, 0}}, {3, {0, 0, 0, 0, 1, 0}}};
        c.exp_D = 6; c.exp_f = 2; c.exp_dim_A = 6; c.exp_dim_m = 4; c.exp_nu = 2;
        c.exp_hilbert = {1, 2}; c.exp_dim_soc = 4; c.exp_gorenstein = false;
        c.exp_embdim = 2; c.exp_min_gen = 1; c.exp_cyclic = true;
        c.exp_active = {1, 1}; c.exp_minimal_active = {2, 3};
        c.exp_s_soc = 2; c.exp_dim_C = 6;   // End_A(A) = A, dim_{F_ell} = 6
        c.exp_socle_bound = 6; c.exp_admits = true;
        c.exp_outcome = CommutantOutcome::NotNeeded;
        cases.push_back(c);
    }
    // (6) V = A + A over A = F[x]/(x^3): certificate (a) must FAIL (multiplicity
    // 2), with min_generators = 2 -- the shape [E] cert:cyclic calls out.
    {
        Case c;
        c.name = "F[x]/(x^3) acting on A + A (non-cyclic)";
        c.A = make_alg(1, lin, 1, {{0}, {1}, {2}});
        c.ops = {{2, {a, 1, 0}}};
        c.copies = 2;
        c.exp_D = 6; c.exp_f = 1; c.exp_dim_A = 3; c.exp_dim_m = 2; c.exp_nu = 3;
        c.exp_hilbert = {1, 1, 1}; c.exp_dim_soc = 1; c.exp_gorenstein = true;
        c.exp_embdim = 1; c.exp_min_gen = 2; c.exp_cyclic = false;
        c.exp_active = {1}; c.exp_minimal_active = {2};
        c.exp_s_soc = 2; c.exp_dim_C = 12;   // End_A(A + A) = M_2(A): 4 * 3 = 12
        c.exp_socle_bound = 6; c.exp_admits = false;
        c.exp_outcome = CommutantOutcome::ExcludedBySocleBound;
        cases.push_back(c);
    }
    // (7) A = F_{ell^2} itself: m_S = 0, nu = 1, embdim 0 -- the degenerate end
    // of the filtration, and the case in which no n_p is active.
    {
        Case c;
        c.name = "F_{ell^2} (a field; m_S = 0)";
        c.A = make_alg(2, pi1, 1, {{0}});
        c.ops = {{2, {0, 1}}};
        c.exp_D = 2; c.exp_f = 2; c.exp_dim_A = 2; c.exp_dim_m = 0; c.exp_nu = 1;
        c.exp_hilbert = {1}; c.exp_dim_soc = 2; c.exp_gorenstein = true;
        c.exp_embdim = 0; c.exp_min_gen = 1; c.exp_cyclic = true;
        c.exp_active = {0}; c.exp_minimal_active = {};
        c.exp_s_soc = 1; c.exp_dim_C = 2;   // End_F(F) = F, dim_{F_ell} = 2
        c.exp_socle_bound = 2; c.exp_admits = true;
        c.exp_outcome = CommutantOutcome::NotNeeded;
        cases.push_back(c);
    }
    // (8) V = A^dual for A = F[x,y]/(x,y)^2 -- WP-5's POSITIVE commutant case
    // ([X] Sec. 2.2 Stage 4(b) step 1): certificate (a) FAILS (t = 2), the
    // socle bound ADMITS (b) (f*t*s = 2 <= dim soc A_S = 2), and (b) PASSES,
    // because End_{A}(A^dual) = A for a local Artinian A (A^dual is the
    // injective hull of the residue field).  This is the only shape on which
    // (b) can certify where (a) cannot.
    {
        Case c;
        c.name = "F[x,y]/(x,y)^2 acting on A^dual (commutant passes, cyclic fails)";
        c.A = make_alg(1, lin, 2, {{0, 0}, {1, 0}, {0, 1}});
        c.ops = {{2, {a, 1, 0}}, {3, {b, 0, 1}}};
        c.summands = { Summand{ {}, true } };
        c.exp_D = 3; c.exp_f = 1; c.exp_dim_A = 3; c.exp_dim_m = 2; c.exp_nu = 2;
        c.exp_hilbert = {1, 2}; c.exp_dim_soc = 2; c.exp_gorenstein = false;
        c.exp_embdim = 2; c.exp_min_gen = 2; c.exp_cyclic = false;
        c.exp_active = {1, 1}; c.exp_minimal_active = {2, 3};
        c.exp_s_soc = 1; c.exp_dim_C = 3;   // End_A(A^dual) = A
        c.exp_socle_bound = 3; c.exp_admits = true;
        c.exp_outcome = CommutantOutcome::Passed;
        cases.push_back(c);
    }
    // (9) V = A + A/m for A = F[eps]/eps^2 -- THE (10, 691) SHAPE of [XA B1]:
    // the unsplit Eisenstein-cuspidal block.  A_S is Gorenstein (dim soc = 1)
    // and V is not cyclic (t = 2), so the socle bound EXCLUDES (b) with
    // f*t*s = 4 > 1, and the bound dim C >= 2 + 4 - 1 = 5 is EXACT:
    // End_A(A + A/m) = Hom(A,A) + Hom(A,A/m) + Hom(A/m,A) + Hom(A/m,A/m)
    //               = 2 + 1 + 1 + 1 = 5.
    {
        Case c;
        c.name = "F[eps]/eps^2 acting on A + A/m (the (10,691) shape)";
        c.A = make_alg(1, lin, 1, {{0}, {1}});
        c.ops = {{2, {a, 1}}};
        c.summands = { Summand{ {}, false }, Summand{ {{0}}, false } };
        c.exp_D = 3; c.exp_f = 1; c.exp_dim_A = 2; c.exp_dim_m = 1; c.exp_nu = 2;
        c.exp_hilbert = {1, 1}; c.exp_dim_soc = 1; c.exp_gorenstein = true;
        c.exp_embdim = 1; c.exp_min_gen = 2; c.exp_cyclic = false;
        c.exp_active = {1}; c.exp_minimal_active = {2};
        c.exp_s_soc = 2; c.exp_dim_C = 5;   // 2 + 1 + 1 + 1, the exact bound
        c.exp_socle_bound = 5; c.exp_admits = false;
        c.exp_outcome = CommutantOutcome::ExcludedBySocleBound;
        cases.push_back(c);
    }
    (void)pi2;
    return cases;
}

// ---------------------------------------------------------------------------
// The [XA B2] negative control: P_p(w) in place of P_p(w_hat).
// ---------------------------------------------------------------------------

void CheckWitnessDirectionTrap(uint32_t ell) {
    const std::string tag = "[XA B2] P(w) trap (ell=" + std::to_string(ell) + ")";
    const std::vector<Case> cases = BuildCases(ell);
    const Case& c = cases[3];   // F_{ell^2}[eps]/eps^2, t = alpha + eps
    LocalAlgebra L;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    if (!RunCase(c, false, L, ops)) return;
    if (!L.ok) { Check(false, tag + ": block certified local"); return; }

    // The correct generator n = t - P(w_hat) is nonzero and m_S has dimension f.
    Check(!L.n[0].is_zero(), tag + ": n_t = t - P(w_hat) is NOT zero");
    CheckEq(L.embdim, 1u, tag + ": embdim over K is 1");

    // The v1.0 recipe: P(w) with w the witness matrix itself.  Here P = X and
    // w = t, so t - P(w) = 0 and the whole tangent direction disappears.
    const ExtMatrix w = L.locality.witness_matrix;
    Check(w.rows == L.D, tag + ": the verdict carries the witness matrix");
    const ExtMatrix n_bad = L.t[0] - hecke::EvalPolyAtMatrix(L.character[0], w);
    Check(n_bad.is_zero(), tag + ": t - P(w) IS zero -- the direction is lost");

    // And an F_ell-span rather than a K-span would report embdim 2, because the
    // one K-direction is two F_ell-directions.
    CheckEq(L.dim_m_S, 2u, tag + ": dim_{F_ell} m_S/m_S^2 = f * embdim = 2");
}

// ---------------------------------------------------------------------------
// The cross-block evaluation test ([XA M5], [X] Sec. 2.2 Stage 2).
// ---------------------------------------------------------------------------

LocalAlgebra BuildOne(const Case& c, std::vector<std::pair<uint64_t, ExtMatrix>>& ops) {
    LocalAlgebra L;
    RunCase(c, false, L, ops);
    return L;
}

void CheckCrossBlock(uint32_t ell) {
    const std::string tag = "[XA M5] cross-block (ell=" + std::to_string(ell) + ")";
    FpPoly pi1(ell), pi2(ell);
    TwoIrreducibleQuadratics(ell, pi1, pi2);

    auto field_case = [&](const FpPoly& pi, uint32_t power) {
        // A = F_{ell^2}, the operator is multiplication by alpha^{ell^power},
        // i.e. the Frobenius conjugate of alpha when power == 1.
        MonomialAlgebra A;
        A.ell = ell; A.f = 2; A.pi = pi; A.nv = 1; A.mons = {{0}};
        A.build_alpha_powers();
        FpPoly e = FpPoly::X(ell);
        for (uint32_t s = 0; s < power; ++s) e = FpPoly::powmod(e, ell, pi);
        std::vector<uint32_t> coeffs(2, 0u);
        for (int64_t d = 0; d <= e.degree(); ++d) {
            coeffs[static_cast<size_t>(d)] = e.coeffs[static_cast<size_t>(d)];
        }
        Case c;
        c.name = "F_{ell^2} (field)";
        c.A = A;
        c.ops = {{2, coeffs}};
        return c;
    };
    auto dual_case = [&](const FpPoly& pi) {
        MonomialAlgebra A;
        A.ell = ell; A.f = 2; A.pi = pi; A.nv = 1; A.mons = {{0}, {1}};
        A.build_alpha_powers();
        Case c;
        c.name = "F_{ell^2}[eps]/eps^2";
        c.A = A;
        c.ops = {{2, {0, 1, 1, 0}}};
        return c;
    };

    std::vector<std::pair<uint64_t, ExtMatrix>> ops_B, ops_conj, ops_dual, ops_other;
    const LocalAlgebra B     = BuildOne(field_case(pi1, 0), ops_B);      // D = 2
    const LocalAlgebra Bconj = BuildOne(field_case(pi1, 1), ops_conj);   // D = 2, Frobenius
    const LocalAlgebra Bdual = BuildOne(dual_case(pi1), ops_dual);       // D = 4, same ideal
    const LocalAlgebra Bother = BuildOne(field_case(pi2, 0), ops_other); // D = 2, other ideal
    if (!B.ok || !Bconj.ok || !Bdual.ok || !Bother.ok) {
        Check(false, tag + ": all four constructed blocks certify local");
        return;
    }

    // Itself.
    Check(hecke_ext::SameMaximalIdeal(B, B.t).same_ideal, tag + ": B matches itself");
    // The Frobenius conjugate: the SAME maximal ideal of T_S, must match.
    Check(hecke_ext::SameMaximalIdeal(B, Bconj.t).same_ideal,
          tag + ": a Frobenius-conjugate character matches");
    Check(hecke_ext::SameMaximalIdeal(Bconj, B.t).same_ideal,
          tag + ": the match is symmetric on the conjugate pair");
    // Same ideal, DIFFERENT dimension (2 vs 4) -- the case v1.1's "first witness
    // in both" could not even state, the candidate sequence being seeded by the
    // block dimension.
    CheckEq(B.D, 2u, tag + ": dim B = 2");
    CheckEq(Bdual.D, 4u, tag + ": dim B' = 4");
    {
        const hecke_ext::EvaluationMatch m = hecke_ext::SameMaximalIdeal(B, Bdual.t);
        Check(m.same_ideal, tag + ": same ideal across dim 2 vs dim 4");
        // N' is taken for the TESTED block ([XA M5]'s correction).
        uint32_t n_expect = 0; uint64_t acc = 1;
        while (acc < 4u) { acc *= ell; ++n_expect; }
        n_expect = ((n_expect + 1u) / 2u) * 2u;
        CheckEq(m.N_other, n_expect, tag + ": N' is computed for B', not for B");
    }
    Check(hecke_ext::SameMaximalIdeal(Bdual, B.t).same_ideal,
          tag + ": and the other way round");
    // A DIFFERENT character must not match, in either direction.
    {
        const hecke_ext::EvaluationMatch m = hecke_ext::SameMaximalIdeal(B, Bother.t);
        Check(!m.same_ideal, tag + ": a distinct character does NOT match");
        CheckEq(m.failure, static_cast<uint32_t>(hecke_ext::kEvalWitnessNotNilpotent),
                tag + ": the rejection is pi_w(w') not nilpotent");
    }
    Check(!hecke_ext::SameMaximalIdeal(Bother, B.t).same_ideal,
          tag + ": non-matching is symmetric");
    Check(!hecke_ext::SameMaximalIdeal(Bdual, Bother.t).same_ideal,
          tag + ": distinct characters at different dimensions do not match");
}

// ---------------------------------------------------------------------------
// The tamper arm: a certificate that reads instead of re-deriving would pass.
// ---------------------------------------------------------------------------

void CheckTampering(uint32_t ell) {
    const std::string tag = "tamper (ell=" + std::to_string(ell) + ")";
    const std::vector<Case> cases = BuildCases(ell);

    // A cyclic case (2) and the non-cyclic case (6).
    LocalAlgebra Lc, Ln;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops_c, ops_n;
    if (!RunCase(cases[1], false, Lc, ops_c) || !RunCase(cases[5], false, Ln, ops_n)) return;
    if (!Lc.ok || !Ln.ok) { Check(false, tag + ": both blocks certify local"); return; }
    CheckEq(Lc.cyclicity.verify(), 0u, tag + ": the untampered cyclic certificate verifies");
    CheckEq(Ln.cyclicity.verify(), 0u, tag + ": the untampered non-cyclic certificate verifies");

    const auto expect = [&](CyclicityCertificate c, uint32_t code, const std::string& what) {
        CheckEq(c.verify(), code, tag + ": " + what);
    };

    { auto c = Lc.cyclicity; c.dim_quotient += 1; expect(c, 10, "dim_quotient incremented"); }
    { auto c = Lc.cyclicity; c.a_basis.pop_back(); expect(c, 7, "A_S basis truncated"); }
    { auto c = Lc.cyclicity; c.t[0].data[0][0] =
          ModArithmetic::mod_add(c.t[0].data[0][0], 1u, ell);
      expect(c, 7, "an operator entry perturbed"); }
    { auto c = Lc.cyclicity; c.cyclic = false; expect(c, 12, "the cyclic verdict flipped off"); }
    { auto c = Lc.cyclicity; c.generator_index = (c.generator_index + 1u) % c.D;
      expect(c, 13, "the generator index moved"); }
    { auto c = Lc.cyclicity; c.m_basis.pop_back(); expect(c, 9, "m_S basis truncated"); }
    { auto c = Lc.cyclicity; c.w_hat.data[0][0] =
          ModArithmetic::mod_add(c.w_hat.data[0][0], 1u, ell);
      expect(c, 5, "w_hat perturbed"); }
    { auto c = Lc.cyclicity;
      std::vector<uint32_t> co(static_cast<size_t>(c.f) + 1u, 0u);
      co.back() = 1u;
      c.character[0] = FpPoly(ell, co);
      expect(c, 6, "a character of degree >= f"); }
    { auto c = Ln.cyclicity; c.cyclic = true; expect(c, 12, "a non-cyclic block claimed cyclic"); }
    { auto c = Lc.cyclicity; c.f += 1; expect(c, 1, "f no longer divides D"); }
}


// ---------------------------------------------------------------------------
// WP-5 (E1a-2): the commutant (b), the socle bound and `CommutantState`.
// ---------------------------------------------------------------------------

// The validation's OWN naive commutant dimension, assembled DIFFERENTLY from
// the shipped `NaiveCommutantDim`: the unknown X_{r,c} sits at column c*D + r
// (column-major, where the shipped oracle uses row-major) and the rows run
// "commutator entry (i, j) row-major, then operator" (where the shipped one
// runs "operator, then entry").  A nullity does not depend on either order, so
// the two must agree -- and neither is read by the structured route.
uint32_t IndependentCommutantDim(const std::vector<ExtMatrix>& t, uint32_t D, uint32_t ell) {
    const uint32_t k = static_cast<uint32_t>(t.size());
    ExtMatrix sys(D * D * k, D * D, ell);
    for (uint32_t i = 0; i < D; ++i) {
        for (uint32_t j = 0; j < D; ++j) {
            for (uint32_t p = 0; p < k; ++p) {
                const uint32_t row = (i * D + j) * k + p;
                for (uint32_t c = 0; c < D; ++c) {    // + X_{i,c} (t_p)_{c,j}
                    const uint32_t col = c * D + i;
                    sys.data[row][col] =
                        ModArithmetic::mod_add(sys.data[row][col], t[p].data[c][j], ell);
                }
                for (uint32_t r = 0; r < D; ++r) {    // - (t_p)_{i,r} X_{r,j}
                    const uint32_t col = j * D + r;
                    sys.data[row][col] =
                        ModArithmetic::mod_sub(sys.data[row][col], t[p].data[i][r], ell);
                }
            }
        }
    }
    return hecke::KernelAsMatrix(sys).cols;
}

// The invariant factors of `a`, re-derived from the NULLITY SEQUENCE of
// pi(a)^j: nullity(pi(a)^j) = deg(pi) * sum_i min(s_i, j).  This is an
// independent route to the RCF data the certificate stores.
bool InvariantSizesAgree(const ExtMatrix& a, const FpPoly& pi,
                         const std::vector<uint32_t>& sizes) {
    const uint32_t D = a.rows;
    const uint32_t deg = static_cast<uint32_t>(pi.degree());
    const ExtMatrix Npi = hecke::EvalPolyAtMatrix(pi, a);
    ExtMatrix acc(D, a.p);                       // identity
    uint32_t top = 0;
    for (uint32_t s : sizes) top = (s > top) ? s : top;
    for (uint32_t j = 1; j <= top + 1; ++j) {
        acc = acc * Npi;
        ExtMatrix copy = acc;
        const uint32_t nullity = D - copy.rank();
        uint64_t want = 0;
        for (uint32_t s : sizes) want += (s < j) ? s : j;
        if (nullity != want * deg) return false;
    }
    return true;
}

// The certificate's own fields, compared field by field -- everything except
// the budget history (R3.1 item 3 vs item 4).
bool SameCertificateModuloHistory(const CommutantCertificate& x,
                                  const CommutantCertificate& y) {
    if (x.ell != y.ell || x.D != y.D || x.f != y.f) return false;
    if (x.dim_A != y.dim_A || x.t_top != y.t_top || x.s_soc != y.s_soc) return false;
    if (x.dim_soc_A != y.dim_soc_A || x.socle_bound != y.socle_bound) return false;
    if (x.admits != y.admits || x.forced != y.forced || x.outcome != y.outcome) return false;
    if (x.candidate_index != y.candidate_index) return false;
    if (!(x.a == y.a) || !(x.R == y.R)) return false;
    if (!(x.pi_a == y.pi_a) || x.inv_sizes != y.inv_sizes) return false;
    if (x.u != y.u || x.rank != y.rank || x.dim_C != y.dim_C) return false;
    return x.state.equals(y.state, true);
}

void CheckCommutantCase(const Case& c, bool conjugate) {
    const std::string tag = "(b) " + c.name + " (ell=" + std::to_string(c.A.ell) +
                            (conjugate ? ", conjugated)" : ")");
    LocalAlgebra L;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    if (!RunCase(c, conjugate, L, ops)) return;
    if (!L.ok) { Check(false, tag + ": block certified local"); return; }
    const uint32_t ell = c.A.ell;

    // --- dim C three ways: the closed form, our own naive, the shipped one.
    const uint32_t mine = IndependentCommutantDim(L.t, L.D, ell);
    CheckEq(mine, c.exp_dim_C, tag + ": naive dim C against the closed form");
    CheckEq(hecke_ext::NaiveCommutantDim(L.t, L.D, ell), mine,
            tag + ": the shipped naive oracle == the independent one");

    // --- the socle bound, on EVERY case ([X] Sec. 5's row). --------------
    CommutantRoute route(L);
    const CommutantOutcome out = route.run();
    CheckEq(route.state().t_top, c.exp_min_gen, tag + ": socle-bound t = dim_F V/m_S V");
    CheckEq(route.state().s_soc, c.exp_s_soc, tag + ": socle-bound s = dim_F soc V");
    CheckEq(route.state().dim_soc_A, c.exp_dim_soc, tag + ": socle-bound dim_F soc A_S");
    CheckEq(route.socle_bound(), c.exp_socle_bound, tag + ": the socle bound value");
    CheckEq(route.admits(), c.exp_admits, tag + ": the socle-bound gate f*t*s <= dim soc A_S");
    Check(mine >= route.socle_bound(), tag + ": dim C >= the socle bound");
    CheckEq(static_cast<uint32_t>(out), static_cast<uint32_t>(c.exp_outcome),
            tag + ": route (b) outcome");
    CheckEq(route.certificate().verify(), 0u, tag + ": verify() on the unforced certificate");
    if (out == CommutantOutcome::Passed || out == CommutantOutcome::Failed) {
        CheckEq(route.dim_C(), c.exp_dim_C, tag + ": structured dim C (unforced)");
    }

    // --- the FORCED structured route: "structured == naive wherever both
    // run" is checked on every case, gates included. ---------------------
    CommutantRoute forced(L, CommutantRoute::kCommutantBudget, true);
    const CommutantOutcome fo = forced.run();
    CheckEq(forced.dim_C(), c.exp_dim_C, tag + ": structured dim C == the closed form");
    CheckEq(forced.dim_C(), mine, tag + ": structured dim C == the naive dim C");
    CheckEq(static_cast<uint32_t>(fo),
            static_cast<uint32_t>(c.exp_dim_C == L.dim_A ? CommutantOutcome::Passed
                                                         : CommutantOutcome::Failed),
            tag + ": forced outcome follows dim C == dim A_S");
    const CommutantCertificate fcert = forced.certificate();
    CheckEq(fcert.verify(), 0u, tag + ": verify() on the forced certificate");

    // The RCF data, re-derived from the nullity sequence of pi_a(a)^j.
    Check(InvariantSizesAgree(fcert.a, fcert.pi_a, fcert.inv_sizes),
          tag + ": the invariant factors match the nullity sequence of pi_a(a)^j");
    {
        uint64_t want_u = 0;
        for (uint32_t si : fcert.inv_sizes) {
            for (uint32_t sj : fcert.inv_sizes) want_u += (si < sj) ? si : sj;
        }
        want_u *= static_cast<uint64_t>(fcert.pi_a.degree());
        CheckEq(static_cast<uint64_t>(fcert.u), want_u,
                tag + ": u = deg(pi_a) * sum_{i,j} min(s_i, s_j)");
        Check(fcert.u >= fcert.dim_C, tag + ": C lies inside End_{F_ell[a]}(V)");
    }

    // Determinism: the same inputs give byte-identical JSON.
    {
        CommutantRoute again(L, CommutantRoute::kCommutantBudget, true);
        again.run();
        Check(again.certificate().to_json() == fcert.to_json(),
              tag + ": CommutantCertificate::to_json() is deterministic");
    }
}

// The R3 arms ([X] Sec. 1.3.1 R3.1 items 2-4, R3.3 item 3): the budget is
// interrupted at EVERY reachable row count, resumed, and the result compared
// entrywise with a fresh full run.
void CheckCommutantResume(const Case& c, bool forced, const std::string& label) {
    const std::string tag = "R3 " + label + " (ell=" + std::to_string(c.A.ell) + ")";
    LocalAlgebra L;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    if (!RunCase(c, false, L, ops)) return;
    if (!L.ok) { Check(false, tag + ": block certified local"); return; }

    const uint64_t BIG = CommutantRoute::kCommutantBudget;
    CommutantRoute full(L, BIG, forced);
    full.run();
    const hecke_ext::CommutantState& fs = full.state();
    Check(fs.prepared, tag + ": the full run prepared the structured system");
    if (!fs.prepared) return;
    Check(fs.outcome != CommutantOutcome::OverBudget, tag + ": the full run completed");

    // The cost model of the header, re-derived here -- so a change to it is a
    // change to this validation too.
    const uint64_t D = fs.D, k = fs.k, u = fs.u;
    const uint64_t cost_formation = 2ull * k * u * D * D * D;
    const uint64_t cost_row = u * u;
    CheckEq(fs.total_rows, k * D * D, tag + ": total_rows = k * D^2");
    CheckEq(fs.modmul_spent, cost_formation + fs.total_rows * cost_row,
            tag + ": modmul_spent = formation + rows * row cost");
    Check(fs.budget_history == std::vector<uint64_t>{BIG}, tag + ": budget_history of a fresh run");

    // (i) Below the formation cost: nothing is formed, nothing is spent.
    {
        CommutantRoute r(L, cost_formation - 1u, forced);
        CheckEq(static_cast<uint32_t>(r.run()), static_cast<uint32_t>(CommutantOutcome::OverBudget),
                tag + ": a budget below the formation cost is over_budget");
        CheckEq(r.state().rows_done, static_cast<uint64_t>(0), tag + ": rows_done = 0");
        CheckEq(r.state().modmul_spent, static_cast<uint64_t>(0), tag + ": modmul_spent = 0");
        CheckEq(r.certificate().verify(), 0u, tag + ": the partial certificate verifies");
        CheckEq(static_cast<uint32_t>(r.resume(BIG)), static_cast<uint32_t>(fs.outcome),
                tag + ": resume from nothing reaches the same outcome");
        Check(r.state().equals(fs, true), tag + ": resumed state == fresh state, entrywise");
        Check(!r.state().equals(fs, false), tag + ": the budget histories DIFFER (R3.1 item 4)");
        Check(r.state().budget_history == std::vector<uint64_t>{cost_formation - 1u, BIG},
              tag + ": budget_history is [b1, b2]");
        Check(SameCertificateModuloHistory(r.certificate(), full.certificate()),
              tag + ": resumed certificate == fresh certificate");
        CheckEq(r.certificate().verify(), 0u, tag + ": the resumed certificate verifies");
    }

    // (ii) Every reachable row count.
    for (uint64_t rows = 0; rows <= fs.total_rows; ++rows) {
        CommutantRoute r(L, cost_formation + rows * cost_row, forced);
        const CommutantOutcome o = r.run();
        CheckEq(r.state().rows_done, rows, tag + ": rows_done is a function of the budget");
        CheckEq(r.state().modmul_spent, cost_formation + rows * cost_row,
                tag + ": modmul_spent is a function of rows_done");
        CheckEq(r.certificate().verify(), 0u, tag + ": the interrupted certificate verifies");
        if (rows < fs.total_rows) {
            CheckEq(static_cast<uint32_t>(o),
                    static_cast<uint32_t>(CommutantOutcome::OverBudget), tag + ": over_budget");
            r.resume(BIG);
            Check(r.state().equals(fs, true), tag + ": resume(full) == fresh, entrywise");
            Check(SameCertificateModuloHistory(r.certificate(), full.certificate()),
                  tag + ": resumed certificate == fresh certificate");
            CheckEq(r.certificate().verify(), 0u, tag + ": the resumed certificate verifies");
        } else {
            CheckEq(static_cast<uint32_t>(o), static_cast<uint32_t>(fs.outcome),
                    tag + ": the last row completes the route");
            // A resume on a complete state is a STRICT no-op.
            const hecke_ext::CommutantState before = r.state();
            r.resume(BIG * 2u);
            Check(r.state().equals(before, false), tag + ": resume on a complete state is a no-op");
        }
    }

    // (iii) A lower budget is refused with the named error.
    {
        CommutantRoute r(L, cost_formation + cost_row, forced);
        r.run();
        bool named = false;
        try {
            r.resume(cost_formation);
        } catch (const std::logic_error& e) {
            named = std::string(e.what()).find("resume: budget below reached state")
                    != std::string::npos;
        }
        Check(named, tag + ": a lower budget is refused with the named error");
    }

    // (iv) run() twice is refused.
    {
        CommutantRoute r(L, BIG, forced);
        r.run();
        bool threw = false;
        try { r.run(); } catch (const std::logic_error&) { threw = true; }
        Check(threw, tag + ": run() twice is refused");
    }
}

void CheckCommutantTampering(uint32_t ell) {
    const std::string tag = "(b) tamper (ell=" + std::to_string(ell) + ")";
    const std::vector<Case> cases = BuildCases(ell);
    LocalAlgebra L8;   // A^dual: (b) passes
    std::vector<std::pair<uint64_t, ExtMatrix>> ops8;
    if (!RunCase(cases[7], false, L8, ops8) || !L8.ok) {
        Check(false, tag + ": the A^dual block certifies local");
        return;
    }
    CommutantRoute r8(L8);
    r8.run();
    const CommutantCertificate base = r8.certificate();
    CheckEq(base.verify(), 0u, tag + ": the untampered certificate verifies");
    CheckEq(static_cast<uint32_t>(base.outcome),
            static_cast<uint32_t>(CommutantOutcome::Passed), tag + ": it is a `passed` one");

    const auto expect = [&](CommutantCertificate x, uint32_t code, const std::string& what) {
        CheckEq(x.verify(), code, tag + ": " + what);
    };

    { auto x = base; x.rank -= 1; expect(x, 16, "the stored rank decremented"); }
    { auto x = base; x.dim_C += 1; expect(x, 16, "dim C incremented"); }
    // The RCF basis.  A single entry of the CYCLIC GENERATOR's column may
    // legitimately move (e.g. by a vector that `a` kills), and a whole cyclic
    // block may be rescaled -- both give ANOTHER valid rational canonical
    // basis, and the certificate must still verify, because what it re-derives
    // is dim C, which is basis-independent ([X] Sec. 1.3).  What must fail is a
    // perturbation that BREAKS the companion relation a * (column j) =
    // (column j+1) inside a block, so these two tampers touch the image column.
    { auto x = base; x.R.data[0][1] = ModArithmetic::mod_add(x.R.data[0][1], 1u, ell);
      expect(x, 12, "the RCF basis perturbed in an image column"); }
    { auto x = base;
      for (uint32_t i = 0; i < x.D; ++i) std::swap(x.R.data[i][0], x.R.data[i][1]);
      expect(x, 12, "two columns of a cyclic block of the RCF basis swapped"); }
    // ... and the positive control: rescaling a whole cyclic block by 2 is a
    // different, equally valid RCF basis and MUST still verify.
    { auto x = base;
      const uint32_t k0 = x.inv_sizes[0] * static_cast<uint32_t>(x.pi_a.degree());
      for (uint32_t i = 0; i < x.D; ++i) {
        for (uint32_t j = 0; j < k0; ++j) {
          x.R.data[i][j] = ModArithmetic::mod_mul(x.R.data[i][j], 2u % ell, ell);
        }
      }
      expect(x, 0, "a rescaled cyclic block is another valid RCF basis"); }
    { auto x = base; x.a.data[0][0] = ModArithmetic::mod_add(x.a.data[0][0], 1u, ell);
      expect(x, 10, "the chosen element perturbed"); }
    { auto x = base; x.socle_bound += 1; expect(x, 8, "the socle bound incremented"); }
    { auto x = base; x.s_soc += 1; expect(x, 6, "s = dim_F soc V incremented"); }
    { auto x = base; x.dim_soc_A -= 1; expect(x, 7, "dim_F soc A_S decremented"); }
    { auto x = base; x.admits = !x.admits; expect(x, 8, "the socle-bound gate flipped"); }
    { auto x = base; x.u += 1; expect(x, 13, "u incremented"); }
    { auto x = base; x.candidate_index += 1; expect(x, 10, "the candidate index moved"); }
    { auto x = base; x.inv_sizes.pop_back(); expect(x, 11, "the invariant factor list truncated"); }
    { auto x = base; x.outcome = CommutantOutcome::Failed;
      expect(x, 18, "a passing commutant claimed failed"); }
    { auto x = base; x.outcome = CommutantOutcome::ExcludedBySocleBound;
      expect(x, 9, "excluded_by_socle_bound claimed where the gate admits (b)"); }
    { auto x = base; x.cyclicity.a_basis.pop_back(); expect(x, 2, "the A_S basis truncated"); }
    { auto x = base; x.cyclicity.t[0].data[0][0] =
          ModArithmetic::mod_add(x.cyclicity.t[0].data[0][0], 1u, ell);
      expect(x, 4, "an operator entry perturbed (route (a)'s own re-derivation)"); }

    // The partial (over_budget) certificate: its ONE claim is that the stored
    // reduction is the RREF of the first rows_done rows.
    {
        const uint64_t D = base.D, k = base.cyclicity.t.size(), u = base.u;
        const uint64_t cost_formation = 2ull * k * u * D * D * D;
        const uint64_t cost_row = u * u;
        CommutantRoute rp(L8, cost_formation + 4u * cost_row);
        rp.run();
        const CommutantCertificate part = rp.certificate();
        CheckEq(part.verify(), 0u, tag + ": the partial certificate verifies");
        CheckEq(static_cast<uint32_t>(part.outcome),
                static_cast<uint32_t>(CommutantOutcome::OverBudget), tag + ": it is over_budget");
        if (!part.state.system_rref.empty()) {
            auto x = part;
            x.state.system_rref.pop_back();
            expect(x, 15, "the partial reduction truncated");
        }
    }
}

// The route-combining verdict and the OQ8 emission rule: an ideal that cannot
// reach closure is EMITTED with `final = false`, its reason and (where a budget
// was involved) its resumable state -- never withheld.
void CheckGenerationVerdicts(uint32_t ell) {
    const std::string tag = "final (ell=" + std::to_string(ell) + ")";
    using GK = hecke_ext::GenerationKind;
    using SR = hecke_ext::SturmRouteStatus;
    using SF = hecke_ext::SturmFinal;
    const auto vd = [&](bool a, CommutantOutcome b, SR c) {
        return hecke_ext::generation_verdict(a, b, c);
    };
    const auto expect = [&](hecke_ext::GenerationVerdict v, GK kind, SF fin,
                            const std::string& reason, const std::string& what) {
        CheckEq(static_cast<uint32_t>(v.kind), static_cast<uint32_t>(kind),
                tag + ": " + what + " kind");
        CheckEq(static_cast<uint32_t>(v.final_value), static_cast<uint32_t>(fin),
                tag + ": " + what + " final");
        CheckEq(v.reason, std::string(reason), tag + ": " + what + " reason");
    };

    expect(vd(true, CommutantOutcome::NotNeeded, SR::NotNeeded), GK::Cyclic, SF::True, "",
           "(a) certifies");
    expect(vd(false, CommutantOutcome::Passed, SR::NotNeeded), GK::Commutant, SF::True, "",
           "(b) certifies");
    expect(vd(false, CommutantOutcome::ExcludedBySocleBound, SR::Passed), GK::Sturm, SF::True, "",
           "(c) certifies");
    expect(vd(false, CommutantOutcome::ExcludedBySocleBound, SR::Unavailable),
           GK::Unavailable, SF::False, "closure_unreachable", "the socle-bound proof");
    expect(vd(false, CommutantOutcome::Failed, SR::Unavailable),
           GK::Unavailable, SF::False, "closure_unreachable", "a computed dim C > dim A_S");
    expect(vd(false, CommutantOutcome::OverBudget, SR::Unavailable),
           GK::Unavailable, SF::False, "budget_exhausted:commutant", "the commutant budget");
    expect(vd(false, CommutantOutcome::ExcludedBySocleBound, SR::Failed),
           GK::Unavailable, SF::False, "certificate_failed:sturm", "a failed sweep");
    expect(vd(false, CommutantOutcome::NotNeeded, SR::Unavailable),
           GK::Unavailable, SF::False, "sturm_unavailable", "no B' below the cap");

    // The emission itself, on the real (10, 691)-shaped certificate.
    const std::vector<Case> cases = BuildCases(ell);
    LocalAlgebra L9;
    std::vector<std::pair<uint64_t, ExtMatrix>> ops9;
    if (!RunCase(cases[8], false, L9, ops9) || !L9.ok) {
        Check(false, tag + ": the A + A/m block certifies local");
        return;
    }
    CommutantRoute r9(L9);
    r9.run();
    const std::string j9 = r9.certificate().to_json();
    Check(j9.find("\"outcome\":\"excluded_by_socle_bound\"") != std::string::npos,
          tag + ": the emitted record names the outcome");
    Check(j9.find("\"socle_bound\":5") != std::string::npos,
          tag + ": the emitted record carries the socle bound 5");
    Check(j9.find("\"admits\":false") != std::string::npos,
          tag + ": the emitted record carries the gate");
    Check(j9.find("\"dim_soc_A_S\":1") != std::string::npos,
          tag + ": the emitted record carries dim soc A_S");
    Check(j9.find("\"resumable\":") != std::string::npos,
          tag + ": the emitted record carries a resumable object");
    Check(j9.find("\"complete\":true") != std::string::npos,
          tag + ": closure_unreachable is not a budget outcome (complete = true)");
    Check(j9.find("\"verify\":0") != std::string::npos,
          tag + ": the emitted record verifies");
    expect(vd(false, r9.outcome(), SR::Unavailable), GK::Unavailable, SF::False,
           "closure_unreachable", "the real excluded certificate");

    // And the budget-exhausted emission, which must carry complete = false.
    {
        LocalAlgebra L8;
        std::vector<std::pair<uint64_t, ExtMatrix>> ops8;
        if (RunCase(cases[7], false, L8, ops8) && L8.ok) {
            CommutantRoute probe(L8);
            probe.run();
            const uint64_t D = probe.state().D, k = probe.state().k, u = probe.state().u;
            CommutantRoute rp(L8, 2ull * k * u * D * D * D + u * u);
            rp.run();
            const std::string jp = rp.certificate().to_json();
            Check(jp.find("\"outcome\":\"over_budget\"") != std::string::npos,
                  tag + ": the budget-exhausted record names the outcome");
            Check(jp.find("\"complete\":false") != std::string::npos,
                  tag + ": budget_exhausted always comes with complete = false");
            Check(jp.find("\"rows_done\":1") != std::string::npos,
                  tag + ": the record says how far it got");
            Check(jp.find("\"state\":") != std::string::npos,
                  tag + ": the state is inlined (R3.1 item 6, under 64 KiB)");
            expect(vd(false, rp.outcome(), SR::Unavailable), GK::Unavailable, SF::False,
                   "budget_exhausted:commutant", "the real over_budget certificate");
        }
    }
}

} // namespace

uint32_t HeckeExtAlgebraValidation(void) {
    std::cout << "=== HeckeExtAlgebraValidation (hecke-extensions E1a / WP-4 + WP-5) ==="
              << std::endl;
    std::cout << "    the local algebra A_S, its invariants and generation certificates (a)"
              << std::endl
              << "    and (b) -- the socle bound, the structured commutant and its resumable"
              << std::endl
              << "    CommutantState -- on constructed truths ([E] sec:localalg / alg:stage3"
              << std::endl
              << "    / cert:cyclic / cert:comm)." << std::endl;

    Check.reset();

    const std::vector<uint32_t> ells = {5u, 13u, 691u, 134217757u};
    for (uint32_t ell : ells) {
        const std::vector<Case> cases = BuildCases(ell);
        for (const Case& c : cases) {
            CheckCase(c, false);
            CheckCase(c, true);
            // WP-5 (E1a-2): the commutant, on every case and in both bases.
            CheckCommutantCase(c, false);
            CheckCommutantCase(c, true);
        }
        CheckWitnessDirectionTrap(ell);
        CheckCrossBlock(ell);
        CheckTampering(ell);
        // WP-5 (E1a-2).
        CheckCommutantResume(cases[7], false, "A^dual (unforced, (b) passes)");
        CheckCommutantTampering(ell);
        CheckGenerationVerdicts(ell);
    }
    // One larger R3 sweep, on the forced non-cyclic A + A (D = 6, u = 12,
    // 36 rows), so the row loop is exercised well past the small cases.
    CheckCommutantResume(BuildCases(691u)[5], true, "A + A (forced)");

    std::cout << "    checks " << Check.checks << ", failing " << Check.fails << std::endl;
    if (Check.fails == 0) {
        std::cout << "=== HeckeExtAlgebraValidation PASSED ===" << std::endl;
    } else {
        std::cout << "=== HeckeExtAlgebraValidation FAILED ===" << std::endl;
    }
    return Check.fails;
}
