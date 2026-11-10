// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S7V -- see HeckeOracleValidation.h for the file-level independence
// statement, the gating rules (amendment A-F4), and the full oracle list.
//
// =====================================================================
// INDEPENDENT DERIVATION OF THE EXPECTED VALUES (read before the numbers
// below: none of them are read off S7's own smoke test
// HeckeOperatorSmokeValidation.cpp -- only its PUBLIC header signatures and
// HeckeCosetTables.h / HeckeTauTable.h / EquivariantCohomologyOnK.h /
// LocalSystemSL2Z.h were consulted for how to call the pipeline).
//
// (1) Eichler-Shimura dimension formula (used by O14 and every "dim H^1"
//     assertion below): for even weight k >= 4,
//         dim H^1(SL(2,Z); Sym^{k-2}(F_ell^2)) = 2*dim S_k + 1   (ell >= 5),
//     where the "+1" is the (1-dimensional) Eisenstein line and each cusp
//     form contributes a conjugate holomorphic/antiholomorphic pair. This
//     is char-0 Eichler-Shimura reduced mod ell; for ell >= 5 the integral
//     comparison has no Tor obstruction (plan amendment A-F4), so the same
//     formula holds mod ell for EVERY ell >= 5, independent of the size of
//     ell relative to n -- this is exactly what licenses dropping the
//     "ell > n+2" gate for the dimension oracle.
//
// (2) Classical dimension formula for M_k(SL(2,Z)) (k even, k >= 4):
//         dim M_k = floor(k/12) + 1          if k !== 2 (mod 12)
//         dim M_k = floor(k/12)              if k === 2 (mod 12)
//     and dim S_k = dim M_k - 1 for k >= 4 (the Eisenstein series E_k spans
//     the 1-dimensional complement). Independently re-tabulated here (NOT
//     copied from the task's constant list, though it agrees with it, which
//     is itself a cross-check):
//         k=4:  dim M_4=1, dim S_4=0.      (n=2)
//         k=6:  dim M_6=1, dim S_6=0.      (n=4)
//         k=8:  dim M_8=1, dim S_8=0.      (n=6)
//         k=10: dim M_10=1, dim S_10=0.    (n=8)
//         k=12: 12 !== 2 (mod 12) => dim M_12 = 1+1 = 2, dim S_12=1 (= C*Delta). (n=10)
//         k=14: 14 === 2 (mod 12) => dim M_14 = 1, dim S_14=0.  (n=12)
//         k=16: dim M_16 = 1+1 = 2, dim S_16=1.  (n=14)
//         k=24: 24 !== 2 (mod 12) => dim M_24 = 2+1 = 3, dim S_24=2. (n=22)
//     giving dim H^1 = 2*dim S_{n+2}+1: n=2,4,6,8,12 -> 1; n=10,14 -> 3;
//     n=22 -> 5. (n=0 is the trivial-coefficient special case H^0=1,H^1=0,
//     not covered by the k>=4 formula above.)
//
// (3) Hecke eigenvalues of the unique level-1 cusp forms of weight 12, 16,
//     24 and the Eisenstein eigenvalue 1+p^{n+1} in weight n+2: these are
//     the TRUSTED constants supplied by the task (independently re-verified
//     by the contract author against LMFDB before this stage started; see
//     the task's explicit statement that weights 12/16/24 + all Eisenstein
//     eigenvalues + the 691 Ramanujan congruence are safe to hard-assert,
//     while weights 18/20/26 are explicitly OUT OF SCOPE here). This file
//     does not re-derive a_2(Delta)=-24, a_3(Delta)=252, a_2(f_16)=216, or
//     the weight-24 pair's trace/norm from first principles (that would
//     require constructing the actual modular forms, out of scope for a
//     validation agent with no network access) -- it takes them as the
//     external ground truth against which the PIPELINE is checked, exactly
//     as the task specifies. The Eisenstein eigenvalue 1+p^{n+1} IS
//     re-derived independently here from the classical fact that the
//     weight-k level-1 Eisenstein series E_k is a T_p-eigenform with
//     eigenvalue 1+p^{k-1} = 1+p^{n+1} (Diamond-Shurman / Serre, standard).
//
// (4) The Ramanujan congruence tau(p) === p^11+1 (mod 691) (Ramanujan's own
//     1916 conjecture, proved by Wilton et al.): 2049 = 1+2^11 and
//     -24 = a_2(Delta); 2049-(-24) = 2073 = 3*691, i.e. 2049 === -24
//     (mod 691), so (X+24)^2*(X-2049) === (X+24)^3 (mod 691) -- an
//     unconditional polynomial identity given the weight-12 constants of
//     (3), re-verified by direct arithmetic here (2049+24 = 2073 = 3*691).
//
// (5) Division-free char-poly: charpoly(A) := det(X*I - A), computed via
//     ordinary Leibniz/cofactor expansion along the first row over the ring
//     F_ell[X] (entries of X*I-A are represented as length-<=2 low->high
//     coefficient lists; polynomial +,-,* only, NEVER a division). This is
//     the elementary definition of the determinant and is valid over ANY
//     commutative ring, in particular F_ell for every prime ell used here
//     (including ell=5,7 <= d_q=5, where Faddeev-LeVerrier's division by k
//     up to d_q would fail). Cost O(d_q!) is trivial for d_q <= 5.
//
// =====================================================================
// TWO GENUINE, INDEPENDENTLY-CONFIRMED MATHEMATICAL FINDINGS (read before
// the failure counts below: neither is a bug in Stage S7's new code --
// both were CROSS-CHECKED by reproducing the identical numeric discrepancy
// on the pre-Hecke-branch merge-base commit 0a931e1 ("CMake configuration
// finalized", the ancestor of "cdad0b9 Phase 7 complete" that both `main`
// and `hecke-action` share), using ONLY the pre-existing, untouched
// `TestSL2ZCohomologySymmetricPower` / `EquivariantCohomologyOnK` /
// `GenerateSL2ZRelations_7VertexModel` machinery -- none of Stage S1-S7's
// Hecke code is on that commit at all, so the discrepancy cannot be an S7
// regression):
//
// FINDING A -- O2 (odd n => H^q=0) FAILS against the actual (K,R) model.
//   The contract's claim itself is mathematically airtight (standard fact:
//   -I is central in SL(2,Z), so for any g in Z(G) the pair (conjugation-
//   by-g = id_G, rho(g)) induces the IDENTITY on H^*(G,M) -- apply this to
//   g=-I, rho_n(-I)=(-1)^n*Id_V: for odd n this forces c = -c, i.e. 2c=0,
//   for EVERY class c in EVERY degree, killing H^* whenever ell is odd).
//   But the codebase's 7-vertex model's SimplexRelation list (both the
//   legacy GenerateSL2ZRelations_7VertexModel AND the S2-refactored
//   GenerateSL2ZRelationsWithGroupData -- IDENTICAL relation data, I3) only
//   encodes the STABILIZER-INDUCED occurrences of -I at the three elliptic
//   points (v6: (M_S)^2=-I via the order-4 relation; v4,v5: (M_ST)^3=-I,
//   (M_TS)^3=-I via the order-6 relations) -- NOT the fact that -I ALSO
//   stabilizes (trivially, since -I fixes every point of the upper half
//   plane by the Moebius formula) every OTHER, "free"-PSL(2,Z)-orbit
//   simplex (v0..v3 on the cusp line, and dependent edges/triangles). Per
//   [C, rmk:orbifold]'s own reasoning ("each chain group is Z[Gamma/Gamma_
//   sigma] with |Gamma_sigma| dividing 12"), Gamma_sigma = Stab_{SL(2,Z)}
//   (point) contains {+-I} (order >= 2) at EVERY point, not just the
//   elliptic ones -- so a faithful combinatorial model needs an explicit
//   f(sigma) = rho_n(-I) f(sigma) relation at the free simplices too, which
//   this model omits. For EVEN n this omission is harmless (rho_n(-I)=I is
//   a tautology there), which is exactly why it was never caught: every
//   existing regression/oracle in this codebase (pre- and post-Hecke) only
//   ever exercises even n. Direct computation confirms this precisely:
//   dim H^1 comes out to EXACTLY n+1 = d_V (a full extra copy of V,
//   ell-independent -- verified identical at ell=5 AND ell=7) instead of 0,
//   for n in {1,3,5}, on BOTH this branch's pipeline and the pre-Hecke
//   commit's TestSL2ZCohomologySymmetricPower(n,ell) directly.
//   CONCLUSION: O2 is kept as a HARD assertion below (per the task's
//   instruction to test the contract's own oracle, and per "FAIL LOUDLY,
//   do not patch S7 to force a pass") -- it WILL fail, honestly reporting a
//   real, pre-existing (dating to at least "Phase 7 complete", well before
//   any Hecke-branch work) gap in the base (K,R) model / relation list,
//   OUTSIDE Stage S7's own scope to fix (S7 only consumes the relations;
//   S2's GenerateSL2ZRelationsWithGroupData is bit-identical to the legacy
//   generator by construction, I3, so the fix -- if made -- belongs to the
//   base model, not to any Hecke stage).
//
// FINDING B -- Dickson-invariant exceptional locus at n === 0 (mod ell+1).
//   Sym^n(F_ell^2) is, as an SL_2(F_ell)-module (not just an SL(2,Z)-
//   module), NON-semisimple/reducible once n >= ell-1 (classical modular
//   representation theory of SL_2(F_q); Steinberg tensor product theorem).
//   In particular Sym^{ell+1}(F_ell^2) contains the DICKSON INVARIANT
//   L(x,y) := x^ell*y - x*y^ell (a bona fide SL_2(F_ell)-invariant of
//   degree ell+1 -- an elementary, classical fact, e.g. Dickson 1911), and
//   its powers L^k give invariants in degree k*(ell+1) for every k >= 1.
//   Since the reduction map SL(2,Z) -> SL_2(F_ell) is surjective and rho_n
//   factors through it, this produces a genuine EXTRA SL(2,Z)-invariant
//   (nonzero dim H^0) -- and, empirically, a matching extra dimension in
//   H^1 -- exactly when (ell+1) | n, n > 0. This directly contradicts
//   amendment A-F4's blanket premise ("no Tor obstruction for ell>=5,
//   independent of n") at these specific (n,ell) pairs: A-F4's own
//   justification is about INTEGRAL cohomology comparison (no p-torsion in
//   the SES defining H^1(Z)⊗F_ell), which is a different phenomenon from
//   Sym^n(F_ell^2) itself becoming a non-simple SL_2(F_ell)-module once
//   n >= ell-1 -- a case A-F4's text does not appear to have anticipated.
//   Confirmed via direct computation AND independently re-derived from
//   Dickson's classical invariant theory (not merely observed): for every
//   (n,ell) pair used by this file's grids, this fires EXACTLY at
//   (n,ell) in {(6,5), (12,5), (8,7)} -- i.e. exactly the multiples of
//   ell+1 present in the tested n-grid -- and nowhere else (verified: none
//   of the other grid points, e.g. ell=101/10007/691/134217757 with
//   ell+1 far larger than any tested n, are affected).
//   CONCLUSION: these three (n,ell) pairs are RECORDED ONLY (not
//   hard-asserted) below, exactly like the codebase's existing convention
//   for a case that is not independently certifiable by first-principles
//   re-derivation (cf. HeckeShapiroValidation.cpp's n=2,ell=5 case) --
//   this is a genuine mathematical exception to A-F4's premise, not a
//   pipeline bug (same reproduction method as FINDING A above).
// =====================================================================

#include "HeckeOracleValidation.h"

#include "DicksonCount.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// ===========================================================================
// Division-free F_ell[X] arithmetic, LOCAL to this validation file (does
// NOT depend on Stage S10's FpPoly, which has not landed). Polynomials are
// stored low -> high degree (constant term first), trimmed so the highest
// stored coefficient is always nonzero (the zero polynomial is {}).
// ===========================================================================

using Poly = std::vector<uint32_t>;

uint32_t ModOfInt64(int64_t x, uint32_t ell) {
    int64_t m = x % static_cast<int64_t>(ell);
    if (m < 0) m += static_cast<int64_t>(ell);
    return static_cast<uint32_t>(m);
}

Poly PolyTrim(Poly p) {
    while (!p.empty() && p.back() == 0u) p.pop_back();
    return p;
}

Poly PolyAdd(const Poly& a, const Poly& b, uint32_t ell) {
    Poly r(std::max(a.size(), b.size()), 0u);
    for (size_t i = 0; i < a.size(); ++i) r[i] = ModArithmetic::mod_add(r[i], a[i], ell);
    for (size_t i = 0; i < b.size(); ++i) r[i] = ModArithmetic::mod_add(r[i], b[i], ell);
    return PolyTrim(r);
}

Poly PolyScale(const Poly& a, uint32_t c, uint32_t ell) {
    Poly r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = ModArithmetic::mod_mul(a[i], c, ell);
    return PolyTrim(r);
}

Poly PolyMul(const Poly& a, const Poly& b, uint32_t ell) {
    if (a.empty() || b.empty()) return {};
    Poly r(a.size() + b.size() - 1, 0u);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0u) continue;
        for (size_t j = 0; j < b.size(); ++j) {
            if (b[j] == 0u) continue;
            r[i + j] = ModArithmetic::mod_add(r[i + j], ModArithmetic::mod_mul(a[i], b[j], ell), ell);
        }
    }
    return PolyTrim(r);
}

bool PolyEqual(const Poly& a, const Poly& b) {
    return PolyTrim(a) == PolyTrim(b);
}

// FINDING B (see the file-header derivation): the Dickson-invariant
// exceptional locus of Sym^n(F_ell^2) as an SL_2(F_ell)-module -- fires
// exactly when n is a non-negative combination of the two Dickson degrees
// (ell+1) and (ell^2-ell). RECORDED, not hard-asserted, at these points
// (below), independent of and never touching Stage S7's own code.
//
// (hecke-extensions WP-2, [X Sec.2.1]) This used to check only the FIRST
// generator, (ell+1) | n -- an under-approximation that misses the second
// Dickson degree (e.g. n=20, ell=5, per HeckeEndToEndValidation.cpp's own
// header note). Unified onto the public `DicksonInvariantCount`, which the
// other two pre-existing copies (HeckeEndToEndValidation.cpp,
// Golden/GoldenHarness.cpp) already computed correctly by looping over both
// generators; this file's grid (max n=14, plus 22) is unaffected -- no row
// below newly fires -- but a future n=20 row now would.
bool IsDicksonExceptional(uint32_t n, uint32_t ell) {
    return hecke::IsDicksonExceptionalCount(n, ell);
}

std::string PolyToString(const Poly& p) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < p.size(); ++i) {
        if (i) oss << ", ";
        oss << p[i];
    }
    oss << "] (coeffs low->high)";
    return oss.str();
}

// (X - root)^mult mod ell.
Poly PolyLinearFactorPow(int64_t root, uint32_t mult, uint32_t ell) {
    const Poly factor = PolyTrim(Poly{ModOfInt64(-root, ell), 1u});
    Poly result{1u};
    for (uint32_t i = 0; i < mult; ++i) result = PolyMul(result, factor, ell);
    return result;
}

// Recursive Leibniz/cofactor-expansion determinant of a matrix of
// polynomials over F_ell (division-free; see file header note (5)).
Poly PolyMatDet(const std::vector<std::vector<Poly>>& M, uint32_t ell) {
    const size_t n = M.size();
    if (n == 0) return Poly{1u};
    if (n == 1) return M[0][0];
    Poly result; // zero
    for (size_t j = 0; j < n; ++j) {
        std::vector<std::vector<Poly>> minor(n - 1, std::vector<Poly>(n - 1));
        for (size_t r = 1; r < n; ++r) {
            size_t cc = 0;
            for (size_t c = 0; c < n; ++c) {
                if (c == j) continue;
                minor[r - 1][cc++] = M[r][c];
            }
        }
        Poly term = PolyMul(M[0][j], PolyMatDet(minor, ell), ell);
        if (j % 2 == 1) term = PolyScale(term, ell - 1u, ell); // negate mod ell
        result = PolyAdd(result, term, ell);
    }
    return result;
}

// charpoly(A) = det(X*I - A), division-free, valid for every prime ell
// (including ell <= A.rows). See file header note (5).
Poly CharPolyDivisionFree(const Matrix& A, uint32_t ell) {
    const uint32_t n = A.rows;
    std::vector<std::vector<Poly>> M(n, std::vector<Poly>(n));
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t aij = A.data[i][j];
            const uint32_t neg_aij = (aij == 0u) ? 0u : (ell - aij);
            M[i][j] = (i == j) ? PolyTrim(Poly{neg_aij, 1u}) : PolyTrim(Poly{neg_aij});
        }
    }
    return PolyMatDet(M, ell);
}

// ===========================================================================
// Pipeline fixtures. CohomologyFixture builds L = Sym^n(F_ell^2) on the
// 7-vertex model and computes H^*_eq(K,R;L) exactly per the task's pipeline
// spec. HeckeFixture builds the S3/S4/S7 tower for a fixed (p_hecke,ell,n)
// on top of an already-computed CohomologyFixture.
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

struct HeckeFixture {
    hecke::HeckeCosetTables cosets;
    hecke::HeckeTauTable tau;
    hecke::HeckeOperatorOnK op;

    HeckeFixture(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                 EquivariantCohomologyOnK& coh, uint64_t p_hecke, uint32_t ell, uint32_t n_sym)
        : cosets(p_hecke, /*r=*/1, ell, n_sym),
          tau(K, model, cosets, ell, n_sym),
          op(&coh, &cosets, &tau) {
        op.compute();
    }
};

// ===========================================================================
// Oracle-checking context: caches fixtures across oracles that reuse the
// same (n,ell) or (p,ell,n) so the ~40-case battery stays cheap, runs every
// check, and prints/records a full diagnosis (offending (q,n,ell,p), the
// computed matrix, computed vs expected char-poly) on any mismatch.
// ===========================================================================

class OracleContext {
public:
    OracleContext(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model)
        : K_(K), model_(model) {}

    void RunAll() {
        RunO1();
        RunO2();
        RunO3();
        RunO4();
        RunO5();
        RunO6();
        RunO7();
        RunO8();
        RunO14();
    }

    uint32_t failures() const { return failures_; }

private:
    const SimplicialComplex& K_;
    const hecke::SL2ZTessellationTables& model_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<CohomologyFixture>> coh_cache_;
    std::map<std::tuple<uint64_t, uint32_t, uint32_t>, std::unique_ptr<HeckeFixture>> hecke_cache_;
    uint32_t failures_ = 0;

    void Fail(const std::string& msg) {
        ++failures_;
        std::cout << "HeckeOracleValidation: FAIL: " << msg << std::endl;
    }

    CohomologyFixture& Coh(uint32_t n, uint32_t ell) {
        const auto key = std::make_pair(n, ell);
        auto it = coh_cache_.find(key);
        if (it == coh_cache_.end()) {
            it = coh_cache_.emplace(key, std::make_unique<CohomologyFixture>(K_, n, ell)).first;
        }
        return *it->second;
    }

    HeckeFixture& Hecke(uint64_t p, uint32_t ell, uint32_t n) {
        const auto key = std::make_tuple(p, ell, n);
        auto it = hecke_cache_.find(key);
        if (it == hecke_cache_.end()) {
            CohomologyFixture& cf = Coh(n, ell);
            it = hecke_cache_.emplace(key, std::make_unique<HeckeFixture>(K_, model_, cf.coh, p, ell, n)).first;
        }
        return *it->second;
    }

    // --- generic check helpers (each catches its own exceptions so one bad
    // case never aborts the rest of the battery) ---------------------------

    void CheckDims(uint32_t n, uint32_t ell, uint32_t expected_h0, uint32_t expected_h1,
                    uint32_t expected_h2, const std::string& tag) {
        try {
            CohomologyFixture& cf = Coh(n, ell);
            const auto& dims = cf.coh.get_H_eq_dimensions();
            if (dims.size() < 3) {
                std::ostringstream oss;
                oss << "[" << tag << "] n=" << n << " ell=" << ell
                    << ": get_H_eq_dimensions() has size " << dims.size() << " (expected >= 3)";
                Fail(oss.str());
                return;
            }
            const bool ok = (dims[0] == expected_h0) && (dims[1] == expected_h1) && (dims[2] == expected_h2);
            if (!ok) {
                std::ostringstream oss;
                oss << "[" << tag << "] DIMENSION MISMATCH at n=" << n << " ell=" << ell
                    << ": got (H^0,H^1,H^2)=(" << dims[0] << "," << dims[1] << "," << dims[2]
                    << "), expected (" << expected_h0 << "," << expected_h1 << "," << expected_h2 << ")";
                Fail(oss.str());
            } else {
                std::cout << "[" << tag << "] dims OK: n=" << n << " ell=" << ell
                          << " (H^0,H^1,H^2)=(" << dims[0] << "," << dims[1] << "," << dims[2] << ")"
                          << std::endl;
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "[" << tag << "] EXCEPTION building cohomology at n=" << n << " ell=" << ell
                << ": " << e.what();
            Fail(oss.str());
        }
    }

    // FINDING B: records (prints, does not assert) the dimension vector at
    // a Dickson-invariant exceptional (n,ell) pair -- see the file-header
    // derivation. Never contributes to failures().
    void RecordDims(uint32_t n, uint32_t ell, const std::string& tag) {
        try {
            CohomologyFixture& cf = Coh(n, ell);
            const auto& dims = cf.coh.get_H_eq_dimensions();
            std::ostringstream oss;
            oss << "[" << tag << "] DICKSON-EXCEPTIONAL (n=" << n << " === 0 mod ell+1="
                << (ell + 1) << "), RECORDED ONLY, not asserted: (H^0,H^1,H^2)=(";
            for (size_t i = 0; i < 3; ++i) {
                if (i) oss << ",";
                oss << (i < dims.size() ? dims[i] : 0u);
            }
            oss << ")";
            std::cout << oss.str() << std::endl;
        } catch (const std::exception& e) {
            Fail(tag + " EXCEPTION while recording Dickson-exceptional case (n=" +
                 std::to_string(n) + ", ell=" + std::to_string(ell) + "): " + e.what());
        }
    }

    void CheckA1x1(uint64_t p, uint32_t ell, uint32_t n, uint32_t q, uint32_t expected,
                    const std::string& tag) {
        try {
            HeckeFixture& hf = Hecke(p, ell, n);
            const auto& H = hf.op.get_H_matrices();
            if (q >= H.size()) {
                Fail(tag + ": degree q=" + std::to_string(q) + " out of range");
                return;
            }
            const Matrix& Aq = H[q];
            if (Aq.rows != 1 || Aq.cols != 1) {
                std::ostringstream oss;
                oss << "[" << tag << "] SHAPE MISMATCH (q=" << q << ", n=" << n << ", ell=" << ell
                    << ", p=" << p << "): A^(" << q << ") is " << Aq.rows << "x" << Aq.cols
                    << ", expected 1x1";
                Fail(oss.str());
                return;
            }
            const uint32_t got = Aq.data[0][0];
            if (got != expected) {
                std::ostringstream oss;
                oss << "[" << tag << "] MISMATCH (q=" << q << ", n=" << n << ", ell=" << ell
                    << ", p=" << p << "): A^(" << q << ")(T_" << p << ") = (" << got
                    << "), expected (" << expected << "). Matrix:\n"
                    << Aq.to_string();
                Fail(oss.str());
            } else {
                std::cout << "[" << tag << "] OK: A^(" << q << ")(T_" << p << ") = (" << got
                          << ") at n=" << n << " ell=" << ell << std::endl;
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "[" << tag << "] EXCEPTION (n=" << n << ", ell=" << ell << ", p=" << p
                << "): " << e.what();
            Fail(oss.str());
        }
    }

    void CheckCharpoly(uint64_t p, uint32_t ell, uint32_t n, uint32_t q, const Poly& expected,
                        const std::string& tag) {
        try {
            HeckeFixture& hf = Hecke(p, ell, n);
            const auto& H = hf.op.get_H_matrices();
            if (q >= H.size()) {
                Fail(tag + ": degree q=" + std::to_string(q) + " out of range");
                return;
            }
            const Matrix& Aq = H[q];
            const Poly computed = CharPolyDivisionFree(Aq, ell);
            if (!PolyEqual(computed, expected)) {
                std::ostringstream oss;
                oss << "[" << tag << "] CHARPOLY MISMATCH at (q=" << q << ", n=" << n
                    << ", ell=" << ell << ", p=" << p << ")\n"
                    << "  computed A^(" << q << ")(T_" << p << ") =\n" << Aq.to_string()
                    << "  computed charpoly = " << PolyToString(computed) << "\n"
                    << "  expected charpoly = " << PolyToString(expected);
                Fail(oss.str());
            } else {
                std::cout << "[" << tag << "] charpoly OK at n=" << n << " ell=" << ell
                          << " p=" << p << ": " << PolyToString(computed) << std::endl;
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "[" << tag << "] EXCEPTION (n=" << n << ", ell=" << ell << ", p=" << p
                << "): " << e.what();
            Fail(oss.str());
        }
    }

    void CheckZeroMatrices(uint64_t p, uint32_t ell, uint32_t n, const std::string& tag) {
        try {
            HeckeFixture& hf = Hecke(p, ell, n);
            const auto& H = hf.op.get_H_matrices();
            if (H.size() != 3) {
                Fail(tag + ": get_H_matrices() has size " + std::to_string(H.size()) + " (expected 3)");
                return;
            }
            bool ok = true;
            for (uint32_t q = 0; q < 3; ++q) {
                if (H[q].rows != 0 || H[q].cols != 0) ok = false;
            }
            if (!ok) {
                std::ostringstream oss;
                oss << "[" << tag << "] EXPECTED ALL-EMPTY MATRICES at n=" << n << " ell=" << ell
                    << " p=" << p << ", got shapes ("
                    << H[0].rows << "x" << H[0].cols << ", "
                    << H[1].rows << "x" << H[1].cols << ", "
                    << H[2].rows << "x" << H[2].cols << ")";
                Fail(oss.str());
            } else {
                std::cout << "[" << tag << "] OK: all A^(q) are 0x0 at n=" << n << " ell=" << ell
                          << " p=" << p << std::endl;
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "[" << tag << "] EXCEPTION (n=" << n << ", ell=" << ell << ", p=" << p
                << "): " << e.what();
            Fail(oss.str());
        }
    }

    // --- oracle implementations ---------------------------------------------

    // O1: n=0, ell in {5,7,101}, p in {2,3,5,7}: dim H^0=1,H^1=H^2=0;
    // A^(0)(T_p) = (p+1 mod ell). [C, sec:example] item 1 ("Trivial
    // coefficients"); independently: T_p = Gamma diag(1,p) Gamma has p+1
    // cosets, and on the trivial 1-dim representation Theta_g sums the
    // rho_0(.) = 1 contribution over all p+1 cosets, giving the scalar p+1.
    void RunO1() {
        for (uint32_t ell : {5u, 7u, 101u}) {
            CheckDims(0, ell, /*h0=*/1, /*h1=*/0, /*h2=*/0, "O1");
            for (uint64_t p : {2u, 3u, 5u, 7u}) {
                const uint32_t expected = static_cast<uint32_t>((p + 1) % ell);
                CheckA1x1(p, ell, /*n=*/0, /*q=*/0, expected, "O1");
            }
        }
    }

    // O2: n odd in {1,3,5}, ell in {5,7} (ell != 2): all H^q = 0, pipeline
    // returns 0x0 matrices in every degree. [C, sec:example] item 2 /
    // FINDING A of the file header: kept as a HARD assertion (the
    // underlying group-cohomology fact is airtight) -- this is EXPECTED TO
    // FAIL against the current (K,R) model for the confirmed, pre-existing
    // (non-S7) reason documented at length in the file header ("FINDING
    // A"). Left as a hard assertion deliberately, per the task's explicit
    // instruction to fail loudly and report rather than suppress a
    // genuine finding.
    void RunO2() {
        for (uint32_t n : {1u, 3u, 5u}) {
            for (uint32_t ell : {5u, 7u}) {
                CheckDims(n, ell, 0, 0, 0, "O2");
                CheckZeroMatrices(/*p=*/2, ell, n, "O2");
            }
        }
    }

    // O3: n=2, ell=5 (weight 4, no cusp forms, [C, sec:example] item 3):
    // dim H^1=1 (pure Eisenstein); A^(1)(T_2) = 1+2^3 = 9 = 4 mod 5;
    // A^(1)(T_3) = 1+3^3 = 28 = 3 mod 5.
    void RunO3() {
        CheckDims(2, 5, 0, 1, 0, "O3");
        CheckA1x1(2, 5, 2, 1, ModOfInt64(1 + 8, 5), "O3-T2");   // 9 mod 5 = 4
        CheckA1x1(3, 5, 2, 1, ModOfInt64(1 + 27, 5), "O3-T3");  // 28 mod 5 = 3
    }

    // O4: n=8, ell in {5,7,101} (weight 10, dim S_10=0, pure Eisenstein):
    // dim H^1=1; A^(1)(T_2) = 1+2^9 = 513 mod ell. EXCEPT ell=7: n=8=ell+1
    // is the FINDING-B Dickson-invariant exceptional locus -- recorded
    // only there (see file header).
    void RunO4() {
        for (uint32_t ell : {5u, 7u, 101u}) {
            if (IsDicksonExceptional(8, ell)) {
                RecordDims(8, ell, "O4-Dickson");
                continue;
            }
            CheckDims(8, ell, 0, 1, 0, "O4");
            CheckA1x1(2, ell, 8, 1, ModOfInt64(513, ell), "O4");
        }
    }

    // O5: n=10, weight 12 (S_12 = C*Delta), ell in {5,7,13,101,10007}
    // (A-F4: NO ell>n+2 gate, so ell=5,7 <= n+2=12 are asserted too):
    // dim H^1=3; charpoly(A^(1)(T_2)) = (X+24)^2(X-2049);
    // charpoly(A^(1)(T_3)) = (X-252)^2(X-177148); both mod ell.
    void RunO5() {
        for (uint32_t ell : {5u, 7u, 13u, 101u, 10007u}) {
            CheckDims(10, ell, 0, 3, 0, "O5");
            const Poly expected_T2 =
                PolyMul(PolyLinearFactorPow(-24, 2, ell), PolyLinearFactorPow(2049, 1, ell), ell);
            CheckCharpoly(2, ell, 10, 1, expected_T2, "O5-T2");
            const Poly expected_T3 =
                PolyMul(PolyLinearFactorPow(252, 2, ell), PolyLinearFactorPow(177148, 1, ell), ell);
            CheckCharpoly(3, ell, 10, 1, expected_T3, "O5-T3");
        }
    }

    // O6: n=10, ell=691 (Ramanujan congruence, HARD PART ONLY per the
    // task's gating): charpoly(A^(1)(T_2)) = (X+24)^3 mod 691 (since
    // 2049+24 = 2073 = 3*691); HARD: A^(1)(T_2) != -24*I. Jordan partition
    // (rank(A+24I)) is RECORDED only (amendment A-F2 / Stage S10V's job).
    void RunO6() {
        const uint32_t ell = 691;
        CheckDims(10, ell, 0, 3, 0, "O6");
        try {
            HeckeFixture& hf = Hecke(2, ell, 10);
            const auto& H = hf.op.get_H_matrices();
            if (H.size() < 2 || H[1].rows != 3 || H[1].cols != 3) {
                Fail("O6: A^(1)(T_2) has unexpected shape");
                return;
            }
            const Matrix& A1 = H[1];

            // Independent cross-check of the Ramanujan congruence itself
            // (not just trusting the task's restatement): 2049 - (-24) must
            // be an exact multiple of 691.
            if ((2049 - (-24)) % 691 != 0) {
                Fail("O6: internal derivation error -- 2049+24 is not a multiple of 691");
                return;
            }
            const Poly expected = PolyLinearFactorPow(-24, 3, ell);
            const Poly computed = CharPolyDivisionFree(A1, ell);
            if (!PolyEqual(computed, expected)) {
                std::ostringstream oss;
                oss << "[O6] CHARPOLY MISMATCH at (q=1, n=10, ell=691, p=2)\n"
                    << "  computed A^(1)(T_2) =\n" << A1.to_string()
                    << "  computed charpoly = " << PolyToString(computed) << "\n"
                    << "  expected charpoly = " << PolyToString(expected) << " ((X+24)^3 mod 691)";
                Fail(oss.str());
            } else {
                std::cout << "[O6] charpoly OK: (X+24)^3 mod 691." << std::endl;
            }

            // HARD: A^(1)(T_2) != -24*I (nonsplit Eisenstein/cuspidal
            // extension -- Deng). diff = A1 - (-24*I) = A1 + 24*I.
            const Matrix neg24I = Matrix(A1.rows, ell) * ModOfInt64(-24, ell);
            const Matrix diff = A1 - neg24I;
            if (diff.is_zero()) {
                Fail("[O6] HARD FAILURE: A^(1)(T_2) == -24*I mod 691 (expected strict "
                     "inequality -- the Eisenstein/cuspidal extension must be nonsplit).");
            } else {
                Matrix diff_mut = diff; // rank() is non-const (caches RREF)
                const uint32_t rk = diff_mut.rank();
                std::cout << "[O6] HARD OK: A^(1)(T_2) != -24*I mod 691. rank(A+24I) = " << rk
                          << " (RECORDED only -- partition (2,1) vs (3) is amendment A-F2 / "
                             "Stage S10V's job, not asserted here)."
                          << std::endl;
            }
        } catch (const std::exception& e) {
            Fail(std::string("O6: EXCEPTION: ") + e.what());
        }
    }

    // O7: n=14, weight 16 (S_16 = C*f_16), ell in {101,10007} (already
    // ell > n+2=16, so the plan-added-extension gate is satisfied
    // automatically): dim H^1=3; charpoly(A^(1)(T_2)) = (X-216)^2(X-32769).
    void RunO7() {
        for (uint32_t ell : {101u, 10007u}) {
            CheckDims(14, ell, 0, 3, 0, "O7");
            const Poly expected =
                PolyMul(PolyLinearFactorPow(216, 2, ell), PolyLinearFactorPow(32769, 1, ell), ell);
            CheckCharpoly(2, ell, 14, 1, expected, "O7");
        }
    }

    // O8: n=22, weight 24 (dim S_24=2, Galois-conjugate pair), ell in
    // {101,10007} (already ell > n+2=24): dim H^1=5;
    // charpoly(A^(1)(T_2)) = (X^2-1080X-20468736)^2*(X-8388609) mod ell.
    // Char-poly identity ONLY; irreducibility of the quadratic factor
    // (Legendre symbol of 144169) is Stage S10V's job.
    void RunO8() {
        for (uint32_t ell : {101u, 10007u}) {
            CheckDims(22, ell, 0, 5, 0, "O8");
            const Poly quad{ModOfInt64(-20468736, ell), ModOfInt64(-1080, ell), 1u};
            const Poly quad_sq = PolyMul(quad, quad, ell);
            const Poly expected = PolyMul(quad_sq, PolyLinearFactorPow(8388609, 1, ell), ell);
            CheckCharpoly(2, ell, 22, 1, expected, "O8");
        }
    }

    // O14: Eichler-Shimura dimension table, ell in {5,7,101} for every
    // listed n (A-F4: no ell>n+2 gate for this invariant class), plus one
    // large-ell anchor (134217757) at n=10. EXCEPT the FINDING-B
    // Dickson-invariant exceptional pairs (n === 0 mod ell+1), recorded
    // only (see file header): (6,5) and (12,5) here.
    void RunO14() {
        struct Row {
            uint32_t n;
            uint32_t dim1; // expected dim H^1 (n=0 is special: H^0=1,H^1=0)
        };
        const std::vector<Row> rows = {
            {0, 0}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 3}, {12, 1}, {14, 3},
        };
        for (const Row& r : rows) {
            for (uint32_t ell : {5u, 7u, 101u}) {
                if (IsDicksonExceptional(r.n, ell)) {
                    RecordDims(r.n, ell, "O14-Dickson");
                    continue;
                }
                const uint32_t expected_h0 = (r.n == 0) ? 1u : 0u;
                CheckDims(r.n, ell, expected_h0, r.dim1, /*h2=*/0, "O14");
            }
        }
        // A-F4: dimension is asserted at EVERY ell >= 5, independent of its
        // size relative to n -- confirm this at a large ell for n=10
        // (134217757 is nowhere near a multiple of n=10, so FINDING B does
        // not apply here).
        CheckDims(10, 134217757u, 0, 3, 0, "O14-largeell");
    }
};

} // namespace

uint32_t HeckeOracleValidation() {
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        OracleContext ctx(K, model);
        ctx.RunAll();

        std::cout << "HeckeOracleValidation: " << ctx.failures() << " failing case(s)." << std::endl;
        return ctx.failures();
    } catch (const std::exception& e) {
        std::cout << "HeckeOracleValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
