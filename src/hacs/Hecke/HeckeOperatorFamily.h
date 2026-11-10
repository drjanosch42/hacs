// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_OPERATOR_FAMILY_H
#define HECKE_OPERATOR_FAMILY_H

// Stage S9 -- HeckeOperatorFamily: the Hecke ALGEBRA structure built on top
// of Stage S7's single-double-coset pipeline (HeckeOperatorOnK). The STANDARD
// FULL Hecke operator T_{p^r} is computed two independent ways that agree:
//   * T_p_pow(p, r): by the algebra recursion of [C, thm:relations]
//     (primary; validated -- reproduces a_{p^2} = sigma_{k-1}(p^2));
//   * T_p_pow_direct(p, r): by the classical double-coset decomposition of
//     the full determinant-p^r operator into PRIMITIVE double cosets (see
//     the T_p_pow_direct / T_p_pow_primitive_direct block below).
// The PRIMITIVE double-coset operator T'_{p^r} of Gamma diag(1,p^r) Gamma
// (what HeckeCosetTables(p, r) directly computes) is a DIFFERENT object for
// r >= 2 and is exposed separately as T_p_pow_primitive_direct(p, r). Also:
// T_n (multiplicativity over the prime factorization), T(p,p) (the
// central/diamond scalar, [C, prop:diamond]), and U_ell (the bad-prime case
// p == ell, [C, cor:ulrecursion]) -- which is NOT new math (invariant I7):
// U_ell is literally T_p evaluated at p == ell, the SAME pipeline, the SAME
// code path. Nothing in this class special-cases p == ell.
//
// Contract: docs/latex/hecke_action.tex ([C]):
//   * [C, thm:relations] (the recursion):
//       T_{p^0} = I,   T_{p^1} = T_p,
//       T_{p^{r+1}} = T_p . T_{p^r}  -  p^{n+1} . T_{p^{r-1}}   (r >= 1),
//     which is contract-backed ONLY under hypothesis (H3), 6 invertible in
//     the coefficient ring -- for R of characteristic ell, only at
//     ell >= 5. The hypothesis is on the COEFFICIENT prime ell, NOT on p:
//     it is required for EVERY Hecke prime p, the genuinely good ones
//     (ell does not divide p) exactly as much as the bad case p == ell
//     (U_ell), because [C, thm:relations] is proved through
//     [C, obl:algebra] (the algebra-homomorphism property) and through
//     [C, thm:classical], both of which assume 6 in R^x. At ell in {2,3}
//     the recursion is RECORDED, not asserted -- plan amendment A-F1
//     (docs/implementation/hecke_action_implementation_plan.md) and the
//     S9V gating described in HeckeRelationsValidation.h (oracle O10).
//     This class does not gate on ell -- it always runs the (cheap)
//     recursion; the ell in {2,3} restriction is a VALIDATION concern
//     (S9V), not something this class enforces.
//   * [C, prop:diamond]: T(p,p) = p^n . I on every H^q (p not divisible by
//     ell), a pure scalar formula -- no pipeline run needed.
//   * [C, cor:ulrecursion]: at p == ell the recursion above degenerates to
//     U_ell's recursion automatically, because p^{n+1} mod ell == 0
//     identically -- SAME CODE, no branch (invariant I7 / [C,
//     obl:integral]).
//
// MULTIPLICATION ORDER (binding for bit-reproducibility even though the
// operators commute, per the plan's S9 risk list): every product below is
// formed in a FIXED left-to-right order --
//   * the recursion multiplies T_p on the LEFT: T_{p^{k+1}} = T_p * T_{p^k}
//     - scalar * T_{p^{k-1}} (never T_{p^k} * T_p);
//   * T_n's prime-power factors are multiplied in ASCENDING prime order
//     (the natural order of trial-division discovery), accumulating on the
//     LEFT: result = result * factor.
//
// CACHING: T_p(p) runs the full pipeline (HeckeCosetTables(p,1,ell,n_sym)
// -> HeckeTauTable -> HeckeOperatorOnK::compute()) at most ONCE per prime p
// and returns a STABLE reference into this object (owned for the object's
// lifetime). T_p_pow_direct(p,r) shares the SAME internal pipeline cache,
// keyed by (p,r) (so T_p_pow_direct(p,1) reuses T_p(p)'s pipeline run
// verbatim), but returns BY VALUE (a copy of get_H_matrices()) per the
// declared interface -- do not assume reference stability for it.
// T_p_pow(p,r) (the recursive route) triggers exactly one pipeline run (for
// T_p(p) itself, via the shared cache) and is otherwise pure per-degree
// linear algebra on already-computed matrices -- cheap to recompute, so it
// is not separately cached.
//
// Placement (documented per the task's instructions): like S7's
// HeckeOperatorOnK (whose pipeline this class assembles repeatedly for
// different Hecke primes/powers), this class consumes ModArithmetic
// (Matrix) and the model/cohomology/tessellation layer, so it is compiled
// into the cohomology_app executable sources (NOT into the pure-integer
// HeckeCore static library). See CMakeLists.txt.

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
// (matrix-engine Phase 3) Matrix.h is no longer included directly. The Matrix
// in the signatures below is the level-(b) A^{(q)} (d_q x d_q, d_q <= ~20) --
// HeckeModuleDecomposition's input type, which the plan deliberately keeps on
// MatrixT -- and it arrives through HeckeOperatorOnK.h -> FlatMatrix.h.

class SimplicialComplex;

namespace hecke {

// (matrix-engine Phase 6) The ONE process-wide lock that serialises every
// HeckeCosetTables / HeckeTauTable CONSTRUCTION in the process.
//
// tau construction is first-match-wins over a fixed transporter order and
// drives the NON-ATOMIC global PredicateStats counters whose exact values
// HeckeFpFilterValidation asserts (audit 02 Sec. 6.5, hazards 1-3), so two
// constructions must never overlap.  HeckeOperatorFamily::build_pipelines()
// has always held it for its own phase A; it is declared here so that any
// other concurrent driver -- notably the Phase 6 campaign, which runs a shape
// group's jobs (same n_sym / p / r, one ell each) on separate threads -- can
// hold the SAME lock around its own coset+tau construction instead of
// introducing a second, independent one.  Only compute() runs concurrently.
// (slot-9 WP-7, lever L2) build_pipelines() itself now holds it only around
// priming K's simplex index; its own constructions run outside it (the
// hazards above are intra-construction or gone since slot-8 WP-6 -- see
// HeckeOperatorFamily.cpp).  The other holders are unchanged.
std::mutex& tau_construction_mutex();

class HeckeOperatorFamily {
public:
    // cohomL: the COMPUTED base system L = Sym^n_sym(F_ell^2) on K
    // (compute_cohomology() already called; checked). K: the 7-vertex
    // model (same object HeckeOperatorOnK's constructor validates the shape
    // of). model: the S2 tessellation tables built from K. ell/n_sym must
    // match cohomL's (prime, representation_dim - 1); checked, throws
    // std::runtime_error otherwise. cohomL, K and model must outlive this
    // object and every reference/value it returns.
    HeckeOperatorFamily(const EquivariantCohomologyOnK* cohomL, const SimplicialComplex& K,
                        const SL2ZTessellationTables& model, uint32_t ell, uint32_t n_sym);

    // T_p: the prime Hecke operator (this IS U_ell when p == ell -- same
    // code path, invariant I7; no special-casing). Cached: the first call
    // for a given p builds and owns HeckeCosetTables(p,1,ell,n_sym) ->
    // HeckeTauTable(...) -> HeckeOperatorOnK::compute() inside this object;
    // later calls with the same p return the SAME reference in O(1).
    // d_q x d_q per degree q; 0x0 for an empty degree.
    const std::vector<Matrix>& T_p(uint64_t p);

    // T_{p^r} via the algebra recursion of [C, thm:relations]:
    //   T_{p^0} = I     (d_q x d_q identity per degree; 0x0 stays 0x0),
    //   T_{p^1} = T_p(p),
    //   T_{p^{r+1}} = T_p(p) * T_{p^r}  -  (p^{n_sym+1} mod ell) * T_{p^{r-1}}.
    // Fixed left-to-right order (T_p(p) on the left of every product) for
    // bit-reproducibility (the risk list of the S9 stage; the operators
    // commute where [C, obl:algebra] applies, i.e. under (H3), 6 in R^x --
    // see the hypothesis note in the file header). For p == ell this degenerates
    // automatically to U_ell's recursion ([C, cor:ulrecursion]) since
    // p^{n_sym+1} mod ell == 0 -- no branch, no special case.
    std::vector<Matrix> T_p_pow(uint64_t p, uint32_t r);

    // PRIMITIVE-vs-FULL Hecke operators at prime powers (binding distinction;
    // see the block comment below). The pipeline of Stages S3/S4/S7 built on
    // HeckeCosetTables(p, r, ...) computes the operator of the PRIMITIVE
    // double coset Gamma diag(1, p^r) Gamma, whose reps are the P^1(Z/p^r)
    // classes of [C, sec:heckedatum]. For r == 1 that primitive operator IS
    // the standard T_p; but for r >= 2 the standard (full) Hecke operator
    // T_{p^r} is NOT the primitive double-coset operator -- it is the sum
    // over the double cosets of determinant p^r,
    //     T_{p^r}^full = sum_{j=0}^{floor(r/2)} p^{j n} . T'_{p^{r-2j}},
    // where T'_{p^m} is the PRIMITIVE double-coset operator (T'_{p^0} := I,
    // T'_{p^1} = T_p) and the scalar p^{j n} is the j-th power of the
    // central/diamond character T(p,p) = p^n I ([C, prop:diamond]). This is
    // the classical decomposition of the full determinant-p^r Hecke operator
    // into primitive double cosets (Shimura, Intro. to the Arithmetic Theory
    // of Automorphic Functions, Thm 3.34 / the T(1,p^r) decomposition); it
    // is precisely what makes T_{p^r}^full satisfy the algebra recursion
    // [C, thm:relations] realized by T_p_pow above (so T_p_pow == this method
    // for every r -- asserted by the S9 smoke and by S9V's O9/O10).
    //
    // CONTRACT CLARIFICATION NOTE (tracked separately by the orchestrator):
    // [C, sec:heckedatum]'s P^1(Z/p^r) representatives describe the PRIMITIVE
    // coset Gamma diag(1,p^r) Gamma; the plan's oracle O9 wrongly equated the
    // r >= 2 primitive operator with the standard T_{p^r} (they differ by
    // exactly the p^n . I offset above, empirically measured during S9). The
    // user (contract author) has RULED that the deliverable's T_{p^r} is the
    // STANDARD FULL operator (= T_p_pow's recursion, validated by
    // a_{p^2} = sigma_{k-1}(p^2)); [C, thm:relations] gives that full
    // operator. The primitive object remains available via
    // T_p_pow_primitive_direct() as a legitimate secondary datum.

    // The INDEPENDENT full-operator route (what O9/O10 compares the recursion
    // T_p_pow above against): assembles T_{p^r}^full from the primitive
    // double-coset operators by the decomposition formula above. r == 0
    // returns the per-degree identity. r == 1 returns T_p(p) (primitive ==
    // full for r == 1). For r >= 2 it sums p^{j n} . T'_{p^{r-2j}} over
    // j = 0 .. floor(r/2), with the primitive terms computed by
    // T_p_pow_primitive_direct() (which shares T_p(p)'s / the (p,m) pipeline
    // cache). Fixed summation order (ascending j) and scalar p^{j n} via
    // mod_pow, for bit-reproducibility. Returns BY VALUE, per the declared
    // interface. By construction this MATCHES T_p_pow(p, r) exactly.
    std::vector<Matrix> T_p_pow_direct(uint64_t p, uint32_t r);

    // The raw PRIMITIVE double-coset operator T'_{p^r} of the double coset
    // Gamma diag(1, p^r) Gamma: runs the pipeline DIRECTLY via
    // HeckeCosetTables(p, r, ell, n_sym) (m = p^r + p^{r-1} cosets) ->
    // HeckeTauTable(...) -> HeckeOperatorOnK::compute(). For r == 1 this IS
    // T_p (primitive == full). For r >= 2 it is NOT the standard T_{p^r}
    // (see the block above): it is the primitive double-coset operator, a
    // legitimate secondary object exposed for study / cross-checks. r == 0
    // returns the per-degree identity directly (bypasses the pipeline --
    // HeckeCosetTables requires r >= 1). r >= 1 shares the (p, r) pipeline
    // cache with T_p(p) (r == 1) and with T_p_pow_direct's primitive terms.
    // Returns BY VALUE.
    std::vector<Matrix> T_p_pow_primitive_direct(uint64_t p, uint32_t r);

    // T_n = product of T_{p_i^{r_i}} over the prime factorization of n
    // (trial division; n < 2^32 assumed). n == 0 is not a Hecke index:
    // throws std::invalid_argument. n == 1 is the empty product: returns
    // T_{p^0} = I per degree. Factors are multiplied via T_p_pow (the
    // recursive route, not T_p_pow_direct) in ASCENDING prime order, a
    // fixed order for bit-reproducibility (the factors commute where
    // [C, obl:algebra] applies, i.e. under (H3), 6 in R^x -- see the
    // hypothesis note in the file header).
    std::vector<Matrix> T_n(uint64_t n);

    // T(p,p) = p^n_sym * I per degree ([C, prop:diamond]); a pure scalar
    // computation (mod_pow(p mod ell, n_sym, ell)) -- no pipeline run. (The
    // plan permits an optional pipeline cross-check "if cheap"; this stage
    // does not add one -- see the .cpp file and the stage report for the
    // rationale: the double coset Gamma diag(p,p) Gamma is the CENTRAL one,
    // not of HeckeCosetTables' supported g = diag(1,N) shape, so a genuine
    // pipeline cross-check would need new coset-table code, out of scope
    // for this stage's "no new math" mandate; the formula itself is
    // contract-pinned, [C, prop:diamond].)
    std::vector<Matrix> T_pp(uint64_t p);

    // Level-(a) passthrough of get_Theta_flat_matrices()[q] for T_p(p)'s
    // r=1 pipeline (building it first via the shared cache if this is the
    // first reference to prime p). NOTE ON THE RETURN TYPE (documented,
    // see the .cpp file and the stage report for the full rationale): the
    // task's / plan's declared signature is
    // `std::vector<Matrix> theta_flat(uint64_t p, uint32_t q) const`, even
    // though HeckeOperatorOnK::get_Theta_flat_matrices() returns a single
    // FlatMatrix per degree q (not a collection indexed further by q this
    // method already selects). Implemented EXACTLY to the literal
    // signature: returns a size-1 vector containing
    // get_Theta_flat_matrices()[q].to_matrix().
    std::vector<Matrix> theta_flat(uint64_t p, uint32_t q) const;

    // Item 7.6 (matrix-engine Phase 1): the same level-(a) passthrough
    // WITHOUT the dim_q^2 Matrix copy that the plan-mandated signature above
    // forces (to_matrix() rebuilds the whole (N_q*d_V)^2 matrix as a
    // vector-of-vectors -- 1.2 M entries at n_sym = 100).  theta_flat(p, q)
    // is kept exactly as specified (plan Q6) and is unchanged; new callers
    // should use this accessor instead.
    // Returns a reference into the family's pipeline cache for (p, r = 1); it
    // stays valid for the lifetime of this HeckeOperatorFamily (cache entries
    // are only ever added, never replaced or erased).
    const FlatMatrix& theta_flat_fm(uint64_t p, uint32_t q) const;

    // (matrix-engine Phase 4, parallel axis (d)) Build, CONCURRENTLY, every
    // pipeline that T_p(p), T_p_pow(p, r) and T_p_pow_direct(p, r) will need:
    // (p, 1) and the primitive terms (p, r - 2j), 0 <= 2j < r.  Each missing
    // (p, m) is one full pipeline (HeckeCosetTables -> HeckeTauTable ->
    // HeckeOperatorOnK::compute()); the coset and tau-table CONSTRUCTIONS run
    // serially in ascending (p, m) order under one process-wide mutex (tau
    // construction is first-match-wins over a fixed order and drives the
    // non-atomic PredicateStats counters -- audit 02 Sec. 6.5 hazards 1-3 --
    // so it is never parallelised), and the compute() calls -- the heavy part
    // -- run as one host task each on the engine pool.  Results are inserted
    // into the cache in key order; every later call is an O(1) hit.  Values
    // are identical to the sequential build (every pipeline is a pure
    // function of its (p, m)); only the wall clock changes.  T_p_pow_direct
    // itself builds its missing primitive terms the same way, so calling
    // prepare() is an optimisation, never a requirement.
    void prepare(uint64_t p, uint32_t r);

    // (slot-24 G6) Memory only: every pipeline cached so far drops its level
    // (a) (HeckeOperatorOnK::release_level_a: Theta_block / Theta_eq /
    // Theta_flat and the fused blocks) and its coset and tau tables, and keeps
    // its operator's A^{(q)} and certificate counters.  T_p(p) of a released
    // prime still returns the same reference (the same object) in O(1), and
    // so do T_p_pow / T_p_pow_direct / T_n built on it; theta_flat /
    // theta_flat_fm of a released prime throw std::logic_error.  Pipelines
    // built later are not released until the next call.  A HeckeExt cell
    // calls it once Stage 1 has copied its operators (HeckeExtDriver, the
    // HECKE_FIELD_DIET lever).  Returns the number of pipelines released by
    // this call.
    uint32_t release_level_a();

private:
    const EquivariantCohomologyOnK* cohomL_;
    const SimplicialComplex* K_;
    const SL2ZTessellationTables* model_;
    uint32_t ell_ = 0;
    uint32_t n_sym_ = 0;
    uint32_t num_degrees_ = 0; // == get_H_eq_dimensions().size()

    struct PipelineEntry {
        std::unique_ptr<HeckeCosetTables> cosets;
        std::unique_ptr<HeckeTauTable> tau;
        std::unique_ptr<HeckeOperatorOnK> op;
    };

    // Keyed by (p_hecke, r); shared by T_p(p) (r=1 entries) and
    // T_p_pow_direct(p,r) (any r >= 1). std::map guarantees pointer/
    // reference stability of EXISTING elements across further insertions
    // (no rehashing/relocation, unlike vector/unordered_map) -- required
    // here since HeckeTauTable and HeckeOperatorOnK store raw pointers into
    // their dependencies (the HeckeCosetTables / HeckeTauTable owned
    // alongside them in the same entry). Declared `mutable` so the const
    // theta_flat() can build-on-demand via the same cache.
    mutable std::map<std::pair<uint64_t, uint32_t>, PipelineEntry> pipeline_cache_;
    // (Phase 4) The map stays a std::map (02 Sec. 6.5(5): no hash containers
    // on a result path) and is only ever touched under this mutex, so that
    // concurrently built pipelines (axis (d)) and concurrent readers agree.
    mutable std::mutex cache_mutex_;

    // Builds (if absent) and returns the pipeline entry for (p, r); r must
    // be >= 1 (HeckeCosetTables' own precondition). Declared const (the
    // cache is `mutable`) so both the non-const T_p/T_p_pow_direct and the
    // const theta_flat can share it.
    PipelineEntry& get_or_build_pipeline(uint64_t p, uint32_t r) const;
    // Builds every listed (p, r) that is not cached yet (axis (d): tau
    // construction serial in key order, compute() concurrent) and inserts
    // them in key order.  Keys must have r >= 1.
    void build_pipelines(std::vector<std::pair<uint64_t, uint32_t>> keys) const;

    // scalar p^{n_sym+1} mod ell via ModArithmetic::mod_pow, the recursion's
    // Eisenstein-type coefficient ([C, thm:relations]).
    uint32_t recursion_scalar(uint64_t p) const;
};

} // namespace hecke

#endif // HECKE_OPERATOR_FAMILY_H
