// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_BOUNDARY_H
#define HECKE_EXT_BOUNDARY_H

// hecke-extensions slot 2, WP-7 (phase E4b) -- HeckeExt/HeckeBoundary: the
// ext-layer CONSUMER of Hecke/HeckeBoundaryDatum + Hecke/HeckeBoundaryOperator
// ([X] hecke_extensions_implementation_plan.md v1.5.1 Sec. 1.1, Sec. 6.2 E4b;
// hecke_extensions_slot2_plan.md Sec. 3 WP-7). It builds, on one computed
// bulk cohomology H^*_eq(K; Sym^n F_ell^2) of the 7-vertex model
// ([C] = docs/latex/gl2_contract.tex, cited by label):
//
//   * the boundary side: the circle model C^*_bd ([C, prop:circlemodel]),
//     its cohomology X_2 = H^0(bd), X_5 = H^1(bd), and the restriction
//     r = j^* ([C, def:pair]) as the shipped EquivariantCochainComplexMorphismOnK,
//     with r^q as matrices in the stored bases ([C, cl:les](2): X_1 -> X_2
//     is r^0, X_4 -> X_5 is r^1);
//   * for every requested (p, nu): the bulk pipeline HeckeCosetTables ->
//     HeckeTauTable -> HeckeOperatorOnK (A^{(q)}(T_{Gamma g Gamma})) and, on
//     the SAME coset tables, the boundary pipeline HeckeBoundaryTauTable ->
//     HeckeBoundaryOperator (A_bd^{(q)});
//   * THE INTERIOR COHOMOLOGY  H^1_! := ker r^1 = im(X_3 -> X_4)
//     ([C, cl:les](2)) as a basis matrix B (dim H^1 x dim H^1_!, columns) in
//     the stored H^1_eq coordinates -- the RREF kernel basis of r^1 (the D5b
//     basis-choice rule of the substrate: free columns in order, so the basis
//     is a deterministic function of r^1) -- together with its Hecke action
//     A_!(p, nu) := the matrix with A^{(1)} B = B A_!, solved EXACTLY
//     (SolveColumnsExact); H^1_! is T_p-stable because r^1 intertwines
//     ([C, obl:taubdry](vii) / [C, cl:les](1)), a theorem at 6 in R^x and a
//     recorded fact at ell in {2, 3} (where the object records `stable = false`
//     instead of throwing);
//   * THE EISENSTEIN LINE, by the same rule HeckeIotaSignValidation's route
//     (a) uses ([C, obl:iota](v)): the joint T_p-eigenspace E of eigenvalue
//     1 + p^{n+1} over p in {2, 3, 5} \ {ell} (nu = 1), stopping once two
//     primes leave a line; the LINE is defined iff E is a line. E is stored
//     in full: on the Dickson loci it may be larger (WP-7 risk 2), and AT
//     EVERY KUMMER CELL IT IS 2-DIMENSIONAL -- the reduced Eisenstein class
//     and the reduction of the congruent cuspidal eigenline share their
//     T_p-eigenvalues and are not separable mod ell ([E] D6: "at (10, 691)
//     both reduce to the same line"; measured here on all ten cells of [E]
//     Claim 1.8). Recorded, never asserted, plus rank(r^1 E): whether E
//     restricts to 0 on the cusp (route (b)'s boundary-visibility), i.e.
//     whether it lies in H^1_!;
//   * HARDER'S CRITERION ([E] hecke_extensions.tex cl:harder, Claim 1.8):
//     for n < ell - 1, r^1 = 0 on the Eisenstein class iff ell | num(B_k/k),
//     k = n + 2. The Bernoulli side comes from a PINNED table of num(B_k/k)
//     for even k <= 26 (the range of [E]'s Kummer list; bernoulli_mod stays
//     phase E6), so `harder_applicable` = (n < ell - 1, k <= 26, E != 0) and
//     the verdict is decided on E as the certificate's table says: E a line
//     -> the line decides; E larger and r^1 E = 0 -> r^1 = 0 on the
//     Eisenstein class; E larger and r^1 injective on E -> r^1 != 0 on it;
//     otherwise INCONCLUSIVE mod ell. RECORDED here; ASSERTED "not refuted"
//     by HeckeBoundaryParabolicValidation on every applicable cell with
//     ell >= 5 (the ten Kummer cells of [E] Claim 1.8 included, where the
//     measured verdict is `consistent` through r^1 E = 0) and RECORDED at
//     (10, 5), (10, 7) ([XA m5]);
//   * THE HARDER SPLIT (the Eisenstein / cuspidal decomposition of H^1 as a
//     Hecke module, [E] pr:eiscusp / D6): dim H^1 = dim H^1_! + rank r^1
//     (certified), the induced action of A^{(1)} on the quotient H^1 / H^1_!
//     being conjugate through r^1 to A_bd^{(1)} on im r^1 -- recorded as the
//     char-poly identity charpoly(A^{(1)}) = charpoly(A_!) * charpoly(A_bd|im
//     r^1) per (p, nu) and, where the Eisenstein line is defined and NOT in
//     H^1_!, the direct sum H^1 = H^1_! (+) Eis (rank certificate;
//     `harder_split_direct` = it exhausts H^1, i.e. rank r^1 = 1);
//   * THE TWO DIAGNOSTICS of Hecke/HeckeBoundaryDiagnostics per (p, nu):
//     cochain_commutes and the cusp-support walk (eta_zero) -- stored,
//     written into to_json(), never branched on ([C, rmk:conestrict]);
//   * THE GOLDEN COLLECTORS ([X] Sec. 6.2 E4b: "defined here, emitted with
//     E2's add-only ext-subgrid emission"): golden_objects(p, nu) returns the
//     (object, tier, rows, cols, hash) records dim_H0_bd, dim_H1_bd, rank_r^0,
//     rank_r^1, charpoly_A_bd^0, charpoly_A_bd^1, chi_multiset (tier A) and
//     r^0, r^1, A_bd^0, A_bd^1 (tier B), plus the E4b additions
//     dim_H1_interior, eis_joint_dim, eis_rank_r1 (tier A) and interior_basis (tier B),
//     hashed EXACTLY as Golden/GoldenHarness.cpp hashes a scalar / a MatrixT /
//     an FpPolyT (engine::FnvU64 over rows, cols, entries row-major; count
//     then coefficients), so E2 can emit them without re-deriving a format.
//     NOTHING here writes GoldenReference.inc.
//
// CERTIFICATE CONVENTION (HeckeExt/HeckeIdentityCertificate.h): certificate()
// returns a plain, copyable HeckeBoundaryCertificate holding exactly the
// stored matrices a third party needs, whose verify() RE-DERIVES every claim
// above from those matrices alone (rank / kernel / solve recomputed, never by
// calling back into the pipeline); HeckeBoundaryParabolicValidation tampers a
// copy and shows verify() rejects it.
//
// Record fields of [C, sec:recordschema] item 5 as decided in slot-2 owner
// confirmation 1 (per COMPLEX, never a maximum): to_json() carries, inside
// the "bulk" and "boundary" objects, q_max / truncated / q_max_source /
// degrees -- bulk 2 / false / "derived" / [0,1,2], boundary 1 / false /
// "derived" / [0,1] on the model complexes -- and the boundary operator's
// boundary_model / boundary_is_group_cohomology flags. Schema string
// "hecke_boundary_parabolic/v1"; the boundary complex reaches the
// extension record at hecke_ext/v2 (slot-5 WP-1), through the cone.
//
// Placement ([X] Sec. 1.1, Sec. 2.3): HeckeExt/, MatrixT<uint32_t> for the
// stored level-(b) objects (the plan's "MatrixT<uint32_t> for everything in
// HeckeExt/" decision -- docs/matrix-engine/tools/check_matrix_includes.sh
// carries this file with that reason). Consumes Hecke/ one-way; no Engine/,
// Cuda/, Console/ or BlockFlatMatrix.h include (rule 2 of the HeckeExt/
// guard); no unordered_ container (rule 4); no raw mod_* call (ModCtx).
// The interfaces WP-8 (the eta-table) builds on: pipeline(p, nu) -- the
// cosets, the bulk tau, tau_bd and both operators on one coset table --,
// restriction(), boundary_cohomology(), K_bd(), and the walk.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeBoundaryDiagnostics.h"
#include "../Hecke/HeckeBoundaryOperator.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/SL2ZBoundaryPair.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "HeckeExtGoldenObject.h"

namespace hecke_ext {

// --- the pinned Bernoulli numerators ([E] cl:harder) ------------------------
// num(B_k/k) for even 2 <= k <= 26 (sign included; the value is what an
// exact Fraction computation of B_k / k gives, checked against [E] Claim
// 1.8's factorisations: 691; 3617; 43867; 174611 = 283 * 617; 77683 = 131 *
// 593; 236364091 = 103 * 2294797; 657931). `known` = false outside the range.
int64_t pinned_bernoulli_numerator(uint32_t k, bool& known);
// ell | num(B_k/k), from the pinned table; `known` as above.
bool pinned_kummer_divides(uint32_t k, uint32_t ell, bool& known);

// --- one golden record ------------------------------------------------------
// (slot 5, WP-3) Moved VERBATIM to HeckeExt/HeckeExtGoldenObject.h, included
// above: HeckeSignSplit.h carried a second, identical definition, and
// Golden/GoldenHarness.cpp is the first TU to include both.  No field, no
// default and no hash changed.

// --- the certificate ----------------------------------------------------------
struct HeckeBoundaryCertificate {
    uint32_t ell = 0, n = 0;
    uint32_t dim_H0_K = 0, dim_H1_K = 0, dim_H0_bd = 0, dim_H1_bd = 0;
    Matrix r0, r1;                 // dim H^q_bd x dim H^q_K, the stored restriction maps
    Matrix interior_basis;         // dim H^1_K x dim H^1_!  (columns), the RREF kernel basis of r1
    struct Op {
        uint64_t p = 0;
        uint32_t nu = 0;
        Matrix A1;                 // A^{(1)}(T_{Gamma g Gamma}) on H^1_K
        Matrix A_bd1;              // A_bd^{(1)} on H^1_bd
        Matrix A_interior;         // dim H^1_! square; A1 * B == B * A_interior
        bool stable = false;       // the solve succeeded
    };
    std::vector<Op> ops;
    // The joint Eisenstein eigenspace E (route (a) of obl:iota(v)): the joint
    // kernel of (A_p - (1 + p^{n+1})) over the stacked primes. Off the Kummer
    // and Dickson loci it is the Eisenstein LINE; at a Kummer cell it is
    // larger, because the reduced Eisenstein class and the reduction of the
    // congruent cuspidal eigenline are not separable mod ell ([E] D6; at
    // (10, 691) both reduce to the same joint eigenspace) -- the object then
    // stores the whole space, and eisenstein_defined is false.
    std::vector<uint64_t> eisenstein_primes; // the primes stacked, in order
    uint32_t joint_eigenspace_dim = 0;      // j = #columns of eisenstein_space (dim H^1 if no prime available)
    Matrix eisenstein_space;                 // dim H^1_K x j, the RREF kernel basis of the stacked matrix
    bool eisenstein_defined = false;         // j == 1
    Matrix eisenstein_vector;                // the one column when defined (empty otherwise)
    uint32_t r1_rank_on_joint_space = 0;     // rank(r1 * E)
    bool eisenstein_in_interior = false;     // r1 * E == 0: r^1 vanishes on the WHOLE joint eigenspace
    // Harder ([E] cl:harder): for n < ell - 1, r^1 = 0 on the Eisenstein class
    // iff ell | num(B_k/k). What is decidable mod ell on E:
    //   j == 1        -> E is the Eisenstein class: consistent iff kummer == (r1 E == 0);
    //   j >= 2, r1 E == 0 -> r^1 vanishes on the Eisenstein class: consistent iff kummer;
    //   j >= 2, r1 injective on E -> r^1 != 0 on the Eisenstein class: consistent iff !kummer;
    //   j >= 2, 0 < rank r1 E < j -> INCONCLUSIVE (the line is not separable mod ell).
    enum class HarderVerdict : uint32_t { kNotApplicable = 0, kConsistent = 1, kRefuted = 2, kInconclusive = 3 };
    bool harder_applicable = false;          // n < ell - 1, k <= 26 (pinned), j >= 1
    bool kummer = false;                     // ell | num(B_k/k)
    HarderVerdict harder_verdict = HarderVerdict::kNotApplicable;
    // The direct-sum claim H^1 = H^1_! (+) Eis (only when the line is defined and r1 v != 0).
    bool harder_split_direct = false;        // dim H^1_! + 1 == dim H^1_K

    // 0 iff every item holds, re-derived from the stored matrices (an item
    // whose data is misshaped fails, never throws):
    //   1  r0 injective (rank r0 == dim H^0_K)                      [C, lem:conerelative](4)
    //   2  r1 * B == 0, rank B == #columns, #columns + rank r1 == dim H^1_K
    //   3  per op: stable && A1 * B == B * A_interior (an unstable op fails
    //      the item unless ell in {2, 3}, where stability is recorded only)
    //   4  the joint eigenspace: (A_p - (1 + p^{n+1})) E == 0 for every stacked
    //      prime, rank E == j == the stacked kernel's dimension (recomputed),
    //      eisenstein_defined == (j == 1) with eisenstein_vector == E then,
    //      r1_rank_on_joint_space == rank(r1 E), eisenstein_in_interior == (that rank == 0)
    //   5  Harder: if applicable -- kummer == the pinned divisibility and
    //      harder_verdict == the verdict recomputed by the table above (never kRefuted)
    //   6  the direct sum: if defined and r1 v != 0 -- rank [B | v] == #B + 1,
    //      and harder_split_direct == (#B + 1 == dim H^1_K)
    uint32_t verify() const;
    // The same, with one line per failing item.
    uint32_t verify(std::vector<std::string>& failures) const;
};

// --- the object ----------------------------------------------------------------
struct HeckeBoundaryOptions {
    // The (p, nu) pipelines to build. The Eisenstein line uses the nu = 1
    // members with p in {2, 3, 5} \ {ell}, in this order.
    std::vector<std::pair<uint64_t, uint32_t>> operators = { { 2, 1 }, { 3, 1 }, { 5, 1 } };
    bool run_diagnostics = true;   // cochain_commutes + the cusp-support walk per pipeline
};

class HeckeBoundary {
public:
    // One (p, nu) pipeline pair on one coset table.
    struct Pipeline {
        uint64_t p = 0;
        uint32_t nu = 0;
        std::unique_ptr<hecke::HeckeCosetTables> cosets;
        std::unique_ptr<hecke::HeckeTauTable> tau;
        std::unique_ptr<hecke::HeckeOperatorOnK> op;
        std::unique_ptr<hecke::HeckeBoundaryTauTable> taub;
        std::unique_ptr<hecke::HeckeBoundaryOperator> bd_op;
        // level (b), copied out of the two operators
        Matrix A_bulk[3];              // A^{(q)}, q = 0, 1, 2
        Matrix A_bd[2];                // A_bd^{(q)}, q = 0, 1
        // the interior action and the split
        Matrix A_interior;             // dim H^1_! square (empty when !interior_stable)
        bool interior_stable = false;
        Matrix A_quotient_bd;          // A_bd^{(1)} restricted to im r^1 (rank r^1 square; empty if not stable)
        bool quotient_stable = false;
        hecke::FpPoly charpoly_A1, charpoly_interior, charpoly_quotient;
        bool charpoly_product_holds = false;   // charpoly_A1 == charpoly_interior * charpoly_quotient
        std::vector<uint32_t> chi;             // chi_b(p), b = 0..floor(n/ell) (nu = 1, p != ell; empty otherwise)
        bool quotient_charpoly_divides_chi = false; // charpoly_quotient | prod_b (X - chi_b)  (recorded)
        // the diagnostics
        bool diagnostics_run = false;
        hecke::CochainCommutes cochain_commutes;
        hecke::CuspSupportWalk walk;
    };

    // coh_K: the COMPUTED bulk cohomology on K (the 7-vertex model), with
    // prime ell and representation dimension n + 1 (checked); K and model
    // must outlive this object. Nothing is computed before compute().
    HeckeBoundary(const EquivariantCohomologyOnK* coh_K, const SimplicialComplex& K,
                  const hecke::SL2ZTessellationTables& model, HeckeBoundaryOptions options = HeckeBoundaryOptions());
    ~HeckeBoundary();
    HeckeBoundary(const HeckeBoundary&) = delete;
    HeckeBoundary& operator=(const HeckeBoundary&) = delete;

    // Builds everything of the header note. Throws std::runtime_error on an
    // inconsistency that is a theorem at 6 in R^x (an unstable H^1_! at
    // ell >= 5, a failing certificate); records instead at ell in {2, 3}.
    // Idempotent.
    void compute();
    bool computed() const { return computed_; }

    // --- parameters ---
    uint32_t ell() const { return ell_; }
    uint32_t n_sym() const { return n_; }
    uint32_t d_V() const { return n_ + 1; }

    // --- the boundary side (the interfaces WP-8 builds on) ---
    const SimplicialComplex& K() const { return *K_; }
    const SimplicialComplex& K_bd() const;
    // The pipeline-side pair itself (slot-3 WP-5), for the cone's consumers.
    const hecke::SL2ZBoundaryPair& pair() const;
    const EquivariantCohomologyOnK& bulk_cohomology() const { return *coh_K_; }
    const EquivariantCohomologyOnK& boundary_cohomology() const;
    const EquivariantCochainComplexMorphismOnK& restriction() const;
    const Matrix& r(uint32_t q) const;          // q in {0, 1}
    uint32_t rank_r(uint32_t q) const;
    uint32_t dim_H_bulk(uint32_t q) const;
    uint32_t dim_H_bd(uint32_t q) const;

    // --- the pipelines ---
    size_t num_pipelines() const { return pipelines_.size(); }
    const Pipeline& pipeline(size_t i) const;
    const Pipeline& pipeline(uint64_t p, uint32_t nu) const;  // throws if absent
    bool has_pipeline(uint64_t p, uint32_t nu) const;

    // --- H^1_! ---
    const Matrix& interior_basis() const;       // dim H^1 x dim H^1_!
    uint32_t dim_interior() const;
    const Matrix& interior_operator(uint64_t p, uint32_t nu) const; // throws if not stable
    bool interior_stable(uint64_t p, uint32_t nu) const;

    // --- the Eisenstein line / Harder / the split (see the certificate) ---
    bool eisenstein_defined() const { return cert_.eisenstein_defined; }
    uint32_t joint_eigenspace_dim() const { return cert_.joint_eigenspace_dim; }
    const std::vector<uint64_t>& eisenstein_primes() const { return cert_.eisenstein_primes; }
    const Matrix& eisenstein_space() const { return cert_.eisenstein_space; }
    const Matrix& eisenstein_vector() const { return cert_.eisenstein_vector; }
    bool eisenstein_in_interior() const { return cert_.eisenstein_in_interior; }
    // rank(r^1 E) on the joint eigenspace E (0 when E = 0).
    uint32_t eisenstein_rank_r1() const { return cert_.r1_rank_on_joint_space; }
    bool harder_applicable() const { return cert_.harder_applicable; }
    bool kummer() const { return cert_.kummer; }
    int64_t bernoulli_numerator() const { return bernoulli_num_; }
    bool bernoulli_known() const { return bernoulli_known_; }
    HeckeBoundaryCertificate::HarderVerdict harder_verdict() const { return cert_.harder_verdict; }
    bool harder_consistent() const { return cert_.harder_verdict == HeckeBoundaryCertificate::HarderVerdict::kConsistent; }
    static const char* harder_verdict_name(HeckeBoundaryCertificate::HarderVerdict v);
    bool harder_split_direct() const { return cert_.harder_split_direct; }

    // --- the certificate (by value; verify() re-derives everything) ---
    const HeckeBoundaryCertificate& certificate() const;

    // --- the golden collectors (never emitted here) ---
    std::vector<GoldenObject> golden_objects(uint64_t p, uint32_t nu) const;
    // FNV-1a-64 over every record of golden_objects(p, nu) for every pipeline,
    // in pipeline order: one digest per object, the validation's stability handle.
    uint64_t golden_digest() const;

    std::string to_string() const { return to_string_offset(0); }
    std::string to_string_offset(uint32_t offset) const;
    // Hand-rolled JSON, schema "hecke_boundary_parabolic/v1".
    std::string to_json() const;

private:
    const EquivariantCohomologyOnK* coh_K_;
    const SimplicialComplex* K_;
    const hecke::SL2ZTessellationTables* model_;
    HeckeBoundaryOptions options_;
    uint32_t ell_ = 0, n_ = 0;
    bool computed_ = false;

    // (hecke-extensions slot 3, WP-5) The boundary side -- K_bd, the circle
    // model, its cohomology, the inclusion and the restriction morphism -- is
    // the pipeline-side hecke::SL2ZBoundaryPair (lifted verbatim out of the
    // former build_boundary_side; value-preserving), consumed one-way.
    std::unique_ptr<hecke::SL2ZBoundaryPair> pair_;
    Matrix r_[2];
    uint32_t rank_r_[2] = { 0, 0 };

    std::vector<std::unique_ptr<Pipeline>> pipelines_;
    HeckeBoundaryCertificate cert_;
    int64_t bernoulli_num_ = 0;
    bool bernoulli_known_ = false;

    void require_computed(const char* who) const;
    void build_boundary_side();
    void build_pipeline(uint64_t p, uint32_t nu);
    void build_interior();
    void build_eisenstein_and_harder();
    void build_split(Pipeline& pl);
    void run_diagnostics(Pipeline& pl);
};

} // namespace hecke_ext

#endif // HECKE_EXT_BOUNDARY_H
