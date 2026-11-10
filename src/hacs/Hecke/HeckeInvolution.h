// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_INVOLUTION_H
#define HECKE_INVOLUTION_H

// ===========================================================================
// The PRODUCTION involution iota^* -- [C, sec:involution], phase E3.
// (hecke-extensions slot 4, WP-3; docs/implementation/slot4/WP-3_report.md)
// ===========================================================================
//
// WHAT THIS IS, AND WHAT IT IS NOT.  Until slot 4 the involution existed only
// inside two validations ([C, sec:codemap]: "Both are validations: no
// PRODUCTION iota^* exists").  This file is the production object [C]
// specified for it, verbatim:
//
//   "iota^* is an EquivariantCochainComplexMorphismOnK built through the
//    general constructor, its FiberBlockTable holding exactly one
//    PerSimplexBlock per simplex --- source simplex |iota_K sigma|, block
//    -+ rho_n(iota), the sign of lem:iotamodel NEGATED per def:iota(iv) and
//    folded into the matrix."
//
// THE SIGN (owner decision OQ3 of 2026-09-16; [C] v1.2.0 def:iota(iv);
// docs/implementation/oq3_resolution.md).  The geometric pullback is
//
//     (iota^*_geo f)(sigma) = rho_n(iota) f(iota_K sigma),
//
// and THE involution of this document is its NEGATIVE,
//
//     iota^* := -iota^*_geo,   (iota^* f)(sigma) = -rho_n(iota) f(iota_K sigma),
//
// so that "+" is BY DEFINITION the eigenspace containing the Eisenstein
// classes.  In the flat coordinates [C, eq:flatindex] (s*d_V + k) that is one
// block  -eps_sigma * rho_n(iota)  at block position (sigma, |iota_K sigma|),
// eps_sigma the orientation sign of [C, lem:iotamodel].  Nothing here may flip
// a sign at a call site ([C, rmk:iotasign]): the ONE global sign lives in
// iota_block(), and the degree-0 anchor iota^* = -1 on H^0(Gamma; F_ell) at
// n = 0 is what pins it.
//
// INDEPENDENCE FROM THE ORACLE.  hecke::iota_oracle (exported by slot-4 WP-3
// from Hecke/HeckeIotaSignValidation.h) builds the same involution inside the
// validation.  This translation unit NEVER calls it -- it re-derives iota_K
// from the shipped exact vertex coordinates, recomputes every orientation sign
// combinatorially, and forms rho_n(iota) through the live
// ComputeSymmetricPowerMatrixFM.  HeckeInvolutionValidation then compares the
// two ENTRYWISE; a production call into the oracle would make that vacuous.
//
// INVARIANT I8 (combinatorial, never geometric).  The vertex permutation is
// found by an EXACT QuadPoint comparison of z -> -conj(z) against the shipped
// coordinates (QuadPoint is exact algebraic arithmetic, not floating point),
// and is then CERTIFIED against [C, lem:iotamodel]'s permutation and sign
// table; the orientation signs themselves are the parity of the sort back into
// increasing vertex order, i.e. pure combinatorics.  Geometry only PROPOSES;
// certification DISPOSES.
//
// WHAT compute() DISCHARGES (aborting -- [C, obl:iota](i), (ii), (iii)):
//   (i)   iota_K is an involutive simplicial automorphism of K and every
//         recomputed orientation sign agrees with lem:iotamodel's table.
//   (ii)  every relation row's image under (iota_K, c_iota) is the inverse of
//         a listed row with equal orientation signs on both sides, and the
//         fibre identity rho_n(iota) rho_n(gamma) = rho_n(iota gamma iota)
//         rho_n(iota) holds entrywise.  The exact Mat22 half runs when exact
//         relation data is supplied (set_exact_relation_data); the fibre half,
//         which is what makes iota^* well defined on C^q_eq, always runs.
//   (iii) iota_C^2 = id and iota_C d = d iota_C in every degree, and iota_C
//         carries the stored C^q_eq / Z^q_eq / B^q_eq bases into their own
//         spans.
// Clause (iv) is NOT re-run here: it is HeckeIotaTauSymmetryValidation's gate
// on the canonical tau, and this object reads no tau.  Clause (v) is
// HeckeInvolutionValidation's and HeckeExt/HeckeSignSplit's.
//
// THE SPLIT ([C, prop:iotasplit]).  At 2 in R^x the projectors
// pr_+- = (id +- iota^*)/2 are complementary idempotents commuting with d, and
// this object stores the +- bases of C^q_eq, Z^q_eq, B^q_eq (flat cochain
// coordinates) and of H^q_eq (stored-basis coordinates), each as the image
// basis of the corresponding projector under the pipeline's own D5b rules.
// At ell = 2 the splitting is FALSE ([C, prop:iotasplit](2)) and this object
// reports sign_split_status() == NotApplicableCharTwo, recording instead the
// replacement statement of that clause: n := iota^* - id has n^2 = 0, and
// H^q is a sum of b = rank n free F_2[x]/(x^2)-modules and a = dim ker n - b
// trivial ones.  No +- basis and no halving of any matrix is offered there.
//
// NOT here: the Hecke commutation [C, obl:iota](v), the per-sign
// decomposition, the Eisenstein line and the golden collectors, which are
// HeckeExt/HeckeSignSplit's (the ext layer is MatrixT-sized); and the record
// integration, which is E2's.
// ===========================================================================

#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "ExactInt.h"
#include "SL2ZTessellationTables.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hecke {

// ---------------------------------------------------------------------------
// iota_K on a stored complex ([C, def:iota](iii), [C, lem:iotamodel]).
// ---------------------------------------------------------------------------
struct IotaSimplicialMap {
    // vertex_perm[v] = the vertex iota_K sends v to.
    std::vector<uint32_t> vertex_perm;
    // image_index[q][s] = |iota_K sigma_s|, the stored index of the image.
    std::vector<std::vector<uint32_t>> image_index;
    // image_sign[q][s] = eps_sigma in {+1, -1}: the parity of the permutation
    // sorting the image vertex list back into increasing order.
    std::vector<std::vector<int>> image_sign;

    uint32_t num_degrees() const { return static_cast<uint32_t>(image_index.size()); }
    uint32_t num_minus_signs() const;
};

// The 7-vertex model: iota_K PROPOSED by an exact QuadPoint match of
// z -> -conj(z) against SL2ZTessellationTables::vertex_position, then the
// signs recomputed combinatorially.  Throws if the map is not simplicial
// (hypothesis (I1) of [C, sec:involution] fails for the given complex).
IotaSimplicialMap BuildIotaSimplicialMapOnModel(const SimplicialComplex& K,
                                                const SL2ZTessellationTables& model);

// A complex with no exact coordinate table (the boundary path): the vertex
// permutation is given, the signs are still recomputed.
IotaSimplicialMap BuildIotaSimplicialMapFromVertexPermutation(const SimplicialComplex& K,
                                                              const std::vector<uint32_t>& vertex_perm);

// The boundary path v0-v1-v2-v3 of [C, prop:circlemodel] is iota-stable with
// the permutation 0<->3, 1<->2 (iota reverses the horocycle).
std::vector<uint32_t> IotaBoundaryPathVertexPermutation();

// [C, lem:iotamodel]'s printed sign table, by VERTEX SET: the six simplices of
// the 7-vertex model whose orientation sign is -1.
const std::vector<std::vector<uint32_t>>& IotaModelMinusSignSimplices();

// ---------------------------------------------------------------------------
// The certificate: [C, obl:iota](i)-(iii) as data, with a re-deriving verify().
// ---------------------------------------------------------------------------
struct InvolutionCertificate {
    uint32_t ell = 0;
    uint32_t n_sym = 0;
    uint32_t d_V = 0;                 // n_sym + 1
    uint32_t num_degrees = 0;
    bool model_is_seven_vertex = false;  // the lem:iotamodel sign table applies

    IotaSimplicialMap iota_K;
    FlatMatrix rho_iota;              // d_V x d_V, the LIVE rho_n(diag(-1,1))

    // Item 1 -- obl:iota(i).
    uint32_t minus_sign_count = 0;

    // Item 2 -- obl:iota(ii), fibre half (always run).
    struct RelationRow {
        uint32_t q_dim = 0;
        uint32_t s_lhs = 0, s_rhs = 0;
        FlatMatrix M;
        bool has_exact = false;
        Mat22 g_r;                    // meaningful iff has_exact
        // The matched image row (iota s_rhs, iota s_lhs, c_iota(gamma)^{-1}).
        bool matched = false;
        uint32_t match_index = 0;
    };
    std::vector<RelationRow> relations;
    bool exact_relation_data = false;     // the Mat22 half ran

    // Item 3 -- obl:iota(iii).  Per degree.
    std::vector<FlatMatrix> iota_H;        // dim H^q x dim H^q, stored basis
    std::vector<uint32_t> dim_H;
    std::vector<uint32_t> flat_dim;        // N_q * d_V

    // The split ([C, prop:iotasplit]).  Empty at ell = 2.
    std::vector<FlatMatrix> C_plus, C_minus;   // flat_dim x .
    std::vector<FlatMatrix> Z_plus, Z_minus;
    std::vector<FlatMatrix> B_plus, B_minus;
    std::vector<FlatMatrix> H_plus, H_minus;   // dim H^q x .
    bool split_applicable = false;
    // prop:iotasplit(2) at ell = 2: n := iota^* - id on H^q.
    std::vector<uint32_t> char_two_rank_n;     // b
    std::vector<uint32_t> char_two_dim_ker_n;  // a + b

    // 0 = every item holds; otherwise the index of the first failing item.
    uint32_t verify() const;
    uint32_t verify(std::vector<std::string>& failures) const;
    std::vector<uint32_t> failing_items() const;
    static const char* verify_message(uint32_t code);
};

enum class SignSplitStatus : uint8_t {
    Split = 0,                  // 2 in R^x: H^q = H^q_+ (+) H^q_-
    NotApplicableCharTwo = 1    // ell = 2: prop:iotasplit(2)'s filtration instead
};
const char* to_string(SignSplitStatus s);

// ---------------------------------------------------------------------------
// The object.
// ---------------------------------------------------------------------------
class HeckeInvolution {
public:
    // `coh` must be computed and must outlive this object.  `iota_K` must be
    // the involution of `coh`'s base complex.
    HeckeInvolution(const EquivariantCohomologyOnK* coh,
                    IotaSimplicialMap iota_K,
                    std::string label);
    ~HeckeInvolution();
    HeckeInvolution(const HeckeInvolution&) = delete;
    HeckeInvolution& operator=(const HeckeInvolution&) = delete;

    // Enables the exact Mat22 half of [C, obl:iota](ii).  Must be called before
    // compute(); the list must be the one `coh`'s local system was built from,
    // in the same order.
    void set_exact_relation_data(std::vector<SL2ZRelationDatum> data);

    // Builds iota^*, runs [C, obl:iota](i)-(iii) and the split; THROWS on any
    // failure (the obligation says "a failure aborts").  Idempotent.
    void compute();
    bool computed() const { return computed_; }

    uint32_t ell() const { return ell_; }
    uint32_t n_sym() const { return n_sym_; }
    uint32_t d_V() const { return d_V_; }
    uint32_t num_degrees() const { return num_degrees_; }
    uint32_t q_max() const { return num_degrees_ - 1u; }
    const std::string& label() const { return label_; }

    const IotaSimplicialMap& iota_K() const { return iota_K_; }
    const FlatMatrix& rho_iota() const { return rho_iota_; }

    // The production morphism: domain == codomain == `coh` (the class's
    // documented contravariant naming makes both the same object here), its
    // FiberBlockTable one PerSimplexBlock per simplex.
    const EquivariantCochainComplexMorphismOnK& morphism() const;

    // iota_C^{(q)} on the flat cochains, materialised on demand (the object
    // itself never stores it: the morphism holds one interned d_V x d_V block).
    //   sign = -1 (the default) -> iota^*    of def:iota(iv);
    //   sign = +1               -> iota^*_geo, the geometric pullback.
    FlatMatrix cochain_matrix(uint32_t q, int global_sign = -1) const;

    // One application without materialising anything.
    FlatMatrix apply_to_flat_cochain(uint32_t q, const FlatMatrix& c_flat) const;

    // iota^* on H^q in the stored basis: P_q iota_C H_eq, the column
    // convention of A^{(q)} ([C, alg:assembly]).
    const FlatMatrix& matrix(uint32_t q) const;
    uint32_t dim_H(uint32_t q) const;

    SignSplitStatus sign_split_status() const;
    // dim H^q x dim H^q_+- , the image basis of (id +- iota^*)/2.  Empty and
    // dim 0 at ell = 2 (the split is not applicable there).
    const FlatMatrix& H_plus_basis(uint32_t q) const;
    const FlatMatrix& H_minus_basis(uint32_t q) const;
    uint32_t dim_H_plus(uint32_t q) const;
    uint32_t dim_H_minus(uint32_t q) const;
    // flat_dim(q) x . , the +- bases of the stored cochain spaces.
    const FlatMatrix& C_eq_plus_basis(uint32_t q) const;
    const FlatMatrix& C_eq_minus_basis(uint32_t q) const;
    const FlatMatrix& Z_eq_plus_basis(uint32_t q) const;
    const FlatMatrix& Z_eq_minus_basis(uint32_t q) const;
    const FlatMatrix& B_eq_plus_basis(uint32_t q) const;
    const FlatMatrix& B_eq_minus_basis(uint32_t q) const;

    const InvolutionCertificate& certificate() const;

    std::string to_string() const;
    std::string to_string_offset(uint32_t offset) const;
    std::string to_json() const;      // schema "hecke_involution/v1"

    // -----------------------------------------------------------------------
    // (slot-23 P4I) The iota phase's host RSS -- value-neutral, P4(a)-class.
    //
    // Streamed (the default): the tasks of check_obl_iii and build_split are
    // ADMITTED first come, first served under a byte budget (the summed estimate of the
    // running tasks <= twice the largest task's; a task is always admitted
    // when nothing runs) -- run_concurrently starts one thread per task, so
    // before P4I all tasks of a phase held their buffers at once.  build_split's
    // flat families are split by one flat x k buffer (X +- iota_C X)/2 per
    // sign, built straight from the stored cochains, whose column rank profile
    // (exactly rref()'s pivots, so exactly image()'s columns) is taken in
    // place before the flat x r output is filled from the pivot columns with
    // the same residue formula; and the C / Z / B +- bases are kept ONCE (in
    // the certificate, which the accessors read) instead of twice.
    //
    // Rollback: HECKE_IOTA_STREAM=0 (read-only; anything else or unset =
    // streamed) or ScopedIotaStreaming(false): the pre-P4I computation
    // (all tasks concurrent, the materialised split, the copies).  The
    // setting is read once per compute(), on the calling thread.
    // (S25 LEVERS-25) HECKE_IOTA_STREAM=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1);
    // ScopedIotaStreaming(false) still selects the materialised route in every build.
    static bool iota_streaming();
    struct ScopedIotaStreaming {
        bool saved_;
        explicit ScopedIotaStreaming(bool on);
        ~ScopedIotaStreaming();
        ScopedIotaStreaming(const ScopedIotaStreaming&) = delete;
        ScopedIotaStreaming& operator=(const ScopedIotaStreaming&) = delete;
    };
    // Validation only: the admission budget in bytes (0 = automatic, twice
    // the largest task's estimate; 1 = one task at a time).
    struct ScopedIotaTaskBudget {
        uint64_t saved_;
        explicit ScopedIotaTaskBudget(uint64_t bytes);
        ~ScopedIotaTaskBudget();
        ScopedIotaTaskBudget(const ScopedIotaTaskBudget&) = delete;
        ScopedIotaTaskBudget& operator=(const ScopedIotaTaskBudget&) = delete;
    };
    // Validation only: a deliberate defect of the streamed route, which the
    // validation's entrywise comparison must detect.  1 = +1 on entry (0, 0)
    // of the lowest-degree streamed C^q_+ basis; 2 = its last pivot replaced
    // by the largest non-pivot column; 3 = the last split task with real work
    // (a non-empty family) not run; 0 = none.
    struct ScopedIotaStreamMutation {
        int saved_;
        explicit ScopedIotaStreamMutation(int kind);
        ~ScopedIotaStreamMutation();
        ScopedIotaStreamMutation(const ScopedIotaStreamMutation&) = delete;
        ScopedIotaStreamMutation& operator=(const ScopedIotaStreamMutation&) = delete;
    };
    struct StreamStats {
        bool streamed = false;
        uint32_t tasks_iii = 0, tasks_split = 0;
        uint32_t max_concurrent_iii = 0, max_concurrent_split = 0;   // most tasks admitted at once
        uint32_t streamed_families = 0;     // flat families split by the streamed route (non-empty)
        uint64_t budget_iii = 0, budget_split = 0;                   // bytes
        uint64_t peak_admitted_iii = 0, peak_admitted_split = 0;     // largest summed estimate admitted, bytes
        uint64_t max_task_iii = 0, max_task_split = 0;               // largest single estimate, bytes
        uint32_t mutations_applied = 0;
    };
    const StreamStats& stream_stats() const { return stream_stats_; }

private:
    const EquivariantCohomologyOnK* coh_ = nullptr;
    IotaSimplicialMap iota_K_;
    std::string label_;
    std::vector<SL2ZRelationDatum> exact_data_;
    bool has_exact_data_ = false;

    uint32_t ell_ = 0, n_sym_ = 0, d_V_ = 0, num_degrees_ = 0;
    bool computed_ = false;

    FlatMatrix rho_iota_;
    std::unique_ptr<EquivariantCochainComplexMorphismOnK> morphism_;
    std::vector<FlatMatrix> iota_H_;
    SignSplitStatus split_status_ = SignSplitStatus::Split;
    std::vector<FlatMatrix> C_plus_, C_minus_, Z_plus_, Z_minus_, B_plus_, B_minus_, H_plus_, H_minus_;
    InvolutionCertificate cert_;
    bool streaming_ = true;          // (P4I) iota_streaming() at compute()
    uint64_t task_budget_ = 0;       // (P4I) validation override, 0 = automatic
    int stream_mutation_ = 0;        // (P4I) validation only
    StreamStats stream_stats_;

    // (slot-22 PERF1, P3) iota_C on C^q as the signed coordinate permutation it
    // is: row r = s*d_V + a of iota_C x is  +-x[iota_src_[q][r]], negated iff
    // iota_neg_[q][r].  Built by build_signed_permutation() from the morphism's
    // own fibre-block table and checked against it entry by entry, so applying
    // it is exactly the morphism's apply_to_flat_cochain, one multi-vector at a
    // time instead of one block GEMV per column.
    std::vector<std::vector<uint32_t>> iota_src_;
    std::vector<std::vector<uint8_t>> iota_neg_;
    // rank of the stored C / Z / B multi-vector of degree q (index 0 / 1 / 2),
    // read off check_obl_iii's elimination of [X | iota_C X] and reused by
    // build_split's exhaustion check; -1 = not computed.
    std::vector<std::array<int64_t, 3>> family_rank_;

    void build_signed_permutation();
    FlatMatrix iota_apply(uint32_t q, const FlatMatrix& X) const;
    void iota_apply_rows(uint32_t q, const uint32_t* in, size_t in_stride, uint32_t* out, size_t out_stride,
                         uint32_t k) const;

    void build_rho();
    void build_morphism();
    void check_obl_i();
    void check_obl_ii();
    void check_obl_iii();
    void build_split();
    void require_computed(const char* who) const;
};

// The restriction morphism's naturality [C, eq:iotacochain]:
// r^q iota^* == iota^*_bdry r^q on H^q, with `r` the pipeline's own restriction
// (rows = boundary, cols = bulk).  Returns true iff the identity holds
// entrywise in every degree both maps have.  Pure; throws only on a shape
// mismatch.
bool IotaCommutesWithRestriction(const HeckeInvolution& bulk,
                                 const HeckeInvolution& boundary,
                                 const std::vector<FlatMatrix>& r_per_degree);

// ---------------------------------------------------------------------------
// (hecke-extensions slot 20, WP-LV3) iota^* OVER Z/ell^k ([E] sec:ring (h):
// the involution honours coefficient_ring = Z/ell^k at ell >= 5).
//
// The SAME cochain formula as HeckeInvolution, def:iota(iv) with the global
// sign negated: (iota^* f)(sigma_s) = -eps_s rho_n(iota) f(iota_K sigma_s),
// rho_n(iota) = diag((-1)^{n-j}) (= the live rho_n(diag(-1,1)) of the field
// object, ENGINE_CHECKed against ComputeSymmetricPowerMatrixFM by the field
// object's certificate), applied to the module basis cocycles b_i of
// H^q = (+)_i Z/ell^{e_i} (EquivariantCohomologyOnKT's module route) and read
// back by H_class_coordinates -- the formula slot-19 WP-LV's
// EisensteinBernoulliRoute applied in degree 1, now in every degree and
// owned here.  At ell >= 5 (2 a unit) H^q = H^q_+ (+) H^q_-, H^q_+- the images
// of (1 +- iota^*)/2, as modules: their invariant factors are what replaces
// dim H^q_+- (there is no dim over Z/ell^k, [E] sec:ring (b3)).
// compute ENGINE_CHECKs iota^2 = id and that the lengths of H^q_+ and H^q_-
// add up to length H^q; a failure throws (obl:iota "a failure aborts").
// ell in {2, 3} refuses (hecke::NotImplementedInThisBuild): the module route's
// stored basis does not exist there and 2 is not a unit at 2.
// ---------------------------------------------------------------------------
struct RingInvolutionDegree {
    std::vector<uint32_t> exps;                     // H^q = (+)_i Z/ell^{e_i}
    std::vector<std::vector<uint64_t>> iota;        // t x t (row-major, [i][j]): iota^* b_j = sum_i iota[i][j] b_i
    std::vector<uint32_t> plus_factors;             // invariant factors of H^q_+, descending
    std::vector<uint32_t> minus_factors;            // of H^q_-
    uint32_t length = 0, length_plus = 0, length_minus = 0;
    int64_t trace_on_free = 0;                      // sum over the exponent-k summands of iota[i][i] read as +-1 (rank+ - rank- of the free part)
};
struct RingInvolution {
    uint32_t ell = 0, n_sym = 0, k = 0;
    std::string label;
    std::vector<RingInvolutionDegree> degrees;
    std::string to_json() const;                    // schema "hecke_involution_ring/v1"
};
// `coh` must be computed over Z/ell^k (k >= 2, ell >= 5); `iota_K` the
// involution of its base complex.
template <class Elt>
RingInvolution BuildRingInvolution(const EquivariantCohomologyOnKT<Elt>& coh, const IotaSimplicialMap& iota_K,
                                   uint32_t n_sym, uint32_t ell, const std::string& label);

#if OC_ENABLE_LEGACY
// Number of (iii-c) carry checks that took the FlatMatrix route (ENGINE_BACKEND=legacy / shadow); PERF1 audit R2.
// (S24 CL-L2) ON builds only: Hecke/legacy/HeckeInvolutionLegacy.inc.
uint64_t iota_carry_legacy_calls();
#endif

} // namespace hecke

#endif // HECKE_INVOLUTION_H
