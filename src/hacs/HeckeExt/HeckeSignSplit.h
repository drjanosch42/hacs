// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKE_SIGN_SPLIT_H
#define HECKE_EXT_HECKE_SIGN_SPLIT_H

// ===========================================================================
// The SIGN SPLIT of the Hecke module -- phase E3, hecke-extensions slot 4 WP-3.
// [C, obl:iota](v), [C, prop:iotasplit](1); [E] sec:tests / Claim 1.8's
// "cyclic per iota-sign"; [X] Sec. 6.2 E3.
// ===========================================================================
//
// WHAT THIS IS.  Hecke/HeckeInvolution builds iota^* ([C, def:iota](iv), the
// v1.2.0 normalisation iota^* = -iota^*_geo) and splits the stored objects.
// This unit takes the RESULT -- the matrix of iota^* on H^q and the two
// eigenbases, in the stored basis -- together with the computed Hecke matrices
// A^{(q)}(T_p), and does the Hecke-module half:
//
//   * [C, obl:iota](v): [iota^*, A^{(q)}] = 0, exactly, per operator.
//     ASSERTED at ell >= 5, where [C, prop:iotacommutes](3c) proves it
//     (6 in R^x); at ell = 3 it is COMPUTED and RECORDED (the only route there
//     is (3b), whose hypothesis tau^iota = tau is a separate obligation), and
//     the restriction A^{(q)}_+- is formed only where it holds exactly.  At
//     ell = 2 the whole split is not_applicable ([C, prop:iotasplit](2)) and
//     this object reports that without forming anything.
//   * A^{(q)}_+- by an EXACT solve  A U_s = U_s A_s  (hecke::SolveColumnsExact,
//     which throws rather than returning a wrong answer, so the solve IS the
//     invariance certificate).
//   * the per-sign primary decomposition (hecke::HeckeModuleDecomposition on
//     the restricted family) beside the unsplit one.
//   * cyclicity per sign, through slot-2 WP-4's generation certificate (a)
//     (hecke_ext::ComputeLocalAlgebra / CyclicityCertificate).
//   * the EISENSTEIN LINE of [E] D6: for a block V = X^+_{4,m} of the PLUS side
//     in degree 1, soc(V) = Ann_V(m_S) = intersect_p ker n_p and m_S V =
//     sum_p im n_p are computed, asserted equal, and the line is DEFINED when
//     that space has F-dimension 1 (F = T_m/m, so dim_{F_ell} / f == 1) and
//     "undefined" otherwise -- which is exactly what happens on the UNSPLIT
//     H^1 at (10, 691), where the socle is 2-dimensional because the Eisenstein
//     and congruent cuspidal reductions are not separable mod ell.
//
// GOLDEN COLLECTORS.  charpoly_iota^q, rank_iota^q, dim_Hplus^q,
// dim_Hminus^q, eisenstein_line_defined (tier A) and iota^q (tier B) are
// DEFINED here and CALLED ONLY FROM HeckeInvolutionValidation.  Nothing writes
// GoldenReference.inc: slot 4 emits no golden entry ([S4] Sec. 4.4), and
// iota^* occurs in no golden object ([C, rmk:iotasign]).
//
// LAYER RULES.  MatrixT<uint32_t> throughout ([X] Sec. 1.1: D <= ~200), no
// Console/, no Cuda/, no Engine/, no BlockFlatMatrix.h, no unordered_ -- and
// no pipeline object at all: this unit takes matrices, so it can be exercised
// without constructing a cohomology.  Hecke/HeckeInvolution is the pipeline
// half and lives in Hecke/ (check_heckeext_includes.sh rule 1 forbids the
// pipeline from depending on this layer).
// ===========================================================================

#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../../ModArithmetic/Matrix.h"
#include "HeckeLocalAlgebra.h"
#include "HeckeExtGoldenObject.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hecke_ext {

enum class SignSide : uint8_t { Plus = 0, Minus = 1 };
const char* ToString(SignSide s);

// Why the split is or is not available on this cell.
enum class SignSplitStatus : uint8_t {
    Split = 0,             // 2 in R^x
    NotApplicable = 1      // ell = 2, [C, prop:iotasplit](2)
};
const char* ToString(SignSplitStatus s);

struct SignSplitOperator {
    uint64_t p = 0;
    uint32_t nu = 1;
    ExtMatrix A;              // A^{(q)}(T_{Gamma g Gamma}) on H^q, dim x dim
    bool commutes = false;    // [iota^*, A] == 0, exactly
    bool restricted = false;  // A_plus / A_minus were formed
    ExtMatrix A_plus;         // dim H^q_+ square
    ExtMatrix A_minus;        // dim H^q_- square
};

// [E] D6 / [X] Sec. 6.2 E3(b): the Eisenstein line at a maximal ideal.
struct EisensteinLine {
    bool applicable = false;      // the plus side exists and carries blocks
    bool defined = false;         // dim_F soc = 1
    uint32_t block_index = 0;     // which block of the plus side
    uint32_t f = 0;               // residue degree of that block
    uint32_t dim_socle = 0;       // dim_{F_ell} soc(V)
    uint32_t dim_socle_over_F = 0;// dim_socle / f
    bool socle_equals_mV = false; // soc(V) == m_S V  ([E]: "= m H^1_+")
    ExtMatrix socle_basis;        // dim V x dim_socle, in the BLOCK's own basis
    ExtMatrix line_in_H;          // dim H^q x dim_socle, in the stored H^q basis
    std::string status;           // "defined" | "undefined" | "not_applicable"
};

// (slot 5, WP-3) `GoldenObject` moved VERBATIM to
// HeckeExt/HeckeExtGoldenObject.h, included above -- HeckeBoundary.h carried a
// second, identical definition and Golden/GoldenHarness.cpp includes both.

struct SignSplitCertificate {
    uint32_t ell = 0, n_sym = 0, degree = 0;
    uint32_t dim_H = 0, dim_plus = 0, dim_minus = 0;
    SignSplitStatus status = SignSplitStatus::Split;
    ExtMatrix iota;                  // dim_H square
    ExtMatrix plus_basis, minus_basis;   // dim_H x dim_+-
    std::vector<SignSplitOperator> ops;
    bool commutation_asserted = false;   // ell >= 5
    uint32_t num_commuting = 0;
    EisensteinLine eisenstein;
    // Per sign: the block dimensions and residue degrees of the decomposition,
    // and whether generation certificate (a) passed on each block.
    std::vector<uint32_t> plus_block_dims, minus_block_dims, unsplit_block_dims;
    std::vector<uint8_t> plus_block_cyclic, minus_block_cyclic;

    // 0 = every item holds; otherwise the index of the first failing item.
    uint32_t verify() const;
    uint32_t verify(std::vector<std::string>& failures) const;
    std::vector<uint32_t> failing_items() const;
    static const char* verify_message(uint32_t code);
};

class HeckeSignSplit {
public:
    // `iota`, `plus_basis`, `minus_basis` are the stored-basis objects of
    // hecke::HeckeInvolution in degree `degree`; `ops` the computed
    // A^{(degree)}(T_p) in any order (sorted ascending by (p, nu) internally).
    // Everything is copied.
    HeckeSignSplit(uint32_t ell, uint32_t n_sym, uint32_t degree,
                   ExtMatrix iota, ExtMatrix plus_basis, ExtMatrix minus_basis,
                   std::vector<std::pair<uint64_t, ExtMatrix>> ops);
    ~HeckeSignSplit();
    HeckeSignSplit(const HeckeSignSplit&) = delete;
    HeckeSignSplit& operator=(const HeckeSignSplit&) = delete;

    // Runs everything above.  Throws only on what is a THEOREM at 6 in R^x
    // (a non-commuting operator at ell >= 5, a failing certificate); records
    // otherwise.  Idempotent.
    void compute();
    bool computed() const { return computed_; }

    uint32_t ell() const { return ell_; }
    uint32_t n_sym() const { return n_sym_; }
    uint32_t degree() const { return degree_; }
    uint32_t dim_H() const { return dim_H_; }
    uint32_t dim_plus() const;
    uint32_t dim_minus() const;
    SignSplitStatus status() const { return status_; }

    size_t num_operators() const;
    const SignSplitOperator& op(size_t i) const;
    const SignSplitOperator& op(uint64_t p, uint32_t nu) const;   // throws if absent

    // The per-sign and the unsplit primary decompositions.  Null when the
    // corresponding side is empty or the split is not applicable.
    const hecke::HeckeModuleDecomposition* decomposition(SignSide s) const;
    const hecke::HeckeModuleDecomposition* decomposition_unsplit() const;

    const EisensteinLine& eisenstein_line() const;
    const SignSplitCertificate& certificate() const;

    // DEFINED HERE, CALLED ONLY FROM THE VALIDATION.  No emission.
    std::vector<GoldenObject> golden_objects() const;
    uint64_t golden_digest() const;

    std::string to_string() const;
    std::string to_string_offset(uint32_t offset) const;
    std::string to_json() const;    // schema "hecke_sign_split/v1"

private:
    uint32_t ell_ = 0, n_sym_ = 0, degree_ = 0, dim_H_ = 0;
    ExtMatrix iota_, plus_, minus_;
    std::vector<std::pair<uint64_t, ExtMatrix>> input_ops_;
    SignSplitStatus status_ = SignSplitStatus::Split;
    bool computed_ = false;

    std::vector<SignSplitOperator> ops_;
    std::unique_ptr<hecke::HeckeModuleDecomposition> dec_plus_, dec_minus_, dec_unsplit_;
    EisensteinLine eis_;
    SignSplitCertificate cert_;

    void restrict_operators();
    void decompose();
    void find_eisenstein_line();
};

} // namespace hecke_ext

#endif // HECKE_EXT_HECKE_SIGN_SPLIT_H
