// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_PRESENTATION_H
#define HECKE_EXT_PRESENTATION_H

// ===========================================================================
// hecke-extensions phase E2 (slot 6, WP-2; OQ11) -- THE ON-REQUEST
// PRESENTATION of a certified-local block V = X_{i,m} over T_m = A_S|_V.
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.9,
// Sec. 2.2 Stage 5 ("On-request presentation (OQ11, DECIDED; v1.5)") and
// Sec. 5's `HeckePresentationValidation` row; [E] =
// docs/latex/hecke_extensions.tex; the commutativity A_S rests on is [C]
// `obl:algebra` / `thm:relations` (see HeckeLocalAlgebra.h).
//
// WHAT THIS COMPUTES.  No canonical form and no T_m-module isomorphism class
// (the deliverable of the record stays (D1)-(D7)); only generators and
// relations, and only when asked:
//
//   * GENERATORS.  A minimal A_S-generating set of V is a lift of a K-basis of
//     V / m_S V (Nakayama; K = A_S / m_S = F_{ell^f}).  The lift is the
//     DETERMINISTIC least-index one: walk the standard basis e_0, e_1, ...
//     and keep e_j iff it is not in  m_S V + sum_{kept i} A_S e_i.  So
//     t = min_generators = dim_{F_ell}(V / m_S V) / f.
//   * RELATIONS.  The relation module is R = ker(A_S^t -> V),
//     (a_1, ..., a_t) |-> sum_i a_i v_i, computed as the F_ell-kernel of the
//     D x (t * dim A_S) matrix whose column (i, j) is b_j v_i over the stored
//     RREF basis b_j of A_S; its canonical (RREF) F_ell-basis is then reduced
//     to a minimal A_S-generating set by Nakayama again: walk that basis and
//     keep r iff it is not in  m_S R + sum_{kept} A_S r'.  So
//     min_relations = dim_{F_ell}(R / m_S R) / f.
//   * OUTPUT.  The generators as columns of V (standard basis vectors, with
//     their indices) and the relations as t-TUPLES OVER THE A_S BASIS: a
//     relation is a vector of length t * dim A_S whose entry i * dim_A + j is
//     the coefficient of b_j in a_i.
//
// At f = 1 (every arm of the pipeline so far) the counts are F_ell-dimensions,
// exactly as [X] states them; at f > 1 [X]'s "dim_F(ker / m_S ker)" is read
// over the residue field K, i.e. divided by f -- the only reading under which
// it counts A_S-generators.  HeckePresentationValidation pins both.
//
// CERTIFICATE -- `verify()`, [X] Sec. 1.3's re-derivation rule (R17).  It runs
// on the stored object alone and re-derives every quantity it concludes from:
//   1  the embedded CyclicityCertificate re-derives A_S and m_S from the
//      stored t_p (so a_basis / m_basis are never trusted);
//   2  shapes and index ranges;
//   3  A_S is closed under multiplication in the stored basis (the
//      multiplication table is rebuilt, every product re-expanded);
//   4  the generators are the least-index Nakayama lift, re-derived, and
//      t * f == D - dim m_S V (a rank);
//   5  the generators span V over A_S (their A_S-span has dimension D);
//   6  every stored relation evaluates to 0 on the generators;
//   7  the stored relations generate R over A_S (A_S-span dimension ==
//      dim R == t * dim A_S - D, R recomputed as a kernel);
//   8  the relation count: min_relations * f == dim R - dim m_S R.
// Codes 6-8 together make the stored relations a MINIMAL generating set of R;
// the tamper arm of HeckePresentationValidation hits each code.
//
// NOT IN THE RECORD.  Slot 6 registers no `presentation` settings key and the
// `hecke_ext/v2` record carries no `presentation` object: the key, the record
// key and the arm "record hash unchanged by turning it on" are deferred to
// slot 7's owner-gated changed-line re-baseline (slot-6 plan Sec. 2.3 item 2,
// Sec. 4.4, Sec. 10 ruling 3).  The object is reached through the free
// function below or `ExtDriverOptions::with_presentation` (default off).
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t> through HeckeLocalAlgebra.h (no
// direct Matrix.h include, so check_matrix_includes.sh is unchanged); no
// Console/, Cuda/, Engine/, BlockFlatMatrix.h, no unordered_ container.
// ===========================================================================

#include "HeckeLocalAlgebra.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

struct Presentation {
    // False exactly when the block was not certified local (then everything
    // below is empty): a presentation over T_m needs the local algebra.
    bool available = false;

    uint32_t ell = 0;
    uint32_t D = 0;       // dim_{F_ell} V
    uint32_t f = 0;       // residue degree
    uint32_t dim_A = 0;   // dim_{F_ell} A_S

    // A_S, m_S and the t_p they are re-derived from.
    CyclicityCertificate algebra;

    // The generators: least-index standard basis vectors, ascending, and the
    // same vectors as the columns of a D x t matrix.
    std::vector<uint32_t> generator_indices;
    ExtMatrix generators;

    // The minimal relations, each a t-tuple over the A_S basis (length
    // t * dim_A, entry i * dim_A + j = coefficient of b_j in a_i).
    std::vector<std::vector<uint32_t>> relations;

    uint32_t min_generators = 0;        // t, over A_S
    uint32_t min_relations = 0;         // over A_S
    uint32_t dim_relation_module = 0;   // dim_{F_ell} R = t * dim_A - D

    bool is_free() const { return available && min_relations == 0; }

    // 0 on success, else the first failing clause (see the header comment).
    uint32_t verify() const;
    static const char* verify_message(uint32_t code);

    // Schema "hecke_presentation/v1"; fixed key order.  Not a record key.
    std::string to_json() const;
};

// The presentation of the block whose local algebra is `L`.  A non-local `L`
// (L.ok == false) gives `available = false`; everything else is computed and
// its certificate is ENGINE_CHECKed before return.
Presentation ComputePresentation(const LocalAlgebra& L);

} // namespace hecke_ext

#endif // HECKE_EXT_PRESENTATION_H
