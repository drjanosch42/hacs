// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_MODULE_STRUCTURE_H
#define HECKE_EXT_MODULE_STRUCTURE_H

// ===========================================================================
// hecke-extensions phase E2a (slot 4, WP-1) -- STAGE 5, the module structure
// of a block V at its maximal ideal m ([E] `alg:stage5`, [X] Sec. 2.2
// "Stage 5 -- module structure").
//
// INPUT.  One `LocalAlgebra L` with `L.ok` -- i.e. a block V = F_ell^D that
// `hecke::CertifyLocality` has certified local, with its algebra A_S, its
// maximal ideal m_S (basis `L.cyclicity.m_basis`), its generators
// n_p = t_p - P_p(w_hat) and its residue degree f.  Nothing is re-searched:
// every quantity below is a deterministic function of L.
//
// OUTPUT ([E] `alg:stage5`(1)-(3),(5); (4), the canonical submodules of D6,
// needs E3 / E4 / E5 objects and is NOT in this unit -- the record writes
// `"canonical": {"status": "unavailable"}` until they arrive):
//
//   (1) LOEWY SERIES   m^{j+1}V = sum_a n_a (m^j V) over a basis {n_a} of m_S
//       (spans + RREF).  `loewy[j] = dim_F(m^j V / m^{j+1} V) = the F_ell
//       dimension divided by f` ([E] `cl:frank`: every layer is a K-space).
//       SOCLE SERIES  soc V = intersect_a ker n_a (one stacked kernel),
//       soc^{j+1}V = {x : n_a x in soc^j V for every a}, which IS the preimage
//       of soc(V / soc^j V).  Layer dimensions again divided by f.
//
//   (2) JORDAN TYPE of each n_p on V IN UNITS OF f: the nullity sequence of
//       n_p^s over F_ell, divided by f, fed to the conjugate-partition
//       construction of HeckeModuleDecomposition's S10 machinery (re-derived
//       here rather than shared -- that function is file-local there, and the
//       arithmetic is eight lines).
//
//   (3) STRUCTURE MAPS theta_j of [E] D4, one per CONSECUTIVE LAYER PAIR
//       (j, j+1) and per generator: the F_ell matrix of `b-bar |-> n_p b` from
//       the layer m^j V / m^{j+1} V to m^{j+1} V / m^{j+2} V, with **rows the
//       bottom-layer basis and columns the top-layer basis** ([E] Sec. 7's
//       `structure_maps[].matrices` convention, [X] Sec. 2.2 Stage 5).  The
//       piece's `rank` is the F_ell rank of the STACKED map over all
//       generators, and `rank_over_f` is that divided by f ([E] `cl:frank`);
//       `active` is the subset of primes whose own matrix is nonzero.
//
//   (5) cyclic <=> dim V/m_S V = f; free of rank one <=> cyclic and
//       dim V = dim A_S; min_generators = dim(V/m_S V)/f.
//
// THE LAYER BASES ARE CANONICAL, which is what makes an ENTRYWISE comparison
// of two runs (and of verify()'s re-derivation) meaningful: every subspace is
// held as a `hecke::RrefRowSpan` over F_ell^D, whose rows are the reduced row
// echelon form of the span and therefore a function of the SET of inserted
// vectors alone; a layer basis is the greedy selection, in that canonical row
// order, of the rows of m^j V that are independent modulo m^{j+1} V.
//
// `verify()` RE-DERIVES ([X] Sec. 1.3 R17) and reads nothing it concludes
// from: the embedded CyclicityCertificate re-derives A_S, m_S and the n_p from
// the stored t_p; the two filtrations, the layer bases, the Jordan types and
// every structure-map matrix are then recomputed and compared ENTRYWISE with
// what is stored.  A truncated series, a permuted layer basis, a decremented
// rank or a perturbed matrix therefore fails.
//
// NO RESUMABLE STATE.  [X] Sec. 1.3.1 R3.3 lists five budgeted instances and
// Stage 5 is not one of them: it is a fixed O(k nu D^3) computation on a block
// of size D <= ~200.
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t> throughout, single-threaded, no
// hash container; it consumes HeckeLocalAlgebra (hence Hecke/SubspaceOps,
// Hecke/FpPoly and Hecke/HeckeModuleDecomposition) and nothing else.
// ===========================================================================

#include "HeckeLocalAlgebra.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// One consecutive Loewy-layer pair, with one matrix per generator.
// ---------------------------------------------------------------------------
struct StructureMap {
    // The pair (j, j+1) of [E] Sec. 7's `"piece": [0,1]`.
    uint32_t top_layer    = 0;   // j
    uint32_t bottom_layer = 0;   // j + 1

    // The two layers' F_ell dimensions and their K-dimensions (/ f).
    uint32_t top_dim_fl    = 0, top_dim    = 0;
    uint32_t bottom_dim_fl = 0, bottom_dim = 0;

    // Index-aligned with the module's `primes`: one bottom_dim_fl x top_dim_fl
    // matrix per generator n_p, rows = bottom-layer basis, columns =
    // top-layer basis.
    std::vector<ExtMatrix> matrices;
    std::vector<uint32_t>  rank_p;     // the F_ell rank of each matrix
    std::vector<uint64_t>  active;     // the primes with rank_p > 0, ascending

    // The F_ell rank of the STACKED map (all generators at once) and its
    // K-rank ([E] `cl:frank`: the F_ell rank is f times the K-rank).
    uint32_t rank         = 0;
    uint32_t rank_over_f  = 0;
};

// ---------------------------------------------------------------------------
// The Stage-5 object.
// ---------------------------------------------------------------------------
struct ModuleStructure {
    // False exactly when the input LocalAlgebra was not `ok`; then every
    // field below is empty.  A non-local block is REPORTED, never thrown.
    bool ok = false;

    uint32_t ell = 0, D = 0, f = 0;
    uint32_t dim_A = 0;

    // The generators, ascending in p; index-aligned with `jordan` and with
    // every StructureMap's `matrices` / `rank_p`.
    std::vector<uint64_t> primes;

    // --- (1) the two filtrations -----------------------------------------
    // m_dims[j] = dim_{F_ell} m^j V for j = 0 .. nu_V, with m_dims[0] = D and
    // m_dims[nu_V] = 0; `loewy[j] = (m_dims[j] - m_dims[j+1]) / f`.
    std::vector<uint32_t> m_dims;
    std::vector<uint32_t> loewy;
    // socle_dims[j] = dim_{F_ell} soc^j V for j = 0 .. and
    // `socle[j] = (socle_dims[j+1] - socle_dims[j]) / f`.
    std::vector<uint32_t> socle_dims;
    std::vector<uint32_t> socle;

    // The canonical layer bases of the Loewy filtration: layer_basis[j] is a
    // D x (loewy[j]*f) matrix whose columns are the chosen representatives of
    // m^j V / m^{j+1} V, and m_basis_cols[j] a D x m_dims[j] matrix whose
    // columns are the canonical RREF basis of m^j V itself.
    std::vector<ExtMatrix> layer_basis;
    std::vector<ExtMatrix> m_basis_cols;

    // --- (2) Jordan types, in units of f ---------------------------------
    std::vector<std::vector<uint32_t>> jordan;   // index-aligned with `primes`

    // --- (3) the structure maps ------------------------------------------
    std::vector<StructureMap> structure_maps;

    // --- (5) cyclic / free ------------------------------------------------
    bool     cyclic         = false;
    bool     free_rank_one  = false;
    uint32_t min_generators = 0;

    // The input's certificate (a), which carries ell, D, f, the primes, the
    // t_p, the characters, pi_w, w_hat and the A_S / m_S bases -- and whose
    // own verify() re-derives all of them.  Stage 5's verify() starts there.
    CyclicityCertificate cyclicity;

    // 0 on success, else the index of the first failing clause.
    uint32_t verify() const;
    static const char* verify_message(uint32_t code);

    // Fixed key order, no pointers, no hash container.
    std::string to_json() const;
};

// Stage 5 on one certified-local block.  Throws std::invalid_argument on a
// malformed `L` (wrong sizes, missing certificate data); reports a non-local
// input through `ok == false`.
ModuleStructure ComputeModuleStructure(const LocalAlgebra& L);

} // namespace hecke_ext

#endif // HECKE_EXT_MODULE_STRUCTURE_H
