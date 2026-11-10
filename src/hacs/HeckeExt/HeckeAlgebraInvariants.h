// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_ALGEBRA_INVARIANTS_H
#define HECKE_EXT_ALGEBRA_INVARIANTS_H

// ===========================================================================
// hecke-extensions slot 9, WP-13 (console phase C5b) -- the LOCAL-ALGEBRA
// INVARIANTS of an arbitrary commutative F_ell-algebra of operators, with no
// locality certificate as input.
//
// WHY A SECOND ROUTE.  HeckeExt/HeckeLocalAlgebra computes A_S, m_S, the socle
// and the tangent table of a block that `hecke::CertifyLocality` has ALREADY
// certified local, reading the witness w_hat and the characters P_p from that
// certificate.  [P] Sec. 3.5's local-algebra verbs (`nilpotent_part`,
// `radical`, `tangent_space`, `socle`, `nilpotency_index`, `loewy`,
// `socle_series`, `is_local`) are asked of ANY commuting family a console user
// hands in -- a family that may not be local, and for which no witness exists.
// So they need a route that decides everything from the operators alone.
//
// THE ROUTE (no search, no randomness).  On a COMMUTATIVE F_ell-algebra A the
// Frobenius x -> x^ell is F_ell-LINEAR, so (the strict-locality linear algebra
// of Hecke/SubspaceOps.h, [E] `sec:localcert` strict mode (1)-(2)):
//
//   rad A            = ker(Frob^e),  ell^e >= D                 (the nilpotents)
//   #maximal ideals  = dim{x : x^ell - x in rad A} - dim rad A   (Berlekamp)
//   A local          <=> that number is 1; then f = dim A - dim rad A
//   rad^{i+1}        = span{ r s : r in rad, s in rad^i }        (A commutative)
//   soc A            = Ann_A(rad A)
//   m/m^2            = rad / rad^2                               (tangent space)
//
// and, for an A-stable subspace V of F_ell^D,
//
//   Loewy series     rad^0 V = V,  rad^{j+1} V = span{ r x : r in rad, x in rad^j V }
//   socle series     soc^0 V = 0,  soc^{j+1} V = { x in V : r x in soc^j V for every r }
//
// which are exactly HeckeExt/HeckeModuleStructure's `m_dims` and `socle_dims`
// when A = A_S and V is the block (the validation asserts the equality).
//
// THE JORDAN-CHEVALLEY SPLIT (`nilpotent_part`).  F_ell is perfect, so every
// t in End(F_ell^D) is s + n with s semisimple, n nilpotent, sn = ns, both
// polynomials in t.  With L the lcm of the degrees of the irreducible factors
// of charpoly(t) and N the least multiple of L with ell^N >= D,
//   s = t^(ell^N)
// (Frobenius: (s + n)^(ell^N) = s^(ell^N) + n^(ell^N) = s^(ell^N), and ell^N
// fixes every eigenvalue of s since each lies in F_{ell^L}).  On a certified-
// local block this s IS HeckeLocalAlgebra's P_p(w_hat) -- the split is unique
// -- so `n` equals LocalAlgebra::n entrywise, which the validation asserts.
// The split is re-verified before it is returned (commutation, nilpotency of
// n, square-freeness of minpoly(s)).
//
// CANONICAL OUTPUT.  Every subspace is returned as the reduced row echelon
// basis of its span (hecke::RrefRowSpan over the row-major vectorisation), so
// two runs -- and a script's transcript -- see the same matrices.
//
// PLACEMENT.  HeckeExt/, MatrixT<uint32_t>, single-threaded, no hash
// container, no Console/ / Cuda/ / Engine/ include (check_heckeext_includes.sh
// rules 2 and 4).  It consumes Hecke/SubspaceOps and Hecke/FpPoly only.
// ===========================================================================

// MatrixT<uint32_t> through HeckeLocalAlgebra.h (ExtMatrix): a DIRECT include of
// the deprecated Matrix.h would need a check_matrix_includes.sh allow-list entry.
#include "HeckeLocalAlgebra.h"

#include <cstdint>
#include <vector>

namespace hecke_ext {

using AlgMatrix = ExtMatrix;

// The invariants of A = F_ell[generators] inside End(F_ell^D).
struct AlgebraInvariants {
    uint32_t ell = 0;
    uint32_t D   = 0;       // the ambient dimension

    // A itself: the canonical RREF basis of the span (hecke::algebra_closure).
    uint32_t               dim_A = 0;
    std::vector<AlgMatrix> basis;

    // The least e >= 1 with ell^e >= D; rad A = ker(Frob^e).
    uint32_t frobenius_exponent = 0;

    // rad A = the nilpotent elements, canonical basis.
    uint32_t               dim_rad = 0;
    std::vector<AlgMatrix> radical;

    // Berlekamp: the number of maximal ideals of A, and the locality verdict.
    uint32_t components = 0;
    bool     local      = false;
    // [A/m : F_ell] iff local, else 0.
    uint32_t residue_degree = 0;

    // rad^i for i = 1, 2, ... while nonzero: radical_power_dims[i-1] = dim rad^i.
    std::vector<uint32_t> radical_power_dims;
    // The least nu >= 1 with rad^nu = 0 (nu == 1 iff A is reduced).
    uint32_t nilpotency_index = 0;

    // rad / rad^2: its F_ell-dimension and canonical representatives (the
    // radical's canonical rows, greedily, independent modulo rad^2).
    uint32_t               dim_tangent = 0;
    std::vector<AlgMatrix> tangent;

    // soc A = Ann_A(rad A), canonical basis; Gorenstein iff local and
    // dim soc == f.
    uint32_t               dim_socle = 0;
    std::vector<AlgMatrix> socle;
    bool                   gorenstein = false;

    // Local only: hilbert[i] = dim(rad^i / rad^{i+1}) / f, 0 <= i < nu
    // (rad^0 = A), HeckeLocalAlgebra's `hilbert`.  Empty when not local.
    std::vector<uint32_t> hilbert;
};

// Throws std::invalid_argument on an empty family, a non-square or
// mismatched-size / mismatched-modulus operator, or a NON-COMMUTING pair
// (the invariants above are those of a commutative algebra).
AlgebraInvariants ComputeAlgebraInvariants(const std::vector<AlgMatrix>& generators,
                                           uint32_t ell);

// The two filtrations of an A-stable subspace V (D x k, independent columns).
// `dims[j]` is the F_ell-dimension of the j-th term, `layers[j]` the
// successive quotient's (Loewy: dims[j] - dims[j+1]; socle: dims[j+1] -
// dims[j]).  Loewy: dims[0] = k, ..., last = 0.  Socle: dims[0] = 0, ...,
// last = k.  Throws std::invalid_argument when V's columns are dependent, of
// the wrong height, or V is not A-stable.
struct ModuleSeries {
    std::vector<uint32_t> dims;
    std::vector<uint32_t> layers;
};
ModuleSeries LoewySeries(const AlgebraInvariants& A, const AlgMatrix& V);
ModuleSeries SocleSeries(const AlgebraInvariants& A, const AlgMatrix& V);

// The Jordan-Chevalley split t = s + n over F_ell, verified.
struct JordanChevalley {
    AlgMatrix semisimple;
    AlgMatrix nilpotent;
    uint32_t  lcm_degree = 0;   // L: lcm of the irreducible factor degrees
    uint32_t  exponent   = 0;   // N: s = t^(ell^N)
};
JordanChevalley JordanChevalleySplit(const AlgMatrix& t);


} // namespace hecke_ext

#endif // HECKE_EXT_ALGEBRA_INVARIANTS_H
