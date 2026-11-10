// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_DICKSON_MULTIPLICATION_H
#define HECKE_EXT_DICKSON_MULTIPLICATION_H

// ===========================================================================
// hecke-extensions slot 5, WP-8 -- `L.`, MULTIPLICATION BY THE DICKSON
// INVARIANT, and its certificate.  [X] Sec. 1.1's inventory gap [I G12] /
// [XA m10]: "the Dickson multiplication L.: Sym^{n-ell-1} -> Sym^n,
// L = x^ell y - x y^ell (an (n+1) x (n-ell) matrix with entries in {0, +-1} in
// the monomial basis; L(g.(x, y)) = det(g).L(x, y) for every 2x2 g over F_ell,
// hence Hecke-equivariant up to the det-twist of [E] `cl:Tlnil`) ... new
// HeckeExt/DicksonMultiplication.{h,cpp} (~100 lines; the det-twisted
// intertwining with rho_n and rho_{n-ell-1} checked entrywise on seeded g, in
// the action convention of [C, def:rhon])".
//
// [X] = docs/implementation/hecke_extensions_implementation_plan.md v1.8;
// [E] = docs/latex/hecke_extensions.tex v1.5.2, `sec:identities` item 6 and
// `cl:Tlnil`; [C] = docs/latex/gl2_contract.tex v1.6.2, cited by label.
//
// THE OBJECT.  In the monomial basis of [C, def:rhon] as
// ComputeSymmetricPowerMatrixFM implements it,
//
//     Sym^d has the ordered basis  u^{(d)}_k = x^{d-k} y^k,  k = 0 .. d,
//     rho_d(A) column j = coordinates of  A_*(u^{(d)}_j),
//     A_* being the algebra map  x |-> a x + c y,  y |-> b x + d y
//     for A = [[a, b], [c, d]]  (so that rho_d(AB) = rho_d(A) rho_d(B)).
//
// With m := n - ell - 1 and L := x^ell y - x y^ell the map
//
//     L. : Sym^m -> Sym^n,   f |-> L f
//
// sends u^{(m)}_j = x^{m-j} y^j to
//
//     x^{ell} y . x^{m-j} y^j  -  x y^{ell} . x^{m-j} y^j
//   = x^{n-1-j} y^{j+1}        -  x^{n-ell-j} y^{j+ell}
//   = u^{(n)}_{j+1}            -  u^{(n)}_{j+ell},
//
// so COLUMN j of the (n+1) x (n-ell) matrix is  +1 at row j+1, -1 at row
// j+ell, and 0 elsewhere -- entries in {0, +-1}, exactly one pair per column,
// and the two rows are distinct because ell >= 2.  In particular the matrix is
// INJECTIVE (rank n - ell): the rows j+1 are distinct across columns and carry
// the leading +1.
//
// THE det-TWIST, over F_ell and for EVERY 2x2 A over F_ell, singular or not.
// In characteristic ell, (a x + c y)^ell = a^ell x^ell + c^ell y^ell
// = a x^ell + c y^ell, so
//
//     A_*(L) = (a x^ell + c y^ell)(b x + d y) - (a x + c y)(b x^ell + d y^ell)
//            = (ad - bc)(x^ell y - x y^ell) = det(A) . L,
//
// and since A_* is a ring map, A_*(L f) = det(A) . L . A_*(f).  In matrices:
//
//     rho_n(A) . L  ==  det(A) . L . rho_m(A)                        (*)
//
// -- the DEFINING identity of this unit, re-derived entrywise by verify() on
// seeded A.  No determinant hypothesis is used: (*) holds for singular A too,
// which is why the certificate's sample list deliberately contains some.
//
// WHY `FlatMatrix` AND NOT `ExtMatrix`.  [X] Sec. 2.3's layer decision is
// "MatrixT<uint32_t> for everything in HeckeExt/ except the cone", and its
// stated reason is that the ext layer's objects are LEVEL-(b), D-sized
// (D = dim H^q <= ~200) matrices that the certificates re-derive with the
// MatrixT RREF/kernel surface.  `L.` is not one of those: it is a FIBRE-level
// object of size (n+1) x (n-ell) -- the same kind of object as rho_n, which
// the pipeline ships as a FlatMatrix -- and its one consumer is the COCHAIN
// layer (`PerSimplexBlock::F` in EquivariantCochainComplexMorphismOnK.h is a
// FlatMatrix, and ComputeSymmetricPowerMatrixFM returns one).  Typing it
// MatrixT would mean two conversions on every use and would put a d_V-sized
// object on the D-sized allow-list.  Consequence: this unit needs NO entry in
// check_matrix_includes.sh's Matrix.h allow-list, because it includes no
// Matrix.h.  The COHOMOLOGY-level induced map L_* IS level-(b) and is stored
// as a MatrixT in HeckeIdentityCertificate, which is already allow-listed.
//
// PLACEMENT.  HeckeExt/, namespace hecke_ext.  This TU consumes only
// ModArithmetic's FlatMatrix / ModArithmetic / EngineCheck -- no Console/, no
// Cuda/, no Engine/, no BlockFlatMatrix.h, no `unordered_` (the four rules of
// docs/matrix-engine/tools/check_heckeext_includes.sh).  It touches no
// pipeline object at all: the cohomology-level statement is the caller's.
// ===========================================================================

#include "../../ModArithmetic/FlatMatrix.h"

#include <cstdint>
#include <string>
#include <vector>

namespace hecke_ext {

// ---------------------------------------------------------------------------
// The source degree m = n - ell - 1 and the applicability predicate.
// ---------------------------------------------------------------------------
//
// `L.` exists as a non-empty map exactly when n >= ell + 1.  At n = ell + 1 it
// is the (ell+2) x 1 matrix of the FIRST Dickson class ([E] Sec. 8.3's
// "ell in {5,7,11}, n = ell+1: first Dickson class L").
inline bool DicksonMultiplicationApplicable(uint32_t n, uint32_t ell) {
    return ell >= 2 && n >= ell + 1;
}

inline uint32_t DicksonSourceDegree(uint32_t n, uint32_t ell) {
    return DicksonMultiplicationApplicable(n, ell) ? (n - ell - 1) : 0u;
}

// ---------------------------------------------------------------------------
// The matrix of L. : Sym^{n-ell-1} -> Sym^n over F_ell, (n+1) x (n-ell).
// ---------------------------------------------------------------------------
//
// Returns the empty 0 x 0 matrix when !DicksonMultiplicationApplicable.
// `ell` is the modulus AND the exponent in L; the two are the same prime by
// definition of the Dickson invariant.
FlatMatrix DicksonMultiplicationMatrix(uint32_t n, uint32_t ell);

// ---------------------------------------------------------------------------
// The certificate ([X] Sec. 1.1: "the det-twisted intertwining with rho_n and
// rho_{n-ell-1} checked entrywise on seeded g").
// ---------------------------------------------------------------------------
//
// Certificate convention of this tree (HeckeModuleDecomposition::certify()):
// the struct holds exactly the data a third party needs, and verify()
// RE-DERIVES every value rather than trusting the producer.  Here that means
//   * the shape, from (n, ell) alone;
//   * every entry, by a GENERIC homogeneous-polynomial convolution against the
//     coefficient vector of L -- a different code path from the direct
//     two-index write that built it;
//   * the rank, by the substrate's own elimination;
//   * identity (*) above, entrywise, on every stored sample.
struct DicksonMultiplicationCertificate {
    uint32_t n = 0;
    uint32_t ell = 0;
    uint32_t m = 0;            // n - ell - 1
    uint32_t rows = 0;         // n + 1
    uint32_t cols = 0;         // n - ell
    bool     applicable = false;

    // The stored map.
    FlatMatrix L;

    // The seeded 2x2 matrices A = [[a, b], [c, d]] over F_ell that (*) was
    // checked on, flattened as (a, b, c, d) in row-major order, 4 entries per
    // sample.  Deterministic: CertifyDicksonMultiplication builds them from a
    // fixed seed, and the list always begins with I, -I, S, T and two singular
    // matrices, so that a caller comparing two certificates compares the same
    // samples.
    std::vector<uint32_t> samples;

    uint32_t num_samples() const { return static_cast<uint32_t>(samples.size() / 4); }

    // 0 on success, else the first failing clause:
    //   1  ell < 2, or the applicability flag disagrees with (n, ell)
    //   2  the shape is not (n+1) x (n-ell), or the modulus is not ell
    //   3  an entry differs from the independently convolved column
    //   4  rank != n - ell (L. is not injective)
    //   5  no samples, or a malformed sample list
    //   6  the det-twisted intertwining (*) fails on some sample
    uint32_t verify() const;
    static const char* verify_message(uint32_t code);

    std::string to_string_offset(uint32_t offset) const;
};

// `num_samples` extra PSEUDO-RANDOM samples on top of the six fixed ones.
DicksonMultiplicationCertificate CertifyDicksonMultiplication(uint32_t n, uint32_t ell,
                                                              uint32_t num_samples = 18);

} // namespace hecke_ext

#endif // HECKE_EXT_DICKSON_MULTIPLICATION_H
