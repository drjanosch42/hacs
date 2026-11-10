// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_SUBSPACE_OPS_H
#define HECKE_SUBSPACE_OPS_H

// hecke-extensions phase E0a -- the exact F_ell subspace toolkit shared by the
// level-(c) machinery.
//
// PLACEMENT (slot-1 plan Sec. 2.3 item 5, which corrects
// hecke_extensions_implementation_plan.md Sec. 1.1): these helpers used to live
// in the anonymous namespace of HeckeModuleDecomposition.cpp.  E0b's locality
// refinement and E1a's local-algebra layer both need them, and
// HeckeModuleDecomposition.cpp must be able to include them -- but the
// HeckeExt/ include guard forbids `Hecke/` from including `HeckeExt/`.  So they
// live HERE, under Hecke/, and the guard stays one-way (HeckeExt -> Hecke).
//
// VALUE-PRESERVING MOVE.  Every function below is the VERBATIM body that
// HeckeModuleDecomposition.cpp carried before the move (same pivot rules, same
// search order, same deterministic seed kRcfSeed, same exception texts), so the
// whole golden reference -- including the basis-dependent tier-B `decomp_*_U`
// hashes that rest on PrimaryRCFBasis' basis choice -- is unchanged by it.
// GoldenRegressionValidation under GOLDEN_STRICT=1 is the proof.
//
// Build placement: compiled into cohomology_app, NOT into the pure-integer
// HeckeCore static library, because it consumes ModArithmetic's Matrix
// (kernel / rank / RREF) -- exactly the reasoning recorded for
// HeckeModuleDecomposition.cpp.

#include "FpPoly.h"

#include "../../ModArithmetic/Matrix.h"

#include <cstdint>
#include <vector>

namespace hecke {

// Deterministic seed for the phi-functional fallback search inside the
// cyclic-vector complement construction (search only; every result is
// certified downstream).
constexpr uint64_t kRcfSeed = 0x510B10CBA5150001ull;

// The deterministic PRNG used by every search in this layer (SplitMix64).
// Search only -- nothing it produces is ever trusted without a downstream
// deterministic certificate.
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t next_mod(uint32_t m) { return static_cast<uint32_t>(next() % m); }
};

// rank of M (rank() lazily caches the RREF, so this works on a copy).
uint32_t RankOf(const Matrix& M);

// Exact solve B * X = C for full-column-rank B (d x k), C (d x m); asserts
// exact solvability (throws otherwise). This is the invariance certificate
// wherever a restriction to a subspace is computed.
Matrix SolveColumnsExact(const Matrix& B, const Matrix& C, const char* who);

// Horizontal concatenation of `parts` (all with `rows` rows) over F_ell.
Matrix HStack(const std::vector<Matrix>& parts, uint32_t rows, uint32_t ell);

// ker M as an (M.cols x dim ker) matrix whose columns span the kernel.
Matrix KernelAsMatrix(Matrix M);

// Smallest r >= 0 with N^r = 0; throws if not nilpotent within `cap` steps.
uint32_t NilpotencyExponent(const Matrix& N, uint32_t cap, std::vector<Matrix>* powers_out);

// Within-block cyclic-vector / quotient-chain construction: for a pi-primary
// operator Aj on F_ell^n (charpoly pi^m, n = m * deg pi), returns an
// invertible R (n x n) whose columns are the concatenated cyclic bases
// v, Aj v, ..., Aj^{k-1} v of a chain of cyclic summands with non-increasing
// heights, so R^{-1} Aj R = (+) companion(pi^{s_t}) (primary rational
// canonical form). sizes_out receives (s_1 >= s_2 >= ...).
//
// The invariant complement of a maximal-height cyclic subspace Z(v) is
// computed as ker Phi for Phi(u) = (phi(u), phi(Aj u), ..., phi(Aj^{k-1} u)),
// where phi is a linear functional with Phi|_{Z(v)} invertible: ker Phi is
// Aj-invariant because the minimal polynomial pi^{e} of Aj on the current
// space has degree exactly k = dim Z(v) (so Aj^k is a combination of lower
// powers), and rank(Phi Z) = k forces ker Phi (+) Z(v) = V. A good phi always
// exists (the dual of the cyclic module F_ell[X]/(pi^e) is cyclic); the
// search tries the dual standard basis first, then deterministic-seed random
// functionals. Every use is followed by an exact-solve restriction whose
// solvability certifies invariance, and by the companion-form assertion in
// compute().
Matrix PrimaryRCFBasis(const Matrix& Aj_in, const FpPoly& pi,
                       std::vector<uint32_t>& sizes_out);

// ---------------------------------------------------------------------------
// E0a additions (new, not a move).
// ---------------------------------------------------------------------------

// Vertical concatenation of A (a x c) and B (b x c) over F_ell, so that
// ker(VStack(A, B)) == ker A INTERSECT ker B -- the cheapest exact way to
// intersect two kernels without an image-basis extraction.
Matrix VStack(const Matrix& A, const Matrix& B);

// True iff N is nilpotent, decided by ceil(log2(cap)) squarings (N^(2^k) == 0
// for the first 2^k >= max(cap, 1)).  `cap` is the ambient dimension.
bool IsNilpotent(const Matrix& N, uint32_t cap);

// A^(ell^N) over F_ell, formed as N successive ell-th powers (each by
// square-and-multiply on the bits of ell).  For a matrix A = s + n with s
// semisimple, n nilpotent and n^cap = 0, choosing ell^N >= cap kills n
// (Frobenius: (s+n)^(ell^N) = s^(ell^N) + n^(ell^N)), so with deg(pi_s) | N
// this is exactly the Teichmueller lift of A's residue ([E] Claim 3.3).
Matrix FrobeniusPower(const Matrix& A, uint32_t ell, uint32_t N);

// ---------------------------------------------------------------------------
// E1s addition (WP-3, hecke-extensions slot 1; [X] Sec. 1.1 / Sec. 2.1).
// ---------------------------------------------------------------------------

// The UNITAL F_ell-subalgebra of Mat_D(F_ell) generated by `generators`, as an
// RREF COORDINATE MATRIX: one row per basis element, D*D columns holding the
// generator's entries in row-major order (entry (i, j) sits at column i*D + j).
// `dim_out` receives its dimension.
//
// The generators must be D x D over the same ell and must COMMUTE pairwise --
// the two consumers ([X] Sec. 2.1) are HeckeLocalAlgebra's A_S on D-sized
// Hecke matrices (E1a) and HeckeSturmBound's <T_q : q <= B> on
// dim M_k-sized Miller-basis matrices (E1s) -- so the span of the products of
// the generators IS the generated algebra and closing the span under RIGHT
// multiplication by each generator suffices.
//
// CANONICAL, hence RESUME-INVARIANT.  The result is the reduced row echelon
// form of the span, which is unique; it is therefore a function of the SET of
// generators alone and not of the order in which they were supplied or of the
// order in which products were formed.  That is exactly what makes the Sturm
// state's `closure_basis` a function of (k, ell, B) and the whole computation
// a prefix cut in the sense of [X] Sec. 1.3.1 R3.1 item 3.  No hash container
// is used, and the algorithm is a fixed sequence of exact F_ell operations.
Matrix algebra_closure(const std::vector<Matrix>& generators, uint32_t D, uint32_t ell,
                       uint32_t& dim_out);


// ---------------------------------------------------------------------------
// E1a addition (WP-4, hecke-extensions slot 2; [X] Sec. 1.1 / Sec. 2.2 Stage 3).
// ---------------------------------------------------------------------------

// An incrementally maintained REDUCED row echelon basis of a subspace of
// F_ell^width.
//
// `pivots()` is strictly increasing and `rows()[i]` has a 1 at pivots()[i] and
// 0 at every other pivot -- i.e. the rows ARE the RREF of their span at every
// moment, which makes the final basis CANONICAL: a function of the SET of
// inserted vectors and not of the order in which they were inserted.  That is
// what lets a certificate compare a stored basis with a re-derived one
// ENTRYWISE rather than by span ([X] Sec. 1.3 R17).
//
// PROVENANCE: the body is the verbatim pre-WP-4 text of the file-local
// `RrefSpan` that E1s' `algebra_closure` (below) carried in the anonymous
// namespace of SubspaceOps.cpp; WP-4 promoted it to the header, with
// `contains()` added, so that HeckeLocalAlgebra's ideal / socle / tangent
// steps use the SAME canonical span object the closure does rather than a
// second copy of it.  `algebra_closure` now calls it, so its output is
// unchanged by construction (GoldenRegressionValidation under GOLDEN_STRICT=1
// and HeckeSturmBoundValidation are the proof).
class RrefRowSpan {
public:
    RrefRowSpan(uint32_t width, uint32_t ell) : width_(width), ell_(ell) {}

    // Reduces `v` against the current basis and, if a nonzero remainder is
    // left, normalises it, back-substitutes it into the existing rows and
    // inserts it at its pivot position.  Returns true iff the rank grew.
    bool insert(std::vector<uint32_t> v);

    // Reduces a COPY of `v` and reports whether the remainder is zero, i.e.
    // whether `v` already lies in the span.  Never mutates the basis.
    bool contains(const std::vector<uint32_t>& v) const;

    uint32_t width() const { return width_; }
    uint32_t dim() const { return static_cast<uint32_t>(rows_.size()); }
    const std::vector<std::vector<uint32_t>>& rows() const { return rows_; }
    const std::vector<uint32_t>& pivots() const { return pivots_; }

    // The basis as a dim() x width() matrix, one row per basis vector.
    Matrix as_matrix() const;

private:
    uint32_t width_;
    uint32_t ell_;
    std::vector<std::vector<uint32_t>> rows_;
    std::vector<uint32_t> pivots_;
};

// The coordinate convention shared by `algebra_closure`, `RrefRowSpan` and
// HeckeLocalAlgebra: a matrix is vectorised ROW-MAJOR, so entry (i, j) of an
// r x c matrix sits at coordinate i*c + j.
std::vector<uint32_t> VecOfMatrix(const Matrix& M);
Matrix MatrixOfVec(const std::vector<uint32_t>& v, uint32_t rows, uint32_t cols, uint32_t ell);

// ---------------------------------------------------------------------------
// E0b-S addition (WP-3, hecke-extensions slot 3; [X] Sec. 3.1 "Strict mode"
// steps 1-2, [E] Cert. `sec:localcert` "Strict mode" (1)-(2)).
//
// STRICT LOCALITY MODE'S LINEAR ALGEBRA.  On a COMMUTATIVE F_ell-algebra the
// Frobenius x -> x^ell is F_ell-LINEAR ((x + y)^ell = x^ell + y^ell and
// (cx)^ell = c x^ell for c in F_ell), so the whole of "is this block local,
// and if not how does it split" is a kernel computation and needs NO search:
//
//   rad A_S = {nilpotents} = ker(Frob^N)  for ell^N >= D      (step 1)
//   Abar    = A_S / rad A_S = prod_i F_{ell^{f_i}},
//   B       = ker(Frob_Abar - id) = F_ell^m,  dim B = m       (step 2, Berlekamp)
//
// and V is local iff dim B = 1.  The TRACE FORM IS NOT USABLE for the radical
// here: it vanishes identically on F_ell[x]/(x^ell), a local algebra of
// dimension ell ([E] `sec:localcert`, strict mode (1)).
//
// The three functions below are the only new primitives; everything else
// strict mode needs (kernels, exact solves, nilpotency, Teichmueller powers)
// already exists above.  They are pure linear algebra over F_ell and contain
// no randomness at all.
// ---------------------------------------------------------------------------

// The pivot columns of an RREF coordinate matrix (one row per basis element,
// as `algebra_closure` returns): `pivots[i]` is the index of the first nonzero
// entry of row i, and the sequence is strictly increasing.  Throws if a row is
// zero or the pivots are not increasing, i.e. if `closure` is not in RREF.
std::vector<uint32_t> RrefPivots(const Matrix& closure);

// The coordinates of `M` (r x c, vectorised ROW-MAJOR as everywhere in this
// layer) in the RREF coordinate basis `closure`.  Because the basis is in
// REDUCED row echelon form the coordinates are read off the pivot positions --
// c_i = vec(M)[pivots[i]] -- and the result is then VERIFIED by re-expanding
// it, so a matrix outside the span throws rather than returning a wrong
// answer (`who` names the caller in the message).
std::vector<uint32_t> AlgebraCoords(const Matrix& closure, const std::vector<uint32_t>& pivots,
                                    const Matrix& M, uint32_t ell, const char* who);

// The matrix of the F_ell-linear Frobenius x -> x^ell on the algebra spanned
// by the RREF coordinate basis `closure` of D x D matrices: a dim x dim matrix
// whose column i holds the coordinates of b_i^ell.  Each b_i^ell is formed by
// square-and-multiply on the bits of ell, so the cost is
// O(dim * D^3 * log ell).  Throws (through AlgebraCoords) if the span is not
// closed under ell-th powers, which for a commutative algebra cannot happen
// and is therefore an exactness certificate on `closure`.
Matrix frobenius_matrix(const Matrix& closure, uint32_t D, uint32_t ell);

// ker(F^N) as a (dim x dim ker) matrix whose COLUMNS span the kernel -- with
// F the Frobenius matrix above and ell^N >= D, this is `rad A_S` in the
// coordinates of `closure` ([X] Sec. 3.1 step 1).  N is taken as given; the
// caller owns the ell^N >= D condition.
Matrix frobenius_kernel(const Matrix& F, uint32_t N);

} // namespace hecke

#endif // HECKE_SUBSPACE_OPS_H
