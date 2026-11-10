// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_SPARSE_H
#define HECKE_LEVEL_N_SPARSE_H
// ===========================================================================
// (hecke-extensions slot 15, WP-8) THE SPARSE COKERNEL AT LARGE N; LEVEL-CHANGE
// AND DEGENERACY MAPS AT COMPOSITE N; THE COST MODEL.
// Binding text: [C] sec:levelN / sec:levelshapiro --
//   prop:levelorbit (1), (2)   Rel_N = M^{<S>} + M^{<ST>}, orbit by orbit; H^1 = M_N / Rel_N;
//   rmk:levelcost (4)          the cokernel: the orbit-local quotient by M^{<S>}, then ONE
//                              sparse elimination against M^{<ST>} (block sparsity; no bound
//                              better than cubic is asserted -- none is claimed here either);
//   prop:heilbronn             H_{N,p} on M_N / Rel_N (HeckeLevelNHeilbronn.h, WP-7);
//   prop:levelcorrespondence (4)(b), (c)  level change as fibre maps (res = the projection
//                              alpha^0, cor = the trace omega) and the degeneracy maps
//                              [Gamma_0(M) diag(1, d) Gamma_0(N)], composite d through
//                              Gamma_0(M d_1).
//
// WHAT IS BUILT.
//  * SparseCokernel -- the RREF of Rel_N over F_p (p >= 5), computed SPARSELY: the
//    generators are the orbit-wise fixed-space vectors (each supported on one orbit:
//    <= 2 blocks for S, <= 3 for ST), eliminated by sparse Gauss--Jordan with the
//    SMALLEST column of each reduced row as its pivot.  That rule keeps every stored row
//    of the form e_c + (entries at columns > c, none at another pivot), so the result
//    IS the reduced row echelon form of Rel_N in the natural column order -- unique --
//    and equals WP-7's dense CokernelPresentation (the oracle) entrywise: the same
//    pivots, the same free columns (the quotient basis), the same rows, hence the same
//    coordinates and the same Hecke matrices.  Value-identical by construction, checked
//    entrywise by HeckeLevelNSparseValidation at every dense-reachable cell.
//  * OnCokernelSparse -- [H_{N,p}] on the free-column basis, column by column from the
//    operator's block terms (no D-length dense apply); RelEscapesOnGenerators checks
//    prop:heilbronn (1) on the sparse generators of Rel_N (equivalent to the RREF rows).
//  * LevelChangeMap -- prop:levelcorrespondence (4)(b) on the induced module, for
//    Gamma_0(N) <= Gamma_0(M) (M | N): alpha^0 (res, M -> N) and omega (cor, the trace,
//    N -> M), each ONE monomial fibre map P^1(Z/N) -> P^1(Z/M) with m_N blocks.  A
//    Gamma-module map, so it preserves Rel (checked) and induces the map on the cokernel
//    presentation by the naturality of prop:levelorbit (2) in the coefficient module.
//  * DegeneracyFactors / DegeneracySpec / TraceSpec / RestrictionSpec -- the specs of
//    prop:levelcorrespondence (4)(b), (c) for BuildLevelCorrespondence (HeckeLevelEll.h)
//    on the model: alpha_d = [Gamma_0(M) diag(1, d) Gamma_0(N)] (dM | N; one term, r = {I})
//    for d a prime power, composite d as the composite through Gamma_0(M d_1) ((4)(c));
//    the trace omega = [Gamma_0(N) 1 Gamma_0(M)] (r = def:cosetrepsN (6)'s transversal).
//  * CokernelCostModel -- rmk:levelcost (1), (4) as operation / byte counts for the dense
//    RREF and the sparse route, the sparse fill from a fitted law; ChooseCokernelRoute.
//  * ReadEisensteinR3 -- the (R3) readings (dim H^1, W_{N,p}, its Loewy layers) on the
//    sparse presentation with H_{N,q}, q in {2, 3, 5, 7} \ {N}: the input WP-9 needs.
//
// Nothing here is reached from a shipped record, report, campaign job id or golden text.
// MP note: no model is read by the sparse route (the degeneracy specs are the model's by
// definition); -I enters only through n even (def:kappaN's hypothesis); 6 must be a unit.
// ===========================================================================

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "HeckeLevelEll.h"
#include "HeckeLevelN.h"
#include "HeckeLevelNHeilbronn.h"
#include "SparseRref.h"
#include "../../ModArithmetic/LegacyConfig.h"

namespace hecke {
namespace level {

// SparseVec (a sparse vector over F_p: strictly increasing columns, nonzero values) is defined in SparseRref.h, with the
// sparse Gauss--Jordan kernel shared with neb::NebSparseCokernel (S24 CL-G2, CODEAUDIT-B B1).

// The orbit-wise generators of M^{<gamma>} (prop:levelorbit (1)), sparse: the same vectors
// as OrbitWiseFixedSpace, except that at a fixed point every nonzero column of the averaging
// idempotent is kept (a spanning set; the elimination drops the dependent ones).
std::vector<SparseVec> OrbitWiseFixedGenerators(const CosetLevel& A, const Mat22& gamma, uint32_t order, uint32_t n_sym,
                                                const engine::ModCtx<uint32_t>& ctx);

// ---- (slot-18 WP-5 (b), [C] rmk:levelNscale (4)) the order in which the generators are eliminated --------
// The stored result is the reduced row echelon form of Rel_N in the natural column order, which is UNIQUE: the
// order in which SparseCokernel::eliminate consumes the generators changes no pivot, no free column, no row entry
// and so no coordinate or Hecke matrix.  It changes only the work: merge_ops, peak_nnz, the occurrence lists
// (which keep stale row ids until their column is pivoted) and the wall time.  Slots 15-17 consumed the
// generators as built (every S generator, orbits ascending, then every ST generator); W5-F1 (N = 99330, k = 2,
// past an 8 GB watchdog) is that order's cost, not the fill: the RREF there has 54.5 M off-pivot entries.
// Shipped since slot 18, and the only order since S24 CL-G2: the generators sorted by their LARGEST column,
// descending (stable), so that a pivot is established before the rows that would otherwise have to be
// back-substituted with it.
// (S24 CL-G2, owner ruling 2 of 2026-10-30) The as-built order and its rollback arm HECKE_LEVEL_N_GEN_ORDER=as_built
// are RETIRED: they live in Hecke/legacy/HeckeLevelNGenOrderLegacy.inc and are compiled only with OC_ENABLE_LEGACY=ON
// (GeneratorOrder, CurrentGeneratorOrder, GeneratorOrderName, ScopedGeneratorOrder, SparseCokernelStats::order below).
// In an OFF build HECKE_LEVEL_N_GEN_ORDER=as_built is REFUSED at the first elimination (stderr, exit 1); unset or
// `sorted` is the one route; anything else throws, as before.
#if OC_ENABLE_LEGACY
enum class GeneratorOrder : uint8_t {
    kLastColumnDescending = 0,   // shipped (slot 18)
    kAsBuilt = 1,                // slots 15-17: S then ST, as built -- the retired rollback arm and the bit-identity gate's twin
};
// The order in force on this thread: a ScopedGeneratorOrder if one is active, else the read-only rollback arm
// HECKE_LEVEL_N_GEN_ORDER (unset or `sorted`: kLastColumnDescending; `as_built`: kAsBuilt; anything else throws).
GeneratorOrder CurrentGeneratorOrder();
const char* GeneratorOrderName(GeneratorOrder o);   // "sorted" / "as_built"
// VALIDATION ONLY (thread-local; no shipped caller sets it): the bit-identity arm builds both orders.
class ScopedGeneratorOrder {
public:
    explicit ScopedGeneratorOrder(GeneratorOrder o);
    ~ScopedGeneratorOrder();
    ScopedGeneratorOrder(const ScopedGeneratorOrder&) = delete;
    ScopedGeneratorOrder& operator=(const ScopedGeneratorOrder&) = delete;
private:
    int prev_;
};
#endif
// The name of the order the next elimination on this thread uses ("sorted"; "as_built" only in an ON build).  Reads
// HECKE_LEVEL_N_GEN_ORDER like the elimination does, so an OFF build refuses `as_built` here too.
const char* GeneratorOrderInForceName();

struct SparseCokernelStats {
    uint64_t generators = 0;        // relation vectors fed in (S then ST)
    uint64_t generator_nnz = 0;
    uint64_t dependent = 0;         // generators reduced to zero
    uint64_t final_nnz = 0;         // off-pivot entries of the stored RREF rows
    uint64_t peak_nnz = 0;
    uint64_t merge_ops = 0;         // entry operations of the reductions and back substitutions
    uint32_t dim_S = 0, dim_ST = 0; // ranks of M^{<S>}, M^{<ST>} (each eliminated alone; disjoint orbits)
    uint64_t occ_entries = 0;       // (slot-18 WP-5) row ids pushed to the occurrence lists (4 bytes each, kept to the end)
#if OC_ENABLE_LEGACY
    GeneratorOrder order = GeneratorOrder::kLastColumnDescending;   // (slot-18 WP-5 (b)) the order used
#endif
    double generators_s = 0, eliminate_s = 0;
};

class SparseCokernel {
public:
    // Rel_N from the two generator lists (S first, then ST), over F_p (p >= 5 prime).
    SparseCokernel(uint32_t D, const std::vector<SparseVec>& gen_S, const std::vector<SparseVec>& gen_ST, uint32_t p);
    // The generators built here (n even, 6 a unit); keep_generators retains them for
    // RelEscapesOnGenerators.
    static SparseCokernel Build(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                bool keep_generators = true);
    // (slot-22 LN23, object=curve) the weight-2 presentation at EVERY prime p, the fixed spaces as orbit sums:
    // M / (M^S + M^{ST}) = H_1(X_0(N), cusps; Z) (x) F_p = H^1(Y_0(N); F_p), the cohomology of the open coarse curve
    // ([C] prop:levelNcurve; Manin's presentation).  n_sym must be 0 (refused by name otherwise).  At p >= 5 it equals
    // Build(A, 0, ctx) entry for entry.
    static SparseCokernel BuildCurve(const CosetLevel& A, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                     bool keep_generators = true);

    uint32_t D() const { return D_; }
    uint32_t p() const { return p_; }
    uint32_t dim_rel() const { return static_cast<uint32_t>(R_.pivots.size()); }
    uint32_t h1() const { return static_cast<uint32_t>(R_.free.size()); }
    uint32_t h0() const { return stats_.dim_S + stats_.dim_ST - dim_rel(); }   // prop:levelorbit (2)
    const std::vector<uint32_t>& pivots() const { return R_.pivots; }          // ascending
    const std::vector<uint32_t>& free_columns() const { return R_.free; }      // ascending: the quotient basis
    // The RREF row of pivot c (c a pivot): e_c + tail, tail at free columns > c.
    const SparseVec& tail_of(uint32_t c) const { return R_.rows[static_cast<size_t>(R_.row_of_col[c])]; }
    bool is_pivot(uint32_t c) const { return R_.row_of_col[c] >= 0; }
    int32_t free_index(uint32_t c) const { return R_.free_index[c]; }          // -1 at a pivot
    const SparseCokernelStats& stats() const { return stats_; }
    const std::vector<SparseVec>& generators() const { return gens_; }        // empty unless kept

    // Coordinates on the free columns of x mod Rel_N (x sparse or dense).
    std::vector<uint32_t> coords(const SparseVec& x) const;
    std::vector<uint32_t> coords_dense(const std::vector<uint32_t>& x) const;
    bool in_rel(const SparseVec& x) const;

private:
    uint32_t D_ = 0, p_ = 0;
    sparse::Rref R_;                       // the stored RREF (SparseRref.h)
    std::vector<SparseVec> gens_;
    SparseCokernelStats stats_;
    struct CurveTag {};
    SparseCokernel(uint32_t D, const std::vector<SparseVec>& gen_S, const std::vector<SparseVec>& gen_ST, uint32_t p, CurveTag);
    void init(const std::vector<SparseVec>& gen_S, const std::vector<SparseVec>& gen_ST);
    void eliminate(const std::vector<SparseVec>& gens);
};

// The rank of a set of sparse vectors whose supports lie in disjoint orbits is the sum of the
// per-orbit ranks; computed by the same sparse elimination (no fill across orbits).
uint32_t SparseRank(uint32_t D, const std::vector<SparseVec>& vecs, uint32_t p);

// (slot-18 WP-5 (b)) FNV-1a-64 (engine/Fnv.h) of the presentation: D, p, the pivots ascending and, per pivot, its RREF
// row (columns and values), then the free columns.  Independent of the generator order by the uniqueness of the
// RREF; the SPARSE_CELL hook prints it so that two runs (e.g. an OC_ENABLE_LEGACY=ON build's
// HECKE_LEVEL_N_GEN_ORDER=as_built against the default) can be compared on cells too large for an in-process twin.
uint64_t PresentationDigest(const SparseCokernel& Q);

// [H] on M_N / Rel_N in the free-column basis (h1 x h1 field HMat), from the block terms.
HMat OnCokernelSparse(const SparseCokernel& Q, const HeilbronnOperator& H);
// prop:heilbronn (1) on the generators of Rel_N (needs the kept generators): the number of
// generators whose image leaves Rel_N.
uint32_t RelEscapesOnGenerators(const SparseCokernel& Q, const HeilbronnOperator& H);
// H applied to a sparse vector (the block terms, grouped by source block).
SparseVec ApplySparse(const HeilbronnOperator& H, const SparseVec& x);
// The same with the source grouping built once (repeated applications).
class HeilbronnSparseApply {
public:
    explicit HeilbronnSparseApply(const HeilbronnOperator& H);
    SparseVec apply(const SparseVec& x) const;
private:
    const HeilbronnOperator* H_;
    std::vector<std::vector<uint32_t>> by_src_;
};

// ---- level change: prop:levelcorrespondence (4)(b) as monomial fibre maps -----------------
enum class LevelChangeKind : uint8_t {
    kRestriction = 0,   // alpha^0 = res^{Gamma_0(M)}_{Gamma_0(N)}: M_M -> M_N, (f)_i = rho(v_i w_{a(i)}^{-1}) f_{a(i)}
    kTrace = 1,         // omega = cor^{Gamma_0(M)}_{Gamma_0(N)}: M_N -> M_M, (f)_a = sum_{i -> a} rho(w_a v_i^{-1}) f_i
};

class LevelChangeMap {
public:
    // big = Gamma_0(M), small = Gamma_0(N), M | N.
    LevelChangeMap(const Gamma0LevelN& big, const Gamma0LevelN& small, LevelChangeKind kind, uint32_t n_sym,
                   const engine::ModCtx<uint32_t>& ctx);
    LevelChangeKind kind() const { return kind_; }
    uint32_t d() const { return d_; }
    uint32_t src_m() const { return src_m_; }
    uint32_t dst_m() const { return dst_m_; }
    uint32_t src_dim() const { return src_m_ * d_; }
    uint32_t dst_dim() const { return dst_m_ * d_; }
    // i in P^1(Z/N) -> a(i) in P^1(Z/M) (1-based): the fibre map of the coset spaces.
    const std::vector<uint32_t>& fibre() const { return fibre_; }
    std::vector<uint32_t> apply(const std::vector<uint32_t>& x) const;       // dense, over ctx
    SparseVec apply_sparse(const SparseVec& x) const;
    std::vector<uint32_t> dense() const;                                    // dst_dim x src_dim, row-major

private:
    LevelChangeKind kind_;
    uint32_t d_ = 1, src_m_ = 1, dst_m_ = 1;
    engine::ModCtx<uint32_t> ctx_;
    std::vector<uint32_t> fibre_;           // 1-based, size m_N + 1
    std::vector<uint32_t> blk_;             // 1-based over i: the block of rho(v_i w_{a(i)}^{-1}) (res) / rho(w_a v_i^{-1}) (trace)
    std::vector<Block> blocks_;
};

// The matrix (dst.h1 x src.h1) of a level-change map on the two presentations; ENGINE_CHECKs
// that every generator of the source's Rel lands in the target's Rel first.
HMat OnCokernelLevelChange(const SparseCokernel& src, const SparseCokernel& dst, const LevelChangeMap& F);

// ---- degeneracy maps on the model: prop:levelcorrespondence (4)(b), (c) ------------------------
// d = p_1^{nu_1} ... p_s^{nu_s} (primes ascending): the prime-power factors.
std::vector<std::pair<uint64_t, uint32_t>> DegeneracyFactors(uint64_t d);
// alpha_d, d = p^nu a prime power, A = Gamma_0(M), B = Gamma_0(N), p^nu M | N: the spec of
// [A diag(1, p^nu) B] (B'' = B, r = {I}), on the level-1 datum of the primitive double coset.
CorrespondenceSpec DegeneracySpec(const Gamma0LevelN& A, const Gamma0LevelN& B, const HeckeCosetTables& g_cosets,
                                  const HeckeTauTable& tau);
// res (A = Gamma_0(M) -> B = Gamma_0(N), identity datum, r = {I}).
CorrespondenceSpec RestrictionSpec(const Gamma0LevelN& A, const Gamma0LevelN& B);
// cor, the trace (A = Gamma_0(N) -> B = Gamma_0(M), identity datum, r = Gamma_0(N) \ Gamma_0(M)).
CorrespondenceSpec TraceSpec(const Gamma0LevelN& A, const Gamma0LevelN& B);

// ---- the cost model: rmk:levelcost (1), (4) ---------------------------------------------------
enum class CokernelRoute : uint8_t { kDense = 0, kSparse = 1, kAuto = 2 };
struct CokernelCost {
    uint64_t D = 0, m = 0;
    uint32_t d = 1;
    double gen_rows = 0;                 // (m - nu2)/2 d + nu2 d + (m - nu3)/3 d + nu3 d  (spanning rows)
    double dense_ops = 0, dense_bytes = 0;
    double sparse_nnz = 0, sparse_ops = 0, sparse_bytes = 0;
    CokernelRoute route = CokernelRoute::kSparse;
};
// The fitted sparse law (HeckeLevelNSparseValidation (C) re-measures it on its ladder and reports the
// ratios): final fill ~ 0.645 m^1.335 d^2 (x 1.7 at d > 1), merge operations ~ 0.034 m^2.14 d^3.
CokernelCost CokernelCostModel(uint64_t N, uint32_t n_sym);
CokernelRoute ChooseCokernelRoute(uint64_t N, uint32_t n_sym);   // kDense or kSparse
// (slot-22 LN23) Which object the cokernel presentation is read as.  kGroup: H^1(Gamma_0(N); V_n) (prop:levelorbit (2),
// 6 a unit).  kCurve: weight 2 only (n = 0), every p: H_1(X_0(N), cusps; Z) (x) F_p = H^1(Y_0(N); F_p) ([C]
// prop:levelNcurve; at p >= 5 the two coincide, value for value).
enum class CokernelObject : uint8_t { kGroup = 0, kCurve = 1 };
// [H_{N,q}] on H^1 = M_N / Rel_N for the listed primes, by the chosen route (kAuto: the cost model).
// Value-identical on either route (the natural RREF; HeckeLevelNSparseValidation (S)); the sparse route
// checks prop:heilbronn (1) on every generator first.  With the default object (kGroup), p in {2, 3} is refused by
// name on every route, kAuto included (both need 6 a unit, prop:levelorbit (2); slot-15 WP-17).  With kCurve, every p
// is served at n = 0 and n > 0 is refused by name.
std::map<uint32_t, HMat> CokernelHeckeMatrices(const Gamma0LevelN& lvl, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                               const std::vector<uint32_t>& primes, CokernelRoute route = CokernelRoute::kAuto,
                                               CokernelRoute* used = nullptr, CokernelObject object = CokernelObject::kGroup);

// ---- (R3) readings on the sparse presentation ---------------------------------------------------
struct R3Reading {
    uint64_t N = 0, p = 0;
    uint32_t dimH1 = 0, dimW = 0, loewy = 0;
    std::vector<uint32_t> layers;
    double presentation_s = 0, operators_s = 0, eisenstein_s = 0;
    uint64_t final_nnz = 0;
};
// N prime (or any N; q | N gives U_q), p >= 5 prime with p !| N, weight 2 (n = 0): W_{N,p} is the
// joint generalised eigenspace of H_{N,q} at 1 + q for q in {2, 3, 5, 7} \ {N}, its layers
// m^i W the spans of the images under the T_q - (1 + q), exactly as HeckeLevelNValidation (R3).
R3Reading ReadEisensteinR3(uint64_t N, uint64_t p);
// (slot-16 WP-13; [C] rmk:levelNcheckabletwo (LN9), R3_k) the same reading at weight k = n + 2 >= 2: W^{(k)}_{N,p}
// is the joint generalised eigenspace of H_{N,q} at 1 + q^{k-1} for every prime q <= qmax with q !| N p, its
// layers m^i W as above.  N any (p !| N), p >= 5 prime, n even.  At n = 0 it differs from ReadEisensteinR3 only
// in dropping q = p (and every q | N) from the family.  A reading: no prediction is asserted here.
R3Reading ReadEisensteinR3k(uint64_t N, uint64_t p, uint32_t n_sym, uint32_t qmax = 7);

// Dense F_p helpers used by the readings (exposed for the validation).
std::vector<uint32_t> FpKernelBasis(const std::vector<uint32_t>& A, uint32_t rows, uint32_t cols, uint32_t p,
                                    uint32_t& dim);   // cols x dim, row-major
uint32_t FpRank(std::vector<uint32_t> A, uint32_t rows, uint32_t cols, uint32_t p);

} // namespace level
} // namespace hecke

#endif // HECKE_LEVEL_N_SPARSE_H
