// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ORBIFOLD_MP_MP_FILL_H
#define ORBIFOLD_MP_MP_FILL_H

// ===========================================================================
// (hecke-extensions slot 22, MP-4) An explicit Z-linear FILLING of the MP
// resolution N Z[F'_.] -> Z ([C, def:mpset], [C, thm:mpcomparison]): for a
// cycle z of N Z[F'_k] (k = 0: augmentation 0) a chain y of N Z[F'_{k+1}] with
// d y = z, exactly over Z.  This is the MP model's own counterpart of
// PeriodicityClass::fill (the contracting homotopy of Iso_.), and what a
// twisted chain datum ON THE MP COMPLEX needs (MPHecke.h): the comparison
// theorem over a non-trivial group homomorphism (conj_g) has no geometric
// simplicial map to read off (X and gX are different truncations, gap G3 of
// [C, rmk:mphecke]), so it is built generator by generator with this filling.
//
// CONCRETE STRINGS.  An element of F'_0 is a translate (h, t), h in Gamma, t a top
// simplex representative (the labelled Cech nerve: (h, t) and (-h, t) are
// different elements of the free Gamma-set); a k-string is a (k+1)-tuple of
// translates whose closed simplices h iota(t) share a vertex (under (SC) the
// common intersection is a face, non-empty iff it has a vertex).  Normalised:
// a string with two equal consecutive translates is 0.  The face d_j drops the
// j-th translate; on the free-module form h_0 . e_s, s = (t_i; h_{i-1}^{-1} h_i),
// this is d e_s = gamma_1 e_{d_0 s} + sum_{i >= 1} (-1)^i e_{d_i s} -- the MP
// differential.
//
// THE FILLING: the proof of [C, thm:mpcomparison] (Steps 2-3) made explicit.
// The double complex D_{a,b} = Z{(x, s) : x an a-simplex of X (vertices in
// orbit order, (IO)), s a normalised b-string with x inside every simplex of s},
//   d_v (x, s) = sum_i (-1)^i (d_i x, s),   d_h (x, s) = sum_j (-1)^j (x, d_j s),
//   d = d_v + (-1)^a d_h,
// has exact COLUMNS (fixed s: the chains of the simplex cap(s), augmented to Z,
// contracted by the cone from v(s), the least vertex of cap(s)) and exact ROWS
// (fixed x: the chaotic complex on the translates containing x, augmented to
// C_a(X), contracted by prepending the apex a(x), the least translate containing x).
//   alpha : Tot -> N Z[F'] ((v, s) -> s at a = 0)   and   beta : Tot -> C_.(X)
//   ((x, (U)) -> x at b = 0) are quasi-isomorphisms.  fill(z):
//   (1) lift c = sum z_s (v(s), s); correct it to a Tot cycle with alpha = z by the
//       column cones (descending b; d c lies in the acyclic ker alpha);
//   (2) fill the Tot cycle by the row cones (descending a); at b = 0 the residue
//       beta(c) is a cycle of C_.(X): degree 0 by a 1-skeleton path (breadth-first
//       search), degree 1 by THE unique 2-chain (X is contractible and
//       2-dimensional, d_2 injective; found over F_p on a growing set of
//       triangles and certified exactly over Z, as PeriodicityClass::fill_x1),
//       degree 2 is 0; lifted by the row section x -> (x, (a(x)));
//   (3) y := alpha(Tot filling); d y = z is ENGINE_CHECKed exactly.
// Deterministic (std::map order, least keys); no randomisation; int64,
// overflow-checked; no golden object; no setting.  Pure host C++.
// ===========================================================================

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "Descriptor.h"
#include "Nerve.h"

namespace orbifold_mp {

// A translate (h, t) of F'_0.
struct Tr {
    uint32_t t = 0;
    GKey h{};
    bool operator<(const Tr& o) const { return t != o.t ? t < o.t : h < o.h; }
    bool operator==(const Tr& o) const { return t == o.t && h == o.h; }
    bool operator!=(const Tr& o) const { return !(*this == o); }
};
using Str = std::vector<Tr>;               // a concrete string (k + 1 translates)
using StrChain = std::map<Str, int64_t>;   // a chain of N Z[F'_k] (concrete form)
using XS = std::vector<PointKey>;          // a simplex of X, vertices in orbit order

struct FillStats {
    uint64_t fills = 0;
    uint64_t x0_fills = 0, x1_fills = 0;
    uint64_t x1_max_triangles = 0;
    uint32_t x1_max_rounds = 0;
    uint64_t bfs_max_visited = 0;
    uint64_t max_tot_terms = 0;
};

class MPFill {
public:
    explicit MPFill(const Descriptor& D);

    const Descriptor& descriptor() const { return *D_; }
    // The vertices (point keys) of h iota(t), in orbit order.
    const XS& points(const Tr& u) const;
    // The common vertices of the simplices of s (sorted, orbit order); empty if none.
    XS common(const Str& s) const;
    static bool degenerate(const Str& s);
    // The MP differential on a concrete chain of degree k >= 1 (degenerate faces dropped).
    StrChain boundary(const StrChain& c) const;
    static int64_t augmentation(const StrChain& c);   // degree 0

    // y with d y = z (z a cycle of degree k; k = 0: augmentation 0).  ENGINE_CHECKs d y = z.
    StrChain fill(uint32_t k, const StrChain& z) const;

    // Every translate (h, t) whose simplex contains the vertex P (h over all of g Gamma_P' g^{-1} etc.).
    const std::vector<Tr>& containing(const PointKey& P) const;
    // The least translate containing the simplex x (the row apex).
    const Tr& apex(const XS& x) const;

    const FillStats& stats() const { return stats_; }

private:
    using TotKey = std::pair<XS, Str>;
    using TotChain = std::map<TotKey, int64_t>;
    using XChain = std::map<XS, int64_t>;

    const Descriptor* D_;
    mutable std::map<Tr, XS> points_;
    mutable std::map<PointKey, std::vector<Tr>> containing_;
    mutable std::map<XS, Tr> apex_;
    mutable FillStats stats_;

    TotChain dtot(const TotChain& c) const;
    static void add(TotChain& c, const TotKey& k, int64_t n);
    static void add(XChain& c, const XS& k, int64_t n);
    static void add(StrChain& c, const Str& k, int64_t n);
    XChain fill_x0(const XChain& beta) const;   // 0-chain of augmentation 0 -> 1-chain
    XChain fill_x1(const XChain& beta) const;   // 1-cycle -> the unique 2-chain
    std::vector<XS> triangles_at_edge(const XS& e) const;
    std::vector<PointKey> neighbours(const PointKey& P) const;
};

// Free-module form <-> concrete form of an MP string.
// The concrete string g0 . e_s of the j-th non-degenerate string of S_k.
Str ConcreteString(const InternedMPSet& S, uint32_t k, uint64_t nd_j, const Mat22& g0);
// (g0, non-degenerate index) of a concrete non-degenerate string; ENGINE_CHECK it lies in S_k.
std::pair<GKey, uint64_t> FreeForm(const InternedMPSet& S, const Str& s);

}  // namespace orbifold_mp

#endif
