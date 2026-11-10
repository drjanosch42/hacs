// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKEZ_LOCAL_SOURCE_H
#define HECKE_EXT_HECKEZ_LOCAL_SOURCE_H

// ===========================================================================
// HeckeZLocalSource -- slot-25 HZ-LOC: the LEVEL-1 source of the lattice module (Integral/HeckeZLocal.h) and the per-block
// invariants of the census cross-check. It lives in HeckeExt/ because it calls the shipped Z/ell^K module route
// (ringalg::RunModule, ringalg::RationalSystems, ringalg::Localize); Integral/ may not include HeckeExt/
// (check_heckeext_includes.sh rule 1), so the routing that needs both is here.
//
// LatticeFromH1Level1: ringalg::RunModule(n, ell, max(K, 2), P) (native: u32 while ell^K < 2^32, the u64 regimes above, ell >= 5,
// n >= 1). Its module is H^1(SL_2(Z); Sym^n Z/ell^K) = H^1 / ell^K ([C, thm:descentcert](1): no ell-torsion in H^2 at ell >= 5).
// It is the lattice L / ell^K iff it is free of rank b1 (b1 = 2d - 1 at level 1; each ell-torsion summand of H^1 adds a summand);
// otherwise ell is a TORSION PRIME of the cell, and the module is returned only as `h1_level1_torsion(A1_of_computed)` over F_ell
// (is_lattice = false: nothing certified is read from it, 01 P2a-2).
// LoadLattice (P plus T_ell by ReadPrimes, audit F1; at a level N where the export refuses T_ell, P and t_ell = omitted(export:..)):
// N = 1 and ell >= 5 -> the H^1 route; at a torsion prime (or N > 1, or ell in {2, 3}) -> the EXPORT free block.
// ell^K >= 2^63 -> refused(word_modulus).
// ===========================================================================

#include <cstdint>
#include <vector>

#include "../Integral/HeckeZLocal.h"

namespace heckez {
namespace local {

LatticeModule LatticeFromH1Level1(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1);

// h1_side (optional): at a level-1 torsion prime the H^1 module as A1_of_computed (for the torsion-routing check).
// (S26 Z1) allow_bad_primes: p | N in P is U_p (the export only; never on the level-1 H^1 route, where N = 1).
LatticeModule LoadLattice(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                          LatticeModule* h1_side = nullptr, bool allow_bad_primes = false);

// (audit F2) one (N, n, ell) read with the K-escalation: K is raised (at most max_steps times) while the at-root read is
// open(precision).  When the escalation stops on a module that is not served (refused(export_word), refused(word_modulus),
// blocked(export), the export's open(precision) above its K_max) the read's root and C_T carry THAT word, and `escalation` says where.
// LoadLattice applies ReadPrimes (T_ell in the read set, audit F1).
struct CellLocalRead {
    LatticeModule M;                 // the module of the last served read (or the first refusal when nothing was read)
    LocalRead L;
    bool read = false;
    std::string escalation;          // empty: no escalation stop; else the word and the K where it stopped
};
CellLocalRead ReadCellLocal(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                            const ReadOptions& opt, uint32_t max_steps = 3, LatticeModule* h1_side = nullptr);

// The per-block invariants of A'_1 on H^1(SL_2(Z); Sym^n F_ell) at the F_ell-rational joint systems of the operators P (the census
// convention: P = {2, 3, 5, 7, ell}), computed independently of the census route: the blocks by ringalg::RationalSystems /
// ringalg::Localize at K = 2 reduced mod ell, the invariants by this package's Howell spans over F_ell (length, Hilbert function of
// the m-adic filtration, embedding dimension, socle dimension).
struct BlockInvariants {
    std::vector<uint64_t> system;    // a_p mod ell, in P order
    uint32_t dim_E = 0;              // rank of the block e_m H^1 / ell
    uint32_t length = 0;             // dim A'_1(e_m) (residue degree 1)
    std::vector<uint32_t> hilbert;
    uint32_t embdim = 0;
    uint32_t socle = 0;
    bool gorenstein = false;
    uint32_t loewy() const { return static_cast<uint32_t>(hilbert.size()); }
};
struct Level1Blocks {
    uint32_t h1_rank = 0;            // rank of H^1 / ell (b1 + the ell-torsion summands)
    std::vector<BlockInvariants> blocks;
};
Level1Blocks BlockInvariantsLevel1(uint32_t n, uint64_t ell, const std::vector<uint32_t>& P);

}  // namespace local
}  // namespace heckez

#endif
