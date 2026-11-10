// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_ORDER_VALIDATION_H
#define HECKE_LEVEL_N_ORDER_VALIDATION_H
// (hecke-extensions slot 18, WP-5) Arms of HeckeLevelNSparseValidation (no new registered name; [C] sec:levelN,
// rmk:levelNscale (4), the code map's "Planned at v1.18.0-pre.1" items (b) and (d)):
//   (W18o) THE GENERATOR ORDER IS VALUE-FREE, a gate: SparseCokernel built from the same generators under
//          GeneratorOrder::kAsBuilt (slots 15-17) and kLastColumnDescending (shipped since slot 18) has the same
//          pivots, free columns, RREF rows entry for entry, ranks, h0, dependent count, final_nnz and
//          PresentationDigest, and CokernelHeckeMatrices (sparse route) are bit-identical under both orders; the
//          work (merge_ops, occ_entries) is reported per cell, and on the composite levels with a large prime factor
//          (the W5-F1 shape, P(N) = 43) the sorted order must not cost more occurrence-list entries than the old one.
//   (W18g) Gamma1LevelN::contains_wide: exact (agrees with contains() on int64 inputs, members and non-members), a
//          product beyond int64 in Gamma_1(N) escalates through SubgroupProductForRho with the residues of an
//          independent u128 computation, the old width (kForceNarrow) refuses it loudly, and the CosetLevel default
//          (the pre-slot-18 behaviour on Gamma_1(N)) refused it too (negative control); on Gamma_1(N) cells the
//          forced-__int128 orbit-wise generators equal the shipped ones entry for entry.
// HeilbronnChi's wide route (now Gamma0LevelN::cocycle_wide) is gated by HeckeNebentypusValidation's existing
// forced-wide arm (ScopedForceWideHeilbronn), unchanged.  Adds its checks to `checks`; returns its failure count.
//
// (S24 CL-G2, owner ruling 2 of 2026-10-30) The as-built order is RETIRED (Hecke/legacy/).  (W18o) now has two parts:
//   (W18o-pin)  every build: on each cell, PresentationDigest of the (only) sorted order equals the value recorded from
//               the slot-24 binary before the retirement (the bit-identity gate's figures, so a change of the RREF shows);
//   (W18o-as_built)  OC_ENABLE_LEGACY=ON only (legacy/HeckeLevelNOrderLegacyArms.cpp): the slot-18 twin above, as_built
//               against sorted, unchanged; an OFF build prints `SKIP-ARM HeckeLevelNOrderValidation arm=as_built: ...`.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "../../ModArithmetic/LegacyConfig.h"

uint32_t HeckeLevelNOrderArms(uint32_t& checks);

// (W18o) one cell: level N, Sym^n, F_p; gate_occ: the W5-F1 shape (as_built arm only); fnv: the pinned digest.
struct LevelNOrderCell {
    uint64_t N;
    uint32_t n, p;
    bool gate_occ;
    uint64_t fnv;
};

#if OC_ENABLE_LEGACY
// legacy/HeckeLevelNOrderLegacyArms.cpp: (W18o-as_built) on `cells`, plus the Hecke-matrix twins.
void HeckeLevelNOrderAsBuiltArms(const std::vector<LevelNOrderCell>& cells, const std::function<void(bool, const std::string&)>& check,
                                 std::vector<std::string>& lines);
#endif

#endif // HECKE_LEVEL_N_ORDER_VALIDATION_H
