// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// (hecke-extensions slot 23, TER-1) TernaryF3Validation: the F_3 rung T0.
//   (0) the bit-sliced and byte F_3 arithmetic exhaustively on the 9 pairs, on every row path;
//   (1) binary::TritMatrix (rref / echelon / rank / kernel / product) and the opt-in ENGINE_TERNARY_F3 route of
//       engine::elim are bit for bit the engine at p = 3 -- rank, pivots, RREF, kernel basis, product -- on random
//       matrices and the real A_Cq relation matrices of the 7-vertex SL_2(Z) model mod 3 (>= 100), every row path,
//       uint32_t and uint64_t; a whole compute_cohomology() at ell = 3 with the route on equals the route off;
//   (2) the degree-1 amalgam reader's F_3 route (l23::h1_mod at ell = 3, k = 1, Ops3F) equals its Ops3<1> route on
//       every compared cell (h0, length, every image length), the negative control mutate = 1 included;
//   (3) TERNARY_F3_TIMING / TERNARY_F3_READER lines (reported, never gated).
// (S24 CL-G3, CODEAUDIT C-04) The former part (3), engine::ring3k::Mod3k<N> against ModCtx Z/3^K and nf::BigInt, was
// deleted with Mod3k.h: its one consumer (Ring3k's unit inverse) now uses ModCtx::inv.
#include <cstdint>

uint32_t TernaryF3Validation();
