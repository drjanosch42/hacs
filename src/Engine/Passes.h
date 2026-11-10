// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// Passes.h -- passes 1-6 of plan D4, dependency analysis, wavefronts, arena
// =====================================================================
//
//   1 constant-fold on BlockRef kinds      (kind-GEMM -> AXPY / GEMM nodes)
//   2 CSE on (op, operand content keys, view) + product/AXPY fusion
//   3 layout assignment per consumer + physical TRANSPOSE insertion (D3)
//   4 k-blocking (k_bound = min(k, k_max); optional node split, ENGINE_KSPLIT)
//   5 batching: same-key independent GEMMs of one wavefront -> BATCHED_GEMM
//   6 device assignment by the static cost model (EngineCostModel.h),
//     per group of would-be batch members (Phase 5: cpu | gpu | hybrid)
//   7 residency: H2D / D2H transfer nodes, hoisted uploads, epilogue
//     downloads, mixed-writer conflicts forced to the CPU (Phase 5)
//   + dependency analysis (RAW / WAR / WAW on overlapping windows, result
//     slots), wavefront levels, Temp lifetimes in levels, arena offsets.
//
// compile() is idempotent on its input program (a compiled program is not
// recompiled) and a pure function of (program, options): the same recording
// under the same options yields the same node list, levels and offsets.

#include "Program.h"

namespace engine {
namespace tape {

template<class Elt>
void compile(Program<Elt>& prog, const PassOptions& opts);

template<class Elt>
void compile(Program<Elt>& prog);   // PassOptions::from_env()

// Per-node access sets, exposed for the validations.
struct Access { ViewRef v; bool write; };
template<class Elt>
void node_accesses(const Program<Elt>& prog, const Node<Elt>& n, std::vector<Access>& out);

// Structural FNV-64 of a program (nodes, buffers, index lists, canonicalised
// content keys); computed by Recorder::finish() and re-computable here.
template<class Elt>
uint64_t structural_hash(const Program<Elt>& prog);

// Fresh program identity, assigned by compile() (keys the executor's device state).
uint64_t next_program_uid();

} // namespace tape
} // namespace engine
