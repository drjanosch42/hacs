// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// =====================================================================
// BlockModelPipelineValidation -- the Phase 3 block model on the REAL
// pipeline matrices, entrywise vs the legacy path, timed (plan Phase 3,
// "suggested first deliverable": EquivariantCohomologyOnK.cpp:214 / :365)
// =====================================================================
//
// No consumer is migrated here.  The validation assembles A_C1 / A_Z1 of the
// SL(2,Z) model at n_sym = 128 (ell = 134217757) through BlockBuilder from
// the very relations EquivariantCohomologyOnK uses (I at lhs, -M at rhs,
// I - M on the diagonal; pencils beta_lhs*I - beta_rhs*M for A_Zq; every
// rho_n(g_r) interned ONCE), runs engine::blk::rref + kernel_basis, and
// compares ENTRYWISE with BlockFlatMatrix::flat_kernel on the legacy
// (pre-Phase-2, engine::elim::ScopedLegacy) and retrofitted paths.  It then
// runs the real compute_cohomology() once, takes the actual Z^1_eq and
// C^1_eq bases, and compares engine::blk::factor_basis (W^1 = C^1/Z^1) and
// the H^1 = Z^1/B^1 projector with the legacy factor_basis_block output,
// again entrywise.  Every arm is timed and the model-alone timings are
// printed next to the legacy ones.  Returns 0 on success.
//
// BuildACqBlockMat / BuildAZqBlockMat are the assembly helpers the Stage B
// migration can call (ENGINE_API.md Sec. 9.7 shows the same code).

#include <cstdint>
#include <memory>

#include "../ModArithmetic/BlockMat.h"
#include "LocalSystemSimplicialComplex.h"

// A_Cq exactly as EquivariantCohomologyOnK::compute_C_eq_bases assembles it,
// honouring the item-7.4 vacuous-relation short-circuit setting.
engine::BlockMat<uint32_t> BuildACqBlockMat(std::shared_ptr<engine::BlockPool<uint32_t>> pool,
                                            const LocalSystemSimplicialComplex& lssc, uint32_t q);

uint32_t BlockModelPipelineValidation();
