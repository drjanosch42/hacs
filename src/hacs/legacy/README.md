<!--
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: CC-BY-4.0
-->

# `src/hacs/legacy/` — retired cohomology-layer routes (compiled only with `OC_ENABLE_LEGACY=ON`)

Owner rulings of 2026-10-30 (`docs/implementation/slot24/CLEANUP_plan_1030.md` §2): a retired route is moved here
verbatim (never rewritten), kept compiling by the close-time `cross_config.sh` arms `legacy` and `legacy-shadow`, and
can be reactivated. The default build (`OC_ENABLE_LEGACY=OFF`) carries the block / tape route only.

| File | What | Retired | Package | Wiring |
|---|---|---|---|---|
| `EquivariantCohomologyOnKLegacy.inc` | the LEGACY arm of `compute_cohomology()` (pre-Stage-B: `BlockFlatMatrix::flat_kernel` + `factor_basis_block` + the flat/unflat/expand loops): `compute_cohomology_legacy`, `adopt_legacy_result`, the four `legacy_*` assembly helpers and the legacy-arm-only `cochains_as_multivector` / `projector_as_block`; it prints the `[TIMING] A_Cq/A_Zq kernel` lines | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | included by `EquivariantCohomologyOnK.cpp` under `#if OC_ENABLE_LEGACY`, before the shadow `.inc`; members and `EquivariantCohomologyOptions::{legacy_cohomology_path_forced, set_legacy_cohomology_path_forced, ScopedLegacyCohomologyPath}` declared in the header under the same guard |
| `SimplicialCohomologyLegacy.inc` | `SimplicialComplex::coboundary_operator_block_elt` (and its two instantiations), and `legacy_trivial_factor_basis` (`CochainComplex::compute_cohomology_bases`' former `factor_basis_block` call, now taken only under `engine::elim::use_legacy()`) | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | included by `SimplicialCohomology.cpp` under `#if OC_ENABLE_LEGACY` |
| `CohomologyBlockModelValidation.{h,cpp}` | the whole validation (moved with `git mv`) | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `cmake/legacy.cmake`; `OC_LEGACY_VALIDATION` in `ValidationRegistry.cpp` |
| `BlockModelPipelineLegacyArms.inc` | the `BlockFlatMatrix::flat_kernel` / `factor_basis_block` halves of `BlockModelPipelineValidation` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `SKIP-ARM BlockModelPipelineValidation arm=legacy` when OFF |
| `DenseElimPipelineLegacyArms.inc` | `BuildACq` (A_Cq as a `BlockFlatMatrix`), the legacy `FlatMatrix` / `flat_kernel` comparisons and Part 2 `whole_pipeline` of `DenseElimPipelineValidation` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | OFF: A_Cq from `BuildACqBlockMat`, `SKIP-ARM DenseElimPipelineValidation arm=legacy-real-matrices` and `arm=whole_pipeline` |
| `EquivariantCohomologyOnKShadow.inc` | the comparison half of the `-DENGINE_SHADOW_LEGACY=ON` build mode for `EquivariantCohomologyOnKT::compute_cohomology()`: after the block arm the legacy arm runs as well, and the `H^q` dimensions, the bases C/Z/B/H and the projectors P are compared entrywise (`same_flat`, `shadow_compare_lists`, the member `shadow_compare_with_legacy()`) | 2026-10-31 | S24 CL-L1 (CODEAUDIT-C C-12 step 1) | included at the end of `EquivariantCohomologyOnK.cpp` under `#ifdef ENGINE_SHADOW_LEGACY`, before the explicit instantiations; the member is declared in `EquivariantCohomologyOnK.h` under the same guard |

Since CL-L2 the legacy arm itself lives here too; `EquivariantCohomologyOnK.cpp` keeps a guarded dispatch in
`compute_cohomology()`. An OFF build runs the block arm only.

Value statement: the shadow only compares, so moving it moves no value.

**Reactivate:** configure with `-DOC_ENABLE_LEGACY=ON -DENGINE_SHADOW_LEGACY=ON` (fresh configure); see
`src/ModArithmetic/legacy/README.md`. Deleting a file here is an owner decision.
