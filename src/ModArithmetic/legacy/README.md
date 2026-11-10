<!--
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: CC-BY-4.0
-->

# `ModArithmetic/legacy/` — retired ModArithmetic routes (compiled only with `OC_ENABLE_LEGACY=ON`)

Owner rulings of 2026-10-30 (`docs/implementation/slot24/CLEANUP_plan_1030.md` §2): a retired route is moved here
verbatim (never rewritten), kept compiling by the close-time `cross_config.sh` arms `legacy` and `legacy-shadow`, and
can be reactivated. The default build (`OC_ENABLE_LEGACY=OFF`) carries the engine route only.

| File | What | Retired | Package | Wiring |
|---|---|---|---|---|
| `BlockFlatMatrix.h`, `BlockFlatMatrix.cpp` | `BlockFlatMatrixT` (the deprecated block container) and the free function `factor_basis_block`; moved with `git mv` (include paths only) | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `cmake/legacy.cmake` (`ModArithmetic`); `check_matrix_includes.sh` admits `BlockFlatMatrix.h` only from files under a `legacy/` directory |
| `BlockFlatMatrixValidation.h`, `BlockFlatMatrixValidation.cpp` | the whole validation (moved with `git mv`) | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `cmake/legacy.cmake`; registered through `#if OC_ENABLE_LEGACY` / `OC_LEGACY_VALIDATION` in `ValidationRegistry.cpp` (`[SKIP] BlockFlatMatrixValidation: OC_ENABLE_LEGACY=OFF`) |
| `DenseElimLegacy.inc` | `engine::elim::ScopedLegacy` and `legacy_forced()` (the thread-local half of `ENGINE_BACKEND=legacy`) | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | included by `DenseElim.cpp` under `#if OC_ENABLE_LEGACY`; declarations, and `use_legacy()`, in `DenseElim.h` under the same guard; the pool hand-over in `Engine/EngineRuntime.cpp` too |
| `FlatMatrixLegacy.inc` | the pre-Phase-2 scalar elimination of `FlatMatrixT`: `_gaussian_elimination_core`, `flat_to_2d` / `flat_from_2d`, and the former `use_legacy()` branches / `legacy_route` lambdas as the members `_legacy_row_echelon_form`, `_legacy_rref_and_transform`, `_legacy_elimination_factor_and_apply`, `_legacy_inverse` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | included by `FlatMatrix.cpp` before the first member definition under `#if OC_ENABLE_LEGACY`; members declared in `FlatMatrix.h` under the same guard; the call sites keep a guarded one-line call |
| `MicroBenchLegacy.inc` | the `legacy` column of the micro-benchmarks: `GemmLegacy`, `GemvLegacy`, `RrefLegacy`, `KernelLegacy`, `TransposeLegacy` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | included by `PerformanceSmokeValidation.cpp` under `#if OC_ENABLE_LEGACY`; an OFF build parses no `legacy` backend (`--bench micro --backend legacy` is a usage error, exit 1) |
| `FlatMatrixLegacyArms.inc` | `FlatMatrixValidation`'s Q11 third class (`BlockFlatMatrixT::flat_image` of `alpha*I`) | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `SKIP-ARM FlatMatrixValidation arm=Q11-BlockFlatMatrix` when OFF |
| `SubstrateDifferentialLegacyArms.inc` | the third substrate (`BlockFlatMatrixT`) of `SubstrateDifferentialValidation` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `SKIP-ARM SubstrateDifferentialValidation arm=BlockFlatMatrix` when OFF; the Matrix-vs-FlatMatrix checks run in both builds |
| `DenseElimLegacyArms.inc` | the `ScopedLegacy` / `factor_basis_block` halves of `DenseElimValidation` / `DenseElimSlowValidation` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `SKIP-ARM <name> arm=legacy` when OFF; the retrofit (engine-route `FlatMatrix`) checks run in both builds |
| `BlockEliminationLegacyArms.inc` | the legacy `BlockFlatMatrix` / `factor_basis_block` / `change_of_basis_fm`-under-`ScopedLegacy` halves of `BlockEliminationValidation` / `BlockEliminationSlowValidation` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `SKIP-ARM <name> arm=legacy` when OFF; the engine checks are the same set in both builds |
| `ShadowLegacy.inc` | the comparison half of the `-DENGINE_SHADOW_LEGACY=ON` build mode (matrix-engine Phase 2) for `FlatMatrix.cpp` and `BlockFlatMatrix.cpp`: `shadow_check` (memcmp of two buffers), `shadow_mismatch`, `shadow_check_equal` (entrywise `FlatMatrix`), and the per-site helpers `shadow_park` / `shadow_restore` (the five entrywise operators), `shadow_check_ref` (`compute_row_echelon_form`), `shadow_check_pair` (`get_rref_and_transform_matrix`, `compute_elimination_factor_and_apply`) | 2026-10-31 | S24 CL-L1 (CODEAUDIT-C C-12 step 1) | included inside each file's anonymous namespace under `#ifdef ENGINE_SHADOW_LEGACY`; the call sites keep a one-line `#ifdef ENGINE_SHADOW_LEGACY` hook |

What stays in the `.cpp` files: the entrywise `run_legacy` loops of `FlatMatrix.cpp` (they are also the `p < 2` /
empty-operand fallback of the five entrywise operators). Each `*LegacyArms.inc` is included at file scope by its
validation under `#if OC_ENABLE_LEGACY` and reopens that file's anonymous namespace after its own `#include`.

Value statement (CL-L2): every moved branch is false at the defaults (`backend_is_legacy()` is false with the variable
unset, `legacy_forced()` outside the validations that set it), so the default route's text is unchanged; the one live
caller of `factor_basis_block` (`CochainComplex::compute_cohomology_bases`, unit block rows) now calls
`engine::elim::complement_basis`, `factor_basis_block`'s rule on the same augmented matrix (pinned entrywise by
`DenseElimValidation`; golden tier A and B unchanged). Value statement (CL-L1): the shadow only compares; it never feeds a value back (the engine result is restored after the
comparison), so moving it moves no value. In an OFF build it does not exist.

**Reactivate (CL-L2):** configure with `-DOC_ENABLE_LEGACY=ON` (fresh configure); then `ENGINE_BACKEND=legacy`, console
`set backend = "legacy"` and `--bench micro --backend legacy` run the legacy route as before, and every gated name and
arm runs. In an OFF build each is refused: `REFUSED ENGINE_BACKEND=legacy: compiled out (OC_ENABLE_LEGACY=OFF)` (exit 1),
`NotImplementedError` (exit 5), `--bench: --backend legacy: compiled out (OC_ENABLE_LEGACY=OFF)` (exit 1).

**Reactivate the shadow:** configure with `-DOC_ENABLE_LEGACY=ON -DENGINE_SHADOW_LEGACY=ON` (fresh configure); `cmake` refuses
`ENGINE_SHADOW_LEGACY=ON` without `OC_ENABLE_LEGACY=ON` (`FATAL_ERROR` naming both options). Then run `--all`: every
retrofitted call runs both routes and throws `std::logic_error` (`ENGINE_SHADOW_LEGACY: …`) on the first differing
entry. Deleting a file here is an owner decision.
