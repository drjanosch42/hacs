<!--
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: CC-BY-4.0
-->

# `Hecke/legacy/` — retired Hecke routes (compiled only with `OC_ENABLE_LEGACY=ON`)

Owner rulings of 2026-10-30 (`docs/implementation/slot24/CLEANUP_plan_1030.md` §2, §3): a retired route is
moved here verbatim (never rewritten), kept compiling by the close-time `cross_config.sh` `legacy` arm, and can be
reactivated. The default build (`OC_ENABLE_LEGACY=OFF`) carries the new route only and **refuses** a request for a
retired one (`REFUSED <request>: compiled out (OC_ENABLE_LEGACY=OFF)` on stderr, exit 1).

| File | What | Retired | Package | Wiring |
|---|---|---|---|---|
| `HeckeOperatorOnKLegacy.inc` | the LEGACY (pre-Phase-3) dense field arm of `HeckeOperatorOnKT::compute()`: `flatten_cochain_legacy`, `assemble_theta_matrix_legacy`, `compute_legacy`, `adopt_legacy_result` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | included by `HeckeOperatorOnK.cpp` under `#if OC_ENABLE_LEGACY`, before the shadow `.inc`; the `Legacy*` types, the members and `HeckeOperatorOptions::{legacy_hecke_operator_path_forced, set_…, ScopedLegacyHeckeOperatorPath}` declared under the same guard |
| `HeckeInvolutionLegacy.inc` | the two `FlatMatrix` pivot routes of `HeckeInvolution.cpp` under `engine::elim::use_legacy()` (the (iii-c) carry check, PERF1 audit R2, and `build_split`'s `pivots_of`) and the counter `iota_carry_legacy_calls()` | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | included inside `namespace hecke` under `#if OC_ENABLE_LEGACY`; `HeckeInvolutionValidation`'s `IOTA_CARRY_ROUTE` block is the arm `SKIP-ARM HeckeInvolutionValidation arm=IOTA_CARRY_ROUTE` when OFF |
| `HeckeThetaBlockValidation.{h,cpp}`, `BarCocycleSolverValidation.{h,cpp}` | the two whole validations (moved with `git mv`) | 2026-10-31 | S24 CL-L2 (CODEAUDIT-C C-12 step 2) | `cmake/legacy.cmake`; `OC_LEGACY_VALIDATION` in `ValidationRegistry.cpp` |
| `HeckeLevelNGenOrderLegacy.inc` | the as-built generator order of `level::SparseCokernel` (slots 15–17: S then ST) and its rollback arm `HECKE_LEVEL_N_GEN_ORDER=as_built`; `GeneratorOrder`, `CurrentGeneratorOrder`, `GeneratorOrderName`, `ScopedGeneratorOrder` (slot 18 WP-5 (b), `d4265bae`) | 2026-10-30 | S24 CL-G2 (CODEAUDIT-B B7) | included at the end of the order section of `HeckeLevelNSparse.cpp` under `#if OC_ENABLE_LEGACY`; declarations in `HeckeLevelNSparse.h` under the same guard |
| `HeckeLevelNOrderLegacyArms.cpp` | the (W18o-as_built) bit-identity twin of `HeckeLevelNSparseValidation` (as_built vs sorted on 12 cells, plus the Hecke-matrix twins) | 2026-10-30 | S24 CL-G2 | `cmake/legacy.cmake`; called under `oc::legacy::RunLegacyArm("HeckeLevelNOrderValidation", "as_built")` |
| `HeckeOperatorLeversLegacy.h` | the setters and `Scoped*` guards of the four slot-13/14 levers: `set_u_ell_low_rank_mode` / `ScopedUEllLowRank` (`HECKE_U_ELL_LOW_RANK`, slot 13 WP-4, `3015a534`), `set_coset_streaming` / `ScopedCosetStreaming` (`HECKE_COSET_STREAM`), `set_parallel_setup` / `ScopedParallelSetup` (`HECKE_PARALLEL_SETUP`, both slot 14 WP-10, `3580160a`), `set_rank_one_closed_form` / `ScopedRankOneClosedForm` (`HECKE_RANK1_CLOSED_FORM`, slot 14 G0, `a249a6d6`), and the validation hook `ScopedParallelSetupMinWork` | 2026-10-31 | S24 CL-R1 (CODEAUDIT-B B6) | included at the end of `HeckeOperatorOnK.h` under `#if OC_ENABLE_LEGACY`; the thread-local state and its capture in `HeckeOperatorOnK.cpp` under the same guard; the env reads are `engine::lever::RetiredLever` |
| `HeckeCosetTablesMaterialised.inc` | the materialised `ρ_n` tables of `HeckeCosetTablesT` (`HECKE_COSET_STREAM=0`): the three fills (`materialise_*`) and the dense reference accessors `rho_s_inv`, `rho_t`, `rho_cocycle`, `rho_cocycle_upper` | 2026-10-31 | S24 CL-R1 | included at the end of `HeckeCosetTables.cpp` under `#if OC_ENABLE_LEGACY`; declarations and the four table members in `HeckeCosetTables.h` under the same guard |
| `HeckeUEllStreamValidation.cpp` | the whole validation (moved with `git mv`; include paths only): the coset-streaming and parallel-setup levers against their opt-outs, alone and composed with the U_ell lever | 2026-10-31 | S24 CL-R1 | `cmake/legacy.cmake`; registered through `OC_LEGACY_VALIDATION` (`Registry/Slot14/WP10.cpp`) |
| `HeckeUEllLowRankLegacyArms.cpp` | the lever arms (A)–(G) of `HeckeUEllLowRankValidation` (the U_ell lever against `ScopedUEllLowRank(false)`, its forced-mode refusals, the module route, the guards, the legacy arm, the contragredient fibre, the RSS print) | 2026-10-31 | S24 CL-R1 | `cmake/legacy.cmake`; called under `oc::legacy::RunLegacyArm("HeckeUEllLowRankValidation", "lever")` |
| `HeckeOperatorOnKShadow.inc` | the comparison half of the `-DENGINE_SHADOW_LEGACY=ON` build mode for `HeckeOperatorOnKT::compute()`: after the block arm the legacy dense arm runs as well, and `Theta_flat`, `Theta_eq`, `A^{(q)}` and the certificate counters are compared entrywise (the member `shadow_compare_with_legacy()`); the legacy arm itself moved to `HeckeOperatorOnKLegacy.inc` in CL-L2; the one-line `needs_omega = true` hook stays in `HeckeOperatorOnK.cpp` | 2026-10-31 | S24 CL-L1 (CODEAUDIT-C C-12 step 1) | included at the end of `HeckeOperatorOnK.cpp` under `#ifdef ENGINE_SHADOW_LEGACY` (admitted only with `OC_ENABLE_LEGACY=ON`), before the explicit instantiations; the member is declared in `HeckeOperatorOnK.h` under the same guard |

Value statement: the generator order is value-free (the stored RREF is unique), so retiring it moves no value. In an
OFF build `HeckeLevelNSparseValidation` pins `PresentationDigest` of the sorted order on the same 12 cells (W18o-pin)
and prints `SKIP-ARM HeckeLevelNOrderValidation arm=as_built: OC_ENABLE_LEGACY=OFF`.

**Reactivate:** configure with `-DOC_ENABLE_LEGACY=ON` (fresh configure); then `HECKE_LEVEL_N_GEN_ORDER=as_built`
behaves as in slots 18–24 and the (W18o-as_built) arm runs. Deleting a file here is an owner decision.

**CL-R1 value statement.** Each of the four levers was value-neutral by its own validation (bit-identity with its
opt-out since slots 13/14), so retiring them moves no value. What stays in the default build: the dense U_ell route
(the route for `ell ∤ N` and the non-SymN fibres, chosen by `Auto`), the dense factoriser (non-field rings), the
serial loop (`for_each_index(…, false, …)`), and the refusals in `check_u_ell_lever_preconditions`. In an OFF build
`HECKE_U_ELL_LOW_RANK=0`, `HECKE_COSET_STREAM=0`, `HECKE_PARALLEL_SETUP=0` and `HECKE_RANK1_CLOSED_FORM=0` are refused
at their first read; `HeckeUEllStreamValidation` prints `[SKIP]`, `HeckeUEllLowRankValidation` runs its (M) arm (the
default `Auto` mode) and prints `SKIP-ARM HeckeUEllLowRankValidation arm=lever: OC_ENABLE_LEGACY=OFF`. The `--bench`
flags `--[no-]u-ell-low-rank`, `--[no-]coset-stream`, `--[no-]parallel-setup` are gone in every build.
