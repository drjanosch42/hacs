<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->

# Validation registration fragments

`RegisterAllValidations()` (`ValidationRegistry.cpp`) registers the validations of the core first, then the slot fragments in slot order.

- **Closed slots 9-25: one file each**, `Registry/Slot<S>.cpp`. It holds the former per-package registration functions
  (`RegisterSlot<S><pkg>()`, kept verbatim) and the aggregate `RegisterSlot<S>()`, which calls them in the old call order; the registry calls
  `RegisterSlot9()` ... `RegisterSlot25()`. The call order fixes the `--list` order, so **do not reorder**. Their sources are listed per module
  in `cmake/sources/<Module>.cmake` (`Registry.cmake` lists the `Slot<S>.cpp`; `Special.cmake` holds the library-target sources,
  compile flags and `OC_BUILD_NOTEBOOK`-conditional sources). The old per-package READMEs are in
  `docs/implementation/registry-slot-readmes-9-25.md`.
- **Slot 26 (open): one pair per package**, `Registry/Slot26/<pkg>.cpp` + `cmake/slot26/<pkg>.cmake`, created by the first slot-26 package (no empty
  directories now); a package edits only its own pair; the registry calls `RegisterSlot26<pkg>()` explicitly at the tail of
  `RegisterAllValidations()` (never a static initialiser) and `CMakeLists.txt` includes the cmake fragment (two one-line shared edits). Each slot's
  `README.md` for the open slot records its roster. At the slot-26 close the same fold is applied (one `Slot26.cpp`, the sources into `cmake/sources/`).
- A fragment includes validation headers only (`check_heckeext_includes.sh`). A validation gated on `OC_ENABLE_LEGACY` is registered through
  `OC_LEGACY_VALIDATION` (`LegacyValidation.h`).
- Any package that registers a validation, verb, type, setting or CLI token regenerates `docs/latex/cli_manual/`
  (`docs/latex/tools/gen_cli_manual_tables.sh <binary>`) and updates the cluster kits' expected roster count (`EXPECT_NAMES`, development repository) in the same commit.
