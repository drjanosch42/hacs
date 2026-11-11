<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->

# Slot 26 registration fragments (open slot)

One pair per package: `Registry/Slot26/<pkg>.cpp` (`RegisterSlot26<pkg>()`, called explicitly at the tail of `RegisterAllValidations()`)
and `cmake/slot26/<pkg>.cmake` (included by `CMakeLists.txt` after the closed slots' `cmake/sources/*.cmake`). A package edits only its own
pair; the two shared edits are one line each. Folded at the slot-26 close (one `Slot26.cpp`, the sources into `cmake/sources/`; this text
joins `docs/implementation/registry-slot-readmes-9-25.md`, renamed `-9-26`).

| Package | Fragment | Validations (tier) |
|---|---|---|
| Z1 (UP-Z (a), `slot26/Z1_report.md`) | `HeckeZUp.cpp`, `cmake/slot26/HeckeZUp.cmake` | `HeckeZUpValidation` (standard) |
| Z9 (HZ-LOC-PAR, `slot26/Z9_report.md`) | `HeckeZLocalPar.cpp`, `cmake/slot26/HeckeZLocalPar.cmake` | `HeckeZLocalParValidation` (standard) |
| HZ-EISUP (`slot26/HZEISUP_report.md`) | `HeckeZEisUp.cpp`, `cmake/slot26/HeckeZEisUp.cmake` | `HeckeZEisUpValidation` (standard) |
