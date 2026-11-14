<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->

# Slot 27 registration fragments (open slot)

One pair per package: `Registry/Slot27/<pkg>.cpp` (`RegisterSlot27<pkg>()`, called explicitly at the tail of `RegisterAllValidations()`)
and `cmake/slot27/<pkg>.cmake` (included by `CMakeLists.txt` after the closed slots' `cmake/sources/*.cmake`). A package edits only its own
pair; the two shared edits are one line each. Folded at the slot-27 close (into `Slot27.cpp`, the sources into `cmake/sources/`; this text
joins `docs/implementation/registry-slot-readmes-9-26.md`, renamed `-9-27`).

| Package | Fragment | Validations (tier) |
|---|---|---|
| TS-FIX / TS (`slot27/TS_design.md`) | `HeckeZThetaSearch.cpp`, `cmake/slot27/HeckeZThetaSearch.cmake` | `HeckeZThetaSearchValidation` (standard) |
| EXW (`slot27/EXW_design.md`) | `HeckeZExportWord.cpp`, `cmake/slot27/HeckeZExportWord.cmake` | `HeckeZExportWordValidation` (standard) |
| BIGNTT (`slot27/BIGNTT_design.md`) | `BigIntNtt.cpp`, `cmake/slot27/BigIntNtt.cmake` (+ `NumberField/BigNtt.cpp` into the NumberField target) | `BigIntNttValidation` (standard) |
| EISUPb (`slot26/HZEISUPb_design.md` v0.4) | `HeckeZEisUpChar.cpp`, `cmake/slot27/HeckeZEisUpChar.cmake` (+ `Integral/HeckeZEisUpChar.cpp`) | `HeckeZEisUpCharValidation` (standard) |
