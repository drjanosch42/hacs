// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (hecke-extensions slot 22, BIN-3) the degree-1 amalgam reader (AmalgamReader.h) against the frozen outputs of
// ladic2.py (even n <= 200, ell = 2 at K = 240, ell = 3 at K = 160) and oddn_l2.py (odd n <= 101), bit for bit; the
// closed forms of [C] prop:oddnltwo / lem:oddnsplit / prop:oddnhecke, T_p = 1 + p on H^1(Gamma; Sym^n F_ell^2)
// ([C] prop:ltwothreescalar), rank T_ell <= 2 ([C] lem:tellrank), the census gates on every lattice cell, the
// ell = 3 Lambda^+- data, the adaptive precision rule and negative controls.  Returns 0 on success (the failure count otherwise).
#include <cstdint>
uint32_t L23AmalgamReaderValidation();
