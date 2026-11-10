// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-5) HeckeHidaControlValidation (standard): Hida
// theory at level one, part 2 -- [C] v1.13.0-pre.2 sec:hidafull,
// rmk:hidacheckable2 (Hi7), (Hi8) with its arms (a), (b), (c), (Hi9);
// def:hidalattice and its variant (H); and the torsion pre-pass beyond 2^64
// (prop:torsionprepass on the engine::ladic digit route).  Writes nothing; one
// HIDA_TOWER line per (cell, level, lattice) on stdout (not published).
//
// The ell = 7 mod-ell^3 pair 10 <-> 304 of (Hi7) is NOT in the suite: measured at
// 187-189 s (1.2 GB), over the slot-13 plan's 3-minute cap for a slow arm; its
// held result is recorded in docs/implementation/slot13/WP-5_report.md.
#pragma once
#ifndef HECKE_HIDA_CONTROL_VALIDATION_H
#define HECKE_HIDA_CONTROL_VALIDATION_H

#include <cstdint>

uint32_t HeckeHidaControlValidation();

#endif
