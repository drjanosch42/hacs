// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// CwLayerValidation.h -- part (9) of NumberFieldValidation: the Coates--Wiles
// layer route (slot-19 WP-NF2; NumberField/CwLayer.h) against R19-7
// (docs/implementation/slot19/R19-7_report.md, r197/out/). Compiled into
// cohomology_app, not into the STL-only NumberField library.
#include <cstdint>
#include <string>

// Runs the default arm and, when NUMBERFIELD_LAYER_ALL=1 / NUMBERFIELD_LAYER_CELLS
// are set, the read-only heavier arms. Adds to *checks / *fails.
void NumberFieldCwLayerGate(const std::string& repo_root, uint32_t* checks, uint32_t* fails);
