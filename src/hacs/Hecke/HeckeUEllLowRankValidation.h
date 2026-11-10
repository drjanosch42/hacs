// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_U_ELL_LOW_RANK_VALIDATION_H
#define HECKE_U_ELL_LOW_RANK_VALIDATION_H

#include <cstdint>
#include <functional>
#include <string>

#include "../../ModArithmetic/LegacyConfig.h"

// (hecke-extensions slot 13, WP-4) The U_ell memory lever
// (HeckeOperatorOptions::ScopedUEllLowRank): the opt-in rank-<= k route of
// HeckeOperatorOnK against the default route, entrywise.  Returns 0 on success.
uint32_t HeckeUEllLowRankValidation();

#if OC_ENABLE_LEGACY
// (S24 CL-R1) The lever arms (A)-(G) of the RETIRED lever (Hecke/legacy/HeckeUEllLowRankLegacyArms.cpp, ON builds).
void HeckeUEllLowRankLeverArms(const std::function<void(bool, const std::string&)>& check);
#endif

#endif // HECKE_U_ELL_LOW_RANK_VALIDATION_H
