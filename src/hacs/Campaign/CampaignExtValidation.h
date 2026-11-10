// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef CAMPAIGN_EXT_VALIDATION_H
#define CAMPAIGN_EXT_VALIDATION_H

// hecke-extensions slot 5, WP-2 / E2 cont. -- the campaign extension seam's
// acceptance gate ([X] Sec. 5 `HeckeExtCampaignValidation`).  Returns 0 on
// success, else the number of failing checks.  Registered in
// ValidationRegistry.cpp; see Campaign/CampaignExt.h for the seam itself.

#include <cstdint>

uint32_t HeckeExtCampaignValidation();

#endif // CAMPAIGN_EXT_VALIDATION_H
