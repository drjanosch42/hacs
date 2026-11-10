// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_HECKEZ_RECORD_SOURCE_H
#define HECKE_EXT_HECKEZ_RECORD_SOURCE_H

// ===========================================================================
// HeckeZRecordSource -- slot-25 HZ-REC: the SHIPPED sources of the record builder (Integral/HeckeZRecord.h, RecordSources).
// HZ-LOC audit F8: the level-1 lattice source (HeckeZLocalSource.h, ReadCellLocal) and the Bernoulli route (HeckeHidaLambda.h,
// BernoulliOverK) live in HeckeExt/, which Integral/ may not include (check_heckeext_includes.sh rule 1).  The builder therefore takes
// two function objects; this file supplies them, and every caller (HeckeZRecordValidation, Campaign/CampaignHeckeZ, the console verb
// `hecke_structure`, HeckeZCertDumpValidation) passes DefaultRecordSources().  Nothing here computes a value: it forwards.
// ===========================================================================

#include "../Integral/HeckeZRecord.h"

namespace heckez {
namespace record {

RecordSources DefaultRecordSources();

}  // namespace record
}  // namespace heckez

#endif
