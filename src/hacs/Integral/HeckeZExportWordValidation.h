// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_EXPORT_WORD_VALIDATION_H
#define INTEGRAL_HECKEZ_EXPORT_WORD_VALIDATION_H

#include <cstdint>

// slot-27 EXW: the at-root read and the (C3) read on the free block over Z of EXPORT-Z step 2 where rule (alpha) fires
// (docs/implementation/slot27/EXW_design.md v0.2 Sec. 6).  Eleven arms: (A) route identity against the EXPORT / level-1 read where both reach,
// (B) class A beyond the export's K_max ((8, 10) @2 at K0 = 33, pinned), (C) class C / k1 at K = 1 with ell > 2^32, (D) the overflow teeth of
// the reduction and the word modulus, (E) four whole records EXW on vs. the seam empty (the diff inside Sec. 5's rule; WriteCertFiles twice,
// M2), (F) route bit-identity (threads, HZ-LOC-PAR par / serial), (G) teeth (the chi_Z check, a free-block RssCeiling, a forced [Terms] of
// T_ell), (H) record-level identity by forcing (the root and the (C3) trigger against the unforged record), (I) the negative case of (alpha)
// (a refusal that stands in no field: no call of the free source, the record byte-identical); (J) (audit M-EXW1) free-block refusals per
// requested set (refused-then-servable, served-then-refused, WriteCertFiles after a refusal on the same source), (K) (audit M-EXW2) the
// RssCeiling decision at threads 1 / 4 / 16 equal (predicted at kFreeReadRssLanes lanes).
// Env arm (not in --all): HECKEZ_EXW_ALL=1 builds the 22 pending records of edition 3 EXW on (HECKEZ_EXW_CELLS="N:n:alg,..." restricts),
// one EFFECTIVE line.  Gate line `HECKEZ_EXW arms=11 failing=0 identity=<k>/<k> pairs=<m> moved=<w> negative=<j>/<j> verdict=OK`.
uint32_t HeckeZExportWordValidation();

#endif
