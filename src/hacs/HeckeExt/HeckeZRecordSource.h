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

#include <atomic>
#include <memory>

namespace heckez {
namespace record {

// local_read_free = FreeLocalRead() (slot-27 EXW); the other sources as before.
RecordSources DefaultRecordSources();

// ---- (slot-27 EXW, docs/implementation/slot27/EXW_design.md Sec. 3.1) the free-block source ------------------------------------------
// A per-source lazy cache (shared by the copies of the function object): on the first call for (N, n, allow_bad_primes) it builds the
// carrier freez::IsoTreeCarrierZ(N, n) and ONE modexport::RunFreeBlock on the read set (P, plus T_ell when ReadPrimes appends it);
// a later call needing an operator the cached block lacks re-runs RunFreeBlock on the union (F is the same integers in the same B:
// the basis does not depend on the operators).  One cell is cached at a time.  Refusals of the route (RssCeiling, Terms, NoProof,
// Inadmissible, NoDevice, and the free basis' own resource refusals) return M.status = "blocked(free_block)" with the name in
// M.detail ("REFUSED [<Name>] .."); Uncertified, SingularG and BoundViolated on a carrier that EXPORT-Z step 2 certified are defects
// (ENGINE_CHECK).  A [Terms] refusal of an appended T_ell is retried once without the appended T_ell's: the pairs that need that T_ell
// are then refused by name, the others proceed (design m2).  The lanes are the CPU lanes unless ENGINE_DEVICE=gpu (bit-identical).
// (EXW fix, audit M-EXW1) Refusals are PER REQUESTED SET: a request whose read set the cached block covers is served, whatever was
// refused before on the cell; otherwise the source tries the union (cached block + read set), and when that is refused the read set
// alone (the cached block is then replaced by it: F is the same integers, so a re-read gives the same module).  A refusal is cached
// under the operator set it was refused for, never for the cell.  So a request is served iff RunFreeBlock serves its read set (for the
// refusals that are monotone in the operator set, RssCeiling among them), independently of the order of the requests.
// (EXW fix, audit M-EXW2) [RssCeiling] is predicted at rss_lanes (default kFreeReadRssLanes) lanes, not at the run's: whether a
// request is served is a pure function of (N, n, the read set, allow_bad_primes, rss_cap_bytes), never of threads / ENGINE_THREADS /
// ENGINE_DEVICE.  The run itself uses free_threads (or `threads`) lanes.
// (audit m-EXW3) Rule 19: the carrier and the block stay resident for the life of the source (in a campaign: until the next cell, and
// through WriteCertFiles), so an EXW cell's peak RSS is the EXPORT read's PLUS the free block's; census kit predictions must add both.
// The mutex is held across RunFreeBlock (correct; it serialises concurrent callers, campaign jobs = 1).
constexpr uint32_t kFreeReadRssLanes = 16;   // = PredictFreeBlockRss' own default for threads = 0
struct FreeSourceStats {
    uint64_t calls = 0;                      // local_read_free calls
    uint64_t runs = 0;                       // RunFreeBlock runs (incl. a [Terms] retry; a cached refusal is not re-run)
    std::vector<uint32_t> ops;               // the cached block's operators
    std::vector<uint64_t> F_fnv;             // its MatrixFnv per operator
    std::string refusal;                     // the last refusal returned ("" when none)
    uint64_t rss_pred = 0;                   // the last RunFreeBlock's predicted lane bytes (FreeBlockResult::rss_pred; 0 if not reached)
    double secs = 0;                         // RunFreeBlock wall, summed
};
struct FreeSourceOptions {
    uint64_t rss_cap_bytes = 3ull << 30;     // ModuleZ2Options' default [RssCeiling]
    uint32_t rss_lanes = kFreeReadRssLanes;  // the lanes [RssCeiling] is predicted at (M-EXW2; never 0 on this route)
    uint32_t threads = 0;                    // > 0 overrides ReadOptions::free_threads (VALIDATION: route bit-identity)
    std::vector<uint32_t> force_terms;       // VALIDATION ONLY: these operators are refused [Terms] (as IsoHeckeN would refuse them)
    std::shared_ptr<FreeSourceStats> stats;  // optional: filled on every call (VALIDATION / provenance)
};
std::function<LocalReadOut(uint32_t, uint32_t, uint64_t, uint32_t, const std::vector<uint32_t>&, uint32_t, const local::ReadOptions&,
                           uint32_t)>
FreeLocalRead(const FreeSourceOptions& fo = {});

}  // namespace record
}  // namespace heckez

#endif
