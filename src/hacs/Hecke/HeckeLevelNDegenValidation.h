// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_LEVEL_N_DEGEN_VALIDATION_H
#define HECKE_LEVEL_N_DEGEN_VALIDATION_H
// (hecke-extensions slot 16, WP-13) The WP-13 arms (K), (A), (W), (L9), (G) of HeckeLevelNSparseValidation
// ([C] v1.16.0-pre.3 sec:levelNdegsec: (LN7)-(LN9)); no new registered name (slot-16 plan WP-13: arms only).
// Adds its checks to `checks` and returns its failure count.  Env hooks (read-only, not emitters):
// HECKE_LEVEL_N_DEGEN_CELLS="M:N:d:n:p,..." and HECKE_LEVEL_N_R3K_CELLS="N:p:k,...", each with an EFFECTIVE line.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

uint32_t HeckeLevelNDegenArms(uint32_t& checks);

// (slot-17 WP-8) The strict hook parser and the rule-19 pre-run RSS refusal, shared by every `*_CELLS` hook of the level-N sparse
// validation. HookParseCells throws (ENGINE_CHECK) on an empty value, an empty trailing / inner cell or field, a non-digit field
// or a wrong field count, by position.
std::vector<std::vector<uint64_t>> HookParseCells(const char* env, const char* name, size_t fields);

// (slot-17 WP-17; WP-8 audit MAJOR-1, MINOR-2, MINOR-3) The calibrated hook RSS model `hook_rss/v1`. The cost model's
// sparse_bytes is NOT a peak-RSS prediction: on WP-5's ladder (docs/implementation/slot17/WP-5_report.md section 3) the measured
// peak RSS / sparse_bytes is 10.0-10.6 at d = n + 1 = 1 (30030, 60060, 90090, 100003), 11-16 on the DEGEN cells against
// sparse_bytes(M) + sparse_bytes(N), 1.9 at 60060:2 (d = 3), and > 50 at 99330:0 (= 2*3*5*7*11*43, W5-F1: past the 8 GB
// watchdog against 164 MiB). Hence, per cell, predicted RSS = kHookRssBaselineMiB + factor x sparse_bytes with
//   factor = 64 when N has >= 2 distinct prime factors and one of them is > 13 (the W5-F1 regime; no cell measured there below 8 GB),
//          = 16 at d = 1,
//          = 4 (SPARSE, calibrated at 60060:2) or 6 (DEGEN / R3K; small DEGEN cells and slot-16's 15015 -> 30030 at n = 2) at d >= 3.
// A coarse CEILING, not a model: the VmHWM watchdog stays mandatory for every hook run (rule 19).
enum class HookKind { kSparse, kDegen, kR3k };
constexpr double kHookRssBaselineMiB = 512.0;
double HookRssFactor(HookKind kind, uint64_t N, uint32_t n);
// kHookRssBaselineMiB is NOT included: add it once per cell (a DEGEN cell sums its M and N presentations, then adds it).
double HookPredictedRssBytes(HookKind kind, uint64_t N, uint32_t n);

// The cap: 8 GiB, or HECKE_LEVEL_N_HOOK_RSS_CAP_GIB=<G> (whole GiB in [1, 1024]) to run a refused cell DELIBERATELY (an
// exclusive window, e.g. the slot-18 90090:2:17). A malformed value throws. Applies to the SPARSE, DEGEN and R3K hooks alike;
// each prints `rss_cap_gib=<G> rss_cap_source=default|HECKE_LEVEL_N_HOOK_RSS_CAP_GIB rss_model=hook_rss/v1` on its EFFECTIVE line.
struct HookRssCap {
    uint64_t gib = 8;
    bool overridden = false;
};
HookRssCap HookRssCapFrom(const char* value);   // nullptr = the default
HookRssCap HookRssCapFromEnv();                 // reads HECKE_LEVEL_N_HOOK_RSS_CAP_GIB
std::string HookRssCapText(const HookRssCap& cap);
// Throws (ENGINE_CHECK, rule 19, before the run) when `predicted_bytes` (baseline included) exceeds the cap; otherwise prints one
// `HOOK_RSS <name> cell=<idx> <what> predicted_mib=<x> cap_gib=<G>` line.
void HookRefuseOverCap(const char* name, size_t idx, const std::string& what, double predicted_bytes, const HookRssCap& cap);

#endif // HECKE_LEVEL_N_DEGEN_VALIDATION_H
