// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#include <cstdint>
// (hecke-extensions slot 23, PERF-P2) The host eliminations of the cone and of the involution iota^* run with
// engine::elim's inner parallel hook bound to the consumer pool (engine::tape::ScopedHostElimHook, EngineRuntime.h).
// This validation proves that no value depends on it, IN PROCESS, on consumer pools of 1, 4 and 16 workers
// (ScopedConsumerPool; the global pool is never touched):
//   D  the direct arms: engine::elim::rref / column_rank_profile / rref_with_transform of rank-deficient matrices
//      (p = 13, the VNNI route; p = 134217757) under the hook against the hook-free route, entrywise;
//   C  the mapping cone (compute() + its certificate) at (n, ell) cells: golden_digest() plus the Z / B blocks;
//   I  the production involution iota^* (compute() + its certificate): every +- basis of C / Z / B / H;
// and that the hook actually ran (pool regions > 0 at 4 and 16 workers, the cone's and iota's counted SEPARATELY;
// = 0 at 1 worker and everywhere under the rollback ENGINE_HOST_ELIM_HOOK=0).  D includes a tall 1600 x 300 case
// (the tiled pivot search); its region count must not depend on the pool size.  The audit repairs (L3) add, on a
// 4-worker pool: B run_concurrently inheritance (0/6 unhooked, 6/6 hooked, caller restored); C exceptions through a
// hook region and a hooked task (caught, guard intact, pool sound), a nested guard (no-op), a foreign hook not
// replaced (precedence, L4); D four concurrent cells against the sequential run.  It prints one HOST_ELIM_DET line whose digests a
// caller compares across separate ENGINE_THREADS=1 / 4 / 16 processes.  Returns the number of failing checks.
uint32_t HostElimThreadDeterminismValidation();
