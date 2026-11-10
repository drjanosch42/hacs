// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/legacy/HeckeLevelNGenOrderLegacy.inc -- the RETIRED generator order of the sparse cokernel (S24 CL-G2; owner
// ruling 2 of 2026-10-30; docs/implementation/slot24/CLEANUP_plan_1030.md Sec. 3.2, CODEAUDIT-B B7).
//
// Included by Hecke/HeckeLevelNSparse.cpp inside namespace hecke::level, under `#if OC_ENABLE_LEGACY` only.  Extracted
// verbatim from HeckeLevelNSparse.cpp (slot 18 WP-5 (b), d4265bae; the PAR-LN thread-context hook of slot 24): the
// thread-local ScopedGeneratorOrder, the read-only rollback arm HECKE_LEVEL_N_GEN_ORDER=as_built (slots 15-17's order:
// S then ST, as built) and CurrentGeneratorOrder / GeneratorOrderName.  Value-free: the stored RREF is unique, so the
// order changes only the work counters (merge_ops, peak_nnz, occ_entries).  HeckeLevelNOrderValidation's bit-identity
// arm (W18o, as_built vs sorted; legacy/HeckeLevelNOrderLegacyArms.cpp) is its user.

namespace {
thread_local int t_gen_order = -1;   // -1: no ScopedGeneratorOrder active
// (slot-24 PAR-LN) a scoped thread-local option: workers of run_concurrently() / par::ForEachIndex receive the launching thread's
const bool g_gen_order_registered = [] {
    engine::tape::register_thread_context([]() -> engine::tape::ThreadContextInstaller {
        const int o = t_gen_order;
        return [o]() -> engine::tape::ThreadContextGuard {
            const int prev = t_gen_order;
            t_gen_order = o;
            return engine::tape::ThreadContextGuard(nullptr, [prev](void*) { t_gen_order = prev; });
        };
    });
    return true;
}();

GeneratorOrder EnvGeneratorOrder() {
    // read once per process; a malformed value throws at every use (loud, never silently ignored)
    static const std::string v = [] {
        const char* e = std::getenv("HECKE_LEVEL_N_GEN_ORDER");
        return e == nullptr ? std::string("sorted") : std::string(e);
    }();
    if (v == "sorted") return GeneratorOrder::kLastColumnDescending;
    if (v == "as_built") return GeneratorOrder::kAsBuilt;
    ENGINE_CHECK_MSG(false, ("HECKE_LEVEL_N_GEN_ORDER='" + v + "': expected 'sorted' (the default) or 'as_built' (the rollback arm)").c_str());
    return GeneratorOrder::kLastColumnDescending;
}
} // namespace

GeneratorOrder CurrentGeneratorOrder() {
    return t_gen_order >= 0 ? static_cast<GeneratorOrder>(t_gen_order) : EnvGeneratorOrder();
}
const char* GeneratorOrderName(GeneratorOrder o) { return o == GeneratorOrder::kAsBuilt ? "as_built" : "sorted"; }
ScopedGeneratorOrder::ScopedGeneratorOrder(GeneratorOrder o) : prev_(t_gen_order) { t_gen_order = static_cast<int>(o); }
ScopedGeneratorOrder::~ScopedGeneratorOrder() { t_gen_order = prev_; }

const char* GeneratorOrderInForceName() { return GeneratorOrderName(CurrentGeneratorOrder()); }
