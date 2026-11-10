// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/legacy/HeckeLevelNOrderLegacyArms.cpp -- (W18o-as_built), the slot-18 bit-identity twin of the RETIRED generator
// order (S24 CL-G2; owner ruling 2 of 2026-10-30).  Compiled only with OC_ENABLE_LEGACY=ON (cmake/legacy.cmake); called
// from HeckeLevelNOrderArms (HeckeLevelNOrderValidation.cpp) under oc::legacy::RunLegacyArm("HeckeLevelNOrderValidation",
// "as_built").  Extracted verbatim from HeckeLevelNOrderValidation.cpp (slot 18 WP-5, d4265bae): ArmOrder / ArmOrderHecke
// and SamePresentation; only the Checker became the `check` callback.
// [C] sec:levelN, rmk:levelNscale (4) (the generator order, a value-free choice).
#include "../HeckeLevelNOrderValidation.h"

#if !OC_ENABLE_LEGACY
#error "HeckeLevelNOrderLegacyArms.cpp is an OC_ENABLE_LEGACY=ON source (cmake/legacy.cmake)"
#endif

#include <chrono>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../HeckeLevelEll.h"
#include "../HeckeLevelN.h"
#include "../HeckeLevelNSparse.h"
#include "../../../ModArithmetic/ModCtx.h"
#include "../../Util/Timing.h"

namespace {

using oc_util::Seconds;

using hecke::Mat22;
using namespace hecke::level;

using Check = std::function<void(bool, const std::string&)>;

constexpr uint32_t kP = 1000003;

std::string Cell(uint64_t N, uint32_t n, uint32_t p) {
    return "(N=" + std::to_string(N) + ", n=" + std::to_string(n) + ", F_" + std::to_string(p) + ")";
}

// Every value of the two presentations, entry for entry (not only the digest).
bool SamePresentation(const SparseCokernel& a, const SparseCokernel& b) {
    if (a.D() != b.D() || a.p() != b.p() || a.pivots() != b.pivots() || a.free_columns() != b.free_columns()) return false;
    for (uint32_t c : a.pivots()) {
        const SparseVec &x = a.tail_of(c), &y = b.tail_of(c);
        if (x.col != y.col || x.val != y.val) return false;
    }
    const SparseCokernelStats &s = a.stats(), &t = b.stats();
    return s.generators == t.generators && s.generator_nnz == t.generator_nnz && s.dependent == t.dependent &&
           s.final_nnz == t.final_nnz && s.dim_S == t.dim_S && s.dim_ST == t.dim_ST && a.h0() == b.h0();
}

// (W18o) one cell: the same generators, both orders.
void ArmOrder(const Check& check, uint64_t N, uint32_t n, uint32_t p, bool gate_occ, std::vector<std::string>& lines) {
    const engine::ModCtx<uint32_t> ctx(p);
    const Gamma0LevelN lvl(N);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<SparseVec> gS = OrbitWiseFixedGenerators(lvl, Mat22::S(), 2, n, ctx);
    const std::vector<SparseVec> gU = OrbitWiseFixedGenerators(lvl, Mat22::ST(), 3, n, ctx);
    const uint32_t D = lvl.m() * (n + 1);
    auto build = [&](GeneratorOrder o) {
        ScopedGeneratorOrder s(o);
        return SparseCokernel(D, gS, gU, p);
    };
    const SparseCokernel A = build(GeneratorOrder::kAsBuilt);
    const SparseCokernel B = build(GeneratorOrder::kLastColumnDescending);
    const std::string tag = "(W18o) " + Cell(N, n, p);
    check(A.stats().order == GeneratorOrder::kAsBuilt && B.stats().order == GeneratorOrder::kLastColumnDescending,
            tag + ": each build records the order it used");
    const bool same = SamePresentation(A, B);
    check(same, tag + ": the RREF of Rel_N is the same under both generator orders (pivots, free columns, every row entry, ranks, h0)");
    check(PresentationDigest(A) == PresentationDigest(B), tag + ": PresentationDigest agrees");
    if (gate_occ)
        check(B.stats().occ_entries <= A.stats().occ_entries,
                tag + ": the sorted order costs no more occurrence-list entries than the as-built order (the W5-F1 shape)");
    std::ostringstream os;
    os << "LEVEL_N_ORDER (W18o) N=" << N << " n=" << n << " p=" << p << " D=" << D << " h1=" << A.h1()
       << " final_nnz=" << A.stats().final_nnz << " identical=" << (same ? 1 : 0) << " presentation_fnv=0x" << std::hex
       << PresentationDigest(B) << std::dec << " merge_ops_as_built=" << A.stats().merge_ops << " merge_ops_sorted=" << B.stats().merge_ops
       << " occ_as_built=" << A.stats().occ_entries << " occ_sorted=" << B.stats().occ_entries << " peak_nnz_as_built="
       << A.stats().peak_nnz << " peak_nnz_sorted=" << B.stats().peak_nnz << " wall_s=" << Seconds(t0);
    lines.push_back(os.str());
}

// (W18o) the Hecke matrices on the sparse route under both orders.
void ArmOrderHecke(const Check& check, uint64_t N, uint32_t n, uint32_t p, const std::vector<uint32_t>& primes) {
    const engine::ModCtx<uint32_t> ctx(p);
    const Gamma0LevelN lvl(N);
    std::map<uint32_t, HMat> a, b;
    {
        ScopedGeneratorOrder s(GeneratorOrder::kAsBuilt);
        a = CokernelHeckeMatrices(lvl, n, ctx, primes, CokernelRoute::kSparse);
    }
    {
        ScopedGeneratorOrder s(GeneratorOrder::kLastColumnDescending);
        b = CokernelHeckeMatrices(lvl, n, ctx, primes, CokernelRoute::kSparse);
    }
    bool same = a.size() == b.size() && a.size() == primes.size();
    for (const auto& kv : a) same = same && b.count(kv.first) && HEqual(kv.second, b.at(kv.first));
    check(same, "(W18o) Hecke " + Cell(N, n, p) + ": [H_{N,q}] on H^1 bit-identical under both generator orders");
}

} // namespace

void HeckeLevelNOrderAsBuiltArms(const std::vector<LevelNOrderCell>& cells, const Check& check, std::vector<std::string>& lines) {
    for (const LevelNOrderCell& c : cells) ArmOrder(check, c.N, c.n, c.p, c.gate_occ, lines);
    ArmOrderHecke(check, 210, 4, kP, {11u, 13u});
    ArmOrderHecke(check, 1000, 0, kP, {3u, 7u});
    ArmOrderHecke(check, 997, 2, kP, {2u, 5u});
    ArmOrderHecke(check, 1290, 0, 7, {11u, 13u});
}
