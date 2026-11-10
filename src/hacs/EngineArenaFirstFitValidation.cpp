// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineArenaFirstFitValidation.cpp -- hecke-extensions slot 9, WP-7 (lever L3).
//
// Lever L3 replaced the tape scheduler's arena placement -- a linear walk over
// the offset-sorted live set per temp, O(T * L) -- by a treap with subtree
// max-gap augmentation and an expiry heap, O(log L) per temp
// (src/Engine/Passes.cpp, ArenaFirstFit).  The claim is that the
// placement RULE is unchanged, so every arena_offset, the arena size and hence
// the compiled program are exactly what they were.  This validation checks
// that claim in every build (the NDEBUG builds included, where schedule()'s
// own ENGINE_DEBUG_CHECK against the linear scan is compiled out):
//
//   (A) on randomized programs -- mixed temp shapes (so footprints differ and
//       gaps are reused partially), chains that stretch lifetimes across many
//       wavefronts, late reads that keep a temp alive -- compiled with the
//       arena on, it re-derives the offset of EVERY temp from the compiled
//       buffers' (first_level, last_level, footprint) with its own linear
//       scan of the rule (the pre-WP-7 algorithm, re-implemented here, not
//       shared with Passes.cpp) and requires equality, and equality of
//       arena_elts;
//   (B) that the stress actually exercised the interesting branches: temps
//       placed in an interior gap (below the current live end) and temps
//       placed past the last live temp both occur;
//   (C) the arena run's outputs equal the owned-storage run's (arena = false)
//       entrywise, under a poisoned arena too.
// Deterministic (fixed seeds); ~1 s.  Writes nothing.
#include "EngineArenaFirstFitValidation.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "../Engine/EngineTestUtil.h"
#include "../Engine/Executor.h"
#include "../Engine/Passes.h"
#include "../Engine/Recorder.h"

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;

struct RefStats { uint64_t temps = 0, interior = 0, appended = 0; };

// The placement rule, re-derived from the compiled buffer descriptors.
template<class Elt>
bool reference_offsets_match(const Program<Elt>& prog, RefStats& st, std::string& why) {
    std::vector<BufId> temps;
    for (BufId b = 0; b < prog.bufs.size(); ++b)
        if (prog.bufs[b].kind == BufKind::Temp && prog.bufs[b].footprint > 0) temps.push_back(b);
    std::stable_sort(temps.begin(), temps.end(), [&](BufId x, BufId y) {
        const BufferDesc& a = prog.bufs[x]; const BufferDesc& b = prog.bufs[y];
        if (a.first_level != b.first_level) return a.first_level < b.first_level;
        if (a.footprint != b.footprint) return a.footprint > b.footprint;
        return x < y;
    });
    struct Live { size_t off, end; uint32_t last; };
    std::vector<Live> live;   // sorted by offset; `end` is the 16-aligned end
    size_t arena = 0;
    for (BufId b : temps) {
        const BufferDesc& d = prog.bufs[b];
        std::vector<Live> kept;
        for (const Live& l : live) if (l.last >= d.first_level) kept.push_back(l);
        live.swap(kept);
        size_t off = 0, pos = live.size();
        for (size_t i = 0; i < live.size(); ++i) {
            if (live[i].off >= off + d.footprint) { pos = i; break; }
            off = live[i].end;
        }
        const bool interior = pos < live.size();
        const size_t end = (off + d.footprint + 15) / 16 * 16;
        live.insert(live.begin() + static_cast<long>(pos), Live{off, end, d.last_level});
        ++st.temps;
        if (interior) ++st.interior; else ++st.appended;
        if (d.arena_offset != off) {
            why = "temp " + std::to_string(b) + ": arena_offset " + std::to_string(d.arena_offset) +
                  " != reference " + std::to_string(off);
            return false;
        }
        arena = std::max(arena, end);
    }
    if (prog.arena_elts != arena) {
        why = "arena_elts " + std::to_string(prog.arena_elts) + " != reference " + std::to_string(arena);
        return false;
    }
    return true;
}

struct ShapeSpec { uint32_t rows, cols; };

// One randomized program: `n_temps` temps over the shapes, chained and read late.
template<class Elt>
struct RandomProgram {
    std::vector<DenseBuf<Elt>> X;
    std::vector<ShapeSpec> shapes;
    Recorder<Elt> rec;
    RandomProgram(const ModCtx<Elt>& ctx, Rng& rng, uint32_t n_temps, const std::vector<ShapeSpec>& sh,
                  std::vector<DenseBuf<Elt>>& Y)
        : shapes(sh), rec(ctx) {
        const Elt p = ctx.p;
        std::vector<BufId> in, out;
        for (const ShapeSpec& s : shapes) X.push_back(random_buf<Elt>(rng, s.rows, s.cols, p));
        for (size_t s = 0; s < shapes.size(); ++s) in.push_back(rec.input(cview(X[s])));
        for (size_t s = 0; s < shapes.size(); ++s) out.push_back(rec.output(view(Y[s])));
        std::vector<std::vector<BufId>> by_shape(shapes.size());
        for (uint32_t i = 0; i < n_temps; ++i) {
            const uint32_t s = rng.below(static_cast<uint32_t>(shapes.size()));
            const BufId t = rec.temp(shapes[s].rows, shapes[s].cols);
            std::vector<BufId>& prev = by_shape[s];
            if (!prev.empty() && rng.below(4) != 0) {
                // a chain step: t = earlier + alpha * X  (lifetimes stretch over levels)
                const BufId src = prev[prev.size() - 1 - rng.below(static_cast<uint32_t>(std::min<size_t>(prev.size(), 8)))];
                rec.scatter(rec.whole(src), rec.whole(t));
                rec.axpy(rng.nonzero<Elt>(p), rec.whole(in[s]), rec.whole(t));
            } else {
                rec.scatter(rec.whole(in[s]), rec.whole(t));
                rec.scal(rng.nonzero<Elt>(p), rec.whole(t));
            }
            // a late read of an OLD temp: keeps it alive across many levels
            if (!prev.empty() && rng.below(3) == 0) {
                const BufId old = prev[rng.below(static_cast<uint32_t>(prev.size()))];
                rec.axpy(rng.nonzero<Elt>(p), rec.whole(old), rec.whole(t));
            }
            if (rng.below(5) == 0) rec.axpy(Elt(1), rec.whole(t), rec.whole(out[s]));
            prev.push_back(t);
        }
        // every temp of the last few steps reaches its output (nothing is dead weight)
        for (size_t s = 0; s < shapes.size(); ++s)
            for (size_t k = 0; k < by_shape[s].size() && k < 3; ++k)
                rec.axpy(Elt(1), rec.whole(by_shape[s][by_shape[s].size() - 1 - k]), rec.whole(out[s]));
    }
};

template<class Elt>
void run_case(Checker& g, Elt p, uint64_t seed, uint32_t n_temps, const std::vector<ShapeSpec>& shapes,
              bool cse, RefStats& total, const std::string& tag) {
    const ModCtx<Elt> ctx(p);
    PassOptions o; o.cse = cse; o.fuse = cse; o.arena = true;
    std::vector<std::vector<Elt>> results;
    for (int mode = 0; mode < 3; ++mode) {   // 0: arena, 1: owned storage, 2: poisoned arena
        Rng rng(seed);
        std::vector<DenseBuf<Elt>> Y;
        for (const ShapeSpec& s : shapes) Y.emplace_back(s.rows, s.cols);
        RandomProgram<Elt> rp(ctx, rng, n_temps, shapes, Y);
        Program<Elt> prog = rp.rec.finish();
        PassOptions om = o; om.arena = (mode != 1);
        compile(prog, om);
        if (mode == 0) {
            RefStats st;
            std::string why;
            const bool ok = reference_offsets_match(prog, st, why);
            g.check(ok, tag + " every arena_offset and arena_elts equal the linear-scan reference" + (ok ? "" : ": " + why));
            total.temps += st.temps; total.interior += st.interior; total.appended += st.appended;
            std::cout << "  " << tag << ": " << st.temps << " temps placed (" << st.interior << " in an interior gap, "
                      << st.appended << " appended), " << prog.levels.size() << " levels, arena " << prog.arena_elts
                      << " of " << prog.stats.temp_elts << " elements" << std::endl;
        }
        Executor<Elt> ex;
        ex.set_poison(mode == 2);
        ex.run(prog, rp.rec.bindings());
        std::vector<Elt> flat;
        for (const DenseBuf<Elt>& y : Y)
            for (uint32_t i = 0; i < y.rows; ++i)
                for (uint32_t j = 0; j < y.cols; ++j) flat.push_back(y.at(i, j));
        results.push_back(std::move(flat));
    }
    g.check(results[0] == results[1], tag + " arena run == owned-storage run (entrywise)");
    g.check(results[2] == results[1], tag + " poisoned-arena run == owned-storage run (entrywise)");
}

} // namespace

uint32_t EngineArenaFirstFitValidation() {
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::cout << "=== EngineArenaFirstFitValidation (slot-9 WP-7, L3: the O(log L) arena first-fit == the linear-scan rule) ==="
              << std::endl;
    try {
        const std::vector<ShapeSpec> mixed = {{8, 16}, {16, 16}, {33, 20}, {5, 48}, {64, 40}, {1, 16}, {17, 3}};
        const std::vector<ShapeSpec> two = {{16, 16}, {40, 32}};
        RefStats total;
        run_case<uint32_t>(g, 134217757u, 0xF1F1, 400, mixed, true, total, "[u32 mixed 400 cse]");
        run_case<uint32_t>(g, 134217757u, 0xF1F2, 3000, mixed, false, total, "[u32 mixed 3000]");
        run_case<uint32_t>(g, 4294967291u, 0xF1F3, 3000, two, true, total, "[u32 two-shape 3000 cse]");
        run_case<uint32_t>(g, 691u, 0xF1F4, 12000, mixed, false, total, "[u32 mixed 12000]");
        run_case<uint64_t>(g, 2305843009213693951ull, 0xF1F5, 2000, mixed, true, total, "[u64 mixed 2000 cse]");
        g.check(total.interior > 0, "(B) some temp was placed in an interior gap");
        g.check(total.appended > 0, "(B) some temp was placed past the last live temp");
    } catch (const std::exception& e) {
        std::cout << "  EXCEPTION: " << e.what() << std::endl;
        ++g.fails;
    }
    std::cout << "EngineArenaFirstFitValidation: " << g.checks << " checks, " << g.fails << " failed, "
              << seconds_since(t0) << " s" << std::endl;
    return g.fails == 0 ? 0 : 1;
}
