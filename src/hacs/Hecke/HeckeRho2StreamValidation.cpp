// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeRho2StreamValidation.cpp -- see the header.
#include "HeckeRho2StreamValidation.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"

namespace {

namespace Opt = hecke::HeckeOperatorOptions;

struct Checker {
    uint64_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) { ++fails; std::cout << "  FAIL: " << what << std::endl; }
    }
};

bool same_flat(const FlatMatrix& a, const FlatMatrix& b) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (uint32_t i = 0; i < a.get_rows(); ++i)
        for (uint32_t j = 0; j < a.get_cols(); ++j)
            if (a.get_entry(i, j) != b.get_entry(i, j)) return false;
    return true;
}
bool same_matrix(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j] != b.data[i][j]) return false;
    return true;
}

// Every point owns its own complex (EngineTapeConsumerValidation.cpp's note).
struct Point {
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    uint32_t n = 0, ell = 0;
};

std::unique_ptr<Point> make_point(uint32_t n, uint32_t ell) {
    auto pt = std::make_unique<Point>();
    pt->K = std::make_unique<SimplicialComplex>(CreateSL2ZFundamentalDomainTriangulation());
    pt->model = std::make_unique<hecke::SL2ZTessellationTables>(*pt->K);
    pt->n = n; pt->ell = ell;
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(*pt->K, n, ell)) rels.push_back(datum.rel);
    pt->lssc = std::make_unique<LocalSystemSimplicialComplex>(*pt->K, rels, n + 1, ell);
    pt->coh = std::make_unique<EquivariantCohomologyOnK>(pt->lssc.get());
    pt->coh->compute_cohomology();
    return pt;
}

struct Run {
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;
    std::unique_ptr<hecke::HeckeOperatorOnK> op;
};

std::unique_ptr<Run> make_run(const Point& pt, uint64_t p) {
    auto r = std::make_unique<Run>();
    r->cosets = std::make_unique<hecke::HeckeCosetTables>(p, 1, pt.ell, pt.n);
    r->tau = std::make_unique<hecke::HeckeTauTable>(*pt.K, *pt.model, *r->cosets, pt.ell, pt.n);
    r->op = std::make_unique<hecke::HeckeOperatorOnK>(pt.coh.get(), r->cosets.get(), r->tau.get());
    return r;
}

// One operator, route 2 streamed or materialised, its route-2 blocks captured.
std::unique_ptr<Run> run_captured(const Point& pt, uint64_t p, bool stream, Opt::Route2Capture& cap) {
    const Opt::ScopedRho2Streaming s(stream);
    const Opt::ScopedRoute2Capture c(&cap);
    std::unique_ptr<Run> r = make_run(pt, p);
    r->op->compute();
    return r;
}

// The first entrywise difference of two captures, "" if none.
std::string capture_diff(const Opt::Route2Capture& a, const Opt::Route2Capture& b) {
    if (a.blocks.size() != b.blocks.size())
        return "block counts " + std::to_string(a.blocks.size()) + " vs " + std::to_string(b.blocks.size());
    auto ib = b.blocks.begin();
    for (auto ia = a.blocks.begin(); ia != a.blocks.end(); ++ia, ++ib) {
        if (ia->first != ib->first) return "block key sets differ";
        if (ia->second != ib->second) {
            size_t e = 0;
            while (e < ia->second.size() && e < ib->second.size() && ia->second[e] == ib->second[e]) ++e;
            return "block (q=" + std::to_string(std::get<0>(ia->first)) + ", sigma=" + std::to_string(std::get<1>(ia->first)) +
                   ", sigma_k=" + std::to_string(std::get<2>(ia->first)) + ") entry " + std::to_string(e);
        }
    }
    return "";
}

// Every captured block equals the fused block of Theta_block(q), entrywise.
bool capture_matches_fused(const Opt::Route2Capture& cap, const hecke::HeckeOperatorOnK& op, uint32_t d) {
    for (const auto& kv : cap.blocks) {
        const uint32_t q = std::get<0>(kv.first), sigma = std::get<1>(kv.first), sk = std::get<2>(kv.first);
        const auto& T = op.Theta_block(q);
        if (kv.second.size() != static_cast<size_t>(d) * d) return false;
        for (uint32_t r = 0; r < d; ++r)
            for (uint32_t c = 0; c < d; ++c)
                if (static_cast<uint64_t>(T.get_entry(sigma * d + r, sk * d + c)) != kv.second[static_cast<size_t>(r) * d + c])
                    return false;
    }
    return true;
}

// compute() under a target mutation: "" if it threw O12(a) naming exactly (R, C), else what happened.
std::string mutation_outcome(const Point& pt, uint64_t p, bool stream, const Opt::Route2TargetMutation& m, uint32_t d) {
    const Opt::ScopedRho2Streaming s(stream);
    const Opt::ScopedRoute2TargetMutation mut(m);
    std::unique_ptr<Run> r = make_run(pt, p);
    const std::string want = "entry (" + std::to_string(m.sigma * d + m.r) + "," + std::to_string(m.sigma_k * d + m.c) + ")";
    try {
        r->op->compute();
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        if (msg.find("O12(a) FAILED") == std::string::npos) return "threw something else: " + msg;
        if (msg.find(want) == std::string::npos) return "O12(a) named another entry: " + msg + " (want " + want + ")";
        return "";
    }
    return "compute() did not throw";
}

} // namespace

uint32_t HeckeRho2StreamValidation() {
    Checker g;
    uint32_t arms = 0, mutations = 0, reduce_arms = 0;
    uint64_t blocks_total = 0;
    struct Cell { uint32_t n, ell; };
    for (const Cell c : {Cell{50, 13}, Cell{42, 7}, Cell{64, 101}}) {
        const std::unique_ptr<Point> pt = make_point(c.n, c.ell);
        const uint32_t d = c.n + 1;
        for (const uint64_t p : {uint64_t{2}, uint64_t{3}, uint64_t{31}, uint64_t{c.ell}}) {
            const std::string t = "(n=" + std::to_string(c.n) + ", ell=" + std::to_string(c.ell) + ", p=" + std::to_string(p) + ")";
            const bool lever = (p == c.ell);   // U_ell: the factored route 2, never streamed
            ++arms;
            Opt::Route2Capture on_cap, off_cap;
            const Opt::Rho2StreamStats s0 = Opt::rho2_stream_stats();
            const std::unique_ptr<Run> on = run_captured(*pt, p, true, on_cap);
            const Opt::Rho2StreamStats s1 = Opt::rho2_stream_stats();
            const std::unique_ptr<Run> off = run_captured(*pt, p, false, off_cap);
            const Opt::Rho2StreamStats s2 = Opt::rho2_stream_stats();
            const hecke::HeckeOperatorOnK& a = *on->op;
            const hecke::HeckeOperatorOnK& b = *off->op;
            // (V) non-vacuity / routing
            if (lever) {
                g.check(s1.streamed_degrees == s0.streamed_degrees && s2.materialised_degrees == s1.materialised_degrees,
                        t + " (V) the U_ell lever's route 2 is neither streamed nor materialised");
                g.check(on_cap.blocks.empty() && off_cap.blocks.empty(), t + " (V) no dense route-2 capture under the lever");
            } else {
                g.check(s1.streamed_degrees - s0.streamed_degrees == 3 && s1.materialised_degrees == s0.materialised_degrees,
                        t + " (V) streamed: 3 degrees streamed, none materialised");
                g.check(s2.materialised_degrees - s1.materialised_degrees == 3 && s2.streamed_degrees == s1.streamed_degrees,
                        t + " (V) rollback: 3 degrees materialised, none streamed");
                g.check(s1.keys > s0.keys && s1.evals - s0.evals == s1.keys - s0.keys,
                        t + " (V) every distinct key evaluated exactly once");
                g.check(s1.blocks - s0.blocks == on_cap.blocks.size() && !on_cap.blocks.empty(),
                        t + " (V) streamed block counter == captured blocks, > 0");
            }
            // (X) the cross-check, entrywise
            const std::string diff = capture_diff(on_cap, off_cap);
            g.check(diff.empty(), t + " (X) streamed == materialised route-2 blocks: " + diff);
            g.check(capture_matches_fused(on_cap, a, d), t + " (X) streamed route-2 blocks == Theta_block (streamed op)");
            g.check(capture_matches_fused(off_cap, b, d), t + " (X) materialised route-2 blocks == Theta_block (rollback op)");
            blocks_total += on_cap.blocks.size();
            const auto& Ta = a.get_Theta_flat_matrices();
            const auto& Tb = b.get_Theta_flat_matrices();
            const auto& Aa = a.get_H_matrices();
            const auto& Ab = b.get_H_matrices();
            g.check(Ta.size() == 3 && Tb.size() == 3 && Aa.size() == 3 && Ab.size() == 3, t + " degree counts");
            if (Ta.size() == 3 && Tb.size() == 3 && Aa.size() == 3 && Ab.size() == 3)
                for (uint32_t q = 0; q < 3; ++q) {
                    g.check(same_flat(Ta[q], Tb[q]), t + " q=" + std::to_string(q) + " (X) Theta_flat on == off");
                    g.check(same_matrix(Aa[q], Ab[q]), t + " q=" + std::to_string(q) + " (X) A^{(q)} on == off");
                }
            g.check(a.certificates().routes_bit_identical && b.certificates().routes_bit_identical, t + " (X) O12(a) on and off");
            // (N) negative control on the comparator
            if (!on_cap.blocks.empty()) {
                Opt::Route2Capture flipped;
                flipped.blocks = on_cap.blocks;
                auto& v = flipped.blocks.rbegin()->second;
                v[v.size() / 2] = (v[v.size() / 2] + 1) % c.ell;
                g.check(!capture_diff(flipped, off_cap).empty(), t + " (N) the comparator flags a flipped entry");
                g.check(!capture_matches_fused(flipped, a, d), t + " (N) the fused comparison flags a flipped entry");
            }
            // (R) (audit F3) the delayed accumulation reduced every r products, so its
            // in-loop reduction branch runs: the blocks equal the rollback's.
            if (p == 31) {
                for (const uint64_t every : {uint64_t{1}, uint64_t{2}}) {
                    const Opt::ScopedRho2StreamReduceEvery re(every);
                    Opt::Route2Capture re_cap;
                    const uint64_t m0 = Opt::rho2_stream_stats().mid_reductions;
                    const std::unique_ptr<Run> rr = run_captured(*pt, p, true, re_cap);
                    const uint64_t m1 = Opt::rho2_stream_stats().mid_reductions;
                    const std::string rdiff = capture_diff(re_cap, off_cap);
                    g.check(m1 > m0, t + " (R) reduce every " + std::to_string(every) + ": the in-loop reduction ran");
                    g.check(rdiff.empty(), t + " (R) reduce every " + std::to_string(every) + " == materialised: " + rdiff);
                    ++reduce_arms;
                }
            }
            // (M) mutations: one entry of the fused target, first and last captured block
            // (p = 31, audit F4: the first block of every degree, multi-batch degrees included)
            if (!lever && p == 31) {
                uint32_t last_q = UINT32_MAX;
                for (const auto& kv : on_cap.blocks) {
                    if (std::get<0>(kv.first) == last_q) continue;
                    last_q = std::get<0>(kv.first);
                    Opt::Route2TargetMutation m;
                    m.on = true;
                    m.q = last_q; m.sigma = std::get<1>(kv.first); m.sigma_k = std::get<2>(kv.first);
                    m.r = d - 1; m.c = d - 1;
                    for (const bool stream : {true, false}) {
                        ++mutations;
                        const std::string out = mutation_outcome(*pt, p, stream, m, d);
                        g.check(out.empty(), t + " (M) " + (stream ? "streamed" : "materialised") + " mutation q=" +
                                                 std::to_string(m.q) + ": " + out);
                    }
                }
            }
            if (!lever && p != 31) {
                for (const bool first : {true, false}) {
                    const auto& key = first ? on_cap.blocks.begin()->first : on_cap.blocks.rbegin()->first;
                    Opt::Route2TargetMutation m;
                    m.on = true;
                    m.q = std::get<0>(key); m.sigma = std::get<1>(key); m.sigma_k = std::get<2>(key);
                    m.r = first ? 0 : d - 1; m.c = first ? d / 2 : 1;
                    for (const bool stream : {true, false}) {
                        ++mutations;
                        const std::string out = mutation_outcome(*pt, p, stream, m, d);
                        g.check(out.empty(), t + " (M) " + (stream ? "streamed" : "materialised") + " mutation q=" +
                                                 std::to_string(m.q) + ": " + out);
                    }
                }
            }
        }
        // the census path: memory-lean, no Theta_eq certification, streamed vs materialised
        {
            const std::string t = "(n=" + std::to_string(c.n) + ", ell=" + std::to_string(c.ell) + ", p=5, memory-lean)";
            ++arms;
            const Opt::ScopedThetaEqCertification no_cert(false);
            const Opt::ScopedMemoryLean lean(true);
            Opt::Route2Capture on_cap, off_cap;
            const std::unique_ptr<Run> on = run_captured(*pt, 5, true, on_cap);
            const std::unique_ptr<Run> off = run_captured(*pt, 5, false, off_cap);
            const std::string diff = capture_diff(on_cap, off_cap);
            g.check(!on_cap.blocks.empty() && diff.empty(), t + " (X) streamed == materialised route-2 blocks: " + diff);
            g.check(capture_matches_fused(on_cap, *on->op, d), t + " (X) streamed route-2 blocks == Theta_block");
            blocks_total += on_cap.blocks.size();
            const auto& Aa = on->op->get_H_matrices();
            const auto& Ab = off->op->get_H_matrices();
            bool same = Aa.size() == Ab.size();
            for (size_t q = 0; same && q < Aa.size(); ++q) same = same_matrix(Aa[q], Ab[q]);
            g.check(same, t + " (X) A^{(q)} on == off");
        }
    }
    // (B) (audit F4) a degree whose key count is 1 mod the batch (n, ell, p) = (12, 101, 61):
    // 257 keys at q = 1, so the last batch holds one key.  Streamed vs materialised,
    // the in-loop reduction (reduce every 1), and a mutation in that degree.
    {
        const std::unique_ptr<Point> pt = make_point(12, 101);
        const uint32_t d = 13;
        const uint64_t p = 61;
        const std::string t = "(n=12, ell=101, p=61)";
        ++arms;
        Opt::Route2Capture on_cap, off_cap, re_cap;
        const Opt::Rho2StreamStats s0 = Opt::rho2_stream_stats();
        const std::unique_ptr<Run> on = run_captured(*pt, p, true, on_cap);
        const Opt::Rho2StreamStats s1 = Opt::rho2_stream_stats();
        const std::unique_ptr<Run> off = run_captured(*pt, p, false, off_cap);
        {
            const Opt::ScopedRho2StreamReduceEvery re(1);
            const std::unique_ptr<Run> rr = run_captured(*pt, p, true, re_cap);
        }
        const Opt::Rho2StreamStats s2 = Opt::rho2_stream_stats();
        ++reduce_arms;
        g.check(s1.keys - s0.keys >= 257 && s1.evals - s0.evals == s1.keys - s0.keys,
                t + " (B) >= 257 keys (" + std::to_string(s1.keys - s0.keys) + "), each evaluated once");
        g.check(s2.mid_reductions > s1.mid_reductions, t + " (R) reduce every 1: the in-loop reduction ran");
        const std::string diff = capture_diff(on_cap, off_cap), rdiff = capture_diff(re_cap, off_cap);
        g.check(!on_cap.blocks.empty() && diff.empty(), t + " (X) streamed == materialised route-2 blocks: " + diff);
        g.check(rdiff.empty(), t + " (R) reduce every 1 == materialised: " + rdiff);
        g.check(capture_matches_fused(on_cap, *on->op, d), t + " (X) streamed route-2 blocks == Theta_block");
        blocks_total += on_cap.blocks.size();
        // the mutation: the last block of the degree with the most captured blocks (q = 1)
        std::tuple<uint32_t, uint32_t, uint32_t> key{UINT32_MAX, 0, 0};
        for (const auto& kv : on_cap.blocks) if (std::get<0>(kv.first) == 1) key = kv.first;
        g.check(std::get<0>(key) == 1, t + " (B) q = 1 has route-2 blocks");
        if (std::get<0>(key) == 1) {
            Opt::Route2TargetMutation m;
            m.on = true;
            m.q = 1; m.sigma = std::get<1>(key); m.sigma_k = std::get<2>(key);
            m.r = d / 2; m.c = 0;
            for (const bool stream : {true, false}) {
                ++mutations;
                const std::string out = mutation_outcome(*pt, p, stream, m, d);
                g.check(out.empty(), t + " (M) " + (stream ? "streamed" : "materialised") + " mutation q=1: " + out);
            }
        }
    }
    const bool ok = g.fails == 0 && mutations > 0 && reduce_arms > 0;
    std::cout << "P4R_RHO2_STREAM arms=" << arms << " checks=" << g.checks << " failing=" << g.fails << " blocks=" << blocks_total
              << " mutations=" << mutations << " reduce_arms=" << reduce_arms << " verdict=" << (ok ? "OK" : "FAIL") << std::endl;
    return ok ? 0u : 1u;
}
