// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// (slot-24 PAR-GPU) The host driver of the device-resident LU (Ring2kDevice.h).  Included at the end of Ring2k.h; the
// one call site is the dispatch at the top of ring2k::echelon_blocked.

#include <chrono>
#include <cstdio>
#include <type_traits>

#include "Ring2k.h"
#include "Ring2kDevice.h"

namespace engine {
namespace ring2k {
namespace device {

namespace detail {

// new[i] = old[src[i]] on the column segment [c0, c0 + w) of every row of A (in place, by cycles, one segment buffer).
template<class E>
void permute_rows(Mat<E>& A, const std::vector<uint32_t>& src, uint32_t c0, uint32_t w) {
    const uint32_t m = A.rows;
    // (slot-24 PAR-F2) on the threaded route: column tiles, each the same cycle walk on its own columns (a permutation of
    // the rows has one result per column, whatever the order)
    if (smithpar::f2_route() && w >= 256 && static_cast<uint64_t>(m) * w >= 2 * smithpar::min_work()) {
        const uint32_t tiles = std::min<uint32_t>(4u * smithpar::threads(), w / 64u);
        smithpar::parallel_for(tiles, [&](uint32_t t) {
            const uint32_t a = smithpar::split_at(w, tiles, t, 64), b = smithpar::split_at(w, tiles, t + 1, 64);
            if (b <= a) return;
            const uint32_t tw = b - a;
            std::vector<uint8_t> done(m, 0);
            std::vector<E> tmp(tw);
            for (uint32_t i = 0; i < m; ++i) {
                if (done[i]) continue;
                if (src[i] == i) { done[i] = 1; continue; }
                std::memcpy(static_cast<void*>(tmp.data()), A.row(i) + c0 + a, sizeof(E) * tw);
                uint32_t j = i;
                while (src[j] != i) {
                    std::memcpy(static_cast<void*>(A.row(j) + c0 + a), A.row(src[j]) + c0 + a, sizeof(E) * tw);
                    done[j] = 1;
                    j = src[j];
                }
                std::memcpy(static_cast<void*>(A.row(j) + c0 + a), tmp.data(), sizeof(E) * tw);
                done[j] = 1;
            }
        });
        return;
    }
    std::vector<uint8_t> done(m, 0);
    std::vector<E> tmp(w);
    for (uint32_t i = 0; i < m; ++i) {
        if (done[i]) continue;
        if (src[i] == i) { done[i] = 1; continue; }
        std::memcpy(static_cast<void*>(tmp.data()), A.row(i) + c0, sizeof(E) * w);
        uint32_t j = i;
        while (src[j] != i) {
            std::memcpy(static_cast<void*>(A.row(j) + c0), A.row(src[j]) + c0, sizeof(E) * w);
            done[j] = 1;
            j = src[j];
        }
        std::memcpy(static_cast<void*>(A.row(j) + c0), tmp.data(), sizeof(E) * w);
        done[j] = 1;
    }
}

} // namespace detail

// Returns false (A and info untouched) when the device route declines or fails before its first host write; then the
// host code runs.  Returns true with A and info exactly as the host echelon_blocked leaves them.
template<class Tr>
bool echelon_device(Mat<typename Tr::E>& A, uint32_t k, PivotRule rule, uint32_t panel, uint32_t climit, EchelonInfo& info) {
    using E = typename Tr::E;
    if constexpr (!(std::is_same<E, uint32_t>::value || std::is_same<E, uint64_t>::value)) {
        (void)A; (void)k; (void)rule; (void)panel; (void)climit; (void)info;
        return false;
    } else {
        const Ops* o = ops();
        if (o == nullptr || o->plan == nullptr || host_only()) return false;
        const uint32_t m = A.rows, n = A.cols;
        if (m == 0 || n == 0 || climit == 0) return false;
        note(&Stats::calls);
        const uint32_t cw = o->plan(sizeof(E), m, n, panel);
        if (cw == 0) return false;
        ENGINE_CHECK_MSG(cw <= n && (cw == n || (cw % panel == 0 && cw >= panel)),
                         "ring2k::device: plan() returned an invalid column window");
        const bool tiled = cw < n;

        const bool verify = o->verify != nullptr && o->verify();
        Mat<E> A_in;
        if (verify) A_in = A;

        // The record of the current phase's panels (tiled route only): pivot rows, swaps, L factor (rows r .. m-1, ld = np).
        struct PanelRec {
            uint32_t r = 0, np = 0;
            std::vector<uint32_t> swaps;
            std::vector<E> L;
        };
        std::vector<PanelRec> recs;
        const uint64_t store_cap = o->store_budget ? o->store_budget() : 0;   // 0 = unlimited

        void* s = nullptr;
        bool host_touched = false;
        EchelonInfo out;
        uint64_t piv_count = 0;
        using clk = std::chrono::steady_clock;
        auto tick = clk::now();
        auto lap = [&](uint64_t Stats::*field) {
            const auto t = clk::now();
            note(field, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t - tick).count()));
            tick = t;
        };
        uint32_t nslabs = 0;
        try {
            s = o->open(sizeof(E), m, cw, panel);
            lap(&Stats::ns_open);

            std::vector<E> L(static_cast<size_t>(m) * panel, Tr::zero());
            std::vector<E> P;
            std::vector<uint32_t> lvl, pj, pbv, swaps;
            std::vector<E> pui;
            uint32_t r = 0, c = 0;
            // PHASES.  Invariant at a phase start (c = the next panel's column): every column >= c has received the
            // operations of every panel < c, every column < c their swaps.  A phase uploads the window [c, c + cw), factors
            // panels right-looking inside it (lookahead), until the window's panels are done or the L store would pass its
            // budget, and downloads the window.  Tiled only: the columns right of the window then pass through the device in
            // slabs and receive the phase's panels (swaps, TRSM, GEMM, replayed from the store), the columns left of the
            // window receive the phase's swaps on the host (one row permutation), and the store is dropped.
            while (c < climit && r < m) {
                const uint32_t s0 = c, s1 = s0 + cw < n ? s0 + cw : n, w = s1 - s0;
                ++nslabs;
                recs.clear();
                uint64_t store = 0;
                o->put(s, A.row(0) + s0, n, 0, m, 0, w);
                lap(&Stats::ns_open);
                const uint32_t cend = climit < s1 ? climit : s1;
                bool have_next = false;
                while (c < cend && r < m) {
                    const uint32_t c1 = c + panel < climit ? c + panel : climit;   // the host code's panel; never past s1
                    ENGINE_CHECK_MSG(c1 <= s1, "ring2k::device: a panel straddles a slab");
                    const uint32_t pw = c1 - c, rows = m - r;
                    P.resize(static_cast<size_t>(rows) * pw);
                    if (have_next) {
                        o->fetch(s, P.data(), pw);
                        note(&Stats::lookaheads);
                    } else {
                        o->get(s, P.data(), pw, r, rows, c - s0, pw);
                    }
                    lap(&Stats::ns_fetch);
                    // THE PANEL: PAR-2K's lazy panel (Ring2k.h, the T > 1 branch of echelon_blocked) on the panel's own
                    // columns.  Row i of P is the global row r + i; every row operation of the host code inside the panel
                    // touches only the panel's columns [j, c1), so the panel is factored exactly as the host code factors
                    // it; the swaps of the other columns are replayed afterwards (device: this slab; later slabs: the
                    // replays; earlier slabs: the host permutation at the end).
                    uint32_t np = 0;
                    lvl.assign(rows, 0);
                    pj.clear();
                    pbv.clear();
                    pui.clear();
                    swaps.clear();
                    auto bring = [&](uint32_t i, uint32_t to) {
                        uint32_t& li = lvl[i];
                        if (li >= to) return;
                        E* R = P.data() + static_cast<size_t>(i) * pw;
                        E* Li = L.data() + static_cast<size_t>(r + i) * panel;
                        for (uint32_t t = li; t < to; ++t) {
                            const uint32_t j = pj[t];
                            if (Tr::zero_mod(R[j], k)) { Li[t] = Tr::zero(); continue; }
                            const E f = Tr::mul(Tr::shr(R[j], pbv[t]), pui[t]);
                            Li[t] = f;
                            Tr::axpy(R + j, P.data() + static_cast<size_t>(t) * pw + j, f, pw - j);
                        }
                        li = to;
                    };
                    for (uint32_t j = 0; j < pw && np < rows; ++j) {
                        const uint32_t prow = np;
                        uint32_t best = rows, bv = k;
                        for (uint32_t i = prow; i < rows; ++i) {
                            bring(i, np);
                            const E& x = P[static_cast<size_t>(i) * pw + j];
                            if (rule == PivotRule::FirstOdd) {
                                if (Tr::odd(x)) { best = i; bv = 0; break; }
                            } else {
                                const uint32_t v = Tr::val(x, k);
                                if (v < bv) { bv = v; best = i; if (v == 0) break; }
                            }
                        }
                        if (best == rows) {
                            ENGINE_CHECK_MSG(rule != PivotRule::FirstOdd,
                                             "ring2k::echelon_blocked: FirstOdd met a column without an odd entry (put the F_2 rank "
                                             "profile first and pass col_limit = the F_2 rank; see the CAUTION)");
                            continue;
                        }
                        if (best != prow) {
                            E* pa = P.data() + static_cast<size_t>(best) * pw;
                            E* pb = P.data() + static_cast<size_t>(prow) * pw;
                            for (uint32_t t = 0; t < pw; ++t) std::swap(pa[t], pb[t]);
                            E* la = L.data() + static_cast<size_t>(r + best) * panel;
                            E* lb = L.data() + static_cast<size_t>(r + prow) * panel;
                            for (uint32_t t = 0; t < panel; ++t) std::swap(la[t], lb[t]);
                            std::swap(lvl[best], lvl[prow]);
                            swaps.push_back(r + best);
                            swaps.push_back(r + prow);
                        }
                        pj.push_back(j);
                        pbv.push_back(bv);
                        pui.push_back(Tr::inv_odd(Tr::shr(P[static_cast<size_t>(prow) * pw + j], bv)));
                        out.pivots.push_back(c + j);
                        out.pivot_v.push_back(bv);
                        ++np;
                    }
                    if (np < rows) {
                        const uint32_t npp = np;
                        smithpar::for_rows(rows - np, static_cast<uint64_t>(rows - np) * npp * pw / 2,
                                           [&](uint32_t i0, uint32_t i1) { for (uint32_t i = i0; i < i1; ++i) bring(npp + i, npp); });
                    }
                    lap(&Stats::ns_panel);
                    // The device step (with the next panel's lookahead when the next panel is in this phase).
                    const uint32_t rn = r + np;
                    const uint64_t lbytes = static_cast<uint64_t>(rows) * np * sizeof(E);
                    const uint64_t next_bytes = rn < m ? static_cast<uint64_t>(m - rn) * panel * sizeof(E) : 0;
                    const bool budget_ok = !tiled || s1 >= n || store_cap == 0 || store + lbytes + next_bytes <= store_cap;
                    const bool next = c1 < cend && rn < m && budget_ok;
                    const uint32_t nc1 = c1 + panel < climit ? c1 + panel : climit;
                    StepArgs a;
                    a.m = m;
                    a.r = r;
                    a.np = np;
                    a.pc0 = c - s0;
                    a.pc1 = c1 - s0;
                    a.panel = P.data();
                    a.ldp = pw;
                    a.swaps = swaps.data();
                    a.nswaps = static_cast<uint32_t>(swaps.size() / 2);
                    a.L = L.data() + static_cast<size_t>(r) * panel;
                    a.ldl = panel;
                    a.u0 = (np > 0 && c1 < s1) ? c1 - s0 : w;
                    a.la0 = next ? c1 - s0 : 0;
                    a.la1 = next ? nc1 - s0 : 0;
                    a.w = w;
                    o->step(s, a);
                    note(&Stats::panels);
                    have_next = next;
                    lap(&Stats::ns_step);
                    // (tiled) record the panel: its swaps for the columns left of the window, its L for those right of it
                    if (tiled && (np > 0 || !swaps.empty())) {
                        PanelRec pr;
                        pr.r = r;
                        pr.np = np;
                        pr.swaps = swaps;
                        if (s1 < n) {
                            pr.L.resize(static_cast<size_t>(rows) * np);
                            // (PAR-F2) row tiles on the threaded route (disjoint rows of the store)
                            auto copy_l = [&](uint32_t i0, uint32_t i1) {
                                for (uint32_t i = i0; i < i1; ++i)
                                    std::memcpy(static_cast<void*>(pr.L.data() + static_cast<size_t>(i) * np),
                                                L.data() + static_cast<size_t>(r + i) * panel, sizeof(E) * np);
                            };
                            if (smithpar::f2_route()) smithpar::for_rows(rows, static_cast<uint64_t>(rows) * np / 4u, copy_l);
                            else copy_l(0, rows);
                            store += lbytes;
                        }
                        recs.push_back(std::move(pr));
                    }
                    lap(&Stats::ns_host);
                    piv_count += np;
                    r = rn;
                    c = c1;
                    if (!budget_ok) break;   // the store is full: end the phase at this panel boundary
                }
                host_touched = true;
                o->get(s, A.row(0) + s0, n, 0, m, 0, w);
                lap(&Stats::ns_close);
                if (!tiled || recs.empty()) continue;
                // the columns right of the window: slabs through the device, the phase's panels replayed in order
                for (uint32_t t0 = s1; t0 < n; t0 += cw) {
                    const uint32_t t1 = t0 + cw < n ? t0 + cw : n, tw = t1 - t0;
                    ++nslabs;
                    o->put(s, A.row(0) + t0, n, 0, m, 0, tw);
                    lap(&Stats::ns_open);
                    for (const PanelRec& pr : recs) {
                        StepArgs a;
                        a.m = m;
                        a.r = pr.r;
                        a.np = pr.np;
                        a.swaps = pr.swaps.data();
                        a.nswaps = static_cast<uint32_t>(pr.swaps.size() / 2);
                        a.L = pr.L.data();
                        a.ldl = pr.np;
                        a.u0 = 0;
                        a.w = tw;
                        o->step(s, a);
                        note(&Stats::replays);
                    }
                    lap(&Stats::ns_host);
                    o->get(s, A.row(0) + t0, n, 0, m, 0, tw);
                    lap(&Stats::ns_close);
                }
                // the columns left of the window: the phase's swaps, composed into one row permutation (new[i] = old[src[i]])
                if (s0 > 0) {
                    std::vector<uint32_t> src(m);
                    for (uint32_t i = 0; i < m; ++i) src[i] = i;
                    bool any = false;
                    for (const PanelRec& pr : recs)
                        for (size_t t = 0; t + 1 < pr.swaps.size(); t += 2) {
                            std::swap(src[pr.swaps[t]], src[pr.swaps[t + 1]]);
                            any = true;
                        }
                    if (any) detail::permute_rows(A, src, 0, s0);
                    lap(&Stats::ns_host);
                }
            }
            out.rank = r;
            o->close(s);
            s = nullptr;
        } catch (const std::runtime_error& e) {
            if (s) o->close(s);
            if (host_touched) throw;
            note(&Stats::fallbacks);
            std::fprintf(stderr, "[ring2k::device] fallback to the host LU (%ux%u, %s): %s\n", m, n, sizeof(E) == 4 ? "u32" : "u64",
                         e.what());
            return false;
        } catch (...) {
            // (PAR-GPU audit L1) any other exception (an ENGINE_CHECK's std::logic_error, std::bad_alloc, ...): free the device
            // session (window, slabs, staging, streams) before it propagates; the host LU would have thrown alike, never a fallback
            if (s) o->close(s);
            throw;
        }
        note(&Stats::device);
        note(tiled ? &Stats::tiled : &Stats::resident);
        if (tiled) note(&Stats::slabs, nslabs);
        Counters d;
        d.lu_calls = 1;
        d.lu_pivots = piv_count;
        add_counters(d);
        if (verify) {
            const ScopedHostOnly host;
            const EchelonInfo h = echelon_blocked<Tr>(A_in, k, rule, panel, climit);
            bool same = h.rank == out.rank && h.pivots == out.pivots && h.pivot_v == out.pivot_v;
            for (size_t i = 0; same && i < A.a.size(); ++i) same = A.a[i] == A_in.a[i];
            if (!same)
                throw std::logic_error("ENGINE_VERIFY_DEVICE (ring2k device LU): the device LU differs from the host LU (" +
                                       std::to_string(m) + "x" + std::to_string(n) + ")");
            note(&Stats::verified);
        }
        info = std::move(out);
        return true;
    }
}

} // namespace device
} // namespace ring2k
} // namespace engine
