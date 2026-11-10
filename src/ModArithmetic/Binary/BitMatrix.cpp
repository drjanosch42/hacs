// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "BitMatrix.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <type_traits>

#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/Fnv.h"
#include "../ModArithmetic/Kernels.h"   // (S24 CL-5) kernels::isa_auto, the ENGINE_KERNEL ceiling
#include "SmithPar.h"

// See BitMatrix.h.  Origin: the L23 master-plan prototype
// the F2 packed-kernel benchmark prototype f2_packed.cpp (Sec. 2.3), rewritten as a
// library with a cheaper pivot search (the reduced bit of a candidate row is computed from the
// strip's pivot rows without XORing whole rows) and a plain subset table instead of Gray order.

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  include <immintrin.h>
#  define BIN_HAVE_X86_TARGETS 1
#  define BIN_AVX2_FN   __attribute__((target("avx2")))
#  define BIN_AVX512_FN __attribute__((target("avx512f")))
#else
#  define BIN_HAVE_X86_TARGETS 0
#endif

namespace engine {
namespace binary {

namespace {

// ---------------------------------------------------------------------------
// Row XOR bodies.  a[k] ^= b[k] (xor2) / d[k] = a[k] ^ b[k] (xor3) for k in [w0, W); W is a
// multiple of 8 (BitMatrix's row padding), w0 arbitrary.
// ---------------------------------------------------------------------------
void xor2_portable(uint64_t* a, const uint64_t* b, uint32_t w0, uint32_t W) {
    for (uint32_t k = w0; k < W; ++k) a[k] ^= b[k];
}
void xor3_portable(uint64_t* d, const uint64_t* a, const uint64_t* b, uint32_t w0, uint32_t W) {
    for (uint32_t k = w0; k < W; ++k) d[k] = a[k] ^ b[k];
}

#if BIN_HAVE_X86_TARGETS
BIN_AVX2_FN void xor2_avx2(uint64_t* a, const uint64_t* b, uint32_t w0, uint32_t W) {
    uint32_t k = w0;
    for (; (k & 3u) != 0 && k < W; ++k) a[k] ^= b[k];
    for (; k + 4 <= W; k += 4) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
        const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + k));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + k), _mm256_xor_si256(x, y));
    }
    for (; k < W; ++k) a[k] ^= b[k];
}
BIN_AVX2_FN void xor3_avx2(uint64_t* d, const uint64_t* a, const uint64_t* b, uint32_t w0, uint32_t W) {
    uint32_t k = w0;
    for (; (k & 3u) != 0 && k < W; ++k) d[k] = a[k] ^ b[k];
    for (; k + 4 <= W; k += 4) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
        const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + k));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + k), _mm256_xor_si256(x, y));
    }
    for (; k < W; ++k) d[k] = a[k] ^ b[k];
}
BIN_AVX512_FN void xor2_avx512(uint64_t* a, const uint64_t* b, uint32_t w0, uint32_t W) {
    uint32_t k = w0;
    for (; (k & 7u) != 0 && k < W; ++k) a[k] ^= b[k];
    for (; k + 8 <= W; k += 8) {
        const __m512i x = _mm512_loadu_si512(a + k);
        const __m512i y = _mm512_loadu_si512(b + k);
        _mm512_storeu_si512(a + k, _mm512_xor_si512(x, y));
    }
    for (; k < W; ++k) a[k] ^= b[k];
}
BIN_AVX512_FN void xor3_avx512(uint64_t* d, const uint64_t* a, const uint64_t* b, uint32_t w0, uint32_t W) {
    uint32_t k = w0;
    for (; (k & 7u) != 0 && k < W; ++k) d[k] = a[k] ^ b[k];
    for (; k + 8 <= W; k += 8) {
        const __m512i x = _mm512_loadu_si512(a + k);
        const __m512i y = _mm512_loadu_si512(b + k);
        _mm512_storeu_si512(d + k, _mm512_xor_si512(x, y));
    }
    for (; k < W; ++k) d[k] = a[k] ^ b[k];
}

// 16 canonical uint32 F_2 entries -> 16 bits; `orv` collects every entry for the 0/1 check.
BIN_AVX512_FN void pack_row_u32_avx512(const uint32_t* src, uint32_t n, uint64_t* dst, uint32_t& orv) {
    __m512i acc = _mm512_setzero_si512();
    uint32_t j = 0;
    for (; j + 64 <= n; j += 64) {
        uint64_t w = 0;
        for (uint32_t q = 0; q < 4; ++q) {
            const __m512i v = _mm512_loadu_si512(src + j + 16 * q);
            acc = _mm512_or_si512(acc, v);
            w |= static_cast<uint64_t>(_mm512_test_epi32_mask(v, v)) << (16 * q);
        }
        dst[j >> 6] = w;
    }
    uint32_t o = static_cast<uint32_t>(_mm512_reduce_or_epi32(acc));
    for (; j < n; ++j) {
        o |= src[j];
        dst[j >> 6] |= static_cast<uint64_t>(src[j] & 1u) << (j & 63);
    }
    orv |= o;
}
BIN_AVX512_FN void unpack_row_u32_avx512(const uint64_t* src, uint32_t n, uint32_t* dst) {
    const __m512i one = _mm512_set1_epi32(1);
    uint32_t j = 0;
    for (; j + 16 <= n; j += 16) {
        const __mmask16 m = static_cast<__mmask16>(src[j >> 6] >> (j & 63));
        _mm512_storeu_si512(dst + j, _mm512_maskz_mov_epi32(m, one));
    }
    for (; j < n; ++j) dst[j] = static_cast<uint32_t>((src[j >> 6] >> (j & 63)) & 1u);
}
#endif

// (S24 CL-5) the CPU facts from CpuFeatures.h (false on a build without the bodies)
bool cpu_has_avx2() { return BIN_HAVE_X86_TARGETS && cpu::avx2(); }
bool cpu_has_avx512() { return BIN_HAVE_X86_TARGETS && cpu::avx512f(); }

std::atomic<uint8_t> g_forced_path{static_cast<uint8_t>(RowPath::Auto)};

struct RowOps {
    void (*xor2)(uint64_t*, const uint64_t*, uint32_t, uint32_t);
    void (*xor3)(uint64_t*, const uint64_t*, const uint64_t*, uint32_t, uint32_t);
};

RowOps ops_for(RowPath p) {
#if BIN_HAVE_X86_TARGETS
    if (p == RowPath::Avx512) return RowOps{&xor2_avx512, &xor3_avx512};
    if (p == RowPath::Avx2) return RowOps{&xor2_avx2, &xor3_avx2};
#endif
    (void)p;
    return RowOps{&xor2_portable, &xor3_portable};
}

inline bool bit_of(const uint64_t* r, uint32_t j) { return (r[j >> 6] >> (j & 63)) & 1u; }

// ---------------------------------------------------------------------------
// M4RI elimination (full = RREF; !full = echelon only).  Invariant, after each strip: every
// row at index >= r is zero in every column < c; the strip's pivot rows are mutually reduced
// (each is zero in the other strip pivot columns) and zero left of its own pivot.
// ---------------------------------------------------------------------------
constexpr uint32_t kStrip = 8;   // pivots per table (2^8 = 256 entries)

BitRref m4ri(BitMatrix& A, bool full) {
    BitRref res;
    const uint32_t m = A.rows(), n = A.cols(), W = A.words_per_row();
    if (m == 0 || n == 0) return res;
    const RowOps ops = ops_for(active_row_path());
    std::vector<uint64_t> T(static_cast<size_t>(1u << kStrip) * W, 0);
    uint32_t r = 0, c = 0;
    while (c < n && r < m) {
        uint32_t pc[kStrip];
        uint32_t np = 0;
        const uint32_t r0 = r;
        while (np < kStrip && c < n && r < m) {
            // The reduced bit of row i in column c w.r.t. the strip pivots found so far is
            // row[c] ^ XOR_{t : row[pc_t] = 1} P_t[c] (the P_t are mutually reduced, so the
            // coefficients are the row's own bits).  No row is touched until a candidate is found.
            uint32_t p = m;
            for (uint32_t i = r; i < m; ++i) {
                const uint64_t* ri = A.row(i);
                bool b = bit_of(ri, c);
                for (uint32_t t = 0; t < np; ++t)
                    if (bit_of(ri, pc[t])) b ^= bit_of(A.row(r0 + t), c);
                if (b) { p = i; break; }
            }
            if (p < m) {
                if (p != r) A.swap_rows(p, r);
                uint64_t* pr = A.row(r);
                for (uint32_t t = 0; t < np; ++t)
                    if (bit_of(pr, pc[t])) ops.xor2(pr, A.row(r0 + t), pc[t] >> 6, W);
                ENGINE_DEBUG_CHECK(bit_of(pr, c));
                const uint32_t wc = c >> 6;
                for (uint32_t t = 0; t < np; ++t) {
                    uint64_t* q = A.row(r0 + t);
                    if (bit_of(q, c)) ops.xor2(q, pr, wc, W);
                }
                pc[np++] = c;
                res.pivots.push_back(c);
                ++r;
            }
            ++c;
        }
        if (np == 0) break;
        // Subset table over [w0, W): T[g] = XOR of P_t for the bits t of g.
        const uint32_t w0 = pc[0] >> 6;
        const uint32_t N = 1u << np;
        // T[0] stays zero from the allocation (never written); BIN-1 audit F9 removed a redundant memset.
        for (uint32_t g = 1; g < N; ++g) {
            const uint32_t low = static_cast<uint32_t>(__builtin_ctz(g));
            ops.xor3(T.data() + static_cast<size_t>(g) * W, T.data() + static_cast<size_t>(g ^ (1u << low)) * W,
                     A.row(r0 + low), w0, W);
        }
        // Words of the pivot columns, for the index extraction.
        uint32_t pw[kStrip], ps[kStrip];
        for (uint32_t t = 0; t < np; ++t) { pw[t] = pc[t] >> 6; ps[t] = pc[t] & 63; }
        auto clear_row = [&](uint32_t i) {
            uint64_t* ri = A.row(i);
            uint32_t idx = 0;
            for (uint32_t t = 0; t < np; ++t) idx |= static_cast<uint32_t>((ri[pw[t]] >> ps[t]) & 1u) << t;
            if (idx != 0) ops.xor2(ri, T.data() + static_cast<size_t>(idx) * W, w0, W);
        };
        if (full)
            for (uint32_t i = 0; i < r0; ++i) clear_row(i);
        for (uint32_t i = r; i < m; ++i) clear_row(i);
    }
    res.rank = r;
    return res;
}

// ---------------------------------------------------------------------------
// (slot-24 PAR-F2) The lazy blocked M4RI (echelon only; see BitMatrix.h).
//
// THE ROW OPERATIONS ARE THE SERIAL ONES.  The strips are m4ri's strips -- a strip ends exactly where the serial one ends
// (8 pivots, the last column or the last row) -- so every pivot row, every row swap and every subset table is the serial
// one.  Only WHEN a row below receives a strip's clear changes: slot s of up to kSuper pending strips holds strip s's table;
// lvl[i] (rows i >= r) = the pending strips already applied to row i.  A row is brought up to date (bring) when it becomes
// a pivot row and, all rows at once over row tiles, at a flush (kSuper strips done, or the search window moves), each row
// exactly the serial clear_row sequence in strip order.
//
// THE PIVOT SEARCH READS A TRANSPOSED WINDOW.  The serial search for column c scans the rows below r for the first one
// whose REDUCED bit c (w.r.t. every pivot so far) is set; on the descent's matrices most columns have none (rank 31 489
// of 88 200 at (1000, 20)), and each such column cost a pass over every row.  Here a window of kWinWords words (512
// columns) of the reduced rows below r is kept TRANSPOSED: one bitset over the rows per column.  It is built from the rows
// right after a flush (rows current; the current strip's partial pivots applied to the copy), and every pivot is applied
// to it at once: rows p and r swap bits, and every window column j > c that is set in the pivot row receives column c's
// bits below r (exactly "row i += pivot row" for the rows i with bit c, on the window).  So the window is the reduced state
// at every moment, and the search for column c is "the first set bit at a row >= r of column c's bitset": the serial
// search's answer, in m / 64 words.  (Unchanged rows of the window would show a missed update; DescentSmithMtValidation's
// (F) arm compares the whole matrix, rank and pivots against the serial echelon.)
// ---------------------------------------------------------------------------
constexpr uint32_t kSuper = 8;      // pending strips (tables) per pass over the rows below
constexpr uint32_t kWinWords = 8;   // the transposed search window: 512 columns

struct LazyStrip {
    uint32_t np = 0, r0 = 0, w0 = 0;
    uint32_t pc[kStrip] = {}, pw[kStrip] = {}, ps[kStrip] = {};
};

BitRref m4ri_lazy(BitMatrix& A) {
    BitRref res;
    const uint32_t m = A.rows(), n = A.cols(), W = A.words_per_row();
    if (m == 0 || n == 0) return res;
    smithpar::count_f2_lazy_call();
    const RowOps ops = ops_for(active_row_path());
    const uint32_t T = smithpar::threads();
    const uint64_t mw = smithpar::min_work();
    std::vector<uint64_t> tab(static_cast<size_t>(kSuper) * 256u * W, 0);
    auto tabp = [&](uint32_t s, uint32_t g) -> uint64_t* { return tab.data() + (static_cast<size_t>(s) * 256u + g) * W; };
    std::vector<uint8_t> lvl(m, 0);
    LazyStrip st[kSuper];
    uint32_t S = 0;   // pending (complete) strips; the strip being built is slot S

    // Row i (>= r) receives the pending strips lvl[i] .. to-1: exactly the serial clear_row of each, in order.
    auto bring = [&](uint32_t i, uint32_t to) {
        uint8_t& li = lvl[i];
        if (li >= to) return;
        uint64_t* ri = A.row(i);
        for (uint32_t t = li; t < to; ++t) {
            const LazyStrip& q = st[t];
            uint32_t idx = 0;
            for (uint32_t u = 0; u < q.np; ++u) idx |= static_cast<uint32_t>((ri[q.pw[u]] >> q.ps[u]) & 1u) << u;
            if (idx != 0) ops.xor2(ri, tabp(t, idx), q.w0, W);
        }
        li = static_cast<uint8_t>(to);
    };
    // Every row [r, m) to level S, then the pending slots are dropped (lvl = 0).  Row-independent: one region.
    auto flush = [&](uint32_t r) {
        if (S == 0 || r >= m) { S = 0; return; }
        smithpar::PhaseTimer tfl(smithpar::Phase::EchFlush);
        const uint32_t rows = m - r, upto = S;
        smithpar::for_rows(rows, static_cast<uint64_t>(rows) * upto * (W - st[0].w0) * 4u, [&](uint32_t i0, uint32_t i1) {
            for (uint32_t i = r + i0; i < r + i1; ++i) { bring(i, upto); lvl[i] = 0; }
        });
        S = 0;
    };

    // The transposed window: words [ww0, wwe) of the reduced rows, column j (0 .. 64 (wwe - ww0)) at cb + j MW.
    const uint32_t MW = (m + 63u) / 64u;
    std::vector<uint64_t> cb(static_cast<size_t>(64u * kWinWords) * MW, 0);
    auto colp = [&](uint32_t j) -> uint64_t* { return cb.data() + static_cast<size_t>(j) * MW; };
    uint32_t ww0 = 0, wwe = 0;   // wwe == 0: no window yet
    // Built right after a flush (rows [r, m) carry every complete strip); `cur` = the strip being built (its np pivots so
    // far are rows r0 .. r-1, not yet applied to the rows below: applied to the copy here).  Tiles of 64 rows: disjoint words.
    auto build_window = [&](uint32_t wa, uint32_t r, const LazyStrip& cur) {
        smithpar::PhaseTimer tw(smithpar::Phase::EchTable);
        ww0 = wa;
        wwe = std::min<uint32_t>(W, wa + kWinWords);
        const uint32_t nw = wwe - ww0, q0 = r >> 6;
        std::fill(cb.begin(), cb.end(), uint64_t(0));
        const uint32_t nq = MW - q0;
        smithpar::for_rows(nq, static_cast<uint64_t>(nq) * 64u * (nw + cur.np), [&](uint32_t a, uint32_t b) {
            for (uint32_t q = q0 + a; q < q0 + b; ++q) {
                const uint32_t i0 = std::max<uint32_t>(r, q * 64u), i1 = std::min<uint32_t>(m, q * 64u + 64u);
                for (uint32_t i = i0; i < i1; ++i) {
                    const uint64_t* ri = A.row(i);
                    uint64_t w[kWinWords];
                    for (uint32_t k = 0; k < nw; ++k) w[k] = ri[ww0 + k];
                    for (uint32_t t = 0; t < cur.np; ++t)
                        if (bit_of(ri, cur.pc[t])) {
                            const uint64_t* P = A.row(cur.r0 + t);
                            for (uint32_t k = 0; k < nw; ++k) w[k] ^= P[ww0 + k];
                        }
                    const uint64_t rb = uint64_t(1) << (i & 63u);
                    for (uint32_t k = 0; k < nw; ++k)
                        for (uint64_t x = w[k]; x != 0; x &= x - 1)
                            colp(k * 64u + static_cast<uint32_t>(__builtin_ctzll(x)))[q] |= rb;
                }
            }
        });
    };

    uint32_t r = 0, c = 0;
    while (c < n && r < m) {
        st[S].np = 0;
        st[S].r0 = r;
        const uint32_t r0 = r;
        while (st[S].np < kStrip && c < n && r < m) {
            const uint32_t wc = c >> 6, sc = c & 63;
            if (wwe == 0 || wc >= wwe) {
                // The window moves: bring every row up to date first (the strip being built becomes slot 0).
                const LazyStrip keep = st[S];
                flush(r);
                st[0] = keep;
                build_window(wc, r, st[0]);
            }
            LazyStrip& sp = st[S];
            const uint32_t np = sp.np;
            // The search: the first row >= r whose reduced bit c is set.
            uint32_t p = m;
            {
                smithpar::PhaseTimer tsr(smithpar::Phase::EchSearch);
                const uint64_t* col = colp(c - ww0 * 64u);
                uint32_t q = r >> 6;
                uint64_t x = col[q] & (~uint64_t(0) << (r & 63u));
                while (x == 0 && ++q < MW) x = col[q];
                if (x != 0) p = q * 64u + static_cast<uint32_t>(__builtin_ctzll(x));
            }
            if (p < m) {
                bring(p, S);
                if (p != r) {
                    A.swap_rows(p, r);
                    std::swap(lvl[p], lvl[r]);
                }
                uint64_t* pr = A.row(r);
                for (uint32_t t = 0; t < np; ++t)
                    if (bit_of(pr, sp.pc[t])) ops.xor2(pr, A.row(r0 + t), sp.pc[t] >> 6, W);
                ENGINE_CHECK_MSG(bit_of(pr, c), "binary::echelon_lazy: the transposed window disagrees with the pivot row");
                for (uint32_t t = 0; t < np; ++t) {
                    uint64_t* q = A.row(r0 + t);
                    if (bit_of(q, c)) ops.xor2(q, pr, wc, W);
                }
                // The window: rows p <-> r, then "row i += pivot row" for the rows i > r with bit c (window columns > c).
                {
                    smithpar::PhaseTimer tsr(smithpar::Phase::EchSearch);
                    const uint32_t ncol = (wwe - ww0) * 64u, jc = c - ww0 * 64u;
                    if (p != r) {
                        const uint32_t qp = p >> 6, qr = r >> 6;
                        const uint64_t bp = uint64_t(1) << (p & 63u), br = uint64_t(1) << (r & 63u);
                        for (uint32_t j = 0; j < ncol; ++j) {
                            uint64_t* cj = colp(j);
                            const bool vp = (cj[qp] & bp) != 0, vr = (cj[qr] & br) != 0;
                            if (vp != vr) { cj[qp] ^= bp; cj[qr] ^= br; }
                        }
                    }
                    std::vector<uint32_t> js;
                    for (uint32_t j = jc + 1; j < ncol; ++j)
                        if ((pr[ww0 + (j >> 6)] >> (j & 63u)) & 1u) js.push_back(j);
                    const uint32_t qa = (r + 1) >> 6;
                    uint64_t* cc = colp(jc);
                    cc[r >> 6] &= ~(uint64_t(1) << (r & 63u));   // the pivot row itself: not a row below
                    const uint64_t head = (r + 1) & 63u ? (~uint64_t(0) << ((r + 1) & 63u)) : ~uint64_t(0);
                    auto upd = [&](uint32_t a, uint32_t b) {
                        for (uint32_t t = a; t < b; ++t) {
                            uint64_t* cj = colp(js[t]);
                            if (qa < MW) cj[qa] ^= cc[qa] & head;
                            for (uint32_t q = qa + 1; q < MW; ++q) cj[q] ^= cc[q];
                        }
                    };
                    const uint64_t work = static_cast<uint64_t>(js.size()) * (MW - std::min(qa, MW));
                    if (T > 1 && js.size() >= 2 && work >= 2 * mw / 16u) {
                        smithpar::count_f2_par_search();
                        const uint32_t nj = static_cast<uint32_t>(js.size());
                        const uint32_t tiles = std::min<uint32_t>(T, nj);
                        smithpar::parallel_for(tiles, [&](uint32_t t) {
                            upd(static_cast<uint32_t>(static_cast<uint64_t>(nj) * t / tiles),
                                static_cast<uint32_t>(static_cast<uint64_t>(nj) * (t + 1) / tiles));
                        });
                    } else {
                        upd(0, static_cast<uint32_t>(js.size()));
                    }
                }
                sp.pc[np] = c;
                sp.pw[np] = wc;
                sp.ps[np] = sc;
                sp.np = np + 1;
                if (np == 0) sp.w0 = wc;
                res.pivots.push_back(c);
                ++r;
            }
            ++c;
        }
        LazyStrip& done = st[S];
        if (done.np == 0) break;
        // The strip's subset table over [w0, W) into slot S (column tiles when large; each word is its own XOR chain).
        {
            smithpar::PhaseTimer ttb(smithpar::Phase::EchTable);
            const uint32_t w0 = done.w0, N = 1u << done.np, s = S, rr0 = done.r0;
            auto build = [&](uint32_t wa, uint32_t wb) {
                for (uint32_t g = 1; g < N; ++g) {
                    const uint32_t low = static_cast<uint32_t>(__builtin_ctz(g));
                    ops.xor3(tabp(s, g), tabp(s, g ^ (1u << low)), A.row(rr0 + low), wa, wb);
                }
            };
            const uint32_t span = W - w0;
            const uint64_t work = static_cast<uint64_t>(N) * span;
            if (T > 1 && span >= 128 && work >= 2 * mw / 8u) {
                const uint32_t tiles = std::min<uint32_t>(T, span / 64u);
                smithpar::parallel_for(tiles, [&](uint32_t t) {
                    const uint32_t a = w0 + smithpar::split_at(span, tiles, t, 8);
                    const uint32_t b = (t + 1 == tiles) ? W : w0 + smithpar::split_at(span, tiles, t + 1, 8);
                    if (b > a) build(a, b);
                });
            } else {
                build(w0, W);
            }
        }
        ++S;
        if (S == kSuper) flush(r);
    }
    flush(r);
    res.rank = r;
    return res;
}

#if BIN_HAVE_X86_TARGETS
BIN_AVX512_FN void pack_low_u32_avx512(const uint32_t* src, uint32_t n, uint64_t* dst) {
    const __m512i one = _mm512_set1_epi32(1);
    uint32_t j = 0;
    for (; j + 64 <= n; j += 64) {
        uint64_t w = 0;
        for (uint32_t q = 0; q < 4; ++q)
            w |= static_cast<uint64_t>(_mm512_test_epi32_mask(_mm512_loadu_si512(src + j + 16 * q), one)) << (16 * q);
        dst[j >> 6] = w;
    }
    if (j < n) {
        uint64_t w = 0;
        for (uint32_t b = 0; j + b < n; ++b) w |= static_cast<uint64_t>(src[j + b] & 1u) << b;
        dst[j >> 6] = w;
    }
}
BIN_AVX512_FN void pack_low_u64_avx512(const uint64_t* src, uint32_t n, uint64_t* dst) {
    const __m512i one = _mm512_set1_epi64(1);
    uint32_t j = 0;
    for (; j + 64 <= n; j += 64) {
        uint64_t w = 0;
        for (uint32_t q = 0; q < 8; ++q)
            w |= static_cast<uint64_t>(_mm512_test_epi64_mask(_mm512_loadu_si512(src + j + 8 * q), one)) << (8 * q);
        dst[j >> 6] = w;
    }
    if (j < n) {
        uint64_t w = 0;
        for (uint32_t b = 0; j + b < n; ++b) w |= static_cast<uint64_t>(src[j + b] & 1u) << b;
        dst[j >> 6] = w;
    }
}
BIN_AVX2_FN void pack_low_u32_avx2(const uint32_t* src, uint32_t n, uint64_t* dst) {
    uint32_t j = 0;
    for (; j + 64 <= n; j += 64) {
        uint64_t w = 0;
        for (uint32_t q = 0; q < 8; ++q) {
            const __m256i v = _mm256_slli_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + j + 8 * q)), 31);
            w |= static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(v)))) << (8 * q);
        }
        dst[j >> 6] = w;
    }
    if (j < n) {
        uint64_t w = 0;
        for (uint32_t b = 0; j + b < n; ++b) w |= static_cast<uint64_t>(src[j + b] & 1u) << b;
        dst[j >> 6] = w;
    }
}
BIN_AVX2_FN void pack_low_u64_avx2(const uint64_t* src, uint32_t n, uint64_t* dst) {
    uint32_t j = 0;
    for (; j + 64 <= n; j += 64) {
        uint64_t w = 0;
        for (uint32_t q = 0; q < 16; ++q) {
            const __m256i v = _mm256_slli_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + j + 4 * q)), 63);
            w |= static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_pd(_mm256_castsi256_pd(v)))) << (4 * q);
        }
        dst[j >> 6] = w;
    }
    if (j < n) {
        uint64_t w = 0;
        for (uint32_t b = 0; j + b < n; ++b) w |= static_cast<uint64_t>(src[j + b] & 1u) << b;
        dst[j >> 6] = w;
    }
}
#endif

template<class Elt>
void pack_low_portable(const Elt* src, uint32_t n, uint64_t* dst) {
    for (uint32_t j = 0; j < n; j += 64) {
        const uint32_t e = std::min<uint32_t>(64, n - j);
        uint64_t w = 0;
        for (uint32_t b = 0; b < e; ++b) w |= static_cast<uint64_t>(src[j + b] & Elt(1)) << b;
        dst[j >> 6] = w;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// BitMatrix
// ---------------------------------------------------------------------------
BitMatrix::BitMatrix(uint32_t rows, uint32_t cols)
    : rows_(rows), cols_(cols), W_(((cols + 511u) / 512u) * 8u),
      d_(static_cast<size_t>(rows) * (((cols + 511u) / 512u) * 8u), 0) {}

BitMatrix BitMatrix::identity(uint32_t n) {
    BitMatrix I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.set(i, i, true);
    return I;
}

void BitMatrix::swap_rows(uint32_t a, uint32_t b) {
    ENGINE_CHECK(a < rows_ && b < rows_);
    if (a == b) return;
    std::swap_ranges(row(a), row(a) + W_, row(b));
}

uint64_t BitMatrix::fnv() const {
    uint64_t h = FnvU64(kFnvOffset64, rows_);
    h = FnvU64(h, cols_);
    const uint32_t sig = (cols_ + 63u) / 64u;
    for (uint32_t i = 0; i < rows_; ++i)
        for (uint32_t k = 0; k < sig; ++k) h = FnvU64(h, row(i)[k]);
    return h;
}

uint32_t BitMatrix::popcount() const {
    uint64_t s = 0;
    for (uint64_t w : d_) s += static_cast<uint64_t>(__builtin_popcountll(w));
    return static_cast<uint32_t>(s);
}

template<class Elt>
BitMatrix BitMatrix::from_dense(DenseView<const Elt> src) {
    const uint32_t m = src.rows(), n = src.cols();
    BitMatrix B(m, n);
    const auto s = src.strides();
    uint64_t orv = 0;
#if BIN_HAVE_X86_TARGETS
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        if (s.cs == 1 && active_row_path() == RowPath::Avx512) {
            uint32_t o = 0;
            for (uint32_t i = 0; i < m; ++i) pack_row_u32_avx512(s.base + i * s.rs, n, B.row(i), o);
            ENGINE_CHECK_MSG(o <= 1u, "BitMatrix::from_dense: entries must be canonical residues mod 2");
            return B;
        }
    }
#endif
    for (uint32_t i = 0; i < m; ++i) {
        const Elt* ri = s.base + i * s.rs;
        uint64_t* bi = B.row(i);
        for (uint32_t j = 0; j < n; ++j) {
            const Elt v = ri[static_cast<size_t>(j) * s.cs];
            orv |= static_cast<uint64_t>(v);
            bi[j >> 6] |= static_cast<uint64_t>(v & Elt(1)) << (j & 63);
        }
    }
    ENGINE_CHECK_MSG(orv <= 1u, "BitMatrix::from_dense: entries must be canonical residues mod 2");
    return B;
}

template<class Elt>
void BitMatrix::to_dense(DenseView<Elt> dst) const {
    ENGINE_CHECK(dst.rows() == rows_ && dst.cols() == cols_);
    const auto s = dst.strides();
#if BIN_HAVE_X86_TARGETS
    if constexpr (std::is_same<Elt, uint32_t>::value) {
        if (s.cs == 1 && active_row_path() == RowPath::Avx512) {
            for (uint32_t i = 0; i < rows_; ++i) unpack_row_u32_avx512(row(i), cols_, s.base + i * s.rs);
            return;
        }
    }
#endif
    for (uint32_t i = 0; i < rows_; ++i) {
        Elt* di = s.base + i * s.rs;
        const uint64_t* bi = row(i);
        for (uint32_t j = 0; j < cols_; ++j)
            di[static_cast<size_t>(j) * s.cs] = static_cast<Elt>((bi[j >> 6] >> (j & 63)) & 1u);
    }
}

template BitMatrix BitMatrix::from_dense<uint32_t>(DenseView<const uint32_t>);
template BitMatrix BitMatrix::from_dense<uint64_t>(DenseView<const uint64_t>);
template void BitMatrix::to_dense<uint32_t>(DenseView<uint32_t>) const;
template void BitMatrix::to_dense<uint64_t>(DenseView<uint64_t>) const;

// ---------------------------------------------------------------------------
// Elimination, kernel, products
// ---------------------------------------------------------------------------
BitRref rref(BitMatrix& A) { return m4ri(A, true); }
BitRref echelon(BitMatrix& A) { return m4ri(A, false); }
BitRref echelon_lazy(BitMatrix& A) { return m4ri_lazy(A); }

void pack_low_bits(const uint32_t* src, uint32_t n, uint64_t* dst) {
#if BIN_HAVE_X86_TARGETS
    const RowPath p = active_row_path();
    if (p == RowPath::Avx512) { pack_low_u32_avx512(src, n, dst); return; }
    if (p == RowPath::Avx2) { pack_low_u32_avx2(src, n, dst); return; }
#endif
    pack_low_portable(src, n, dst);
}
void pack_low_bits(const uint64_t* src, uint32_t n, uint64_t* dst) {
#if BIN_HAVE_X86_TARGETS
    const RowPath p = active_row_path();
    if (p == RowPath::Avx512) { pack_low_u64_avx512(src, n, dst); return; }
    if (p == RowPath::Avx2) { pack_low_u64_avx2(src, n, dst); return; }
#endif
    pack_low_portable(src, n, dst);
}

uint32_t rank(const BitMatrix& A) {
    BitMatrix B = A;
    return echelon(B).rank;
}

BitMatrix kernel_basis(const BitMatrix& R, const std::vector<uint32_t>& pivots) {
    const uint32_t n = R.cols();
    const uint32_t k = static_cast<uint32_t>(pivots.size());
    ENGINE_CHECK(k <= n && k <= R.rows());
    std::vector<char> is_piv(n, 0);
    for (uint32_t c : pivots) { ENGINE_CHECK(c < n); is_piv[c] = 1; }
    BitMatrix K(n - k, n);
    uint32_t t = 0;
    for (uint32_t jf = 0; jf < n; ++jf) {
        if (is_piv[jf]) continue;
        K.set(t, jf, true);
        const uint32_t w = jf >> 6, s = jf & 63;
        for (uint32_t i = 0; i < k; ++i)
            if ((R.row(i)[w] >> s) & 1u) K.set(t, pivots[i], true);
        ++t;
    }
    return K;
}

BitMatrix kernel(const BitMatrix& A) {
    BitMatrix R = A;
    const BitRref res = rref(R);
    return kernel_basis(R, res.pivots);
}

BitMatrix multiply_m4rm(const BitMatrix& A, const BitMatrix& B) {
    ENGINE_CHECK_MSG(A.cols() == B.rows(), "binary::multiply_m4rm: inner dimensions differ");
    const uint32_t m = A.rows(), k = A.cols(), n = B.cols();
    BitMatrix C(m, n);
    if (m == 0 || k == 0 || n == 0) return C;
    const uint32_t W = C.words_per_row();
    const RowOps ops = ops_for(active_row_path());
    std::vector<uint64_t> T(static_cast<size_t>(256) * W, 0);
    for (uint32_t kb = 0; kb < k; kb += 8) {
        const uint32_t nb = std::min<uint32_t>(8, k - kb);
        const uint32_t N = 1u << nb;
        for (uint32_t g = 1; g < N; ++g) {
            const uint32_t low = static_cast<uint32_t>(__builtin_ctz(g));
            ops.xor3(T.data() + static_cast<size_t>(g) * W, T.data() + static_cast<size_t>(g ^ (1u << low)) * W,
                     B.row(kb + low), 0, W);
        }
        const uint32_t wa = kb >> 6, sa = kb & 63;   // kb is a multiple of 8: the slice is one byte of a word
        const uint64_t mask = (uint64_t(1) << nb) - 1u;
        for (uint32_t i = 0; i < m; ++i) {
            const uint32_t idx = static_cast<uint32_t>((A.row(i)[wa] >> sa) & mask);
            if (idx != 0) ops.xor2(C.row(i), T.data() + static_cast<size_t>(idx) * W, 0, W);
        }
    }
    return C;
}

BitMatrix multiply_naive(const BitMatrix& A, const BitMatrix& B) {
    ENGINE_CHECK_MSG(A.cols() == B.rows(), "binary::multiply_naive: inner dimensions differ");
    BitMatrix C(A.rows(), B.cols());
    const uint32_t W = C.words_per_row();
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t t = 0; t < A.cols(); ++t)
            if (A.get(i, t)) xor2_portable(C.row(i), B.row(t), 0, W);
    return C;
}

BitRref rref_naive(BitMatrix& A) {
    BitRref res;
    const uint32_t m = A.rows(), n = A.cols(), W = A.words_per_row();
    uint32_t r = 0;
    for (uint32_t c = 0; c < n && r < m; ++c) {
        uint32_t p = m;
        for (uint32_t i = r; i < m; ++i) if (A.get(i, c)) { p = i; break; }
        if (p == m) continue;
        A.swap_rows(p, r);
        for (uint32_t i = 0; i < m; ++i)
            if (i != r && A.get(i, c)) xor2_portable(A.row(i), A.row(r), 0, W);
        res.pivots.push_back(c);
        ++r;
    }
    res.rank = r;
    return res;
}

// ---------------------------------------------------------------------------
// Row paths
// ---------------------------------------------------------------------------
bool row_path_available(RowPath p) {
    switch (p) {
        case RowPath::Auto:
        case RowPath::Portable: return true;
        case RowPath::Avx2: return cpu_has_avx2();
        case RowPath::Avx512: return cpu_has_avx512();
    }
    return false;
}

void set_row_path(RowPath p) {
    ENGINE_CHECK_MSG(row_path_available(p), "binary::set_row_path: path not available on this CPU / build");
    g_forced_path.store(static_cast<uint8_t>(p), std::memory_order_relaxed);
}

RowPath active_row_path() {
    const RowPath f = static_cast<RowPath>(g_forced_path.load(std::memory_order_relaxed));
    if (f != RowPath::Auto) return f;
    return kernels::isa_auto(cpu_has_avx2(), cpu_has_avx512());   // (S24 CL-5) under the ENGINE_KERNEL ceiling
}

const char* row_path_name(RowPath p) { return cpu::isa_path_name(p); }

} // namespace binary
} // namespace engine
