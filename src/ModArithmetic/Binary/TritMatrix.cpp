// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "TritMatrix.h"
#include "SmithPar.h"

#include <algorithm>
#include <atomic>
#include <type_traits>

#include "BitMatrix.h"   // RowPath / active_row_path: one row-path switch for the F_2 and F_3 rungs
#include "../ModArithmetic/EngineCheck.h"
#include "../ModArithmetic/Fnv.h"

// See TritMatrix.h.  Origin: the L23 master-plan prototype measurement (Sec. 2.3, "bit-sliced F_3,
// two bitplanes, 4.2-4.7x the engine at 4096^2"), written as a library on BIN-1's row layout.

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  include <immintrin.h>
#  define TER_HAVE_X86_TARGETS 1
#  define TER_AVX2_FN     __attribute__((target("avx2")))
#  define TER_AVX512_FN   __attribute__((target("avx512f")))
#  define TER_AVX512BW_FN __attribute__((target("avx512f,avx512bw")))
#else
#  define TER_HAVE_X86_TARGETS 0
#endif

namespace engine {
namespace binary {

namespace {

// ---------------------------------------------------------------------------
// Plane bodies.  d = [d1 | d2], s = [s1 | s2], each plane W words; operate on words [w0, W).
// Sub = false: d += s;  Sub = true: d -= s (= d + (s2, s1)).
// ---------------------------------------------------------------------------
template<bool Sub>
void addrow_portable(uint64_t* d, const uint64_t* s, uint32_t w0, uint32_t W) {
    uint64_t* d1 = d;
    uint64_t* d2 = d + W;
    const uint64_t* y1p = Sub ? s + W : s;
    const uint64_t* y2p = Sub ? s : s + W;
    for (uint32_t k = w0; k < W; ++k) {
        const uint64_t x1 = d1[k], x2 = d2[k], y1 = y1p[k], y2 = y2p[k];
        const uint64_t t = (x1 | y2) ^ (x2 | y1);
        d1[k] = (x2 | y2) ^ t;
        d2[k] = (x1 | y1) ^ t;
    }
}

#if TER_HAVE_X86_TARGETS
template<bool Sub>
TER_AVX2_FN void addrow_avx2(uint64_t* d, const uint64_t* s, uint32_t w0, uint32_t W) {
    uint64_t* d1 = d;
    uint64_t* d2 = d + W;
    const uint64_t* y1p = Sub ? s + W : s;
    const uint64_t* y2p = Sub ? s : s + W;
    uint32_t k = w0;
    for (; (k & 3u) != 0 && k < W; ++k) {
        const uint64_t x1 = d1[k], x2 = d2[k], y1 = y1p[k], y2 = y2p[k];
        const uint64_t t = (x1 | y2) ^ (x2 | y1);
        d1[k] = (x2 | y2) ^ t;
        d2[k] = (x1 | y1) ^ t;
    }
    for (; k + 4 <= W; k += 4) {
        const __m256i x1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d1 + k));
        const __m256i x2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d2 + k));
        const __m256i y1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y1p + k));
        const __m256i y2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y2p + k));
        const __m256i t = _mm256_xor_si256(_mm256_or_si256(x1, y2), _mm256_or_si256(x2, y1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d1 + k), _mm256_xor_si256(_mm256_or_si256(x2, y2), t));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d2 + k), _mm256_xor_si256(_mm256_or_si256(x1, y1), t));
    }
    for (; k < W; ++k) {
        const uint64_t x1 = d1[k], x2 = d2[k], y1 = y1p[k], y2 = y2p[k];
        const uint64_t t = (x1 | y2) ^ (x2 | y1);
        d1[k] = (x2 | y2) ^ t;
        d2[k] = (x1 | y1) ^ t;
    }
}
template<bool Sub>
TER_AVX512_FN void addrow_avx512(uint64_t* d, const uint64_t* s, uint32_t w0, uint32_t W) {
    uint64_t* d1 = d;
    uint64_t* d2 = d + W;
    const uint64_t* y1p = Sub ? s + W : s;
    const uint64_t* y2p = Sub ? s : s + W;
    uint32_t k = w0;
    for (; (k & 7u) != 0 && k < W; ++k) {
        const uint64_t x1 = d1[k], x2 = d2[k], y1 = y1p[k], y2 = y2p[k];
        const uint64_t t = (x1 | y2) ^ (x2 | y1);
        d1[k] = (x2 | y2) ^ t;
        d2[k] = (x1 | y1) ^ t;
    }
    for (; k + 8 <= W; k += 8) {
        const __m512i x1 = _mm512_loadu_si512(d1 + k);
        const __m512i x2 = _mm512_loadu_si512(d2 + k);
        const __m512i y1 = _mm512_loadu_si512(y1p + k);
        const __m512i y2 = _mm512_loadu_si512(y2p + k);
        // t = (x1|y2) ^ (x2|y1); z1 = (x2|y2) ^ t; z2 = (x1|y1) ^ t
        const __m512i t = _mm512_xor_si512(_mm512_or_si512(x1, y2), _mm512_or_si512(x2, y1));
        _mm512_storeu_si512(d1 + k, _mm512_xor_si512(_mm512_or_si512(x2, y2), t));
        _mm512_storeu_si512(d2 + k, _mm512_xor_si512(_mm512_or_si512(x1, y1), t));
    }
    for (; k < W; ++k) {
        const uint64_t x1 = d1[k], x2 = d2[k], y1 = y1p[k], y2 = y2p[k];
        const uint64_t t = (x1 | y2) ^ (x2 | y1);
        d1[k] = (x2 | y2) ^ t;
        d2[k] = (x1 | y1) ^ t;
    }
}
#endif

// ---------------------------------------------------------------------------
// Byte bodies on canonical F_3 entries: s = a + b (or a + 3 - b, or 3 - a) in [0, 5], then
// s = min(s, s - 3) as unsigned bytes (s - 3 wraps above 250 when s < 3).
// ---------------------------------------------------------------------------
inline uint8_t f3_red(unsigned s) {
    const uint8_t a = static_cast<uint8_t>(s), b = static_cast<uint8_t>(s - 3u);
    return a < b ? a : b;
}
void add_u8_portable(uint8_t* d, const uint8_t* s, size_t n) { for (size_t i = 0; i < n; ++i) d[i] = f3_red(unsigned(d[i]) + s[i]); }
void sub_u8_portable(uint8_t* d, const uint8_t* s, size_t n) { for (size_t i = 0; i < n; ++i) d[i] = f3_red(unsigned(d[i]) + 3u - s[i]); }
void neg_u8_portable(uint8_t* d, size_t n) { for (size_t i = 0; i < n; ++i) d[i] = f3_red(3u - d[i]); }

#if TER_HAVE_X86_TARGETS
TER_AVX2_FN void add_u8_avx2(uint8_t* d, const uint8_t* s, size_t n) {
    const __m256i three = _mm256_set1_epi8(3);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
        const __m256i t = _mm256_add_epi8(a, b);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + i), _mm256_min_epu8(t, _mm256_sub_epi8(t, three)));
    }
    add_u8_portable(d + i, s + i, n - i);
}
TER_AVX2_FN void sub_u8_avx2(uint8_t* d, const uint8_t* s, size_t n) {
    const __m256i three = _mm256_set1_epi8(3);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
        const __m256i t = _mm256_sub_epi8(_mm256_add_epi8(a, three), b);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + i), _mm256_min_epu8(t, _mm256_sub_epi8(t, three)));
    }
    sub_u8_portable(d + i, s + i, n - i);
}
TER_AVX2_FN void neg_u8_avx2(uint8_t* d, size_t n) {
    const __m256i three = _mm256_set1_epi8(3);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(d + i));
        const __m256i t = _mm256_sub_epi8(three, a);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + i), _mm256_min_epu8(t, _mm256_sub_epi8(t, three)));
    }
    neg_u8_portable(d + i, n - i);
}
TER_AVX512BW_FN void add_u8_avx512(uint8_t* d, const uint8_t* s, size_t n) {
    const __m512i three = _mm512_set1_epi8(3);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512(d + i);
        const __m512i b = _mm512_loadu_si512(s + i);
        const __m512i t = _mm512_add_epi8(a, b);
        _mm512_storeu_si512(d + i, _mm512_min_epu8(t, _mm512_sub_epi8(t, three)));
    }
    add_u8_portable(d + i, s + i, n - i);
}
TER_AVX512BW_FN void sub_u8_avx512(uint8_t* d, const uint8_t* s, size_t n) {
    const __m512i three = _mm512_set1_epi8(3);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512(d + i);
        const __m512i b = _mm512_loadu_si512(s + i);
        const __m512i t = _mm512_sub_epi8(_mm512_add_epi8(a, three), b);
        _mm512_storeu_si512(d + i, _mm512_min_epu8(t, _mm512_sub_epi8(t, three)));
    }
    sub_u8_portable(d + i, s + i, n - i);
}
TER_AVX512BW_FN void neg_u8_avx512(uint8_t* d, size_t n) {
    const __m512i three = _mm512_set1_epi8(3);
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        const __m512i a = _mm512_loadu_si512(d + i);
        const __m512i t = _mm512_sub_epi8(three, a);
        _mm512_storeu_si512(d + i, _mm512_min_epu8(t, _mm512_sub_epi8(t, three)));
    }
    neg_u8_portable(d + i, n - i);
}
#endif

bool cpu_has_avx512bw() {
#if TER_HAVE_X86_TARGETS
    return cpu::avx512bw();   // (S24 CL-5) CpuFeatures.h
#else
    return false;
#endif
}

struct TerOps {
    void (*add)(uint64_t*, const uint64_t*, uint32_t, uint32_t);
    void (*sub)(uint64_t*, const uint64_t*, uint32_t, uint32_t);
    void (*add8)(uint8_t*, const uint8_t*, size_t);
    void (*sub8)(uint8_t*, const uint8_t*, size_t);
    void (*neg8)(uint8_t*, size_t);
    const char* name;
};

TerOps ter_ops() {
    const RowPath p = active_row_path();
#if TER_HAVE_X86_TARGETS
    if (p == RowPath::Avx512) {
        if (cpu_has_avx512bw())
            return TerOps{&addrow_avx512<false>, &addrow_avx512<true>, &add_u8_avx512, &sub_u8_avx512, &neg_u8_avx512, "avx512(planes)+avx512bw(bytes)"};
        return TerOps{&addrow_avx512<false>, &addrow_avx512<true>, &add_u8_avx2, &sub_u8_avx2, &neg_u8_avx2, "avx512(planes)+avx2(bytes)"};
    }
    if (p == RowPath::Avx2)
        return TerOps{&addrow_avx2<false>, &addrow_avx2<true>, &add_u8_avx2, &sub_u8_avx2, &neg_u8_avx2, "avx2"};
#endif
    (void)p;
    return TerOps{&addrow_portable<false>, &addrow_portable<true>, &add_u8_portable, &sub_u8_portable, &neg_u8_portable, "portable"};
}

// ---------------------------------------------------------------------------
// Gauss--Jordan (full = RREF; !full = echelon only).  Invariant: before the step for column c,
// every row at index >= r is zero in every column < c, so the pivot row is zero left of c and a
// row operation may start at the pivot's word.
// ---------------------------------------------------------------------------
TritRref gauss(TritMatrix& A, bool full) {
    TritRref res;
    const uint32_t m = A.rows(), n = A.cols(), W = A.words_per_plane();
    const TerOps ops = ter_ops();
    uint32_t r = 0;
    for (uint32_t c = 0; c < n && r < m; ++c) {
        const uint32_t w = c >> 6;
        const uint64_t bit = uint64_t(1) << (c & 63);
        uint32_t p = m;
        for (uint32_t i = r; i < m; ++i) {
            const uint64_t* ri = A.row(i);
            if ((ri[w] | ri[W + w]) & bit) { p = i; break; }
        }
        if (p == m) continue;
        if (p != r) A.swap_rows(p, r);
        if (full && (A.row(r)[W + w] & bit)) A.negate_row(r, w);   // normalise the pivot to 1
        const uint64_t* pr = A.row(r);
        const bool piv_one = (pr[w] & bit) != 0;
        for (uint32_t i = full ? 0u : r + 1; i < m; ++i) {
            if (i == r) continue;
            uint64_t* ri = A.row(i);
            const bool e1 = (ri[w] & bit) != 0, e2 = (ri[W + w] & bit) != 0;
            if (!e1 && !e2) continue;
            // row_i -= (e / pv) * pivot: e == pv -> subtract the pivot row, else add it.
            if (e1 == piv_one) ops.sub(ri, pr, w, W);
            else ops.add(ri, pr, w, W);
        }
        res.pivots.push_back(c);
        ++r;
    }
    res.rank = r;
    return res;
}

// ---------------------------------------------------------------------------
// (slot-24 PAR-F2) The lazy blocked Gauss (echelon only; see TritMatrix.h).  The serial gauss(A, false) applies pivot t
// to every row below right after choosing it; here the rows below receive the pivots of a block (8 .. 64 pivots) only
// when they are needed: a row that becomes a pivot row first, every other row at the block's end (a flush: one region,
// each row independently), each row exactly the serial operations in pivot order (the same test of the pivot-column
// entry, the same add / sub from the pivot's word).  Pivot rows are never touched after they are chosen (echelon only),
// so pivot t of the block is row R0 + t.  Block boundaries are free: a pivot's operations do not depend on the grouping.
//
// THE SEARCH READS A TRANSPOSED WINDOW (as binary::echelon_lazy): 512 columns of the reduced rows below r, two bitsets per
// column (plane one, plane two), built after a flush and updated by every pivot: rows p and r swap bits, and every window
// column j > c where the pivot row is non-zero receives, on the rows i > r with e_i = entry (i, c) != 0, the serial
// change -(e_i / pv) P_j = -(e_i pv P_j) (1 / pv = pv over F_3) as one plane-wise trit addition.  So the search for column
// c is "the first row >= r set in either plane": the serial search's answer in m / 64 words.
// ---------------------------------------------------------------------------
constexpr uint32_t kTerWinWords = 8;   // the transposed search window: 512 columns

TritRref gauss_lazy(TritMatrix& A) {
    TritRref res;
    const uint32_t m = A.rows(), n = A.cols(), W = A.words_per_plane();
    if (m == 0 || n == 0) return res;
    smithpar::count_f2_lazy_call();
    const TerOps ops = ter_ops();
    const uint32_t T = smithpar::threads();
    const uint64_t mw = smithpar::min_work();
    // Block size: the block's pivot rows (2 W words each) stay in L2 while a row receives them (never a value).
    const uint32_t kBlk = std::max<uint32_t>(8u, std::min<uint32_t>(64u, (1u << 18) / (16u * W)));
    std::vector<uint8_t> lvl(m, 0);
    std::vector<uint32_t> pw, pb;   // the block's pivots: word, bit
    std::vector<uint8_t> pone;      // pivot entry == 1 (else 2)
    pw.reserve(kBlk); pb.reserve(kBlk); pone.reserve(kBlk);
    uint32_t R0 = 0;                // the block's first pivot row

    auto bring = [&](uint32_t i, uint32_t to) {
        uint8_t& li = lvl[i];
        if (li >= to) return;
        uint64_t* ri = A.row(i);
        for (uint32_t t = li; t < to; ++t) {
            const uint32_t w = pw[t];
            const uint64_t bit = uint64_t(1) << pb[t];
            const bool e1 = (ri[w] & bit) != 0, e2 = (ri[W + w] & bit) != 0;
            if (!e1 && !e2) continue;
            const uint64_t* pr = A.row(R0 + t);
            if (e1 == (pone[t] != 0)) ops.sub(ri, pr, w, W);
            else ops.add(ri, pr, w, W);
        }
        li = static_cast<uint8_t>(to);
    };
    auto flush = [&](uint32_t r) {
        const uint32_t np = static_cast<uint32_t>(pw.size());
        if (np > 0 && r < m) {
            smithpar::PhaseTimer tfl(smithpar::Phase::EchFlush);
            const uint32_t rows = m - r;
            smithpar::for_rows(rows, static_cast<uint64_t>(rows) * np * (W - pw[0]) * 2u, [&](uint32_t i0, uint32_t i1) {
                for (uint32_t i = r + i0; i < r + i1; ++i) { bring(i, np); lvl[i] = 0; }
            });
        }
        pw.clear(); pb.clear(); pone.clear();
        R0 = r;
    };

    // The window: words [ww0, wwe); column j: plane one at c1(j), plane two at c2(j) (MW words each).
    const uint32_t MW = (m + 63u) / 64u;
    std::vector<uint64_t> cb(static_cast<size_t>(2u * 64u * kTerWinWords) * MW, 0);
    auto c1p = [&](uint32_t j) -> uint64_t* { return cb.data() + static_cast<size_t>(2u * j) * MW; };
    auto c2p = [&](uint32_t j) -> uint64_t* { return cb.data() + static_cast<size_t>(2u * j + 1u) * MW; };
    uint32_t ww0 = 0, wwe = 0;
    auto build_window = [&](uint32_t wa, uint32_t r) {   // right after a flush: rows [r, m) are current
        smithpar::PhaseTimer tw(smithpar::Phase::EchTable);
        ww0 = wa;
        wwe = std::min<uint32_t>(W, wa + kTerWinWords);
        const uint32_t nw = wwe - ww0, q0 = r >> 6, nq = MW - q0;
        std::fill(cb.begin(), cb.end(), uint64_t(0));
        smithpar::for_rows(nq, static_cast<uint64_t>(nq) * 64u * nw * 2u, [&](uint32_t a, uint32_t b) {
            for (uint32_t q = q0 + a; q < q0 + b; ++q) {
                const uint32_t i0 = std::max<uint32_t>(r, q * 64u), i1 = std::min<uint32_t>(m, q * 64u + 64u);
                for (uint32_t i = i0; i < i1; ++i) {
                    const uint64_t* ri = A.row(i);
                    const uint64_t rb = uint64_t(1) << (i & 63u);
                    for (uint32_t k = 0; k < nw; ++k) {
                        for (uint64_t x = ri[ww0 + k]; x != 0; x &= x - 1) c1p(k * 64u + static_cast<uint32_t>(__builtin_ctzll(x)))[q] |= rb;
                        for (uint64_t x = ri[W + ww0 + k]; x != 0; x &= x - 1) c2p(k * 64u + static_cast<uint32_t>(__builtin_ctzll(x)))[q] |= rb;
                    }
                }
            }
        });
    };

    uint32_t r = 0;
    std::vector<uint32_t> js;
    std::vector<uint8_t> jk;
    for (uint32_t c = 0; c < n && r < m; ++c) {
        const uint32_t w = c >> 6, sb = c & 63;
        const uint64_t bit = uint64_t(1) << sb;
        if (wwe == 0 || w >= wwe) {
            flush(r);
            build_window(w, r);
        } else if (pw.size() >= kBlk) {
            flush(r);
        }
        const uint32_t np = static_cast<uint32_t>(pw.size());
        const uint32_t jc = c - ww0 * 64u;
        uint32_t p = m;
        {
            smithpar::PhaseTimer tsr(smithpar::Phase::EchSearch);
            const uint64_t* x1 = c1p(jc);
            const uint64_t* x2 = c2p(jc);
            uint32_t q = r >> 6;
            uint64_t x = (x1[q] | x2[q]) & (~uint64_t(0) << (r & 63u));
            while (x == 0 && ++q < MW) x = x1[q] | x2[q];
            if (x != 0) p = q * 64u + static_cast<uint32_t>(__builtin_ctzll(x));
        }
        if (p >= m) continue;
        bring(p, np);
        if (p != r) {
            A.swap_rows(p, r);
            std::swap(lvl[p], lvl[r]);
        }
        const uint64_t* pr = A.row(r);
        ENGINE_CHECK_MSG(((pr[w] | pr[W + w]) & bit) != 0, "binary::echelon_lazy (F_3): the transposed window disagrees with the pivot row");
        const bool po = (pr[w] & bit) != 0;
        {
            smithpar::PhaseTimer tsr(smithpar::Phase::EchSearch);
            const uint32_t ncol = (wwe - ww0) * 64u;
            if (p != r) {
                const uint32_t qp = p >> 6, qr = r >> 6;
                const uint64_t bp = uint64_t(1) << (p & 63u), br = uint64_t(1) << (r & 63u);
                for (uint32_t j = 0; j < 2u * ncol; ++j) {
                    uint64_t* cj = cb.data() + static_cast<size_t>(j) * MW;
                    const bool vp = (cj[qp] & bp) != 0, vr = (cj[qr] & br) != 0;
                    if (vp != vr) { cj[qp] ^= bp; cj[qr] ^= br; }
                }
            }
            // columns j > c where the pivot row is non-zero; k = pv * P_j (mod 3): 1 when equal, else 2
            js.clear();
            jk.clear();
            for (uint32_t j = jc + 1; j < ncol; ++j) {
                const uint32_t ww = ww0 + (j >> 6);
                const uint64_t jb = uint64_t(1) << (j & 63u);
                const bool p1 = (pr[ww] & jb) != 0, p2 = (pr[W + ww] & jb) != 0;
                if (!p1 && !p2) continue;
                js.push_back(j);
                jk.push_back(p1 == po ? 1 : 2);
            }
            const uint32_t qa = (r + 1) >> 6;
            const uint64_t head = ((r + 1) & 63u) ? (~uint64_t(0) << ((r + 1) & 63u)) : ~uint64_t(0);
            const uint64_t* e1 = c1p(jc);
            const uint64_t* e2 = c2p(jc);
            auto upd = [&](uint32_t a, uint32_t b) {
                for (uint32_t t = a; t < b; ++t) {
                    // delta for row i: -(e_i k): 2 where e_i == k, 1 where e_i == 3 - k
                    uint64_t* x1 = c1p(js[t]);
                    uint64_t* x2 = c2p(js[t]);
                    const bool k1 = jk[t] == 1;
                    for (uint32_t q = qa; q < MW; ++q) {
                        const uint64_t mk = (q == qa) ? head : ~uint64_t(0);
                        const uint64_t a1 = e1[q] & mk, a2 = e2[q] & mk;
                        const uint64_t y2 = k1 ? a1 : a2;   // rows receiving + 2
                        const uint64_t y1 = k1 ? a2 : a1;   // rows receiving + 1
                        if ((y1 | y2) == 0) continue;
                        const uint64_t v1 = x1[q], v2 = x2[q];
                        const uint64_t tt = (v1 | y2) ^ (v2 | y1);
                        x1[q] = (v2 | y2) ^ tt;
                        x2[q] = (v1 | y1) ^ tt;
                    }
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
        pw.push_back(w);
        pb.push_back(sb);
        pone.push_back(po ? 1 : 0);
        lvl[r] = static_cast<uint8_t>(np + 1);   // (a pivot row is never read through lvl again)
        res.pivots.push_back(c);
        ++r;
    }
    flush(r);
    res.rank = r;
    return res;
}

} // namespace

// ---------------------------------------------------------------------------
// TritMatrix
// ---------------------------------------------------------------------------
TritMatrix::TritMatrix(uint32_t rows, uint32_t cols)
    : rows_(rows), cols_(cols), W_(((cols + 511u) / 512u) * 8u),
      d_(static_cast<size_t>(rows) * 2 * (((cols + 511u) / 512u) * 8u), 0) {}

TritMatrix TritMatrix::identity(uint32_t n) {
    TritMatrix I(n, n);
    for (uint32_t i = 0; i < n; ++i) I.set(i, i, 1);
    return I;
}

void TritMatrix::set(uint32_t i, uint32_t j, uint32_t v) {
    ENGINE_CHECK_MSG(i < rows_ && j < cols_ && v < 3u, "TritMatrix::set: index or value out of range");
    uint64_t* r = row(i);
    const uint32_t w = j >> 6;
    const uint64_t m = uint64_t(1) << (j & 63);
    r[w] &= ~m;
    r[W_ + w] &= ~m;
    if (v == 1) r[w] |= m;
    else if (v == 2) r[W_ + w] |= m;
}

void TritMatrix::swap_rows(uint32_t a, uint32_t b) {
    if (a == b) return;
    std::swap_ranges(row(a), row(a) + 2 * W_, row(b));
}

void TritMatrix::negate_row(uint32_t i, uint32_t w0) {
    uint64_t* r = row(i);
    std::swap_ranges(r + w0, r + W_, r + W_ + w0);
}

uint64_t TritMatrix::fnv() const {
    uint64_t h = FnvU64(kFnvOffset64, rows_);
    h = FnvU64(h, cols_);
    for (uint32_t i = 0; i < rows_; ++i)
        for (uint32_t j = 0; j < cols_; ++j) h = FnvU64(h, get(i, j));
    return h;
}

template<class Elt>
TritMatrix TritMatrix::from_dense(DenseView<const Elt> src) {
    const uint32_t m = src.rows(), n = src.cols();
    TritMatrix T(m, n);
    const auto s = src.strides();
    Elt mx = 0;
    for (uint32_t i = 0; i < m; ++i) {
        const Elt* ri = s.base + i * s.rs;
        uint64_t* t1 = T.row(i);
        uint64_t* t2 = t1 + T.W_;
        for (uint32_t j = 0; j < n; ++j) {
            const Elt v = ri[static_cast<size_t>(j) * s.cs];
            mx = v > mx ? v : mx;
            t1[j >> 6] |= static_cast<uint64_t>(v == Elt(1)) << (j & 63);
            t2[j >> 6] |= static_cast<uint64_t>(v == Elt(2)) << (j & 63);
        }
    }
    ENGINE_CHECK_MSG(mx < Elt(3), "TritMatrix::from_dense: entries must be canonical residues mod 3");
    return T;
}

template<class Elt>
void TritMatrix::to_dense(DenseView<Elt> dst) const {
    ENGINE_CHECK(dst.rows() == rows_ && dst.cols() == cols_);
    const auto s = dst.strides();
    for (uint32_t i = 0; i < rows_; ++i) {
        Elt* di = s.base + i * s.rs;
        const uint64_t* t1 = row(i);
        const uint64_t* t2 = t1 + W_;
        for (uint32_t j = 0; j < cols_; ++j) {
            const uint64_t a = (t1[j >> 6] >> (j & 63)) & 1u, b = (t2[j >> 6] >> (j & 63)) & 1u;
            di[static_cast<size_t>(j) * s.cs] = static_cast<Elt>(a | (b << 1));
        }
    }
}

template TritMatrix TritMatrix::from_dense<uint32_t>(DenseView<const uint32_t>);
template TritMatrix TritMatrix::from_dense<uint64_t>(DenseView<const uint64_t>);
template void TritMatrix::to_dense<uint32_t>(DenseView<uint32_t>) const;
template void TritMatrix::to_dense<uint64_t>(DenseView<uint64_t>) const;

TritMatrix TritMatrix::from_u8(const uint8_t* a, uint32_t rows, uint32_t cols, size_t ld) {
    TritMatrix T(rows, cols);
    uint8_t mx = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        const uint8_t* ri = a + static_cast<size_t>(i) * ld;
        uint64_t* t1 = T.row(i);
        uint64_t* t2 = t1 + T.W_;
        for (uint32_t j0 = 0; j0 < cols; j0 += 64) {
            const uint32_t e = std::min<uint32_t>(64, cols - j0);
            uint64_t w1 = 0, w2 = 0;
            for (uint32_t q = 0; q < e; ++q) {
                const uint8_t v = ri[j0 + q];
                mx = v > mx ? v : mx;
                w1 |= static_cast<uint64_t>(v == 1) << q;
                w2 |= static_cast<uint64_t>(v == 2) << q;
            }
            t1[j0 >> 6] = w1;
            t2[j0 >> 6] = w2;
        }
    }
    ENGINE_CHECK_MSG(mx < 3, "TritMatrix::from_u8: entries must be canonical residues mod 3");
    return T;
}

void TritMatrix::to_u8(uint8_t* a, size_t ld) const {
    for (uint32_t i = 0; i < rows_; ++i) {
        uint8_t* di = a + static_cast<size_t>(i) * ld;
        const uint64_t* t1 = row(i);
        const uint64_t* t2 = t1 + W_;
        for (uint32_t j = 0; j < cols_; ++j)
            di[j] = static_cast<uint8_t>(((t1[j >> 6] >> (j & 63)) & 1u) | (((t2[j >> 6] >> (j & 63)) & 1u) << 1));
    }
}

// ---------------------------------------------------------------------------
// Elimination, kernel, products
// ---------------------------------------------------------------------------
TritRref rref(TritMatrix& A) { return gauss(A, true); }
TritRref echelon(TritMatrix& A) { return gauss(A, false); }
TritRref echelon_lazy(TritMatrix& A) { return gauss_lazy(A); }

uint32_t rank(const TritMatrix& A) {
    TritMatrix B = A;
    return echelon(B).rank;
}

TritMatrix kernel_basis(const TritMatrix& R, const std::vector<uint32_t>& pivots) {
    const uint32_t n = R.cols();
    const uint32_t k = static_cast<uint32_t>(pivots.size());
    ENGINE_CHECK(k <= n && k <= R.rows());
    std::vector<char> is_piv(n, 0);
    for (uint32_t c : pivots) { ENGINE_CHECK(c < n); is_piv[c] = 1; }
    TritMatrix K(n - k, n);
    uint32_t t = 0;
    for (uint32_t jf = 0; jf < n; ++jf) {
        if (is_piv[jf]) continue;
        K.set(t, jf, 1);
        for (uint32_t i = 0; i < k; ++i) {
            const uint32_t v = R.get(i, jf);
            if (v) K.set(t, pivots[i], 3u - v);
        }
        ++t;
    }
    return K;
}

TritMatrix kernel(const TritMatrix& A) {
    TritMatrix R = A;
    const TritRref res = rref(R);
    return kernel_basis(R, res.pivots);
}

TritMatrix multiply(const TritMatrix& A, const TritMatrix& B) {
    ENGINE_CHECK_MSG(A.cols() == B.rows(), "binary::multiply (F_3): inner dimensions differ");
    TritMatrix C(A.rows(), B.cols());
    const uint32_t W = C.words_per_plane();
    const TerOps ops = ter_ops();
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t t = 0; t < A.cols(); ++t) {
            const uint32_t v = A.get(i, t);
            if (v == 1) ops.add(C.row(i), B.row(t), 0, W);
            else if (v == 2) ops.sub(C.row(i), B.row(t), 0, W);
        }
    return C;
}

TritMatrix multiply_naive(const TritMatrix& A, const TritMatrix& B) {
    ENGINE_CHECK_MSG(A.cols() == B.rows(), "binary::multiply_naive (F_3): inner dimensions differ");
    TritMatrix C(A.rows(), B.cols());
    for (uint32_t i = 0; i < A.rows(); ++i)
        for (uint32_t j = 0; j < B.cols(); ++j) {
            uint32_t s = 0;
            for (uint32_t t = 0; t < A.cols(); ++t) s += A.get(i, t) * B.get(t, j);
            C.set(i, j, s % 3u);
        }
    return C;
}

TritRref rref_naive(TritMatrix& A) {
    TritRref res;
    const uint32_t m = A.rows(), n = A.cols();
    uint32_t r = 0;
    for (uint32_t c = 0; c < n && r < m; ++c) {
        uint32_t p = m;
        for (uint32_t i = r; i < m; ++i) if (A.get(i, c)) { p = i; break; }
        if (p == m) continue;
        A.swap_rows(p, r);
        const uint32_t inv = A.get(r, c);   // 1^-1 = 1, 2^-1 = 2
        for (uint32_t j = 0; j < n; ++j) A.set(r, j, (A.get(r, j) * inv) % 3u);
        for (uint32_t i = 0; i < m; ++i) {
            if (i == r) continue;
            const uint32_t f = A.get(i, c);
            if (!f) continue;
            for (uint32_t j = 0; j < n; ++j) A.set(i, j, (A.get(i, j) + 3u * 3u - f * A.get(r, j)) % 3u);
        }
        res.pivots.push_back(c);
        ++r;
    }
    res.rank = r;
    return res;
}

// ---------------------------------------------------------------------------
// Byte kernels
// ---------------------------------------------------------------------------
void f3_add_u8(uint8_t* dst, const uint8_t* src, size_t n) { ter_ops().add8(dst, src, n); }
void f3_sub_u8(uint8_t* dst, const uint8_t* src, size_t n) { ter_ops().sub8(dst, src, n); }
void f3_neg_u8(uint8_t* dst, size_t n) { ter_ops().neg8(dst, n); }

const char* ternary_row_body_name() { return ter_ops().name; }

} // namespace binary
} // namespace engine
