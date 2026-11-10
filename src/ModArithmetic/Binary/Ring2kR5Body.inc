// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// The R5 kernel bodies (slot-23 R5V), included by Ring2kR5.cpp once per kernel path (portable / AVX2 /
// AVX-512), each time inside its own namespace R5_NS and, for AVX2 / AVX-512, under `#pragma GCC target`
// (so every template below is compiled for that ISA).  The including file defines:
//   R5_NS            the namespace;
//   R5_VB            the vector width in bytes (16 / 32 / 64);
//   R5_MUL(a, b)     the lane-wise (a mod 2^32)(b mod 2^32) as a 64-bit lane (the operands are < 2^28 here);
//   R5_ROWMASKS      1 if R5_LTMASK(a, b) (bit l = a_l < b_l, unsigned) and R5_EQMASK(a, b) (bit l = a_l == b_l)
//                    exist; then the row kernels propagate carries by masks, else they run the scalar chain.
// and, before the first inclusion, the constexpr helpers r5_limb_bits / r5_chunk (Ring2kR5.cpp).
//
// THE LIMB SCHEME.  An element x mod 2^(64 N) is cut into NL limbs of B bits (B NL >= 64 N; the top limb
// holds the bits >= 64 N as zeros).  A truncated product mod 2^(B NL) needs the limb products f_s1 u_s2
// with s1 + s2 < NL only; each is < 2^(2B) and is added, uncarried, into a 64-bit lane acc[s1 + s2].  Per
// rank-one update a lane gains at most NL such products, so TC = r5_chunk(N) updates fit (TC NL 2^(2B) <
// 2^64 - 2^(2B)); after TC updates the lanes are carried (acc[s+1] += acc[s] >> B, acc[s] &= 2^B - 1, the
// carry out of the top limb dropped: mod 2^(B NL)).  The result C - S is formed limb-wise with a borrow
// chain and unpacked; the bits >= 64 N are dropped, so everything is mod 2^(64 N) exactly -- the same
// residue the scalar Traits<WordN<N>> computes, so the two are bit-identical.

namespace R5_NS {

constexpr uint32_t VW = R5_VB / 8;   // 64-bit lanes per vector
typedef uint64_t V __attribute__((vector_size(R5_VB)));

template<uint32_t N> struct Cfg {
    static constexpr uint32_t B = r5_limb_bits(N);
    static constexpr uint32_t NL = (64 * N + B - 1) / B;
    static constexpr uint32_t TC = r5_chunk(N);
    static constexpr uint64_t M = (uint64_t(1) << B) - 1;
    static_assert(TC >= 1, "R5V: r5_chunk(N) = 0 would never carry");
};

template<uint32_t N>
R5_INL void to_limbs(const uint64_t* w, uint64_t* l) {
    using C = Cfg<N>;
    for (uint32_t s = 0; s < C::NL; ++s) {
        const uint32_t bit = C::B * s, q = bit / 64, r = bit % 64;
        uint64_t x = q < N ? (w[q] >> r) : 0;
        if (r + C::B > 64 && q + 1 < N) x |= w[q + 1] << (64 - r);
        l[s] = x & C::M;
    }
}
// l[s] < 2^B (carried); the bits >= 64 N are dropped.
template<uint32_t N>
R5_INL void from_limbs(const uint64_t* l, uint64_t* w) {
    using C = Cfg<N>;
    for (uint32_t q = 0; q < N; ++q) w[q] = 0;
    for (uint32_t s = 0; s < C::NL; ++s) {
        const uint32_t bit = C::B * s, q = bit / 64, r = bit % 64;
        if (q >= N) break;
        w[q] |= l[s] << r;
        if (r + C::B > 64 && q + 1 < N) w[q + 1] |= l[s] >> (64 - r);
    }
}

// acc[S1 + S2] += f_S1 u_S2 for S2 < NL - S1 (the truncated product), unrolled: acc stays in registers.
template<uint32_t N, size_t S1, size_t... S2>
R5_INL void mac1(V* acc, V fb, const V* u, std::index_sequence<S2...>) {
    ((acc[S1 + S2] += R5_MUL(fb, u[S2])), ...);
}
template<uint32_t N, size_t... S1>
R5_INL void mac(V* acc, const uint64_t* f, uint32_t nz, const V* u, std::index_sequence<S1...>) {
    ((S1 < nz ? mac1<N, S1>(acc, V{} + f[S1], u, std::make_index_sequence<Cfg<N>::NL - S1>{}) : void()), ...);
}
// The carry pass, unrolled (constant indices only, so acc is never addressed in memory).
template<uint32_t N, size_t... S>
R5_INL void carry_seq(V* acc, std::index_sequence<S...>) {
    ((acc[S + 1] += acc[S] >> Cfg<N>::B, acc[S] &= Cfg<N>::M), ...);
}
template<uint32_t N>
R5_INL void carry(V* acc) {
    carry_seq<N>(acc, std::make_index_sequence<Cfg<N>::NL - 1>{});
    acc[Cfg<N>::NL - 1] &= Cfg<N>::M;
}
template<size_t... S>
R5_INL void zero_seq(V* acc, std::index_sequence<S...>) { ((acc[S] = V{}), ...); }
template<size_t... S>
R5_INL void copy_seq(V* dst, const V* acc, std::index_sequence<S...>) { ((dst[S] = acc[S]), ...); }

// C (mrows x ncols) -= L (mrows x np) U (np x ncols); element strides; disjoint C and U (as every caller has).
template<uint32_t N>
void gemm_sub(uint64_t* C, size_t ldc, uint32_t mrows, const uint64_t* L, size_t ldl, const uint64_t* U, size_t ldu,
              uint32_t np, uint32_t ncols, uint32_t fault) {
    using Cf = Cfg<N>;
    // Fault injection (validation only, r5k::set_fault; 0 in production): 1 drops the final borrow chain,
    // 2 never carries between rank-one updates (the lanes overflow past TC updates of large limbs).
    const uint32_t tc = fault == 2 ? 0xffffffffu : Cf::TC;
    constexpr uint32_t NL = Cf::NL;
    if (mrows == 0 || np == 0 || ncols == 0) return;
    // Column tiles: the packed U tile (np x jt elements, NL limbs each) stays <= 256 KiB.
    size_t jt = (size_t(1) << 18) / (size_t(np) * NL * 8);
    jt = jt < VW ? VW : jt - jt % VW;
    const size_t ncr = (size_t(ncols) + VW - 1) / VW * VW;
    if (jt > ncr) jt = ncr;
    const size_t nb = jt / VW;
    // (std::vector<V> does not honour V's alignment on every toolchain: a raw buffer, aligned by hand.)
    std::vector<uint64_t> up_store(nb * np * NL * VW + 8);
    V* const Upd = reinterpret_cast<V*>((reinterpret_cast<uintptr_t>(up_store.data()) + 63) & ~uintptr_t(63));
    std::vector<uint64_t> Lf(size_t(np) * NL);
    std::vector<uint32_t> nz(np);
    alignas(64) uint64_t tmp[NL][VW];
    uint64_t lb[NL];
    for (size_t j0 = 0; j0 < ncols; j0 += jt) {
        const size_t jn = j0 + jt < ncols ? jt : ncols - j0;
        const size_t nbb = (jn + VW - 1) / VW;
        // Pack U's tile: Up[(b np + t) NL + s] = limb s of U[t][j0 + b VW + lane], lane-wise (zero past ncols).
        for (size_t b = 0; b < nbb; ++b)
            for (uint32_t t = 0; t < np; ++t) {
                for (uint32_t lane = 0; lane < VW; ++lane) {
                    const size_t j = j0 + b * VW + lane;
                    if (j < ncols) to_limbs<N>(U + (size_t(t) * ldu + j) * N, lb);
                    else for (uint32_t s = 0; s < NL; ++s) lb[s] = 0;
                    for (uint32_t s = 0; s < NL; ++s) tmp[s][lane] = lb[s];
                }
                V* dst = Upd + (b * np + t) * NL;
                for (uint32_t s = 0; s < NL; ++s) std::memcpy(&dst[s], tmp[s], R5_VB);
            }
        for (uint32_t i = 0; i < mrows; ++i) {
            // The row's factors as limbs; nz[t] = the number of limbs up to the top non-zero one (0: f == 0, skipped).
            const uint64_t* Li = L + size_t(i) * ldl * N;
            bool any = false;
            for (uint32_t t = 0; t < np; ++t) {
                uint64_t* f = Lf.data() + size_t(t) * NL;
                to_limbs<N>(Li + size_t(t) * N, f);
                uint32_t z = NL;
                while (z > 0 && f[z - 1] == 0) --z;
                nz[t] = z;
                any |= z != 0;
            }
            if (!any) continue;
            uint64_t* Ci = C + size_t(i) * ldc * N;
            for (size_t b = 0; b < nbb; ++b) {
                V acc[NL];
                zero_seq(acc, std::make_index_sequence<NL>{});
                uint32_t cnt = 0;
                const V* ub = Upd + b * np * NL;
                for (uint32_t t = 0; t < np; ++t) {
                    if (nz[t] == 0) continue;
                    mac<N>(acc, Lf.data() + size_t(t) * NL, nz[t], ub + size_t(t) * NL, std::make_index_sequence<NL>{});
                    if (++cnt == tc) { carry<N>(acc); cnt = 0; }
                }
                carry<N>(acc);
                V sum[NL];   // (a copy with constant indices: acc itself is never addressed, so it stays in registers)
                copy_seq(sum, acc, std::make_index_sequence<NL>{});
                // C - S limb-wise with a borrow chain: d = c + 2^B - s - borrow in [0, 2^(B+1)).
                const size_t jb = j0 + b * VW;
                for (uint32_t lane = 0; lane < VW; ++lane) {
                    if (jb + lane < ncols) to_limbs<N>(Ci + (jb + lane) * N, lb);
                    else for (uint32_t s = 0; s < NL; ++s) lb[s] = 0;
                    for (uint32_t s = 0; s < NL; ++s) tmp[s][lane] = lb[s];
                }
                V bw = V{};
                for (uint32_t s = 0; s < NL; ++s) {
                    V c;
                    std::memcpy(&c, tmp[s], R5_VB);
                    const V d = c + (Cf::M + 1) - sum[s] - bw;
                    const V o = d & Cf::M;
                    bw = fault == 1 ? V{} : 1 - (d >> Cf::B);
                    std::memcpy(tmp[s], &o, R5_VB);
                }
                for (uint32_t lane = 0; lane < VW && jb + lane < ncols; ++lane) {
                    for (uint32_t s = 0; s < NL; ++s) lb[s] = tmp[s][lane];
                    from_limbs<N>(lb, Ci + (jb + lane) * N);
                }
            }
        }
    }
}

template<uint32_t N>
void axpy(uint64_t* r, const uint64_t* p, const uint64_t* f, size_t n, uint32_t fault) {
    if (n == 0) return;
    // r (1 x n) -= f (1 x 1) p (1 x n).
    gemm_sub<N>(r, n, 1, f, 1, p, n, 1, static_cast<uint32_t>(n), fault);
}

// OP 0: d += s;  OP 1: d -= s;  OP 2: d = 0 - d (s unused).
template<uint32_t N, int OP>
R5_INL void row_scalar(uint64_t* d, const uint64_t* s, size_t n) {
    for (size_t e = 0; e < n; ++e) {
        uint64_t* x = d + e * N;
        const uint64_t* y = OP == 2 ? nullptr : s + e * N;
        uint64_t c = 0;
        for (uint32_t q = 0; q < N; ++q) {
            const uint64_t a = OP == 2 ? 0 : x[q], b = OP == 2 ? x[q] : y[q];
            if (OP == 0) {
                const uint64_t t = a + b, u = t + c;
                c = static_cast<uint64_t>(t < a) | static_cast<uint64_t>(u < t);
                x[q] = u;
            } else {
                const uint64_t t = a - b, u = t - c;
                c = static_cast<uint64_t>(a < b) | static_cast<uint64_t>(t < c);
                x[q] = u;
            }
        }
    }
}

#if R5_ROWMASKS
// Chunks of CW words = E whole elements, CW the largest multiple of lcm(N, VW) <= 64, so a chunk is NV whole
// vectors (direct loads / stores): the word sums / differences lane-wise, then the carries of the whole chunk at
// once.  G = the words that generate a carry (borrow), P = the words that propagate one (sum all ones /
// difference zero), both cleared at each element's top word (its carry out is dropped and must not reach the
// next element); the carry INTO word i is bit i of ((G << 1) + P) ^ P.  The elements past the last whole chunk
// take the scalar chain.
constexpr uint32_t r5_gcd(uint32_t a, uint32_t b) { return b == 0 ? a : r5_gcd(b, a % b); }
template<uint32_t N> struct RowCfg {
    static constexpr uint32_t LCM = N / r5_gcd(N, VW) * VW;
    static constexpr uint32_t CW = 64 / LCM * LCM;
    static constexpr uint32_t E = CW / N, NV = CW / VW;
    static_assert(E >= 1, "R5V: lcm(N, VW) > 64 leaves no whole element per row chunk (row_op would not advance)");
    static constexpr uint64_t TOP = [] {
        uint64_t t = 0;
        for (uint32_t e = 0; e < E; ++e) t |= uint64_t(1) << (e * N + N - 1);
        return t;
    }();
};
template<uint32_t N, int OP>
R5_INL void row_chunk(uint64_t* d, const uint64_t* s, uint64_t top) {
    using RC = RowCfg<N>;
    V r[RC::NV];
    uint64_t G = 0, P = 0;
    for (uint32_t v = 0; v < RC::NV; ++v) {
        V x, y;
        if (OP == 2) {
            x = V{};
            std::memcpy(&y, d + v * VW, R5_VB);
        } else {
            std::memcpy(&x, d + v * VW, R5_VB);
            std::memcpy(&y, s + v * VW, R5_VB);
        }
        if (OP == 0) {
            const V z = x + y;
            G |= uint64_t(R5_LTMASK(z, x)) << (v * VW);
            P |= uint64_t(R5_EQMASK(z, V{} + ~uint64_t(0))) << (v * VW);
            r[v] = z;
        } else {
            const V z = x - y;
            G |= uint64_t(R5_LTMASK(x, y)) << (v * VW);
            P |= uint64_t(R5_EQMASK(z, V{})) << (v * VW);
            r[v] = z;
        }
    }
    G &= ~top;
    P &= ~top;
    const uint64_t c = ((G << 1) + P) ^ P;
    for (uint32_t v = 0; v < RC::NV; ++v) {
        const unsigned bits = static_cast<unsigned>(c >> (v * VW)) & ((1u << VW) - 1);
        const V z = OP == 0 ? R5_ADDBITS(r[v], bits) : R5_SUBBITS(r[v], bits);
        std::memcpy(d + v * VW, &z, R5_VB);
    }
}
#endif

template<uint32_t N, int OP>
void row_op(uint64_t* d, const uint64_t* s, size_t n, uint32_t fault) {
    // Fault 3 (validation only): the element top-word mask is not applied, so carries leak into the next element.
    size_t e = 0;
#if R5_ROWMASKS
    constexpr uint32_t E = RowCfg<N>::E;
    const uint64_t top = fault == 3 ? 0 : RowCfg<N>::TOP;
    for (; e + E <= n; e += E) row_chunk<N, OP>(d + e * N, OP == 2 ? nullptr : s + e * N, top);
#endif
    (void)fault;
    if (e < n) row_scalar<N, OP>(d + e * N, OP == 2 ? nullptr : s + e * N, n - e);
}

} // namespace R5_NS
