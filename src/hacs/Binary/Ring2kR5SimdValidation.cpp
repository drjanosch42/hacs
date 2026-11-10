// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Ring2kR5SimdValidation (slot-23 R5V).  See Ring2kR5SimdValidation.h, Ring2k.h (r5k::) and ENGINE_API.md Sec. 8.2.
//
// The oracle is the scalar Traits<WordN<N>> body (axpy_scalar / gemm_sub_scalar / *_row_scalar, and the whole
// ladder under set_r5_kernel(Scalar)).  The scalar body itself is checked against nf::BigInt (R6), ModCtx and
// engine::ladic by Ring2kLadderValidation; here every SIMD path must reproduce it entrywise.

#include "Ring2kR5SimdValidation.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../ModArithmetic/Binary/Ring2k.h"
#include "../Orbifold/L23/AmalgamReader.h"

namespace {

namespace r2 = engine::ring2k;
using r2::KernelPath;
using r2::WordN;

uint32_t g_fail = 0, g_checks = 0;
uint64_t g_seed = 0x5235565f53494d44ull;   // "R5V_SIMD"

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 25) std::cout << "  [FAIL] " << what << std::endl;
    }
}
uint64_t splitmix() {
    uint64_t z = (g_seed += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
uint32_t urand(uint32_t m) { return m == 0 ? 0 : static_cast<uint32_t>(splitmix() % m); }
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<KernelPath> available_paths() {
    std::vector<KernelPath> v;
    for (KernelPath p : {KernelPath::Portable, KernelPath::Avx2, KernelPath::Avx512})
        if (r2::kernel_path_available(p)) v.push_back(p);
    return v;
}

// Restores the global switches whatever happens.
struct Restore {
    ~Restore() {
        r2::r5k::set_fault(0);
        r2::set_r5_kernel(r2::R5Kernel::Simd);
        r2::set_kernel_path(KernelPath::Auto);
    }
};

// One element of the fuzz distribution: zero, small +-, all ones, one random word, a value masked to k bits, random.
template<uint32_t N>
WordN<N> rnd_elem(uint32_t kind) {
    WordN<N> e;
    for (uint32_t q = 0; q < N; ++q) e.w[q] = 0;
    switch (kind % 8) {
    case 0: break;
    case 1: e.w[0] = urand(9); break;
    case 2: { const uint64_t v = 1 + urand(9); e.w[0] = uint64_t(0) - v; for (uint32_t q = 1; q < N; ++q) e.w[q] = ~uint64_t(0); break; }
    case 3: for (uint32_t q = 0; q < N; ++q) e.w[q] = ~uint64_t(0); break;
    case 4: e.w[urand(N)] = splitmix(); break;
    case 5: {
        const uint32_t k = 1 + urand(64 * N);
        for (uint32_t q = 0; q < N; ++q) e.w[q] = splitmix();
        e = r2::Traits<WordN<N>>::mask(e, k);
        break;
    }
    default: for (uint32_t q = 0; q < N; ++q) e.w[q] = splitmix(); break;
    }
    return e;
}
template<uint32_t N>
std::vector<WordN<N>> rnd_vec(size_t n, uint32_t bias) {
    std::vector<WordN<N>> v(n);
    for (auto& e : v) e = rnd_elem<N>(bias == 8 ? urand(8) : (urand(4) == 0 ? urand(8) : bias));
    return v;
}
template<uint32_t N>
bool same(const std::vector<WordN<N>>& a, const std::vector<WordN<N>>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(WordN<N>)) == 0);
}

// ---------------------------------------------------------------------------
// (A) kernel fuzz
// ---------------------------------------------------------------------------
template<uint32_t N>
uint32_t part_kernels(KernelPath p) {
    using Tr = r2::Traits<WordN<N>>;
    using E = WordN<N>;
    r2::set_kernel_path(p);
    r2::set_r5_kernel(r2::R5Kernel::Simd);
    const std::string tag = "N=" + std::to_string(N) + " path=" + r2::kernel_path_name(p);
    const uint32_t tc = r2::r5k::layout(N, p).TC;
    uint32_t cases = 0;
    for (uint32_t it = 0; it < 40; ++it) {
        const uint32_t mr = urand(10), np = 1 + urand(it % 4 == 0 ? 2 * tc + 8 : 70), nc = urand(it % 5 == 0 ? 300 : 70);
        const uint32_t bias = urand(9);
        const size_t ldc = nc + urand(3), ldl = np + urand(3), ldu = nc + urand(3);
        std::vector<E> C = rnd_vec<N>(size_t(mr) * ldc, bias), L = rnd_vec<N>(size_t(mr) * ldl, bias), U = rnd_vec<N>(size_t(np) * ldu, bias);
        std::vector<E> C1 = C, C2 = C;
        Tr::gemm_sub_scalar(C1.data(), ldc, mr, L.data(), ldl, U.data(), ldu, np, nc);
        Tr::gemm_sub(C2.data(), ldc, mr, L.data(), ldl, U.data(), ldu, np, nc);
        check(same(C1, C2), tag + " gemm " + std::to_string(mr) + "x" + std::to_string(np) + "x" + std::to_string(nc));
        ++cases;
    }
    {   // the delayed-carry stress: all-ones operands, np >> TC
        const uint32_t mr = 3, np = 3 * tc + 5, nc = 19;
        std::vector<E> C(size_t(mr) * nc), L(size_t(mr) * np), U(size_t(np) * nc);
        for (auto& e : C) e = rnd_elem<N>(7);
        for (auto& e : L) e = rnd_elem<N>(3);
        for (auto& e : U) e = rnd_elem<N>(3);
        std::vector<E> C1 = C, C2 = C;
        Tr::gemm_sub_scalar(C1.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
        Tr::gemm_sub(C2.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
        check(same(C1, C2), tag + " gemm all-ones stress np=" + std::to_string(np));
        ++cases;
    }
    for (uint32_t it = 0; it < 30; ++it) {
        const size_t n = urand(it % 3 == 0 ? 400 : 40);
        std::vector<E> r = rnd_vec<N>(n, 8), q = rnd_vec<N>(n, 8);
        const E f = rnd_elem<N>(urand(8));
        std::vector<E> r1 = r, r2v = r;
        Tr::axpy_scalar(r1.data(), q.data(), f, n);
        Tr::axpy(r2v.data(), q.data(), f, n);
        check(same(r1, r2v), tag + " axpy n=" + std::to_string(n));
        ++cases;
    }
    for (uint32_t it = 0; it < 60; ++it) {
        const size_t n = urand(it % 3 == 0 ? 500 : 70);
        const uint32_t bias = urand(9);
        std::vector<E> d = rnd_vec<N>(n, bias), s = rnd_vec<N>(n, bias);
        std::vector<E> a1 = d, a2 = d, b1 = d, b2 = d, c1 = d, c2 = d;
        Tr::add_row_scalar(a1.data(), s.data(), n);
        Tr::add_row(a2.data(), s.data(), n);
        Tr::sub_row_scalar(b1.data(), s.data(), n);
        Tr::sub_row(b2.data(), s.data(), n);
        Tr::neg_row_scalar(c1.data(), n);
        Tr::neg_row(c2.data(), n);
        check(same(a1, a2), tag + " add_row n=" + std::to_string(n));
        check(same(b1, b2), tag + " sub_row n=" + std::to_string(n));
        check(same(c1, c2), tag + " neg_row n=" + std::to_string(n));
        cases += 3;
    }
    return cases;
}

// ---------------------------------------------------------------------------
// (B) end to end: smith / howell / kernel_module, SIMD vs scalar
// ---------------------------------------------------------------------------
template<uint32_t N>
WordN<N> from_i64(int64_t v) {
    using Tr = r2::Traits<WordN<N>>;
    const uint64_t u = v < 0 ? uint64_t(0) - static_cast<uint64_t>(v) : static_cast<uint64_t>(v);
    const WordN<N> x = Tr::from_u64(u);
    return v < 0 ? Tr::neg(x) : x;
}
template<uint32_t N>
using EM = r2::Mat<WordN<N>>;
template<uint32_t N>
EM<N> emul(const EM<N>& A, const EM<N>& B) {
    using Tr = r2::Traits<WordN<N>>;
    EM<N> C(A.rows, B.cols);
    for (uint32_t i = 0; i < A.rows; ++i)
        for (uint32_t t = 0; t < A.cols; ++t) {
            if (Tr::is_zero(A.at(i, t))) continue;
            for (uint32_t j = 0; j < B.cols; ++j) C.at(i, j) = Tr::add(C.at(i, j), Tr::mul(A.at(i, t), B.at(t, j)));
        }
    return C;
}
template<uint32_t N>
EM<N> eadd(EM<N> A, const EM<N>& B, int64_t sb) {
    using Tr = r2::Traits<WordN<N>>;
    for (size_t x = 0; x < A.a.size(); ++x) A.a[x] = Tr::add(A.a[x], Tr::mul(from_i64<N>(sb), B.a[x]));
    return A;
}
template<uint32_t N>
EM<N> eident(uint32_t d) {
    EM<N> I(d, d);
    for (uint32_t i = 0; i < d; ++i) I.at(i, i) = r2::Traits<WordN<N>>::one();
    return I;
}
// rho_n(g), entry (k, j) = the coefficient of y^k in (a + c y)^(n-j) (b + d y)^j, mod 2^(64 N).
template<uint32_t N>
EM<N> rho(int64_t a, int64_t b, int64_t c, int64_t d, uint32_t n) {
    using Tr = r2::Traits<WordN<N>>;
    auto ppow = [&](int64_t x, int64_t y, uint32_t e) {
        std::vector<WordN<N>> p(1, Tr::one());
        for (uint32_t t = 0; t < e; ++t) {
            std::vector<WordN<N>> q(p.size() + 1, Tr::zero());
            for (size_t i = 0; i < p.size(); ++i) {
                q[i] = Tr::add(q[i], Tr::mul(p[i], from_i64<N>(x)));
                q[i + 1] = Tr::add(q[i + 1], Tr::mul(p[i], from_i64<N>(y)));
            }
            p.swap(q);
        }
        return p;
    };
    EM<N> M(n + 1, n + 1);
    for (uint32_t j = 0; j <= n; ++j) {
        const auto p1 = ppow(a, c, n - j), p2 = ppow(b, d, j);
        for (size_t x = 0; x < p1.size(); ++x)
            for (size_t y = 0; y < p2.size(); ++y)
                M.at(static_cast<uint32_t>(x + y), j) = Tr::add(M.at(static_cast<uint32_t>(x + y), j), Tr::mul(p1[x], p2[y]));
    }
    return M;
}
template<uint32_t N>
void put(EM<N>& T, const EM<N>& B, uint32_t r0, uint32_t c0) {
    for (uint32_t i = 0; i < B.rows; ++i)
        for (uint32_t j = 0; j < B.cols; ++j) T.at(r0 + i, c0 + j) = B.at(i, j);
}
// The exact Sym^n coboundary [rho(S) - 1; rho(ST) - 1] and the cocycle relation matrix of Z/4 *_{Z/2} Z/6.
template<uint32_t N>
std::vector<EM<N>> amalgam(uint32_t n) {
    const uint32_t d = n + 1;
    const EM<N> S = rho<N>(0, -1, 1, 0, n), U = rho<N>(0, -1, 1, 1, n), I = eident<N>(d);
    const EM<N> S2 = emul(S, S), S3 = emul(S2, S), U2 = emul(U, U), U3 = emul(U2, U), U4 = emul(U3, U), U5 = emul(U4, U);
    EM<N> d0(2 * d, d);
    put(d0, eadd(S, I, -1), 0, 0);
    put(d0, eadd(U, I, -1), d, 0);
    const EM<N> NS = eadd(eadd(eadd(I, S, 1), S2, 1), S3, 1);
    const EM<N> NU = eadd(eadd(eadd(eadd(eadd(I, U, 1), U2, 1), U3, 1), U4, 1), U5, 1);
    EM<N> z1(3 * d, 2 * d);
    put(z1, NS, 0, 0);
    put(z1, NU, d, d);
    put(z1, eadd(I, S, 1), 2 * d, 0);
    put(z1, eadd(eadd(I, U, 1), U2, 1), 2 * d, d);
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t j = 0; j < d; ++j) z1.at(2 * d + i, d + j) = r2::Traits<WordN<N>>::neg(z1.at(2 * d + i, d + j));
    return {d0, z1};
}
template<uint32_t N>
r2::WideMatrix to_wide_k(const EM<N>& M, uint32_t k) { return r2::to_wide(M, k); }

template<uint32_t N>
uint32_t e2e_one(const r2::WideMatrix& A, const std::vector<KernelPath>& paths, const std::string& what) {
    const r2::RungChoice rung{r2::Rung::R5, N, 64 * N};
    r2::set_r5_kernel(r2::R5Kernel::Scalar);
    const r2::SmithResult s0 = r2::smith_on(rung, A);
    const r2::WideHowell h0 = r2::howell_on(rung, A);
    const r2::WideHowell k0 = r2::kernel_module_on(rung, A);
    r2::set_r5_kernel(r2::R5Kernel::Simd);
    for (KernelPath p : paths) {
        r2::set_kernel_path(p);
        const r2::SmithResult s1 = r2::smith_on(rung, A);
        const std::string tag = what + " N=" + std::to_string(N) + " k=" + std::to_string(A.k) + " path=" + r2::kernel_path_name(p);
        check(s1.sigma == s0.sigma && s1.level_ranks == s0.level_ranks, tag + " smith");
        check(r2::howell_on(rung, A) == h0, tag + " howell");
        check(r2::kernel_module_on(rung, A) == k0, tag + " kernel_module");
    }
    r2::set_kernel_path(KernelPath::Auto);
    return 1;
}

template<uint32_t N>
uint32_t part_e2e(const std::vector<KernelPath>& paths) {
    using Tr = r2::Traits<WordN<N>>;
    uint32_t mats = 0;
    auto pick_k = [] { const uint32_t r = urand(3); return r == 0 ? 64 * N : r == 1 ? 64 * (N - 1) + 1 + urand(63) : 65 + urand(64 * N - 65); };
    for (uint32_t it = 0; it < 4; ++it) {   // random
        const uint32_t m = 1 + urand(60), n = 1 + urand(60), k = pick_k();
        r2::WideMatrix A(m, n, k);
        for (auto& w : A.w) w = splitmix();
        A.canonicalise();
        mats += e2e_one<N>(A, paths, "random " + std::to_string(m) + "x" + std::to_string(n));
    }
    for (uint32_t it = 0; it < 4; ++it) {   // planted: X (m x r) diag(2^v) Y (r x n), rank-deficient with mixed valuations
        const uint32_t m = 8 + urand(50), n = 8 + urand(50), r = 1 + urand(std::min(m, n) - 1), k = pick_k();
        EM<N> X(m, r), D(r, r), Y(r, n);
        for (auto& e : X.a) e = rnd_elem<N>(urand(3) == 0 ? 1 : 7);
        for (auto& e : Y.a) e = rnd_elem<N>(urand(3) == 0 ? 2 : 7);
        for (uint32_t i = 0; i < r; ++i) D.at(i, i) = Tr::shl(Tr::one(), urand(std::min<uint32_t>(k, 200)));
        mats += e2e_one<N>(to_wide_k<N>(emul(emul(X, D), Y), k), paths,
                           "planted " + std::to_string(m) + "x" + std::to_string(n) + " r=" + std::to_string(r));
    }
    const uint32_t nlist[] = {6, 17, 30};
    for (uint32_t n : nlist) {   // amalgam
        const std::vector<EM<N>> am = amalgam<N>(n);
        for (size_t t = 0; t < am.size(); ++t)
            mats += e2e_one<N>(to_wide_k<N>(am[t], pick_k()), paths, std::string(t == 0 ? "amalgam d0" : "amalgam z1") + " n=" + std::to_string(n));
    }
    return mats;
}

// ---------------------------------------------------------------------------
// (C) the BIN-3 reader on an R5 rung, SIMD vs scalar
// ---------------------------------------------------------------------------
std::string reader_fingerprint(const l23::LatticeCell& c) {
    std::string s = c.ladic2_line() + " rung=" + c.rung + " K=" + std::to_string(c.K) + " m=" + std::to_string(c.m) +
                    " ok=" + std::to_string(c.ok) + " failures=" + std::to_string(c.failures.size()) + " tell=" + c.ed_tell.census() +
                    " tell2=" + c.ed_tell2.census() + " commute=" + std::to_string(c.commute);
    for (const auto& kv : c.ed_tp) s += " tp" + std::to_string(kv.first) + "=" + kv.second.census();
    for (const auto& kv : c.ed_eis) s += " eis" + std::to_string(kv.first) + "=" + kv.second.census();
    for (const std::string& l : c.log) s += " | " + l;
    return s;
}
uint32_t part_reader(const std::vector<KernelPath>& paths) {
    struct Cell { uint32_t n, k_min; };
    const Cell cells[] = {{12, 150}, {30, 150}, {46, 150}, {30, 200}, {24, 330}};
    uint32_t runs = 0;
    for (const Cell& cl : cells) {
        l23::ReaderOptions o;
        o.ell = 2;
        o.k_min = cl.k_min;
        o.k_cap = 1024;
        r2::set_r5_kernel(r2::R5Kernel::Scalar);
        const l23::LatticeCell c0 = l23::read_lattice(cl.n, o);
        const std::string f0 = reader_fingerprint(c0);
        check(c0.ok && c0.rung.rfind("R5(", 0) == 0, "reader n=" + std::to_string(cl.n) + " k_min=" + std::to_string(cl.k_min) +
                                                         " scalar run ok on an R5 rung (" + c0.rung + ")");
        r2::set_r5_kernel(r2::R5Kernel::Simd);
        for (KernelPath p : paths) {
            r2::set_kernel_path(p);
            const l23::LatticeCell c1 = l23::read_lattice(cl.n, o);
            const bool eq = reader_fingerprint(c1) == f0;
            check(eq, "reader n=" + std::to_string(cl.n) + " " + c0.rung + " path=" + r2::kernel_path_name(p) + " = scalar");
            if (!eq) std::cout << "    scalar: " << f0 << "\n    simd:   " << reader_fingerprint(c1) << std::endl;
            ++runs;
        }
        r2::set_kernel_path(KernelPath::Auto);
        if (cl.n == 46) std::cout << "  reader (2, 46) " << c0.rung << ": " << c0.ladic2_line() << std::endl;
    }
    return runs;
}

// ---------------------------------------------------------------------------
// (D) mutation controls
// ---------------------------------------------------------------------------
uint32_t g_armed = 0, g_caught = 0;
void mutation(bool caught, const std::string& what) {
    ++g_armed;
    if (caught) ++g_caught;
    check(caught, "mutation caught: " + what);
}
template<uint32_t N>
void part_mutations(const std::vector<KernelPath>& paths) {
    using Tr = r2::Traits<WordN<N>>;
    using E = WordN<N>;
    r2::set_r5_kernel(r2::R5Kernel::Simd);
    for (KernelPath p : paths) {
        r2::set_kernel_path(p);
        const std::string tag = std::string(" N=") + std::to_string(N) + " path=" + r2::kernel_path_name(p);
        const uint32_t tc = r2::r5k::layout(N, p).TC;
        // 1: the borrow chain dropped (random operands).
        {
            const uint32_t mr = 4, np = 9, nc = 33;
            std::vector<E> C = rnd_vec<N>(size_t(mr) * nc, 7), L = rnd_vec<N>(size_t(mr) * np, 7), U = rnd_vec<N>(size_t(np) * nc, 7);
            std::vector<E> C1 = C, C2 = C;
            Tr::gemm_sub_scalar(C1.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
            r2::r5k::set_fault(1);
            Tr::gemm_sub(C2.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
            r2::r5k::set_fault(0);
            mutation(!same(C1, C2), "fault 1 (no borrow chain)" + tag);
        }
        // 2: no carries between rank-one updates (the all-ones stress with np >> TC overflows the lanes).
        {
            const uint32_t mr = 2, np = 3 * tc + 5, nc = 9;
            std::vector<E> C(size_t(mr) * nc), L(size_t(mr) * np), U(size_t(np) * nc);
            for (auto& e : L) e = rnd_elem<N>(3);
            for (auto& e : U) e = rnd_elem<N>(3);
            std::vector<E> C1 = C, C2 = C;
            Tr::gemm_sub_scalar(C1.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
            r2::r5k::set_fault(2);
            Tr::gemm_sub(C2.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
            r2::r5k::set_fault(0);
            mutation(!same(C1, C2), "fault 2 (no intermediate carries)" + tag);
        }
        // 3: the row kernels' element top-word mask dropped (only the mask-carry paths have one).
        if (p != KernelPath::Portable) {
            const size_t n = 200;
            std::vector<E> d = rnd_vec<N>(n, 7), s = rnd_vec<N>(n, 7);
            std::vector<E> a1 = d, a2 = d;
            Tr::add_row_scalar(a1.data(), s.data(), n);
            r2::r5k::set_fault(3);
            Tr::add_row(a2.data(), s.data(), n);
            r2::r5k::set_fault(0);
            mutation(!same(a1, a2), "fault 3 (no top-word mask)" + tag);
        }
        // 4: one flipped output bit is seen by the comparator.
        {
            const uint32_t mr = 3, np = 5, nc = 11;
            std::vector<E> C = rnd_vec<N>(size_t(mr) * nc, 7), L = rnd_vec<N>(size_t(mr) * np, 7), U = rnd_vec<N>(size_t(np) * nc, 7);
            std::vector<E> C1 = C, C2 = C;
            Tr::gemm_sub_scalar(C1.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
            Tr::gemm_sub(C2.data(), nc, mr, L.data(), np, U.data(), nc, np, nc);
            check(same(C1, C2), "mutation baseline equal" + tag);
            C2[urand(static_cast<uint32_t>(C2.size()))].w[urand(N)] ^= uint64_t(1) << urand(64);
            mutation(!same(C1, C2), "flipped output bit" + tag);
        }
    }
    r2::set_kernel_path(KernelPath::Auto);
}

// ---------------------------------------------------------------------------
// (E) timing (reported, never gated)
// ---------------------------------------------------------------------------
template<uint32_t N>
void part_timing() {
    using Tr = r2::Traits<WordN<N>>;
    using E = WordN<N>;
    const uint32_t m = 192, np = 64, nc = 384;
    std::vector<E> C = rnd_vec<N>(size_t(m) * nc, 7), L = rnd_vec<N>(size_t(m) * np, 7), U = rnd_vec<N>(size_t(np) * nc, 7);
    std::vector<E> C1 = C, C2 = C;
    double t = now_s();
    Tr::gemm_sub_scalar(C1.data(), nc, m, L.data(), np, U.data(), nc, np, nc);
    const double gs = now_s() - t;
    r2::set_kernel_path(KernelPath::Auto);
    t = now_s();
    Tr::gemm_sub(C2.data(), nc, m, L.data(), np, U.data(), nc, np, nc);
    const double gv = now_s() - t;
    check(same(C1, C2), "timing gemm equal N=" + std::to_string(N));
    const size_t rows = 64, cols = 1000;
    std::vector<E> R = rnd_vec<N>(rows * cols, 7), R1 = R, R2 = R;
    t = now_s();
    for (int z = 0; z < 20; ++z)
        for (size_t r = 0; r + 1 < rows; ++r) Tr::add_row_scalar(R1.data() + r * cols, R1.data() + (r + 1) * cols, cols);
    const double rs = now_s() - t;
    t = now_s();
    for (int z = 0; z < 20; ++z)
        for (size_t r = 0; r + 1 < rows; ++r) Tr::add_row(R2.data() + r * cols, R2.data() + (r + 1) * cols, cols);
    const double rv = now_s() - t;
    check(same(R1, R2), "timing rows equal N=" + std::to_string(N));
    char buf[400];
    std::snprintf(buf, sizeof(buf),
                  "R5V_TIMING N=%u path=%s gemm(%ux%ux%u) scalar_s=%.4f simd_s=%.4f speedup=%.1f rows(add %zux%zu x20) scalar_s=%.4f "
                  "simd_s=%.4f speedup=%.1f",
                  N, r2::kernel_path_name(r2::active_kernel_path()), m, np, nc, gs, gv, gv > 0 ? gs / gv : 0.0, rows - 1, cols, rs, rv,
                  rv > 0 ? rs / rv : 0.0);
    std::cout << "  " << buf << std::endl;
}

template<uint32_t N>
void for_n(const std::vector<KernelPath>& paths, uint32_t& kernel_cases, uint32_t& e2e_mats) {
    for (KernelPath p : paths) kernel_cases += part_kernels<N>(p);
    e2e_mats += part_e2e<N>(paths);
    part_mutations<N>(paths);
}

} // namespace

uint32_t Ring2kR5SimdValidation() {
    const double t0 = now_s();
    g_fail = 0;
    g_checks = 0;
    g_armed = 0;
    g_caught = 0;
    std::cout << "=== Ring2kR5SimdValidation (ring2k R5: the SoA / SIMD kernels vs the scalar WordN<N> bodies) ===" << std::endl;
    Restore restore;
    const std::vector<KernelPath> paths = available_paths();
    std::string pstr;
    for (KernelPath p : paths) pstr += std::string(pstr.empty() ? "" : ",") + r2::kernel_path_name(p);
    std::cout << "  kernel paths available: " << pstr << " (auto = " << r2::kernel_path_name(r2::active_kernel_path()) << ")" << std::endl;
    std::string ns;
    for (uint32_t N : r2::r5_word_counts())
        if (r2::r5_simd_instantiated(N)) {
            const r2::r5k::Layout l = r2::r5k::layout(N, KernelPath::Auto);
            ns += std::string(ns.empty() ? "" : ",") + std::to_string(N);
            std::cout << "  N=" << N << ": limbs B=" << l.B << " NL=" << l.NL << " carry every TC=" << l.TC << " updates, lanes=" << l.lanes
                      << std::endl;
        }
    uint32_t kernel_cases = 0, e2e_mats = 0, reader_runs = 0;
    try {
        check(r2::r5_kernel() == r2::R5Kernel::Simd && r2::r5k::fault() == 0, "production defaults: SIMD kernels, no fault");
        check(!r2::r5_simd_instantiated(20) && r2::r5_simd_instantiated(16), "SIMD bodies exactly for N in {3,4,5,6,8,10,12,16}");
        for_n<3>(paths, kernel_cases, e2e_mats);
        for_n<4>(paths, kernel_cases, e2e_mats);
        for_n<5>(paths, kernel_cases, e2e_mats);
        for_n<6>(paths, kernel_cases, e2e_mats);
        for_n<8>(paths, kernel_cases, e2e_mats);
        for_n<10>(paths, kernel_cases, e2e_mats);
        for_n<12>(paths, kernel_cases, e2e_mats);
        for_n<16>(paths, kernel_cases, e2e_mats);
        std::cout << "  kernels: " << kernel_cases << " cases; end to end: " << e2e_mats << " matrices; " << g_checks << " checks so far, "
                  << g_fail << " failing" << std::endl;
        reader_runs = part_reader(paths);
        std::cout << "  reader: " << reader_runs << " SIMD runs vs scalar (" << g_checks << " checks so far, " << g_fail << " failing)"
                  << std::endl;
        part_timing<3>();
        part_timing<6>();
        part_timing<12>();
        part_timing<16>();
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    const double secs = now_s() - t0;
    std::cout << "RING2K_R5_SIMD Ns=" << ns << " paths=" << pstr << " kernel=" << kernel_cases << " e2e=" << e2e_mats << " reader=" << reader_runs
              << " mutations=" << g_caught << "/" << g_armed << " checks=" << g_checks << " failing=" << g_fail
              << " verdict=" << (g_fail == 0 ? "OK" : "FAIL") << std::endl;
    std::cout << "  " << secs << " s" << std::endl;
    std::cout << (g_fail == 0 ? "Ring2kR5SimdValidation: PASS" : "Ring2kR5SimdValidation: FAIL") << std::endl;
    return g_fail;
}
