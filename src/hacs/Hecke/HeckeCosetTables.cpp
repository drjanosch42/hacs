// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S3 -- see HeckeCosetTables.h for the contract references, the
// binding indexing convention (1-based coset indices) and the documented
// r >= 2 representative rule.

#include "HeckeCosetTables.h"

#include "HeckeOperatorOnK.h"   // (slot-14 WP-10) HeckeOperatorOptions: coset streaming, parallel setup
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <stdexcept>
#include <cstdint>
#include <type_traits>

namespace hecke {

namespace {

// Fixed-seed xorshift64 (deterministic across runs and build settings) --
// same construction as Hecke/HeckeTessellationSmokeValidation.cpp, reused
// here for the self_check() 100 random-word chain-rule battery.
struct XorShift64 {
    uint64_t state;
    explicit XorShift64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    uint64_t below(uint64_t n) { return next() % n; }
};

} // namespace

// --- generator dictionary ------------------------------------------------

template<class Elt>
Mat22 HeckeCosetTablesT<Elt>::generator_matrix(uint32_t gen) {
    switch (gen) {
        case 0: return Mat22::T();
        case 1: return Mat22::S();
        case 2: return Mat22::ST();
        case 3: return Mat22::TS();
        default:
            throw std::out_of_range(
                "HeckeCosetTables: gen must be in {0,1,2,3} "
                "(0:T, 1:S, 2:ST, 3:TS)");
    }
}

// --- bounds checks ---------------------------------------------------------

template<class Elt>
void HeckeCosetTablesT<Elt>::check_index(uint32_t i) const {
    if (i == 0 || i > m_) {
        throw std::out_of_range("HeckeCosetTables: coset index out of range "
                                 "(valid range is [1, num_cosets()])");
    }
}

template<class Elt>
void HeckeCosetTablesT<Elt>::check_gen(uint32_t gen) const {
    if (gen >= 4) {
        throw std::out_of_range("HeckeCosetTables: gen must be in {0,1,2,3}");
    }
}

// --- rho_n(.) via ComputeSymmetricPowerMatrixFM (invariant I2) -------------

template<class Elt>
FlatMatrixT<Elt> HeckeCosetTablesT<Elt>::rho_of(const Mat22& m) const {
    std::array<Elt, 4> gm;
    if constexpr (std::is_same_v<Elt, uint32_t>) {
        gm = m.reduced_mod(ell_);
    } else {
        gm = m.reduced_mod64(ell_);
    }
    const FlatMatrixT<Elt> g_fm(2, 2, ell_,
                           std::vector<Elt>{ gm[0], gm[1], gm[2], gm[3] });
    return ComputeSymmetricPowerMatrixFM(g_fm, n_sym_);
}

// --- (P1) class map, entirely mod N (I6 early reduction) -------------------
//
// N = p^r has the SINGLE prime factor p, so for any (c,d) mod N with
// gcd(c,d,N) == 1 exactly one of the following holds: (a) p does not divide
// d, in which case d is invertible mod N and (c:d) ~ (c*d^{-1} : 1); or
// (b) p divides d, in which case p does not divide c (else gcd(c,d,N) would
// be divisible by p), c is invertible mod N, and (c:d) ~ (1 : d*c^{-1})
// with d*c^{-1} automatically divisible by p (c^{-1} is a unit, so
// multiplying the p-divisible d by it cannot remove the factor of p).
// ctxN_.inv forwards to ModArithmetic::mod_inv, the general extended-Euclid
// inverse (valid for any modulus with gcd(a,N) == 1, not just prime N), so
// this is correct for every r >= 1.
template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::class_index(uint32_t c_mod_N, uint32_t d_mod_N) const {
    const uint32_t c = c_mod_N % N32_;
    const uint32_t d = d_mod_N % N32_;

    if (d % p32_ != 0) {
        const uint32_t d_inv = ctxN_.inv(d);
        const uint32_t k = ctxN_.mul(c, d_inv);
        return k + 1; // branch A: i = k + 1, k in [0, N)
    }

    if (c % p32_ == 0) {
        throw std::invalid_argument(
            "HeckeCosetTables::class_index: (c:d) is not a valid P^1(Z/N) "
            "point (gcd(c,d,N) != 1)");
    }
    const uint32_t c_inv = ctxN_.inv(c);
    const uint32_t e = ctxN_.mul(d, c_inv);
    if (e % p32_ != 0) {
        throw std::runtime_error(
            "HeckeCosetTables::class_index: internal invariant violated "
            "(expected d*c^{-1} divisible by p)");
    }
    const uint32_t c_val = e / p32_;
    return N32_ + c_val + 1; // branch B: i = N + c_val + 1, c_val in [0, p^{r-1})
}

template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::class_index_of(const Mat22& gamma) const {
    const std::array<uint32_t, 4> rm = gamma.reduced_mod(N32_);
    return class_index(rm[2], rm[3]); // bottom row: [a,b,c,d] -> (c,d) = (rm[2], rm[3])
}

// --- the shared (T1) split-and-conjugate step (WP-4a) ------------------------

template<class Elt>
Mat22 HeckeCosetTablesT<Elt>::twisted_conjugate(const Mat22& gamma, uint32_t& class_out) const {
    // gamma = h * s_j with j = class_index_of(gamma), h in Gamma_0(N) ((P1)).
    const uint32_t j = class_index_of(gamma);
    const Mat22 h = gamma * s_inv_[j];
    const int64_t Ni = static_cast<int64_t>(N_);
    if (det_tiered(h) != 1 || (h.c % Ni) != 0) {   // (slot-24 ELL-W) the wide tier on the determinant
        throw std::runtime_error(
            "HeckeCosetTables: (T1) helper cocycle is not in Gamma_0(N) (internal (P1) error).");
    }
    // g^{-1} h g = [[h.a, N h.b],[h.c / N, h.d]] for g = diag(1, N): exact
    // and integral BECAUSE N | h.c; in Gamma' subset Gamma ([C, def:tau]).
    const Mat22 conj{ h.a, checked_mul<int64_t>(h.b, Ni), h.c / Ni, h.d };
    if (det_tiered(conj) != 1) {
        throw std::runtime_error("HeckeCosetTables: (T1) conjugate is not unimodular (internal error).");
    }
    class_out = j;
    return conj;
}

// --- general (non-generator-restricted) perm / cocycle ---------------------

template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::perm_of(uint32_t i, const Mat22& gamma) const {
    check_index(i);
    const std::array<uint32_t, 2>& cd = class_repr_[i];
    const std::array<uint32_t, 4> gm = gamma.reduced_mod(N32_); // [a,b,c',d']
    const uint32_t a = gm[0], b = gm[1], cp = gm[2], dp = gm[3];
    // (P1), bottom-row right action: (c:d) -> (c*a + d*c' : c*b + d*d') mod N.
    const uint32_t new_c = ctxN_.add(ctxN_.mul(cd[0], a), ctxN_.mul(cd[1], cp));
    const uint32_t new_d = ctxN_.add(ctxN_.mul(cd[0], b), ctxN_.mul(cd[1], dp));
    return class_index(new_c, new_d);
}

template<class Elt>
Mat22 HeckeCosetTablesT<Elt>::cocycle_of(uint32_t i, const Mat22& gamma) const {
    check_index(i);
    const uint32_t j = perm_of(i, gamma);
    // h_i(gamma) = s_i * gamma * s_{i.gamma}^{-1}; s_{i.gamma} in SL_2(Z),
    // so inv_unimodular() is safe (invariant I7 -- never invert det == N).
    return (s_[i] * gamma) * s_inv_[j];
}

// --- Gamma^0(N) upper-triangular analogues (Stage S5 addition) -------------
//
// Gamma^0(N) = { gamma in Gamma : b == 0 mod N }; its elements have TOP row
// (a, 0 mod N) with a a unit mod N. For h in Gamma^0(N), the top row of
// h*gamma is (a,0)*gamma = a*(top row of gamma) -- a unit multiple, i.e. the
// SAME P^1(Z/N) class -- so (unlike the bottom-row Gamma_0(N) case, which
// uses the class of s_i . gamma directly) the relevant class map input here
// is likewise the class of t_i . gamma, computed as (top row of t_i) acted
// on by gamma from the right, exactly mirroring perm_of's row-vector action
// with (a,b) in place of (c,d). class_index() itself is representation-
// agnostic (just a canonical enumeration of P^1(Z/N) points), so it is
// reused verbatim; only the reverse lookup t_index_of_class_ is new (see the
// header comment on class_repr_upper_ for why it is needed).
template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::perm_upper_of(uint32_t i, const Mat22& gamma) const {
    check_index(i);
    const std::array<uint32_t, 2>& ab = class_repr_upper_[i];
    const std::array<uint32_t, 4> gm = gamma.reduced_mod(N32_); // [a,b,c,d]
    const uint32_t a = gm[0], b = gm[1], c = gm[2], d = gm[3];
    // Top-row right action: (a_i, b_i) -> (a_i*a + b_i*c, a_i*b + b_i*d) mod N.
    const uint32_t new_a = ctxN_.add(ctxN_.mul(ab[0], a), ctxN_.mul(ab[1], c));
    const uint32_t new_b = ctxN_.add(ctxN_.mul(ab[0], b), ctxN_.mul(ab[1], d));
    return t_index_of_class_[class_index(new_a, new_b)];
}

template<class Elt>
Mat22 HeckeCosetTablesT<Elt>::cocycle_upper_of(uint32_t i, const Mat22& gamma) const {
    check_index(i);
    const uint32_t j = perm_upper_of(i, gamma);
    // h_i(gamma) = t_i * gamma * t_{i.gamma}^{-1}; t_{i.gamma} in SL_2(Z),
    // so inv_unimodular() is safe (invariant I7).
    return (t_[i] * gamma) * t_inv_[j];
}

// --- construction ------------------------------------------------------

template<class Elt>
void HeckeCosetTablesT<Elt>::init_common(uint64_t p_hecke, uint32_t r, Elt ell,
                                    uint32_t n_sym) {
    if (p_hecke < 2) {
        throw std::invalid_argument("HeckeCosetTables: p_hecke must be >= 2");
    }
    if (r < 1) {
        throw std::invalid_argument("HeckeCosetTables: r must be >= 1");
    }
    if (ell < 2) {
        throw std::invalid_argument("HeckeCosetTables: ell must be >= 2");
    }

    p_ = p_hecke;
    r_ = r;
    ell_ = ell;
    n_sym_ = n_sym;
    // (slot-14 WP-10) The two levers (default on since the owner decision of
    // 2026-09-28, Q8; each with an opt-out), read once here (thread-local
    // options; HeckeOperatorOnK.h).  Neither can change a value: a streamed
    // table evaluates rho_of(...) of the same exact matrix on demand, and the
    // parallel fill writes each pre-assigned slot with the same rho_of call.
    streamed_ = HeckeOperatorOptions::coset_streaming();
    parallel_setup_ = HeckeOperatorOptions::parallel_setup();

    // N = p^r, overflow-checked, and must additionally fit the uint32_t
    // class-map domain used by class_index/perm/Mat22::reduced_mod (this is
    // an explicit design constraint of the declared interface, not merely a
    // DoD-range convenience: class_index takes uint32_t c_mod_N/d_mod_N).
    uint64_t N = 1;
    for (uint32_t k = 0; k < r; ++k) {
        if (N > UINT64_MAX / p_hecke) {
            throw std::overflow_error("HeckeCosetTables: N = p^r overflows uint64_t");
        }
        N *= p_hecke;
    }
    if (N > static_cast<uint64_t>(UINT32_MAX)) {
        throw std::overflow_error(
            "HeckeCosetTables: N = p^r exceeds the uint32_t class-map domain");
    }
    if (p_hecke > static_cast<uint64_t>(UINT32_MAX)) {
        throw std::overflow_error("HeckeCosetTables: p_hecke exceeds the uint32_t domain");
    }

    N_ = N;
    N32_ = static_cast<uint32_t>(N_);
    ctxN_ = engine::ModCtx<uint32_t>(N32_); // the level's modulus context (see the header)
    p32_ = static_cast<uint32_t>(p_);

    const uint64_t p_pow_rm1 = N_ / p_; // p^{r-1}, exact (N = p^r)
    m_ = static_cast<uint32_t>(N_ + p_pow_rm1);

    g_ = Mat22{ 1, 0, 0, static_cast<int64_t>(N_) };

    class_repr_.assign(static_cast<size_t>(m_) + 1, std::array<uint32_t, 2>{ 0, 0 });
    s_.assign(static_cast<size_t>(m_) + 1, Mat22::I());
    t_.assign(static_cast<size_t>(m_) + 1, Mat22::I());

    // Branch A: class (k:1), k = 0 .. N-1, i = k + 1. Closed-form SL_2(Z)
    // lift s_i = [[1,0],[k,1]] (det == 1*1 - 0*k == 1); specializes to
    // s_1 = I (k = 0) and [C, sec:heckedatum]'s s_{1+k} = [[1,0],[k,1]] for
    // r = 1 exactly (see the class-level header comment for why this direct
    // closed form is used instead of a generic extended-gcd Bezout solve).
    for (uint64_t k = 0; k < N_; ++k) {
        const uint32_t i = static_cast<uint32_t>(k) + 1;
        class_repr_[i] = { static_cast<uint32_t>(k), 1u };
        s_[i] = Mat22{ 1, 0, static_cast<int64_t>(k), 1 };
        t_[i] = Mat22{ 1, static_cast<int64_t>(k), 0, 1 }; // T^k
    }
    // Branch B: class (1:p*c), c = 0 .. p^{r-1}-1, i = N + c + 1. Closed-form
    // lift s_i = [[0,-1],[1,p*c]] (det == 0*(p*c) - (-1)*1 == 1);
    // specializes to s_{p+1} = [[0,-1],[1,0]] = S for r = 1, c = 0 exactly.
    for (uint64_t c = 0; c < p_pow_rm1; ++c) {
        const uint32_t i = N32_ + static_cast<uint32_t>(c) + 1;
        const int64_t pc = static_cast<int64_t>(p_) * static_cast<int64_t>(c);
        class_repr_[i] = { 1u, static_cast<uint32_t>(pc % static_cast<int64_t>(N_)) };
        s_[i] = Mat22{ 0, -1, 1, pc };
        t_[i] = Mat22{ -pc, -1, 1, 0 };
    }

    // Constructor-time internal consistency asserts (I8: certification, not
    // geometry/heuristics) -- every constructed s_i really is an SL_2(Z)
    // lift of its assigned class.
    for (uint32_t i = 1; i <= m_; ++i) {
        if (s_[i].det() != 1) {
            throw std::runtime_error("HeckeCosetTables: internal error, s_i has det != 1");
        }
        const std::array<uint32_t, 4> rm = s_[i].reduced_mod(N32_);
        if (rm[2] != class_repr_[i][0] || rm[3] != class_repr_[i][1]) {
            throw std::runtime_error("HeckeCosetTables: internal error, s_i bottom row "
                                      "does not match its assigned class");
        }
        if (t_[i].det() != 1) {
            throw std::runtime_error("HeckeCosetTables: internal error, t_i has det != 1");
        }
    }

#if OC_ENABLE_LEGACY
    materialise_rho_t();   // (S24 CL-R1) the materialised table: ON builds, HECKE_COSET_STREAM=0 only
#endif

    // --- Gamma^0(N) upper-triangular coset data (Stage S5 addition) --------
    // t_ is fixed data (like rho_t_ above), independent of the s_ rebuilt by
    // apply_custom_reps(), so this block runs exactly once here.
    t_inv_.assign(static_cast<size_t>(m_) + 1, Mat22::I());
    class_repr_upper_.assign(static_cast<size_t>(m_) + 1, std::array<uint32_t, 2>{ 0, 0 });
    t_index_of_class_.assign(static_cast<size_t>(m_) + 1, 0);
    for (uint32_t i = 1; i <= m_; ++i) {
        t_inv_[i] = t_[i].inv_unimodular(); // t_i in SL_2(Z): safe (I7)
        const std::array<uint32_t, 4> tm = t_[i].reduced_mod(N32_);
        class_repr_upper_[i] = { tm[0], tm[1] }; // native top row (a,b) mod N
        const uint32_t cidx = class_index(tm[0], tm[1]);
        if (cidx == 0 || cidx > m_ || t_index_of_class_[cidx] != 0) {
            throw std::runtime_error(
                "HeckeCosetTables: internal error, Gamma^0(N) representatives "
                "t_i do not biject onto P^1(Z/N) (duplicate or out-of-range class)");
        }
        t_index_of_class_[cidx] = i;
    }
    for (uint32_t cidx = 1; cidx <= m_; ++cidx) {
        if (t_index_of_class_[cidx] == 0) {
            throw std::runtime_error(
                "HeckeCosetTables: internal error, Gamma^0(N) representatives "
                "t_i do not cover all of P^1(Z/N)");
        }
    }

    perm_upper_.assign(4, std::vector<uint32_t>(static_cast<size_t>(m_) + 1, 0));
    cocycle_upper_.assign(4, std::vector<Mat22>(static_cast<size_t>(m_) + 1, Mat22::I()));
    for (uint32_t gen = 0; gen < 4; ++gen) {
        const Mat22 gm = generator_matrix(gen);
        for (uint32_t i = 1; i <= m_; ++i) {
            const uint32_t j = perm_upper_of(i, gm);
            // h_i(gamma) = t_i * gamma * t_{i.gamma}^{-1}; t_{i.gamma} in
            // SL_2(Z), so inv_unimodular() is safe (invariant I7).
            const Mat22 h = (t_[i] * gm) * t_inv_[j];
            perm_upper_[gen][i] = j;
            cocycle_upper_[gen][i] = h;
        }
    }
#if OC_ENABLE_LEGACY
    materialise_rho_cocycle_upper();   // (S24 CL-R1) ON builds, HECKE_COSET_STREAM=0 only
#endif

    // perm_ is representative-independent (depends only on class_repr_, not
    // on which SL_2(Z) lift is stored in s_), so it is computed once here
    // and never touched by apply_custom_reps().
    perm_.assign(4, std::vector<uint32_t>(static_cast<size_t>(m_) + 1, 0));
    for (uint32_t gen = 0; gen < 4; ++gen) {
        const Mat22 gm = generator_matrix(gen);
        for (uint32_t i = 1; i <= m_; ++i) {
            perm_[gen][i] = perm_of(i, gm);
        }
    }

    s_inv_.assign(static_cast<size_t>(m_) + 1, Mat22::I());
    delta_.assign(static_cast<size_t>(m_) + 1, Mat22::I());
#if OC_ENABLE_LEGACY
    rho_s_inv_.assign(static_cast<size_t>(m_) + 1, FlatMatrixT<Elt>());
    rho_cocycle_.assign(4, std::vector<FlatMatrixT<Elt>>(static_cast<size_t>(m_) + 1));
#endif
    cocycle_.assign(4, std::vector<Mat22>(static_cast<size_t>(m_) + 1, Mat22::I()));

    rebuild_from_s(); // fills s_inv_, delta_, rho_s_inv_, cocycle_, rho_cocycle_ from s_
}

template<class Elt>
void HeckeCosetTablesT<Elt>::rebuild_from_s() {
    for (uint32_t i = 1; i <= m_; ++i) {
        s_inv_[i] = s_[i].inv_unimodular(); // s_i in SL_2(Z): safe (I7)
        delta_[i] = s_inv_[i] * g_;          // delta_i = s_i^{-1} g ([C, prop:cosets])
    }
    for (uint32_t gen = 0; gen < 4; ++gen) {
        const Mat22 gm = generator_matrix(gen);
        for (uint32_t i = 1; i <= m_; ++i) {
            const uint32_t j = perm_[gen][i];
            const Mat22 h = (s_[i] * gm) * s_inv_[j]; // h_i(gamma) = s_i gamma s_{i.gamma}^{-1}
            cocycle_[gen][i] = h;
        }
    }
#if OC_ENABLE_LEGACY
    materialise_rho_s_inv_cocycle();   // (S24 CL-R1) ON builds, HECKE_COSET_STREAM=0 only
#endif
}

template<class Elt>
HeckeCosetTablesT<Elt>::HeckeCosetTablesT(uint64_t p_hecke, uint32_t r, Elt ell,
                                    uint32_t n_sym) {
    init_common(p_hecke, r, ell, n_sym);
}

template<class Elt>
void HeckeCosetTablesT<Elt>::apply_custom_reps(const std::vector<Mat22>& custom_s) {
    if (custom_s.size() != m_) {
        throw std::invalid_argument(
            "HeckeCosetTables: custom_s.size() must equal num_cosets(); "
            "custom_s[i-1] is the caller-supplied representative for coset i");
    }
    for (uint32_t i = 1; i <= m_; ++i) {
        const Mat22& cand = custom_s[i - 1];
        if (cand.det() != 1) {
            throw std::invalid_argument(
                "HeckeCosetTables: custom representative has det != 1 "
                "(must be an SL_2(Z) element)");
        }
        if (class_index_of(cand) != i) {
            throw std::invalid_argument(
                "HeckeCosetTables: custom representative is not in the "
                "expected Gamma_0(N) coset (O12(b) requires the SAME "
                "classes as the canonical representatives)");
        }
        s_[i] = cand;
    }
    rebuild_from_s();
}

template<class Elt>
HeckeCosetTablesT<Elt>::HeckeCosetTablesT(uint64_t p_hecke, uint32_t r, Elt ell,
                                    uint32_t n_sym, const std::vector<Mat22>& custom_s) {
    init_common(p_hecke, r, ell, n_sym); // canonical build first (needed for class_index_of, t_/rho_t_)
    apply_custom_reps(custom_s);
}

// --- public accessors --------------------------------------------------

template<class Elt>
const Mat22& HeckeCosetTablesT<Elt>::s(uint32_t i) const { check_index(i); return s_[i]; }
template<class Elt>
const Mat22& HeckeCosetTablesT<Elt>::s_inv(uint32_t i) const { check_index(i); return s_inv_[i]; }
template<class Elt>
const Mat22& HeckeCosetTablesT<Elt>::delta(uint32_t i) const { check_index(i); return delta_[i]; }

template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::perm(uint32_t gen, uint32_t i) const {
    check_gen(gen); check_index(i);
    return perm_[gen][i];
}
template<class Elt>
const Mat22& HeckeCosetTablesT<Elt>::cocycle(uint32_t gen, uint32_t i) const {
    check_gen(gen); check_index(i);
    return cocycle_[gen][i];
}

template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::perm_upper(uint32_t gen, uint32_t i) const {
    check_gen(gen); check_index(i);
    return perm_upper_[gen][i];
}

// (slot-14 WP-10) The block accessors: borrowed on a default table, evaluated
// by rho_of on a streamed one (the function that fills the default table).
template<class Elt>
CosetRhoT<Elt> HeckeCosetTablesT<Elt>::rho_cocycle_block(uint32_t gen, uint32_t i) const {
    check_gen(gen); check_index(i);
#if OC_ENABLE_LEGACY
    if (!streamed_) return CosetRhoT<Elt>(&rho_cocycle_[gen][i]);
#endif
    return CosetRhoT<Elt>(rho_of(cocycle_[gen][i]));
}
template<class Elt>
CosetRhoT<Elt> HeckeCosetTablesT<Elt>::rho_s_inv_block(uint32_t i) const {
    check_index(i);
#if OC_ENABLE_LEGACY
    if (!streamed_) return CosetRhoT<Elt>(&rho_s_inv_[i]);
#endif
    return CosetRhoT<Elt>(rho_of(s_inv_[i]));
}
template<class Elt>
CosetRhoT<Elt> HeckeCosetTablesT<Elt>::rho_t_block(uint32_t i) const {
    check_index(i);
#if OC_ENABLE_LEGACY
    if (!streamed_) return CosetRhoT<Elt>(&rho_t_[i]);
#endif
    return CosetRhoT<Elt>(rho_of(t_[i]));
}
template<class Elt>
CosetRhoT<Elt> HeckeCosetTablesT<Elt>::rho_cocycle_upper_block(uint32_t gen, uint32_t i) const {
    check_gen(gen); check_index(i);
#if OC_ENABLE_LEGACY
    if (!streamed_) return CosetRhoT<Elt>(&rho_cocycle_upper_[gen][i]);
#endif
    return CosetRhoT<Elt>(rho_of(cocycle_upper_[gen][i]));
}

// --- self_check(): the S3 Definition-of-Done battery -----------------------

template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::self_check() const {
    // (i) coset count.
    if (static_cast<uint64_t>(m_) != N_ + N_ / p_) return 1;

    const Mat22 gens[4] = {
        generator_matrix(0), generator_matrix(1), generator_matrix(2), generator_matrix(3)
    };

    // (ii) every h_i(gamma), gamma in {T,S,ST,TS}: det == 1, lower-left == 0 mod N.
    for (uint32_t gen = 0; gen < 4; ++gen) {
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22& h = cocycle_[gen][i];
            if (h.det() != 1) return 2;
            const std::array<uint32_t, 4> hm = h.reduced_mod(N32_);
            if (hm[2] != 0) return 3; // lower-left entry
        }
    }

    // (v) reconstruction: s_i * gamma == h_i(gamma) * s_{i.gamma}.
    for (uint32_t gen = 0; gen < 4; ++gen) {
        for (uint32_t i = 1; i <= m_; ++i) {
            const uint32_t j = perm_[gen][i];
            const Mat22 lhs = s_[i] * gens[gen];
            const Mat22 rhs = cocycle_[gen][i] * s_[j];
            if (!(lhs == rhs)) return 4;
        }
    }

    // (iii) chain rule + (iv) rho_n multiplicativity, all 16 ordered pairs
    // from the generator set G = {T,S,ST,TS}. h_direct/perm_direct are
    // recomputed via the GENERAL (non-cached) perm_of/cocycle_of on the
    // exact composite matrix gens[g1]*gens[g2], so this is a genuine
    // cross-check against the generator-indexed caches, not a tautology.
    for (uint32_t g1 = 0; g1 < 4; ++g1) {
        for (uint32_t g2 = 0; g2 < 4; ++g2) {
            const Mat22 prod = gens[g1] * gens[g2];
            for (uint32_t i = 1; i <= m_; ++i) {
                const uint32_t j = perm_[g1][i];
                const uint32_t k_direct = perm_of(i, prod);
                const uint32_t k_chain = perm_[g2][j];
                if (k_direct != k_chain) return 5;

                const Mat22 h_direct = cocycle_of(i, prod);
                const Mat22 h_chain = cocycle_[g1][i] * cocycle_[g2][j];
                if (!(h_direct == h_chain)) return 6;

                // (iv): rho_n multiplicativity spot check on the same pair.
                const FlatMatrixT<Elt> rho_direct = rho_of(h_direct);
                const FlatMatrixT<Elt> rho_chain =
                    rho_cocycle_block(g1, i).get() * rho_cocycle_block(g2, j).get();
                if (!(rho_direct == rho_chain)) return 7;
            }
        }
    }

    // (iii cont'd): 100 fixed-seed random length-<=6 words gamma1, gamma2
    // (each itself a random word in G), same two identities, via the
    // GENERAL perm_of/cocycle_of (gamma1, gamma2 need not be single
    // generators here).
    XorShift64 rng(0x53335343'6F736574ULL); // fixed seed ("S3CosetTables"-ish), deterministic
    for (int trial = 0; trial < 100; ++trial) {
        const int len1 = 1 + static_cast<int>(rng.below(6));
        Mat22 w1 = Mat22::I();
        for (int t = 0; t < len1; ++t) w1 = w1 * gens[rng.below(4)];

        const int len2 = 1 + static_cast<int>(rng.below(6));
        Mat22 w2 = Mat22::I();
        for (int t = 0; t < len2; ++t) w2 = w2 * gens[rng.below(4)];

        const Mat22 prod = w1 * w2;
        for (uint32_t i = 1; i <= m_; ++i) {
            const uint32_t j = perm_of(i, w1);
            const uint32_t k_direct = perm_of(i, prod);
            const uint32_t k_chain = perm_of(j, w2);
            if (k_direct != k_chain) return 8;

            const Mat22 h_direct = cocycle_of(i, prod);
            const Mat22 h1 = cocycle_of(i, w1);
            const Mat22 h2 = cocycle_of(j, w2);
            if (!(h_direct == (h1 * h2))) return 9;
        }
    }

    return 0;
}

// --- self_check_upper(): Stage S5 addition, Gamma^0(N) DoD battery ---------
// Mirrors self_check() (i)-(v) above exactly, for the NEW t_i /
// perm_upper_ / cocycle_upper_ / rho_cocycle_upper_ tables.
template<class Elt>
uint32_t HeckeCosetTablesT<Elt>::self_check_upper() const {
    const Mat22 gens[4] = {
        generator_matrix(0), generator_matrix(1), generator_matrix(2), generator_matrix(3)
    };

    // (ii) every h_i(gamma), gamma in {T,S,ST,TS}: det == 1, upper-right == 0 mod N.
    for (uint32_t gen = 0; gen < 4; ++gen) {
        for (uint32_t i = 1; i <= m_; ++i) {
            const Mat22& h = cocycle_upper_[gen][i];
            if (h.det() != 1) return 1;
            const std::array<uint32_t, 4> hm = h.reduced_mod(N32_);
            if (hm[1] != 0) return 2; // upper-right entry
        }
    }

    // (v) reconstruction: t_i * gamma == h_i(gamma) * t_{i.gamma}.
    for (uint32_t gen = 0; gen < 4; ++gen) {
        for (uint32_t i = 1; i <= m_; ++i) {
            const uint32_t j = perm_upper_[gen][i];
            const Mat22 lhs = t_[i] * gens[gen];
            const Mat22 rhs = cocycle_upper_[gen][i] * t_[j];
            if (!(lhs == rhs)) return 3;
        }
    }

    // (iii) chain rule + (iv) rho_n multiplicativity, all 16 ordered pairs
    // from G = {T,S,ST,TS}, via the GENERAL (non-cached) perm_upper_of/
    // cocycle_upper_of on the exact composite matrix gens[g1]*gens[g2].
    for (uint32_t g1 = 0; g1 < 4; ++g1) {
        for (uint32_t g2 = 0; g2 < 4; ++g2) {
            const Mat22 prod = gens[g1] * gens[g2];
            for (uint32_t i = 1; i <= m_; ++i) {
                const uint32_t j = perm_upper_[g1][i];
                const uint32_t k_direct = perm_upper_of(i, prod);
                const uint32_t k_chain = perm_upper_[g2][j];
                if (k_direct != k_chain) return 5;

                const Mat22 h_direct = cocycle_upper_of(i, prod);
                const Mat22 h_chain = cocycle_upper_[g1][i] * cocycle_upper_[g2][j];
                if (!(h_direct == h_chain)) return 6;

                const FlatMatrixT<Elt> rho_direct = rho_of(h_direct);
                const FlatMatrixT<Elt> rho_chain =
                    rho_cocycle_upper_block(g1, i).get() * rho_cocycle_upper_block(g2, j).get();
                if (!(rho_direct == rho_chain)) return 7;
            }
        }
    }

    // (iii cont'd): 100 fixed-seed random length-<=6 words (a DIFFERENT seed
    // from self_check()'s, so this is not merely replaying the same words).
    XorShift64 rng(0x53335343'557070ULL); // fixed seed ("S3CosetTablesUpp"-ish), deterministic
    for (int trial = 0; trial < 100; ++trial) {
        const int len1 = 1 + static_cast<int>(rng.below(6));
        Mat22 w1 = Mat22::I();
        for (int t = 0; t < len1; ++t) w1 = w1 * gens[rng.below(4)];

        const int len2 = 1 + static_cast<int>(rng.below(6));
        Mat22 w2 = Mat22::I();
        for (int t = 0; t < len2; ++t) w2 = w2 * gens[rng.below(4)];

        const Mat22 prod = w1 * w2;
        for (uint32_t i = 1; i <= m_; ++i) {
            const uint32_t j = perm_upper_of(i, w1);
            const uint32_t k_direct = perm_upper_of(i, prod);
            const uint32_t k_chain = perm_upper_of(j, w2);
            if (k_direct != k_chain) return 8;

            const Mat22 h_direct = cocycle_upper_of(i, prod);
            const Mat22 h1 = cocycle_upper_of(i, w1);
            const Mat22 h2 = cocycle_upper_of(j, w2);
            if (!(h_direct == (h1 * h2))) return 9;
        }
    }

    return 0;
}

#if OC_ENABLE_LEGACY
#include "legacy/HeckeCosetTablesMaterialised.inc"   // (S24 CL-R1) the materialised rho_n tables (HECKE_COSET_STREAM=0)
#endif

// Stage S13 explicit instantiations.
template class HeckeCosetTablesT<uint32_t>;
template class HeckeCosetTablesT<uint64_t>;

} // namespace hecke
