// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S1 -- Exact bounded-integer layer (see ExactInt.h for the contract
// references: [C, sec:reduction], [C, thm:backend], [C, obl:backend]).

#include "ExactInt.h"

#include "EngineCheck.h" // ENGINE_DEBUG_CHECK (NDEBUG policy)

#include <cstdio>
#include <cstdlib>
#include <string>

namespace hecke {

// --- Mat22 ------------------------------------------------------------------

Mat22 Mat22::I()  { return Mat22{ 1, 0, 0, 1 }; }
Mat22 Mat22::T()  { return Mat22{ 1, 1, 0, 1 }; }
Mat22 Mat22::S()  { return Mat22{ 0, -1, 1, 0 }; }
Mat22 Mat22::ST() { return Mat22{ 0, -1, 1, 1 }; }  // S*T
Mat22 Mat22::TS() { return Mat22{ 1, -1, 1, 0 }; }  // T*S

Mat22 Mat22::operator*(const Mat22& rhs) const {
    // Each entry is a 2-term dot product; every multiply/add is
    // overflow-checked and throws HeckeOverflow ([C, obl:backend]).
    return Mat22{
        checked_add(checked_mul(a, rhs.a), checked_mul(b, rhs.c)),
        checked_add(checked_mul(a, rhs.b), checked_mul(b, rhs.d)),
        checked_add(checked_mul(c, rhs.a), checked_mul(d, rhs.c)),
        checked_add(checked_mul(c, rhs.b), checked_mul(d, rhs.d))
    };
}

Mat22 Mat22::inv_unimodular() const {
    const int64_t dt = det();
    // Debug-only: the ALWAYS-ON guard for this precondition is the
    // `throw HeckeOverflow` below, which every configuration reaches. Kept as
    // a debug check so the failure surfaces at the assertion site too.
    ENGINE_DEBUG_CHECK_MSG(dt == 1 || dt == -1, "Mat22::inv_unimodular requires det == +-1");
    if (dt == 1) {
        return Mat22{ d, checked_neg(b), checked_neg(c), a };
    }
    if (dt == -1) {
        return Mat22{ checked_neg(d), b, c, checked_neg(a) };
    }
    // Hard guard in release builds too: abort rather than corrupt.
    throw HeckeOverflow("Mat22::inv_unimodular: det != +-1");
}

int64_t Mat22::det() const {
    return checked_sub(checked_mul(a, d), checked_mul(b, c));
}

int64_t Mat22::max_abs() const {
    const int64_t entries[4] = { a, b, c, d };
    uint64_t m = 0;
    for (int64_t e : entries) {
        const uint64_t ae = uabs64(e);
        if (ae > m) m = ae;
    }
    if (m > static_cast<uint64_t>(INT64_MAX)) {
        // Only possible for an INT64_MIN entry.
        throw HeckeOverflow("Mat22::max_abs: |entry| not representable as int64_t");
    }
    return static_cast<int64_t>(m);
}

bool Mat22::operator==(const Mat22& rhs) const {
    return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d;
}

Mat22 Mat22::neg() const {
    return Mat22{ checked_neg(a), checked_neg(b), checked_neg(c), checked_neg(d) };
}

Mat22 Mat22::adj() const {
    // [C, sec:contragredient]: adj [[a,b],[c,d]] = [[d,-b],[-c,a]].
    return Mat22{ d, checked_neg(b), checked_neg(c), a };
}

std::array<uint32_t, 4> Mat22::reduced_mod(uint32_t m) const {
    // Debug-only: the always-on guard is the `throw HeckeOverflow` below.
    ENGINE_DEBUG_CHECK_MSG(m > 0, "Mat22::reduced_mod requires m > 0");
    if (m == 0) throw HeckeOverflow("Mat22::reduced_mod: modulus is 0");
    const int64_t mm = static_cast<int64_t>(m);
    const int64_t entries[4] = { a, b, c, d };
    std::array<uint32_t, 4> out{};
    for (int i = 0; i < 4; ++i) {
        int64_t r = entries[i] % mm; // C++ truncation: r in (-m, m)
        if (r < 0) r += mm;          // canonical representative in [0, m)
        out[static_cast<size_t>(i)] = static_cast<uint32_t>(r);
    }
    return out;
}

std::array<uint64_t, 4> Mat22::reduced_mod64(uint64_t m) const {
    if (m == 0) throw HeckeOverflow("Mat22::reduced_mod64: modulus is 0");
    const int64_t entries[4] = { a, b, c, d };
    std::array<uint64_t, 4> out{};
    for (int i = 0; i < 4; ++i) {
        const int64_t e = entries[i];
        uint64_t r;
        if (e >= 0) {
            r = static_cast<uint64_t>(e) % m;
        } else {
            // |e| as uint64 (safe for INT64_MIN via two's-complement magnitude).
            const uint64_t pos = (e == INT64_MIN)
                ? (static_cast<uint64_t>(INT64_MAX) + 1ull)
                : static_cast<uint64_t>(-e);
            const uint64_t rem = pos % m;
            r = (rem == 0) ? 0ull : (m - rem);
        }
        out[static_cast<size_t>(i)] = r;
    }
    return out;
}

// --- Mat22W -----------------------------------------------------------------

Mat22W Mat22W::I() { return Mat22W{ 1, 0, 0, 1 }; }

Mat22W Mat22W::from(const Mat22& m) {
    return Mat22W{ m.a, m.b, m.c, m.d };
}

Mat22W Mat22W::operator*(const Mat22W& rhs) const {
    return Mat22W{
        checked_add(checked_mul(a, rhs.a), checked_mul(b, rhs.c)),
        checked_add(checked_mul(a, rhs.b), checked_mul(b, rhs.d)),
        checked_add(checked_mul(c, rhs.a), checked_mul(d, rhs.c)),
        checked_add(checked_mul(c, rhs.b), checked_mul(d, rhs.d))
    };
}

Mat22W Mat22W::inv_unimodular() const {
    const __int128 dt = det();
    // Debug-only: the ALWAYS-ON guard for this precondition is the
    // `throw HeckeOverflow` below, which every configuration reaches. Kept as
    // a debug check so the failure surfaces at the assertion site too.
    ENGINE_DEBUG_CHECK_MSG(dt == 1 || dt == -1, "Mat22W::inv_unimodular requires det == +-1");
    if (dt == 1) {
        return Mat22W{ d, checked_neg(b), checked_neg(c), a };
    }
    if (dt == -1) {
        return Mat22W{ checked_neg(d), b, c, checked_neg(a) };
    }
    throw HeckeOverflow("Mat22W::inv_unimodular: det != +-1");
}

__int128 Mat22W::det() const {
    return checked_sub(checked_mul(a, d), checked_mul(b, c));
}

__int128 Mat22W::max_abs() const {
    const __int128 entries[4] = { a, b, c, d };
    __int128 m = 0;
    for (__int128 e : entries) {
        // checked_neg throws for INT128_MIN (magnitude not representable).
        const __int128 ae = (e < 0) ? checked_neg(e) : e;
        if (ae > m) m = ae;
    }
    return m;
}

bool Mat22W::operator==(const Mat22W& rhs) const {
    return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d;
}

Mat22W Mat22W::neg() const {
    return Mat22W{ checked_neg(a), checked_neg(b), checked_neg(c), checked_neg(d) };
}

std::array<uint32_t, 4> Mat22W::reduced_mod(uint32_t m) const {
    // Debug-only: the always-on guard is the `throw HeckeOverflow` below.
    ENGINE_DEBUG_CHECK_MSG(m > 0, "Mat22W::reduced_mod requires m > 0");
    if (m == 0) throw HeckeOverflow("Mat22W::reduced_mod: modulus is 0");
    const __int128 mm = static_cast<__int128>(m);
    const __int128 entries[4] = { a, b, c, d };
    std::array<uint32_t, 4> out{};
    for (int i = 0; i < 4; ++i) {
        __int128 r = entries[i] % mm;
        if (r < 0) r += mm;
        out[static_cast<size_t>(i)] = static_cast<uint32_t>(r);
    }
    return out;
}

// --- The exact-tier switch (slot 3 WP-1; see ExactInt.h) ---------------------

namespace {
thread_local ExactTier g_exact_tier = ExactTier::Auto;
thread_local uint64_t g_exact_tier_escalations = 0;
} // namespace

ExactTier current_exact_tier() { return g_exact_tier; }
void set_exact_tier(ExactTier t) { g_exact_tier = t; }
uint64_t exact_tier_escalations() { return g_exact_tier_escalations; }
void reset_exact_tier_escalations() { g_exact_tier_escalations = 0; }
void note_exact_tier_escalation() { ++g_exact_tier_escalations; }

// --- The stored wide tier (slot-24 ELL-W; see ExactInt.h) --------------------

namespace {
thread_local uint64_t g_wide_product_escalations = 0;

struct TauWideLever {
    bool enabled = true;
    std::string line;
    TauWideLever() {
        const char* v = std::getenv("HECKE_TAU_WIDE");
        enabled = !(v != nullptr && std::string(v) == "0");
        // (ELL-W audit F2, the HECKE_RHO_SIMD precedent) "0" = rollback, "1" = on; anything else is named and stays on.
        if (v != nullptr && std::string(v) != "0" && std::string(v) != "1")
            std::fprintf(stderr, "WARNING HECKE_TAU_WIDE=%s is neither 0 nor 1: the wide tier stays on\n", v);
        line = std::string("EFFECTIVE HeckeTauWide wide_tier=") + (enabled ? "on" : "off") +
               " HECKE_TAU_WIDE=" + (v ? std::string(v) : std::string("unset"));
    }
};
const TauWideLever& tau_wide_lever() {
    static const TauWideLever lever;   // read once per process (thread-safe static init)
    return lever;
}

int64_t narrow_entry(__int128 x) {
    if (x > static_cast<__int128>(INT64_MAX) || x < static_cast<__int128>(INT64_MIN))
        throw HeckeOverflow("hecke: wide-tier result exceeds the int64 storage tier");
    return static_cast<int64_t>(x);
}

// The wide tier may run (else the narrow HeckeOverflow propagates).
bool wide_allowed() { return tau_wide_lever().enabled && current_exact_tier() != ExactTier::ForceNarrow; }
bool wide_forced() { return tau_wide_lever().enabled && current_exact_tier() == ExactTier::ForceWide; }

Mat22 mul_wide(const Mat22& A, const Mat22& B) {
    const Mat22W w = Mat22W::from(A) * Mat22W::from(B);   // exact: < 2^123 on stored-class operands under the wide clause; checked
    return Mat22{ narrow_entry(w.a), narrow_entry(w.b), narrow_entry(w.c), narrow_entry(w.d) };
}
int64_t det_wide(const Mat22& A) { return narrow_entry(Mat22W::from(A).det()); }
void print_tau_wide_effective_at_exit() {
    std::fprintf(stdout, "%s\n", tau_wide_lever().line.c_str());
    std::fflush(stdout);
}
// (ELL-W audit F2) registered at load: a SET HECKE_TAU_WIDE always prints its EFFECTIVE line at exit, in every run
// (bench, campaign, census shard), so an ignored or mistyped value is visible.  Reporting only; no value depends on it.
[[maybe_unused]] const bool g_tau_wide_noted_at_load = [] {
    if (std::getenv("HECKE_TAU_WIDE") != nullptr) {
        (void)tau_wide_lever();   // the WARNING (if any) at load, not at the first wide table
        std::atexit(&print_tau_wide_effective_at_exit);
    }
    return true;
}();
} // namespace

uint32_t tau_bound_clause(uint64_t N, unsigned __int128 B, bool wide_allowed) {
    // [C, lem:bounds](5),(7), [C, thm:backend](1),(4), [C, obl:backend](ii).  Narrow: 2 B^2 < 2^63 <=> B <= 2^31 - 1.
    if (B <= static_cast<unsigned __int128>(2147483647ull)) return 0;
    if (!wide_allowed) return 51;
    // Wide: 2 N B <= 2^63 - 1.  N < 2^64 and B < 2^128 would overflow the product; but B > 2^63 alone already fails
    // (N >= 1), so test that first and form 2 N B only for B <= 2^63, where it is < 2^(1 + 64 + 63) -- exact.
    if (B > static_cast<unsigned __int128>(INT64_MAX)) return 52;
    const unsigned __int128 two_N_B = static_cast<unsigned __int128>(2) * N * B;
    return two_N_B > static_cast<unsigned __int128>(INT64_MAX) ? 52 : 0;
}

bool tau_wide_tier_enabled() { return tau_wide_lever().enabled; }
const char* tau_wide_tier_effective_line() { return tau_wide_lever().line.c_str(); }
uint64_t wide_product_escalations() { return g_wide_product_escalations; }
void reset_wide_product_escalations() { g_wide_product_escalations = 0; }

Mat22 mul_tiered(const Mat22& A, const Mat22& B) {
    if (wide_forced()) return mul_wide(A, B);
    try {
        return A * B;   // the narrow tier, verbatim
    } catch (const HeckeOverflow&) {
        if (!wide_allowed()) throw;
    }
    ++g_wide_product_escalations;
    return mul_wide(A, B);
}

int64_t det_tiered(const Mat22& A) {
    if (wide_forced()) return det_wide(A);
    try {
        return A.det();
    } catch (const HeckeOverflow&) {
        if (!wide_allowed()) throw;
    }
    ++g_wide_product_escalations;
    return det_wide(A);
}

Mat22 inv_unimodular_tiered(const Mat22& A) {
    const int64_t dt = det_tiered(A);
    // (ELL-W audit F8) parity with Mat22::inv_unimodular: the debug check at the assertion site, the throw below always.
    ENGINE_DEBUG_CHECK_MSG(dt == 1 || dt == -1, "inv_unimodular_tiered requires det == +-1");
    // Same branches as Mat22::inv_unimodular (entries are A's own, negated).
    if (dt == 1) return Mat22{ A.d, checked_neg(A.b), checked_neg(A.c), A.a };
    if (dt == -1) return Mat22{ checked_neg(A.d), A.b, A.c, checked_neg(A.a) };
    throw HeckeOverflow("hecke::inv_unimodular_tiered: det != +-1");
}

// --- GroupElt ---------------------------------------------------------------

GroupElt GroupElt::from_exact(const Mat22& g, uint32_t ell, uint32_t N) {
    GroupElt e;
    e.exact = g;
    e.mod_ell = g.reduced_mod(ell);
    e.mod_N = g.reduced_mod(N);
    return e;
}

} // namespace hecke
