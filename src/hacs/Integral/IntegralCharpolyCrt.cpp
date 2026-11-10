// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// IntegralCharpolyCrt -- Line 2, M-0 (slot 8, WP-7). See the header for the
// statement, the certificate and the "nothing new" discipline.

#include "IntegralCharpolyCrt.h"

#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ScalarMod.h"

#include <cstring>
#include <utility>

namespace integral {

namespace {

using u128 = unsigned __int128;

// Number of bits of x (0 for x = 0).
uint32_t BitLen(u128 x) {
    uint32_t b = 0;
    while (x) { ++b; x >>= 1; }
    return b;
}

uint64_t CeilDiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// (S24 CL-G3, C-07) the shared scalar helpers (ScalarMod.h); m < 2^32 here, where they equal the old narrow copies.
uint64_t MulMod(uint64_t a, uint64_t b, uint64_t m) { return engine::scalar::mulmod(a, b, m); }
uint64_t PowMod(uint64_t base, uint64_t e, uint64_t m) { return engine::scalar::powmod<uint64_t>(base, e, m); }

// Decimal digit count of the magnitude of a signed decimal string (validated).
uint32_t DecimalMagnitudeDigits(const char* s) {
    ENGINE_CHECK_MSG(s != nullptr && *s != '\0', "IntegralCharpolyCrt: empty decimal string");
    if (*s == '-' || *s == '+') ++s;
    uint32_t d = 0;
    for (; *s; ++s) {
        ENGINE_CHECK_MSG(*s >= '0' && *s <= '9', "IntegralCharpolyCrt: non-digit in decimal string");
        ++d;
    }
    ENGINE_CHECK_MSG(d > 0, "IntegralCharpolyCrt: sign without digits");
    return d;
}

} // namespace

// ---------------------------------------------------------------------------
// O5, frozen. [IB] Sec. 3.4, verbatim in the `verbatim` column:
//
//   | n  | k  | char(T_2 | Lambda^1)       |
//   | 10 | 12 | (x - 2049)(x + 24)^2       |
//   | 12 | 14 | x - 8193                   |
//   | 14 | 16 | (x - 32769)(x - 216)^2     |
//   | 16 | 18 | (x - 131073)(x + 528)^2    |
//   | 18 | 20 | (x - 524289)(x - 456)^2    |
//   | 20 | 22 | (x - 2097153)(x + 288)^2   |
//   and at n = 10: char(T_3 | Lambda^1) = (x - 177148)(x - 252)^2.
//
// probe_rank_h1: docs/implementation/integral_line_probe.py 10,12,14,16,18,20 2
// prints H^1 = Z^3, Z^1, Z^3, Z^3, Z^3, Z^3 (+ torsion) for these n.
// ---------------------------------------------------------------------------
const std::vector<O5Row>& O5Table() {
    static const std::vector<O5Row> kTable = {
        {10, 2, "(x - 2049)(x + 24)^2",    "2049",    "-24",  2, 3},
        {10, 3, "(x - 177148)(x - 252)^2", "177148",  "252",  2, 3},
        {12, 2, "x - 8193",                "8193",    nullptr, 0, 1},
        {14, 2, "(x - 32769)(x - 216)^2",  "32769",   "216",  2, 3},
        {16, 2, "(x - 131073)(x + 528)^2", "131073",  "-528", 2, 3},
        {18, 2, "(x - 524289)(x - 456)^2", "524289",  "456",  2, 3},
        {20, 2, "(x - 2097153)(x + 288)^2", "2097153", "-288", 2, 3},
    };
    return kTable;
}

uint32_t DimMk(uint32_t k) {
    if (k % 2 != 0 || k == 2) return 0;
    return (k % 12 == 2) ? (k / 12) : (k / 12 + 1);
}

uint32_t RankLambda1(uint32_t n) {
    if (n == 0 || n % 2 != 0) return 0;
    return 2u * DimMk(n + 2u) - 1u;
}

DeligneBitBound CharpolyCoeffBitBound(uint32_t n, uint32_t p) {
    ENGINE_CHECK_MSG(p >= 2, "CharpolyCoeffBitBound: p >= 2");
    ENGINE_CHECK_MSG(n >= 2 && n % 2 == 0, "CharpolyCoeffBitBound: even n >= 2");
    DeligneBitBound B;
    // Largest den with p^den <= 2^126, then num = ceil(log2 p^den) = bitlen(p^den - 1):
    // p^den <= 2^num, hence log2 p <= num / den, all exact in unsigned __int128.
    const u128 cap = static_cast<u128>(1) << 126;
    u128 pw = p;
    uint32_t den = 1;
    while (pw <= cap / p) { pw *= p; ++den; }
    B.log2p_num = BitLen(pw - 1);
    B.log2p_den = den;
    const uint64_t k = n + 2u;
    // p^{k-1} <= 2^{(k-1) num / den} <= 2^{ceil((k-1) num / den)}.
    B.bits_eis = CeilDiv((k - 1) * static_cast<uint64_t>(B.log2p_num), B.log2p_den);
    B.bits_sqrt = CeilDiv(B.bits_eis, 2);
    // Eisenstein factor: 1 + |1 + E| = 2 + E <= 2E <= 2^{bits_eis + 1} (E = p^{k-1} >= 2).
    // Cusp factor: 1 + 2 sqrt(E) <= 3 sqrt(E) < 2^{bits_sqrt + 2}, 2 (d_k - 1) of them.
    const uint64_t d = DimMk(static_cast<uint32_t>(k));
    ENGINE_CHECK_MSG(d >= 1, "CharpolyCoeffBitBound: d_k >= 1 for even k >= 4");
    B.bits = B.bits_eis + 1u + 2u * (d - 1u) * (B.bits_sqrt + 2u);
    return B;
}

uint32_t ReduceDecimalMod(const std::string& dec, uint32_t ell) {
    ENGINE_CHECK_MSG(ell >= 2, "ReduceDecimalMod: ell >= 2");
    (void)DecimalMagnitudeDigits(dec.c_str());   // validates
    const char* s = dec.c_str();
    const bool neg = (*s == '-');
    if (*s == '-' || *s == '+') ++s;
    uint64_t r = 0;
    for (; *s; ++s) r = (r * 10u + static_cast<uint64_t>(*s - '0')) % ell;   // < 2^36
    if (neg && r != 0) r = ell - r;
    return static_cast<uint32_t>(r);
}

hecke::FpPoly OracleCharpolyMod(const O5Row& row, uint32_t ell) {
    const uint32_t e = ReduceDecimalMod(row.eis_root, ell);
    hecke::FpPoly P(ell, std::vector<uint32_t>{ (ell - e) % ell, 1u });
    if (row.cusp_root != nullptr && row.cusp_mult > 0) {
        const uint32_t c = ReduceDecimalMod(row.cusp_root, ell);
        const hecke::FpPoly lin(ell, std::vector<uint32_t>{ (ell - c) % ell, 1u });
        P = P * hecke::FpPoly::pow(lin, row.cusp_mult);
    }
    return P;
}

uint64_t OracleCoeffBitBound(const O5Row& row) {
    // |coefficient| <= prod (1 + |r|); 1 + |r| <= 10^D (D = digits of |r|) <= 2^{ceil(10 D / 3)}
    // since 10 <= 2^{10/3}.
    auto root_bits = [](const char* s) -> uint64_t {
        return CeilDiv(10u * static_cast<uint64_t>(DecimalMagnitudeDigits(s)), 3u);
    };
    uint64_t bits = root_bits(row.eis_root);
    if (row.cusp_root != nullptr) bits += static_cast<uint64_t>(row.cusp_mult) * root_bits(row.cusp_root);
    return bits;
}

bool IsPrimeU32(uint32_t x) {
    if (x < 2) return false;
    for (uint32_t q : {2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u, 47u, 53u, 59u, 61u}) {
        if (x % q == 0) return x == q;
    }
    uint64_t d = x - 1u;
    int s = 0;
    while ((d & 1u) == 0) { d >>= 1; ++s; }
    // Bases {2, 7, 61}: deterministic for every x < 4 759 123 141 > 2^32.
    for (uint64_t a : {2ull, 7ull, 61ull}) {
        uint64_t y = PowMod(a, d, x);
        if (y == 1 || y == x - 1u) continue;
        bool composite = true;
        for (int i = 1; i < s; ++i) {
            y = MulMod(y, y, x);
            if (y == x - 1u) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

std::vector<uint32_t> CrtPrimes(uint32_t count) {
    std::vector<uint32_t> out;
    out.reserve(count);
    for (uint32_t x = (1u << 30) - 1u; out.size() < count; x -= 2u) {
        ENGINE_CHECK_MSG(x > (1u << 29), "CrtPrimes: ran below 2^29 (count too large)");
        if (IsPrimeU32(x)) out.push_back(x);
    }
    return out;
}

uint32_t CertificatePrefix(const std::vector<uint32_t>& primes, uint64_t bits_needed,
                           uint64_t* bits_covered) {
    uint64_t acc = 0;
    uint32_t i = 0;
    while (acc < bits_needed) {
        ENGINE_CHECK_MSG(i < primes.size(), "CertificatePrefix: prime list too short for the bound");
        acc += BitLen(primes[i]) - 1u;   // ell >= 2^{bitlen - 1}
        ++i;
    }
    if (bits_covered) *bits_covered = acc;
    return i;
}

FellRun RunPipelineModEll(uint32_t n, uint32_t ell, const std::vector<uint32_t>& ps) {
    // (hecke-extensions slot 9, WP-9; [X] open item 9; slot-8 WP-12 review, m-5)
    // M-0's pipeline run is F_ell-only: it builds a field EquivariantCohomologyOnK
    // directly (no CoefficientRingContext), so it has no structural way to honour
    // a non-default `coefficient_ring`.  This is a NAMED guard rather than the
    // structural refusal alone, so a caller under a composite ring in force gets
    // [E] sec:ring (b2)'s message ("RunPipelineModEll (Integral): coefficient_ring=
    // ... is not honoured in this build") instead of an unrelated failure deeper
    // inside the pipeline.
    hecke::RequireFieldCoefficientRing("RunPipelineModEll (Integral)");
    // Good prime: ell > n + 1 and ell != p (T_p, not U_ell), [IB] Sec. 6.2 M-0.
    ENGINE_CHECK_MSG(ell > n + 1u, "RunPipelineModEll: ell must exceed n + 1 (good prime)");
    for (uint32_t p : ps) ENGINE_CHECK_MSG(ell != p, "RunPipelineModEll: ell == p");

    FellRun run;
    run.n = n;
    run.ell = ell;

    // The shipped pipeline, exactly the `--bench hecke` recipe (public API only).
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    const hecke::SL2ZTessellationTables model(K);
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n, ell)) rels.push_back(datum.rel);
    LocalSystemSimplicialComplex lssc(K, std::move(rels), n + 1, ell);
    EquivariantCohomologyOnK coh(&lssc);
    coh.compute_cohomology();
    run.dims = coh.get_H_eq_dimensions();

    for (uint32_t p : ps) {
        hecke::HeckeCosetTables cosets(p, /*r=*/1, ell, n);
        hecke::HeckeTauTable tau(K, model, cosets, ell, n);    // ctor certifies (obl:tau)
        hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
        op.compute();                                          // asserts the two-route identity
        const auto& A = op.get_H_matrices();
        ENGINE_CHECK_MSG(A.size() >= 2, "RunPipelineModEll: no A^{(1)}");
        run.charpoly[p] = hecke::CharPolyBerkowitz(A[1]);
    }
    return run;
}

bool EisensteinFactorDivides(const hecke::FpPoly& chi, uint32_t n, uint32_t p, uint32_t ell,
                             hecke::FpPoly* cofactor_out) {
    const uint64_t e = (1u + PowMod(p, n + 1u, ell)) % ell;   // 1 + p^{k-1}, k = n + 2
    const hecke::FpPoly lin(ell, std::vector<uint32_t>{ static_cast<uint32_t>((ell - e) % ell), 1u });
    hecke::FpPoly q(ell), r(ell);
    hecke::FpPoly::divrem(chi, lin, q, r);
    if (cofactor_out) *cofactor_out = q;
    return r.is_zero();
}

bool IsSquareMonic(const hecke::FpPoly& f) {
    const uint32_t ell = f.ell;
    ENGINE_CHECK_MSG(ell % 2u == 1u, "IsSquareMonic: ell odd");
    if (!f.is_monic()) return false;
    const int64_t deg = f.degree();
    if (deg % 2 != 0) return false;
    // Over a field of characteristic != 2 a monic square has a unique monic
    // square root S, determined top-down by the top m + 1 coefficients; then
    // f is a square iff S * S == f.
    const uint32_t m = static_cast<uint32_t>(deg / 2);
    const uint64_t inv2 = (ell + 1u) / 2u;
    std::vector<uint32_t> s(m + 1u, 0u);
    s[m] = 1u;
    for (uint32_t j = 1; j <= m; ++j) {
        uint64_t acc = 0;
        for (uint32_t i = 1; i < j; ++i) acc = (acc + MulMod(s[m - i], s[m - j + i], ell)) % ell;
        const uint64_t c = f.coeff(2u * m - j);
        s[m - j] = static_cast<uint32_t>(MulMod((c + ell - acc) % ell, inv2, ell));
    }
    const hecke::FpPoly S(ell, s);
    return S * S == f;
}

} // namespace integral
