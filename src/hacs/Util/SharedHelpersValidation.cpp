// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// SharedHelpersValidation (`fast`; S24 CL-1, docs/implementation/slot24/CLEANUP_plan_1030.md)
// ===========================================================================
//
// The CL-1 helpers must reproduce the copies they will replace, so that the
// later migrations (CL-2 ... CL-6) are value- and behaviour-neutral.  Each arm
// compares a helper with a LIVE copy wherever one is callable, else with a
// table or a reference transcribed from the copy (named in the arm):
//   (A) ScalarMod.h      vs nf::{mulmod64, mulmod64_portable, powmod32, invmod32},
//                        lvalue::{PowModU64, InvModU64}, eis::{MulModP, PowModP, InvModP},
//                        and the naive routines, on random and edge inputs
//   (B) ModCtx::chain    vs ModCtx(ell) and ModCtx::prime_power(ell, k)
//   (C) EnvLever.h       the parse matrix vs the copies' readers (FASTMUL, GOOD_SPARSE,
//                        GOOD_DENSE, ENGINE_GPU_IMMA live, SMITH2K_VERIFY), the WARNING
//                        texts, Scoped nesting, say_once, and the live DescentGood* /
//                        FormsFastMul readers
//   (D) Threads.h        vs ThreadPool::default_threads(), birch::MeasureThreads(),
//                        tower::TowerThreads() (live, ENGINE_THREADS set per input and
//                        restored), smithpar::default_threads() (as configured), and the
//                        SmithPar / DescentParAsm tables with their WARNING texts
//   (E) CpuFeatures.h    the predicates' implications, kernels::avx*_kernel_available(),
//                        the IsaPath names vs kernels::kernel_path_name
//   (F) ValidationCheck  the prefix / newline / print-limit output styles
//   (G) CellList.h       ParseStrict(Hook) vs HookParseCells, ParseStrict(SatCert) vs
//                        HeckeSatCertDumpParseCells, ParseLax vs census::ParseCellList,
//                        every error text those pin (HeckeSmallCarriesValidation)
//   (H) TextIo.h         the three hex styles vs atlas_index::Hex16 and std::hex; ReadFile
//   (I) Timing.h, (J) ParallelFor.h (the three chunk plans vs their copies' formulas,
//                        every index visited once)
//   (K) the legacy flag  (S24 CL-L0) LegacyConfig.h's refusal text, EnvLever's RETIRED mode
//                        (OFF: one route, source=retired, X=0 refused; ON: RetiredLever ==
//                        EnvLever), Registry/LegacyValidation.h's OC_LEGACY_VALIDATION stub and
//                        SKIP-ARM line.  Its LIVE probe reads OC_CL1_RETIRED_PROBE as the process
//                        was started: `OC_CL1_RETIRED_PROBE=0` in an OFF build is REFUSED (stderr
//                        `REFUSED OC_CL1_RETIRED_PROBE=0: compiled out (OC_ENABLE_LEGACY=OFF)`,
//                        exit 1) -- the process-level smoke of a retired lever's refusal.
// Read-only: it writes no file; it sets only its probe variables, ENGINE_THREADS and
// ENGINE_GPU_IMMA, each restored at once, and only around readers that re-read per call.
// ===========================================================================

#include "SharedHelpersValidation.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../ModArithmetic/Binary/SmithPar.h"
#include "../../Cuda/CudaKernels.h"
#include "../../Engine/EngineCostModel.h"
#include "../../Engine/ThreadPool.h"
#include "../../Engine/Threads.h"
#include "../../ModArithmetic/CpuFeatures.h"
#include "../../ModArithmetic/EnvLever.h"
#include "../../ModArithmetic/Kernels.h"
#include "../../ModArithmetic/LegacyConfig.h"
#include "../Registry/LegacyValidation.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../../ModArithmetic/ScalarMod.h"
#include "../Hecke/HeckeGammaHTower.h"
#include "../Hecke/HeckeLevelNDegenValidation.h"
#include "../HeckeExt/HeckeCensusJoin.h"
#include "../HeckeExt/HeckeFormsHeckeAlgebra.h"
#include "../HeckeExt/HeckeSatCertDumpValidation.h"
#include "../HeckeExt/LValue/Bernoulli.h"
#include "../HeckeExt/LValue/EisensteinMeasure.h"
#include "../HeckeExt/LValue/ModularSymbolMeasure.h"
#include "../Integral/Descent/DescentAssembly.h"
#include "../NumberField/ModUtil.h"
#include "AtlasIndex.h"
#include "CellList.h"
#include "ParallelFor.h"
#include "TextIo.h"
#include "Timing.h"
#include "ValidationCheck.h"

namespace {

using engine::scalar::u128;
namespace sc = engine::scalar;

oc_util::Checker g_c("  [FAIL] ");

void Check(bool ok, const std::string& what) { g_c(ok, what); }

std::string U(uint64_t v) { return std::to_string(v); }

uint64_t g_rng = 0x5eed0c11ull;
uint64_t Rand() {   // SplitMix64
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Sets `name` to `value` (nullptr = unset) for its lifetime, then restores the previous value.
class EnvSet {
public:
    EnvSet(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name);
        had_ = old != nullptr;
        if (had_) old_ = old;
        if (value) setenv(name, value, 1);
        else unsetenv(name);
    }
    ~EnvSet() {
        if (had_) setenv(name_, old_.c_str(), 1);
        else unsetenv(name_);
    }
    EnvSet(const EnvSet&) = delete;
    EnvSet& operator=(const EnvSet&) = delete;

private:
    const char* name_;
    bool had_ = false;
    std::string old_;
};

std::string Show(const char* v) { return v ? std::string("'") + v + "'" : std::string("unset"); }

// ---------------------------------------------------------------------------- (A)
uint64_t NaivePow(uint64_t b, uint64_t e, uint64_t m) {
    u128 r = 1 % m;
    for (uint64_t i = 0; i < e; ++i) r = r * (b % m) % m;
    return static_cast<uint64_t>(r);
}
bool IsPrimeNaive(uint64_t v) {
    if (v < 2) return false;
    for (uint64_t d = 2; d * d <= v; ++d)
        if (v % d == 0) return false;
    return true;
}

// compile-time evaluation (the header is constexpr)
static_assert(sc::powmod<uint64_t>(3, 4, 100) == 81, "powmod constexpr");
static_assert(sc::powmod<uint32_t>(5, 0, 1) == 0, "powmod: 1 % m at m == 1");
static_assert(sc::powfit(2, 63) == (uint64_t(1) << 63) && sc::powfit(2, 64) == 0, "powfit edges");
static_assert(sc::invmod_or_zero<uint64_t>(3, 7) == 5 && sc::invmod_or_zero<uint64_t>(6, 9) == 0, "invmod_or_zero");
static_assert(sc::modi(-1, 7) == 6 && sc::modi(INT64_MIN, 3) == 1, "modi");

void ArmA() {
    const uint64_t kM61 = (uint64_t(1) << 61) - 1, kP64 = 18446744073709551557ull, kP32 = 4294967291u;
    // (A1) products: u64 (128-bit / portable) vs nf, u32 vs the u64 product, u128 vs u64 on the u64 domain
    uint32_t bad = 0;
    for (int i = 0; i < 4000; ++i) {
        const uint64_t m = (i % 4 == 0) ? (Rand() >> (Rand() % 63)) | 1u : (i % 4 == 1) ? kP64 : (i % 4 == 2) ? kM61 : (Rand() % 1000) + 1;
        const uint64_t a = Rand(), b = Rand();
        const uint64_t want = static_cast<uint64_t>(static_cast<u128>(a) * b % m);
        if (sc::mulmod(a, b, m) != want || sc::mulmod_portable(a, b, m) != want || nf::mulmod64(a, b, m) != want ||
            nf::mulmod64_portable(a, b, m) != want || static_cast<uint64_t>(sc::mulmod(u128(a), u128(b), u128(m))) != want)
            ++bad;
        const uint32_t m32 = static_cast<uint32_t>(m % kP32) | 1u, a32 = static_cast<uint32_t>(a), b32 = static_cast<uint32_t>(b);
        if (sc::mulmod(a32, b32, m32) != static_cast<uint32_t>(static_cast<uint64_t>(a32) * b32 % m32) ||
            sc::mulmod(a32, b32, m32) != nf::mulmod32(a32, b32, m32))
            ++bad;
    }
    Check(bad == 0, "(A1) mulmod u32 / u64 / u128 / portable == the 128-bit product == nf::mulmod{32,64}{,_portable}: 8000 cases (" +
                        U(bad) + " bad)");
    // u128 beyond 2^64: (2^100 + 7) * 3 mod (2^101 - 1), checked by the identity x * 3 == x + x + x
    {
        const u128 m = (u128(1) << 101) - 1, x = (u128(1) << 100) + 7;
        const u128 want = sc::addmod(sc::addmod(x % m, x % m, m), x % m, m);
        Check(sc::mulmod(x, u128(3), m) == want, "(A1) u128 mulmod beyond 2^64 == the repeated sum");
    }
    // (A2) powmod vs the naive product, nf::powmod32, lvalue::PowModU64, eis::PowModP; Fermat at 2^61-1 and 2^64-59
    bad = 0;
    for (int i = 0; i < 1500; ++i) {
        const uint64_t m = (i % 3 == 0) ? (Rand() % 5000) + 1 : (i % 3 == 1) ? (Rand() >> 1) | 1u : (Rand() % 3) + 1;
        const uint64_t b = Rand(), e = Rand() % 70;
        const uint64_t want = NaivePow(b, e, m);
        if (sc::powmod<uint64_t>(b, e, m) != want || hecke_ext::lvalue::PowModU64(b, e, m) != want) ++bad;
        if (sc::powmod<u128>(b, e, m) != want) ++bad;
        if (m < (uint64_t(1) << 32)) {
            const uint32_t m32 = static_cast<uint32_t>(m);
            if (sc::powmod<uint32_t>(static_cast<uint32_t>(b % m32), e, m32) != want || nf::powmod32(b, e, m32) != want) ++bad;
        }
        const uint64_t e2 = Rand();
        if (sc::powmod<uint64_t>(b, e2, m) != hecke_ext::lvalue::PowModU64(b, e2, m) ||
            sc::powmod<uint64_t>(b, e2, m) != hecke_ext::lvalue::eis::PowModP(b, e2, m))
            ++bad;
    }
    Check(bad == 0, "(A2) powmod u32 / u64 / u128 == naive (e < 70) == nf::powmod32 == lvalue::PowModU64 == eis::PowModP: 1500 x 4 (" +
                        U(bad) + " bad)");
    Check(sc::powmod<uint64_t>(123456789, kM61 - 1, kM61) == 1 && sc::powmod<uint64_t>(987654321, kP64 - 1, kP64) == 1 &&
              sc::powmod<uint64_t>(0, 0, 7) == 1 && sc::powmod<uint64_t>(5, 3, 1) == 0,
          "(A2) Fermat at 2^61-1 and 2^64-59; 0^0 = 1; mod 1 = 0");
    // (A3) inverses: try_invmod (any m) vs gcd, InvModU64, nf::invmod32; invmod_fermat vs eis::InvModP
    bad = 0;
    uint32_t units = 0, nonunits = 0;
    for (int i = 0; i < 3000; ++i) {
        const uint64_t m = (i % 5 == 0) ? 1 : (i % 5 == 1) ? (Rand() % 60) + 2 : (i % 5 == 2) ? Rand() | 1u : (i % 5 == 3) ? kP64 : Rand() % 100000 + 2;
        const uint64_t a = (i % 7 == 0) ? 0 : Rand();
        uint64_t x = 12345;
        const bool ok = sc::try_invmod(a, m, x);
        const bool unit = sc::gcd(a % m, m) == 1 || m == 1;
        if (ok != unit) ++bad;
        if (!ok) {
            ++nonunits;
            if (sc::invmod_or_zero(a, m) != 0) ++bad;
            continue;
        }
        ++units;
        if (sc::mulmod(a % m, x, m) != 1 % m || x >= m) ++bad;
        if (m >= 2 && m < (uint64_t(1) << 62)) {
            if (hecke_ext::lvalue::InvModU64(a % m, m) != x) ++bad;
        }
        if (m >= 2 && m < (uint64_t(1) << 31) && nf::invmod32(a, static_cast<uint32_t>(m)) != x) ++bad;
        // u32 and u128 instantiations agree
        if (m < (uint64_t(1) << 32)) {
            uint32_t x32 = 0;
            if (!sc::try_invmod(static_cast<uint32_t>(a % m), static_cast<uint32_t>(m), x32) || x32 != x) ++bad;
        }
        u128 x128 = 0;
        if (!sc::try_invmod(u128(a), u128(m), x128) || static_cast<uint64_t>(x128) != x) ++bad;
    }
    Check(bad == 0 && units > 0 && nonunits > 0,
          "(A3) try_invmod: ok iff gcd == 1 (m == 1 -> 0), a x == 1, == InvModU64 == nf::invmod32, u32 / u128 agree: " + U(units) +
              " units, " + U(nonunits) + " non-units (" + U(bad) + " bad)");
    bad = 0;
    for (uint64_t p : {uint64_t(3), uint64_t(5), uint64_t(691), uint64_t(1000003), kP32, kM61}) {
        for (int i = 0; i < 50; ++i) {
            const uint64_t a = Rand() % p;
            if (a == 0) continue;
            if (sc::invmod_fermat(a, p) != hecke_ext::lvalue::eis::InvModP(a, p) || sc::invmod_fermat(a, p) != sc::invmod_or_zero(a, p)) ++bad;
        }
        if (sc::invmod_fermat(uint64_t(0), p) != 0) ++bad;   // the unchecked Fermat copies: 0 for a == 0 (p > 2)
    }
    Check(bad == 0 && sc::invmod_fermat(uint64_t(0), uint64_t(2)) == 1 && sc::invmod_fermat(uint64_t(1), uint64_t(2)) == 1,
          "(A3) invmod_fermat == eis::InvModP == try_invmod on units; 0 -> 0 (p > 2); p == 2 -> 1 for every a (verbatim a^(p-2))");
    // (A4) powfit vs the u128 loop of the PowFit / Pow64Fit / FitsU64 copies; pow_wrap vs the wrapping loop; gcd; modi
    bad = 0;
    for (uint64_t b : {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(691), uint64_t(65535), uint64_t(65536),
                       uint64_t(4294967295u), uint64_t(4294967296ull), kM61, UINT64_MAX})
        for (uint32_t k = 0; k <= 70; ++k) {
            u128 r = 1;
            bool fit = true;
            for (uint32_t i = 0; i < k && fit; ++i) {
                r *= b;
                if (r >> 64) fit = false;
            }
            const uint64_t want = fit ? static_cast<uint64_t>(r) : 0;
            uint64_t w = 1;
            for (uint32_t i = 0; i < k; ++i) w *= b;
            if (sc::powfit(b, k) != want || sc::pow_wrap(b, k) != w) ++bad;
        }
    Check(bad == 0, "(A4) powfit == the u128 loop (0 past 2^64), pow_wrap == the mod-2^64 loop: 11 bases x k = 0..70 (" + U(bad) + " bad)");
    bad = 0;
    for (int i = 0; i < 2000; ++i) {
        const int64_t x = static_cast<int64_t>(Rand());
        const uint64_t m = (i % 2) ? (Rand() % 1000000) + 1 : (Rand() >> 2) + 1;   // m < 2^62: the copies' cast domain
        const int64_t r = x % static_cast<int64_t>(m);
        const uint64_t want = static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(m) : r);
        if (sc::modi(x, m) != want) ++bad;
        const uint64_t a = Rand() >> (Rand() % 64), b = Rand() >> (Rand() % 64);
        uint64_t g0 = a, g1 = b;
        while (g1) { const uint64_t t = g0 % g1; g0 = g1; g1 = t; }
        if (sc::gcd(a, b) != g0) ++bad;
    }
    Check(bad == 0 && sc::modi(INT64_MIN, UINT64_MAX) == (uint64_t(1) << 63) - 1 && sc::modi(-5, 1) == 0,
          "(A4) modi == the ModU(int64_t, m) copies on m < 2^63 (and exact beyond); gcd == Euclid: 2000 cases");
}

// ---------------------------------------------------------------------------- (B)
template <class F>
bool Throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void ArmB() {
    using C32 = engine::ModCtx<uint32_t>;
    using C64 = engine::ModCtx<uint64_t>;
    const C32 f = C32::chain(7, 1), g = C32(7);
    Check(f.is_field() && f.p == 7 && f.ell == 7 && f.k == 1 && f.p == g.p && f.barrett_m == g.barrett_m && f.regime == g.regime,
          "(B) chain(7, 1) == ModCtx(7): the field (is_field, p, ell, k, barrett_m, regime)");
    bool same = true;
    for (uint32_t k = 2; k <= 9; ++k) {
        const C32 a = C32::chain(7, k), b = C32::prime_power(7, k);
        same = same && !a.is_field() && a.p == b.p && a.ell == b.ell && a.k == b.k && a.k_max == b.k_max && a.regime == b.regime;
        for (uint32_t v = 0; v <= k; ++v) same = same && a.ell_pow(v) == b.ell_pow(v);
    }
    const C64 c = C64::chain(3, 30), d = C64::prime_power(3, 30), e = C64::chain(1000003, 1);
    same = same && c.p == d.p && c.k == 30 && !c.is_field() && e.is_field() && e.p == 1000003;
    Check(same, "(B) chain(ell, k >= 2) == prime_power(ell, k) (u32 7^2..7^9, every ell_pow; u64 3^30); u64 chain(1000003, 1) a field");
    Check(Throws([] { (void)C32::chain(7, 0); }) && Throws([] { (void)C32::chain(1, 1); }) && Throws([] { (void)C32::chain(2, 32); }) &&
              Throws([] { (void)C32::chain(65537, 2); }),
          "(B) chain refuses k = 0, ell = 1, 2^32 and 65537^2 in u32 (ENGINE_CHECK)");
}

// ---------------------------------------------------------------------------- (C)
namespace lv = engine::lever;

constexpr lv::Policy kPolFast{true, lv::Empty::Invalid, lv::Match::Exact01, lv::Cache::Once,
                              "WARNING HeckeFormsHeckeAlgebra: HECKE_FORMS_FASTMUL='%s' is neither 0 nor 1; the fast route stays ON "
                              "(only exactly 0 selects the naive route)\n",
                              "default", "env", "env"};
constexpr lv::Policy kPolSparse{true, lv::Empty::Unset, lv::Match::Exact01, lv::Cache::Once,
                                "WARNING: HECKE_DESCENT_GOOD_SPARSE=%s is neither 0 nor 1; the sparse good prime stays on (default)\n",
                                "default", "env", "default"};
constexpr lv::Policy kPolDense{true, lv::Empty::Invalid, lv::Match::Exact01, lv::Cache::Once,
                               "WARNING: HECKE_DESCENT_GOOD_DENSE=%s is neither 0 nor 1; the dense good prime stays on (default)\n",
                               "default", "env", "env-invalid(default kept)"};
constexpr lv::Policy kPolImma{true, lv::Empty::Unset, lv::Match::FirstChar, lv::Cache::PerCall, nullptr, "default", "env", "env"};
constexpr lv::Policy kPolVerify{false, lv::Empty::Unset, lv::Match::FirstChar, lv::Cache::Once, nullptr, "default", "env", "env"};

// The copies' readers, transcribed (enabled, source).  The FASTMUL copy: HeckeFormsHeckeAlgebra.cpp fast_env +
// FormsFastMulEffectiveLine; GOOD_SPARSE / GOOD_DENSE: DescentAssembly.cpp good_{sparse,dense}_env + *Source;
// SMITH2K_VERIFY: DescentSmithGpu.cpp env_verify.
std::pair<bool, std::string> CopyFast(const char* e) {
    if (e == nullptr) return {true, "default"};
    return {std::string(e) != "0", "env"};
}
std::pair<bool, std::string> CopySparse(const char* e) {
    if (!e || !*e) return {true, "default"};
    if (std::strcmp(e, "0") == 0) return {false, "env"};
    if (std::strcmp(e, "1") == 0) return {true, "env"};
    return {true, "default"};
}
std::pair<bool, std::string> CopyDense(const char* e) {
    if (e == nullptr) return {true, "default"};
    if (std::strcmp(e, "0") == 0) return {false, "env"};
    if (std::strcmp(e, "1") == 0) return {true, "env"};
    return {true, "env-invalid(default kept)"};
}
bool CopyVerify(const char* e) { return e != nullptr && e[0] == '1'; }

void ArmC() {
    const char* inputs[] = {nullptr, "", "0", "1", "00", "01", "10", "x", " 0", "0 ", "true"};
    uint32_t bad = 0;
    std::string first;
    auto note = [&](const std::string& s) {
        ++bad;
        if (first.empty()) first = s;
    };
    for (const char* in : inputs) {
        const auto f = CopyFast(in);
        const lv::State sf = lv::read(in, kPolFast);
        if (lv::state_enabled(sf, kPolFast) != f.first || lv::state_source(sf, kPolFast) != f.second) note("FASTMUL " + Show(in));
        const auto s = CopySparse(in);
        const lv::State ss = lv::read(in, kPolSparse);
        if (lv::state_enabled(ss, kPolSparse) != s.first || lv::state_source(ss, kPolSparse) != s.second) note("GOOD_SPARSE " + Show(in));
        const auto d = CopyDense(in);
        const lv::State sd = lv::read(in, kPolDense);
        if (lv::state_enabled(sd, kPolDense) != d.first || lv::state_source(sd, kPolDense) != d.second) note("GOOD_DENSE " + Show(in));
        if (lv::state_enabled(lv::read(in, kPolVerify), kPolVerify) != CopyVerify(in)) note("SMITH2K_VERIFY " + Show(in));
        // ENGINE_GPU_IMMA: the live copy (re-read per call), against a PerCall lever on the same variable.
        // (S24 CL-R2) The live copy is a retired lever: an OC_ENABLE_LEGACY=OFF build refuses the "0..." readings
        // (exit 1), so there the live comparison runs on the other inputs only (the parse itself is in every build).
        if (oc::legacy::kLegacyBuild || lv::read(in, kPolImma) != lv::State::Off) {
            EnvSet env("ENGINE_GPU_IMMA", in);
            const lv::EnvLever imma("ENGINE_GPU_IMMA", kPolImma);
            if (imma.enabled() != engine::cuda::imma_policy()) note("ENGINE_GPU_IMMA (live) " + Show(in));
        }
    }
    Check(bad == 0, "(C1) the parse matrix: 11 values x {FASTMUL, GOOD_SPARSE, GOOD_DENSE, SMITH2K_VERIFY, ENGINE_GPU_IMMA live}: enabled "
                    "and source as the copies (" + U(bad) + " bad" + (first.empty() ? "" : ", first " + first) + ")");
    // (C2) the WARNING texts, byte for byte
    Check(lv::warning_text("x", kPolFast) ==
                  "WARNING HeckeFormsHeckeAlgebra: HECKE_FORMS_FASTMUL='x' is neither 0 nor 1; the fast route stays ON (only exactly 0 "
                  "selects the naive route)\n" &&
              lv::warning_text("", kPolFast) ==
                  "WARNING HeckeFormsHeckeAlgebra: HECKE_FORMS_FASTMUL='' is neither 0 nor 1; the fast route stays ON (only exactly 0 "
                  "selects the naive route)\n" &&
              lv::warning_text("2", kPolSparse) ==
                  "WARNING: HECKE_DESCENT_GOOD_SPARSE=2 is neither 0 nor 1; the sparse good prime stays on (default)\n" &&
              lv::warning_text("", kPolSparse).empty() &&
              lv::warning_text("", kPolDense) ==
                  "WARNING: HECKE_DESCENT_GOOD_DENSE= is neither 0 nor 1; the dense good prime stays on (default)\n" &&
              lv::warning_text("1", kPolDense).empty() && lv::warning_text(nullptr, kPolDense).empty() &&
              lv::warning_text("x", kPolImma).empty(),
          "(C2) WARNING texts: FASTMUL 'x' and '' (it warns on ''), GOOD_SPARSE '2' (silent on ''), GOOD_DENSE '' ; none for valid / "
          "unset / silent policies");
    // (C3) a live lever: caching, Scoped nesting, source, the lines, say_once
    {
        const lv::Policy quiet{true, lv::Empty::Unset, lv::Match::Exact01, lv::Cache::Once, nullptr, "default", "env", "default"};
        EnvSet env("OC_CL1_LEVER_PROBE", "0");
        const lv::EnvLever l("OC_CL1_LEVER_PROBE", quiet);
        bool ok = !l.enabled() && std::string(l.source()) == "env";
        {
            const lv::EnvLever::Scoped a(l, true);
            ok = ok && l.enabled() && std::string(l.source()) == "scoped" && !l.env_enabled();
            {
                const lv::EnvLever::Scoped b(l, false);
                ok = ok && !l.enabled();
            }
            ok = ok && l.enabled();
        }
        ok = ok && !l.enabled() && std::string(l.source()) == "env";
        setenv("OC_CL1_LEVER_PROBE", "1", 1);
        ok = ok && !l.enabled();   // Cache::Once: the first reading stays
        const lv::Policy percall{true, lv::Empty::Unset, lv::Match::Exact01, lv::Cache::PerCall, nullptr, "default", "env", "default"};
        const lv::EnvLever m("OC_CL1_LEVER_PROBE", percall);
        ok = ok && m.enabled();
        Check(ok, "(C3) Scoped nests and restores, source scoped / env, env_enabled ignores the override, Once caches, PerCall re-reads");
        Check(l.effective_line("mul", "fast", "naive") == "EFFECTIVE OC_CL1_LEVER_PROBE mul=naive source=env" &&
                  l.short_line("mul", "fast", "naive") == "EFFECTIVE OC_CL1_LEVER_PROBE mul=naive (OC_CL1_LEVER_PROBE=1)",
              "(C3) effective_line / short_line texts (the short line prints the cached env reading and the value now)");
        std::ostringstream os;
        const bool p1 = l.say_once(os, "LINE-A"), p2 = l.say_once(os, "LINE-B");
        unsetenv("OC_CL1_LEVER_PROBE");
        std::ostringstream os2;
        const bool p3 = m.say_once(os2, "LINE-C");
        Check(p1 && !p2 && os.str() == "LINE-A\n" && !p3 && os2.str().empty(), "(C3) say_once prints once while set, never while unset");
    }
    // (C4) the live levers: DescentGoodSparse / DescentGoodDense (under this process's environment), FormsFastMul
    {
        const lv::Policy sparse_quiet{true, lv::Empty::Unset, lv::Match::Exact01, lv::Cache::Once, nullptr, "default", "env", "default"};
        const lv::Policy dense_quiet{true, lv::Empty::Invalid, lv::Match::Exact01, lv::Cache::Once, nullptr, "default", "env",
                                     "env-invalid(default kept)"};
        const lv::EnvLever sp("HECKE_DESCENT_GOOD_SPARSE", sparse_quiet), de("HECKE_DESCENT_GOOD_DENSE", dense_quiet);
        bool ok = sp.enabled() == integral_descent::DescentGoodSparseEnabled() && sp.source() == integral_descent::DescentGoodSparseSource() &&
                  de.enabled() == integral_descent::DescentGoodDenseEnabled() && de.source() == integral_descent::DescentGoodDenseSource();
        {
            const integral_descent::ScopedDescentGoodSparse a(false);
            const integral_descent::ScopedDescentGoodDense b(false);
            const lv::EnvLever::Scoped c(sp, false), d(de, false);
            ok = ok && sp.enabled() == integral_descent::DescentGoodSparseEnabled() && sp.source() == integral_descent::DescentGoodSparseSource() &&
                 de.enabled() == integral_descent::DescentGoodDenseEnabled() && de.source() == integral_descent::DescentGoodDenseSource();
        }
        Check(ok, "(C4) live: DescentGood{Sparse,Dense}{Enabled,Source}() == the lever, unscoped and scoped off");
        if (std::getenv("HECKE_FORMS_FASTMUL") == nullptr) {   // the real line marks its short line said: harmless only when unset
            const lv::Policy fast_quiet{true, lv::Empty::Invalid, lv::Match::Exact01, lv::Cache::Once, nullptr, "default", "env", "env"};
            const lv::EnvLever fm("HECKE_FORMS_FASTMUL", fast_quiet);
            bool same = hecke_ext::forms::FormsFastMulEffectiveLine().rfind(fm.effective_line("mul", "fast", "naive") + " ntt=", 0) == 0;
            {
                const hecke_ext::forms::ScopedFormsFastMul s(false);
                const lv::EnvLever::Scoped t(fm, false);
                same = same && hecke_ext::forms::FormsFastMulEffectiveLine().rfind(fm.effective_line("mul", "fast", "naive") + " ntt=", 0) == 0;
            }
            Check(same, "(C4) live: FormsFastMulEffectiveLine() starts with the lever's effective_line, unscoped and scoped naive");
        } else {
            std::cout << "  (C4) FormsFastMul live comparison skipped: HECKE_FORMS_FASTMUL is set\n";
        }
    }
}

// ---------------------------------------------------------------------------- (D)
namespace th = engine::threads;

static_assert(th::kThreadPool.env_cap == engine::tape::kMaxWorkers && th::kThreadPool.default_cap == engine::tape::kMaxWorkers,
              "kThreadPool caps == ThreadPool's kMaxWorkers");
static_assert(th::kSmithPar.default_cap == engine::smithpar::kDefaultThreadCap && th::kSmithPar.env_cap == engine::smithpar::kMaxThreads,
              "kSmithPar caps == SmithPar.h");

struct ThreadRow {
    const char* raw;
    uint32_t hw, aff, want;
    const char* src;
    const char* warn;   // nullptr = none
};

void ArmD() {
    const char* inputs[] = {nullptr, "", "0", "1", "8", "16", "17", "33", "300", "-1", "-5", "abc", "8x", " 8", "8 ",
                            "4294967296", "4294967297", "99999999999999999999"};
    const uint32_t hw = th::hardware();
    uint32_t bad = 0;
    std::string first;
    for (const char* in : inputs) {
        EnvSet env("ENGINE_THREADS", in);
        const uint32_t pool = engine::tape::ThreadPool::default_threads();
        const uint32_t meas = hecke_ext::lvalue::birch::MeasureThreads();
        const uint32_t tower = hecke::level::tower::TowerThreads();
        const uint32_t a = th::configured(th::kThreadPool), b = th::configured(th::kCap16), c = th::configured(th::kGammaHTower);
        if (a != pool || b != meas || c != tower) {
            ++bad;
            if (first.empty())
                first = Show(in) + ": pool " + U(pool) + "/" + U(a) + " measure " + U(meas) + "/" + U(b) + " tower " + U(tower) + "/" + U(c);
        }
    }
    Check(bad == 0, "(D1) live, 18 values of ENGINE_THREADS: kThreadPool == ThreadPool::default_threads(), kCap16 == birch::MeasureThreads(), "
                    "kGammaHTower == tower::TowerThreads() (hardware " + U(hw) + "; " + U(bad) + " bad" + (first.empty() ? "" : ", first " + first) + ")");
    {   // SmithPar as this process configured it (its reader is cached; the environment is unchanged since start-up)
        const th::Result r = th::resolve(std::getenv("ENGINE_THREADS"), th::kSmithPar, th::hardware(), th::affinity());
        Check(r.threads == engine::smithpar::default_threads() && std::string(r.source) == engine::smithpar::default_threads_source(),
              "(D2) live: kSmithPar == smithpar::default_threads() and its source (" + U(r.threads) + ", " + r.source + ")");
    }
    // (D3) the tables transcribed from the copies that cannot be called per value
    const ThreadRow smith[] = {
        {nullptr, 12, 8, 8, "affinity", nullptr},
        {"", 12, 8, 8, "affinity", nullptr},
        {"0", 12, 8, 8, "affinity", nullptr},
        {"1", 12, 8, 1, "env", nullptr},
        {" 8", 12, 8, 8, "env", nullptr},
        {"8 \t\n", 12, 8, 8, "env", nullptr},
        {"256", 12, 8, 256, "env", nullptr},
        {"300", 12, 8, 256, "env", "WARNING: ENGINE_THREADS=300 exceeds 256; the descent Smith pool is clamped to 256 threads\n"},
        {"-1", 12, 8, 8, "env-invalid",
         "WARNING: ENGINE_THREADS=-1 is not a non-negative integer; the descent Smith pool uses the default 8 threads (affinity)\n"},
        {"8x", 12, 8, 8, "env-invalid",
         "WARNING: ENGINE_THREADS=8x is not a non-negative integer; the descent Smith pool uses the default 8 threads (affinity)\n"},
        {"99999999999999999999", 12, 8, 8, "env-invalid",
         "WARNING: ENGINE_THREADS=99999999999999999999 is not a non-negative integer; the descent Smith pool uses the default 8 threads "
         "(affinity)\n"},
        {nullptr, 100, 0, 64, "hardware", nullptr},
        {"abc", 100, 0, 64, "env-invalid",
         "WARNING: ENGINE_THREADS=abc is not a non-negative integer; the descent Smith pool uses the default 64 threads (hardware)\n"},
        {nullptr, 0, 0, 1, "hardware", nullptr},
        {nullptr, 4, 200, 64, "affinity", nullptr},
    };
    const ThreadRow parasm[] = {
        {nullptr, 12, 0, 12, "hardware", nullptr}, {"", 12, 0, 12, "hardware", nullptr},   {"0", 12, 0, 12, "hardware", nullptr},
        {"1", 12, 0, 1, "env", nullptr},            {"8x", 12, 0, 8, "env", nullptr},       {"300", 12, 0, 256, "env", nullptr},
        {"-1", 12, 0, 12, "hardware", nullptr},     {"abc", 12, 0, 12, "hardware", nullptr}, {"4294967296", 12, 0, 1, "env", nullptr},
        {"4294967297", 12, 0, 1, "env", nullptr},   {"99999999999999999999", 12, 0, 256, "env", nullptr},
        {nullptr, 0, 0, 1, "hardware", nullptr},    {nullptr, 512, 0, 256, "hardware", nullptr},
    };
    const ThreadRow tower[] = {
        {nullptr, 12, 0, 8, "hardware", nullptr},      {"0", 12, 0, 8, "hardware", nullptr}, {"100", 12, 0, 64, "env", nullptr},
        {"4294967297", 12, 0, 64, "env", nullptr},     {"5", 12, 0, 5, "env", nullptr},      {nullptr, 0, 0, 1, "hardware", nullptr},
        {nullptr, 3, 0, 3, "hardware", nullptr},
    };
    const ThreadRow pool[] = {
        {"-1", 12, 0, 32, "env", nullptr},          {"4294967296", 12, 0, 12, "hardware", nullptr}, {"4294967297", 12, 0, 1, "env", nullptr},
        {"abc", 12, 0, 12, "hardware", nullptr},    {nullptr, 0, 0, 1, "hardware", nullptr},        {nullptr, 64, 0, 32, "hardware", nullptr},
    };
    bad = 0;
    first.clear();
    auto run = [&](const char* tag, const th::Policy& p, const ThreadRow* rows, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            const ThreadRow& w = rows[i];
            const th::Result r = th::resolve(w.raw, p, w.hw, w.aff);
            const std::string want_warn = w.warn ? w.warn : "";
            if (r.threads != w.want || std::string(r.source) != w.src || r.warning != want_warn) {
                ++bad;
                if (first.empty()) first = std::string(tag) + " " + Show(w.raw) + " -> " + U(r.threads) + " " + r.source;
            }
        }
    };
    run("SmithPar", th::kSmithPar, smith, sizeof(smith) / sizeof(smith[0]));
    run("DescentParAsm", th::kDescentParAsm, parasm, sizeof(parasm) / sizeof(parasm[0]));
    run("GammaHTower", th::kGammaHTower, tower, sizeof(tower) / sizeof(tower[0]));
    run("ThreadPool", th::kThreadPool, pool, sizeof(pool) / sizeof(pool[0]));
    Check(bad == 0, "(D3) tables: SmithPar (strict, affinity, 64 / 256, both WARNING texts), DescentParAsm (256), GammaHTower (8 / 64, "
                    "clamp before the cast), ThreadPool (strtoul wrap and truncation): 41 rows (" + U(bad) + " bad" +
                        (first.empty() ? "" : ", first " + first) + ")");
}

// ---------------------------------------------------------------------------- (E)
void ArmE() {
    namespace cpu = engine::cpu;
    const bool chain = (!cpu::avx512_vnni() || cpu::avx512_f_bw_dq_vl()) && (!cpu::avx512_f_bw_dq_vl() || (cpu::avx512f() && cpu::avx512bw())) &&
                       (!cpu::avx2_fma() || cpu::avx2());
    Check(chain, "(E) avx512_vnni => avx512_f_bw_dq_vl => avx512f, avx512bw; avx2_fma => avx2");
    // "available" = the module's build flag && the CPU: available implies the predicate
    Check((!engine::kernels::avx2_kernel_available() || cpu::avx2_fma()) &&
              (!engine::kernels::avx512_kernel_available() || cpu::avx512_f_bw_dq_vl()),
          "(E) kernels::avx2_kernel_available() => avx2_fma(), avx512_kernel_available() => avx512_f_bw_dq_vl()");
    using KP = engine::kernels::KernelPath;
    using IP = cpu::IsaPath;
    bool names = std::string(cpu::isa_path_name(IP::Auto)) == engine::kernels::kernel_path_name(KP::Auto) &&
                 std::string(cpu::isa_path_name(IP::Portable)) == engine::kernels::kernel_path_name(KP::Portable) &&
                 std::string(cpu::isa_path_name(IP::Avx2)) == engine::kernels::kernel_path_name(KP::Avx2) &&
                 std::string(cpu::isa_path_name(IP::Avx512)) == engine::kernels::kernel_path_name(KP::Avx512);
    IP p = IP::Auto;
    names = names && cpu::isa_path_from_name("scalar", p) && p == IP::Portable && cpu::isa_path_from_name("avx512", p) && p == IP::Avx512 &&
            cpu::isa_path_from_name("auto", p) && p == IP::Auto && !cpu::isa_path_from_name("AVX2", p) && !cpu::isa_path_from_name("", p) &&
            !cpu::isa_path_from_name(nullptr, p);
    Check(names, "(E) IsaPath names == kernels::kernel_path_name per path (by name: the enum orders differ); ENGINE_KERNEL spellings, "
                 "'scalar' alias, case-sensitive");
    std::cout << "  (E) cpu: avx2=" << cpu::avx2() << " fma=" << cpu::features().fma << " avx512f=" << cpu::avx512f()
              << " avx512_f_bw_dq_vl=" << cpu::avx512_f_bw_dq_vl() << " vnni=" << cpu::avx512_vnni() << "\n";
}

// ---------------------------------------------------------------------------- (F)
void ArmF() {
    std::ostringstream a, b, c;
    oc_util::Checker x("  *** FAIL [census-cell-list] ", true, 0, &a), y("  FAIL: ", false, 2, &b), z("    [FAIL] ", true, 0, &c);
    const bool r1 = x(true, "fine"), r2 = x(false, "one");
    y(false, "p");
    y(true, "q");
    y(false, "r");
    y(false, "s");   // the third failure: counted, not printed
    z(false, "deep");
    Check(r1 && !r2 && a.str() == "  *** FAIL [census-cell-list] one\n" && x.checks == 2 && x.fails == 1,
          "(F) Checker: returns ok, counts, a tagged prefix");
    Check(b.str() == "  FAIL: p\n  FAIL: r\n" && y.checks == 4 && y.fails == 3 && y.exit_code() == 1 && c.str() == "    [FAIL] deep\n" &&
              y.summary("V") == "V: 4 checks, 3 failing",
          "(F) Checker: print_limit 2 prints the first two failures (counts all), indentation kept, summary text");
}

// ---------------------------------------------------------------------------- (G)
namespace cl = oc_util::cells;

bool CellsEq(const std::vector<std::vector<uint64_t>>& a, const std::vector<cl::Cell>& b) { return a == b; }

void ArmG() {
    // (G1) Style::Hook vs HookParseCells (HeckeLevelNDegenValidation.cpp; HeckeSmallCarriesValidation (1))
    cl::Grammar hook;
    hook.style = cl::Style::Hook;
    hook.name = "HECKE_LEVEL_N_SPARSE_CELLS";
    hook.fields = 3;
    hook.max_digits = 18;
    const char* hook_in[] = {"1571:0:5,210:4:7", "", "1571:0:5,", ",1571:0:5", "1571:0:5,,210:4:7", "1571:0", "1571:0:5:", "1571:0:x",
                             "1571:-2:5", "1:2:3:4", "1::3", ":1:2", "1234567890123456789:1:2", "123456789012345678:1:2", "1:2:3,4:5:6,7:8:9",
                             " 1:2:3", "1:2:3 "};
    uint32_t bad = 0, refused = 0;
    std::string first;
    for (const char* in : hook_in) {
        std::vector<std::vector<uint64_t>> real;
        std::string real_err;
        bool real_ok = true;
        try {
            real = HookParseCells(in, hook.name.c_str(), 3);
        } catch (const std::logic_error& e) {
            real_ok = false;
            real_err = e.what();
        }
        std::vector<cl::Cell> mine;
        std::string why;
        const bool ok = cl::ParseStrict(in, hook, mine, why);
        const std::string tail = " -- " + why;
        const bool same = ok == real_ok && (ok ? CellsEq(real, mine)
                                               : real_err.size() >= tail.size() && real_err.compare(real_err.size() - tail.size(), tail.size(), tail) == 0);
        if (!ok) ++refused;
        if (!same) {
            ++bad;
            if (first.empty()) first = Show(in) + " real '" + real_err + "' mine '" + why + "'";
        }
    }
    Check(bad == 0 && refused == 14, "(G1) ParseStrict(Hook) == HookParseCells on 17 values: the cells, and every ENGINE_CHECK message (" +
                                         U(refused) + " refused, " + U(bad) + " bad" + (first.empty() ? "" : ", first " + first) + ")");
    // (G2) Style::SatCert vs HeckeSatCertDumpParseCells (HeckeSmallCarriesValidation (3))
    cl::Grammar sat;
    sat.style = cl::Style::SatCert;
    sat.fields = 3;
    sat.max_digits = 12;
    sat.ranges = {{2, uint64_t(1) << 20}, {5, (uint64_t(1) << 31) - 1}, {2, 12}};
    sat.accept = [](const cl::Cell& v) { return v[0] % 2 == 0 && IsPrimeNaive(v[1]) && engine::scalar::powfit(v[1], static_cast<uint32_t>(v[2])) != 0; };
    sat.want = "want n:ell:kmax, n even, ell >= 5 prime, 2 <= kmax <= 12";
    const char* sat_in[] = {"168:13:2,220:17:2,258:37:2", "168:13:2,", ",168:13:2", "168:13:2,,220:17:2", "168:13:2:", "", "168:13", "169:13:2",
                            "168:12:2", "168:13:1", "1048576:13:2", "1048578:13:2", "168:2147483647:2", "168:2147483647:3", "1234567890123:13:2",
                            "168:13:12", "168::2", "a:b:c", "168:13:2:4"};
    bad = 0;
    refused = 0;
    first.clear();
    for (const char* in : sat_in) {
        std::string real_why, why;
        const bool real_ok = HeckeSatCertDumpParseCells(in, &real_why);
        std::vector<cl::Cell> mine;
        const bool ok = cl::ParseStrict(in, sat, mine, why);
        if (!ok) ++refused;
        if (ok != real_ok || (!ok && why != real_why)) {
            ++bad;
            if (first.empty()) first = Show(in) + " real '" + real_why + "' mine '" + why + "'";
        }
    }
    Check(bad == 0 && refused == 15, "(G2) ParseStrict(SatCert) == HeckeSatCertDumpParseCells on 19 values: accepted, and the `why` text (" +
                                         U(refused) + " refused, " + U(bad) + " bad" + (first.empty() ? "" : ", first " + first) + ")");
    // (G3) ParseLax vs census::ParseCellList (it reads a variable: a probe variable of this validation's own)
    const char* lax_in[] = {nullptr, "", "654:37,82:7", "654-37,:7,82:7", "654:37,", ",654:37", "1:2:3", "1234567890:1", "123456789:1", "a:b",
                            "654:37,,82:7", " 1:2"};
    bad = 0;
    first.clear();
    for (const char* in : lax_in) {
        EnvSet env("OC_CL1_CELLS_PROBE", in);
        const hecke_ext::census::CellList real = hecke_ext::census::ParseCellList("OC_CL1_CELLS_PROBE");
        const cl::LaxList mine = cl::ParseLax(in);
        if (real.cells != mine.cells || real.bad != mine.bad || real.set != mine.set) {
            ++bad;
            if (first.empty()) first = Show(in);
        }
    }
    const cl::LaxList swallow = cl::ParseLax("654:37,");
    Check(bad == 0 && swallow.cells.size() == 1 && swallow.bad.empty(),
          "(G3) ParseLax == census::ParseCellList on 12 values (cells, bad tokens, set); a trailing ',' is swallowed there (" + U(bad) +
              " bad" + (first.empty() ? "" : ", first " + first) + ")");
    Check(cl::EffectiveLine("HeckeSatCertDumpValidation", "DUMP", 3) == "EFFECTIVE HeckeSatCertDumpValidation arm=DUMP cells=3",
          "(G4) EffectiveLine == the EFFECTIVE convention's text");
}

// ---------------------------------------------------------------------------- (H) (I) (J)
void ArmH() {
    bool ok = true;
    for (uint64_t v : {uint64_t(0), uint64_t(1), uint64_t(0xdeadbeefull), uint64_t(0x0123456789abcdefull), UINT64_MAX}) {
        std::ostringstream padded, bare, minimal;
        padded << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
        bare << std::hex << std::setw(16) << std::setfill('0') << v;
        minimal << "0x" << std::hex << v;
        ok = ok && oc_util::HexU64(v) == padded.str() && oc_util::HexU64(v) == atlas_index::Hex16(v) && oc_util::HexU64(v, false) == bare.str() &&
             oc_util::HexU64(v, true, false) == minimal.str();
    }
    Check(ok, "(H) HexU64: '0x' + 16 digits == atlas_index::Hex16 == the setw(16) copies; no prefix; minimal == BenchDriver's std::hex");
#ifdef OC_SOURCE_DIR
    {
        const std::string lic = std::string(OC_SOURCE_DIR) + "/LICENSE";
        std::string a;
        const bool r = oc_util::ReadFile(lic, a);
        std::ifstream in(lic, std::ios::binary);
        std::ostringstream o;
        o << in.rdbuf();
        std::string none;
        const bool r2 = oc_util::ReadFile(std::string(OC_SOURCE_DIR) + "/no/such/file", none);
        Check(r && !a.empty() && a == o.str() && oc_util::ReadFile(lic) == a && !r2 && none.empty() &&
                  oc_util::ReadFile(std::string(OC_SOURCE_DIR) + "/no/such/file").empty(),
              "(H) ReadFile: the whole file, both shapes; a missing file -> false / \"\"");
    }
#endif
}

void ArmI() {
    const auto t0 = oc_util::SteadyClock::now();
    const double a = oc_util::Seconds(t0), b = oc_util::Seconds(t0);
    Check(a >= 0.0 && b >= a && b < 60.0, "(I) Seconds: non-negative and monotone");
}

// The three copies' splits, transcribed (Gamma0NShapiro.cpp / ModularSymbolMeasure.cpp ParallelChunks, EisensteinMeasure.cpp ParallelFor).
std::vector<std::pair<uint64_t, uint64_t>> CopyRanges(int which, uint64_t total, unsigned threads) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    unsigned T = 0;
    if (which == 0 || which == 1) {
        const uint64_t below = which == 0 ? 512 : 256, per = which == 0 ? 128 : 64;
        if (threads <= 1 || total < below) return {{0, total}};
        T = static_cast<unsigned>(std::min<uint64_t>(threads, total / per + 1));
    } else {
        threads = std::max(1u, std::min<unsigned>(threads, 16));
        if (threads == 1 || total < 4096) return {{0, total}};
        T = threads;
    }
    const uint64_t chunk = (total + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
        const uint64_t s = t * chunk, e = std::min<uint64_t>(total, s + chunk);
        if (s >= e) break;
        out.push_back({s, e});
    }
    return out;
}

void ArmJ() {
    const oc_util::ChunkPlan plans[3] = {oc_util::kShapiroChunks, oc_util::kMeasureChunks, oc_util::kEisensteinChunks};
    uint32_t bad = 0, cases = 0;
    for (int w = 0; w < 3; ++w)
        for (uint64_t total : {uint64_t(0), uint64_t(1), uint64_t(255), uint64_t(256), uint64_t(511), uint64_t(512), uint64_t(1000), uint64_t(4095),
                               uint64_t(4096), uint64_t(4097), uint64_t(100003)})
            for (unsigned threads : {0u, 1u, 2u, 3u, 7u, 16u, 17u, 64u}) {
                ++cases;
                if (oc_util::ChunkRanges(total, threads, plans[w]) != CopyRanges(w, total, threads)) ++bad;
            }
    Check(bad == 0, "(J) ChunkRanges == the three copies' splits: " + U(cases) + " (plan, total, threads) cases (" + U(bad) + " bad)");
    bad = 0;
    for (unsigned threads : {1u, 4u, 16u}) {
        std::mutex mu;
        std::vector<std::pair<uint64_t, uint64_t>> seen;
        oc_util::ParallelChunks(10007, threads, oc_util::kMeasureChunks, [&](uint64_t s, uint64_t e) {
            std::lock_guard<std::mutex> g(mu);
            seen.push_back({s, e});
        });
        std::sort(seen.begin(), seen.end());
        if (seen != oc_util::ChunkRanges(10007, threads, oc_util::kMeasureChunks)) ++bad;
    }
    for (size_t n : {size_t(0), size_t(1), size_t(255), size_t(256), size_t(257), size_t(5000)})
        for (unsigned threads : {0u, 1u, 2u, 7u, 16u, 64u}) {
            std::vector<std::atomic<int>> hit(n);
            for (auto& h : hit) h.store(0);
            oc_util::ParallelForBlocks(n, threads, [&](size_t i) { hit[i].fetch_add(1); });
            for (auto& h : hit)
                if (h.load() != 1) ++bad;
            const size_t want = std::max<size_t>(1, std::min<size_t>({static_cast<size_t>(threads), size_t{16}, n / 256u + 1u}));
            if (oc_util::BlockWorkers(n, threads, 256, 16) != want) ++bad;
        }
    Check(bad == 0, "(J) ParallelChunks runs exactly ChunkRanges; ParallelForBlocks visits every index once and its worker count == "
                    "HeckeAtlas.cpp's formula (" + U(bad) + " bad)");
}

// ---------------------------------------------------------------------------- (K)
[[maybe_unused]] uint32_t LegacyGateProbe() { return 7u; }

void ArmK() {
    namespace lg = oc::legacy;
    std::cout << "  (K) OC_ENABLE_LEGACY=" << (lg::kLegacyBuild ? 1 : 0) << "\n";
    const lv::Policy kRetiredPol{true, lv::Empty::Unset, lv::Match::Exact01, lv::Cache::Once, nullptr, "default", "env", "default"};
    // (K1) the LIVE probe, first (before anything sets the variable): the process environment as given.  In an OFF
    // build `OC_CL1_RETIRED_PROBE=0` exits 1 right here; unset / "1" / invalid read the one route.
    {
        const lv::RetiredLever live("OC_CL1_RETIRED_PROBE", kRetiredPol);
        const bool on = live.enabled();
        std::cout << "  (K1) live " << live.effective_line("route", "new", "old") << "\n";
        Check(lg::kLegacyBuild || (on && std::string(live.source()) == "retired"),
              "(K1) live probe: in an OFF build a retired lever reads its one route, source=retired");
    }
    // (K2) the texts, byte for byte
    Check(lg::compiled_out_text("ENGINE_BACKEND=legacy") == "ENGINE_BACKEND=legacy: compiled out (OC_ENABLE_LEGACY=OFF)" &&
              lg::SkipArmLine("FooValidation", "b") == "SKIP-ARM FooValidation arm=b: OC_ENABLE_LEGACY=OFF",
          "(K2) the refusal text and the SKIP-ARM line");
    // (K3) which reading a retired lever refuses: the valid non-default one, nothing else
    {
        const lv::Policy off_default{false, lv::Empty::Unset, lv::Match::Exact01, lv::Cache::Once, nullptr, "default", "env", "default"};
        const lv::Policy first_char{true, lv::Empty::Unset, lv::Match::FirstChar, lv::Cache::Once, nullptr, "default", "env", "default"};
        const char* inputs[] = {nullptr, "", "0", "1", "00", "x", " 0"};
        const bool want_on[] = {false, false, true, false, false, false, false};    // default-on, Exact01: exactly "0"
        const bool want_off[] = {false, false, false, true, false, false, false};   // default-off: exactly "1"
        const bool want_fc[] = {false, false, true, false, true, false, false};     // FirstChar: "0..."
        uint32_t bad = 0;
        for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
            if (lv::retired_refuses(lv::read(inputs[i], kRetiredPol), kRetiredPol) != want_on[i]) ++bad;
            if (lv::retired_refuses(lv::read(inputs[i], off_default), off_default) != want_off[i]) ++bad;
            if (lv::retired_refuses(lv::read(inputs[i], first_char), first_char) != want_fc[i]) ++bad;
        }
        Check(bad == 0, "(K3) retired_refuses: 7 values x {default-on Exact01, default-off, FirstChar} (" + U(bad) + " bad)");
    }
    // (K4) the mode itself, on a PerCall probe set per value (never to the refused value in an OFF build)
    {
        const lv::Policy percall{true, lv::Empty::Unset, lv::Match::Exact01, lv::Cache::PerCall, nullptr, "default", "env", "default"};
        const lv::RetiredLever r("OC_CL1_RETIRED_PROBE", percall);
        bool ok = true;
#if OC_ENABLE_LEGACY
        static_assert(std::is_same<lv::RetiredLever, lv::EnvLever>::value, "ON: RetiredLever is EnvLever");
        {
            EnvSet e("OC_CL1_RETIRED_PROBE", "0");
            ok = ok && !r.enabled() && std::string(r.source()) == "env";
            const lv::EnvLever::Scoped s(r, true);
            ok = ok && r.enabled() && std::string(r.source()) == "scoped";
        }
        Check(ok, "(K4) ON: a retired lever behaves as the live lever (X=0 selects the old route, Scoped works)");
#else
        for (const char* v : {static_cast<const char*>(nullptr), "1", "x", ""}) {
            EnvSet e("OC_CL1_RETIRED_PROBE", v);
            ok = ok && r.enabled() && r.env_enabled() && std::string(r.source()) == "retired" && r.refusal_text().empty() &&
                 r.effective_line("route", "new", "old") == "EFFECTIVE OC_CL1_RETIRED_PROBE route=new source=retired";
        }
        {
            EnvSet e("OC_CL1_RETIRED_PROBE", "0");   // only the pure query: enabled() would exit 1 here
            ok = ok && r.refusal_text() == "OC_CL1_RETIRED_PROBE=0: compiled out (OC_ENABLE_LEGACY=OFF)";
        }
        Check(ok, "(K4) OFF: a retired lever reads its one route (unset, 1, invalid, empty), source=retired, and refuses exactly X=0");
#endif
    }
    // (K5) OC_LEGACY_VALIDATION: ON = the validation itself, OFF = the [SKIP] stub (exit 0)
    {
        uint32_t (*const f)() = OC_LEGACY_VALIDATION(LegacyGateProbe);   // a ValidationFn
        const uint32_t rc = f();
        Check(rc == (lg::kLegacyBuild ? 7u : 0u), "(K5) OC_LEGACY_VALIDATION: the validation when ON, the [SKIP] stub (0) when OFF");
        const bool arm = lg::RunLegacyArm("SharedHelpersValidation", "K5");
        Check(arm == lg::kLegacyBuild, "(K5) RunLegacyArm: true when ON, SKIP-ARM + false when OFF");
    }
}

} // namespace

uint32_t SharedHelpersValidation() {
    g_c.reset();
    g_rng = 0x5eed0c11ull;
    std::cout << "=== SharedHelpersValidation (S24 CL-1: the shared helpers reproduce the copies they replace) ===\n";
    ArmA();
    ArmB();
    ArmC();
    ArmD();
    ArmE();
    ArmF();
    ArmG();
    ArmH();
    ArmI();
    ArmJ();
    ArmK();
    std::cout << g_c.summary("SharedHelpersValidation") << std::endl;
    return g_c.fails ? 1 : 0;
}
