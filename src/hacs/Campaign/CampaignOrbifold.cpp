// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, L23-1) The orbifold section of the campaign -- see CampaignOrbifold.h.

#include "CampaignOrbifold.h"

#include "../Hecke/HeckeSettings.h"
#include "../HeckeExt/HeckeExtRecord.h"
#include "../NumberField/Sha256.h"
#include "../Integral/Model/IntegralHecke.h"
#include "../Integral/Model/IsotropyModel.h"
#include "../Integral/Model/PeriodicityClass.h"
#include "../Integral/Model/StrictLift.h"
#include "../Orbifold/L23/AmalgamReader.h"
#include "../Orbifold/L23/SumRuleOracle.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/Binary/Ring2k.h"
#include "../../ModArithmetic/Binary/Ring3k.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/EnvLever.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

namespace campaign {

namespace {

using engine::DenseBuf;
using engine::ModCtx;
using U32 = uint32_t;
using Clock = std::chrono::steady_clock;

double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

std::string hex16(uint64_t x) {
    char b[24];
    std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(x));
    return b;
}
std::string fmt(double x, int prec) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.*f", prec, x);
    return b;
}
std::string join(const std::vector<uint32_t>& v, const char* sep = ",") {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += sep;
        s += std::to_string(v[i]);
    }
    return s;
}
std::string jlist(const std::vector<uint32_t>& v) { return "[" + join(v) + "]"; }
std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o + "\"";
}
const char* jb(bool b) { return b ? "true" : "false"; }

bool is_pow2(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

// h^0 = #{(a, b) >= 0 : a (ell + 1) + b ell (ell - 1) = n} (the Dickson degrees 3, 2 resp. 4, 6).
uint32_t h0_closed(uint32_t ell, uint32_t n) {
    const uint32_t A = ell + 1, B = ell * (ell - 1);
    uint32_t c = 0;
    for (uint32_t a = 0; a * A <= n; ++a)
        if ((n - a * A) % B == 0) ++c;
    return c;
}

// v_ell(p^n - 1) for p != ell (exact: p^n mod ell^K, K = 62 / 39, and v < K for every n < 2^31).
uint32_t v_pn_minus_1(uint32_t ell, uint32_t p, uint32_t n) {
    using u128 = unsigned __int128;
    uint64_t M = 1;
    const uint32_t K = (ell == 2) ? 62 : 39;
    for (uint32_t i = 0; i < K; ++i) M *= ell;
    uint64_t x = 1, b = p % M;
    for (uint32_t e = n; e; e >>= 1) {
        if (e & 1) x = static_cast<uint64_t>(static_cast<u128>(x) * b % M);
        b = static_cast<uint64_t>(static_cast<u128>(b) * b % M);
    }
    x = (x + M - 1) % M;
    if (x == 0) return K;
    uint32_t v = 0;
    while (x % ell == 0) { x /= ell; ++v; }
    return v;
}

// The Hatada--Rustom floors of [P3] Sec. 2.3 (LIT; a per-cell integer gate): min finite ed_eis_p.
uint32_t eis_floor(uint32_t ell, uint32_t p) {
    if (ell == 2) return p == 3 ? 7 : p == 5 ? 9 : p == 7 ? 8 : 0;
    return p == 2 ? 1 : p == 5 ? 1 : p == 7 ? 3 : 0;
}

// ---- the shared C_I data: one StrictLiftBundle per Hecke prime, q_max = 3 ------------------------
struct Models {
    std::mutex mu;
    std::map<uint32_t, std::unique_ptr<integral_model::StrictLiftBundle>> bundle;
    std::unique_ptr<integral_model::PeriodicityClass> per;
    const integral_model::StrictLiftBundle& get(uint32_t p) {
        std::lock_guard<std::mutex> g(mu);
        auto it = bundle.find(p);
        if (it == bundle.end()) it = bundle.emplace(p, std::make_unique<integral_model::StrictLiftBundle>(p, 3)).first;
        return *it->second;
    }
    // The reference model (the first bundle's), on which the cohomology and e u - are read.
    const integral_model::IsotropyModel& model() { return get(2).model(); }
    const integral_model::PeriodicityClass& periodicity() {
        const integral_model::IsotropyModel& M = model();
        std::lock_guard<std::mutex> g(mu);
        if (!per) per = std::make_unique<integral_model::PeriodicityClass>(M);
        return *per;
    }
};
Models& models() {
    static Models m;
    return m;
}

DenseBuf<U32> Sub(const ModCtx<U32>& F, const DenseBuf<U32>& A, const DenseBuf<U32>& B) {
    return integral_model::LinComb(F, U32(1), A, F.neg(1), B);
}
uint32_t RankOf(const ModCtx<U32>& F, const DenseBuf<U32>& A) { return integral_model::ColumnLength(F, A); }

JordanType Jordan(const ModCtx<U32>& F, const DenseBuf<U32>& A) {
    JordanType J;
    J.dim = A.rows;
    if (J.dim == 0) return J;
    J.rank = RankOf(F, A);
    uint32_t gen[2] = {0, 0};
    for (U32 lam : {U32(0), U32(1)}) {
        const DenseBuf<U32> X = integral_model::AddScalar(F, A, F.neg(lam % F.p));
        std::vector<uint32_t> r = {J.dim};
        DenseBuf<U32> P = X;
        for (uint32_t j = 1; j <= J.dim; ++j) {
            const uint32_t rj = RankOf(F, P);
            if (rj == r.back()) break;
            r.push_back(rj);
            if (rj == 0) break;
            P = integral_model::MatMul(F, P, X);
        }
        // b_j = r_{j-1} - r_j blocks of size >= j; exactly j: b_j - b_{j+1}.
        std::vector<uint32_t> blocks;
        for (size_t j = 1; j < r.size(); ++j) {
            const uint32_t bj = r[j - 1] - r[j];
            const uint32_t bj1 = (j + 1 < r.size()) ? r[j] - r[j + 1] : 0;
            for (uint32_t t = 0; t < bj - bj1; ++t) blocks.push_back(static_cast<uint32_t>(j));
        }
        std::sort(blocks.rbegin(), blocks.rend());
        gen[lam] = J.dim - r.back();
        (lam == 0 ? J.eig0 : J.eig1) = blocks;
    }
    J.other = J.dim - gen[0] - gen[1];
    return J;
}

std::string Divs(const l23::Divisors& D, uint32_t kernel) {
    std::string s;
    auto add = [&](const std::string& t) { s += (s.empty() ? "" : ",") + t; };
    for (uint32_t x : D.finite) add(std::to_string(x));
    const uint32_t k = std::min(kernel, D.at_least);
    for (uint32_t i = 0; i < k; ++i) add("inf");
    for (uint32_t i = k; i < D.at_least; ++i) add(">=" + std::to_string(D.cap));
    return s.empty() ? "-" : s;
}

JordanType FromJ01(const l23::Jordan01& j) {
    JordanType J;
    J.dim = j.dim;
    J.rank = j.rank;
    J.eig0 = j.eig0;
    J.eig1 = j.eig1;
    J.other = j.other;
    return J;
}
bool SameJordan(const JordanType& a, const JordanType& b) {
    return a.dim == b.dim && a.rank == b.rank && a.eig0 == b.eig0 && a.eig1 == b.eig1 && a.other == b.other;
}
uint64_t FiniteSum(const l23::Divisors& D) {
    uint64_t s = 0;
    for (uint32_t x : D.finite) s += x;
    return s;
}

// `src` = "<class>:<ref>" (see OrbifoldGate, L23-1 audit R2).
void gate(OrbifoldCell& c, const std::string& name, const char* kind, const std::string& src, bool ok, const std::string& detail) {
    const size_t colon = src.find(':');
    OrbifoldGate g{name, kind, src.substr(0, colon), colon == std::string::npos ? std::string() : src.substr(colon + 1), ok, detail};
    ENGINE_CHECK_MSG(!g.source_class.empty() && !g.source_ref.empty(), "orbifold: a gate without a source (L23-1 audit R2)");
    if (!ok) {
        const std::string tag = " [" + std::string(kind) + " " + g.source_class + ": " + g.source_ref + "]";
        if (std::string(kind) == "G") c.failures.push_back(name + tag + ": " + detail + " -- " + OrbifoldGateTriage(g.source_class));
        else c.findings.push_back(name + tag + ": " + detail);
    }
    c.gates.push_back(std::move(g));
}
const JordanType& OrbifoldLayerJordan(const OrbifoldCell& c, uint32_t q, const std::string& layer) {
    for (const OrbifoldLayer& L : c.layers)
        if (L.q == q && L.layer == layer) return L.jordan;
    const std::string m = "orbifold: no layer (" + std::to_string(q) + ", " + layer + ") in the record (L23_3_audit.md n1)";
    ENGINE_CHECK_MSG(false, m.c_str());
    return c.layers.front().jordan;   // unreachable
}
std::string eqs(uint64_t got, uint64_t want) { return "got " + std::to_string(got) + ", expected " + std::to_string(want); }

uint32_t effective_k_cap(const CampaignSpec& spec, uint32_t ell) {
    const uint32_t inst = (ell == 2) ? 1024u : 312u;
    return spec.orbifold.k_cap ? std::min(spec.orbifold.k_cap, inst) : inst;
}

}  // namespace

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------
uint32_t OrbifoldAutoMTarget(uint32_t ell, uint32_t n) {
    // [P3] Sec. 3.3 / [MP] Sec. 2.2: max ed_tell <= (n+1)/(ell+1) + 4 (observed), plus a margin; and never below 40,
    // which covers ed_tp / ed_eis (<= 21 / 18 at n <= 200, heuristically <= 30 at n = 700).
    return std::max<uint32_t>((n + 1) / (ell + 1) + 10, 40);
}

uint32_t OrbifoldMTarget(const CampaignSpec& spec, uint32_t ell, uint32_t n) {
    return spec.orbifold.m_target ? spec.orbifold.m_target : OrbifoldAutoMTarget(ell, n);
}

uint64_t OrbifoldFingerprint(const CampaignSpec& spec) {
    const uint64_t mf = spec.math.fingerprint();
    const OrbifoldSpec& o = spec.orbifold;
    if (mf == 0 && o.primes_default && o.m_target == 0 && o.k_cap == 0) return 0;
    std::string t = "orbifold_cell/v1 settings";
    if (mf) t += ";math=" + hex16(mf);
    if (!o.primes_default) t += ";orbifold_primes=" + join(o.primes);
    if (o.m_target) t += ";orbifold_m_target=" + std::to_string(o.m_target);
    if (o.k_cap) t += ";orbifold_k_cap=" + std::to_string(o.k_cap);
    const uint64_t h = engine::FnvBytes(engine::kFnvOffset64, t);
    return h ? h : 1;   // never 0 off the defaults
}

std::string OrbifoldCellId(const CampaignSpec& spec, uint32_t ell, uint32_t n) {
    std::string id = "orb_ell" + std::to_string(ell) + "_n" + std::to_string(n);
    const uint64_t f = OrbifoldFingerprint(spec);
    if (f) id += "_s" + hex16(f);
    return id;
}

std::vector<OrbifoldCellPlan> PlanOrbifoldCells(const CampaignSpec& spec) {
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (const SpecSection& sec : spec.sections)
        for (uint32_t n : sec.n_sym)
            for (uint64_t e : sec.ell) seen.insert({static_cast<uint32_t>(e), n});
    std::vector<OrbifoldCellPlan> out;
    for (const auto& [e, n] : seen) out.push_back({e, n, OrbifoldCellId(spec, e, n)});
    return out;
}

void ValidateOrbifoldSpec(const CampaignSpec& spec) {
    const std::string who = "campaign spec " + spec.origin;
    const std::string at = " (`orbifold = c_i`, line " + std::to_string(spec.orbifold.line) + ")";
    if (spec.ext.ext) throw std::runtime_error(who + ": `ext = true` cannot be combined with `orbifold`" + at);
    if (spec.math.coefficient_ring.k != 1)
        throw hecke::NotImplementedInThisBuild(who + ": coefficient_ring=" + hecke::CoefficientRingText(spec.math.coefficient_ring.k) +
                                               " is not honoured by the orbifold section (the reader chooses its own Z/ell^K)" + at);
    if (spec.jobs > 1)
        throw std::runtime_error(who + ": `jobs` must be 1 under `orbifold` (cells run one after another; shard by spec)" + at);
    for (const SpecSection& sec : spec.sections) {
        const std::string sl = " -- the section at line " + std::to_string(sec.first_line);
        if (!sec.p.empty() || !sec.r.empty() || !sec.q.empty())
            throw std::runtime_error(who + ": an orbifold section is per (n_sym, ell) cell; `p`, `r` and `q` are not its axes "
                                           "(set `orbifold_primes`)" + sl);
        for (uint64_t e : sec.ell) {
            if (e != 2 && e != 3)
                throw hecke::NotImplementedInThisBuild(who + ": orbifold = c_i at ell = " + std::to_string(e) +
                                                       ": the orbifold section is ell in {2, 3} only" + sl);
            for (uint32_t n : sec.n_sym) {
                const uint32_t need = OrbifoldMTarget(spec, static_cast<uint32_t>(e), n) + 16;   // + the losses e_A + e_1
                if (need > effective_k_cap(spec, static_cast<uint32_t>(e)))
                    throw hecke::NotImplementedInThisBuild(
                        who + ": the orbifold cell (n_sym = " + std::to_string(n) + ", ell = " + std::to_string(e) + ") needs a working "
                        "precision K ~ " + std::to_string(need) + " > " + std::to_string(effective_k_cap(spec, static_cast<uint32_t>(e))) +
                        " (the reader's instantiated range or orbifold_k_cap)" + sl);
            }
        }
    }
    for (uint32_t p : spec.orbifold.primes)
        if (p >= 65536) throw std::runtime_error(who + ": orbifold_primes must be < 2^16" + at);
}

// ---------------------------------------------------------------------------
// JordanType / OrbifoldCell
// ---------------------------------------------------------------------------
std::string JordanType::text() const {
    if (dim == 0) return "-";
    std::string s;
    if (!eig0.empty()) s += "0:" + jlist(eig0);
    if (!eig1.empty()) s += std::string(s.empty() ? "" : " ") + "1:" + jlist(eig1);
    if (other) s += std::string(s.empty() ? "" : " ") + "other:" + std::to_string(other);
    return s;
}
bool JordanType::rank_one_idempotent() const {
    if (other || eig1 != std::vector<uint32_t>{1}) return false;
    for (uint32_t b : eig0) if (b != 1) return false;
    return true;
}
uint32_t OrbifoldCell::gate_failures() const {
    uint32_t k = 0;
    for (const auto& g : gates) if (!g.ok && g.kind == "G") ++k;
    return k;
}
uint32_t OrbifoldCell::conj_findings() const {
    uint32_t k = 0;
    for (const auto& g : gates) if (!g.ok && g.kind == "G-conj") ++k;
    return k;
}

// ---------------------------------------------------------------------------
// (slot-23 Z2K) The (Z) phase at ell = 2 on the Z/2^k ladder engine::ring2k (ENGINE_API.md Sec. 8.2)
// ---------------------------------------------------------------------------
namespace {
std::atomic<uint64_t> g_ring2k_smith{0}, g_ring2k_kernel{0}, g_engine_smith{0}, g_engine_kernel{0};
// L23_RING2K, default on; exactly "0" = the rollback; (the TER-1 audit F4 pattern) anything else warns and stays on.
// One engine::lever::EnvLever (S24 CL-7, F10).
constexpr engine::lever::Policy kRing2kPolicy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                    engine::lever::Cache::Once,
                                    "WARNING campaign: L23_RING2K='%s' is neither 0 nor 1; the ring2k route stays ON "
                                    "(only exactly 0 turns it off)\n"};
engine::lever::EnvLever g_ring2k("L23_RING2K", kRing2kPolicy);
// The ladder rung for Z/2^k, k <= 32 (u16 lanes for k <= 16: BIN-2 O2, the u8 multiply is emulated).
template<class E>
engine::ring2k::Mat<E> ToRing2k(const DenseBuf<U32>& D) {
    engine::ring2k::Mat<E> M(D.rows, D.cols);
    for (uint32_t i = 0; i < D.rows; ++i)
        for (uint32_t j = 0; j < D.cols; ++j) M.at(i, j) = static_cast<E>(D.at(i, j));
    return M;
}
template<class E>
OrbifoldHowell KernelOnLadder(const DenseBuf<U32>& D, uint32_t k) {
    using Tr = engine::ring2k::Traits<E>;
    const engine::ring2k::HowellForm<E> h = engine::ring2k::kernel_module<Tr>(ToRing2k<E>(D), k);
    OrbifoldHowell out;
    out.H = DenseBuf<U32>(h.H.rows, h.H.cols);
    for (uint32_t i = 0; i < h.H.rows; ++i)
        for (uint32_t j = 0; j < h.H.cols; ++j) out.H.at(i, j) = static_cast<U32>(h.H.at(i, j));
    out.pivots = h.pivots;
    out.pivot_v = h.pivot_v;
    return out;
}
}  // namespace

bool OrbifoldRing2kEnabled() { return g_ring2k.enabled(); }
ScopedOrbifoldRing2k::ScopedOrbifoldRing2k(bool on) : guard_(g_ring2k, on) {}
std::string OrbifoldRing2kEffectiveLine() {
    return std::string("EFFECTIVE l23_ring2k=") + (OrbifoldRing2kEnabled() ? "on" : "off") + " source=" + g_ring2k.source() +
           " (orbifold (Z) phase at ell=2: engine::ring2k smith / kernel_module when on, elim::invariant_factors / kernel_module when off)";
}
OrbifoldRing2kCounters OrbifoldRing2kCalls() {
    OrbifoldRing2kCounters c;
    c.ring2k_smith = g_ring2k_smith.load();
    c.ring2k_kernel = g_ring2k_kernel.load();
    c.engine_smith = g_engine_smith.load();
    c.engine_kernel = g_engine_kernel.load();
    return c;
}

// ---------------------------------------------------------------------------
// (slot-23 Z3K) The (Z) phase at ell = 3 on the Z/3^k rung engine::ring3k (Ring3k.h, ENGINE_API.md Sec. 8.4)
// ---------------------------------------------------------------------------
namespace {
std::atomic<uint64_t> g_mod3k_smith{0}, g_mod3k_kernel{0};
// L23_MOD3K, default on; exactly "0" = the rollback; (the TER-1 audit F4 pattern) anything else warns and stays on.
// One engine::lever::EnvLever (S24 CL-7, F10).
constexpr engine::lever::Policy kMod3kPolicy{true, engine::lever::Empty::Invalid, engine::lever::Match::Exact01,
                                    engine::lever::Cache::Once,
                                    "WARNING campaign: L23_MOD3K='%s' is neither 0 nor 1; the ring3k route stays ON "
                                    "(only exactly 0 turns it off)\n"};
engine::lever::EnvLever g_mod3k("L23_MOD3K", kMod3kPolicy);
OrbifoldHowell FromRing3k(engine::ring3k::HowellForm h) {
    OrbifoldHowell out;
    out.H = std::move(h.H);
    out.pivots = std::move(h.pivots);
    out.pivot_v = std::move(h.pivot_v);
    return out;
}
}  // namespace

bool OrbifoldMod3kEnabled() { return g_mod3k.enabled(); }
ScopedOrbifoldMod3k::ScopedOrbifoldMod3k(bool on) : guard_(g_mod3k, on) {}
std::string OrbifoldMod3kEffectiveLine() {
    return std::string("EFFECTIVE l23_mod3k=") + (OrbifoldMod3kEnabled() ? "on" : "off") + " source=" + g_mod3k.source() +
           " (orbifold (Z) phase at ell=3: engine::ring3k smith / kernel_module when on, elim::invariant_factors / kernel_module when off)";
}
OrbifoldMod3kCounters OrbifoldMod3kCalls() {
    OrbifoldMod3kCounters c;
    c.ring3k_smith = g_mod3k_smith.load();
    c.ring3k_kernel = g_mod3k_kernel.load();
    c.engine_smith = g_engine_smith.load();
    c.engine_kernel = g_engine_kernel.load();
    return c;
}

// ---------------------------------------------------------------------------
// (slot-23 F3L) the F_3 field half at ell = 3 (rho_n recurrence + bit-sliced F_3 elimination)
// ---------------------------------------------------------------------------
namespace {
std::atomic<uint64_t> g_f3l_ternary_cells{0};
}  // namespace

std::string OrbifoldF3lEffectiveLine() {
    const bool on = integral_model::F3lRouteEnabled();
    return std::string("EFFECTIVE l23_f3l=") + (on ? "on" : "off") + " source=" + integral_model::F3lRouteSource() +
           " rho_n=" + (on ? "recurrence" : "cubic") + " ell3_f3_elim=" + (on ? "ternary" : (engine::elim::ternary_f3_enabled() ? "ternary(env)" : "engine")) +
           " (orbifold F_3 field half: rho_n by the O(d^2) column recurrence and, at ell=3, TritMatrix RREF when on; the O(d^3) sum and the engine PLU when off)";
}
OrbifoldF3lCounters OrbifoldF3lCalls() {
    OrbifoldF3lCounters c;
    c.ternary_cells = g_f3l_ternary_cells.load();
    const integral_model::RhoNTotals t = integral_model::rho_n_totals();
    c.rho_recurrence = t.recurrence;
    c.rho_cubic = t.cubic;
    c.rho_fallback = t.fallback;
    return c;
}

std::vector<uint32_t> OrbifoldSmithTorsion(uint32_t ell, uint32_t K, const DenseBuf<U32>& D, bool ladder) {
    // The torsion exponents of colspan(D) over Z/ell^K below K, descending: invariant_factors e < K give K - e;
    // on the ladder, the Smith valuations sigma (colspan = sum Z/2^(K - sigma)) with sigma >= 1 are exactly those K - e.
    std::vector<uint32_t> tors;
    if (D.rows == 0 || D.cols == 0) return tors;
    if (ladder && ell == 3) {
        // (Z3K) the ring3k Smith valuations sigma >= 1 are the exponents K - e of invariant_factors' e < K.
        ENGINE_CHECK_MSG(K >= 2 && K <= engine::ring3k::kMaxK, "OrbifoldSmithTorsion: the ring3k route covers 2 <= K <= 19 (the engine route is composite-only: K = 1 has no rollback)");
        const engine::ring3k::SmithResult S = engine::ring3k::smith(D, K);
        ENGINE_CHECK_MSG(S.k == K, "OrbifoldSmithTorsion: ring3k precision");
        for (uint32_t s : S.sigma) {
            ENGINE_CHECK_MSG(s < K, "OrbifoldSmithTorsion: a ring3k Smith valuation at or above K");
            if (s > 0) tors.push_back(s);
        }
        g_mod3k_smith.fetch_add(1, std::memory_order_relaxed);
    } else if (ladder && ell == 2) {
        ENGINE_CHECK_MSG(K >= 2 && K <= 32, "OrbifoldSmithTorsion: the ring2k route covers 2 <= K <= 32 (the engine route is composite-only: K = 1 has no rollback)");
        const engine::ring2k::SmithResult S =
            K <= 16 ? engine::ring2k::smith<engine::ring2k::Traits<uint16_t>>(ToRing2k<uint16_t>(D), K)
                    : engine::ring2k::smith<engine::ring2k::Traits<uint32_t>>(ToRing2k<uint32_t>(D), K);
        ENGINE_CHECK_MSG(S.k == K, "OrbifoldSmithTorsion: ladder precision");
        for (uint32_t s : S.sigma) {
            ENGINE_CHECK_MSG(s < K, "OrbifoldSmithTorsion: a Smith valuation at or above K");
            if (s > 0) tors.push_back(s);
        }
        g_ring2k_smith.fetch_add(1, std::memory_order_relaxed);
    } else {
        const ModCtx<U32> R = integral_model::RingContext(ell, K);
        for (uint32_t e : engine::elim::invariant_factors(R, engine::cview(D)))
            if (e < K) tors.push_back(K - e);
        g_engine_smith.fetch_add(1, std::memory_order_relaxed);
    }
    std::sort(tors.rbegin(), tors.rend());
    return tors;
}

OrbifoldHowell OrbifoldKernelModule(uint32_t ell, uint32_t E, const DenseBuf<U32>& D, bool ladder) {
    if (ladder && ell == 3) {
        ENGINE_CHECK_MSG(E >= 2 && E <= engine::ring3k::kMaxK, "OrbifoldKernelModule: the ring3k route covers 2 <= E <= 19 (the engine route is composite-only: E = 1 has no rollback)");
        g_mod3k_kernel.fetch_add(1, std::memory_order_relaxed);
        return FromRing3k(engine::ring3k::kernel_module(D, E));
    }
    if (ladder && ell == 2) {
        ENGINE_CHECK_MSG(E >= 2 && E <= 32, "OrbifoldKernelModule: the ring2k route covers 2 <= E <= 32 (the engine route is composite-only: E = 1 has no rollback)");
        g_ring2k_kernel.fetch_add(1, std::memory_order_relaxed);
        return E <= 16 ? KernelOnLadder<uint16_t>(D, E) : KernelOnLadder<uint32_t>(D, E);
    }
    const ModCtx<U32> ZE = integral_model::RingContext(ell, E);
    auto km = engine::elim::kernel_module(ZE, engine::cview(D));
    g_engine_kernel.fetch_add(1, std::memory_order_relaxed);
    OrbifoldHowell out;
    out.H = std::move(km.H);
    out.pivots = std::move(km.pivots);
    out.pivot_v = std::move(km.pivot_v);
    return out;
}

const integral_model::IsotropyModel& OrbifoldReferenceModel() { return models().model(); }
const integral_model::StrictLift& OrbifoldStrictLift(uint32_t p) { return models().get(p).lift(); }

// ---------------------------------------------------------------------------
// One cell
// ---------------------------------------------------------------------------
OrbifoldCell ComputeOrbifoldCell(const CampaignSpec& spec, uint32_t ell, uint32_t n) {
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "ComputeOrbifoldCell: ell in {2, 3}");
    const auto t0 = Clock::now();
    OrbifoldCell c;
    c.ell = ell;
    c.n = n;
    c.id = OrbifoldCellId(spec, ell, n);
    c.formal = (ell == 2 && (n % 2) == 1);
    c.h0_closed = h0_closed(ell, n);
    const bool even = (n % 2) == 0;
    // (slot-23 F3L) at ell = 3 every F_3 field elimination of the cell runs bit-sliced (TER-1's TritMatrix);
    // process-wide for the cell's duration (cells run one at a time); value-neutral (unique RREF).
    std::optional<engine::elim::ScopedTernaryF3> f3l_ternary;
    if (ell == 3 && integral_model::F3lRouteEnabled()) {
        f3l_ternary.emplace(true);
        g_f3l_ternary_cells.fetch_add(1, std::memory_order_relaxed);
    }
    std::vector<uint32_t> hp = spec.orbifold.primes;
    if (std::find(hp.begin(), hp.end(), ell) == hp.end()) hp.push_back(ell);
    std::sort(hp.begin(), hp.end());
    try {
        // ---------------- (F) the F_ell part on C_I ------------------------------------------------
        const auto tf = Clock::now();
        Models& MM = models();
        const integral_model::IsotropyModel& M = MM.model();
        for (uint32_t p : hp)
            for (uint32_t q = 0; q <= 4; ++q)
                ENGINE_CHECK_MSG(MM.get(p).model().rank(q) == M.rank(q), "orbifold: bundle models differ");
        const ModCtx<U32> F(ell);
        auto lap_t = Clock::now();
        auto lap = [&](const char* what) {
            const auto t = Clock::now();
            c.log.push_back(std::string("PROFILE ") + what + " " + fmt(secs(lap_t, t), 3) + " s");
            lap_t = t;
        };
        const integral_model::ModelCohomology<U32> MF(M, F, n);
        lap("model_cohomology_F");
        // H^q bases (complements of B^q in Z^q), then T_p one prime at a time (peak memory: one Theta^ set), only
        // ever applied to the h^q basis columns; one elimination per degree reads every T_p matrix at once.
        std::vector<integral_model::InducedOperator<U32>> N(4);
        for (uint32_t q = 0; q <= 3; ++q) {
            c.h[q] = MF.length_H(q);
            N[q] = integral_model::Subquotient(F, MF.Z(q), MF.B(q));
            ENGINE_CHECK_MSG(N[q].rank() == c.h[q], "orbifold: H^q basis size");
        }
        const integral_model::PeriodicityClass& P = MM.periodicity();
        const DenseBuf<U32> E0 = P.matrix(F, n, 0), E1 = P.matrix(F, n, 1);
        const DenseBuf<U32> E1b = integral_model::MatMul(F, E1, N[1].basis);   // e u (H^1 basis), in C^3
        std::vector<DenseBuf<U32>> TB(4);    // [Theta_p basis_q]_p side by side, per degree
        for (uint32_t q = 0; q <= 3; ++q) TB[q] = DenseBuf<U32>(MF.dim(q), 0);
        DenseBuf<U32> EQ(MF.dim(3), 0);      // [E1 Theta_p basis_1 - Theta_p E1 basis_1]_p
        std::vector<DenseBuf<U32>> ThL12(3); // Theta_ell^q, q = 1, 2 (kept for the layers)
        for (uint32_t p : hp) {
            DenseBuf<U32> E1T;
            for (uint32_t q = 0; q <= 3; ++q) {
                DenseBuf<U32> T = MM.get(p).lift().theta_hat(F, n, q);
                const DenseBuf<U32> Y = integral_model::MatMul(F, T, N[q].basis);
                TB[q] = integral_model::HStack(TB[q], Y);
                if (q == 1) E1T = integral_model::MatMul(F, E1, Y);
                if (q == 3) EQ = integral_model::HStack(EQ, Sub(F, E1T, integral_model::MatMul(F, T, E1b)));
                if (p == ell && (q == 1 || q == 2)) ThL12[q] = std::move(T);
            }
        }
        c.e_cup_equivariant = EQ.cols == 0 || RankOf(F, integral_model::HStack(MF.B(3), EQ)) == MF.length_B(3);
        lap("theta");
        // coordinates of every T_p basis column in [basis | B]: one solve per degree
        std::map<uint32_t, std::vector<DenseBuf<U32>>> A;   // p -> [A_p^q]
        for (uint32_t q = 0; q <= 3; ++q) {
            const uint32_t h = c.h[q];
            for (uint32_t p : hp) A[p].push_back(DenseBuf<U32>(h, h));
            if (h == 0) continue;
            const DenseBuf<U32> Bq = MF.B(q).cols ? MF.B(q) : DenseBuf<U32>(MF.dim(q), 1);
            const auto co = engine::elim::solve_multi_rhs(F, engine::cview(integral_model::HStack(N[q].basis, Bq)), engine::cview(TB[q]));
            for (size_t pi = 0; pi < hp.size(); ++pi)
                for (uint32_t j = 0; j < h; ++j)
                    for (uint32_t i = 0; i < h; ++i) A[hp[pi]][q].at(i, j) = co[pi * h + j][i];
        }
        TB.clear();
        for (uint32_t p : hp) {
            if (p == ell) continue;
            bool ok = true;
            const U32 cpl = (1 + p) % ell;
            for (uint32_t q = 0; q <= 3; ++q)
                for (uint32_t i = 0; i < c.h[q]; ++i)
                    for (uint32_t j = 0; j < c.h[q]; ++j) ok = ok && A[p][q].at(i, j) == (i == j ? cpl : 0u);
            c.tp_scalar[p] = ok;
        }
        const std::vector<DenseBuf<U32>> Tl = A[ell];
        for (uint32_t q = 0; q <= 3; ++q) {
            OrbifoldLayer L;
            L.q = q;
            L.layer = "whole";
            L.jordan = Jordan(F, Tl[q]);
            c.t_ell_rank[q] = L.jordan.rank;
            c.layers.push_back(L);
        }
        {
            auto image = [&](uint32_t qd, const DenseBuf<U32>& Y) {
                return Y.cols == 0 ? 0u : RankOf(F, integral_model::HStack(MF.B(qd), Y)) - MF.length_B(qd);
            };
            c.e_cup_h0_h2 = image(2, integral_model::MatMul(F, E0, N[0].basis));
            c.e_cup_h1_h3 = image(3, E1b);
        }
        lap("t_ell_jordan_e_cup");
        c.t_field = secs(tf, Clock::now());

        // ---------------- (Z) the UCT readings over Z/ell^K ----------------------------------------------
        // Torsion of H^q(V) = the torsion of coker(delta^{q-1}) (Z^q is saturated in the free C^q): the Smith
        // invariants of delta^{q-1} over Z/ell^K, certified below K.  K = 31 / 19 at q = 1 (the reader's T^1, exponents
        // <= 9 / 5 at n <= 700), K = E = 4 / 2 at q = 2, 3 (one above the bound 3 / 1 of [C] cl:amalgam(4)).
        const auto tz = Clock::now();
        c.E = (ell == 2) ? 4 : 2;
        // (slot-23 Z2K) at ell = 2 the Smith / Howell readings run on engine::ring2k (equal entrywise, ENGINE_API.md
        // Sec. 8.2; OrbifoldRing2kValidation compares the records route on vs off); L23_RING2K=0 is the rollback.
        // (slot-23 Z3K) at ell = 3 on engine::ring3k (Sec. 8.4; OrbifoldMod3kValidation); L23_MOD3K=0 is the rollback.
        const bool ladder = (ell == 2) ? OrbifoldRing2kEnabled() : OrbifoldMod3kEnabled();
        for (uint32_t q = 1; q <= 3; ++q) {
            const uint32_t Kq = (q == 1) ? (ell == 2 ? 31u : 19u) : c.E;
            const ModCtx<U32> R = integral_model::RingContext(ell, Kq);
            const DenseBuf<U32> D = M.coboundary(R, n, q - 1);
            const std::vector<uint32_t> tors = OrbifoldSmithTorsion(ell, Kq, D, ladder);
            if (q == 1) c.ci_q1_exponents = tors;
            else c.ladic_type[q] = tors;
        }
        lap("smith_readings");
        // (U): H^q(V)/ell inside H^q(M) = the reduction of the cocycles mod ell^E (E - 1 >= exp T^{q+1}), q = 1, 2.
        const ModCtx<U32> ZE = integral_model::RingContext(ell, c.E);
        for (uint32_t q = 1; q <= 2; ++q) {
            const DenseBuf<U32> D = M.coboundary(ZE, n, q);
            const OrbifoldHowell km = OrbifoldKernelModule(ell, c.E, D, ladder);
            DenseBuf<U32> Gr(km.H.cols, km.H.rows);
            for (uint32_t i = 0; i < km.H.rows; ++i)
                for (uint32_t j = 0; j < km.H.cols; ++j) Gr.at(j, i) = km.H.at(i, j) % ell;
            const auto Ns = integral_model::Subquotient(F, Gr, MF.B(q));
            c.sub_dim[q] = Ns.rank();
            OrbifoldLayer Ls, Lq;
            Ls.q = Lq.q = q;
            Ls.layer = "sub";
            Lq.layer = "quot";
            Ls.jordan = Jordan(F, integral_model::OperatorMatrix(F, Ns, MF.B(q), ThL12[q]));
            const DenseBuf<U32> Rl2 = integral_model::HStack(MF.B(q), Gr);
            const auto Nq = integral_model::Subquotient(F, MF.Z(q), Rl2);
            Lq.jordan = Jordan(F, integral_model::OperatorMatrix(F, Nq, Rl2, ThL12[q]));
            c.layers.push_back(Ls);
            c.layers.push_back(Lq);
        }
        ThL12.clear();
        c.t2 = static_cast<uint32_t>(c.ladic_type[2].size());
        c.t3 = static_cast<uint32_t>(c.ladic_type[3].size());
        lap("layers");
        c.t_ladic = secs(tz, Clock::now());

        // ---------------- (L) the degree-1 amalgam reader --------------------------------------------------
        const auto tr = Clock::now();
        c.m_target = OrbifoldMTarget(spec, ell, n);
        if (n == 0) {
            // H^1(Gamma; Z) = 0: no lattice, no torsion (the reader starts at n = 1).
            c.reader_ran = false;
            c.reader_ok = true;
        } else {
            l23::ReaderOptions o;
            o.ell = ell;
            o.m_target = c.m_target;
            o.cap = c.m_target;
            o.k_cap = effective_k_cap(spec, ell);
            o.primes = hp;
            o.pm = (ell == 3);
            const l23::LatticeCell L = l23::read_lattice(n, o);
            c.reader_ran = true;
            c.reader_ok = L.ok;
            c.K = L.K;
            c.m = L.m;
            c.e_A = L.e_A;
            c.e_1 = L.e_1;
            c.reruns = L.reruns;
            c.rung = L.rung;
            c.rank_lambda = L.rank_lambda;
            c.torsion1 = L.torsion;
            c.log.insert(c.log.end(), L.log.begin(), L.log.end());
            for (const std::string& f : L.failures) c.log.push_back("reader: " + f);
            const bool lat = c.rank_lambda > 0;
            for (const auto& [p, D] : L.ed_tp) {
                c.ed_tp[p] = Divs(D, 0);
                c.fallback_divisors += D.at_least;
                if (lat) c.c_p[p] = D.finite.empty() ? D.cap : D.finite.front();
            }
            for (const auto& [p, D] : L.ed_eis) {
                c.ed_eis[p] = Divs(D, lat ? 1 : 0);
                c.fallback_divisors += D.at_least > 0 ? D.at_least - 1 : 0;
            }
            if (lat) {
                c.ed_tell = Divs(L.ed_tell, 0);
                c.fallback_divisors += L.ed_tell.at_least;
            }
            if (ell == 3 && L.pm_done) {
                c.pm = true;
                c.dim_plus = L.dim_plus;
                c.dim_minus = L.dim_minus;
                for (const auto& [p, D] : L.ed_tp_plus) { c.ed_tp_plus[p] = Divs(D, 0); c.fallback_divisors += D.at_least; }
                for (const auto& [p, D] : L.ed_tp_minus) { c.ed_tp_minus[p] = Divs(D, 0); c.fallback_divisors += D.at_least; }
                for (const auto& [p, D] : L.ed_eis_plus) {
                    c.ed_eis_plus[p] = Divs(D, 1);
                    c.fallback_divisors += D.at_least > 0 ? D.at_least - 1 : 0;
                }
                for (const auto& [p, D] : L.ed_eis_minus) { c.ed_eis_minus[p] = Divs(D, 0); c.fallback_divisors += D.at_least; }
                c.ed_tell_plus = Divs(L.ed_tell_plus, 0);
                c.ed_tell_minus = Divs(L.ed_tell_minus, 0);
                c.fallback_divisors += L.ed_tell_plus.at_least + L.ed_tell_minus.at_least;
                // the cusp halves of T_3: ed(T_3 | Lambda^+) minus its one unit vs ed(T_3 | Lambda^-) (BIN3_audit.md Sec. 4.4)
                std::vector<uint32_t> plus = L.ed_tell_plus.finite;
                auto it = std::find(plus.begin(), plus.end(), 0u);
                if (it != plus.end()) plus.erase(it);
                c.pm_tell_halves_identical = (plus == L.ed_tell_minus.finite && L.ed_tell_plus.at_least == L.ed_tell_minus.at_least);
            }
            // ---- (L) gates ----
            const bool precision_only = !L.ok && L.failures.empty();
            gate(c, "reader_internal_checks", "G", "internal:l23::read_lattice self-checks (sigma(A) vs the exact rank over Q, cocycles)", L.ok || precision_only,
                 L.failures.empty() ? std::string("ok") : L.failures.front());
            if (!precision_only) gate(c, "reader_t_p_commute", "G", "LIT:the Hecke algebra is commutative", L.commute || !lat, L.commute ? "ok" : "the T_p do not commute mod ell^m");
            if (L.ok) {
                const uint32_t dimS = lat ? (c.rank_lambda - 1) / 2 : 0;
                if (lat) {
                    uint32_t units = 0;
                    for (uint32_t x : L.ed_tell.finite) units += (x == 0);
                    gate(c, "ed_tell_unit_count", "G", "LIT:Hida control; S_2(Gamma_1(4)) = S_2(Gamma_1(9)) = 0", units == 1, eqs(units, 1));
                    for (const auto& [p, D] : L.ed_eis)
                        gate(c, "ed_eis_kernel_p" + std::to_string(p), "G", "LIT:the Eisenstein class, T_p E = (1 + p^{n+1}) E", D.at_least >= 1,
                             "the Eisenstein line: at_least " + std::to_string(D.at_least) + " >= 1");
                    for (const auto& [p, D] : L.ed_tp) {
                        const uint32_t mn = D.finite.empty() ? D.cap : D.finite.front();
                        gate(c, "ed_tp_no_unit_p" + std::to_string(p), "G", "C:prop:ltwothreescalar", mn >= 1, "min ed_tp = " + std::to_string(mn));
                        const l23::Divisors& Ee = L.ed_eis.at(p);
                        const uint32_t me = Ee.finite.empty() ? Ee.cap : Ee.finite.front();
                        const uint32_t want = std::min({v_pn_minus_1(ell, p, n), me, D.cap});
                        gate(c, "cp_rule_p" + std::to_string(p), "G", "C:lem:cprule", std::min(mn, D.cap) == want,
                             "c_p = " + std::to_string(mn) + ", min(v(p^n - 1), min ed_eis) = " + std::to_string(want) + " ([C] lem:cprule)");
                        const uint32_t fl = eis_floor(ell, p);
                        if (fl && !Ee.finite.empty())
                            gate(c, "eis_floor_p" + std::to_string(p), "G", "LIT:Hatada; Rustom (the Eisenstein congruence floors)", Ee.finite.front() >= fl,
                                 "min finite ed_eis = " + std::to_string(Ee.finite.front()) + " >= " + std::to_string(fl) + " (Hatada--Rustom, LIT)");
                    }
                }
                if (ell == 3 && c.pm && even && n > 0) {
                    gate(c, "pm_dims", "G", "LIT:Eichler--Shimura (dim Lambda^+ = dim S_k + 1, dim Lambda^- = dim S_k)", c.dim_plus == dimS + 1 && c.dim_minus == dimS,
                         "dim Lambda^+ = " + std::to_string(c.dim_plus) + ", dim Lambda^- = " + std::to_string(c.dim_minus) + ", dim S_k = " +
                             std::to_string(dimS));
                    if (n % 6 == 4)
                        gate(c, "pm_tell_halves_n4mod6", "G-conj", "CONJ:+-C (BIN3_audit.md Sec. 4.4)", c.pm_tell_halves_identical,
                             "ed(T_3|Lambda^+_!) = ed(T_3|Lambda^-) (CONJ +-C, BIN3_audit.md Sec. 4.4)");
                }
                // ---- (S) the level-1 q-expansion sum rules (L23-3; census_l23_plan.md Sec. 4.3) ----
                if (lat) {
                    const auto ts = Clock::now();
                    const l23::SumRuleRow S = l23::sum_rule_oracle(ell, n, hp);
                    c.sumrules_ran = true;
                    c.sumrules_resolved = S.resolved;
                    c.sumrule_K = S.K;
                    c.sumrule_max_invariant = S.max_invariant;
                    c.vdet_tell = S.vdet_tell;
                    c.vdet_tp = S.vdet_tp;
                    c.vdet_eis = S.vdet_eis;
                    c.v_pn1 = S.v_pn1;
                    c.log.insert(c.log.end(), S.log.begin(), S.log.end());
                    // (L23_3_audit.md n7) a precision shortfall at k_cap is not a refutation: no G gate (it would stop the campaign),
                    // a SUMRULE_UNRESOLVED log line and a record finding instead; the rules below are then not gated
                    if (S.resolved) {
                        // (orbifold_cell/v2, L23-H1) the oracle's wall time ("... -> resolved (0.274 s)") is not a value: it leaves the hashed
                        // gate detail and goes to the unhashed timings_s.sumrule_oracle (c.sumrule_oracle_s, the digits as printed)
                        std::string det = S.log.empty() ? std::string("-") : S.log.back();
                        const size_t lp = det.rfind(" (");
                        if (lp != std::string::npos && det.size() > lp + 5 && det.compare(det.size() - 3, 3, " s)") == 0) {
                            c.sumrule_oracle_s = det.substr(lp + 2, det.size() - 3 - (lp + 2));
                            det.erase(lp);
                        }
                        gate(c, "sumrule_oracle_resolved", "G", "internal:l23::sum_rule_oracle (every Smith invariant on S_k below K)", true, det);
                    } else {
                        const std::string u = "SUMRULE_UNRESOLVED ell=" + std::to_string(ell) + " n=" + std::to_string(n) + " K=" + std::to_string(S.K) +
                                              (S.log.empty() ? std::string() : " -- " + S.log.back());
                        c.log.push_back(u);
                        c.findings.push_back(u);
                    }
                    const std::string src = "LIT:Eichler--Shimura; T_p on level-1 q-expansions (census_l23_plan.md Sec. 4.3)";
                    auto rule = [&](const std::string& name, uint64_t got, uint64_t want, const std::string& what) {
                        gate(c, name, "G", src, got == want, what + ": " + eqs(got, want));
                    };
                    // (L23_3_audit.md m2) a rule whose divisor list is at the cap is not gated: say so, and count it (data)
                    auto skipped = [&](const std::string& name, uint32_t at_least) {
                        ++c.sumrules_skipped;
                        c.log.push_back("SUMRULE_SKIPPED " + name + " at_least=" + std::to_string(at_least) + " cap=" + std::to_string(c.m_target));
                    };
                    if (S.resolved) {
                        if (L.ed_tell.at_least == 0)
                            rule("sumrule_t_ell", FiniteSum(L.ed_tell), 2ull * S.vdet_tell, "sum ed_tell vs 2 v det(T_ell | S_k)");
                        else
                            skipped("sumrule_t_ell", L.ed_tell.at_least);
                        for (const auto& [p, D] : L.ed_tp)
                            if (D.at_least == 0)
                                rule("sumrule_tp_p" + std::to_string(p), FiniteSum(D), 2ull * S.vdet_tp.at(p) + S.v_pn1.at(p),
                                     "sum ed_tp vs 2 v det(T_p - (1+p) | S_k) + v(p^n - 1)");
                            else
                                skipped("sumrule_tp_p" + std::to_string(p), D.at_least);
                        for (const auto& [p, D] : L.ed_eis)
                            if (D.at_least == 1)
                                rule("sumrule_eis_p" + std::to_string(p), FiniteSum(D), 2ull * S.vdet_eis.at(p),
                                     "sum of the finite ed_eis vs 2 v det(T_p - (1+p^{n+1}) | S_k)");
                            else
                                skipped("sumrule_eis_p" + std::to_string(p), D.at_least);
                        if (ell == 3 && c.pm && L.pm_done) {
                            // Lambda^- ~ S_k, Lambda^+ ~ S_k + the Eisenstein line (over Q): v det on S_k, + v(lambda_Eis) on Lambda^+
                            const std::string hs = "LIT:Eichler--Shimura on the iota halves (census_l23_plan.md Sec. 4.3)";
                            auto half = [&](const std::string& name, const l23::Divisors& D, uint32_t at_least, uint64_t want, const std::string& what) {
                                if (D.at_least == at_least) gate(c, name, "G", hs, FiniteSum(D) == want, what + ": " + eqs(FiniteSum(D), want));
                                else skipped(name, D.at_least);
                            };
                            half("sumrule_t_ell_plus", L.ed_tell_plus, 0, S.vdet_tell, "sum ed(T_3 | Lambda^+) vs v det(T_3 | S_k)");
                            half("sumrule_t_ell_minus", L.ed_tell_minus, 0, S.vdet_tell, "sum ed(T_3 | Lambda^-) vs v det(T_3 | S_k)");
                            for (const auto& [p, D] : L.ed_tp_plus)
                                half("sumrule_tp_plus_p" + std::to_string(p), D, 0, S.vdet_tp.at(p) + S.v_pn1.at(p),
                                     "sum ed(T_p - (1+p) | Lambda^+) vs v det + v(p^n - 1)");
                            for (const auto& [p, D] : L.ed_tp_minus)
                                half("sumrule_tp_minus_p" + std::to_string(p), D, 0, S.vdet_tp.at(p), "sum ed(T_p - (1+p) | Lambda^-) vs v det");
                            for (const auto& [p, D] : L.ed_eis_plus)
                                half("sumrule_eis_plus_p" + std::to_string(p), D, 1, S.vdet_eis.at(p), "sum of the finite ed_eis on Lambda^+ vs v det");
                            for (const auto& [p, D] : L.ed_eis_minus)
                                half("sumrule_eis_minus_p" + std::to_string(p), D, 0, S.vdet_eis.at(p), "sum ed_eis on Lambda^- vs v det");
                        }
                    }
                    c.t_sumrules = secs(ts, Clock::now());
                }
            }
        }
        c.t_reader = secs(tr, Clock::now());
        c.t1 = static_cast<uint32_t>(c.torsion1.size());

        // ---------------- (F) / (Z) gates -----------------------------------------------------------------
        const bool l2 = (ell == 2);
        gate(c, "h0_closed_form", "G", "LIT:Dickson (the invariants of SL_2(F_ell), degrees ell + 1, ell (ell - 1))", c.h[0] == c.h0_closed, eqs(c.h[0], c.h0_closed));
        if (l2) {
            for (uint32_t q = 1; q <= 3; ++q)
                gate(c, "h" + std::to_string(q) + "_closed_form", "G", even ? "R:Thm A" : "C:prop:oddnltwo(3)", c.h[q] == n / 2 + 1, eqs(c.h[q], n / 2 + 1));
        } else if (!even) {
            for (uint32_t q = 0; q <= 3; ++q)
                gate(c, "h" + std::to_string(q) + "_zero_odd_n", "G", "C:lem:centralminusone", c.h[q] == 0, eqs(c.h[q], 0) + " ([C] lem:centralminusone)");
        } else {
            const uint32_t w = (n % 6 == 0 || n % 6 == 4) ? 1 : 0;
            for (uint32_t q = 2; q <= 3; ++q)
                gate(c, "h" + std::to_string(q) + "_closed_form", "G", "R:Thm C(1)", c.h[q] == w, eqs(c.h[q], w));
        }
        for (const auto& [p, ok] : c.tp_scalar)
            gate(c, "tp_scalar_p" + std::to_string(p), "G", "C:prop:ltwothreescalar", ok, "T_p - (1 + p) induces 0 on H^0..H^3 ([C] prop:ltwothreescalar)");
        if (l2 || even) {
            const uint32_t b1 = 2, bq = l2 ? 2u : 1u;
            bool rk = c.t_ell_rank[0] <= 1 && c.t_ell_rank[1] <= b1;
            for (uint32_t q = 2; q <= 3; ++q) rk = rk && c.t_ell_rank[q] <= bq;
            gate(c, "t_ell_rank_bound", "G", "C:lem:tellrank", rk,
                 "ranks " + std::to_string(c.t_ell_rank[0]) + "," + std::to_string(c.t_ell_rank[1]) + "," + std::to_string(c.t_ell_rank[2]) + "," +
                     std::to_string(c.t_ell_rank[3]) + " ([C] lem:tellrank)");
        }
        if (n > 0 && c.h[0] > 0) {
            gate(c, "t_ell_sq_zero_h0", "G", l2 ? "R:Thm B(2)" : "R:Thm D(2)", RankOf(F, integral_model::MatMul(F, Tl[0], Tl[0])) == 0, std::string("T_ell^2 = 0 on H^0 (") + (l2 ? "[R] Thm B(2)" : "[R] Thm D(2)") + "; [C] lem:tellrank gives only rank <= 1)");
        }
        if (l2 && n > 0) {
            const uint32_t want = (even && !is_pow2(n)) ? 1 : 0;
            gate(c, "t_ell_h0_rank", "G", "R:Thm B(2)", c.t_ell_rank[0] == want, eqs(c.t_ell_rank[0], want) + " (rank T_2 on H^0 = [n even, n != 2^k])");
        }
        if (c.h[1] > 0)
            gate(c, "t_ell_h1_rank_one_idempotent", "G-conj", "CONJ:[R] Thm B(2) / D(2) OBSERVED; [C] lem:tellrank does not assert it", c.layers[1].jordan.rank_one_idempotent(),
                 "T_ell on H^1(M): " + c.layers[1].jordan.text() + " (CONJ, [C] lem:tellrank / prop:oddnhecke(4))");
        gate(c, "periodicity_equivariant", "G", "C:lem:bulkperiodic(3)", c.e_cup_equivariant, "e u - : H^1 -> H^3 commutes with every T_p ([C] lem:bulkperiodic(3))");
        if (l2) {
            gate(c, "periodicity_h1_h3", "G", "C:lem:bulkperiodic(2)", c.e_cup_h1_h3 == c.h[1] && c.h[1] == c.h[3],
                 "rank e u - = " + std::to_string(c.e_cup_h1_h3) + ", h1 = " + std::to_string(c.h[1]) + ", h3 = " + std::to_string(c.h[3]));
            // (L23-1 audit R5) e u - : H^0 -> H^2 is injective at ell = 2 for every n ([R] Thm B(4), atom by atom).
            if (c.h[0] > 0)
                gate(c, "e_cup_h0_h2_injective", "G", "R:Thm B(4)", c.e_cup_h0_h2 == c.h[0],
                     "rank e u - on H^0 = " + std::to_string(c.e_cup_h0_h2) + ", h0 = " + std::to_string(c.h[0]));
        } else {
            gate(c, "periodicity_h1_onto_h3", "G", "C:lem:bulkperiodic(2)", c.e_cup_h1_h3 == c.h[3], eqs(c.e_cup_h1_h3, c.h[3]));
            const uint32_t w = even && (n % 6 == 0 || n % 6 == 4) ? 1 : 0;
            gate(c, "dim_coker_eps", "G", "R:Thm C(1), Thm E", c.e_cup_h1_h3 == w, eqs(c.e_cup_h1_h3, w) + " (H^1 / H^1_Br = H^1 / ker(e u -), [R] Thm C, E)");
            if (even && n > 0 && n % 6 == 0)
                gate(c, "cup_h0_h2", "G-conj", "CONJ:[R] Thm C(5)", c.e_cup_h0_h2 == c.h[2], eqs(c.e_cup_h0_h2, c.h[2]) + " (e u H^0 = H^2, CONJ [R] Thm C(5))");
        }
        // the UCT and the layers
        const uint32_t bound = l2 ? 3u : 1u;
        bool expok = true;
        for (uint32_t q = 2; q <= 3; ++q)
            for (uint32_t e : c.ladic_type[q]) expok = expok && e <= bound;
        gate(c, "ladic_exponent_bound_q2q3", "G", "C:cl:amalgam(4)", expok,
             "T^2 = " + jlist(c.ladic_type[2]) + ", T^3 = " + jlist(c.ladic_type[3]) + " <= " + std::to_string(bound) + " ([C] cl:amalgam(4))");
        gate(c, "uct_h1", "G", "UCT:h1 = dim H^1(V)/ell + t2", c.h[1] == c.sub_dim[1] + c.t2, eqs(c.h[1], c.sub_dim[1] + c.t2) + " (h1 = dim H^1(V)/ell + t2)");
        gate(c, "uct_h2", "G", "UCT:h2 = t2 + t3", c.h[2] == c.t2 + c.t3, eqs(c.h[2], c.t2 + c.t3));
        gate(c, "uct_h3", "G", "UCT:h3 = t3 + t4, t4 = t2 by [C] lem:bulkperiodic", c.h[3] == c.t3 + c.t2, eqs(c.h[3], c.t3 + c.t2) + " (T^4 = T^2, [C] lem:bulkperiodic)");
        gate(c, "sub_dim2_eq_t2", "G", "UCT:dim H^2(V)/ell = t2 (H^2(V) is torsion)", c.sub_dim[2] == c.t2, eqs(c.sub_dim[2], c.t2));
        if (c.reader_ok) {
            gate(c, "rank_lambda", "G", "LIT:Eichler--Shimura (rank H^1(Gamma; Sym^n Z^2) = 2 dim M_{n+2} - 1)", c.rank_lambda == (n == 0 ? 0 : l23::expected_rank_lambda(n)),
                 eqs(c.rank_lambda, n == 0 ? 0 : l23::expected_rank_lambda(n)) + " (2 dim M_{n+2} - 1)");
            if (n > 0) gate(c, "t1_eq_h0", "G", "C:lem:bocksteinallell", c.t1 == c.h[0], eqs(c.t1, c.h[0]) + " ([C] lem:bocksteinallell)");
            gate(c, "sub_dim1", "G", "cross-model:C_I dim H^1(V)/ell vs the reader rank + t1", c.sub_dim[1] == c.rank_lambda + c.t1, eqs(c.sub_dim[1], c.rank_lambda + c.t1));
            gate(c, "ci_vs_reader_q1", "G", "cross-model:C_I Smith form of delta^0 vs the amalgam reader T^1", c.torsion1 == c.ci_q1_exponents,
                 "C_I torsion of H^1(V) (Smith of delta^0) = " + jlist(c.ci_q1_exponents) + ", reader " + jlist(c.torsion1));
        }
        if (l2 && even && c.reader_ok)
            gate(c, "rank_lambda_plus_t1_eq_t3", "G", "UCT:with [C] lem:bulkperiodic(2)", c.rank_lambda + c.t1 == c.t3, eqs(c.rank_lambda + c.t1, c.t3));
        if (!l2 && even) {
            gate(c, "t2_closed_form", "G", "R:Thm C(2)", c.t2 == (n % 6 == 0 ? 1u : 0u), eqs(c.t2, n % 6 == 0 ? 1 : 0) + " ([R] Thm C(2))");
            gate(c, "t3_closed_form", "G", "R:Thm C(2)", c.t3 == (n % 6 == 4 ? 1u : 0u), eqs(c.t3, n % 6 == 4 ? 1 : 0) + " ([R] Thm C(2))");
        }
        if (c.formal) {
            const uint32_t m = (n + 1) / 2 - c.h0_closed;
            bool elem = true;
            for (uint32_t q = 2; q <= 3; ++q)
                for (uint32_t e : c.ladic_type[q]) elem = elem && e == 1;
            for (uint32_t e : c.torsion1) elem = elem && e == 1;
            gate(c, "oddn_groups", "G", "C:prop:oddnltwo(3)", c.t1 == c.h0_closed && c.t3 == c.h0_closed && c.t2 == m && elem && c.rank_lambda == 0,
                 "t1 = " + std::to_string(c.t1) + ", t3 = " + std::to_string(c.t3) + " (h0 = " + std::to_string(c.h0_closed) + "), t2 = " +
                     std::to_string(c.t2) + " (m = " + std::to_string(m) + "), exponent 1: " + jb(elem) + " ([C] prop:oddnltwo(3))");
            const OrbifoldLayer& s1 = c.layers[4];   // q = 1 sub
            gate(c, "oddn_t2_zero_h1V", "G", "C:prop:oddnhecke(2)", s1.jordan.rank == 0, "T_2 on H^1(V)/2: " + s1.jordan.text() + " ([C] prop:oddnhecke(2))");
            const OrbifoldLayer& s2 = c.layers[6];   // q = 2 sub
            if (s2.jordan.dim > 0)
                gate(c, "oddn_t2_h2V_rank_one_idempotent", "G-conj", "CONJ:[C] prop:oddnhecke(4)", s2.jordan.rank_one_idempotent(),
                     "T_2 on H^2(V)/2: " + s2.jordan.text() + " (CONJ [C] prop:oddnhecke(4))");
        }

        // ---------------- (A) the independent amalgam oracle (L23-1 audit R3) -----------------------------------
        // l23::h1_mod reads H^1(Gamma; V/ell^k) on the amalgam, with no code shared with C_I; the specification is
        // docs/implementation/slot22/fl231/compare_h1mod.py.  Two models of the same group: a mismatch is a bug.
        if (n > 0 && (l2 || even)) {
            const auto ta = Clock::now();
            const l23::ModCell c1 = l23::h1_mod(ell, n, 1, {{ell, 0}, {ell, 1}});
            ENGINE_CHECK_MSG(c1.image_length.size() == 2, "orbifold: h1_mod image lengths");
            c.amalgam_ran = true;
            c.amalgam_h0 = c1.h0;
            c.amalgam_h1 = c1.length;
            c.amalgam_rank_t_ell = c1.image_length[0];
            c.amalgam_rank_t_ell_minus_1 = c1.image_length[1];
            c.amalgam_L.push_back(c1.length);
            const uint32_t kmax = l2 ? 4 : 2;
            for (uint32_t k = 2; k <= kmax; ++k) c.amalgam_L.push_back(l23::h1_mod(ell, n, k, {}).length);
            const JordanType& J1 = c.layers.at(1).jordan;
            const std::string src = "cross-model:l23::h1_mod (amalgam) vs C_I";
            gate(c, "amalgam_h0", "G", src, c.amalgam_h0 == c.h[0], eqs(c.h[0], c.amalgam_h0) + " (C_I vs amalgam)");
            gate(c, "amalgam_h1", "G", src, c.amalgam_h1 == c.h[1], eqs(c.h[1], c.amalgam_h1) + " (C_I vs amalgam)");
            gate(c, "amalgam_t_ell_rank", "G", src, c.amalgam_rank_t_ell == J1.rank,
                 "rank T_ell on H^1(M): " + eqs(J1.rank, c.amalgam_rank_t_ell) + " (C_I vs amalgam)");
            const uint32_t rm1 = J1.dim - static_cast<uint32_t>(J1.eig1.size());
            gate(c, "amalgam_t_ell_minus_one_rank", "G", src, c.amalgam_rank_t_ell_minus_1 == rm1,
                 "rank (T_ell - 1) on H^1(M): " + eqs(rm1, c.amalgam_rank_t_ell_minus_1) + " (C_I vs amalgam)");
            if (c.reader_ok) {
                std::vector<uint32_t> want;
                for (uint32_t k = 1; k <= kmax; ++k) {
                    uint32_t L = k * c.rank_lambda;
                    for (uint32_t a : c.torsion1) L += std::min(k, a);
                    for (uint32_t b : c.ladic_type[2]) L += std::min(k, b);
                    want.push_back(L);
                }
                gate(c, "amalgam_L_k", "G", src, want == c.amalgam_L,
                     "k rank + sum min(k, T^1) + sum min(k, T^2) = " + jlist(want) + ", amalgam L_k = " + jlist(c.amalgam_L));
            }
            lap("amalgam_oracle");
            c.t_amalgam = secs(ta, Clock::now());
        }

        // ---------------- (B/!/L) the layers of H^1(M) on the amalgam (L23-3; L23_1_audit.md Sec. 7 items 4-6) -----------
        // l23::h1_layers: (B) = H^1_Br, (U) = H^1(V)/ell and T^1/ell (Bockstein), (!) = H^1_!, (L) at ell = 3 -- over F_ell on
        // the amalgam, sharing no code with C_I; no cmpI^* (as heavy as the cell at n = 700, [P3] R5).
        if (n > 0 && (l2 || even)) {
            const auto tl0 = Clock::now();
            uint32_t amax = 0;
            for (uint32_t a : c.torsion1) amax = std::max(amax, a);
            for (uint32_t a : c.ci_q1_exponents) amax = std::max(amax, a);
            const l23::LayerCell lc = l23::h1_layers(ell, n, amax, c.E);
            c.layers_ran = true;
            c.am_dim_br = lc.dim_br;
            c.am_dim_sub = lc.dim_sub;
            c.am_dim_t1 = lc.dim_t1;
            c.am_dim_t1_br = lc.dim_t1_br;
            c.dim_h1_cusp = lc.dim_cusp;
            c.dim_cusp_br = lc.dim_cusp_br;
            c.dim_cusp_sub = lc.dim_cusp_sub;
            c.dim_cusp_t1 = lc.dim_cusp_t1;
            for (const auto& [name, J] : {std::make_pair("br_sub", &lc.j_br), std::make_pair("br_quot", &lc.j_quot), std::make_pair("cusp", &lc.j_cusp)}) {
                OrbifoldLayer Lr;
                Lr.q = 1;
                Lr.layer = name;
                Lr.jordan = FromJ01(*J);
                c.layers.push_back(Lr);
            }
            const std::string xm = "cross-model:l23::h1_layers (amalgam) vs C_I";
            gate(c, "layers_h1", "G", xm, lc.h1 == c.h[1], eqs(c.h[1], lc.h1) + " (C_I vs amalgam)");
            gate(c, "layers_dim_h1V_mod_ell", "G", xm, lc.dim_sub == c.sub_dim[1], "dim H^1(V)/ell: " + eqs(c.sub_dim[1], lc.dim_sub) + " (C_I vs amalgam)");
            gate(c, "layers_t_ell_whole", "G", xm, SameJordan(FromJ01(lc.j_whole), c.layers.at(1).jordan),
                 "T_ell on H^1(M): C_I " + c.layers.at(1).jordan.text() + ", amalgam " + FromJ01(lc.j_whole).text());
            gate(c, "layers_jordan_valid", "G", "internal:l23::h1_layers (T_ell preserves each subquotient; the rank sequences decrease)",
                 lc.j_whole.valid && lc.j_br.valid && lc.j_quot.valid && lc.j_cusp.valid, "Jordan types readable on H^1, H^1_Br, H^1 / H^1_Br, H^1_!");
            gate(c, "layers_bockstein_cocycles", "G", "internal:l23::h1_layers (delta u / ell^a are cocycles mod ell)", lc.t1_cocycles,
                 lc.t1_cocycles ? "ok" : "a Bockstein column is not a cocycle");
            if (c.reader_ok)
                gate(c, "layers_dim_t1", "G", "cross-model:the Bockstein image T^1/ell on the amalgam vs the reader's T^1", lc.dim_t1 == c.t1,
                     "dim T^1/ell: " + eqs(lc.dim_t1, c.t1) + " (amalgam vs reader; [C] lem:bocksteinallell)");
            const uint32_t codim = lc.h1 - lc.dim_br;
            if (l2) {
                gate(c, "br_codim_eq_h0", "G", "R:Thm B(5)", codim == c.h[0], "dim H^1 / H^1_Br: " + eqs(codim, c.h[0]) + " ([R] Thm B(5))");
            } else {
                const uint32_t w = (n % 6 == 0 || n % 6 == 4) ? 1 : 0;
                gate(c, "br_codim", "G", "R:Thm C(1)", codim == w, "dim H^1 / H^1_Br: " + eqs(codim, w) + " ([R] Thm C(1))");
                gate(c, "br_eq_ker_e_cup", "G", "R:Thm E", lc.dim_br == c.h[1] - c.e_cup_h1_h3,
                     "dim H^1_Br (amalgam) = " + std::to_string(lc.dim_br) + ", h1 - rank e u (C_I) = " + std::to_string(c.h[1] - c.e_cup_h1_h3));
            }
            gate(c, "br_hecke_stable", "G", "R:Thm E", lc.br_stable, "T_ell H^1_Br in H^1_Br ([R] Thm E)");
            // (L23_3_audit.md m1) dim H^1_! = 0 on every cell computed (L23_3_report.md Sec. 4(2)): the gate is then vacuous -- say so
            if (lc.dim_cusp == 0) c.log.push_back("CUSP_EMPTY dim_h1_cusp=0 (gate cusp_hecke_stable vacuous)");
            gate(c, "cusp_hecke_stable", "G", "R:Q10 (H^1_! = ker(res to Gamma_inf), canonical and Hecke-stable)", lc.cusp_stable,
                 lc.dim_cusp == 0 ? std::string("vacuous: dim H^1_! = 0") : "T_ell H^1_! in H^1_! ([R] Q10), dim H^1_! = " + std::to_string(lc.dim_cusp));
            if (!l2 && lc.l_done) {
                c.l_done = true;
                c.dim_L = lc.dim_L;
                c.dim_L_br = lc.dim_L_br;
                c.L_in_T1 = lc.L_in_T1;
                c.L_in_sub = lc.L_in_sub;
                c.L_stable = lc.L_stable;
                const uint32_t w = (n % 6 == 0 || n % 6 == 4) ? 1 : 0;
                gate(c, "L_dim", "G", "R:Thm C(3)", lc.dim_L == w, "dim L = dim ker(res to Gamma(3)): " + eqs(lc.dim_L, w) + " ([R] Thm C(3))");
                gate(c, "L_direct_sum", "G", "R:Thm C(3)", lc.dim_L_br == 0 && lc.dim_L + lc.dim_br == lc.h1,
                     "H^1 = H^1_Br + L: dim L cap H^1_Br = " + std::to_string(lc.dim_L_br) + ", dim L + dim H^1_Br = " +
                         std::to_string(lc.dim_L + lc.dim_br) + ", h1 = " + std::to_string(lc.h1));
                // [R] Thm C(4) / [P3] Sec. 2.1 `L_in_T1` read as a SUBSPACE statement ("L in T^1/3 iff n = 4 mod 6") is false: the
                // Bockstein line restricts non-trivially to T^3 in Gamma(3) (L23_3_report.md; l233/t1_check.py), so L is recorded as
                // data (`L.in_T1`).  What C(4) observes integrally (n = 4, 10, 16, 22; "as it must be" at n = 0 mod 6) is that a
                // Dickson class restricts non-trivially to <U> exactly when n = 4 mod 6, i.e. T^1/3 is not inside H^1_Br:
                // dim(T^1/3 cap H^1_Br) = dim T^1/3 - [n = 4 mod 6].  Gated as G-conj (OBSERVED in [R], not proved there).
                const uint32_t want_t1br = lc.dim_t1 - ((n % 6 == 4 && lc.dim_t1 > 0) ? 1u : 0u);
                gate(c, "t1_mod_3_leaves_br_iff_n4mod6", "G-conj", "CONJ:[R] Thm C(4) OBSERVED (n = 4, 10, 16, 22), restated", lc.dim_t1_br == want_t1br,
                     "dim(T^1/3 cap H^1_Br) = " + std::to_string(lc.dim_t1_br) + ", dim T^1/3 = " + std::to_string(lc.dim_t1) + ", expected " +
                         std::to_string(want_t1br));
            }
            lap("amalgam_layers");
            c.t_layers = secs(tl0, Clock::now());
        }
    } catch (const std::exception& e) {
        c.failures.push_back(std::string("exception: ") + e.what());
    }
    if (!c.failures.empty()) c.status = "failed";
    else if (c.fallback_divisors > 0 || !c.reader_ok) c.status = "fallback";
    else c.status = "ok";
    c.t_total = secs(t0, Clock::now());
    return c;
}

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------
std::string OrbifoldSettingsJson(const CampaignSpec& spec) {
    const OrbifoldSpec& o = spec.orbifold;
    std::ostringstream s;
    s << "{\"fingerprint\":" << jstr(hex16(OrbifoldFingerprint(spec))) << ",\"math_fingerprint\":" << jstr(hex16(spec.math.fingerprint()))
      << ",\"orbifold\":{\"model\":\"c_i\",\"primes\":" << jlist(o.primes) << ",\"m_target\":"
      << (o.m_target ? std::to_string(o.m_target) : std::string("\"auto\"")) << ",\"k_cap\":" << o.k_cap << "}}";
    return s.str();
}

std::string OrbifoldGateTriage(const std::string& k) {
    if (k == "C") return "a bug, a refutation of [C], or a misquoted citation";
    if (k == "R") return "a bug, a refutation, a misquoted citation, or an error in the review's proof ([R] is not refereed into [C])";
    if (k == "LIT") return "a bug, a refutation of the literature, or a misquoted reference";
    if (k == "UCT" || k == "cross-model") return "a bug (two readings of the same group disagree)";
    if (k == "internal") return "a bug (the reader's self-check)";
    return "unclassified";
}

std::string OrbifoldProvenanceJson() {
    static const std::string json = [] {
        namespace fs = std::filesystem;
        std::error_code ec;
        const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
        std::string sha = "unknown";
        if (!ec) {
            std::ifstream in(exe, std::ios::binary);
            if (in) {
                nf::Sha256 h;
                std::vector<char> buf(1 << 20);
                while (in) {
                    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
                    if (in.gcount() > 0) h.update(buf.data(), static_cast<size_t>(in.gcount()));
                }
                sha = h.hex();
            }
        }
        return "{\"commit\":" + jstr(hecke_ext::SourceCommit()) + ",\"binary\":" + jstr(ec ? std::string("unknown") : exe.string()) +
               ",\"binary_sha256\":" + jstr(sha) + "}";
    }();
    return json;
}

namespace {
std::string pmap(const std::map<uint32_t, std::string>& m) {
    std::string s = "{";
    bool first = true;
    for (const auto& [p, v] : m) {
        s += (first ? "" : ",") + jstr(std::to_string(p)) + ":" + jstr(v);
        first = false;
    }
    return s + "}";
}
std::string pmapu(const std::map<uint32_t, uint32_t>& m) {
    std::string s = "{";
    bool first = true;
    for (const auto& [p, v] : m) {
        s += (first ? "" : ",") + jstr(std::to_string(p)) + ":" + std::to_string(v);
        first = false;
    }
    return s + "}";
}
std::string pflat(const std::map<uint32_t, std::string>& m) {
    std::string s;
    for (const auto& [p, v] : m) s += (s.empty() ? "" : "|") + std::to_string(p) + ":" + v;
    return s.empty() ? "-" : s;
}
std::string jordan_json(const JordanType& J) {
    return "{\"dim\":" + std::to_string(J.dim) + ",\"rank\":" + std::to_string(J.rank) + ",\"eig0\":" + jlist(J.eig0) + ",\"eig1\":" + jlist(J.eig1) +
           ",\"other\":" + std::to_string(J.other) + ",\"text\":" + jstr(J.text()) + "}";
}
}  // namespace

std::string OrbifoldCensusColumnsJson(const OrbifoldCell& c) {
    // [P3] Sec. 2.4: how the hecke_census/v1 + dataset/v2 columns read at ell in {2, 3}, per maximal ideal of
    // F_ell[T_ell] acting on H^1(Gamma; Sym^n F_ell^2) (the only Hecke algebra there, [C] prop:ltwothreescalar).
    std::ostringstream s;
    s << "{";
    if (c.layers.size() < 8) return "{}";   // a cell that failed before its layers were read
    const JordanType& J = c.layers.at(1).jordan;
    bool firstIdeal = true;
    for (int lam = 1; lam >= 0; --lam) {
        const std::vector<uint32_t>& B = lam ? J.eig1 : J.eig0;
        if (B.empty()) continue;
        uint32_t dimE = 0;
        for (uint32_t b : B) dimE += b;
        const uint32_t sz = B.front();
        bool fr = true;
        for (uint32_t b : B) fr = fr && b == sz;
        std::string hil;
        for (uint32_t i = 0; i < sz; ++i) hil += std::string(i ? "," : "") + "1";
        std::vector<std::pair<std::string, std::pair<std::string, std::string>>> col;
        auto put = [&](const std::string& k, const std::string& kind, const std::string& v) { col.push_back({k, {kind, v}}); };
        const std::string res = (c.ell == 2) ? "1+1" : "1+omega";
        put("ideal_id", "formal", lam ? "ord" : "nonord");
        put("own_group", "formal", lam ? "ord" : "nonord");
        put("groups", "formal", lam ? "ord" : "nonord");
        put("witness_poly", "formal", lam ? "(x-1)^" + std::to_string(sz) : "x^" + std::to_string(sz));
        put("residue_degree", "formal", "1");
        put("residue_degree_certified", "formal", "true");
        put("residue_degree_full", "formal", "1");
        put("dim_E", "formal", std::to_string(dimE));
        put("length_A", "formal", std::to_string(sz));
        put("reduced", "formal", sz == 1 ? "true" : "false");
        put("loewy_A", "formal", std::to_string(sz));
        put("embdim", "formal", sz > 1 ? "1" : "0");
        put("gorenstein", "formal", "true");
        put("hilbert", "formal", hil);
        put("module_D", "formal", jlist(B));
        put("module_length", "formal", std::to_string(dimE));
        put("module_loewy", "formal", std::to_string(sz));
        put("module_loewy_length", "formal", std::to_string(sz));
        put("min_generators", "formal", std::to_string(B.size()));
        put("cyclic", "formal", B.size() == 1 ? "true" : "false");
        put("free", "formal", fr ? "true" : "false");
        for (const char* k : {"label_status", "label_type", "label_system", "label_theta_twist"}) put(k, "formal", "eisenstein:" + res);
        put("residual_type", "formal", res);
        put("residual_status", "formal", "theorem ([C] prop:ltwothreescalar)");
        put("residual_reducible", "formal", "true");
        for (const char* k : {"final", "final_reason", "certificate", "certificate_routes", "certified"})
            put(k, "formal", "theorem gates (orbifold_cell/v1 gates)");
        for (const char* k : {"sturm_B_prime", "sturm_cap", "sturm_proof_range", "sturm_outcome"}) put(k, "formal", "not_needed (F_ell[T_ell])");
        for (const char* k : {"kummer_tag", "kummer_exact_agree", "bernoulli_v", "predicted", "confirmed", "unpredicted", "theta_image_dim",
                              "theta_image", "hidden_congruence", "depth_window", "chi_order", "extremal", "glover_multiplicity", "glover_main",
                              "glover_correction", "glover", "saturated", "mult_one", "ext_sturm_l5", "residual_confidence", "rank_r1"})
            put(k, "not_applicable", "-");
        if (c.ell == 3 && c.pm) {
            put("sign", "computed", "all");
            put("dim_plus", "computed", std::to_string(c.dim_plus));
            put("dim_minus", "computed", std::to_string(c.dim_minus));
        } else {
            for (const char* k : {"sign", "dim_plus", "dim_minus"}) put(k, "not_computed", "-");
        }
        put("boundary_split", "replaced", c.layers_ran ? "dim_h1_cusp = " + std::to_string(c.dim_h1_cusp) : std::string("dim_h1_cusp (not_computed)"));
        put("zl_depth", "replaced", "ladic_type_1 = rank " + std::to_string(c.rank_lambda) + " torsion " + jlist(c.torsion1));
        put("zl_depth_source", "replaced", "l23::read_lattice (BIN-3)");
        for (const char* k : {"edition", "source_set", "record_path", "record_sha256"}) put(k, "ingest", "-");
        put("n", "computed", std::to_string(c.n));
        put("ell", "computed", std::to_string(c.ell));
        put("weight", "computed", std::to_string(c.n + 2));
        s << (firstIdeal ? "" : ",") << jstr(lam ? "ord" : "nonord") << ":{";
        for (size_t i = 0; i < col.size(); ++i)
            s << (i ? "," : "") << jstr(col[i].first) << ":{\"kind\":" << jstr(col[i].second.first) << ",\"value\":" << jstr(col[i].second.second)
              << "}";
        s << "}";
        firstIdeal = false;
    }
    s << "}";
    return s.str();
}

std::string OrbifoldCellBody(const CampaignSpec& spec, const OrbifoldCell& c) {
    std::ostringstream s;
    s << "\"schema\":\"orbifold_cell/v2\",\"cell_id\":" << jstr(c.id) << ",\"ell\":" << c.ell << ",\"n\":" << c.n << ",\"weight\":" << c.n + 2
      << ",\"model\":\"c_i\",\"formal\":" << jb(c.formal) << ",\"status\":" << jstr(c.status);
    // the hash input carries the settings by their fingerprints (MathSettings sources are outside it)
    s << ",\"settings\":" << OrbifoldSettingsJson(spec);
    s << ",\"dims\":{\"h\":[" << c.h[0] << "," << c.h[1] << "," << c.h[2] << "," << c.h[3] << "],\"h0_closed\":" << c.h0_closed
      << ",\"t1\":" << c.t1 << ",\"rank_lambda\":" << c.rank_lambda << ",\"t2\":" << c.t2 << ",\"t3\":" << c.t3 << ",\"dim_h1V_mod_ell\":"
      << c.sub_dim[1] << ",\"dim_h2V_mod_ell\":" << c.sub_dim[2];
    if (c.ell == 2) {
        // [R] Thm A: M = 1^t + P^i + St_2^m, t = [n even], i = h0 - t, m = (n + 1 - c)/2, c = #{j : j = -n mod 3}
        uint32_t cc = 0;
        for (uint32_t j = 0; j <= c.n; ++j) cc += ((j + c.n) % 3 == 0);
        const uint32_t t = (c.n % 2 == 0) ? 1 : 0;
        s << ",\"atoms_t_i_m\":[" << t << "," << (c.h0_closed - t) << "," << (c.n + 1 - cc) / 2 << "]";
    }
    if (c.ell == 3)
        s << ",\"dim_coker_eps\":" << c.e_cup_h1_h3;
    else s << ",\"dim_coker_eps\":{\"kind\":\"formal\",\"value\":" << c.h0_closed << ",\"source\":\"[R] Thm B(5)\"}";
    // (L23-3) H^1_Br computed on the amalgam (both primes; at ell = 3 gated against h1 - rank e u, [R] Thm E)
    if (c.layers_ran)
        s << ",\"dim_h1_br\":{\"kind\":\"computed\",\"value\":" << c.am_dim_br << ",\"source\":\"l23::h1_layers (amalgam); [R] Thm E\"}";
    else if (c.ell == 3)
        s << ",\"dim_h1_br\":{\"kind\":\"derived\",\"value\":" << (c.h[1] - c.e_cup_h1_h3) << ",\"source\":\"[R] Thm E (H^1_Br = ker e u -)\"}";
    else s << ",\"dim_h1_br\":{\"kind\":\"not_computed\"}";
    if (c.layers_ran) {
        s << ",\"dim_h1_cusp\":{\"kind\":\"computed\",\"value\":" << c.dim_h1_cusp << ",\"cap_br\":" << c.dim_cusp_br << ",\"cap_h1V_mod_ell\":"
          << c.dim_cusp_sub << ",\"cap_t1\":" << c.dim_cusp_t1 << ",\"source\":\"l23::h1_layers: ker(H^1 -> H^1(Gamma_inf)), Gamma_inf = +-<T> ([C] def:bdrygroup)\"}"
          << ",\"dim_t1_mod_ell\":" << c.am_dim_t1 << ",\"dim_t1_cap_br\":" << c.am_dim_t1_br;
    } else {
        s << ",\"dim_h1_cusp\":{\"kind\":\"not_computed\"}";
    }
    // L.t_ell_stable is false wherever dim L = 1 (L23_3_audit.md m4): T_3 is a rank-one idempotent with image in H^1_Br and
    // L cap H^1_Br = 0, so T_3 L is not in L -- the splitting H^1 = H^1_Br + L ([R] C(3)) is T_p-stable for p != 3 (Thm D(1)), not T_3-stable
    if (c.l_done)
        s << ",\"L\":{\"dim\":" << c.dim_L << ",\"cap_br\":" << c.dim_L_br << ",\"in_T1\":" << jb(c.L_in_T1) << ",\"in_h1V_mod_ell\":"
          << jb(c.L_in_sub) << ",\"t_ell_stable\":" << jb(c.L_stable) << ",\"source\":\"l23::h1_layers: ker(res to Gamma(3)); [R] Thm C(3), C(4)\"}";
    s << "}";
    s << ",\"mod_hecke\":{\"tp_scalar_ok\":{";
    bool f = true;
    for (const auto& [p, ok] : c.tp_scalar) { s << (f ? "" : ",") << jstr(std::to_string(p)) << ":" << jb(ok); f = false; }
    s << "},\"t_ell_jordan\":[";
    for (uint32_t q = 0; q <= 3 && q < c.layers.size(); ++q) s << (q ? "," : "") << jordan_json(c.layers.at(q).jordan);
    s << "],\"t_ell_h0_rank\":" << c.t_ell_rank[0] << ",\"e_cup_h0_h2_rank\":" << c.e_cup_h0_h2 << ",\"e_cup_h1_h3_rank\":" << c.e_cup_h1_h3
      << ",\"e_cup_equivariant\":" << jb(c.e_cup_equivariant);
    // the (U) triples and obstruction ranks: rank(T|whole) - rank(T|sub) - rank(T|quot)
    s << ",\"t_ell_triple_U\":[";
    for (uint32_t q = 1; q <= 2 && c.layers.size() >= 8; ++q) {
        const JordanType& W = c.layers.at(q).jordan;
        const JordanType& S = c.layers.at(4 + 2 * (q - 1)).jordan;
        const JordanType& Q = c.layers.at(5 + 2 * (q - 1)).jordan;
        s << (q > 1 ? "," : "") << "{\"q\":" << q << ",\"sub\":" << jstr(S.text()) << ",\"quot\":" << jstr(Q.text()) << ",\"whole\":"
          << jstr(W.text()) << ",\"obstruction_rank\":" << static_cast<int64_t>(W.rank) - S.rank - Q.rank << "}";
    }
    s << "]";
    // (L23-3) the (B) triple at q = 1 (sub H^1_Br, quotient H^1 / H^1_Br, whole) and T_ell on H^1_!
    if (c.layers_ran) {
        const JordanType& W = OrbifoldLayerJordan(c, 1, "whole");
        const JordanType& S = OrbifoldLayerJordan(c, 1, "br_sub");
        const JordanType& Q = OrbifoldLayerJordan(c, 1, "br_quot");
        s << ",\"t_ell_triple_B\":{\"q\":1,\"sub\":" << jstr(S.text()) << ",\"quot\":" << jstr(Q.text()) << ",\"whole\":" << jstr(W.text())
          << ",\"obstruction_rank\":" << static_cast<int64_t>(W.rank) - S.rank - Q.rank << "},\"t_ell_cusp\":" << jstr(OrbifoldLayerJordan(c, 1, "cusp").text());
    }
    s << "}";
    s << ",\"ladic\":{\"ladic_type\":[{\"q\":0,\"free_rank\":" << (c.n == 0 ? 1 : 0) << ",\"torsion\":[]},{\"q\":1,\"free_rank\":" << c.rank_lambda
      << ",\"torsion\":" << jlist(c.torsion1) << "},{\"q\":2,\"free_rank\":0,\"torsion\":" << jlist(c.ladic_type[2]) << "},{\"q\":3,\"free_rank\":0,\"torsion\":"
      << jlist(c.ladic_type[3]) << "}],\"ci_torsion_h1\":{\"K\":" << (c.ell == 2 ? 31 : 19) << ",\"exponents\":" << jlist(c.ci_q1_exponents)
      << "},\"E\":" << c.E;
    // (L23-1 audit R3) the amalgam oracle's lengths L_k = log_ell |H^1(Gamma; V/ell^k)| and its T_ell ranks on H^1(M)
    s << ",\"amalgam\":{\"ran\":" << jb(c.amalgam_ran) << ",\"h0\":" << c.amalgam_h0 << ",\"h1\":" << c.amalgam_h1 << ",\"rank_t_ell\":"
      << c.amalgam_rank_t_ell << ",\"rank_t_ell_minus_1\":" << c.amalgam_rank_t_ell_minus_1 << ",\"L_k\":" << jlist(c.amalgam_L) << "}";
    // the certification only (L23-1 audit R4): the precision path (rung, K, e_A, e_1, reruns, m) is in `reader_path`, outside
    s << ",\"reader\":{\"ran\":" << jb(c.reader_ran) << ",\"ok\":" << jb(c.reader_ok) << ",\"m_target\":" << c.m_target << ",\"cap\":"
      << c.m_target << ",\"certified\":" << jb(c.reader_ok && c.fallback_divisors == 0) << "}";
    s << ",\"ed_tp\":" << pmap(c.ed_tp) << ",\"ed_eis\":" << pmap(c.ed_eis) << ",\"c_p\":" << pmapu(c.c_p) << ",\"ed_tell\":"
      << jstr(c.ed_tell.empty() ? "-" : c.ed_tell);
    if (c.ell == 3)
        s << ",\"pm\":{\"done\":" << jb(c.pm) << ",\"dim_plus\":" << c.dim_plus << ",\"dim_minus\":" << c.dim_minus << ",\"ed_tp_plus\":"
          << pmap(c.ed_tp_plus) << ",\"ed_tp_minus\":" << pmap(c.ed_tp_minus) << ",\"ed_eis_plus\":" << pmap(c.ed_eis_plus) << ",\"ed_eis_minus\":"
          << pmap(c.ed_eis_minus) << ",\"ed_tell_plus\":" << jstr(c.ed_tell_plus.empty() ? "-" : c.ed_tell_plus) << ",\"ed_tell_minus\":"
          << jstr(c.ed_tell_minus.empty() ? "-" : c.ed_tell_minus) << ",\"tell_cusp_halves_identical\":" << jb(c.pm_tell_halves_identical) << "}";
    // (L23-3) the level-1 sum rules: v_ell det(X | S_k) (the oracle's K is in `sumrule_path`, outside the body)
    s << ",\"sumrules\":{\"ran\":" << jb(c.sumrules_ran) << ",\"resolved\":" << jb(c.sumrules_resolved) << ",\"vdet_t_ell\":" << c.vdet_tell
      << ",\"vdet_tp\":" << pmapu(c.vdet_tp) << ",\"vdet_eis\":" << pmapu(c.vdet_eis) << ",\"v_pn_minus_1\":" << pmapu(c.v_pn1)
      << ",\"skipped\":" << c.sumrules_skipped << "}";
    s << ",\"fallback_divisors\":" << c.fallback_divisors << "}";
    s << ",\"layers\":[";
    for (size_t i = 0; i < c.layers.size(); ++i)
        s << (i ? "," : "") << "{\"q\":" << c.layers[i].q << ",\"layer\":" << jstr(c.layers[i].layer) << ",\"t_ell\":" << jordan_json(c.layers[i].jordan)
          << "}";
    s << "],\"gates\":[";
    for (size_t i = 0; i < c.gates.size(); ++i)
        s << (i ? "," : "") << "{\"name\":" << jstr(c.gates[i].name) << ",\"kind\":" << jstr(c.gates[i].kind) << ",\"source\":{\"class\":"
          << jstr(c.gates[i].source_class) << ",\"ref\":" << jstr(c.gates[i].source_ref) << "},\"ok\":" << jb(c.gates[i].ok)
          << ",\"detail\":" << jstr(c.gates[i].detail) << "}";
    s << "],\"gate_failures\":" << c.gate_failures() << ",\"conj_findings\":" << c.conj_findings() << ",\"failures\":[";
    for (size_t i = 0; i < c.failures.size(); ++i) s << (i ? "," : "") << jstr(c.failures[i]);
    s << "],\"census_columns\":" << OrbifoldCensusColumnsJson(c);
    return s.str();
}

std::string OrbifoldCellJson(const CampaignSpec& spec, const OrbifoldCell& c) {
    const std::string body = OrbifoldCellBody(spec, c);
    const uint64_t h = engine::FnvBytes(engine::kFnvOffset64, body);
    std::ostringstream s;
    s << "{" << body << ",\"record_fnv1a64\":" << jstr(hex16(h)) << ",\"settings_full\":{\"math\":" << spec.math.to_json() << "}"
      << ",\"log\":[";
    for (size_t i = 0; i < c.log.size(); ++i) s << (i ? "," : "") << jstr(c.log[i]);
    s << "],\"reader_path\":{\"rung\":" << jstr(c.rung) << ",\"K\":" << c.K << ",\"m\":" << c.m << ",\"e_A\":" << c.e_A << ",\"e_1\":" << c.e_1
      << ",\"reruns\":" << c.reruns << "},\"sumrule_path\":{\"K\":" << c.sumrule_K << ",\"max_invariant\":" << c.sumrule_max_invariant
      << "},\"provenance\":" << OrbifoldProvenanceJson();
    s << ",\"timings_s\":{\"field\":" << fmt(c.t_field, 3) << ",\"ladic\":" << fmt(c.t_ladic, 3) << ",\"reader\":" << fmt(c.t_reader, 3)
      << ",\"amalgam\":" << fmt(c.t_amalgam, 3) << ",\"layers\":" << fmt(c.t_layers, 3) << ",\"sumrules\":" << fmt(c.t_sumrules, 3)
      << (c.sumrule_oracle_s.empty() ? std::string() : ",\"sumrule_oracle\":" + c.sumrule_oracle_s) << ",\"total\":" << fmt(c.t_total, 3) << "},\"peak_rss_mb\":" << fmt(c.peak_rss_mb, 1) << ",\"orbifold_cell_complete\":true}\n";
    return s.str();
}

std::string OrbifoldSummaryHeader() {
    return "cell_id\tell\tn\tweight\tmodel\tformal\tstatus\th0\th1\th2\th3\tt1\trank_lambda\tt2\tt3\ttorsion1\tladic_type2\tladic_type3\t"
           "t_ell_jordan_0\tt_ell_jordan_1\tt_ell_jordan_2\tt_ell_jordan_3\ttp_scalar_ok\te_cup_h1_h3\tK\tm\tm_target\ted_tp\ted_eis\tc_p\t"
           "ed_tell\tdim_plus\tdim_minus\ted_tell_plus\ted_tell_minus\tgate_failures\tconj_findings\tfallback\trecord_fnv1a64\ttime_s\trss_mb\tfile";
}

std::string OrbifoldSummaryRow(const OrbifoldCell& c, uint64_t rf) {
    std::ostringstream s;
    bool tp = true;
    for (const auto& kv : c.tp_scalar) tp = tp && kv.second;
    std::string cp;
    for (const auto& [p, v] : c.c_p) cp += (cp.empty() ? "" : "|") + std::to_string(p) + ":" + std::to_string(v);
    auto J = [&](uint32_t q) { return c.layers.size() > q ? c.layers[q].jordan.text() : std::string("-"); };
    s << c.id << "\t" << c.ell << "\t" << c.n << "\t" << c.n + 2 << "\tc_i\t" << (c.formal ? 1 : 0) << "\t" << c.status << "\t" << c.h[0] << "\t"
      << c.h[1] << "\t" << c.h[2] << "\t" << c.h[3] << "\t" << c.t1 << "\t" << c.rank_lambda << "\t" << c.t2 << "\t" << c.t3 << "\t"
      << (c.torsion1.empty() ? "()" : join(c.torsion1, ";")) << "\t" << (c.ladic_type[2].empty() ? "()" : join(c.ladic_type[2], ";")) << "\t"
      << (c.ladic_type[3].empty() ? "()" : join(c.ladic_type[3], ";")) << "\t" << J(0) << "\t" << J(1) << "\t" << J(2) << "\t" << J(3) << "\t"
      << (tp ? "true" : "false") << "\t" << c.e_cup_h1_h3 << "\t" << c.K << "\t" << c.m << "\t" << c.m_target << "\t" << pflat(c.ed_tp) << "\t"
      << pflat(c.ed_eis) << "\t" << (cp.empty() ? "-" : cp) << "\t" << (c.ed_tell.empty() ? "-" : c.ed_tell) << "\t"
      << (c.pm ? std::to_string(c.dim_plus) : "-") << "\t" << (c.pm ? std::to_string(c.dim_minus) : "-") << "\t"
      << (c.pm ? c.ed_tell_plus : "-") << "\t" << (c.pm ? c.ed_tell_minus : "-") << "\t" << c.gate_failures() << "\t" << c.conj_findings() << "\t"
      << c.fallback_divisors << "\t" << hex16(rf) << "\t" << fmt(c.t_total, 2) << "\t" << fmt(c.peak_rss_mb, 1) << "\tcells/" << c.id << ".json";
    return s.str();
}

std::string OrbifoldLayersHeader() { return "cell_id\tell\tn\tq\tlayer\teigenvalue\tdim\tt_ell_jordan"; }

std::string OrbifoldLayersRows(const OrbifoldCell& c) {
    std::ostringstream s;
    for (const OrbifoldLayer& L : c.layers) {
        auto row = [&](const std::string& eig, const std::vector<uint32_t>& B, uint32_t dimx) {
            s << c.id << "\t" << c.ell << "\t" << c.n << "\t" << L.q << "\t" << L.layer << "\t" << eig << "\t" << dimx << "\t" << jlist(B) << "\n";
        };
        uint32_t d0 = 0, d1 = 0;
        for (uint32_t b : L.jordan.eig0) d0 += b;
        for (uint32_t b : L.jordan.eig1) d1 += b;
        if (d0) row("0", L.jordan.eig0, d0);
        if (d1) row("1", L.jordan.eig1, d1);
        if (L.jordan.other) row("other", {}, L.jordan.other);
    }
    return s.str();
}

// ---------------------------------------------------------------------------
// The campaign
// ---------------------------------------------------------------------------
namespace {
// (L23-1 audit R1) A record on disk: complete (the trailer), its `status`, its first failure.
struct CellOnDisk {
    bool complete = false;
    std::string status, first_failure, schema;
};
// The JSON string value starting at s[a] == '"' (escapes kept verbatim but for \" and \\).
std::string JsonStringAt(const std::string& s, size_t a) {
    std::string v;
    if (a >= s.size() || s[a] != '"') return v;
    for (size_t i = a + 1; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) { v += s[++i]; continue; }
        if (s[i] == '"') break;
        v += s[i];
    }
    return v;
}
CellOnDisk ReadCellOnDisk(const std::string& path) {
    CellOnDisk d;
    std::ifstream in(path, std::ios::binary);
    if (!in) return d;
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t tail = text.size() > 256 ? text.size() - 256 : 0;
    d.complete = text.find("\"orbifold_cell_complete\":true", tail) != std::string::npos;
    const std::string sk = "\"status\":";
    const size_t a = text.find(sk);   // the body's first key after cell_id / ell / n / weight / model / formal
    if (a != std::string::npos) d.status = JsonStringAt(text, a + sk.size());
    const std::string ck = "\"schema\":";   // (L23-H1 audit F2) orbifold_cell/v1 or /v2: a resume may meet either
    const size_t sc = text.find(ck);
    if (sc != std::string::npos && sc < 4) d.schema = JsonStringAt(text, sc + ck.size());
    const std::string fk = "\"failures\":[";
    const size_t b = text.find(fk);
    if (b != std::string::npos) d.first_failure = JsonStringAt(text, b + fk.size());
    return d;
}
}  // namespace

OrbifoldCampaignResult RunOrbifoldCampaign(const CampaignSpec& spec, bool verbose) {
    namespace fs = std::filesystem;
    ENGINE_CHECK_MSG(spec.orbifold.enabled, "RunOrbifoldCampaign: `orbifold` is not enabled");
    OrbifoldCampaignResult res;
    res.output_dir = spec.output_dir;
    res.plan = PlanOrbifoldCells(spec);
    std::error_code ec;
    fs::create_directories(spec.output_dir + "/cells", ec);
    if (!fs::is_directory(spec.output_dir + "/cells")) throw std::runtime_error("campaign: cannot create '" + spec.output_dir + "/cells'");
    const uint64_t fp = OrbifoldFingerprint(spec);
    if (verbose) {
        std::cout << "=== orbifold campaign '" << spec.name << "' ===\n"
                  << "EFFECTIVE campaign arm=orbifold model=c_i cells=" << res.plan.size() << " primes=" << join(spec.orbifold.primes)
                  << " m_target=" << (spec.orbifold.m_target ? std::to_string(spec.orbifold.m_target) : std::string("auto"))
                  << " k_cap=" << spec.orbifold.k_cap << " fingerprint=" << hex16(fp) << " resume=" << (spec.resume ? 1 : 0)
                  << " out=" << spec.output_dir << std::endl;
        std::cout << l23::ternary_f3_effective_line() << std::endl;   // (TER-1 audit F4) the route of the ell = 3, k = 1 amalgam oracle
        std::cout << OrbifoldRing2kEffectiveLine() << std::endl;      // (slot-23 Z2K) the route of the ell = 2 (Z) phase
        std::cout << OrbifoldMod3kEffectiveLine() << std::endl;       // (slot-23 Z3K) the route of the ell = 3 (Z) phase
        std::cout << OrbifoldF3lEffectiveLine() << std::endl;         // (slot-23 F3L) the route of the F_3 field half
        std::cout << l23::bin3o1_effective_line() << std::endl;       // (slot-23 BIN3O1) the route of the degree-1 amalgam reader
    }
    const auto t0 = Clock::now();
    std::ofstream sum(spec.output_dir + "/summary.tsv"), lay(spec.output_dir + "/layers.tsv");
    sum << OrbifoldSummaryHeader() << "\n";
    lay << OrbifoldLayersHeader() << "\n";
    std::ostringstream idx;
    idx << "{\"schema\":\"orbifold_campaign/v1\",\"name\":" << jstr(spec.name) << ",\"spec_origin\":" << jstr(spec.origin)
        << ",\"settings\":" << OrbifoldSettingsJson(spec) << ",\"cells\":[";
    bool firstCell = true;
    for (const OrbifoldCellPlan& pl : res.plan) {
        const std::string path = spec.output_dir + "/cells/" + pl.id + ".json";
        const CellOnDisk disk = spec.resume ? ReadCellOnDisk(path) : CellOnDisk{};
        if (disk.complete && disk.status != "ok" && disk.status != "fallback") {
            // [P3] R7 across restarts: a failed record is a bug or a refutation -- never resumed past, never re-run over.
            ++res.failed;
            res.stopped = true;
            res.stop_reason = "(resume on a " + (disk.status.empty() ? std::string("status-less") : disk.status) + " cell; [P3] R7): " + pl.id +
                              ": " + (disk.first_failure.empty() ? std::string("no failure recorded") : disk.first_failure) +
                              " -- remove or rename cells/" + pl.id + ".json to re-run it";
            sum << pl.id << "\t" << pl.ell << "\t" << pl.n << "\t" << pl.n + 2 << "\tc_i\t-\t" << (disk.status.empty() ? "?" : disk.status);
            for (int i = 0; i < 33; ++i) sum << "\t-";
            sum << "\tcells/" << pl.id << ".json\n";
            idx << (firstCell ? "" : ",") << "{\"id\":" << jstr(pl.id) << ",\"ell\":" << pl.ell << ",\"n\":" << pl.n << ",\"status\":"
                << jstr(disk.status.empty() ? "?" : disk.status) << ",\"from\":\"resume\"}";
            firstCell = false;
            if (verbose) std::cout << "  [FAIL] orbifold ell=" << pl.ell << " n=" << pl.n << " (resume: the record says " << disk.status << ")" << std::endl;
            break;
        }
        if (disk.complete) {
            ++res.skipped;
            if (disk.schema != "orbifold_cell/v2") ++res.resumed_not_v2;   // (L23-H1 audit F2) a mixed-format shard
            sum << pl.id << "\t" << pl.ell << "\t" << pl.n << "\t" << pl.n + 2 << "\tc_i\t-\tresumed";
            for (int i = 0; i < 33; ++i) sum << "\t-";
            sum << "\tcells/" << pl.id << ".json\n";
            idx << (firstCell ? "" : ",") << "{\"id\":" << jstr(pl.id) << ",\"ell\":" << pl.ell << ",\"n\":" << pl.n << ",\"status\":\"resumed\"}";
            firstCell = false;
            if (verbose) std::cout << "  [skip] orbifold ell=" << pl.ell << " n=" << pl.n << " (resume)" << std::endl;
            continue;
        }
        const bool rss = oc_util::ResetPeakRss();
        OrbifoldCell c = ComputeOrbifoldCell(spec, pl.ell, pl.n);
        c.peak_rss_mb = oc_util::PeakRssMiB();
        (void)rss;
        const std::string body = OrbifoldCellBody(spec, c);
        const uint64_t rf = engine::FnvBytes(engine::kFnvOffset64, body);
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("campaign: cannot write '" + path + "'");
            out << OrbifoldCellJson(spec, c);
        }
        sum << OrbifoldSummaryRow(c, rf) << "\n";
        lay << OrbifoldLayersRows(c);
        idx << (firstCell ? "" : ",") << "{\"id\":" << jstr(c.id) << ",\"ell\":" << c.ell << ",\"n\":" << c.n << ",\"status\":" << jstr(c.status)
            << ",\"record_fnv1a64\":" << jstr(hex16(rf)) << "}";
        firstCell = false;
        ++res.ran;
        if (c.status == "failed") ++res.failed;
        if (c.status == "fallback") ++res.fallback;
        res.findings += c.conj_findings();
        if (verbose) {
            std::cout << "  [" << (c.status == "ok" ? "ok  " : c.status == "fallback" ? "fbk " : "FAIL") << "] orbifold ell=" << c.ell << " n=" << c.n
                      << " h=(" << c.h[0] << "," << c.h[1] << "," << c.h[2] << "," << c.h[3] << ") rankL=" << c.rank_lambda << " T1=" << jlist(c.torsion1)
                      << " T_ell|H1=" << (c.layers.size() > 1 ? c.layers[1].jordan.text() : std::string("?")) << " K=" << c.K << " m=" << c.m << " gates=" << c.gates.size()
                      << " G_fail=" << c.gate_failures() << " conj=" << c.conj_findings() << " fbk=" << c.fallback_divisors << " "
                      << fmt(c.t_total, 2) << " s " << fmt(c.peak_rss_mb, 0) << " MB" << std::endl;
            for (const std::string& f : c.failures) std::cout << "      [G] " << f << std::endl;
            for (const std::string& f : c.findings) std::cout << "      [finding] " << f << std::endl;
        }
        res.cells.push_back(std::move(c));
        if (res.cells.back().status == "failed") {
            res.stopped = true;
            res.stop_reason = "(a G gate failed; [P3] R7): cell " + res.cells.back().id + ": " +
                              (res.cells.back().failures.empty() ? std::string("failed") : res.cells.back().failures.front());
            break;   // [P3] R7: a G failure is a bug or a refutation -- stop and report, never run past it
        }
    }
    res.wall_s = secs(t0, Clock::now());
    res.peak_rss_mb = oc_util::PeakRssMiB();
    idx << "],\"ran\":" << res.ran << ",\"skipped\":" << res.skipped << ",\"failed\":" << res.failed << ",\"fallback\":" << res.fallback
        << ",\"refused\":" << res.refused << ",\"stopped\":" << jb(res.stopped) << ",\"stop_reason\":" << jstr(res.stop_reason)
        << ",\"provenance\":" << OrbifoldProvenanceJson() << "}\n";
    std::ofstream(spec.output_dir + "/index.json") << idx.str();
    if (verbose) {
        // (L23-H1 audit F2) resume skips a complete orbifold_cell/v1 record as before (its values stand), but the shard then holds two
        // record formats: say so, so that a census ingest converts those records first (docs/implementation/slot24/l23_record_convert.py)
        if (res.resumed_not_v2)
            std::cout << "RESUME_FORMAT resumed_not_v2=" << res.resumed_not_v2 << " (cells kept in an older record format: mixed-format output; "
                      << "convert them with docs/implementation/slot24/l23_record_convert.py before an ingest)" << std::endl;
        if (res.stopped) std::cout << "STOP " << res.stop_reason << std::endl;
        std::cout << l23::bin3o1_route_line() << std::endl;   // (BIN3O1 audit F1) the degree-1 reader's route counters; fallback=0 expected
        std::cout << "EFFECTIVE campaign arm=orbifold cells=" << res.plan.size() << " ran=" << res.ran << " skipped=" << res.skipped
                  << " failed=" << res.failed << " refused=" << res.refused << " fallback:" << res.fallback << " findings=" << res.findings
                  << " stopped=" << (res.stopped ? 1 : 0) << " wall_s=" << fmt(res.wall_s, 2) << " peak_rss_mb=" << fmt(res.peak_rss_mb, 1)
                  << std::endl;
    }
    return res;
}

int RunOrbifoldCampaignCli(const CampaignSpec& spec, bool dry_run, bool verbose) {
    if (spec.jobs > 1) {
        std::cerr << "campaign: `jobs` must be 1 under `orbifold`" << std::endl;
        return 1;
    }
    if (dry_run) {
        const std::vector<OrbifoldCellPlan> plan = PlanOrbifoldCells(spec);
        std::cout << "orbifold campaign '" << spec.name << "': " << plan.size() << " cells (orbifold_cell/v2, model c_i, fingerprint "
                  << hex16(OrbifoldFingerprint(spec)) << ")\n";
        for (const OrbifoldCellPlan& p : plan)
            std::cout << "    " << p.id << "  ell=" << p.ell << " n=" << p.n << " m_target=" << OrbifoldMTarget(spec, p.ell, p.n) << "\n";
        std::cout << "  " << plan.size() << " cells total\n";
        return 0;
    }
    const OrbifoldCampaignResult r = RunOrbifoldCampaign(spec, verbose);
    return r.failed == 0 ? 0 : 2;
}

}  // namespace campaign
