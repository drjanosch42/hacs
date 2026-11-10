// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions phase E2b (slot 4, WP-2) -- [E] `cl:ramanujan` (Claim 8.1)
// THROUGH THE DRIVER, plus the streamed-operator arm and the `u_ell` resume.
//
// [E] = docs/latex/hecke_extensions.tex v1.4.2 `sec:tests`; [X] =
// hecke_extensions_implementation_plan.md v1.7 Sec. 5 (the Ramanujan row),
// Sec. 2.2 Stage 4(c), Sec. 1.3.1 R3.3 item 4; [C] = gl2_contract.tex v1.5.2.
//
// THE ORDER IS [XA m19]'s: the PRIMARY, basis-INDEPENDENT checks first (the
// four matrix identities, the ratios, the eigenvalues, the Jordan types, the
// algebra invariants, the generation verdict), and only then the
// basis-DEPENDENT pins (`T_q = a_q I + c'_q E_31` with the stored `c'_q`),
// labelled as such -- they depend on the stored `H^1` basis, a tier-B golden
// object, and are asserted in this tree but documented as a basis pin.
//
// NO GOLDEN ENTRY.  The collector `sturm_bound` below is defined and called
// from this file alone; nothing here reaches Golden/GoldenHarness.cpp.
// ===========================================================================

#include "HeckeExtRamanujanValidation.h"

#include "HeckeExtDriver.h"
#include "HeckeExtReport.h"

#include "../Hecke/SubspaceOps.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModArithmetic.h"
#include "../Util/ValidationCheck.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using hecke_ext::ExtMatrix;
using hecke_ext::HeckeExtDriver;
using hecke_ext::ExtDriverOptions;

oc_util::Checker Check("    [FAIL] ");

template <class A, class B>
void CheckEq(const A& got, const B& want, const std::string& what) {
    ++Check.checks;
    if (!(got == static_cast<A>(want))) {
        ++Check.fails;
        std::cout << "    [FAIL] " << what << ": got " << got << ", want " << want << std::endl;
    }
}

std::string HexU64(uint64_t v) {
    std::ostringstream os;
    os << std::hex << std::setfill('0') << std::setw(16) << v;
    return os.str();
}

// --- small F_ell matrix helpers (the validation's own, never the unit's) ---

ExtMatrix Identity(uint32_t D, uint32_t ell) {
    ExtMatrix I(D, D, ell);
    for (uint32_t i = 0; i < D; ++i) I.data[i][i] = 1 % ell;
    return I;
}

// A - c*I over F_ell, with c an ORDINARY integer (possibly negative).
ExtMatrix ShiftBy(const ExtMatrix& A, long long c, uint32_t ell) {
    ExtMatrix R = A;
    const long long m = static_cast<long long>(ell);
    uint32_t cm = static_cast<uint32_t>(((c % m) + m) % m);
    for (uint32_t i = 0; i < A.rows; ++i) {
        R.data[i][i] = ModArithmetic::mod_sub(R.data[i][i], cm, ell);
    }
    return R;
}

ExtMatrix ScaleBy(const ExtMatrix& A, long long c, uint32_t ell) {
    ExtMatrix R = A;
    const long long m = static_cast<long long>(ell);
    const uint32_t cm = static_cast<uint32_t>(((c % m) + m) % m);
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) {
            R.data[i][j] = ModArithmetic::mod_mul(R.data[i][j], cm, ell);
        }
    }
    return R;
}

bool Same(const ExtMatrix& A, const ExtMatrix& B) {
    if (A.rows != B.rows || A.cols != B.cols || A.p != B.p) return false;
    for (uint32_t i = 0; i < A.rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) {
            if (A.data[i][j] != B.data[i][j]) return false;
        }
    }
    return true;
}

// The nilpotent Jordan type of N (partition of D, parts descending), from the
// nullity sequence -- the conjugate-partition construction.
std::vector<uint32_t> JordanType(const ExtMatrix& N, uint32_t D, uint32_t ell) {
    std::vector<uint32_t> nullity;
    ExtMatrix P = Identity(D, ell);
    for (uint32_t s = 1; s <= D; ++s) {
        P = P * N;
        nullity.push_back(D - hecke::RankOf(P));
        if (nullity.back() == D) break;
    }
    std::vector<uint32_t> conj;   // conj[j] = #parts > j
    uint32_t prev = 0;
    for (uint32_t s = 0; s < nullity.size(); ++s) {
        conj.push_back(nullity[s] - prev);
        prev = nullity[s];
    }
    std::vector<uint32_t> parts;
    for (uint32_t i = 0; i < (conj.empty() ? 0u : conj[0]); ++i) {
        uint32_t len = 0;
        for (uint32_t j = 0; j < conj.size(); ++j) {
            if (conj[j] > i) ++len;
        }
        parts.push_back(len);
    }
    return parts;
}

const ExtMatrix* OperatorFor(const std::vector<std::pair<uint64_t, ExtMatrix>>& ops, uint64_t q) {
    for (const auto& o : ops) {
        if (o.first == q) return &o.second;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// The golden collector of [X] Sec. 1.6's ext subgrid -- DEFINED HERE, CALLED
// HERE, and nowhere else.  Slot 4 emits no golden entry (the slot-4 plan
// Sec. 2.3 item 1), so this is a stability check inside one process, not a
// reference lookup.
// ---------------------------------------------------------------------------
uint64_t CollectSturmBound(const hecke_ext::HeckeExtRecord& r) {
    uint64_t h = engine::kFnvOffset64;
    for (const hecke_ext::IdealRecord& id : r.ideals) {
        h = engine::FnvU64(h, id.sturm.present ? 1u : 0u);
        h = engine::FnvU64(h, id.sturm.has_B_prime ? id.sturm.B_prime : 0u);
        h = engine::FnvU64(h, id.sturm.cap);
        h = engine::FnvU64(h, id.sturm.closure_dim);
        h = engine::FnvU64(h, id.sturm.dim_M_k);
        for (uint64_t q : id.sturm.sturm_primes) h = engine::FnvU64(h, q);
    }
    return h;
}

// ===========================================================================
// Arm 1 + 2 -- [E] Claim 8.1 at (10, 691), through the driver.
// ===========================================================================
void RunRamanujanArm() {
    std::cout << "-- arm 1: [E] cl:ramanujan at (10, 691) through HeckeExtDriver\n";

    const uint32_t n = 10, ell = 691;
    ExtDriverOptions opts;
    HeckeExtDriver d(n, ell, opts);
    d.run();
    const hecke_ext::HeckeExtRecord& r = d.record();

    // --- Stage 1: U_691 is inside the budget (the frozen model predicts
    // ~104 MB against the default 2048 MB), so 691 is in S_0.  [X] Sec. 2.2
    // Stage 1 / [XA B3].
    const std::vector<uint64_t> want_S0 = {2, 3, 5, 7, 691};
    Check(d.stage_one().primes == want_S0, "(10,691): S_0 = {2,3,5,7,691}");
    Check(d.stage_one().includes_u_ell(), "(10,691): U_691 is in S_0");
    CheckEq(std::string(hecke_ext::to_string(d.stage_one().u_ell.reason)),
            std::string("within_budget"), "(10,691): the u_ell reason");
    Check(d.stage_one().u_ell.predicted_mb > 100.0 && d.stage_one().u_ell.predicted_mb < 110.0,
          "(10,691): the predicted U_691 RSS is ~104 MB");
    std::cout << "    U_691 predicted " << std::fixed << std::setprecision(2)
              << d.stage_one().u_ell.predicted_mb << " MB, budget "
              << d.stage_one().u_ell.budget_mb << " MB ("
              << d.stage_one().u_ell.model_version << ")\n";

    // --- the cell.
    const std::vector<uint32_t>& dims = d.H_dimensions();
    Check(dims.size() == 3, "(10,691): three degrees");
    CheckEq(dims[0], 0u, "(10,691): dim H^0");
    CheckEq(dims[1], 3u, "(10,691): dim H^1");
    CheckEq(dims[2], 0u, "(10,691): dim H^2");
    CheckEq(static_cast<uint32_t>(r.ideals.size()), 1u, "(10,691): exactly one maximal ideal");
    if (r.ideals.empty()) return;
    const hecke_ext::IdealRecord& id = r.ideals[0];
    const hecke_ext::LocalAlgebra& L = d.local_algebra(0);

    // --- the algebra: T_m = F_691[eps]/eps^2.
    Check(id.local, "(10,691): the block is certified local");
    CheckEq(id.dim, 3u, "(10,691): D = 3");
    CheckEq(L.f, 1u, "(10,691): f = 1");
    CheckEq(id.algebra_dim, 2u, "(10,691): dim A_S = 2, i.e. F_691[eps]/eps^2");
    CheckEq(id.embdim, 1u, "(10,691): embedding dimension 1");
    CheckEq(id.nilpotency, 2u, "(10,691): nu = 2");
    Check(id.gorenstein, "(10,691): A_S is Gorenstein");
    CheckEq(id.coefficient_field.w_hat_exponent, 1u, "(10,691): w_hat_exponent = 1");

    // --- the operators, in the record's own stored restrictions.
    const std::vector<std::pair<uint64_t, ExtMatrix>>& ops = d.cell_operators(1);
    const uint64_t qs[5] = {2, 3, 5, 7, 691};
    const ExtMatrix* T[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 5; ++i) {
        T[i] = OperatorFor(ops, qs[i]);
        Check(T[i] != nullptr, "(10,691): A^{(1)}(T_q) is present for every q in S_0");
        if (T[i] == nullptr) return;
    }

    // === PRIMARY (basis-independent) checks =============================
    //
    // [E] cl:ramanujan's four matrix identities on all of H^1.
    struct Identity4 { uint64_t q; long long a; long long lhs_mul; long long rhs_mul; const char* text; };
    static const Identity4 ids[4] = {
        {3,    252,    3,  256, "3(T_3 - 252) = 256(T_2 + 24)"},
        {5,   4830,    3,  174, "3(T_5 - 4830) = 174(T_2 + 24)"},
        {7, -16744,    3,  137, "3(T_7 + 16744) = 137(T_2 + 24)"},
        {691,    1,    3, -542, "3(U_691 - 1) = -542(T_2 + 24)"},
    };
    const ExtMatrix T2plus24 = ShiftBy(*T[0], -24, ell);
    for (const Identity4& e : ids) {
        const ExtMatrix* Tq = OperatorFor(ops, e.q);
        const ExtMatrix lhs = ScaleBy(ShiftBy(*Tq, e.a, ell), e.lhs_mul, ell);
        const ExtMatrix rhs = ScaleBy(T2plus24, e.rhs_mul, ell);
        Check(Same(lhs, rhs), std::string("(10,691): ") + e.text + " exactly in M_3(F_691)");
    }

    // The eigenvalues a_q = sigma_11(q) mod 691 ([E] cl:ramanujan's "Verified").
    const uint32_t want_a[5] = {667, 252, 684, 531, 1};
    for (int i = 0; i < 5; ++i) {
        // a_q is the ONLY eigenvalue: T_q - a_q I is nilpotent.
        const ExtMatrix N = ShiftBy(*T[i], static_cast<long long>(want_a[i]), ell);
        Check(hecke::IsNilpotent(N, 3), "(10,691): T_q - a_q I is nilpotent (a_q is THE eigenvalue)");
        // ... and the Jordan type is (2, 1).
        const std::vector<uint32_t> jt = JordanType(N, 3, ell);
        const std::vector<uint32_t> want_jt = {2, 1};
        Check(jt == want_jt, "(10,691): T_q - tau(q) has Jordan type (2,1)");
    }

    // The RATIOS c'_q / c'_2 == c_q / c_2 ([E]'s "moreover ... which is
    // stronger"), read off the rank-one parts and therefore basis-independent
    // as ratios.  c_q = (-3, -256, -174, -137, 542).
    const long long c_[5] = {-3, -256, -174, -137, 542};
    // c'_q is recovered from T_q - a_q I = c'_q * E, with E := T_2 - a_2 I
    // scaled: the four identities above already prove T_q - a_q I is a scalar
    // multiple of T_2 - a_2 I; extract that scalar.
    const ExtMatrix N2 = ShiftBy(*T[0], static_cast<long long>(want_a[0]), ell);
    uint32_t pivot_i = 0, pivot_j = 0;
    bool found = false;
    for (uint32_t i = 0; i < 3 && !found; ++i) {
        for (uint32_t j = 0; j < 3 && !found; ++j) {
            if (N2.data[i][j] != 0) { pivot_i = i; pivot_j = j; found = true; }
        }
    }
    Check(found, "(10,691): T_2 - 667 I is non-zero (O6: T_2 is not scalar)");
    const uint32_t inv2 = ModArithmetic::mod_inv(N2.data[pivot_i][pivot_j], ell);
    const long long want_ratio[5] = {1, 546, 58, 276, -1};   // -1 = derive
    for (int i = 0; i < 5; ++i) {
        const ExtMatrix Nq = ShiftBy(*T[i], static_cast<long long>(want_a[i]), ell);
        const uint32_t lam = ModArithmetic::mod_mul(Nq.data[pivot_i][pivot_j], inv2, ell);
        // lam must be the SAME scalar in every entry.
        Check(Same(Nq, ScaleBy(N2, lam, ell)),
              "(10,691): T_q - a_q I is a scalar multiple of T_2 - a_2 I");
        // c'_q / c'_2 == c_q / c_2, i.e. lam * c_2 == c_q in F_691.
        const long long m = static_cast<long long>(ell);
        const uint32_t c2 = static_cast<uint32_t>(((c_[0] % m) + m) % m);
        const uint32_t cq = static_cast<uint32_t>(((c_[i] % m) + m) % m);
        CheckEq(ModArithmetic::mod_mul(lam, c2, ell), cq,
                "(10,691): c'_q/c'_2 == c_q/c_2 in F_691");
        if (want_ratio[i] >= 0) {
            CheckEq(lam, static_cast<uint32_t>(want_ratio[i]), "(10,691): the pinned ratio");
        }
    }
    // 221 == 542 * 23 * (-3)^{-1} (mod 691) -- [E]'s own arithmetic, checked
    // independently of the stored matrices.
    {
        const uint32_t inv_minus3 = ModArithmetic::mod_inv(ell - 3, ell);
        const uint32_t v = ModArithmetic::mod_mul(ModArithmetic::mod_mul(542, 23, ell),
                                                  inv_minus3, ell);
        CheckEq(v, 221u, "(10,691): 221 == 542*23*(-3)^{-1} mod 691");
    }

    // Loewy / socle / generators ([E] cl:ramanujan: H^1_m = T_m (+) F).
    Check(!id.modules.empty(), "(10,691): Stage 5 ran");
    if (!id.modules.empty()) {
        const hecke_ext::ModuleStructure& ms = id.modules[0].structure;
        const std::vector<uint32_t> want = {2, 1};
        Check(ms.loewy == want, "(10,691): Loewy layers (2,1)");
        Check(ms.socle == want, "(10,691): socle layers (2,1)");
        CheckEq(ms.min_generators, 2u, "(10,691): min_generators = 2, not cyclic");
        Check(!ms.cyclic, "(10,691): H^1_m is not cyclic");
        CheckEq(ms.verify(), 0u, "(10,691): ModuleStructure::verify()");
    }
    // Every T_q with c_q != 0 is an ACTIVE direction.
    for (size_t i = 0; i < id.tangent_primes.size(); ++i) {
        Check(id.tangent_active[i] != 0, "(10,691): every T_q is an active direction");
    }

    // --- generation ([XA B1]): (a) fails, (b) excluded by the socle bound,
    // (c) sturm(B' = 2).
    CheckEq(id.certificate_routes.a, std::string("not_cyclic"),
            "(10,691): (a) fails -- dim V/m_S V = 2 != f = 1");
    CheckEq(id.certificate_routes.b, std::string("excluded_by_socle_bound"),
            "(10,691): (b) is a PROOF of failure by the socle bound");
    CheckEq(hecke_ext::NaiveCommutantDim(L.t, L.D, ell), 5u,
            "(10,691): the naive oracle confirms dim End_{A_S}(H^1_m) = 5 != 2");
    CheckEq(id.certificate_routes.c, std::string("found"), "(10,691): (c) found a B'");
    CheckEq(id.certificate, std::string("sturm(B')"), "(10,691): the certificate is (c)");
    Check(id.sturm.present, "(10,691): the sturm object is emitted");
    CheckEq(id.sturm.B_prime, 2u, "(10,691): B'(12, 691) = 2");
    CheckEq(id.sturm.cap, 24u, "(10,691): the default cap 12*B(12) = 24");
    CheckEq(id.sturm.closure_dim, 2u, "(10,691): closure_dim = 2");
    CheckEq(id.sturm.dim_M_k, 2u, "(10,691): dim M_12 = 2");
    const std::vector<uint64_t> want_primes = {2};
    Check(id.sturm.sturm_primes == want_primes, "(10,691): sturm_primes = [2]");
    Check(id.generators_certified == want_primes, "(10,691): generators_certified = [2]");
    CheckEq(static_cast<uint32_t>(id.final.value),
            static_cast<uint32_t>(hecke_ext::Final::True), "(10,691): final = true");
    Check(id.final.reason.empty(), "(10,691): a `true` explains nothing");
    CheckEq(d.sturm_certificate(0).verify(), 0u, "(10,691): the Sturm certificate verifies");

    // --- the record ------------------------------------------------------
    CheckEq(r.verify(), 0u, "(10,691): HeckeExtRecord::verify()");
    CheckEq(r.resumable_items(), 0u, "(10,691): nothing is incomplete");
    CheckEq(r.weight, 12u, "(10,691): weight k = n + 2");
    CheckEq(r.settings.fingerprint(), static_cast<uint64_t>(0),
            "(10,691): the default configuration fingerprints to 0");
    Check(r.operators_computed == want_S0, "(10,691): operators_computed = S_0");

    const std::string j = hecke_ext::BuildHeckeExtReportJson(r);
    // A before/after harness for a schema revision: the (10, 691) record is the
    // one whose text a schema bump must be able to account for line by line
    // (slot-5 WP-1 acceptance (c)).  Off unless the variable is set, so the
    // suite stays read-only.
    if (const char* out = std::getenv("HECKE_EXT_JSON_OUT")) {
        std::ofstream f(out, std::ios::binary);
        f << j << "\n";
    }
    static const char* keys[] = {
        "\"schema\":\"hecke_ext/v5\"", "\"n_sym\":10", "\"ell\":691", "\"weight\":12",
        "\"sign\":\"all\"", "\"model\":", "\"iota_sign\":", "\"settings\":",
        "\"coefficient_ring\":", "\"complexes\":", "\"groups\":", "\"operators_computed\":",
        "\"ideals\":", "\"certificates_ran\":", "\"resumable_items\":0",
        "\"certificate\":\"sturm(B')\"", "\"B_prime\":2", "\"final\":true",
        "\"u_ell\":", "\"within_budget\"", "\"loewy\":[2,1]",
    };
    for (const char* k : keys) {
        Check(j.find(k) != std::string::npos,
              std::string("(10,691): the emitted record carries ") + k);
    }
    Check(hecke_ext::BuildHeckeExtReportJson(r) == j, "(10,691): the writer is deterministic");

    // === BASIS-DEPENDENT PINS ([XA m19]) ================================
    // `T_q = a_q I + c'_q E_31` with c'_q = 23, 120, 643, 129, 221 depends on
    // the STORED H^1 basis (a tier-B golden object).  Asserted in this tree,
    // documented as a basis pin: a legitimate change of the stored basis moves
    // these five numbers and NONE of the primary checks above.
    std::cout << "-- arm 2: the BASIS-DEPENDENT pins of [E] cl:ramanujan\n";
    const uint32_t want_cprime[5] = {23, 120, 643, 129, 221};
    for (int i = 0; i < 5; ++i) {
        const ExtMatrix N = ShiftBy(*T[i], static_cast<long long>(want_a[i]), ell);
        bool matches = true;
        for (uint32_t a = 0; a < 3 && matches; ++a) {
            for (uint32_t b = 0; b < 3 && matches; ++b) {
                const uint32_t want = (a == 2 && b == 0) ? want_cprime[i] : 0u;
                if (N.data[a][b] != want) matches = false;
            }
        }
        Check(matches, "(10,691) [basis pin]: T_q = a_q I + c'_q E_31 with the stored c'_q");
    }

    const uint64_t h_sturm = CollectSturmBound(r);
    std::cout << "RAMANUJAN cell=(10,691) ideals=" << r.ideals.size()
              << " S_0=2,3,5,7,691 certificate=" << id.certificate
              << " sturm_bound=0x" << HexU64(h_sturm) << std::endl;
    CheckEq(CollectSturmBound(r), h_sturm, "(10,691): the sturm_bound collector is stable");
}

// ===========================================================================
// Arm 3 -- the streamed route ([X] Sec. 2.2 Stage 4(c), "reuse before
// streaming"): same S_0, same blocks, only the ROUTE by which a prime outside
// S_0 is produced differs.
// ===========================================================================
void RunStreamedArm() {
    std::cout << "-- arm 3: the streamed operator route (reuse before streaming)\n";

    // (22, 5): B'(24, 5) = 5, so the sweep needs T_2, T_3, T_5.  With the good
    // primes cut to {2}, S_0 = {2} u {5} (U_5 is 1.5 MB, well inside the
    // budget), so T_3 must come from OUTSIDE S_0.
    // (26, 7): B'(28, 7) = 7, good primes {2, 3}, so S_0 = {2, 3, 7} and T_5
    // must come from outside S_0.
    struct Cell { uint32_t n; uint32_t ell; std::vector<uint64_t> good; uint64_t streamed_q; };
    const Cell cells[2] = {
        {22, 5, {2}, 3},
        {26, 7, {2, 3}, 5},
    };

    for (const Cell& c : cells) {
        std::ostringstream tag;
        tag << "(" << c.n << "," << c.ell << ")";

        ExtDriverOptions a_opts;
        a_opts.good_primes = c.good;
        a_opts.allow_streaming = true;
        HeckeExtDriver a(c.n, c.ell, a_opts);
        a.run();

        ExtDriverOptions b_opts;
        b_opts.good_primes = c.good;
        b_opts.allow_streaming = false;     // the family builds T_q instead
        HeckeExtDriver b(c.n, c.ell, b_opts);
        b.run();

        Check(a.streamed().built > 0, tag.str() + ": the streaming arm streamed at least one T_q");
        CheckEq(b.streamed().built, 0u, tag.str() + ": the family arm streamed nothing");
        std::cout << "    " << tag.str() << " streamed " << a.streamed().built
                  << " pipeline(s), " << std::fixed << std::setprecision(4)
                  << a.timings().streamed_s << " s, VmHWM "
                  << std::setprecision(1) << a.streamed().peak_rss_mb << " MB; last q = "
                  << a.streamed().last_prime << "\n";
        CheckEq(a.streamed().last_prime, c.streamed_q, tag.str() + ": the streamed prime");

        const hecke_ext::HeckeExtRecord& ra = a.record();
        const hecke_ext::HeckeExtRecord& rb = b.record();
        CheckEq(static_cast<uint32_t>(ra.ideals.size()),
                static_cast<uint32_t>(rb.ideals.size()), tag.str() + ": the same ideal count");
        if (ra.ideals.size() != rb.ideals.size()) continue;

        for (size_t i = 0; i < ra.ideals.size(); ++i) {
            // The STORED T_q|_V of the sweep, entrywise.
            const hecke_ext::SturmCertificate& ca = a.sturm_certificate(static_cast<uint32_t>(i));
            const hecke_ext::SturmCertificate& cb = b.sturm_certificate(static_cast<uint32_t>(i));
            Check(ca.sturm_primes == cb.sturm_primes, tag.str() + ": the same swept prime list");
            CheckEq(static_cast<uint32_t>(ca.t_q.size()),
                    static_cast<uint32_t>(cb.t_q.size()), tag.str() + ": the same number of T_q|_V");
            if (ca.t_q.size() == cb.t_q.size()) {
                for (size_t k = 0; k < ca.t_q.size(); ++k) {
                    Check(Same(ca.t_q[k], cb.t_q[k]),
                          tag.str() + ": the STREAMED T_q|_V equals the family-built one entrywise");
                }
            }
            Check(ca.member == cb.member, tag.str() + ": the same membership flags");
            // A cyclic ideal never reaches (c), so its certificate is the
            // default-constructed one and has nothing to verify.
            if (ca.ell != 0) {
                CheckEq(ca.verify(), 0u, tag.str() + ": the streamed certificate verifies");
            }
            // The record's `sturm` object is identical.
            const hecke_ext::SturmRecord& sa = ra.ideals[i].sturm;
            const hecke_ext::SturmRecord& sb = rb.ideals[i].sturm;
            Check(sa.present == sb.present && sa.has_B_prime == sb.has_B_prime &&
                          sa.B_prime == sb.B_prime && sa.cap == sb.cap &&
                          sa.closure_dim == sb.closure_dim && sa.dim_M_k == sb.dim_M_k &&
                          sa.mode == sb.mode && sa.outcome == sb.outcome &&
                          sa.sturm_primes == sb.sturm_primes,
                  tag.str() + ": the record's `sturm` object is identical");
        }
        // And the whole record text.
        Check(hecke_ext::BuildHeckeExtReportJson(ra) == hecke_ext::BuildHeckeExtReportJson(rb),
              tag.str() + ": the whole record is byte-identical on the two routes");
        // The B(k) counterexample: B' is 5 resp. 7, never B(k) = 3.
        if (!ra.ideals.empty() && ra.ideals[0].sturm.present) {
            Check(ra.ideals[0].sturm.B_prime > 3,
                  tag.str() + ": B' > B(k) = 3 -- the naive sweep would be unsound here");
        }
    }
}

// ===========================================================================
// Arm 4 -- the `u_ell` resume ([X] Sec. 1.3.1 R3.3 item 4).
// ===========================================================================
void RunUEllResumeArm() {
    std::cout << "-- arm 4: the u_ell resume at (10, 691)\n";

    const uint32_t n = 10, ell = 691;

    // The reference: a fresh `u_ell = auto` run (the default).
    ExtDriverOptions opts;
    HeckeExtDriver fresh(n, ell, opts);
    fresh.run();
    const std::string fresh_json = hecke_ext::BuildHeckeExtReportJson(fresh.record());

    // The budgeted run: 64 MB, against a prediction of ~104 MB -- refused.
    std::string cut_json;
    {
        hecke::UEllSetting u;
        u.mode = hecke::UEllMode::BudgetMb;
        u.budget_mb = 64;
        const hecke::ScopedUEll guard(u);

        HeckeExtDriver d(n, ell, opts);
        d.run();
        const std::vector<uint64_t> want_cut = {2, 3, 5, 7};
        Check(d.stage_one().primes == want_cut, "u_ell = budget_mb=64: S_0 = {2,3,5,7}");
        Check(!d.stage_one().includes_u_ell(), "u_ell = budget_mb=64: U_691 is refused");
        CheckEq(std::string(hecke_ext::to_string(d.stage_one().u_ell.reason)),
                std::string("u_ell_over_budget"), "the refusal names the budget");
        Check(!d.record().ideals.empty(), "the refused run still produces a record");
        {
            // v2: ONE cell-level object, whatever the ideal count.
            const hecke_ext::UEllRecord& u_rec = d.record().u_ell;
            Check(u_rec.present, "the refused run records the cell's u_ell verdict");
            Check(u_rec.resumable.present, "a refused U_ell carries a resumable");
            CheckEq(u_rec.resumable.kind, std::string("u_ell"), "of kind u_ell");
            Check(!u_rec.resumable.complete, "and it is incomplete");
            Check(u_rec.resumable.budget_history_json == "[64]",
                  "whose budget_history is [64]");
        }
        CheckEq(d.record().resumable_items(), 1u,
                "one refusal counts ONCE, whatever the ideal count (hecke_ext/v2)");
        cut_json = hecke_ext::BuildHeckeExtReportJson(d.record());

        // A LOWER budget is refused with the named error.
        bool threw = false;
        try {
            d.resume_u_ell(32);
        } catch (const std::exception& e) {
            threw = std::string(e.what()).find("below reached state") != std::string::npos;
        }
        Check(threw, "resume(32) on a state that reached 64 is refused with the named error");

        // The resume to the shipped default budget.
        d.resume_u_ell(hecke::kDefaultUEllBudgetMb);
        const std::vector<uint64_t> want_full = {2, 3, 5, 7, 691};
        Check(d.stage_one().primes == want_full, "after resume: S_0 = {2,3,5,7,691}");
        Check(d.stage_one().includes_u_ell(), "after resume: U_691 is attached");
        const std::vector<uint32_t> want_hist = {64, hecke::kDefaultUEllBudgetMb};
        Check(d.u_ell_budget_history() == want_hist, "budget_history = [64, 2048]");
        // Entrywise equal to the fresh `auto` run: the resumed run installs
        // `auto` at the shipped default, which IS the default setting, so even
        // the settings block and its fingerprint agree.
        Check(hecke_ext::BuildHeckeExtReportJson(d.record()) == fresh_json,
              "the resumed record equals a fresh `u_ell = auto` run's, byte for byte");
        Check(cut_json != fresh_json, "and the cut record was genuinely different");

        // A resume on an INCLUDED verdict is a strict no-op.
        const std::string after = hecke_ext::BuildHeckeExtReportJson(d.record());
        d.resume_u_ell(4096);
        Check(hecke_ext::BuildHeckeExtReportJson(d.record()) == after,
              "a resume on an included verdict is a strict no-op");
        Check(d.u_ell_budget_history() == want_hist, "and the history does not move");
    }

    // `u_ell = off` short-circuits WITHOUT a prediction ([X] Sec. 1.5).
    {
        hecke::UEllSetting u;
        u.mode = hecke::UEllMode::Off;
        const hecke::ScopedUEll guard(u);
        HeckeExtDriver d(n, ell, opts);
        d.run();
        Check(!d.stage_one().includes_u_ell(), "u_ell = off: U_691 is not in S_0");
        CheckEq(std::string(hecke_ext::to_string(d.stage_one().u_ell.reason)),
                std::string("off"), "u_ell = off: the reason is the user's, not the budget's");
        CheckEq(d.stage_one().u_ell.predicted_mb, 0.0,
                "u_ell = off: no prediction is formed");
    }
}

// ===========================================================================
// Arm 5 -- the Stage-3 fast path is a PERFORMANCE change, not a mathematical
// one (the slot-4 plan Sec. 2.3 item 2 / [X] Sec. 2.2 Stage 3).
// ===========================================================================
void RunFastPathArm() {
    std::cout << "-- arm 5: consuming the block's locality data == re-deriving it\n";

    struct Cell { uint32_t n; uint32_t ell; };
    const Cell cells[3] = {{10, 691}, {12, 5}, {22, 5}};
    for (const Cell& c : cells) {
        std::ostringstream tag;
        tag << "(" << c.n << "," << c.ell << ")";

        ExtDriverOptions fast_opts;   // consume_block_locality = true (shipped)
        HeckeExtDriver fast(c.n, c.ell, fast_opts);
        fast.run();

        ExtDriverOptions slow_opts;
        slow_opts.consume_block_locality = false;   // re-run CertifyLocality
        HeckeExtDriver slow(c.n, c.ell, slow_opts);
        slow.run();

        CheckEq(static_cast<uint32_t>(fast.record().ideals.size()),
                static_cast<uint32_t>(slow.record().ideals.size()),
                tag.str() + ": the same ideal count on both routes");
        if (fast.record().ideals.size() != slow.record().ideals.size()) continue;
        for (size_t i = 0; i < fast.record().ideals.size(); ++i) {
            const hecke_ext::LocalAlgebra& A = fast.local_algebra(static_cast<uint32_t>(i));
            const hecke_ext::LocalAlgebra& B = slow.local_algebra(static_cast<uint32_t>(i));
            CheckEq(A.ok ? 1u : 0u, B.ok ? 1u : 0u, tag.str() + ": the same locality verdict");
            if (!A.ok || !B.ok) continue;
            CheckEq(A.f, B.f, tag.str() + ": the same residue degree");
            CheckEq(A.N, B.N, tag.str() + ": the same Teichmueller exponent");
            Check(A.witness_poly.coeffs == B.witness_poly.coeffs, tag.str() + ": the same pi_w");
            Check(Same(A.w_hat, B.w_hat), tag.str() + ": the same w_hat, ENTRYWISE");
            CheckEq(A.dim_A, B.dim_A, tag.str() + ": the same dim A_S");
            CheckEq(A.embdim, B.embdim, tag.str() + ": the same embedding dimension");
            CheckEq(A.nu, B.nu, tag.str() + ": the same nilpotency index");
            Check(A.hilbert == B.hilbert, tag.str() + ": the same Hilbert function");
            for (size_t k = 0; k < A.n.size() && k < B.n.size(); ++k) {
                Check(Same(A.n[k], B.n[k]), tag.str() + ": the same generator n_p, ENTRYWISE");
            }
            for (size_t k = 0; k < A.basis.size() && k < B.basis.size(); ++k) {
                Check(Same(A.basis[k], B.basis[k]), tag.str() + ": the same A_S basis, ENTRYWISE");
            }
        }
        Check(hecke_ext::BuildHeckeExtReportJson(fast.record()) ==
                      hecke_ext::BuildHeckeExtReportJson(slow.record()),
              tag.str() + ": the whole record is byte-identical on the two routes");
        std::cout << "    " << tag.str() << " ext_s fast " << std::fixed << std::setprecision(4)
                  << fast.timings().ext_s() << " s vs slow " << slow.timings().ext_s()
                  << " s (Stage 3: " << fast.timings().stage3_s << " vs "
                  << slow.timings().stage3_s << ")\n";
    }
}

// ===========================================================================
// Arm 6 -- `not_applicable`, and a degenerate cell the driver must survive.
// ===========================================================================
void RunEdgeArm() {
    std::cout << "-- arm 6: n = 0 (dim M_2 = 0) and an odd n (H^* = 0)\n";

    {
        // n = 0: k = 2, dim M_2 = 0, so certificate (c) is NOT APPLICABLE
        // ([E] Def. 2.8) and the record says so rather than claiming a bound.
        ExtDriverOptions opts;
        HeckeExtDriver d(0, 691, opts);
        d.run();
        CheckEq(d.record().verify(), 0u, "n = 0: the record verifies");
        for (const hecke_ext::IdealRecord& id : d.record().ideals) {
            if (!id.sturm.present) continue;
            CheckEq(id.sturm.outcome, std::string("not_applicable"),
                    "n = 0: certificate (c) is not applicable");
        }
        std::cout << "    n = 0: dim H^0 = " << d.H_dimensions()[0]
                  << ", ideals = " << d.record().ideals.size() << "\n";
    }
    {
        // Odd n: every H^q vanishes on this model (the -I / odd-n gap), so the
        // driver runs no group at all and the record is empty but well formed.
        ExtDriverOptions opts;
        HeckeExtDriver d(11, 691, opts);
        d.run();
        CheckEq(static_cast<uint32_t>(d.record().ideals.size()), 0u,
                "odd n: no maximal ideal");
        CheckEq(d.record().verify(), 0u, "odd n: the record still verifies");
        CheckEq(d.record().resumable_items(), 0u, "odd n: nothing is resumable");
    }
}

} // namespace

uint32_t HeckeExtRamanujanValidation() {
    std::cout << "=== HeckeExtRamanujanValidation ([E] cl:ramanujan through HeckeExtDriver) ==="
              << std::endl;
    Check.reset();

    RunRamanujanArm();
    RunStreamedArm();
    RunUEllResumeArm();
    RunFastPathArm();
    RunEdgeArm();

    std::cout << "RAMANUJAN checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    if (Check.fails == 0) {
        std::cout << "=== HeckeExtRamanujanValidation PASSED (" << Check.checks << " checks) ==="
                  << std::endl;
    } else {
        std::cout << "=== HeckeExtRamanujanValidation FAILED (" << Check.fails << " of "
                  << Check.checks << " checks) ===" << std::endl;
    }
    return Check.fails;
}
