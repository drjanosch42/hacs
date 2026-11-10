// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 9, WP-8; C5c) The campaign's job loop under a
// non-default coefficient_ring: hecke_ring_report/v1.  See CampaignRing.h and
// [E] sec:ring (g) (docs/latex/hecke_extensions.tex); campaign.md Sec. 15.

#include "CampaignRing.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../Hecke/CoefficientRing.h"
#include "../Hecke/HeckeCosetTables.h"
#include "../Hecke/HeckeModelDescriptor.h"
#include "../Hecke/HeckeOperatorFamily.h"   // hecke::tau_construction_mutex()
#include "../Hecke/HeckeOperatorOnK.h"
#include "../Hecke/HeckeReduction.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/HeckeTauTable.h"
#include "../Hecke/LadicCell.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/DenseBuf.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModCtx.h"
#include "../Util/TextIo.h"

namespace campaign {

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
double secs(TimePoint a, TimePoint b) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}

constexpr uint64_t kFnvOffset = engine::kFnvOffset64;
using engine::FnvU64;

std::string Fmt(double v, int prec) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << v;
    return os.str();
}

using oc_util::HexU64;   // S24 CL-7 (F11): Util/TextIo.h
// JSON string escaping -- the driver's hand-rolled writer's, verbatim.
std::string JsonEsc(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned>(c) & 0xffu);
                    out += b;
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

std::string JsonU32List(const std::vector<uint32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

// summary.tsv's list form ([E] sec:ring (g)): semicolon separated, `()` empty.
std::string TsvList(const std::vector<uint32_t>& v) {
    if (v.empty()) return "()";
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ";";
        s += std::to_string(v[i]);
    }
    return s;
}

uint64_t HashU32List(uint64_t h, const std::vector<uint32_t>& v) {
    h = FnvU64(h, v.size());
    for (uint32_t e : v) h = FnvU64(h, e);
    return h;
}

// ell^k as an exact integer (spec validation has refused ell^k >= 2^64).
uint64_t PowU64(uint64_t ell, uint32_t k) {
    unsigned __int128 acc = 1;
    for (uint32_t i = 0; i < k; ++i) acc *= ell;
    ENGINE_CHECK_MSG(acc < (static_cast<unsigned __int128>(1) << 64),
                     "campaign ring job: ell^k >= 2^64 reached the job loop (spec validation refuses it)");
    return static_cast<uint64_t>(acc);
}

// One refused part ((g14)).
struct Refusal {
    std::string part;      // "hecke" | "bockstein" | "reduction"
    std::string who;
    std::string message;   // carries `not_implemented_in_this_build:`
};

// Everything one ring job produced, in the record's vocabulary.
struct RingOut {
    // (g6) the bulk complex's per-complex degree fields
    uint32_t num_degrees = 0;
    // (g7)
    std::string kmode = "not_applicable";
    // (g8), all degrees of the complex (the record reports point.degrees)
    std::vector<uint32_t> lengths;
    std::vector<std::vector<uint32_t>> inv;
    // (g9) present iff has_ladic
    bool has_ladic = false;
    uint32_t r_K0 = 0, r_K1 = 0;
    bool certified = false;
    struct Type { uint32_t free_rank = 0; std::vector<uint32_t> torsion; };
    std::vector<Type> type;                               // certified only
    uint32_t need_r0 = 0, need_r1 = 0;                    // the uncertified state's need
    // (g10) present iff has_bockstein
    bool has_bockstein = false;
    uint32_t bockstein_rank0 = 0, higher_torsion_count = 0, dim_H0_F_ell = 0;
    // (g11) present iff has_hecke
    bool has_hecke = false;
    bool has_charpoly = false;
    uint32_t charpoly_precision = 0;
    std::vector<uint64_t> charpoly_coeffs;
    // (g12), (g13)
    std::vector<CanonicalObject> objects;
    std::vector<Refusal> refusals;
    // (slot-11 WP-18c; [E] v1.12.0-pre.2 sec:ring (g)) a DIGIT job (ell^k >= 2^64)
    bool digit = false;
    bool precision_refused = false;       // PrecisionInsufficient: the resumable state below
    hecke::ladic_cell::Precision prec;
    uint32_t state_K = 0, state_r0 = 0, state_r1 = 0, state_need0 = 0, state_need1 = 0;
    std::vector<std::string> charpoly_text;
    hecke::ladic_cell::HidaOut hida;
};

// Runs `f` and records the NotImplementedInThisBuild it must throw as a
// refusal of `part` by `who`.  A guard that no longer refuses is a loud
// failure of the job, never a silently different record.
template<class F>
void ExpectRefusal(RingOut& out, const std::string& part, const std::string& who, F&& f) {
    try {
        f();
    } catch (const hecke::NotImplementedInThisBuild& e) {
        out.refusals.push_back(Refusal{part, who, e.what()});
        return;
    }
    throw std::logic_error("campaign ring job: " + who + " was expected to refuse the " + part +
                           " part under the ring and did not ([E] sec:ring (g14)); a lifted guard needs a "
                           "schema amendment before the campaign may carry it");
}

// The per-job computation, element-width generic: uint32_t when ell^k < 2^32,
// uint64_t otherwise (CoefficientRingContext's own rule).
template<class Elt>
void RunRingPipeline(const CampaignSpec& spec, const Job& job, JobResult& res, RingOut& out) {
    const TimePoint t0 = Clock::now();
    const uint32_t k = spec.math.coefficient_ring.k;
    const uint32_t n = job.n_sym;
    const uint64_t ell = job.ell;
    const bool bredon = hecke::IsBredonPrime(ell);

    const hecke::SimplicialModelDescriptor* desc = hecke::FindModelDescriptor(job.model);
    if (desc == nullptr) throw std::runtime_error("campaign: unknown model '" + job.model + "'");
    // Serialised exactly as the field job: the tessellation tables drive the
    // non-atomic predicate counters (audit 02 Sec. 6.5).
    std::unique_ptr<const SimplicialComplex> K_owned;
    std::unique_ptr<const hecke::SL2ZTessellationTables> model_owned;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        K_owned.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        hecke::RequireModel(*desc, K_owned.get(), "campaign");
        model_owned.reset(new hecke::SL2ZTessellationTables(*K_owned));
    }
    const SimplicialComplex& K = *K_owned;
    const hecke::SL2ZTessellationTables& model = *model_owned;

    // The relation list's context.  (S9-merge, step 0 of e078600) At ell in
    // {2, 3} it is the raw prime-power context, exactly as the Bredon
    // validations build it: the Bredon lift itself belongs to
    // EquivariantCohomologyOnK ALONE (decision 4 (A)) -- its constructor
    // re-derives the context under the Bredon policy and takes the
    // flat-coordinate route -- so this TU never names the policy
    // (SmallEllPolicyScopeValidation).  Every other ell goes through
    // CoefficientRingContext's refusal rule, as before.  Same context value
    // either way (CoefficientRingContext returns ModCtx::prime_power).
    auto relations_ctx = [&](uint32_t kk, const char* who) -> engine::ModCtx<Elt> {
        return bredon ? engine::ModCtx<Elt>::prime_power(static_cast<Elt>(ell), kk)
                      : hecke::CoefficientRingContext<Elt>(who, ell, kk);
    };
    // The job's object over Z/ell^k (the ring is in force: RunRingJob installed
    // the spec's settings on this thread).
    const engine::ModCtx<Elt> ctx = relations_ctx(k, "campaign job loop");
    LocalSystemSimplicialComplexT<Elt> lssc(K, hecke::GenerateSL2ZRelationsRing<Elt>(K, n, ctx), n + 1, ctx.p);
    const TimePoint t1 = Clock::now();
    res.t_setup = secs(t0, t1);
    EquivariantCohomologyOnKT<Elt> coh(&lssc);
    coh.compute_cohomology();
    const TimePoint t2 = Clock::now();
    res.t_cohomology = secs(t1, t2);

    out.num_degrees = coh.num_degrees();
    out.lengths = coh.H_lengths();
    for (uint32_t q = 0; q < out.num_degrees; ++q) out.inv.push_back(coh.H_invariant_factors(q));

    // The F_ell comparison object: built with F_ell installed for its
    // construction only (a field object is refused under Z/ell^k).
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc_f;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh_f;
    auto field_object = [&]() -> const EquivariantCohomologyOnKT<Elt>& {
        if (!coh_f) {
            const hecke::ScopedCoefficientRing field(1);
            lssc_f.reset(new LocalSystemSimplicialComplexT<Elt>(K, GenerateSL2ZRelationsElt<Elt>(K, n, static_cast<Elt>(ell)),
                                                                n + 1, static_cast<Elt>(ell)));
            coh_f.reset(new EquivariantCohomologyOnKT<Elt>(lssc_f.get()));
            coh_f->compute_cohomology();
        }
        return *coh_f;
    };
    // The cell's Z/ell^2 object (the Bockstein's): this job's own at k = 2,
    // else one built with Z/ell^2 installed for its construction only.
    std::unique_ptr<LocalSystemSimplicialComplexT<Elt>> lssc_2;
    std::unique_ptr<EquivariantCohomologyOnKT<Elt>> coh_2;
    auto ring2_object = [&]() -> const EquivariantCohomologyOnKT<Elt>& {
        if (k == 2) return coh;
        if (!coh_2) {
            const hecke::ScopedCoefficientRing ring2(2);
            const engine::ModCtx<Elt> ctx2 = relations_ctx(2, "campaign job loop (Z/ell^2)");
            lssc_2.reset(new LocalSystemSimplicialComplexT<Elt>(K, hecke::GenerateSL2ZRelationsRing<Elt>(K, n, ctx2),
                                                                n + 1, ctx2.p));
            coh_2.reset(new EquivariantCohomologyOnKT<Elt>(lssc_2.get()));
            coh_2->compute_cohomology();
        }
        return *coh_2;
    };

    // ---- the Hecke part ((g11)).  r >= 2 is refused at THIS dispatch; r = 1
    // ---- runs A(T_p) over the ring when q = 1 is reported (at ell in {2, 3}
    // ---- HeckeOperatorOnK refuses by its own guard).
    const bool wants_q1 = std::find(job.degrees.begin(), job.degrees.end(), 1u) != job.degrees.end();
    std::unique_ptr<hecke::HeckeCosetTablesT<Elt>> cosets;
    std::unique_ptr<hecke::HeckeTauTableT<Elt>> tau;
    std::unique_ptr<hecke::HeckeOperatorOnKT<Elt>> op;
    TimePoint t3 = Clock::now(), t4 = t3, t5 = t3;
    auto build_tables = [&]() {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        cosets.reset(new hecke::HeckeCosetTablesT<Elt>(job.p, /*r=*/1, static_cast<Elt>(ell), n));
        t3 = Clock::now();
        tau.reset(new hecke::HeckeTauTableT<Elt>(K, model, *cosets, static_cast<Elt>(ell), n));   // C3
        t4 = Clock::now();
    };
    if (job.r >= 2) {
        ExpectRefusal(out, "hecke", "campaign job loop", [&] {
            hecke::RefuseCoefficientRing("campaign job loop", k,
                                         " for T_{p^r} with r = " + std::to_string(job.r) +
                                             " (the Hecke part over Z/ell^k is validated for r = 1 only in this "
                                             "build; [E] sec:ring (g14))");
        });
    } else if (bredon) {
        build_tables();
        ExpectRefusal(out, "hecke", "HeckeOperatorOnK",
                      [&] { hecke::HeckeOperatorOnKT<Elt> probe(&coh, cosets.get(), tau.get()); });
        t5 = Clock::now();
    } else if (wants_q1) {
        build_tables();
        op.reset(new hecke::HeckeOperatorOnKT<Elt>(&coh, cosets.get(), tau.get()));
        op->compute();   // Theta both ways, asserted equal (the in-process certificate)
        t5 = Clock::now();
        out.has_hecke = true;
    }
    res.t_cosets = cosets ? secs(t2, t3) : 0.0;
    res.t_tau = tau ? secs(t3, t4) : 0.0;
    res.t_theta = tau ? secs(t4, t5) : 0.0;
    const TimePoint t6 = Clock::now();

    // ---- the objects of (d), under their golden names ((g12)) ------------
    const std::string pt = "E8 ring=Z/" + std::to_string(ell) + "^" + std::to_string(k) + " n=" + std::to_string(n) +
                           " ell=" + std::to_string(ell);
    auto scalar = [&](const std::string& name, uint64_t v) {
        out.objects.push_back(CanonicalObject{pt, name, 'A', v, 0, FnvU64(kFnvOffset, v)});
    };
    auto raw = [&](const std::string& name, char tier, uint64_t rows, uint64_t cols, uint64_t h) {
        out.objects.push_back(CanonicalObject{pt, name, tier, rows, cols, h});
    };
    auto Q = [](uint32_t q) { return "^" + std::to_string(q); };
    for (uint32_t q = 0; q < out.num_degrees; ++q) scalar("lengths_H" + Q(q), out.lengths[q]);
    for (uint32_t q = 0; q < out.num_degrees; ++q)
        raw("invariant_factors_H" + Q(q), 'A', out.inv[q].size(), 1, HashU32List(kFnvOffset, out.inv[q]));

    if (bredon) {
        // (g7) not_applicable, (g14) the Bockstein and the reduction refused by
        // HeckeReduction's own path (the Bredon object has no stored basis).
        const EquivariantCohomologyOnKT<Elt>& cf = field_object();
        const EquivariantCohomologyOnKT<Elt>& c2 = ring2_object();
        ExpectRefusal(out, "bockstein", "HeckeReduction (BocksteinCounts)",
                      [&] { (void)hecke::BocksteinCounts(c2, cf); });
        ExpectRefusal(out, "reduction", "HeckeReduction (ReduceHToField)",
                      [&] { (void)hecke::ReduceHToField(coh, cf, 0); });
        out.kmode = "not_applicable";
    } else {
        // (g9) the K-mode, on its applicability range (n > 0 even) only.
        if (coh.ladic_applicable()) {
            out.has_ladic = true;
            out.r_K0 = coh.r_K(0);
            out.r_K1 = coh.r_K(1);
            out.certified = coh.ladic_type_certified();
            out.kmode = out.certified ? "certified" : "uncertified";
            scalar("r_K(d^0)", out.r_K0);
            scalar("r_K(d^1)", out.r_K1);
            scalar("ladic_type_certified", out.certified ? 1 : 0);
            if (out.certified) {
                for (uint32_t q = 0; q < out.num_degrees; ++q) {
                    const auto t = coh.ladic_type(q);
                    out.type.push_back(RingOut::Type{t.free_rank, t.torsion});
                    raw("ladic_type" + Q(q), 'A', t.torsion.size() + 1, 1,
                        HashU32List(FnvU64(kFnvOffset, t.free_rank), t.torsion));
                }
            } else {
                const auto st = coh.ladic_level_state();
                out.need_r0 = st.need_r0;
                out.need_r1 = st.need_r1;
            }
        } else {
            out.kmode = "not_applicable";
        }
        // (g10) the mod-ell Bockstein, from the cell's Z/ell^2 object.
        const EquivariantCohomologyOnKT<Elt>& cf = field_object();
        const hecke::BocksteinBookkeeping bk = hecke::BocksteinCounts(ring2_object(), cf);
        ENGINE_CHECK_MSG(bk.bockstein_rank0 + bk.higher_torsion_count == bk.dim_H0,
                         "campaign ring job: the Bockstein bookkeeping identity failed");
        if (out.certified) {
            // [E] sec:ring (d): its MEANING, asserted wherever the K-mode certifies.
            uint32_t ge2 = 0;
            for (uint32_t e : out.type[1].torsion) ge2 += (e >= 2) ? 1u : 0u;
            ENGINE_CHECK_MSG(bk.higher_torsion_count == ge2,
                             "campaign ring job: higher_torsion_count != #{e_i >= 2} of ladic_type^1");
        }
        out.has_bockstein = true;
        out.bockstein_rank0 = bk.bockstein_rank0;
        out.higher_torsion_count = bk.higher_torsion_count;
        out.dim_H0_F_ell = bk.dim_H0;
        scalar("bockstein_rank^0", bk.bockstein_rank0);
        scalar("higher_torsion_count", bk.higher_torsion_count);
        // (g11) the free characteristic polynomial, at a certified level only.
        if (out.has_hecke && out.certified) {
            const auto fc = op->free_charpoly_H1();
            out.has_charpoly = true;
            out.charpoly_precision = fc.precision;
            uint64_t h = FnvU64(kFnvOffset, fc.precision);
            h = FnvU64(h, fc.coeffs.size());
            for (Elt a : fc.coeffs) {
                out.charpoly_coeffs.push_back(static_cast<uint64_t>(a));
                h = FnvU64(h, static_cast<uint64_t>(a));
            }
            raw("charpoly_A(T_p)^1_free_p" + std::to_string(job.p), 'A', fc.coeffs.size(), fc.precision, h);
        }
        // red_H^q (tier B): the reduction to the F_ell object's D5b basis.
        for (uint32_t q = 0; q < out.num_degrees; ++q) {
            const engine::DenseBuf<Elt> R = hecke::ReduceHToField(coh, cf, q);
            uint64_t h = FnvU64(kFnvOffset, R.rows);
            h = FnvU64(h, R.cols);
            for (uint32_t i = 0; i < R.rows; ++i)
                for (uint32_t j = 0; j < R.cols; ++j) h = FnvU64(h, static_cast<uint64_t>(R.at(i, j)));
            raw("red_H" + Q(q), 'B', R.rows, R.cols, h);
        }
    }
    res.t_report = secs(t6, Clock::now());
}

// (slot-11 WP-18c; [E] v1.12.0-pre.2 sec:ring, (g)) A DIGIT job: ell^k >= 2^64
// at ell >= 5 (spec validation has refused every other cell beyond 2^64).  The
// route is Hecke/LadicCell; the width is the engine-family switch
// ENGINE_LADIC_WIDTH (it moves no value, id or hash).  Members outside the
// route's list are recorded refusals ((g14)), each from the route's own guard.
void RunDigitPipeline(const CampaignSpec& spec, const Job& job, JobResult& res, RingOut& out) {
    namespace lc = hecke::ladic_cell;
    const TimePoint t0 = Clock::now();
    const uint32_t k = spec.math.coefficient_ring.k;
    out.digit = true;
    lc::Request rq;
    rq.n_sym = job.n_sym;
    rq.ell = job.ell;
    rq.k = k;
    rq.width = lc::EngineLadicWidth();
    const bool wants_q1 = std::find(job.degrees.begin(), job.degrees.end(), 1u) != job.degrees.end();
    if (job.r >= 2) {
        ExpectRefusal(out, "hecke", "campaign job loop", [&] {
            hecke::RefuseCoefficientRing("campaign job loop", k,
                                         " for T_{p^r} with r = " + std::to_string(job.r) +
                                             " (the Hecke part over Z/ell^k is validated for r = 1 only in this "
                                             "build; [E] sec:ring (g14))");
        });
    } else if (wants_q1) {
        rq.primes = {job.p};
    }
    lc::CellRecord rec;
    try {
        rec = lc::RunCell(rq);
    } catch (const lc::PrecisionInsufficient& e) {
        out.precision_refused = true;
        out.kmode = "uncertified";
        out.state_K = e.budget_K;
        out.state_r0 = e.reached_r0;
        out.state_r1 = e.reached_r1;
        out.state_need0 = e.need_r0;
        out.state_need1 = e.need_r1;
        out.refusals.push_back(Refusal{"precision", "LadicCell", e.what()});
        res.t_cohomology = secs(t0, Clock::now());
        return;
    }
    // (S13 WP-12) the Hecke + Hida share is booked as theta, the rest as cohomology.
    res.t_theta = rec.hecke_seconds;
    res.t_cohomology = std::max(0.0, secs(t0, Clock::now()) - rec.hecke_seconds);
    out.prec = rec.precision;
    ENGINE_CHECK_MSG(rec.precision.certified, "campaign digit job: the precision bound did not certify (a certificate failed)");
    out.num_degrees = static_cast<uint32_t>(rec.degrees.size());
    for (const lc::Degree& d : rec.degrees) {
        out.lengths.push_back(d.length_k);
        out.inv.push_back(d.invariant_factors_k);
    }
    out.has_ladic = true;
    out.r_K0 = rec.r_K0;
    out.r_K1 = rec.r_K1;
    out.certified = rec.kmode_certified;
    out.kmode = out.certified ? "certified" : "uncertified";
    for (const lc::TypeQ& t : rec.type) out.type.push_back(RingOut::Type{t.free_rank, t.torsion});
    const std::string pt = "E8 ring=Z/" + std::to_string(job.ell) + "^" + std::to_string(k) + " n=" +
                           std::to_string(job.n_sym) + " ell=" + std::to_string(job.ell);
    auto scalar = [&](const std::string& name, uint64_t v) {
        out.objects.push_back(CanonicalObject{pt, name, 'A', v, 0, FnvU64(kFnvOffset, v)});
    };
    auto raw = [&](const std::string& name, char tier, uint64_t rows, uint64_t cols, uint64_t h) {
        out.objects.push_back(CanonicalObject{pt, name, tier, rows, cols, h});
    };
    auto Q = [](uint32_t q) { return "^" + std::to_string(q); };
    for (uint32_t q = 0; q < out.num_degrees; ++q) scalar("lengths_H" + Q(q), out.lengths[q]);
    for (uint32_t q = 0; q < out.num_degrees; ++q)
        raw("invariant_factors_H" + Q(q), 'A', out.inv[q].size(), 1, HashU32List(kFnvOffset, out.inv[q]));
    scalar("r_K(d^0)", out.r_K0);
    scalar("r_K(d^1)", out.r_K1);
    scalar("ladic_type_certified", out.certified ? 1 : 0);
    for (uint32_t q = 0; q < out.type.size(); ++q)
        raw("ladic_type" + Q(q), 'A', out.type[q].torsion.size() + 1, 1,
            HashU32List(FnvU64(kFnvOffset, out.type[q].free_rank), out.type[q].torsion));
    if (!rec.hecke.empty()) {
        const lc::HeckeOut& h = rec.hecke.front();
        out.has_hecke = true;
        out.has_charpoly = true;
        out.charpoly_precision = h.precision;
        uint64_t hh = FnvU64(kFnvOffset, h.precision);
        hh = FnvU64(hh, h.charpoly.size());
        for (const lc::Residue& c : h.charpoly) {
            out.charpoly_text.push_back(lc::ResidueText(c, job.ell));
            for (uint32_t dg : c) hh = FnvU64(hh, dg);
        }
        raw("charpoly_A(T_p)^1_free_p" + std::to_string(job.p), 'A', h.charpoly.size(), h.precision, hh);
    }
    out.hida = rec.hida;
    // (g14) the members outside the digit route's list, from its own guard.
    ExpectRefusal(out, "bockstein", "LadicCell",
                  [&] { lc::RefuseMember("the Bockstein", job.n_sym, job.ell, k); });
    ExpectRefusal(out, "reduction", "LadicCell",
                  [&] { lc::RefuseMember("the reduction red_H^q", job.n_sym, job.ell, k); });
}

} // namespace

JobResult RunRingJob(const CampaignSpec& spec, const Job& job, const std::string& path, bool verbose,
                     bool allow_rss_reset, std::ostream* log) {
    JobResult res;
    res.job = job;
    res.output_path = path;
    res.lean = spec.lean;
    res.ring_job = true;
    res.level_a_state = "not_applicable";

    // The spec's mathematical settings, the ring among them, on this thread
    // (thread_local, propagated onto the engine's task threads).
    const hecke::ScopedMathSettings math_guard(spec.math);
    const uint32_t k = spec.math.coefficient_ring.k;
    ENGINE_CHECK_MSG(k >= 2, "RunRingJob: needs a non-default coefficient_ring");

    const bool per_job_rss = allow_rss_reset && oc_util::ResetPeakRss();
    res.peak_rss_scope = per_job_rss ? "job" : (allow_rss_reset ? "process" : "process-concurrent");

    const TimePoint t0 = Clock::now();
    RingOut out;
    std::string status;
    try {
        if (!hecke::ladic_cell::FitsU64(job.ell, k)) RunDigitPipeline(spec, job, res, out);   // (slot-11 WP-18c)
        else if (PowU64(job.ell, k) <= 0xffffffffull) RunRingPipeline<uint32_t>(spec, job, res, out);
        else                                           RunRingPipeline<uint64_t>(spec, job, res, out);
        res.ok = true;
        status = out.refusals.empty() ? "ok" : "ok_partial";
    } catch (const std::exception& e) {
        res.ok = false;
        res.error = e.what();
        status = "failed";
    }
    res.ran = true;
    res.t_total = secs(t0, Clock::now());
    res.peak_rss_mb = oc_util::PeakRssMiB();
    res.ring_status = status;
    res.objects = out.objects;
    for (const Refusal& r : out.refusals) res.ring_refusals.push_back(r.part);

    // ---- the record, (g1)-(g16), assembled in memory ---------------------
    const hecke::SimplicialModelDescriptor* d = hecke::FindModelDescriptor(job.model);
    std::ostringstream o;
    o << "{\"schema\":\"hecke_ring_report/v1\",";                                  // (g1)
    o << "\"job_id\":\"" << JsonEsc(job.id()) << "\",";                             // (g2)
    o << "\"point\":{\"n_sym\":" << job.n_sym << ",\"ell\":" << job.ell << ",\"p\":" << job.p
      << ",\"r\":" << job.r << ",\"model\":\"" << JsonEsc(job.model) << "\",\"degrees\":"
      << JsonU32List(job.degrees) << "},";                                          // (g3)
    o << "\"model_descriptor\":{\"name\":\"" << JsonEsc(job.model)
      << "\",\"dimension\":" << (d ? d->dimension : 0) << ",\"simplex_counts\":[";
    if (d) for (size_t i = 0; i < d->simplex_counts.size(); ++i) { if (i) o << ","; o << d->simplex_counts[i]; }
    o << "],\"relation_source\":\"" << JsonEsc(d ? d->relation_source : std::string()) << "\"},";
    o << "\"shape_signature\":\"" << JsonEsc(job.shape().to_string()) << "\",";
    o << "\"coefficient_ring\":\"" << JsonEsc(hecke::CoefficientRingText(k)) << "\",";   // (g4)
    if (!hecke::ladic_cell::FitsU64(job.ell, k)) {
        // (slot-11 WP-18c; [E] v1.12.0-pre.2 (g)) a digit job: the modulus as text,
        // the route and the precision bound's keys (digit jobs only).
        o << "\"modulus\":\"" << job.ell << "^" << k << "\",";
        o << "\"ladic_level\":" << k << ",";
        o << "\"route\":\"digit\",";
        if (res.ok && !out.precision_refused) {
            const hecke::ladic_cell::Precision& P = out.prec;
            o << "\"precision\":{\"precision_requested\":" << P.requested << ",\"precision_working\":" << P.working
              << ",\"e_max\":" << P.e_max << ",\"word_level\":" << P.word_level
              << ",\"precision_certified\":" << (P.certified ? "true" : "false") << ",\"digit_width\":" << P.digit_width
              << ",\"digits_L\":" << P.digits_L << "},";
        } else if (out.precision_refused) {
            o << "\"precision\":{\"precision_requested\":" << k << ",\"precision_certified\":false,\"state\":{"
              << "\"kind\":\"ladic_precision\",\"budget\":{\"K\":" << out.state_K << "},\"reached\":{\"r_K0\":"
              << out.state_r0 << ",\"r_K1\":" << out.state_r1 << "},\"need\":{\"r_K0\":" << out.state_need0
              << ",\"r_K1\":" << out.state_need1 << "},\"complete\":false}},";
        }
    } else {
        o << "\"modulus\":\"" << PowU64(job.ell, k) << "\",";
        o << "\"ladic_level\":" << k << ",";
    }
    o << "\"settings\":" << spec.math.to_json() << ",";                             // (g5)
    o << "\"certify\":{\"mode\":\"" << JsonEsc(spec.certify.text) << "\",\"applicable\":false},";
    if (res.ok) {
        o << "\"degree_fields\":{\"q_max\":" << (out.num_degrees ? out.num_degrees - 1 : 0)
          << ",\"truncated\":false,\"q_max_source\":\"derived\",\"degrees\":[";         // (g6)
        for (uint32_t q = 0; q < out.num_degrees; ++q) { if (q) o << ","; o << q; }
        o << "]},";
        o << "\"kmode\":\"" << out.kmode << "\",";                                  // (g7)
        o << "\"cohomology\":[";                                                    // (g8)
        bool first = true;
        for (uint32_t q : job.degrees) {
            if (q >= out.num_degrees) continue;
            if (!first) o << ",";
            first = false;
            o << "{\"q\":" << q << ",\"length\":" << out.lengths[q]
              << ",\"invariant_factors\":" << JsonU32List(out.inv[q]) << "}";
        }
        o << "],";
        if (out.has_ladic) {                                                        // (g9)
            o << "\"ladic\":{\"r_K\":[" << out.r_K0 << "," << out.r_K1 << "],\"certified\":"
              << (out.certified ? "true" : "false");
            if (out.certified) {
                o << ",\"type\":[";
                for (size_t q = 0; q < out.type.size(); ++q) {
                    if (q) o << ",";
                    o << "{\"q\":" << q << ",\"free_rank\":" << out.type[q].free_rank
                      << ",\"torsion\":" << JsonU32List(out.type[q].torsion) << "}";
                }
                o << "]";
            } else {
                o << ",\"state\":{\"kind\":\"ladic_level\",\"budget\":{\"K\":" << k
                  << "},\"reached\":{\"r_K0\":" << out.r_K0 << ",\"r_K1\":" << out.r_K1
                  << "},\"need\":{\"r_K0\":" << out.need_r0 << ",\"r_K1\":" << out.need_r1
                  << "},\"complete\":false}";
            }
            o << "},";
        }
        if (out.has_bockstein) {                                                    // (g10)
            o << "\"bockstein\":{\"bockstein_rank_0\":" << out.bockstein_rank0
              << ",\"higher_torsion_count\":" << out.higher_torsion_count
              << ",\"dim_H0_F_ell\":" << out.dim_H0_F_ell << "},";
        }
        if (out.has_hecke) {                                                        // (g11)
            o << "\"hecke\":{\"p\":" << job.p << ",\"r\":1";
            if (out.has_charpoly) {
                o << ",\"charpoly_free\":{\"precision\":" << out.charpoly_precision << ",\"coeffs\":[";
                for (size_t i = 0; i < out.charpoly_coeffs.size(); ++i) {
                    if (i) o << ",";
                    // Decimal STRINGS (lead decision, slot-9 WP-8): a coefficient
                    // lies in [0, ell^precision) and may exceed 2^53, the JSON
                    // number limit -- the reason (g4) writes `modulus` as one.
                    o << "\"" << out.charpoly_coeffs[i] << "\"";
                }
                for (size_t i = 0; i < out.charpoly_text.size(); ++i) {   // (slot-11 WP-18c) a digit job
                    if (i) o << ",";
                    o << "\"" << JsonEsc(out.charpoly_text[i]) << "\"";
                }
                o << "]}";
            }
            o << "},";
        }
        if (out.hida.present) {                                                     // (slot-11 WP-18c)
            const hecke::ladic_cell::HidaOut& h = out.hida;
            o << "\"hida\":{\"precision\":" << h.precision << ",\"rank\":" << h.rank << ",\"rank_e_mod_ell\":"
              << h.rank_e_mod_ell << ",\"ordinary_rank_mod_ell\":" << h.ordinary_rank_mod_ell << ",\"certified\":"
              << (h.certified ? "true" : "false") << ",\"e\":[";
            for (size_t i = 0; i < h.e.size(); ++i) {
                if (i) o << ",";
                o << "\"" << JsonEsc(hecke::ladic_cell::ResidueText(h.e[i], job.ell)) << "\"";
            }
            o << "]},";
        }
        o << "\"objects\":[";                                                       // (g12)
        for (size_t i = 0; i < out.objects.size(); ++i) {
            const CanonicalObject& c = out.objects[i];
            if (i) o << ",";
            o << "{\"name\":\"" << JsonEsc(c.object) << "\",\"tier\":\"" << c.tier << "\",\"rows\":" << c.rows
              << ",\"cols\":" << c.cols << ",\"fnv1a64\":\"" << HexU64(c.hash, false) << "\"}";
        }
        o << "],";
    }
    o << "\"refusals\":[";                                                          // (g13)
    for (size_t i = 0; i < out.refusals.size(); ++i) {
        if (i) o << ",";
        o << "{\"part\":\"" << out.refusals[i].part << "\",\"who\":\"" << JsonEsc(out.refusals[i].who)
          << "\",\"message\":\"" << JsonEsc(out.refusals[i].message) << "\"}";
    }
    o << "],";
    o << "\"status\":\"" << status << "\",";                                        // (g14)
    if (!res.ok) o << "\"error\":\"" << JsonEsc(res.error) << "\",";
    o << "\"timings_s\":{\"setup\":" << Fmt(res.t_setup, 6)                         // (g15)
      << ",\"cohomology\":" << Fmt(res.t_cohomology, 6) << ",\"cosets\":" << Fmt(res.t_cosets, 6)
      << ",\"tau\":" << Fmt(res.t_tau, 6) << ",\"theta\":" << Fmt(res.t_theta, 6)
      << ",\"family\":" << Fmt(res.t_family, 6) << ",\"external\":" << Fmt(res.t_external, 6)
      << ",\"report\":" << Fmt(res.t_report, 6) << ",\"total\":" << Fmt(res.t_total, 6) << "},";
    o << "\"peak_rss_mb\":" << Fmt(res.peak_rss_mb, 1) << ",\"peak_rss_scope\":\"" << res.peak_rss_scope << "\",";
    o << "\"campaign_job_complete\":" << (res.ok ? "true" : "false") << "}\n";     // (g16)
    {
        std::ofstream os(path, std::ios::out | std::ios::trunc);
        if (!os) throw std::runtime_error("campaign: cannot open '" + path + "' for writing");
        os << o.str();
        os.flush();
        if (!os) throw std::runtime_error("campaign: write failed on '" + path + "'");
    }

    // ---- summary.tsv's 15 appended values ([E] sec:ring (g)) -------------
    {
        std::vector<std::string>& c = res.ring_columns;
        auto in_degrees = [&](uint32_t q) {
            return res.ok && q < out.num_degrees &&
                   std::find(job.degrees.begin(), job.degrees.end(), q) != job.degrees.end();
        };
        c.push_back(hecke::CoefficientRingText(k));
        c.push_back(std::to_string(k));
        c.push_back(res.ok ? out.kmode : "-");
        for (uint32_t q = 0; q < 3; ++q) c.push_back(in_degrees(q) ? std::to_string(out.lengths[q]) : "-");
        for (uint32_t q = 0; q < 3; ++q) c.push_back(in_degrees(q) ? TsvList(out.inv[q]) : "-");
        c.push_back(out.has_ladic && res.ok ? std::to_string(out.r_K0) : "-");
        c.push_back(out.has_ladic && res.ok ? std::to_string(out.r_K1) : "-");
        c.push_back(out.has_bockstein && res.ok ? std::to_string(out.bockstein_rank0) : "-");
        c.push_back(out.has_bockstein && res.ok ? std::to_string(out.higher_torsion_count) : "-");
        if (out.has_charpoly && res.ok) {
            std::string s;
            for (size_t i = 0; i < out.charpoly_coeffs.size(); ++i) {
                if (i) s += ";";
                s += std::to_string(out.charpoly_coeffs[i]);
            }
            for (size_t i = 0; i < out.charpoly_text.size(); ++i) {   // (slot-11 WP-18c) a digit job
                if (i) s += ";";
                s += out.charpoly_text[i];
            }
            c.push_back(s + "@" + std::to_string(out.charpoly_precision));
        } else {
            c.push_back("-");
        }
        std::string refused;
        for (const Refusal& r : out.refusals) refused += (refused.empty() ? "" : ";") + r.part;
        c.push_back(refused.empty() ? "-" : refused);
    }

    if (verbose) {
        std::ostringstream line;
        line << "  [" << (!res.ok ? "FAIL" : (out.refusals.empty() ? "ok  " : "part")) << "] " << job.to_string()
             << "  " << hecke::CoefficientRingText(k) << "  lengths=(";
        for (size_t i = 0; i < out.lengths.size(); ++i) { if (i) line << ","; line << out.lengths[i]; }
        line << ")  kmode=" << out.kmode << "  " << Fmt(res.t_total, 3) << " s  " << Fmt(res.peak_rss_mb, 1) << " MB";
        if (!out.refusals.empty()) {
            line << "  refused=";
            for (size_t i = 0; i < out.refusals.size(); ++i) line << (i ? ";" : "") << out.refusals[i].part;
        }
        line << "\n";
        if (!res.ok) line << "        error: " << res.error << "\n";
        if (log != nullptr) (*log) << line.str();
        else                std::cout << line.str() << std::flush;
    }
    return res;
}

std::string RingSummaryHeader() {
    return "\tcoefficient_ring\tladic_level\tkmode\tlength_H0\tlength_H1\tlength_H2"
           "\tinvariant_factors_H0\tinvariant_factors_H1\tinvariant_factors_H2\tr_K0\tr_K1"
           "\tbockstein_rank0\thigher_torsion_count\tcharpoly_free_A1\trefusals";
}

std::string RingSummaryColumns(const CampaignSpec& spec, const JobResult& r) {
    std::vector<std::string> c = r.ring_columns;
    if (c.size() != 15) {
        // A job `resume` skipped: the ring and level are the campaign's; the
        // rest is in the job file already written, not re-read here.
        const uint32_t k = spec.math.coefficient_ring.k;
        c.assign(15, "-");
        c[0] = hecke::CoefficientRingText(k);
        c[1] = std::to_string(k);
    }
    std::string s;
    for (const std::string& v : c) s += "\t" + v;
    return s;
}

} // namespace campaign
