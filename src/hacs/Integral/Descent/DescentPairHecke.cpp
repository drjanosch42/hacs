// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D4, the Hecke half (code map (f)): T~^{(N)}_{Iso,p} and T~^{(N)}_{Isob,p} by eq:levelcorr from the pair lift of
// Integral/Model/StrictLift, on the free model's full layout.  [C] = docs/latex/gl2_contract.tex (v1.23.0-pre.27,
// prop:descentpairhecke, prop:strictliftN(3)), cited by label.  See DescentPair.h.
#include "DescentPair.h"

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>

#include "../Model/StrictLift.h"
#include "../../Hecke/ExactInt.h"
#include "../../Hecke/HeckeLevelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../../ModArithmetic/EngineCheck.h"

namespace integral_descent {
namespace pair {

using engine::DenseBuf;
using U32Ctx = engine::ModCtx<uint32_t>;
using hecke::Mat22;

namespace {

struct BundleEntry {
    std::unique_ptr<integral_model::StrictLiftBundle> bundle;
    integral_model::StrictLiftCertificate cert;
    PairLiftCost cost;
};

// The pair lift of T_p for ANY prime p (owner scope addition 2026-10-25; prop:descentpairhecke(1): "produced for every p"), built
// on first request by the terminating construction of prop:strictlift(1) (boundary step first: tau_bd, the strict datum tau^s of
// prop:strictpair, then Lemma strictlifting through the columns), degrees 0 .. 6 (q_max = 5 >= the pair's 4), and certified by
// StrictLift::certify() -- (T1) on the Schreier generators, (T2), (T3), the strict identity (S) and the boundary items (B) -- before
// first use; cached per p.  A named cap (ClassifyPairException: IntegerCap, DomainCap) or a failed input-table certification is
// recorded with its PairRefusal code and the operator REFUSED (no size bound is asserted, rmk:integralscope(2)); any other exception
// propagates (C-D7p referee M1).  Rule 19 (audit m5): PredictPairLiftRss(p) is read
// BEFORE anything is built and the lift REFUSED above kPairLiftRssCap.  Locking (audit m5): the global mutex guards only the map of
// per-p slots; each slot has its own mutex, held while that p is built and certified, so a caller at another p never waits (callers
// at the SAME p wait for the one construction; certification is ~p^{2.4}: 26 s at p = 127).
struct BundleSlot {
    std::mutex mu;
    std::unique_ptr<BundleEntry> entry;   // set once, under mu; never reset
};

std::unique_ptr<BundleEntry> BuildBundle(uint32_t p) {
    auto e = std::make_unique<BundleEntry>();
    e->cost.p = p;
    e->cost.rss_pred = PredictPairLiftRss(p);
    if (e->cost.rss_pred > kPairLiftRssCap) {
        e->cost.ok = false;
        e->cost.code = PairRefusal::RssCeiling;
        e->cost.why = "rule 19: the pair lift at p = " + std::to_string(p) + " predicts " + std::to_string(e->cost.rss_pred >> 20) +
                      " MiB above the ceiling " + std::to_string(kPairLiftRssCap >> 20) + " MiB (refused before building)";
        return e;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t base = CurrentRssBytes();
    RssSampler rs;
    try {
        e->bundle = std::make_unique<integral_model::StrictLiftBundle>(p, 5u);
        e->cost.build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto t1 = std::chrono::steady_clock::now();
        e->cert = e->bundle->lift().certify();
        e->cost.certify_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        e->cost.ok = e->cert.all();
        if (!e->cost.ok) {
            e->cost.code = PairRefusal::LiftCertificate;
            e->cost.why = "certificate: " + e->cert.first_failure;
        }
        for (uint64_t t : e->cert.terms) e->cost.terms += t;
        e->cost.max_abs_coefficient = e->cert.max_abs_coefficient;
        e->cost.max_abs_entry = e->cert.max_abs_entry;
    } catch (const std::exception& ex) {
        // C-D7p referee M1: only the named caps and an input table's failed self-certification refuse; anything else (an
        // ENGINE_CHECK, an internal error) is a genuine error and propagates -- the slot stays empty, a later call retries
        const PairRefusal code = ClassifyPairException(ex);
        if (code == PairRefusal::None) throw;
        e->bundle.reset();
        e->cost.ok = false;
        e->cost.code = code;
        e->cost.why = std::string("construction stopped (") + PairRefusalName(code) + "): " + ex.what();
    }
    const uint64_t peak = rs.stop();
    e->cost.rss_growth = peak > base ? peak - base : 0;
    return e;
}

const BundleEntry& Bundle(uint32_t p) {
    static std::map<uint32_t, std::unique_ptr<BundleSlot>> slots;
    static std::mutex mu;
    BundleSlot* slot = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto& s = slots[p];
        if (!s) s = std::make_unique<BundleSlot>();
        slot = s.get();
    }
    std::lock_guard<std::mutex> lk(slot->mu);
    if (!slot->entry) slot->entry = BuildBundle(p);
    return *slot->entry;
}

const LocalModel& FreeModelH() { return FreeIsoModel(); }

uint32_t Signed(const U32Ctx& ctx, int64_t v) {
    const int64_t m = static_cast<int64_t>(ctx.p);
    int64_t r = v % m;
    if (r < 0) r += m;
    return static_cast<uint32_t>(r);
}

}  // namespace

const char* PairRefusalName(PairRefusal r) {
    switch (r) {
        case PairRefusal::None: return "None";
        case PairRefusal::NotPrime: return "NotPrime";
        case PairRefusal::PDividesN: return "PDividesN";
        case PairRefusal::BulkOnly: return "BulkOnly";
        case PairRefusal::DegreeRange: return "DegreeRange";
        case PairRefusal::RssCeiling: return "RssCeiling";
        case PairRefusal::IntegerCap: return "IntegerCap";
        case PairRefusal::DomainCap: return "DomainCap";
        case PairRefusal::InputTableCertificate: return "InputTableCertificate";
        case PairRefusal::LiftCertificate: return "LiftCertificate";
        case PairRefusal::IsobClosure: return "IsobClosure";
        case PairRefusal::StrictIdentity: return "StrictIdentity";
    }
    return "?";
}

PairRefusal ClassifyPairException(const std::exception& e) {
    // a HeckeInternalError (a violated precondition or internal check, e.g. CuspKey::reduce_to_infinity, QuadPoint's domain) is a
    // genuine error and propagates; only a real cap (any other HeckeOverflow) refuses (D4 follow-up, D5 audit round)
    if (dynamic_cast<const hecke::HeckeInternalError*>(&e)) return PairRefusal::None;
    if (dynamic_cast<const hecke::HeckeOverflow*>(&e)) return PairRefusal::IntegerCap;
    if (dynamic_cast<const std::overflow_error*>(&e)) return PairRefusal::DomainCap;
    if (dynamic_cast<const std::runtime_error*>(&e) && std::string(e.what()).find("certification FAILED") != std::string::npos)
        return PairRefusal::InputTableCertificate;
    return PairRefusal::None;
}

uint64_t PredictPairLiftRss(uint32_t p) {
    // the measured stored-term count (DESCENT4_report.md Sec. 4: 1159 at p = 3 ... 186 791 at p = 127, ~ p^{1.4}), bounded above by
    // 1200 (p / 3)^{1.4} + 700, times 256 bytes per term (measured 146: a GKey map node with its int64 multiplicity, and the
    // certificate's working copies), plus 64 MiB
    const double terms = 1200.0 * std::pow(static_cast<double>(p) / 3.0, 1.4) + 700.0;
    return static_cast<uint64_t>(terms * 256.0) + (64ull << 20);
}

const PairLiftCost& PairLiftCostOf(uint32_t p) {
    ENGINE_CHECK_MSG(p >= 2 && IsPrime(p), "PairLiftCostOf: p prime");
    return Bundle(p).cost;
}

const integral_model::StrictLiftCertificate& PairLiftCertificate(uint32_t p) {
    ENGINE_CHECK_MSG(p >= 2 && IsPrime(p), "PairLiftCertificate: p prime");
    const BundleEntry& B = Bundle(p);
    ENGINE_CHECK_MSG(B.bundle != nullptr, ("PairLiftCertificate: the lift was not built: " + B.cost.why).c_str());
    return B.cert;
}

const integral_model::StrictLift& PairLift(uint32_t p) {
    ENGINE_CHECK_MSG(p >= 2 && IsPrime(p), "PairLift: p prime");
    const BundleEntry& B = Bundle(p);
    ENGINE_CHECK_MSG(B.bundle != nullptr && B.cost.ok, ("PairLift: no certified lift: " + B.cost.why).c_str());
    return B.bundle->lift();
}

ConeHecke BuildConeHecke(uint32_t N, uint32_t n, uint32_t p, const U32Ctx& ctx, uint32_t q_top, ConeDatum datum) {
    ConeHecke H;
    H.N = N;
    H.n = n;
    H.p = p;
    H.q_top = q_top;
    auto refuse = [&](PairRefusal code, const std::string& w) {
        H.refused = true;
        H.code = code;
        H.refusal = "REFUSED [" + std::string(PairRefusalName(code)) + "] T_" + std::to_string(p) + " on the pair at (N, n) = (" +
                    std::to_string(N) + ", " + std::to_string(n) + "): " + w;
        return H;
    };
    if (datum == ConeDatum::BulkOnly)
        return refuse(PairRefusal::BulkOnly, "the bulk-only lift need not preserve Isob and is not a pair datum; it is never used on the cone "
                      "(prop:descentpairhecke(5), rmk:conestrict; code map (f))");
    if (p < 2 || !IsPrime(p)) return refuse(PairRefusal::NotPrime, "p is not a prime");
    if (N % p == 0) return refuse(PairRefusal::PDividesN, "p | N: U_p is deferred ([R-Q4]; prop:descentpairhecke covers p !| N only)");
    if (q_top > kPairLiftTop)
        return refuse(PairRefusal::DegreeRange, "q_top = " + std::to_string(q_top) + " > " + std::to_string(kPairLiftTop) +
                                                    " (the lift is built to degree 6; the pair is served for q_max <= 5, higher degrees come "
                                                    "from prop:descentpairhecke(7), never from extending the lift)");
    const BundleEntry& B = Bundle(p);
    if (!B.cost.ok) return refuse(B.cost.code, "no certified pair lift: " + B.cost.why);
    if (datum == ConeDatum::TestPerturbedLift) {
        // VALIDATION ONLY (planted fault): one value tau_Iso(s_1 e_0) in degree 1 perturbed by e_0; its certificate must fail and
        // the operator be refused -- never computed from an uncertified lift
        const integral_model::StrictLift bad = B.bundle->lift().perturbed(1, 1, 0, 1);
        const integral_model::StrictLiftCertificate c = bad.certify();
        if (!c.all()) return refuse(PairRefusal::LiftCertificate, "the (perturbed) pair lift fails its certificate: " + c.first_failure);
        return refuse(PairRefusal::None, "planted fault NOT detected by the certificate");
    }
    if (datum == ConeDatum::TestHomotopyIntoIsob || datum == ConeDatum::TestHomotopyIntoBulk) {
        // VALIDATION ONLY (C-D7p referee m4): tau' = tau + dK + Kd, K(s_1 e_v) = e_e (the referee's Arm F); (T1)-(T3) hold by
        // construction, so only (S) and the boundary items (B) can refuse it
        const integral_model::StrictLift& good = B.bundle->lift();
        const auto& c0 = good.model().cells(0);
        const auto& c1 = good.model().cells(1);
        const bool into_isob = datum == ConeDatum::TestHomotopyIntoIsob;
        uint32_t v = UINT32_MAX, e = UINT32_MAX;
        for (uint32_t c = 0; c < c0.size() && v == UINT32_MAX; ++c)
            if (c0[c].boundary && c0[c].t == 0) v = c;
        for (uint32_t c = 0; c < c1.size() && e == UINT32_MAX; ++c)
            if (c1[c].t == 0 && c1[c].boundary == into_isob) e = c;
        ENGINE_CHECK_MSG(v != UINT32_MAX && e != UINT32_MAX, "BuildConeHecke: no cells for the planted homotopy");
        integral_model::IsoChain k(c1.size());
        k.c[e].add(integral_model::key_of(Mat22::I()), 1);
        const integral_model::StrictLiftCertificate c = good.with_homotopy(0, 1, v, k).certify();
        const std::string flags = std::string(" [T1=") + (c.t1_twisted_equivariance ? "1" : "0") + " T2=" + (c.t2_chain_map ? "1" : "0") +
                                  " T3=" + (c.t3_augmentation ? "1" : "0") + " S=" + (c.strict_eps ? "1" : "0") + " Bpres=" +
                                  (c.boundary_preserved ? "1" : "0") + " Beps=" + (c.boundary_eps ? "1" : "0") + "]";
        if (!(c.t1_twisted_equivariance && c.t2_chain_map && c.t3_augmentation))
            return refuse(PairRefusal::None, "planted homotopy fault breaks (T1)-(T3) (not the intended fault)" + flags);
        if (!c.all()) return refuse(PairRefusal::LiftCertificate, "the homotopic pair lift fails (S) / (B): " + c.first_failure + flags);
        return refuse(PairRefusal::None, "planted homotopy fault NOT detected by the certificate" + flags);
    }
    const integral_model::StrictLift& lift = B.bundle->lift();
    const LocalModel& L = FreeModelH();
    ENGINE_CHECK_MSG(q_top <= lift.q_top() && q_top <= L.q_top(), "BuildConeHecke: q_top beyond the built lift");
    // the lift's model and the free LocalModel have the same cells, in the same order
    for (uint32_t q = 0; q <= q_top; ++q) {
        const auto& a = lift.model().cells(q);
        const auto& b = L.cells(q);
        ENGINE_CHECK_MSG(a.size() == b.size(), "BuildConeHecke: the cell lists differ");
        for (size_t i = 0; i < a.size(); ++i)
            ENGINE_CHECK_MSG(a[i].s == b[i].s && a[i].rep == b[i].rep && a[i].t == b[i].t && a[i].boundary == b[i].boundary,
                             "BuildConeHecke: the cell lists differ");
    }
    // C-D7p referee m3: a cap hit while the operator is assembled (hecke::HeckeOverflow in the exact products E = v_i s_j^{-1} g gamma
    // w^{-1}) REFUSES; any other exception propagates
    try {
        const hecke::level::Gamma0LevelN A(N);
        const uint32_t m = A.m(), d = n + 1, D = m * d;
        H.D = D;
        const std::vector<Mat22> r = hecke::level::Gamma0SubTransversal(N, p);
        ENGINE_CHECK_MSG(r.size() == p + 1, "BuildConeHecke: the transversal of Gamma_0(pN) \\ Gamma_0(N) has the wrong size");
        std::vector<Mat22> w(m + 1), w_inv(m + 1);
        for (uint32_t a = 1; a <= m; ++a) {
            w[a] = A.rep(a);
            w_inv[a] = w[a].inv_unimodular();
        }
        // per (i, t): j, a, v_i s_j^{-1} g
        struct It {
            uint32_t i, j, a;
            Mat22 vd;
        };
        std::vector<It> its;
        for (uint32_t i = 1; i <= m; ++i)
            for (const Mat22& rt : r) {
                const Mat22 x = rt * w[i];
                uint32_t j = 0;
                const Mat22 k = lift.twisted_decomposition(x, j);   // x = h s_j, k = g^{-1} h g
                ENGINE_CHECK_MSG(k.det() == 1, "BuildConeHecke: g^{-1} h g is not in Gamma (prop:levelcorrespondence(1))");
                const uint32_t a = A.class_of(k);
                ENGINE_CHECK_MSG(A.contains(k * w_inv[a]), "BuildConeHecke: k w_a^{-1} not in A");
                its.push_back(It{i, j, a, w[i] * lift.coset_delta_matrix(j)});
            }
        // the values read: tau_Iso(s_j e_c) from the lift; under TestOffBoundaryTerm one term off Isob planted (validation only)
        const bool plant = datum == ConeDatum::TestOffBoundaryTerm;
        uint32_t plant_b = UINT32_MAX, plant_c2 = UINT32_MAX;
        integral_model::IsoChain planted;
        if (plant && q_top >= 1) {
            const auto& c1 = L.cells(1);
            for (uint32_t c = 0; c < c1.size(); ++c) {
                if (c1[c].boundary && plant_b == UINT32_MAX) plant_b = c;
                if (!c1[c].boundary && plant_c2 == UINT32_MAX) plant_c2 = c;
            }
            ENGINE_CHECK_MSG(plant_b != UINT32_MAX && plant_c2 != UINT32_MAX, "BuildConeHecke: no boundary / bulk cell in degree 1");
            planted = lift.value(1, 1, plant_b);
            planted.c[plant_c2].add(integral_model::key_of(Mat22::I()), 1);
        }
        auto value = [&](uint32_t q, uint32_t j, uint32_t c) -> const integral_model::IsoChain& {
            if (plant && q == 1 && j == 1 && c == plant_b) return planted;
            return lift.value(q, j, c);
        };
        // one term (gamma, mul) of tau_Iso(s_j e_c) at e_c2, cell c in row block i: accumulated into T at (row cell rc, column cell cc)
        auto accumulate = [&](std::map<integral_model::GKey, DenseBuf<uint32_t>>& memo, DenseBuf<uint32_t>& T, const It& it,
                              uint32_t rc, uint32_t cc, const integral_model::GKey& gk, int64_t mul) {
            const Mat22 gam = integral_model::mat_of(gk);
            const uint32_t a2 = A.class_of(w[it.a] * gam);
            const Mat22 E = it.vd * gam * w_inv[a2];   // v_i s_j^{-1} g gamma w_{a.gamma}^{-1}, det p
            ENGINE_CHECK_MSG(E.det() == static_cast<int64_t>(p), "BuildConeHecke: a block matrix has det != p");
            const integral_model::GKey key = integral_model::key_of(E);
            auto mi = memo.find(key);
            if (mi == memo.end()) mi = memo.emplace(key, integral_model::rho_n(ctx, E, n)).first;
            const DenseBuf<uint32_t>& Rb = mi->second;
            const uint32_t s = Signed(ctx, mul);
            const uint32_t r0 = rc * D + (it.i - 1) * d, c0 = cc * D + (a2 - 1) * d;
            for (uint32_t x = 0; x < d; ++x)
                for (uint32_t y = 0; y < d; ++y) {
                    uint32_t& e = T.at(r0 + x, c0 + y);
                    e = ctx.add(e, ctx.mul(s, Rb.at(x, y)));
                }
        };
        std::map<integral_model::GKey, DenseBuf<uint32_t>> memo, memo_b;
        H.strict_identity = H.bdry_closed = true;
        for (uint32_t q = 0; q <= q_top; ++q) {
            const auto& cells = L.cells(q);
            const uint32_t W = static_cast<uint32_t>(cells.size());
            // T~_Iso on the full layout
            DenseBuf<uint32_t> T(W * D, W * D);
            for (uint32_t c = 0; c < W; ++c)
                for (const It& it : its) {
                    const integral_model::IsoChain& v = value(q, it.j, c);
                    for (uint32_t c2 = 0; c2 < W; ++c2)
                        for (const auto& [gk, mul] : v.c[c2].terms) {
                            if (cells[c].boundary && !cells[c2].boundary) H.bdry_closed = false;   // tau_Iso(Isob) in Isob
                            accumulate(memo, T, it, c, c2, gk, mul);
                            ++H.terms;
                        }
                }
            // T~_Isob on Isob's own layout (boundary cells renumbered 0 .. Wb - 1), assembled SEPARATELY from tau^d_Iso: the
            // Isob-component of tau_Iso(s_j e_b), b a boundary cell; a term off Isob is not part of tau^d_Iso and is dropped (counted)
            std::vector<uint32_t> bpos(W, UINT32_MAX), bcell;
            for (uint32_t c = 0; c < W; ++c)
                if (cells[c].boundary) {
                    bpos[c] = static_cast<uint32_t>(bcell.size());
                    bcell.push_back(c);
                }
            const uint32_t Wb = static_cast<uint32_t>(bcell.size());
            DenseBuf<uint32_t> Tb(Wb * D, Wb * D);
            for (uint32_t b = 0; b < Wb; ++b)
                for (const It& it : its) {
                    const integral_model::IsoChain& v = value(q, it.j, bcell[b]);
                    for (uint32_t c2 = 0; c2 < W; ++c2) {
                        if (bpos[c2] == UINT32_MAX) {
                            H.bdry_dropped += v.c[c2].terms.size();
                            continue;
                        }
                        for (const auto& [gk, mul] : v.c[c2].terms) {
                            accumulate(memo_b, Tb, it, b, bpos[c2], gk, mul);
                            ++H.bdry_terms;
                        }
                    }
                }
            // the strict identity rres T~_Iso = T~_Isob rres, entrywise: rres is the block selection of the boundary cells, so
            // (rres T~_Iso)(b x, col) = T~_Iso(bcell[b] x, col) and (T~_Isob rres)(b x, col) = T~_Isob(b x, bpos(col) y), 0 off Isob
            for (uint32_t b = 0; b < Wb; ++b)
                for (uint32_t x = 0; x < D; ++x)
                    for (uint32_t c = 0; c < W; ++c)
                        for (uint32_t y = 0; y < D; ++y) {
                            const uint32_t lhs = T.at(bcell[b] * D + x, c * D + y);
                            const uint32_t rhs = bpos[c] == UINT32_MAX ? 0u : Tb.at(b * D + x, bpos[c] * D + y);
                            ++H.entries_checked;
                            if (lhs != rhs) H.strict_identity = false;
                        }
            H.bulk.push_back(std::move(T));
            H.bdry.push_back(std::move(Tb));
        }
    } catch (const std::exception& ex) {
        const PairRefusal code = ClassifyPairException(ex);
        if (code == PairRefusal::None) throw;
        H.bulk.clear();
        H.bdry.clear();
        return refuse(code, std::string("the operator's assembly stopped: ") + ex.what());
    }
    if (!H.strict_identity)
        return refuse(PairRefusal::StrictIdentity, "the strict identity rres T~_Iso = T~_Isob rres fails entrywise (T~_Isob assembled separately from tau^d_Iso)");
    if (!H.bdry_closed && datum != ConeDatum::TestOffBoundaryTerm) return refuse(PairRefusal::IsobClosure, "the pair lift does not preserve Isob on this layout (tau_Iso(Isob) not in Isob)");
    if (datum == ConeDatum::TestOffBoundaryTerm) return refuse(PairRefusal::None, "planted off-boundary term NOT detected by the strict identity");
    return H;
}

}  // namespace pair
}  // namespace integral_descent
