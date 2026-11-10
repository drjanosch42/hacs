// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeLadicCellValidation.cpp -- see the header.
//
// hecke-extensions slot 11, WP-18c: the digit pipeline route Hecke/LadicCell
// against the pipeline-level oracle matrix ([E] v1.12.0-pre.2 sec:ring).
#include "HeckeLadicCellValidation.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "LadicCell.h"
#include "CoefficientRing.h"
#include "HeckeCosetTables.h"
#include "HeckeOperatorFamily.h"
#include "HeckeOperatorOnK.h"
#include "HeckeSettings.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Integral/IntegralCharpolyCrt.h"
#include "../Campaign/CampaignDriver.h"
#include "../Campaign/CampaignSpec.h"
#include "../../ModArithmetic/DenseElim.h"
#include "../../ModArithmetic/LadicInduction.h"
#include "../../ModArithmetic/ModCtx.h"

namespace {

using namespace hecke::ladic_cell;
using engine::u128;
using Serial = std::vector<uint32_t>;

struct Checker {
    uint32_t checks = 0, fails = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++fails;
            if (fails <= 40) std::cout << "  FAIL: " << what << "\n";
        }
    }
};

// Keep the first k' of K base-ell digits of every entry of a serialisation.
Serial Truncate(const Serial& s, uint32_t K, uint32_t kp) {
    Serial out{s[0], s[1]};
    const size_t n = static_cast<size_t>(s[0]) * s[1];
    for (size_t e = 0; e < n; ++e)
        for (uint32_t d = 0; d < kp; ++d) out.push_back(s[2 + e * K + d]);
    return out;
}

Residue DigitsOfU64(uint64_t v, uint64_t ell, uint32_t j) {
    Residue d(j);
    for (uint32_t i = 0; i < j; ++i) { d[i] = static_cast<uint32_t>(v % ell); v /= ell; }
    return d;
}

// ---------------------------------------------------------------------------
// The shipped E8 route at (n, ell, k), uint64_t, with its objects serialised.
// ---------------------------------------------------------------------------
struct Native {
    std::vector<Serial> C, X, Zc, H, decB, decG, decU;
    std::vector<std::vector<uint32_t>> decE, decP, dK, inv;
    std::vector<std::vector<Serial>> Theta, A;
    uint32_t r0 = 0, r1 = 0;
    bool certified = false;
    std::vector<TypeQ> type;
    std::vector<std::vector<Residue>> charpoly;   // per prime (empty if uncertified)
};

Native RunNative(uint32_t n, uint64_t ell, uint32_t k, const std::vector<uint64_t>& primes) {
    Native out;
    const hecke::ScopedCoefficientRing ring(k);
    const engine::ModCtx<uint64_t> ctx = hecke::CoefficientRingContext<uint64_t>("HeckeLadicCellValidation", ell, k);
    std::unique_ptr<SimplicialComplex> K;
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        K.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        model.reset(new hecke::SL2ZTessellationTables(*K));
    }
    LocalSystemSimplicialComplexT<uint64_t> lssc(*K, hecke::GenerateSL2ZRelationsRing<uint64_t>(*K, n, ctx), n + 1, ctx.p);
    EquivariantCohomologyOnKT<uint64_t> coh(&lssc);
    coh.compute_cohomology();
    for (uint32_t q = 0; q < coh.num_degrees(); ++q) {
        const auto& M = coh.module_degree(q);
        out.C.push_back(SerialiseNative(M.C, ell, k));
        out.X.push_back(SerialiseNative(M.X, ell, k));
        out.Zc.push_back(SerialiseNative(M.Zc, ell, k));
        out.H.push_back(SerialiseNative(M.H, ell, k));
        out.decB.push_back(SerialiseNative(M.dec.basis, ell, k));
        out.decG.push_back(SerialiseNative(M.dec.Gamma, ell, k));
        out.decU.push_back(SerialiseNative(M.dec.U, ell, k));
        out.decE.push_back(M.dec.exponents);
        out.decP.push_back(M.dec.position);
        out.dK.push_back(M.dK);
        out.inv.push_back(coh.H_invariant_factors(q));
    }
    out.r0 = coh.r_K(0);
    out.r1 = coh.r_K(1);
    out.certified = coh.ladic_type_certified();
    if (out.certified)
        for (uint32_t q = 0; q < 3; ++q) {
            const auto t = coh.ladic_type(q);
            out.type.push_back(TypeQ{t.free_rank, t.torsion});
        }
    for (uint64_t p : primes) {
        std::unique_ptr<hecke::HeckeCosetTablesT<uint64_t>> cosets;
        std::unique_ptr<hecke::HeckeTauTableT<uint64_t>> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTablesT<uint64_t>(p, 1, ell, n));
            tau.reset(new hecke::HeckeTauTableT<uint64_t>(*K, *model, *cosets, ell, n));
        }
        hecke::HeckeOperatorOnKT<uint64_t> op(&coh, cosets.get(), tau.get());
        op.compute();
        std::vector<Serial> th, a;
        for (uint32_t q = 0; q < 3; ++q) {
            th.push_back(SerialiseNative(op.Theta_module(q), ell, k));
            a.push_back(SerialiseNative(op.H_matrix_module(q), ell, k));
        }
        out.Theta.push_back(th);
        out.A.push_back(a);
        std::vector<Residue> cp;
        if (out.certified) {
            const auto fc = op.free_charpoly_H1();
            for (uint64_t c : fc.coeffs) cp.push_back(DigitsOfU64(c, ell, fc.precision));
        }
        out.charpoly.push_back(cp);
    }
    return out;
}

std::string Cell(uint32_t n, uint64_t ell, uint32_t k) {
    return "(" + std::to_string(n) + ", " + std::to_string(ell) + ") k=" + std::to_string(k);
}

// ---------------------------------------------------------------------------
// (R) the per-cell rule and the grammar
// ---------------------------------------------------------------------------
template<class F> bool ThrowsNI(F&& f, const std::string& needle) {
    try { f(); } catch (const hecke::NotImplementedInThisBuild& e) {
        return needle.empty() || std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) { return false; }
    return false;
}

void ArmRule(Checker& g) {
    g.check(ClassifyCell(691, 6) == CellRoute::Native, "(R) 691^6 < 2^64: native");
    g.check(ClassifyCell(691, 7) == CellRoute::Digit, "(R) 691^7 >= 2^64: digit");
    g.check(ClassifyCell(5, 27) == CellRoute::Native && ClassifyCell(5, 28) == CellRoute::Digit, "(R) 5^27 native, 5^28 digit");
    g.check(ClassifyCell(2, 63) == CellRoute::Native && ClassifyCell(2, 64) == CellRoute::RefusedSmallEll,
            "(R) 2^63 native, 2^64 refused (ell in {2, 3})");
    g.check(ClassifyCell(3, 40) == CellRoute::Native && ClassifyCell(3, 41) == CellRoute::RefusedSmallEll,
            "(R) 3^40 native, 3^41 refused (ell in {2, 3})");
    g.check(CeilLog2(691) == 10 && ClassifyCell(691, 819) == CellRoute::Digit &&
                ClassifyCell(691, 820) == CellRoute::RefusedBits,
            "(R) the 8192-bit cap: 691^819 digit (8190 bits), 691^820 refused (8200 bits)");
    g.check(WordLevelMax(691) == 6 && WordLevelMax(5) == 27 && WordLevelMax(2) == 63, "(R) K_w(691) = 6, K_w(5) = 27, K_w(2) = 63");
    g.check(ThrowsNI([] { RequireDigitCell("probe", 10, 3, 41, Width::W64); }, "ell = 3"),
            "(R) ell = 3 beyond 2^64 is refused by name");
    g.check(ThrowsNI([] { RequireDigitCell("probe", 10, 691, 820, Width::W64); }, "8200 bits"),
            "(R) the bit cap refuses with the predicted figure");
    g.check(ThrowsNI([] { RequireDigitCell("probe", 200000, 691, 7, Width::W64); }, "GB > 40 GB"),
            "(R) the memory cap refuses with the predicted figure");
    g.check(ThrowsNI([] { RequireDigitCell("probe", 11, 691, 7, Width::W64); }, "n > 0 even"),
            "(R) odd n is refused (the K-mode's range)");
    g.check(ThrowsNI([] { RequireDigitCell("probe", 0, 691, 7, Width::W64); }, "n > 0 even"), "(R) n = 0 is refused");
    bool native_logic = false;
    try { RequireDigitCell("probe", 10, 691, 6, Width::W64); } catch (const std::logic_error&) { native_logic = true; }
    g.check(native_logic, "(R) a native cell never reaches the digit route (logic_error)");
    // the grammar: 4096 valid, 4097 invalid, 64 now valid
    for (const std::string v : {"Z/ell^64", "Z/ell^128", "Z/ell^4096"}) {
        hecke::MathSettings s;
        bool ok = false;
        try { ok = s.set("coefficient_ring", v, hecke::SettingSource::Console) && s.text("coefficient_ring") == v; }
        catch (...) { ok = false; }
        g.check(ok, "(R) '" + v + "' parses and round-trips (cap 4096)");
    }
    for (const std::string v : {"Z/ell^4097", "Z/ell^10000", "Z/ell^04096"}) {
        hecke::MathSettings s;
        bool inv = false;
        try { s.set("coefficient_ring", v, hecke::SettingSource::Console); } catch (const std::invalid_argument&) { inv = true; } catch (...) {}
        g.check(inv, "(R) '" + v + "' is std::invalid_argument");
    }
    // ENGINE_LADIC_WIDTH (restored afterwards)
    const char* saved = std::getenv("ENGINE_LADIC_WIDTH");
    const std::string saved_s = saved ? saved : "";
    unsetenv("ENGINE_LADIC_WIDTH");
    g.check(EngineLadicWidth() == Width::W64, "(R) ENGINE_LADIC_WIDTH unset -> 64");
    setenv("ENGINE_LADIC_WIDTH", "32", 1);
    g.check(EngineLadicWidth() == Width::W32, "(R) ENGINE_LADIC_WIDTH=32");
    setenv("ENGINE_LADIC_WIDTH", "128", 1);
    g.check(EngineLadicWidth() == Width::W128, "(R) ENGINE_LADIC_WIDTH=128");
    setenv("ENGINE_LADIC_WIDTH", "48", 1);
    bool bad = false;
    try { (void)EngineLadicWidth(); } catch (const std::invalid_argument&) { bad = true; }
    g.check(bad, "(R) ENGINE_LADIC_WIDTH=48 is std::invalid_argument");
    if (saved) setenv("ENGINE_LADIC_WIDTH", saved_s.c_str(), 1);
    else unsetenv("ENGINE_LADIC_WIDTH");
    // residue text
    g.check(ResidueText(ResidueOfInt(-2001, 691, 40), 691) == "-2001" &&
                ResidueText(ResidueOfInt(1180224, 691, 7), 691) == "1180224",
            "(R) ResidueText: the signed decimal of least absolute value");
    Residue big(40, 690);
    big[39] = 345;
    g.check(ResidueText(big, 691).rfind("ell-adic:[", 0) == 0, "(R) ResidueText: base-ell digits beyond int64");
    std::cout << "  (R) the per-cell rule, the grammar cap 4096, ENGINE_LADIC_WIDTH, the residue text\n";
}

// ---------------------------------------------------------------------------
// (i) the digit route forced below 2^64 vs the shipped u64 route
// ---------------------------------------------------------------------------
void CompareWithNative(Checker& g, const CellRecord& r, const Native& nv, const std::string& tag) {
    const Objects& o = r.objects;
    bool ok = o.C.size() == 3;
    for (uint32_t q = 0; q < 3 && ok; ++q) {
        const std::string t = tag + " q=" + std::to_string(q);
        g.check(o.C[q] == nv.C[q], t + ": C^q_eq");
        g.check(o.X[q] == nv.X[q], t + ": X_q");
        g.check(o.Zc[q] == nv.Zc[q], t + ": Z^q");
        g.check(o.H[q] == nv.H[q], t + ": the cyclic basis b_i (flat)");
        g.check(o.dec_basis[q] == nv.decB[q] && o.dec_Gamma[q] == nv.decG[q] && o.dec_U[q] == nv.decU[q] &&
                    o.dec_exponents[q] == nv.decE[q] && o.dec_position[q] == nv.decP[q],
                t + ": the (c1)-(c5) decomposition");
        g.check(r.degrees[q].dK == nv.dK[q], t + ": dK");
        g.check(r.degrees[q].invariant_factors_K == nv.inv[q], t + ": invariant factors of H^q");
    }
    g.check(ok, tag + ": objects kept");
    g.check(r.r_K0 == nv.r0 && r.r_K1 == nv.r1 && r.kmode_certified == nv.certified, tag + ": r_K and the K-mode verdict");
    if (nv.certified) {
        bool same = nv.type.size() == r.type.size();
        for (size_t q = 0; same && q < nv.type.size(); ++q)
            same = nv.type[q].free_rank == r.type[q].free_rank && nv.type[q].torsion == r.type[q].torsion;
        g.check(same, tag + ": ladic_type^q");
    }
    g.check(o.Theta.size() == nv.Theta.size() && r.hecke.size() == nv.Theta.size(), tag + ": the Hecke list");
    for (size_t i = 0; i < nv.Theta.size() && i < o.Theta.size(); ++i) {
        const std::string t = tag + " T_" + std::to_string(r.hecke[i].p);
        for (uint32_t q = 0; q < 3; ++q) {
            g.check(o.Theta[i][q] == nv.Theta[i][q], t + " q=" + std::to_string(q) + ": Theta^{(q)}");
            g.check(o.A[i][q] == nv.A[i][q], t + " q=" + std::to_string(q) + ": A^{(q)} (c5)");
        }
        g.check(r.hecke[i].routes_equal, t + ": both routes equal");
        if (nv.certified) g.check(r.hecke[i].charpoly == nv.charpoly[i], t + ": the free charpoly");
    }
}

void ArmNative(Checker& g, std::map<std::string, CellRecord>& keep) {
    struct P { uint32_t n; uint64_t ell; uint32_t k; std::vector<uint64_t> primes; };
    const std::vector<P> points = {
        {10, 691, 2, {2, 3, 5}}, {6, 5, 2, {2, 3}},  {16, 5, 2, {2, 3}}, {30, 5, 2, {2, 3}},
        {30, 5, 3, {2, 3}},      {60, 5, 2, {2, 3}}, {60, 5, 3, {2, 3}},
        {10, 691, 6, {2, 691}},  {30, 5, 26, {2, 5}},   // the u64 limit: 2 x 32 digits, U_ell's split
    };
    uint32_t cells = 0;
    for (const P& p : points) {
        const Native nv = RunNative(p.n, p.ell, p.k, p.primes);
        for (Width w : {Width::W32, Width::W64}) {
            Request rq;
            rq.n_sym = p.n;
            rq.ell = p.ell;
            rq.k = p.k;
            rq.primes = p.primes;
            rq.hida = false;
            rq.width = w;
            rq.keep_objects = true;
            rq.allow_native_range = true;
            rq.fixed_K = p.k;
            const std::string tag = "(i) " + Cell(p.n, p.ell, p.k) + " w" + std::to_string(static_cast<uint32_t>(w));
            try {
                const CellRecord r = RunCell(rq);
                g.check(r.precision.uct_match, tag + ": UCT (iii)");
                CompareWithNative(g, r, nv, tag + " L=" + std::to_string(r.precision.digits_L));
                if (w == Width::W32) std::cout << "  " << tag << " L=" << r.precision.digits_L << " vs shipped u64\n";
                if (w == Width::W64) keep[Cell(p.n, p.ell, p.k)] = r;
            } catch (const std::exception& e) {
                g.check(false, tag + ": threw " + e.what());
            }
            ++cells;
        }
    }
    std::cout << "  (i) " << cells << " digit runs vs the shipped u64 route (the seven E8 points and the u64 limit)\n";
}

// ---------------------------------------------------------------------------
// (ii) M-0 at (10, 691) and the O5 rows; (P) the bound; (H) the idempotent
// ---------------------------------------------------------------------------
std::vector<int64_t> PolyFromRoots(int64_t eis, int64_t cusp, uint32_t mult) {
    std::vector<int64_t> p{-eis, 1};   // low -> high
    for (uint32_t i = 0; i < mult; ++i) {
        std::vector<int64_t> q(p.size() + 1, 0);
        for (size_t j = 0; j < p.size(); ++j) { q[j] -= cusp * p[j]; q[j + 1] += p[j]; }
        p = q;
    }
    return p;
}

void ArmM0(Checker& g, std::map<std::string, CellRecord>& keep) {
    const std::vector<int64_t> want = {-1180224, -97776, -2001, 1};
    g.check(PolyFromRoots(2049, -24, 2) == want, "(ii) (x - 2049)(x + 24)^2 = x^3 - 2001 x^2 - 97776 x - 1180224");
    for (uint32_t k : {7u, 12u, 20u, 40u}) {
        Request rq;
        rq.n_sym = 10;
        rq.ell = 691;
        rq.k = k;
        rq.primes = {2, 691};
        rq.width = EngineLadicWidth();
        rq.keep_objects = (k == 12 || k == 40);
        const std::string tag = "(ii) " + Cell(10, 691, k);
        try {
            const CellRecord r = RunCell(rq);
            g.check(r.precision.certified && r.precision.e_max == 0 && r.precision.working == k, tag + ": certified, e_max = 0, K = k");
            g.check(r.precision.uct_match, tag + ": UCT (iii)");
            std::vector<Residue> wr;
            for (int64_t c : want) wr.push_back(ResidueOfInt(c, 691, k));
            g.check(r.hecke.size() == 2 && r.hecke[0].rank_free == 3 && r.hecke[0].precision == k &&
                        r.hecke[0].charpoly == wr,
                    tag + ": free_charpoly_H1(T_2) == (x - 2049)(x + 24)^2 mod 691^k");
            std::string txt;
            for (const Residue& c : r.hecke[0].charpoly) txt += " " + ResidueText(c, 691);
            std::cout << "  " << tag << ": chi(T_2) =" << txt << "  (L = " << r.precision.digits_L << ", w"
                      << r.precision.digit_width << ")\n";
            if (k == 7 || k == 20) {
                g.check(r.hida.present && r.hida.certified && r.hida.idempotent && r.hida.commutes &&
                            r.hida.complement_nilpotent && r.hida.rank_match,
                        "(H) " + Cell(10, 691, k) + ": the ordinary idempotent is certified");
                std::cout << "  (H) " << Cell(10, 691, k) << ": e certified, rank_F691(e) = " << r.hida.rank_e_mod_ell
                          << " of r = " << r.hida.rank << " (U_691^r ordinary rank " << r.hida.ordinary_rank_mod_ell
                          << "), chi_W = x^" << r.hida.nilpotent_dim << " p(x)\n";
            }
            if (rq.keep_objects) keep[Cell(10, 691, k)] = r;
        } catch (const std::exception& e) {
            g.check(false, tag + ": threw " + e.what());
        }
    }
    // Every O5 row at ell = 691, k = 7.
    for (const integral::O5Row& row : integral::O5Table()) {
        Request rq;
        rq.n_sym = row.n;
        rq.ell = 691;
        rq.k = 7;
        rq.primes = {row.p};
        rq.hida = false;
        rq.width = EngineLadicWidth();
        const std::string tag = "(ii) O5 " + Cell(row.n, 691, 7) + " T_" + std::to_string(row.p);
        try {
            const CellRecord r = RunCell(rq);
            const std::vector<int64_t> poly =
                PolyFromRoots(std::stoll(row.eis_root), row.cusp_root ? std::stoll(row.cusp_root) : 0, row.cusp_mult);
            std::vector<Residue> wr;
            for (int64_t c : poly) wr.push_back(ResidueOfInt(c, 691, 7));
            g.check(r.precision.certified && r.hecke[0].charpoly == wr, tag + ": == " + std::string(row.verbatim) + " mod 691^7");
        } catch (const std::exception& e) {
            g.check(false, tag + ": threw " + e.what());
        }
    }
    std::cout << "  (ii) M-0: (10, 691) at k in {7, 12, 20, 40} and the " << integral::O5Table().size()
              << " O5 rows at 691^7\n";
}

void ArmBound(Checker& g, std::map<std::string, CellRecord>& keep) {
    struct B { uint32_t n; uint64_t ell; uint32_t e_max; };
    for (const B& b : std::vector<B>{{20, 5, 1}, {30, 5, 2}, {60, 5, 2}}) {
        Request rq;
        rq.n_sym = b.n;
        rq.ell = b.ell;
        rq.k = 40;
        rq.primes = {2};
        rq.hida = false;
        rq.width = EngineLadicWidth();
        rq.keep_objects = (b.n == 30);
        const std::string tag = "(P) " + Cell(b.n, b.ell, 40);
        try {
            const CellRecord r = RunCell(rq);
            g.check(r.precision.e_max == b.e_max, tag + ": e_max = " + std::to_string(b.e_max) + " (got " +
                                                      std::to_string(r.precision.e_max) + ")");
            g.check(r.precision.working == 40 + b.e_max && r.precision.certified && r.precision.digit_certified &&
                        r.precision.uct_match,
                    tag + ": K = k + e_max, both certificates and UCT");
            if (!r.hecke.empty()) g.check(r.hecke[0].precision == 40, tag + ": the free data at precision k");
            std::string tried;
            for (uint32_t l : r.precision.word_levels_tried) tried += " " + std::to_string(l);
            std::cout << "  " << tag << ": e_max = " << r.precision.e_max << ", K = " << r.precision.working
                      << ", word levels tried" << tried << "\n";
            if (rq.keep_objects) keep[Cell(b.n, b.ell, 40)] = r;
            keep["(P) " + Cell(b.n, b.ell, 40)] = r;
        } catch (const std::exception& e) {
            g.check(false, tag + ": threw " + e.what());
        }
    }
    // the forced-uncertified control
    Request rq;
    rq.n_sym = 30;
    rq.ell = 5;
    rq.k = 40;
    rq.hida = false;
    rq.word_level_override = 2;
    bool refused = false;
    try {
        (void)RunCell(rq);
    } catch (const PrecisionInsufficient& e) {
        refused = e.kind == "ladic_precision" && e.budget_K == 2 && !e.complete &&
                  (e.reached_r0 < e.need_r0 || e.reached_r1 < e.need_r1) &&
                  std::string(e.what()).find("precision insufficient") != std::string::npos;
        std::cout << "  (P) (30, 5) at word level 2: refused, resumable -- r_K(d^0) " << e.reached_r0 << "/" << e.need_r0
                  << ", r_K(d^1) " << e.reached_r1 << "/" << e.need_r1 << "\n";
    } catch (...) {
    }
    g.check(refused, "(P) (30, 5) with the word level forced to 2: PrecisionInsufficient (resumable, never a guess)");
    bool is_ni = false;
    try { (void)RunCell(rq); } catch (const hecke::NotImplementedInThisBuild&) { is_ni = true; } catch (...) {}
    g.check(is_ni, "(P) and it is a NotImplementedInThisBuild (console exit 5)");
}

// ---------------------------------------------------------------------------
// (P') the precision claim where it does work (WP-14 pass 2, M-2): at the
// e_max > 0 cells, the free charpoly of T_2 at k = 40 (K = 40 + e_max) against
// (a) a digit run at k = 30 on the shared 30 digits and (b) the SHIPPED u64
// route at the largest word level K_w = 27, whose free charpoly has precision
// 27 - e_max; and (H') U_5 at (30, 5): a genuine Fitting split (a > 0).
// ---------------------------------------------------------------------------
Residue Head(const Residue& r, uint32_t j) { return Residue(r.begin(), r.begin() + j); }

void ArmPrimeBound(Checker& g, const std::map<std::string, CellRecord>& keep) {
    for (uint32_t n : {20u, 30u, 60u}) {
        const std::string tag = "(P') " + Cell(n, 5, 40);
        auto it = keep.find("(P) " + Cell(n, 5, 40));
        if (it == keep.end() || it->second.hecke.empty()) { g.check(false, tag + ": the (P) run is missing"); continue; }
        const CellRecord& r40 = it->second;
        const uint32_t e = r40.precision.e_max;
        g.check(e > 0, tag + ": e_max > 0 (the bound does work here)");
        try {
            Request rq;
            rq.n_sym = n;
            rq.ell = 5;
            rq.k = 30;
            rq.primes = {2};
            rq.hida = false;
            rq.width = EngineLadicWidth();
            const CellRecord r30 = RunCell(rq);
            const auto& a = r40.hecke[0].charpoly;
            const auto& b = r30.hecke[0].charpoly;
            bool same = a.size() == b.size() && r30.precision.e_max == e;
            for (size_t i = 0; same && i < a.size(); ++i) same = Head(a[i], 30) == b[i];
            g.check(same, tag + ": chi(T_2) at k = 40 == chi(T_2) at k = 30 on the shared 30 digits");
            const Native nv = RunNative(n, 5, 27, {2});
            g.check(nv.certified && nv.charpoly.size() == 1 && nv.charpoly[0].size() == a.size(),
                    tag + ": the shipped u64 route at K_w = 27 certifies, same degree");
            bool nat = nv.certified && nv.charpoly[0].size() == a.size();
            for (size_t i = 0; nat && i < a.size(); ++i) nat = nv.charpoly[0][i].size() == 27 - e && Head(a[i], 27 - e) == nv.charpoly[0][i];
            g.check(nat, tag + ": chi(T_2) at k = 40 == the shipped u64 route's at precision 27 - e_max = " +
                             std::to_string(27 - e));
            std::cout << "  " << tag << ": e_max = " << e << ", chi(T_2) degree " << a.size() - 1
                      << " agrees with k = 30 (30 digits) and with the shipped u64 route (" << 27 - e << " digits)\n";
        } catch (const std::exception& ex) {
            g.check(false, tag + ": threw " + ex.what());
        }
    }
    // (H') U_5 at (30, 5): rank 5, ordinary rank 1, chi(U_5) = x^4 (x - 1) mod 5.
    try {
        Request rq;
        rq.n_sym = 30;
        rq.ell = 5;
        rq.k = 30;   // a cell of the route (5^30 >= 2^64); K = 32
        rq.primes = {2, 5};
        rq.width = EngineLadicWidth();
        const CellRecord r = RunCell(rq);
        const HidaOut& h = r.hida;
        g.check(h.present && h.certified && h.idempotent && h.commutes && h.complement_nilpotent && h.rank_match,
                "(H') (30, 5) U_5: the ordinary idempotent is certified");
        g.check(h.rank == 5 && h.nilpotent_dim > 0 && h.rank_e_mod_ell == 1 && h.ordinary_rank_mod_ell == 1,
                "(H') (30, 5) U_5: r = 5, a genuine Fitting split (a > 0), rank_F5(e) = ordinary rank = 1");
        std::vector<uint32_t> low;
        for (const Residue& c : r.hecke[1].charpoly) low.push_back(c[0]);
        g.check(low == std::vector<uint32_t>({0, 0, 0, 0, 4, 1}), "(H') (30, 5): chi(U_5) == x^4 (x - 1) mod 5");
        std::cout << "  (H') (30, 5) k=30: U_5 on r = " << h.rank << ", chi_W = x^" << h.nilpotent_dim
                  << " p(x), rank_F5(e) = " << h.rank_e_mod_ell << ", certified = " << h.certified << "\n";
    } catch (const std::exception& ex) {
        g.check(false, std::string("(H') (30, 5): threw ") + ex.what());
    }
}

// ---------------------------------------------------------------------------
// (iv) cross-width, (v) truncation, (vi) native u128
// ---------------------------------------------------------------------------
void ArmCrossWidth(Checker& g) {
    struct C { uint32_t n; uint64_t ell; uint32_t k; std::vector<uint64_t> primes; };
    for (const C& c : std::vector<C>{{10, 691, 40, {2, 691}}, {30, 5, 40, {2, 5}}}) {
        std::string ref;
        for (Width w : {Width::W32, Width::W64, Width::W128}) {
            Request rq;
            rq.n_sym = c.n;
            rq.ell = c.ell;
            rq.k = c.k;
            rq.primes = c.primes;
            rq.width = w;
            rq.keep_objects = true;
            const std::string tag = "(iv) " + Cell(c.n, c.ell, c.k) + " w" + std::to_string(static_cast<uint32_t>(w));
            try {
                const CellRecord r = RunCell(rq);
                const std::string t = r.canonical_text();
                if (ref.empty()) ref = t;
                else g.check(t == ref, tag + ": the whole cell record equals w32's");
                std::cout << "  " << tag << " L=" << r.precision.digits_L << " m=" << r.precision.digit_m
                          << " record " << t.size() << " bytes\n";
            } catch (const std::exception& e) {
                g.check(false, tag + ": threw " + e.what());
            }
        }
    }
}

void ArmTruncation(Checker& g, const std::map<std::string, CellRecord>& keep) {
    struct T { std::string big; uint32_t K; std::string small; uint32_t kp; };
    for (const T& t : std::vector<T>{{Cell(10, 691, 40), 40, Cell(10, 691, 6), 6},
                                     {Cell(10, 691, 12), 12, Cell(10, 691, 6), 6},
                                     {Cell(30, 5, 40), 42, Cell(30, 5, 26), 26},
                                     {Cell(30, 5, 40), 42, Cell(30, 5, 3), 3}}) {
        auto a = keep.find(t.big), b = keep.find(t.small);
        const std::string tag = "(v) " + t.big + " -> " + std::to_string(t.kp);
        if (a == keep.end() || b == keep.end()) { g.check(false, tag + ": runs missing"); continue; }
        const Objects& A = a->second.objects;
        const Objects& Bo = b->second.objects;
        for (uint32_t q = 0; q < 3; ++q) {
            g.check(Truncate(A.C[q], t.K, t.kp) == Bo.C[q], tag + " q=" + std::to_string(q) + ": C^q_eq truncates");
            g.check(Truncate(A.X[q], t.K, t.kp) == Bo.X[q], tag + " q=" + std::to_string(q) + ": X_q truncates");
            if (!A.Theta.empty() && !Bo.Theta.empty())
                g.check(Truncate(A.Theta[0][q], t.K, t.kp) == Bo.Theta[0][q],
                        tag + " q=" + std::to_string(q) + ": Theta(T_2) truncates");
        }
    }
    std::cout << "  (v) truncation of C, X, Theta to native k'\n";
}

template<class W> engine::ladic::LadicMat<W> FromSerial(const Serial& s, uint64_t ell, uint32_t K) {
    const auto lay = engine::ladic::DigitLayout<W>::make(ell, K);
    engine::ladic::LadicMat<W> M(lay, s[0], s[1]);
    size_t o = 2;
    for (uint32_t i = 0; i < s[0]; ++i)
        for (uint32_t j = 0; j < s[1]; ++j) {
            M.set_ell_digits(i, j, std::vector<uint32_t>(s.begin() + o, s.begin() + o + K));
            o += K;
        }
    return M;
}

void ArmU128(Checker& g, const std::map<std::string, CellRecord>& keep) {
    auto it = keep.find(Cell(10, 691, 12));
    if (it == keep.end()) { g.check(false, "(vi) the (10, 691) k=12 run is missing"); return; }
    const CellRecord& r = it->second;
    const uint32_t K = 12;
    const engine::ModCtx<u128> ctx = engine::ModCtx<u128>::prime_power(691, K);
    auto check_matrix = [&](const Serial& s, const std::string& tag) {
        const auto M = FromSerial<uint64_t>(s, 691, K);
        if (M.rows() == 0 || M.cols() == 0) return;
        const engine::DenseBuf<u128> Mw = M.template to_word<u128>();
        const auto kn = engine::elim::kernel_module<u128>(ctx, engine::cview(Mw));
        const auto kd = engine::ladic::kernel_module(M);
        bool same = kn.pivots == kd.pivots && kn.pivot_v == kd.pivot_v;
        if (same) {
            const engine::DenseBuf<u128> Hd = kd.H.template to_word<u128>();
            same = Hd.rows == kn.H.rows && Hd.cols == kn.H.cols;
            for (uint32_t i = 0; same && i < Hd.rows; ++i)
                for (uint32_t j = 0; same && j < Hd.cols; ++j) same = Hd.at(i, j) == kn.H.at(i, j);
        }
        g.check(same, tag + ": kernel_module (digits) == native u128");
        g.check(engine::elim::invariant_factors<u128>(ctx, engine::cview(Mw)) == engine::ladic::invariant_factors(M),
                tag + ": invariant_factors (digits) == native u128");
    };
    for (uint32_t q = 0; q < 2; ++q) {
        check_matrix(r.objects.X[q], "(vi) (10, 691) 691^12 X_" + std::to_string(q));
        g.check(engine::elim::invariant_factors<u128>(ctx, engine::cview(FromSerial<uint64_t>(r.objects.X[q], 691, K)
                                                                             .template to_word<u128>())) ==
                    r.degrees[q].dK,
                "(vi) X_" + std::to_string(q) + ": the cell's dK == native u128");
    }
    check_matrix(r.objects.Theta[0][1], "(vi) (10, 691) 691^12 Theta^{(1)}(T_2)");
    std::cout << "  (vi) the cell's X_0, X_1, Theta^{(1)} at 691^12 (~2^113) through native u128\n";
}


// ---------------------------------------------------------------------------
// (C) the campaign's digit jobs ([E] v1.12.0-pre.2 (g)) -- into a temp dir,
// removed afterwards.
// ---------------------------------------------------------------------------
bool Contains(const std::string& h, const std::string& n) { return h.find(n) != std::string::npos; }

void ArmCampaign(Checker& g) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const fs::path dir = (ec ? fs::path(".") : base) / ("oc_wp18c_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    const std::string spec = (dir / "cell.spec").string(), out = (dir / "out").string();
    {
        std::ofstream f(spec);
        f << "coefficient_ring = Z/ell^7\nn_sym = 10\nell = 691\np = 2,691\nr = 1\n";
    }
    std::string a0 = spec, a1 = "--output", a2 = out, a3 = "--quiet";
    char* av[] = {&a0[0], &a1[0], &a2[0], &a3[0]};
    const int rc = campaign::RunCampaign(4, av);
    g.check(rc == 0, "(C) a digit campaign (10, 691) Z/ell^7 runs, exit 0");
    auto slurp = [](const std::string& path) {
        std::ifstream f(path);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    };
    std::string p2, p691;
    for (const auto& e : fs::directory_iterator(fs::path(out) / "jobs", ec)) {
        const std::string name = e.path().filename().string();
        if (Contains(name, "_p2_")) p2 = slurp(e.path().string());
        if (Contains(name, "_p691_")) p691 = slurp(e.path().string());
    }
    g.check(Contains(p2, "\"route\":\"digit\"") && Contains(p2, "\"modulus\":\"691^7\"") &&
                Contains(p2, "\"precision\":{\"precision_requested\":7,\"precision_working\":7,\"e_max\":0,") &&
                Contains(p2, "\"precision_certified\":true"),
            "(C) the digit job states the route, the textual modulus and the precision bound");
    g.check(Contains(p2, "\"coeffs\":[\"-1180224\",\"-97776\",\"-2001\",\"1\"]"),
            "(C) the digit job's free charpoly is (x - 2049)(x + 24)^2, signed decimals");
    g.check(Contains(p2, "\"status\":\"ok_partial\"") && Contains(p2, "\"part\":\"bockstein\",\"who\":\"LadicCell\"") &&
                Contains(p2, "\"part\":\"reduction\",\"who\":\"LadicCell\""),
            "(C) the Bockstein and the reduction are recorded refusals, by the route's guard");
    g.check(Contains(p691, "\"hida\":{\"precision\":7,\"rank\":3,") && Contains(p691, "\"certified\":true,\"e\":["),
            "(C) at p = ell the record carries the certified ordinary idempotent");
    const std::string tsv = slurp((fs::path(out) / "summary.tsv").string());
    g.check(Contains(tsv, "-1180224;-97776;-2001;1@7"), "(C) summary.tsv carries the digit charpoly");
    auto refused = [&](const std::string& text, const std::string& needle) {
        try {
            std::istringstream is(text);
            (void)campaign::ParseCampaignSpec(is, "<wp18c-spec>");
        } catch (const hecke::NotImplementedInThisBuild& e) {
            return Contains(e.what(), needle) && Contains(e.what(), "is not honoured in this build");
        } catch (...) {
            return false;
        }
        return false;
    };
    g.check(refused("coefficient_ring = Z/ell^41\nn_sym = 4\nell = 3\np = 2\nr = 1\n", "ell = 3"),
            "(C) spec: 3^41 refused (ell in {2, 3} beyond 2^64)");
    g.check(refused("coefficient_ring = Z/ell^815\nn_sym = 10\nell = 691\np = 2\nr = 1\n", "8200 bits"),
            "(C) spec: 691^815 refused (the 8192-bit cap at K = k + K_w - 1 = 820, m-3)");
    g.check(refused("coefficient_ring = Z/ell^7\nn_sym = 11\nell = 691\np = 2\nr = 1\n", "n > 0 even"),
            "(C) spec: odd n at a digit cell refused");
    bool admitted = true;
    try {
        std::istringstream is("coefficient_ring = Z/ell^814\nn_sym = 10\nell = 691\np = 2\nr = 1\n");
        (void)campaign::ParseCampaignSpec(is, "<wp18c-spec>");
    } catch (...) { admitted = false; }
    g.check(admitted, "(C) spec: 691^814 admitted (K <= 819, 8190 bits)");
    fs::remove_all(dir, ec);
    std::cout << "  (C) the campaign's digit jobs and the spec's per-cell rule\n";
}

} // namespace

uint32_t HeckeLadicCellValidation() {
    std::cout << "=== HeckeLadicCellValidation (slot-11 WP-18c: the digit pipeline route) ===\n";
    const auto t0 = std::chrono::steady_clock::now();
    Checker g;
    std::map<std::string, CellRecord> keep;
    auto guarded = [&](const char* name, const std::function<void()>& f) {
        try { f(); } catch (const std::exception& e) { g.check(false, std::string(name) + ": threw " + e.what()); }
    };
    guarded("(R)", [&] { ArmRule(g); });
    guarded("(i)", [&] { ArmNative(g, keep); });
    guarded("(ii)", [&] { ArmM0(g, keep); });
    guarded("(P)", [&] { ArmBound(g, keep); });
    guarded("(P')", [&] { ArmPrimeBound(g, keep); });
    guarded("(iv)", [&] { ArmCrossWidth(g); });
    guarded("(v)", [&] { ArmTruncation(g, keep); });
    guarded("(vi)", [&] { ArmU128(g, keep); });
    guarded("(C)", [&] { ArmCampaign(g); });
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeLadicCellValidation: " << g.checks << " checks, " << g.fails << " failing, " << s << " s\n";
    return g.fails;
}
