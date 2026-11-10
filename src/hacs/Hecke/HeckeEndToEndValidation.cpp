// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S12 -- End-to-end validation battery + benchmark driver.
//
// See HeckeEndToEndValidation.h for the file-level statement of what the three
// registered validations do and the binding gating rules. This file aggregates
// the whole integral-Hecke-action pipeline (Stages S1-S11 + FIX-negI) into one
// acceptance-gate battery. It re-exercises the pipeline directly (the same
// public construction recipe every V-stage uses: 7-vertex model K -> S2
// tessellation tables -> base local system L = Sym^n(F_ell^2) via the
// single-source-of-truth relation generator GenerateSL2ZRelationsWithGroupData
// -> EquivariantCohomologyOnK::compute_cohomology() -> HeckeCosetTables ->
// HeckeTauTable -> HeckeOperatorOnK::compute()), and maps every executed check
// back to at least one contract obligation.
//
// GATING PROVENANCE (matches the already-green V-stages, verified in the S12
// report):
//   * A-F4 + Sec 0.2 (S7V): assert dimension (O14) and char-poly (O5/O7) for
//     ALL ell >= 5, EXCEPT the mod-ell Dickson exceptional loci
//     (IsDicksonExceptional: n > 0 and (ell+1) | n), which are RECORDED. The
//     weight-12 (n=10) charpoly still holds at ell=5,7 -- the exceptional set
//     is specific, not simply n >= ell.
//   * FIX-negI (Sec 0.2): the base 7-vertex model now carries the central -I
//     self-relation on every simplex, so for ODD n the model gives H^* = 0
//     (oracle O2) for ell != 2 -- HARD-asserted here (post-FIX-negI it PASSES,
//     unlike the pre-FIX-negI S7V run recorded in HeckeOracleValidation.cpp).
//   * A-F1 (S9V): the Hecke-algebra relations are contract-backed only for
//     ell >= 5; the bad-prime p = ell case at ell in {2,3} asserts only
//     completion/well-typedness and RECORDS the rest (HeckeBadPrimeValidation).
//   * Auditor-verified constant set (Q7): only weights 12/16/24 a_p, the
//     Eisenstein eigenvalue 1 + p^{n+1}, and the Ramanujan 691 congruence are
//     hard-asserted; weights 18/20/22/26 are RUN and RECORDED only.

#include "HeckeEndToEndValidation.h"

#include "DicksonCount.h"
#include "HeckeCosetTables.h"
#include "HeckeTauTable.h"
#include "HeckeOperatorOnK.h"
#include "HeckeOperatorFamily.h"
#include "HeckeModuleDecomposition.h"
#include "InducedLocalSystem.h"
#include "FpPoly.h"
#include "Predicates.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

// ===========================================================================
// Small helpers.
// ===========================================================================

uint32_t ModOfInt64(int64_t x, uint32_t ell) {
    int64_t m = x % static_cast<int64_t>(ell);
    if (m < 0) m += static_cast<int64_t>(ell);
    return static_cast<uint32_t>(m);
}

// The mod-ell Dickson-invariant exceptional locus of Sym^n(F_ell^2) as an
// SL_2(F_ell)-module (FINDING B of Stage S7V's HeckeOracleValidation.cpp, and
// the A-F4 refinement of plan Sec. 0.2 -- FURTHER REFINED HERE, see the S12
// report).
//
// The reduction SL(2,Z) -> SL_2(F_ell) is surjective and rho_n factors through
// it, so H^0(SL(2,Z); Sym^n F_ell) = (Sym^n F_ell)^{SL_2(F_ell)}. The classical
// theorem of Dickson gives F_ell[x,y]^{SL_2(F_ell)} = F_ell[u, c] as a POLYNOMIAL
// ring on TWO generators: u = x^ell*y - x*y^ell of degree ell+1, and the second
// Dickson invariant c of degree ell*(ell-1) (the GL_2 invariant c_{2,0};
// c_{2,1} of degree ell^2-1 equals u^{ell-1}). Hence Sym^n acquires a nonzero
// SL_2(F_ell)-invariant -- and, empirically, a matching extra dim in H^1 --
// EXACTLY when n lies in the numerical semigroup < ell+1, ell*(ell-1) >, i.e.
//     n = a*(ell+1) + b*ell*(ell-1),   a,b >= 0,   n > 0.
// dim H^* legitimately differs from the char-0 value at these (n,ell), so
// dimension/charpoly are RECORDED there, not asserted.
//
// NOTE: Stage S7V's predicate used only the FIRST generator ((ell+1) | n),
// which sufficed for its grid (max tested n was 14, plus 22). The S12 grid
// includes n=20, which at ell=5 is EXACTLY the second Dickson degree
// ell*(ell-1) = 20 -- a genuine exceptional locus the (ell+1)|n test misses.
// This refinement adds precisely (20,5) over the S7V set; everything else
// (18/24 at 5; 8/16/24 at 7; 14 at 13) is unchanged.
// (hecke-extensions WP-2) Unified onto the public `Hecke/DicksonCount.h`
// (`DicksonInvariantCount`), which counts solutions of this same
// diophantine equation instead of stopping at the first; the boolean value
// returned here is unchanged bit-for-bit.
//
// (slot-7 WP-8, gap G9) The predicate is PUBLIC: `hecke::IsDicksonExceptional`
// in Hecke/DicksonCount.h (the name [P] Sec. 14.2 gives the gap), which the
// console's `dickson_exceptional(n, ell)` also calls.  The file-local copy that
// stood here is gone; the call below is the public function.

// Eichler-Shimura dim H^1(SL(2,Z); Sym^n(F_ell^2)) for even n and ell >= 5
// (no Tor obstruction, A-F4): 2*dim S_{n+2} + 1 for n >= 2, and the special
// trivial-coefficient value 0 for n = 0 (whose H^0 = 1 is handled separately).
// dim S_k (k even, k >= 4) = dim M_k - 1, dim M_k = floor(k/12) + [k !=2 (mod 12)].
uint32_t DimSk(uint32_t k) {
    if (k < 4) return 0;
    const uint32_t mk = (k % 12 == 2) ? (k / 12) : (k / 12 + 1);
    return mk - 1; // dim S_k = dim M_k - 1 for k >= 4
}

uint32_t ExpectedDimH1(uint32_t n) {
    if (n == 0) return 0;              // trivial coefficients: H^0 = 1, H^1 = 0
    if (n % 2 == 1) return 0;          // odd n => H^* = 0 (O2, ell != 2)
    return 2u * DimSk(n + 2u) + 1u;    // even n >= 2
}

// (X - root) over F_ell.
hecke::FpPoly XMinus(uint32_t ell, int64_t root) {
    return hecke::FpPoly(ell, std::vector<uint32_t>{ModOfInt64(-root, ell), 1u});
}

// ===========================================================================
// Obligation -> executed-check coverage tracker.
// ===========================================================================

class Coverage {
public:
    // Registers that a concrete executed check `test_id` (an oracle name, a
    // sub-battery name, or a fixture id) exercised obligation `obl`.
    void hit(const std::string& obl, const std::string& test_id) {
        auto& e = data_[obl];
        e.first.insert(test_id);
        ++e.second;
    }

    void print(const char* who) const {
        static const std::vector<std::pair<std::string, std::string>> kObligations = {
            {"obl:welldef",   "Theta_g well-defined: maps C^q_eq->C^q_eq, commutes with d "
                              "(HeckeOperatorOnK::compute runtime asserts i/ii/iii)"},
            {"obl:agreement", "two evaluation routes bit-identical (O12a); three-map factorization"},
            {"obl:tau",       "tau certification d.tau = tau.d, augmentation (HeckeTauTable::certify)"},
            {"obl:algebra",   "Hecke relations: [T2,T3]=0, T4=T2^2-p^{n+1}I, T(p,p)=p^n I (O9)"},
            {"obl:integral",  "integral normalization (A^(0)=p+1, recursion factor p^{n+1}); "
                              "never invert a det-p matrix (I7)"},
            {"obl:backend",   "exact int64/int128 backend, certified bound, overflow-checked (O13-adjacent)"},
            {"obl:fp",        "FP filter sign == exact sign; PURE_INTEGER on/off bit-identical (O13)"},
            {"obl:shapiro",   "Shapiro: H^*(K,R;L_{Gamma_0(N)}) dims (O11)"},
            {"obl:oracle",    "numerical oracles O1-O16 with exact expected values"},
        };
        std::cout << "\n==================================================================\n";
        std::cout << who << ": OBLIGATION -> EXECUTED-CHECK COVERAGE TABLE\n";
        std::cout << "==================================================================\n";
        bool all_covered = true;
        for (const auto& ob : kObligations) {
            auto it = data_.find(ob.first);
            const uint64_t count = (it == data_.end()) ? 0 : it->second.second;
            std::cout << "  " << ob.first << "  [" << count << " executed check(s)]\n";
            std::cout << "      " << ob.second << "\n";
            std::cout << "      test IDs: ";
            if (it == data_.end() || it->second.first.empty()) {
                std::cout << "(NONE)";
                all_covered = false;
            } else {
                bool first = true;
                for (const auto& id : it->second.first) {
                    if (!first) std::cout << ", ";
                    std::cout << id;
                    first = false;
                }
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------------------------------------------\n";
        std::cout << "  Every obligation exercised by >= 1 executed check: "
                  << (all_covered ? "YES" : "NO") << "\n";
        std::cout << "==================================================================\n";
    }

    bool all_covered() const {
        static const char* kNames[] = {
            "obl:welldef", "obl:agreement", "obl:tau", "obl:algebra", "obl:integral",
            "obl:backend", "obl:fp", "obl:shapiro", "obl:oracle"};
        for (const char* nm : kNames) {
            auto it = data_.find(nm);
            if (it == data_.end() || it->second.first.empty()) return false;
        }
        return true;
    }

private:
    std::map<std::string, std::pair<std::set<std::string>, uint64_t>> data_;
};

// ===========================================================================
// Pipeline fixtures (same public construction recipe as every V-stage).
// ===========================================================================

std::vector<LocalSystemSimplicialComplex::SimplexRelation> BuildRelations(
        const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
    std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
    for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
        rels.push_back(datum.rel);
    }
    return rels;
}

struct CohomologyFixture {
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;

    CohomologyFixture(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell)
        : lssc(K, BuildRelations(K, n_sym, ell), n_sym + 1, ell), coh(&lssc) {
        coh.compute_cohomology();
    }
};

struct HeckeFixture {
    hecke::HeckeCosetTables cosets;
    hecke::HeckeTauTable tau;      // ctor MANDATORILY certifies (obl:tau); throws on failure
    hecke::HeckeOperatorOnK op;

    HeckeFixture(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                 EquivariantCohomologyOnK& coh, uint64_t p_hecke, uint32_t ell, uint32_t n_sym)
        : cosets(p_hecke, /*r=*/1, ell, n_sym),
          tau(K, model, cosets, ell, n_sym),
          op(&coh, &cosets, &tau) {
        op.compute();                // runtime asserts i/ii/iii + O12(a) route bit-identity
    }
};

// ===========================================================================
// The full oracle-matrix driver.
// ===========================================================================

class E2EContext {
public:
    E2EContext(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model)
        : K_(K), model_(model) {}

    uint32_t failures() const { return failures_; }

    // ------------------------------------------------------------------
    // Fixture caches (base cohomology per (n,ell); Hecke op per (p,n,ell)).
    // ------------------------------------------------------------------
    CohomologyFixture& Coh(uint32_t n, uint32_t ell) {
        const auto key = std::make_pair(n, ell);
        auto it = coh_cache_.find(key);
        if (it == coh_cache_.end()) {
            it = coh_cache_.emplace(key, std::make_unique<CohomologyFixture>(K_, n, ell)).first;
        }
        return *it->second;
    }

    // Returns nullptr (and records a failure) if the Hecke pipeline throws.
    HeckeFixture* Hecke(uint64_t p, uint32_t ell, uint32_t n) {
        const auto key = std::make_tuple(p, ell, n);
        auto it = hecke_cache_.find(key);
        if (it != hecke_cache_.end()) return it->second.get();
        try {
            CohomologyFixture& cf = Coh(n, ell);
            auto hf = std::make_unique<HeckeFixture>(K_, model_, cf.coh, p, ell, n);
            HeckeFixture* raw = hf.get();
            hecke_cache_.emplace(key, std::move(hf));
            // Building the fixture already exercised these obligations:
            const std::string id = "n=" + std::to_string(n) + ",ell=" + std::to_string(ell) +
                                   ",p=" + std::to_string(p);
            cov_.hit("obl:welldef", id);
            cov_.hit("obl:tau", id);
            cov_.hit("obl:agreement", id);
            cov_.hit("obl:backend", id);
            return raw;
        } catch (const std::exception& e) {
            Fail("PIPELINE BUILD FAILED at n=" + std::to_string(n) + " ell=" + std::to_string(ell) +
                 " p=" + std::to_string(p) + ": " + e.what());
            return nullptr;
        }
    }

    Coverage& coverage() { return cov_; }

    // ------------------------------------------------------------------
    // Main matrix.
    // ------------------------------------------------------------------
    void RunMainMatrix() {
        const std::vector<uint32_t> ns = {0, 1, 2, 3, 8, 10, 11, 14, 16, 18, 20, 22, 24};
        const std::vector<uint32_t> ells = {5, 7, 13, 101, 691, 10007, 134217757};
        for (uint32_t n : ns) {
            for (uint32_t ell : ells) {
                RunCell(n, ell);
            }
        }
    }

    // O11 Shapiro cross-check (obl:shapiro).
    void RunShapiro() {
        for (uint32_t ell : {5u, 101u}) {
            try {
                const hecke::HeckeCosetTables cosets(/*p=*/2, /*r=*/1, ell, /*n=*/0);
                const LocalSystemSimplicialComplex L_H =
                    hecke::MakeInducedLocalSystemGamma0(K_, cosets, ell, /*n=*/0);
                EquivariantCohomologyOnK eq(&L_H);
                eq.compute_cohomology();
                const auto& d = eq.get_H_eq_dimensions();
                const uint32_t h0 = d.size() > 0 ? d[0] : 0;
                const uint32_t h1 = d.size() > 1 ? d[1] : 0;
                const uint32_t h2 = d.size() > 2 ? d[2] : 0;
                cov_.hit("obl:shapiro", "O11");
                cov_.hit("obl:oracle", "O11");
                if (h0 != 1 || h1 != 1) {
                    Fail("O11 Shapiro: L_{Gamma_0(2)} n=0 ell=" + std::to_string(ell) +
                         ": expected (H^0,H^1)=(1,1), got (" + std::to_string(h0) + "," +
                         std::to_string(h1) + ")");
                } else {
                    std::cout << "[O11] Shapiro OK: L_{Gamma_0(2)} n=0 ell=" << ell
                              << " (H^0,H^1,H^2)=(" << h0 << "," << h1 << "," << h2
                              << ") [H^2 recorded]" << std::endl;
                }
            } catch (const std::exception& e) {
                cov_.hit("obl:shapiro", "O11");
                Fail(std::string("O11 Shapiro: EXCEPTION at ell=") + std::to_string(ell) + ": " + e.what());
            }
        }
    }

    // O9 Hecke-algebra relations (obl:algebra + obl:integral).
    void RunAlgebra() {
        for (uint32_t ell : {13u, 101u}) {
            for (uint32_t n : {0u, 2u, 10u}) {
                try {
                    CohomologyFixture& cf = Coh(n, ell);
                    hecke::HeckeOperatorFamily fam(&cf.coh, K_, model_, ell, n);
                    const auto& dims = cf.coh.get_H_eq_dimensions();
                    const std::vector<Matrix>& A2 = fam.T_p(2);
                    const std::vector<Matrix>& A3 = fam.T_p(3);
                    const std::vector<Matrix> T4 = fam.T_p_pow_direct(2, 2); // independent full T_4
                    const std::vector<Matrix> Tpp2 = fam.T_pp(2);
                    const uint32_t s_np1 = ModArithmetic::mod_pow(2u % ell, n + 1u, ell);
                    const uint32_t s_n = ModArithmetic::mod_pow(2u % ell, n, ell);
                    cov_.hit("obl:algebra", "O9");
                    cov_.hit("obl:integral", "O9-recursion-p^{n+1}");
                    cov_.hit("obl:oracle", "O9");
                    bool ok = true;
                    for (uint32_t q = 0; q < dims.size(); ++q) {
                        const uint32_t d = dims[q];
                        if (A2[q] * A3[q] != A3[q] * A2[q]) { ok = false; break; }
                        Matrix I(d, ell);
                        const Matrix expT4 = A2[q] * A2[q] - I * s_np1;
                        if (T4[q] != expT4) { ok = false; break; }
                        if (Tpp2[q] != I * s_n) { ok = false; break; }
                    }
                    if (!ok) {
                        Fail("O9 Hecke-algebra relations FAILED at ell=" + std::to_string(ell) +
                             " n=" + std::to_string(n));
                    } else {
                        std::cout << "[O9] algebra OK at ell=" << ell << " n=" << n
                                  << " ([T2,T3]=0, T4=T2^2-2^{n+1}I, T(2,2)=2^n I)" << std::endl;
                    }
                } catch (const std::exception& e) {
                    Fail("O9 Hecke-algebra: EXCEPTION at ell=" + std::to_string(ell) + " n=" +
                         std::to_string(n) + ": " + e.what());
                }
            }
        }
    }

    // O13-style FP-filter differential check (obl:fp): the certified sign
    // predicate must equal the exact __int128 sign on every sample, under BOTH
    // HECKE_PURE_INTEGER settings (the cross-config bit-identity is the DoD
    // protocol; here we verify the in-process correctness directly).
    void RunFpFilter() {
        uint64_t seed = 0x123456789abcdef0ull;
        auto next = [&seed]() {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            return seed;
        };
        uint64_t mismatches = 0, samples = 0;
        for (uint64_t B : {uint64_t(1) << 10, uint64_t(1) << 24, uint64_t(1) << 31}) {
            for (int i = 0; i < 20000; ++i) {
                auto pick = [&]() -> int64_t {
                    uint64_t r = next() % (2 * B + 1);
                    return static_cast<int64_t>(r) - static_cast<int64_t>(B);
                };
                const int64_t a1 = pick(), b1 = pick(), a2 = pick(), b2 = pick();
                const int s = (next() & 1) ? 1 : -1;
                const __int128 x = static_cast<__int128>(a1) * b1 +
                                   static_cast<__int128>(s) * (static_cast<__int128>(a2) * b2);
                const int exact = (x > 0) ? 1 : (x < 0 ? -1 : 0);
                int filtered;
                try {
                    filtered = hecke::sign_dot2(a1, b1, a2, b2, s, B);
                } catch (const std::exception&) {
                    ++mismatches; ++samples; continue;
                }
                if (filtered != exact) ++mismatches;
                ++samples;
            }
        }
        cov_.hit("obl:fp", "O13");
        cov_.hit("obl:oracle", "O13");
        if (mismatches != 0) {
            Fail("O13 FP-filter: " + std::to_string(mismatches) + "/" + std::to_string(samples) +
                 " filtered-vs-exact sign mismatches");
        } else {
            std::cout << "[O13] FP-filter OK: " << samples
                      << " samples, filtered sign == exact sign in all cases (B in {2^10,2^24,2^31})."
                      << std::endl;
        }
    }

    void PrintSummary() const {
        std::cout << "\n=============== PER-(ell,n) SUMMARY ===============\n";
        for (const auto& line : summary_) std::cout << line << "\n";
        std::cout << "==================================================\n";
        if (!dickson_recorded_.empty()) {
            std::cout << "Dickson exceptional loci RECORDED (not asserted): ";
            bool first = true;
            for (const auto& pr : dickson_recorded_) {
                if (!first) std::cout << ", ";
                std::cout << "(n=" << pr.first << ",ell=" << pr.second << ")";
                first = false;
            }
            std::cout << "\n";
        }
    }

private:
    const SimplicialComplex& K_;
    const hecke::SL2ZTessellationTables& model_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<CohomologyFixture>> coh_cache_;
    std::map<std::tuple<uint64_t, uint32_t, uint32_t>, std::unique_ptr<HeckeFixture>> hecke_cache_;
    Coverage cov_;
    uint32_t failures_ = 0;
    std::vector<std::string> summary_;
    std::vector<std::pair<uint32_t, uint32_t>> dickson_recorded_;

    void Fail(const std::string& msg) {
        ++failures_;
        std::cout << "  *** FAIL: " << msg << std::endl;
    }

    // ------------------------------------------------------------------
    // Dimension check (O14/O1/O2) -- asserted from the base cohomology fixture.
    // ------------------------------------------------------------------
    bool CheckDims(uint32_t n, uint32_t ell, uint32_t eh0, uint32_t eh1, uint32_t eh2,
                   const std::string& oracle) {
        try {
            CohomologyFixture& cf = Coh(n, ell);
            const auto& d = cf.coh.get_H_eq_dimensions();
            cov_.hit("obl:oracle", oracle);
            if (d.size() < 3) {
                Fail(oracle + " n=" + std::to_string(n) + " ell=" + std::to_string(ell) +
                     ": get_H_eq_dimensions() size " + std::to_string(d.size()) + " (< 3)");
                return false;
            }
            if (d[0] != eh0 || d[1] != eh1 || d[2] != eh2) {
                std::ostringstream oss;
                oss << oracle << " DIM MISMATCH n=" << n << " ell=" << ell << ": got ("
                    << d[0] << "," << d[1] << "," << d[2] << "), expected (" << eh0 << ","
                    << eh1 << "," << eh2 << ")";
                Fail(oss.str());
                return false;
            }
            return true;
        } catch (const std::exception& e) {
            Fail(oracle + " EXCEPTION building cohomology n=" + std::to_string(n) + " ell=" +
                 std::to_string(ell) + ": " + e.what());
            return false;
        }
    }

    // Records (prints) the dimension vector without asserting (Dickson loci).
    void RecordDims(uint32_t n, uint32_t ell) {
        try {
            CohomologyFixture& cf = Coh(n, ell);
            const auto& d = cf.coh.get_H_eq_dimensions();
            std::ostringstream oss;
            oss << "  [DICKSON-RECORD] n=" << n << " in <ell+1=" << (ell + 1) << ", ell*(ell-1)="
                << (static_cast<uint64_t>(ell) * (ell - 1)) << "> (extra SL_2(F_ell)-invariant) ell="
                << ell << ": (H^0,H^1,H^2)=(";
            for (size_t i = 0; i < 3; ++i) { if (i) oss << ","; oss << (i < d.size() ? d[i] : 0); }
            oss << ") -- RECORDED, not asserted";
            std::cout << oss.str() << std::endl;
        } catch (const std::exception& e) {
            Fail("Dickson record EXCEPTION n=" + std::to_string(n) + " ell=" + std::to_string(ell) +
                 ": " + e.what());
        }
    }

    // Assert A^(q) is 0x0.
    bool CheckEmpty(HeckeFixture* hf, uint32_t q, uint32_t n, uint32_t ell, uint64_t p,
                    const std::string& oracle) {
        if (!hf) return false;
        const auto& H = hf->op.get_H_matrices();
        if (q >= H.size()) { Fail(oracle + ": degree q out of range"); return false; }
        if (H[q].rows != 0 || H[q].cols != 0) {
            std::ostringstream oss;
            oss << oracle << " q=" << q << " n=" << n << " ell=" << ell << " p=" << p
                << ": expected 0x0, got " << H[q].rows << "x" << H[q].cols;
            Fail(oss.str());
            return false;
        }
        return true;
    }

    // Assert a 1x1 A^(q) equals `expected`.
    bool CheckScalar(HeckeFixture* hf, uint32_t q, uint32_t n, uint32_t ell, uint64_t p,
                     uint32_t expected, const std::string& oracle) {
        if (!hf) return false;
        const auto& H = hf->op.get_H_matrices();
        if (q >= H.size() || H[q].rows != 1 || H[q].cols != 1) {
            std::ostringstream oss;
            oss << oracle << " q=" << q << " n=" << n << " ell=" << ell << " p=" << p
                << ": expected 1x1, got "
                << (q < H.size() ? std::to_string(H[q].rows) + "x" + std::to_string(H[q].cols) : "<none>");
            Fail(oss.str());
            return false;
        }
        const uint32_t got = H[q].data[0][0];
        cov_.hit("obl:oracle", oracle);
        if (got != expected) {
            std::ostringstream oss;
            oss << oracle << " q=" << q << " n=" << n << " ell=" << ell << " p=" << p
                << ": A^(" << q << ")(T_" << p << ") = (" << got << "), expected (" << expected << ")";
            Fail(oss.str());
            return false;
        }
        return true;
    }

    // Compute charpoly of the degree-q matrix (empty for empty degrees).
    bool GetCharpoly(HeckeFixture* hf, uint32_t q, hecke::FpPoly& out, uint32_t& dim_out) {
        if (!hf) return false;
        const auto& H = hf->op.get_H_matrices();
        if (q >= H.size()) return false;
        dim_out = H[q].rows;
        if (H[q].rows == 0) { out = hecke::FpPoly(H[q].p); return true; }
        out = hecke::CharPolyBerkowitz(H[q]);
        return true;
    }

    // Assert charpoly(A^(q)) == expected.
    bool CheckCharpoly(HeckeFixture* hf, uint32_t q, uint32_t n, uint32_t ell, uint64_t p,
                       const hecke::FpPoly& expected, const std::string& oracle) {
        if (!hf) return false;
        hecke::FpPoly computed;
        uint32_t dim = 0;
        if (!GetCharpoly(hf, q, computed, dim)) { Fail(oracle + ": charpoly unavailable"); return false; }
        cov_.hit("obl:oracle", oracle);
        if (computed != expected) {
            std::ostringstream oss;
            oss << oracle << " CHARPOLY MISMATCH q=" << q << " n=" << n << " ell=" << ell << " p="
                << p << "\n      computed = " << computed.to_string()
                << "\n      expected = " << expected.to_string();
            Fail(oss.str());
            return false;
        }
        return true;
    }

    // Records (prints) a charpoly without asserting.
    void RecordCharpoly(HeckeFixture* hf, uint32_t q, uint32_t n, uint32_t ell, uint64_t p,
                        const std::string& note) {
        if (!hf) return;
        hecke::FpPoly computed;
        uint32_t dim = 0;
        if (!GetCharpoly(hf, q, computed, dim)) return;
        std::cout << "  [RECORD] " << note << " n=" << n << " ell=" << ell << " p=" << p
                  << " q=" << q << ": dim=" << dim << " charpoly(A) = " << computed.to_string()
                  << std::endl;
    }

    uint32_t EisMod(uint64_t p, uint32_t n, uint32_t ell) {
        // (1 + p^{n+1}) mod ell.
        const uint32_t pn1 = ModArithmetic::mod_pow(static_cast<uint32_t>(p % ell), n + 1u, ell);
        return ModArithmetic::mod_add(1u % ell, pn1, ell);
    }

    // ------------------------------------------------------------------
    // Which reliable a_p (weight n+2) is available for a hard charpoly assert?
    // Returns true and sets a_p iff (n, p) is in the auditor-verified set for
    // the single-cusp-form weights 12 (n=10) and 16 (n=14).
    // ------------------------------------------------------------------
    bool ReliableApSingle(uint32_t n, uint64_t p, int64_t& ap) {
        if (n == 10) { // weight 12: a_2..a_7 all verified
            switch (p) { case 2: ap = -24; return true; case 3: ap = 252; return true;
                         case 5: ap = 4830; return true; case 7: ap = -16744; return true; }
        }
        if (n == 14) { // weight 16: only a_2, a_3 verified
            switch (p) { case 2: ap = 216; return true; case 3: ap = -3348; return true; }
        }
        return false;
    }

    // Expected charpoly for a single-cusp-form weight: (X - a_p)^2 (X - Eis).
    hecke::FpPoly ExpectedSingleCusp(uint32_t n, uint64_t p, uint32_t ell, int64_t ap) {
        const hecke::FpPoly cusp = hecke::FpPoly::pow(XMinus(ell, ap), 2);
        const hecke::FpPoly eis = XMinus(ell, static_cast<int64_t>(EisMod(p, n, ell)));
        return cusp * eis;
    }

    // Expected charpoly for weight 24 (n=22), p=2: (X^2-1080X-20468736)^2 (X-Eis).
    hecke::FpPoly ExpectedWeight24T2(uint32_t ell) {
        const hecke::FpPoly quad(ell, std::vector<uint32_t>{
            ModOfInt64(-20468736, ell), ModOfInt64(-1080, ell), 1u});
        const hecke::FpPoly cusp = quad * quad;
        const hecke::FpPoly eis = XMinus(ell, static_cast<int64_t>(EisMod(2, 22, ell)));
        return cusp * eis;
    }

    // ------------------------------------------------------------------
    // One (n,ell) cell of the full matrix.
    // ------------------------------------------------------------------
    void RunCell(uint32_t n, uint32_t ell) {
        std::ostringstream sm;
        sm << "  n=" << n << " ell=" << ell << ": ";

        // Dickson exceptional locus (A-F4 refinement): RECORD, do not assert.
        if (hecke::IsDicksonExceptional(n, ell)) {
            RecordDims(n, ell);
            dickson_recorded_.emplace_back(n, ell);
            HeckeFixture* hf = Hecke(2, ell, n); // still exercise the pipeline (obl:welldef)
            RecordCharpoly(hf, 1, n, ell, 2, "Dickson H^1");
            sm << "DICKSON-RECORDED";
            summary_.push_back(sm.str());
            return;
        }

        const uint32_t before = failures_;

        if (n % 2 == 1) {
            // O2: odd n => H^* = 0 (ell != 2), post-FIX-negI.
            const bool dims_ok = CheckDims(n, ell, 0, 0, 0, "O2");
            HeckeFixture* hf = Hecke(2, ell, n);
            bool empty_ok = true;
            if (hf) {
                for (uint32_t q = 0; q < 3; ++q) empty_ok &= CheckEmpty(hf, q, n, ell, 2, "O2");
                cov_.hit("obl:oracle", "O2");
            } else {
                empty_ok = false;
            }
            sm << "O2 odd-n H^*=0 " << ((dims_ok && empty_ok) ? "ASSERTED-OK" : "FAILED");
            summary_.push_back(sm.str());
            return;
        }

        // Even n: dimensions (O14 / O1).
        const uint32_t eh0 = (n == 0) ? 1u : 0u;
        const uint32_t eh1 = ExpectedDimH1(n);
        const bool dims_ok = CheckDims(n, ell, eh0, eh1, /*h2=*/0u, (n == 0) ? "O1" : "O14");

        // Which primes to build (see the S12 report for the coverage/runtime
        // trade-off: full p-grid on the ASSERTION weights; p=2 on the
        // record-only large-n weights, whose prime-agnostic pipeline is already
        // exercised at all four primes by the smaller weights).
        std::vector<uint64_t> primes;
        if (n == 0 || n == 2 || n == 8 || n == 10 || n == 14) primes = {2, 3, 5, 7};
        else if (n == 22) primes = {2, 3};
        else primes = {2}; // n in {16,18,20,24}: record-only

        std::string cell_note = "dims " + std::string(dims_ok ? "OK" : "FAIL");
        bool any_assert = false, any_record = false;

        for (uint64_t p : primes) {
            HeckeFixture* hf = Hecke(p, ell, n);
            if (!hf) continue;

            // Degrees with no cohomology are always 0x0.
            if (n == 0) {
                CheckScalar(hf, 0, n, ell, p, static_cast<uint32_t>((p + 1) % ell), "O1");
                CheckEmpty(hf, 1, n, ell, p, "O1");
                CheckEmpty(hf, 2, n, ell, p, "O1");
                cov_.hit("obl:integral", "O1"); // A^(0)(T_p) = p+1 is the integral normalization
                any_assert = true;
                continue;
            }

            CheckEmpty(hf, 0, n, ell, p, "dimq0");
            CheckEmpty(hf, 2, n, ell, p, "dimq2");

            // Degree 1 -- the arithmetic heart.
            if (eh1 == 1) {
                // Pure-Eisenstein weight (n=2 weight 4, n=8 weight 10): the
                // 1x1 T_p eigenvalue is the Eisenstein eigenvalue 1 + p^{n+1}.
                CheckScalar(hf, 1, n, ell, p, EisMod(p, n, ell), (n == 2) ? "O3" : "O4");
                cov_.hit("obl:integral", "Eisenstein-eigenvalue");
                any_assert = true;
                continue;
            }

            int64_t ap = 0;
            if (ReliableApSingle(n, p, ap)) {
                const hecke::FpPoly expected = ExpectedSingleCusp(n, p, ell, ap);
                const char* oracle = (n == 10) ? "O5" : "O7";
                CheckCharpoly(hf, 1, n, ell, p, expected, oracle);
                any_assert = true;
                // O6: the Ramanujan-congruence non-semisimplicity at ell=691.
                if (n == 10 && ell == 691 && p == 2) CheckO6NonSemisimple(hf);
            } else if (n == 22 && p == 2) {
                CheckCharpoly(hf, 1, n, ell, p, ExpectedWeight24T2(ell), "O8");
                any_assert = true;
            } else {
                // Record-only (weights 18/20/22/26; weight-16/24 non-verified primes).
                RecordCharpoly(hf, 1, n, ell, p, "cusp-weight (a_p not in auditor-verified set)");
                any_record = true;
            }
        }

        // O12(a) explicit route bit-identity spot-check on one fixture per cell
        // (beyond the compute()-internal assertion) -- an executed obl:agreement
        // check with a visible verdict.
        if (n == 10 && (ell == 13 || ell == 101)) RunRouteBitIdentity(n, ell, 2);

        if (any_assert) cell_note += "; charpoly/eigenvalue ASSERTED";
        if (any_record) cell_note += "; charpoly RECORDED";
        sm << cell_note;
        summary_.push_back(sm.str());
        (void)before;
    }

    // O6 hard part: A^(1)(T_2) != -24*I mod 691 (non-split Eisenstein/cuspidal
    // extension; Deng). The (X+24)^3 charpoly itself is checked by the generic
    // O5 path (the general (X-a_p)^2(X-Eis) formula collapses to (X+24)^3 at
    // ell=691 via the Ramanujan congruence 2049 = -24 (mod 691)).
    void CheckO6NonSemisimple(HeckeFixture* hf) {
        const auto& H = hf->op.get_H_matrices();
        if (H.size() < 2 || H[1].rows != 3) { Fail("O6: A^(1)(T_2) wrong shape at ell=691 n=10"); return; }
        cov_.hit("obl:oracle", "O6");
        if ((2049 - (-24)) % 691 != 0) { Fail("O6: 2049+24 is not a multiple of 691 (internal)"); return; }
        const Matrix& A = H[1];
        Matrix diff = A - Matrix(A.rows, 691u) * ModOfInt64(-24, 691u); // A + 24 I
        if (diff.is_zero()) {
            Fail("O6 HARD: A^(1)(T_2) == -24*I mod 691 (extension must be non-split)");
        } else {
            const uint32_t rk = diff.rank();
            std::cout << "[O6] HARD OK: A^(1)(T_2) != -24*I mod 691; rank(A+24I) = " << rk
                      << " (partition recorded; assertion of (2,1) is S10V's A-F2 job)." << std::endl;
        }
    }

    // Explicit apply_route2 vs apply_Theta bit-identity on the standard basis
    // (obl:agreement, O12(a)).
    void RunRouteBitIdentity(uint32_t n, uint32_t ell, uint64_t p) {
        HeckeFixture* hf = Hecke(p, ell, n);
        if (!hf) return;
        try {
            cov_.hit("obl:agreement", "O12a-explicit");
            cov_.hit("obl:oracle", "O12a");
            for (uint32_t q = 0; q < 3; ++q) {
                const auto& flat = hf->op.get_Theta_flat_matrices();
                if (q >= flat.size()) continue;
                const uint32_t dim = flat[q].get_rows(); // (N_q * d_V)
                if (dim == 0) continue;
                // Test a few standard basis columns.
                const uint32_t step = (dim > 8) ? (dim / 8) : 1;
                for (uint32_t col = 0; col < dim; col += step) {
                    FlatMatrix e(dim, 1u, ell);
                    e.set_entry(col, 0, 1u);
                    const FlatMatrix r1 = hf->op.apply_Theta(q, e);
                    const FlatMatrix r2 = hf->op.apply_route2(q, e);
                    if (!(r1 == r2)) {
                        Fail("O12a explicit: apply_Theta != apply_route2 at n=" + std::to_string(n) +
                             " ell=" + std::to_string(ell) + " q=" + std::to_string(q) + " col=" +
                             std::to_string(col));
                        return;
                    }
                }
            }
            std::cout << "[O12a] route bit-identity OK (apply_Theta == apply_route2) at n=" << n
                      << " ell=" << ell << " p=" << p << std::endl;
        } catch (const std::exception& e) {
            Fail(std::string("O12a explicit: EXCEPTION: ") + e.what());
        }
    }
};

} // namespace

// ===========================================================================
// 1) HeckeEndToEndValidation -- the full oracle matrix.
// ===========================================================================

uint32_t HeckeEndToEndValidation() {
    try {
        std::cout << "================= HeckeEndToEndValidation (S12 acceptance gate) =================\n";
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        E2EContext ctx(K, model);
        ctx.RunMainMatrix();   // O1-O8, O14, O2 (odd-n), Dickson records
        ctx.RunAlgebra();      // O9  (obl:algebra, obl:integral)
        ctx.RunShapiro();      // O11 (obl:shapiro)
        ctx.RunFpFilter();     // O13 (obl:fp)

        ctx.PrintSummary();
        ctx.coverage().print("HeckeEndToEndValidation");

        const uint32_t fails = ctx.failures();
        std::cout << "\nHeckeEndToEndValidation: aggregate failing case(s) = " << fails << "\n";
        std::cout << "  (coverage: every obligation exercised by >= 1 executed check: "
                  << (ctx.coverage().all_covered() ? "YES" : "NO") << ")\n";
        std::cout << "  The two formerly-known pre-existing, non-Hecke defects "
                     "(BlockFlatMatrixValidation hang; SimplicialCohomologyValidation crash) "
                     "are fixed; --all now exits 0.\n";
        // A coverage gap is itself a failure of the acceptance gate.
        return fails + (ctx.coverage().all_covered() ? 0u : 1u);
    } catch (const std::exception& e) {
        std::cout << "HeckeEndToEndValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}

// ===========================================================================
// 2) HeckeBadPrimeValidation -- O10 (p = ell in {2,5}), A-F1 gating.
// ===========================================================================

namespace {

struct BadPrimeStack {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;
    std::unique_ptr<hecke::HeckeOperatorFamily> family;

    BadPrimeStack(uint32_t ell, uint32_t n_sym)
        : K(CreateSL2ZFundamentalDomainTriangulation()),
          model(K),
          lssc(K, BuildRelations(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc) {
        coh.compute_cohomology();
        family = std::make_unique<hecke::HeckeOperatorFamily>(&coh, K, model, ell, n_sym);
    }
};

// p = ell = 5 (>= 5, contract-backed): HARD assertions.
uint32_t BadPrimeGood(uint32_t n_sym) {
    const uint32_t ell = 5;
    BadPrimeStack st(ell, n_sym);
    auto& fam = *st.family;
    const auto& dims = st.coh.get_H_eq_dimensions();
    const uint32_t num_q = static_cast<uint32_t>(dims.size());

    const std::vector<Matrix>& U5 = fam.T_p(5);
    const std::vector<Matrix> U25_direct = fam.T_p_pow_direct(5, 2); // independent N=25 pipeline
    uint32_t fails = 0;
    for (uint32_t q = 0; q < num_q; ++q) {
        const Matrix U5sq = U5[q] * U5[q];
        if (U25_direct[q] != U5sq) {
            std::cout << "  *** FAIL O10(ell=5,n=" << n_sym << ",q=" << q
                      << "): U_25 (independent) != U_5^2." << std::endl;
            ++fails;
        }
    }
    // rank(U_5^r) non-increasing and stationary by r = d_q.
    uint32_t max_d = 0; for (uint32_t d : dims) max_d = std::max(max_d, d);
    const uint32_t R = max_d + 3;
    std::vector<std::vector<uint32_t>> ranks(num_q);
    for (uint32_t r = 1; r <= R; ++r) {
        const std::vector<Matrix> Ur = fam.T_p_pow(5, r);
        for (uint32_t q = 0; q < num_q; ++q) { Matrix c = Ur[q]; ranks[q].push_back(c.rank()); }
    }
    for (uint32_t q = 0; q < num_q; ++q) {
        for (size_t i = 1; i < ranks[q].size(); ++i)
            if (ranks[q][i] > ranks[q][i - 1]) { std::cout << "  *** FAIL O10(ell=5): rank(U_5^r) not non-increasing at q=" << q << std::endl; ++fails; break; }
        const uint32_t d = dims[q];
        if (d >= 1) {
            const uint32_t stable = ranks[q][d - 1];
            for (size_t i = d - 1; i < ranks[q].size(); ++i)
                if (ranks[q][i] != stable) { std::cout << "  *** FAIL O10(ell=5): rank(U_5^r) not stationary by r=d_q at q=" << q << std::endl; ++fails; break; }
        }
    }
    if (fails == 0)
        std::cout << "  [O10 p=ell=5 n=" << n_sym << "] HARD OK: U_25=U_5^2 (independent), "
                     "rank(U_5^r) non-increasing & stationary by r=d_q." << std::endl;
    return fails;
}

// p = ell = 2 (< 5, A-F1): assert only completion/well-typedness; RECORD the rest.
uint32_t BadPrimeGated(uint32_t n_sym) {
    const uint32_t ell = 2;
    try {
        BadPrimeStack st(ell, n_sym);
        auto& fam = *st.family;
        const auto& dims = st.coh.get_H_eq_dimensions();
        const uint32_t num_q = static_cast<uint32_t>(dims.size());
        const std::vector<Matrix>& U = fam.T_p(2);
        const std::vector<Matrix> U4_direct = fam.T_p_pow_direct(2, 2);
        // HARD: well-typedness only.
        for (uint32_t q = 0; q < num_q; ++q) {
            if (U[q].rows != dims[q] || U[q].cols != dims[q] || U[q].p != ell ||
                U4_direct[q].rows != dims[q]) {
                std::cout << "  *** FAIL O10(ell=2,n=" << n_sym << ",q=" << q
                          << "): ill-typed U_ell matrix." << std::endl;
                return 1;
            }
        }
        // RECORD (not asserted): U_4 vs U_2^2 and rank(U_2^r) monotonicity.
        std::cout << "  [O10 p=ell=2 n=" << n_sym << "] A-F1 RECORD ONLY (not asserted):" << std::endl;
        uint32_t max_d = 0; for (uint32_t d : dims) max_d = std::max(max_d, d);
        const uint32_t R = max_d + 3;
        for (uint32_t q = 0; q < num_q; ++q) {
            const Matrix Usq = U[q] * U[q];
            const bool eq = (U4_direct[q] == Usq);
            std::vector<uint32_t> ranks;
            for (uint32_t r = 1; r <= R; ++r) { Matrix Ur = fam.T_p_pow(2, r)[q]; ranks.push_back(Ur.rank()); }
            bool mono = true;
            for (size_t i = 1; i < ranks.size(); ++i) if (ranks[i] > ranks[i - 1]) mono = false;
            std::cout << "    q=" << q << " (dim=" << dims[q] << "): U_4(direct)==U_2^2 ? "
                      << (eq ? "true" : "false") << "; rank(U_2^r) r=1.." << R << " = [";
            for (size_t i = 0; i < ranks.size(); ++i) { if (i) std::cout << ","; std::cout << ranks[i]; }
            std::cout << "]; non-increasing? " << (mono ? "true" : "false") << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cout << "  *** FAIL O10(ell=2,n=" << n_sym
                  << "): pipeline did not complete (A-F1 still requires completion): "
                  << e.what() << std::endl;
        return 1;
    }
}

} // namespace

uint32_t HeckeBadPrimeValidation() {
    try {
        std::cout << "================= HeckeBadPrimeValidation (O10, p=ell in {2,5}) =================\n";
        uint32_t fails = 0;
        // p = ell = 5 (>= 5): hard asserts (A-F1 contract-backed).
        for (uint32_t n : {0u, 2u, 10u}) fails += BadPrimeGood(n);
        // p = ell = 2 (< 5): completion/well-typedness asserted; rest recorded.
        for (uint32_t n : {0u, 2u, 10u}) fails += BadPrimeGated(n);
        std::cout << "HeckeBadPrimeValidation: aggregate failing case(s) = " << fails << "\n";
        return fails;
    } catch (const std::exception& e) {
        std::cout << "HeckeBadPrimeValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}

// ===========================================================================
// 3) HeckeBenchmarkSmoke -- (ell,n) = (134217757, 100), p = 2.
// ===========================================================================

uint32_t HeckeBenchmarkSmoke() {
    try {
        std::cout << "================= HeckeBenchmarkSmoke (ell=134217757, n=100, p=2) =================\n";
        const uint32_t ell = 134217757u;
        const uint32_t n = 100u;

        const auto t0 = std::chrono::steady_clock::now();

        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        const hecke::SL2ZTessellationTables model(K);

        // Base cohomology L = Sym^100(F_ell^2).
        LocalSystemSimplicialComplex lssc(K, BuildRelations(K, n, ell), n + 1, ell);
        EquivariantCohomologyOnK coh(&lssc);
        coh.compute_cohomology();
        const auto& dims = coh.get_H_eq_dimensions();
        const uint32_t d1 = dims.size() > 1 ? dims[1] : 0;
        const auto t_coh = std::chrono::steady_clock::now();

        // Full Hecke pipeline for p=2 (all runtime certs run inside compute()).
        hecke::HeckeCosetTables cosets(/*p=*/2, /*r=*/1, ell, n);
        hecke::HeckeTauTable tau(K, model, cosets, ell, n); // ctor certifies (obl:tau)
        hecke::HeckeOperatorOnK op(&coh, &cosets, &tau);
        op.compute();
        const Matrix& A1 = op.get_H_matrices()[1]; // A^{(1)}(T_2), d1 x d1
        const auto t_op = std::chrono::steady_clock::now();

        // Single-operator block decomposition + certify().
        std::vector<std::pair<uint64_t, Matrix>> ops = {{2ull, A1}};
        hecke::HeckeModuleDecomposition decomp(std::move(ops), ell);
        decomp.compute();
        const uint32_t cert = decomp.certify();
        const auto t_dec = std::chrono::steady_clock::now();

        auto secs = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
        };
        const double total = secs(t0, t_dec);

        std::cout << "  dim H^1 = " << d1 << " (expected 17; d_V = " << (n + 1) << ")\n";
        std::cout << "  A^{(1)}(T_2) is " << A1.rows << "x" << A1.cols << " (level-(a) Theta_flat NOT dumped; n=100 size-gated)\n";
        std::cout << "  block decomposition: " << decomp.blocks().size() << " primary block(s); certify() = "
                  << cert << (cert == 0 ? " (CERTIFIED)" : " (FAILED)") << "\n";
        std::cout << "  --- WALL TIME (baseline for Q8) ---\n";
        std::cout << "    cohomology (Sym^100): " << secs(t0, t_coh) << " s\n";
        std::cout << "    Hecke pipeline + compute(): " << secs(t_coh, t_op) << " s\n";
        std::cout << "    block decomposition + certify(): " << secs(t_op, t_dec) << " s\n";
        std::cout << "    TOTAL: " << total << " s\n";

        uint32_t fails = 0;
        if (d1 != 17u) {
            std::cout << "  *** FAIL: dim H^1 = " << d1 << " (expected 17)." << std::endl;
            ++fails;
        }
        if (cert != 0u) {
            std::cout << "  *** FAIL: block decomposition certify() = " << cert << " (expected 0)." << std::endl;
            ++fails;
        }
        std::cout << "HeckeBenchmarkSmoke: " << (fails == 0 ? "COMPLETED and CERTIFIED" : "FAILED")
                  << " (aggregate failing case(s) = " << fails << ")\n";
        return fails;
    } catch (const std::exception& e) {
        std::cout << "HeckeBenchmarkSmoke: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
