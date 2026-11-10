// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 16, WP-5 -- HeckeSaturationDefectValidation (standard).
//
// HeckeExt/HeckeSaturationDefect against the predictions of
// docs/implementation/slot16/WP-5_tor_note.md Sec. 4 (committed before the code)
// and against itself across routes.
//
//   (P) planted modules over Z/ell^K with a known saturation:
//       (P1) x^2 = 7 on Z_7^2 (L ~= T free of rank one): I = 0, saturated;
//       (P2) x = 7 sigma, sigma^2 = 1: T^sat / T = Z/7, I = m, principal, Loewy 1;
//       (P3) t = diag(0, 5, 25) ([C, cor:stablecert](2)'s suborder): T / 5 =
//            F_5[t] / t^3, I = m = (t), principal, layers [1,1], soc 1, Ann 1;
//            the filtration eps = [2,3,4,4] gives T^sat / T = Z/125 (+) Z/5 (the
//            Smith form of the Vandermonde lattice), stable from K = 4 = c^sat + 1;
//       (P4) the refusals: a torsion module (exponents {3, 1}) and a system at
//            which A / ell A is not local throw; SaturationStructure refuses a
//            non-filtration.
//   (A) the note's predictions at (82, 7), (142, 7): (142, 7) (5,6,4,0) the [M1]
//       hidden datum: I = m, dim 2, principal, Loewy 2; (142, 7) Delta: I = m^2,
//       dim 2, not principal, Loewy 1, soc I = I, Ann I = m; (82, 7) Delta: dim 1,
//       principal, I in soc, Ann I = m.
//   (B) every block of the default cells: the Tor identity's two routes (R-ker
//       dim I == R-len eps_1 = rank_T - dim A_1); (A / ell) / I has A_1's Hilbert
//       function (the isomorphism (T / ell) / I ~= A_1); dim A_K / ell == rank_T at
//       K_stab; the image has dim A_1; I is an ideal inside m; reduced rank-1
//       blocks give I = 0; the hidden blocks (A_1 = F, rank_T >= 2) give I = m;
//       the filtration is final with dim I summands; the forms cross-route over
//       F_ell: joint == Tc (A_1 a quotient of T^f_m / ell, [C, prop:cohforms](5)),
//       dim T^f_m / ell == dim M_k(F_ell)_m == rank_T, kernel <= dim I, == dim I
//       when the S-primes generate.
//   (C) the route agrees with ComputeRingLoewyCell (verdict, K_stab, A_1 and A_K
//       Hilbert functions, eps) at (82, 7).
//   (D) torsion blocks are refused by name at (18, 5) (a Dickson-locus cell).
//   (ENV) HECKE_SAT_DEFECT_CELLS="n:ell[:KMAX],..." (read-only, not an emitter;
//       KMAX defaults to 6, capped so that ell^KMAX < 2^64): the cells listed
//       instead, one SAT_DEFECT line per block, one SAT_DEFECT_CELL line per cell,
//       one EFFECTIVE line; gated as in (B).
//
//   (slot-16 WP-5 audit repairs, docs/implementation/slot16/WP-3_WP-5_audit.md)
//   (H) DEFAULT: the hidden-chain law L-F1 at its cheapest instance, (168, 13) (j = 1, KMAX = 2, ~13 s):
//       the block at a_p = p tau(p) mod 13 has I = m, dim 1, principal, Loewy 1, Ann I = m, T^sat / T =
//       Z/13, and every other ok block I = 0.  The note's own cells are HOOK-ONLY (28-124 s each):
//       listing (220, 17), (238, 17), (246, 19) or (258 | 294 | 296 | 334, 37) in HECKE_SAT_DEFECT_CELLS
//       gates the same hidden-chain expectation, resp. I = 0 on every block with a saturated Kummer block
//       present -- one SAT_DEFECT_EXPECT line per such cell, expectations=<held>/<k> on EFFECTIVE.
//   (ENV) a set-but-empty variable (or one with no valid cell) fails; every field is range-checked and ell
//       must be prime, a bad cell refused by its position; refused blocks are SAT_DEFECT_REFUSED lines,
//       counted on the EFFECTIVE line (now printed after the run); a cell with no ok block fails unless
//       every block is a torsion refusal; HECKE_SAT_DEFECT_STRICT=1 also fails not_stable_at_KMAX.
//   The forms cross-route runs at k = n + 2 >= 12 only (below it, no forms gate).
// ===========================================================================

#include "HeckeSaturationDefectValidation.h"

#include "HeckeSaturationDefect.h"
#include "HeckeFormsHeckeAlgebra.h"   // forms::kFormsMaxEllExclusive (the hook's pre-check, slot-24 ELL-W audit F6)
#include "HeckeRingHeckeAlgebra.h"
#include "../Hecke/CoefficientRing.h"   // hecke::IsPrimeU64 (the hook's ell check)

#include "../../ModArithmetic/EngineCheck.h"
#include "../Util/ValidationCheck.h"
#include "../Util/CellList.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace hecke_ext::satdefect;
using hecke_ext::ringalg::OpModule;

oc_util::Checker Check("  FAIL: ", false);

using H = std::vector<uint32_t>;
std::string Txt(const std::vector<uint32_t>& v) {
    std::ostringstream s;
    s << "[";
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    return s.str() + "]";
}
std::string S(uint64_t a, uint64_t b) { return "(" + std::to_string(a) + ", " + std::to_string(b) + ")"; }

uint64_t P64(uint64_t b, uint32_t e) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < e; ++i) r *= b;
    return r;
}

// (Z/ell^K)^D with the given integer matrices (row-major, entries reduced mod ell^K).
OpModule Planted(uint64_t ell, uint32_t K, uint32_t D, const std::vector<std::vector<int64_t>>& mats) {
    OpModule M;
    M.ell = ell;
    M.exps.assign(D, K);
    const uint64_t q = P64(ell, K);
    for (size_t i = 0; i < mats.size(); ++i) {
        hecke_ext::ringalg::RingMat A;
        A.ell = ell;
        A.row_exp = M.exps;
        for (int64_t x : mats[i]) {
            const int64_t r = x % static_cast<int64_t>(q);
            A.a.push_back(static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(q) : r));
        }
        M.op.push_back(A);
        M.primes.push_back(2 + i);
    }
    return M;
}

// eps_j = j r - length(A_j) for j = 1..Kmax on a planted free module (A_1 from the reduction).
std::vector<uint32_t> PlantedEps(uint64_t ell, uint32_t Kmax, uint32_t D, uint32_t r, const std::vector<std::vector<int64_t>>& mats,
                                 const std::vector<uint64_t>& sys) {
    std::vector<uint32_t> eps;
    eps.push_back(r - hecke_ext::ringalg::ReadAlgebra(hecke_ext::ringalg::ReduceModEll(Planted(ell, 2, D, mats)), sys).length);
    for (uint32_t K = 2; K <= Kmax; ++K)
        eps.push_back(K * r - hecke_ext::ringalg::ReadAlgebra(Planted(ell, K, D, mats), sys).length);
    return eps;
}

void ArmP() {
    {   // (P1) x^2 = 7
        const DefectReading R = ReadDefect(Planted(7, 3, 2, {{0, 7, 1, 0}}), {0});
        Check(R.dim_A == 2 && R.hilbert == H({1, 1}) && R.dim_I == 0 && R.I_loewy == 0 && R.generators.empty(),
              "(P1) x^2 = 7: A / 7 = F_7[x]/x^2, I = 0 (L ~= T, saturated); got dim_A " + std::to_string(R.dim_A) + " dim_I " +
                  std::to_string(R.dim_I));
        Check(R.quotient_hilbert == H({1, 1}) && R.dim_image == 2, "(P1) (A/7)/I = A_1 = (1,1)");
        Check(PlantedEps(7, 3, 2, 2, {{0, 7, 1, 0}}, {0}) == H({0, 0, 0}), "(P1) eps = [0,0,0]");
    }
    {   // (P2) x = 7 sigma
        const std::vector<std::vector<int64_t>> m = {{0, 7, 7, 0}};
        const DefectReading R = ReadDefect(Planted(7, 3, 2, m), {0});
        Check(R.dim_A == 2 && R.A_exponents == H({3, 2}) && R.length_A == 5, "(P2) A_3 = Z/343 (+) Z/49, got " + Txt(R.A_exponents));
        Check(R.dim_I == 1 && R.I_is_m && R.principal && R.I_layers == H({1}) && R.I_loewy == 1 && R.I_socle == 1 && R.I_ann == 1,
              "(P2) I = m, dim 1, principal, Loewy 1, soc 1, Ann = m (dim 1)");
        Check(R.generators.size() == 1 && R.generators[0] == "T2", "(P2) I = (T2), got " + (R.generators.empty() ? std::string("-") : R.generators[0]));
        Check(R.quotient_hilbert == H({1}), "(P2) (A/7)/I = F_7 = A_1");
        const std::vector<uint32_t> eps = PlantedEps(7, 3, 2, 2, m, {0});
        Check(eps == H({1, 1, 1}), "(P2) eps = [1,1,1], got " + Txt(eps));
        Check(SaturationStructure({1}) == H({1}), "(P2) T^sat / T = Z/7");
    }
    {   // (P3) t = diag(0, 5, 25)
        const std::vector<std::vector<int64_t>> m = {{0, 0, 0, 0, 5, 0, 0, 0, 25}};
        const std::vector<uint32_t> eps = PlantedEps(5, 6, 3, 3, m, {0});
        Check(eps == H({2, 3, 4, 4, 4, 4}), "(P3) eps = [2,3,4,4,4,4], got " + Txt(eps));
        Check(SaturationStructure({2, 3, 4}) == H({3, 1}), "(P3) T^sat / T = Z/125 (+) Z/5 (Vandermonde Smith form)");
        const DefectReading R3 = ReadDefect(Planted(5, 3, 3, m), {0});
        Check(R3.dim_A == 2, "(P3) not stable at K = 3 (dim A_3 / 5 = 2 < 3 = r)");
        const DefectReading R = ReadDefect(Planted(5, 4, 3, m), {0});
        Check(R.dim_A == 3 && R.hilbert == H({1, 1, 1}), "(P3) stable at K = 4 = c + 1: A_4 / 5 = F_5[t]/t^3");
        Check(R.dim_I == 2 && R.I_is_m && R.principal && R.I_layers == H({1, 1}) && R.I_loewy == 2 && R.I_socle == 1 && R.I_ann == 1,
              "(P3) I = m = (t): dim 2, principal, layers [1,1], soc 1, Ann = m^2 (dim 1); got dim " + std::to_string(R.dim_I) + " layers " +
                  Txt(R.I_layers) + " soc " + std::to_string(R.I_socle) + " ann " + std::to_string(R.I_ann));
        Check(R.generators.size() == 1 && R.generators[0] == "T2", "(P3) I = (T2)");
        Check(R.dim_I == eps[0], "(P3) R-ker dim I == R-len eps_1");
        const DefectReading R6 = ReadDefect(Planted(5, 6, 3, m), {0});
        Check(R6.hilbert == R.hilbert && R6.dim_I == R.dim_I && R6.I_layers == R.I_layers, "(P3) the reading is the same at K = 6 (T / 5)");
    }
    {   // (P4) refusals
        bool threw = false;
        try { (void)ReadDefect([] { OpModule M = Planted(7, 3, 2, {{0, 0, 0, 0}}); M.exps = {3, 1}; for (auto& a : M.op) a.row_exp = M.exps; return M; }(), {0}); }
        catch (const std::logic_error&) { threw = true; }
        Check(threw, "(P4) a torsion module is refused (ENGINE_CHECK)");
        threw = false;
        try { (void)ReadDefect(Planted(7, 3, 2, {{0, 7, 7, 0}}), {1}); } catch (const std::logic_error&) { threw = true; }
        Check(threw, "(P4) a system at which A / ell A is not local is refused");
        threw = false;
        try { (void)SaturationStructure({2, 1}); } catch (const std::logic_error&) { threw = true; }
        Check(threw, "(P4) a decreasing sequence is not a filtration");
        threw = false;
        try { (void)SaturationStructure({1, 3}); } catch (const std::logic_error&) { threw = true; }
        Check(threw, "(P4) [1,3] is not the filtration of a finite Z_ell-module (more summands of order >= ell^2 than of order >= ell)");
        threw = false;   // (slot-16 WP-5 audit repair, finding 3)
        try { (void)ComputeSaturationDefectCell(10, 9, SatOptions()); } catch (const std::logic_error&) { threw = true; }
        Check(threw, "(P4) a composite ell (9) is refused by name");
    }
}

// ---- the per-block gates of (B) and (ENV) ------------------------------------------
struct Tally {
    uint32_t blocks = 0, ok = 0, torsion = 0, not_stable = 0, nonzero_I = 0, hidden = 0, forms_eq = 0, forms_le = 0;
};

void GateBlock(const SatCell& c, const SatBlock& b, const std::string& arm, Tally& t) {
    const std::string tag = arm + " " + S(c.n, c.ell) + " system=" + Txt(std::vector<uint32_t>(b.system.begin(), b.system.end()));
    ++t.blocks;
    if (b.status == "refused:torsion_block") { ++t.torsion; Check(!b.torsion.empty(), tag + ": refused as torsion only with torsion"); return; }
    if (b.status == "refused:not_stable_at_KMAX") { ++t.not_stable; return; }
    Check(b.status == "ok", tag + ": status ok");
    ++t.ok;
    const DefectReading& r = b.at;
    Check(b.stable_cert == "certified", tag + ": the stable verdict is certified ([C, cor:stablecert])");
    Check(r.dim_A == b.rank_T, tag + ": dim A_K / ell == rank_T at K_stab");
    Check(r.dim_I == b.eps1_len, tag + ": the Tor identity: R-ker dim I == R-len eps_1 (" + std::to_string(r.dim_I) + " vs " +
                                     std::to_string(b.eps1_len) + ")");
    Check(r.dim_image == b.dim_A1, tag + ": the image in End(B / ell) has dim A_1");
    Check(r.quotient_hilbert == b.A1_hilbert, tag + ": (T / ell) / I has A_1's Hilbert function " + Txt(b.A1_hilbert) + " (got " +
                                                   Txt(r.quotient_hilbert) + ")");
    Check(r.I_in_m, tag + ": I is inside m");
    Check(r.generators.size() == r.min_generators, tag + ": one polynomial per minimal generator");
    if (b.rank_T == 1) Check(r.dim_I == 0, tag + ": rank 1: T = Z_ell is saturated, I = 0");
    if (b.dim_A1 == 1 && b.rank_T >= 2) {
        ++t.hidden;
        Check(r.I_is_m, tag + ": a hidden block (A_1 = F, rank_T >= 2): I = m");
    }
    if (r.dim_I) ++t.nonzero_I;
    if (b.sat_final) {
        uint32_t summands = static_cast<uint32_t>(b.sat_structure.size()), len = 0;
        for (uint32_t e : b.sat_structure) len += e;
        Check(summands == r.dim_I && len == b.sat_length, tag + ": T^sat / T has dim I cyclic summands and length delta");
    }
    if (b.forms.ran) {
        const FormsCross& f = b.forms;
        Check(f.M_gen, tag + ": forms: the generator primes give T / ell of M_k ([E] cl:mkfaithful)");
        Check(f.dim_V == b.rank_T && f.T_dim == b.rank_T, tag + ": forms: dim M_k(F_ell)_m == dim T^f_m / ell == rank_T (got " +
                                                                std::to_string(f.dim_V) + ", " + std::to_string(f.T_dim) + ")");
        Check(f.joint == f.Tc, tag + ": forms: A_1 is a quotient of the S-algebra on M_k(F_ell)_m (joint == Tc)");
        Check(f.kernel <= r.dim_I, tag + ": forms: dim ker(J -> A_1) <= dim I");
        if (f.S_gen) {
            Check(f.kernel == r.dim_I, tag + ": forms: S generates T^f_m / ell, so dim ker(J -> A_1) == dim I (" + std::to_string(f.kernel) +
                                           " vs " + std::to_string(r.dim_I) + ")");
            ++t.forms_eq;
        } else {
            ++t.forms_le;
        }
    }
}

std::map<std::pair<uint32_t, uint64_t>, SatCell> g_cells;
const SatCell& Cell(uint32_t n, uint64_t ell, uint32_t kmax) {
    const auto key = std::make_pair(n, ell);
    auto it = g_cells.find(key);
    if (it != g_cells.end()) return it->second;
    SatOptions o;
    o.kmax = kmax;
    SatCell c = ComputeSaturationDefectCell(n, ell, o);
    for (const SatBlock& b : c.blocks) std::cout << SatDefectLine(c, b) << "\n";
    return g_cells.emplace(key, std::move(c)).first->second;
}

const SatBlock* Find(const SatCell& c, const std::vector<uint64_t>& sys) {
    for (const SatBlock& b : c.blocks)
        if (b.system == sys) return &b;
    return nullptr;
}

void ArmA() {
    {
        const SatCell& c = Cell(142, 7, 5);
        const SatBlock* h = Find(c, {5, 6, 4, 0});
        Check(h != nullptr, "(A) (142, 7) (5,6,4,0) present");
        if (h) {
            const DefectReading& r = h->at;
            Check(h->status == "ok" && h->A1_hilbert == H({1}) && r.hilbert == H({1, 1, 1}), "(A) (142, 7) (5,6,4,0): A_1 = F, T / 7 = (1,1,1)");
            Check(r.dim_I == 2 && r.I_is_m && r.principal && r.I_loewy == 2 && r.I_layers == H({1, 1}) && r.I_socle == 1 && r.I_ann == 1,
                  "(A) (142, 7) (5,6,4,0): I = m, dim 2, principal, Loewy 2, soc 1, Ann = m^2 (prediction)");
            Check(h->eps.size() >= 3 && h->eps[2] == 3, "(A) (142, 7) (5,6,4,0): eps_3 = 3 (the route's eps)");
            Check(h->sat_final && h->sat_structure.size() == 2, "(A) (142, 7) (5,6,4,0): T^sat / T has two cyclic summands, final; got " +
                                                                  Txt(h->sat_structure));
        }
        const SatBlock* d = Find(c, {4, 0, 0, 0});
        Check(d != nullptr, "(A) (142, 7) Delta present");
        if (d) {
            const DefectReading& r = d->at;
            Check(d->A1_hilbert == H({1, 3}) && r.hilbert == H({1, 3, 2}), "(A) (142, 7) Delta: A_1 = (1,3), T / 7 = (1,3,2)");
            Check(r.dim_I == 2 && !r.I_is_m && !r.principal && r.min_generators == 2 && r.I_loewy == 1 && r.I_socle == 2 && r.I_ann == 5,
                  "(A) (142, 7) Delta: I = m^2: dim 2, two generators, Loewy 1, soc I = I, Ann I = m (dim 5 = 3 + 2) (prediction)");
        }
    }
    {
        const SatCell& c = Cell(82, 7, 4);
        const SatBlock* d = Find(c, {4, 0, 0, 0});
        Check(d != nullptr, "(A) (82, 7) Delta present");
        if (d) {
            const DefectReading& r = d->at;
            Check(d->A1_hilbert == H({1, 2}) && r.hilbert == H({1, 3}), "(A) (82, 7) Delta: A_1 = (1,2), T / 7 = (1,3)");
            Check(r.dim_I == 1 && r.principal && r.I_loewy == 1 && r.I_socle == 1 && r.I_ann == 3,
                  "(A) (82, 7) Delta: dim I = 1, principal, I in soc, Ann I = m (dim 3) (prediction)");
            Check(d->sat_final && d->sat_structure.size() == 1, "(A) (82, 7) Delta: T^sat / T cyclic");
        }
    }
}

void ArmB() {
    Tally t;
    for (const auto& kv : g_cells)
        for (const SatBlock& b : kv.second.blocks) GateBlock(kv.second, b, "(B)", t);
    std::cout << "SAT_DEFECT_SUMMARY arm=B blocks=" << t.blocks << " ok=" << t.ok << " torsion_refused=" << t.torsion
              << " not_stable=" << t.not_stable << " nonzero_I=" << t.nonzero_I << " hidden=" << t.hidden << " forms_eq=" << t.forms_eq
              << " forms_le=" << t.forms_le << "\n";
    Check(t.nonzero_I >= 3 && t.hidden >= 1 && t.forms_eq >= 3, "(B) the default cells exercise non-zero I, a hidden block and the forms equality");
    Check(t.torsion >= 1, "(B) the default cells include a refused torsion block");
}

void ArmC() {
    hecke_ext::ringalg::LoewyOptions o;
    o.kmax = 4;
    const hecke_ext::ringalg::LoewyCell L = hecke_ext::ringalg::ComputeRingLoewyCell(82, 7, o);
    const SatCell& c = Cell(82, 7, 4);
    uint32_t compared = 0;
    for (const auto& lb : L.blocks) {
        const SatBlock* b = Find(c, lb.system);
        Check(b != nullptr, "(C) (82, 7): every LOEWY_RING block has a SAT_DEFECT block");
        if (!b || !lb.torsion.empty()) continue;
        ++compared;
        Check(b->verdict == lb.verdict && b->K_stab == lb.K_stab && b->A1_hilbert == lb.A1.hilbert,
              "(C) (82, 7): verdict, K_stab and A_1 agree with ComputeRingLoewyCell");
        if (lb.verdict == "stable" && lb.last_T()) {
            Check(b->at.hilbert == lb.last_T()->hilbert, "(C) (82, 7): T / ell's Hilbert function == the route's AK_hilbert");
            Check(b->eps.size() > lb.K_stab - 1 && static_cast<int64_t>(b->eps[lb.K_stab - 1]) == lb.eps, "(C) (82, 7): eps at K_stab agrees");
        }
    }
    uint32_t free_blocks = 0;
    for (const SatBlock& b : c.blocks) free_blocks += b.torsion.empty() ? 1 : 0;
    Check(L.blocks.size() == c.blocks.size() && compared == free_blocks && compared >= 3,
          "(C) (82, 7): the same blocks (" + std::to_string(compared) + " torsion-free compared)");
}

void ArmD() {
    const SatCell& c = Cell(18, 5, 4);
    uint32_t tors = 0;
    for (const SatBlock& b : c.blocks)
        if (!b.torsion.empty()) {
            ++tors;
            Check(b.status == "refused:torsion_block" && Columns(b).sat_defect_dim == "not_computed:torsion_block",
                  "(D) (18, 5): a torsion block is refused by name, its census columns read not_computed:torsion_block");
        }
    Check(c.e_max > 0 && tors >= 1, "(D) (18, 5): a Dickson-locus cell (e_max > 0) with a torsion block");
}

// ---- (slot-16 WP-5 audit repair, finding 1) the note's cell predictions as GATES -------------------
// The hidden-chain law L-F1 (WP-12; rho_Delta (x) omega^j at its onset n = 12 ell + j (ell + 1) - 2) and the
// ell = 37 Kummer cells, with the expectations of WP-5_tor_note.md Sec. 4 hard-coded:
//   hidden_chain: the block at the system a_p = p^j tau(p) mod ell (p = 2, 3, 5, 7), a_ell = 0 exists, is ok,
//     D = 4, rank_T = 2, A_1 = F, T / ell = F[x] / x^2, K_stab = 2, and I = m: dim 1, principal, Loewy 1,
//     Ann I = m (dim 1), T^sat / T = Z / ell (final, delta = 1); every other ok block of the cell has I = 0.
//   kummer_I0 (ell = 37): every block ok (torsion-free, stable by KMAX), I = 0 on every block, and a Kummer
//     block (A_1 = F_37[x] / x^4) is present, saturated (final, delta = 0).
// (168, 13) is the law's j = 1 onset at ell = 13 -- not one of WP-5's three hook cells, but the cheapest
// instance of the same law (13 s at KMAX = 2, against 28-37 s for (220, 17), (238, 17), (246, 19)), so it is
// the DEFAULT arm (H); the other seven are gated whenever HECKE_SAT_DEFECT_CELLS lists them (hook only:
// 28-124 s per cell, beyond the standard tier's budget).
struct Expect {
    uint32_t n;
    uint64_t ell;
    const char* kind;   // hidden_chain | kummer_I0
    uint32_t j;         // the twist of the hidden chain
};
const std::vector<Expect>& Expectations() {
    static const std::vector<Expect> e = {{168, 13, "hidden_chain", 1}, {220, 17, "hidden_chain", 1}, {238, 17, "hidden_chain", 2},
                                          {246, 19, "hidden_chain", 1}, {258, 37, "kummer_I0", 0},    {294, 37, "kummer_I0", 0},
                                          {296, 37, "kummer_I0", 0},    {334, 37, "kummer_I0", 0}};
    return e;
}
const Expect* FindExpect(uint32_t n, uint64_t ell) {
    for (const Expect& e : Expectations())
        if (e.n == n && e.ell == ell) return &e;
    return nullptr;
}

// p^j tau(p) mod ell, tau(2) = -24, tau(3) = 252, tau(5) = 4830, tau(7) = -16744.
uint64_t ThetaDelta(uint64_t p, uint32_t j, uint64_t ell) {
    int64_t tau = p == 2 ? -24 : p == 3 ? 252 : p == 5 ? 4830 : -16744;
    int64_t t = tau % static_cast<int64_t>(ell);
    if (t < 0) t += static_cast<int64_t>(ell);
    uint64_t r = static_cast<uint64_t>(t);
    for (uint32_t i = 0; i < j; ++i) r = r * (p % ell) % ell;
    return r;
}

// Gates the cell against its expectation; prints one SAT_DEFECT_EXPECT line.  Returns true when held.
bool GateExpectation(const SatCell& c, const Expect& e, const std::string& arm) {
    const std::string tag = arm + " " + S(c.n, c.ell) + " " + e.kind;
    const uint32_t g0 = Check.fails;
    if (std::string(e.kind) == "hidden_chain") {
        std::vector<uint64_t> sys;
        for (uint64_t p : c.primes) sys.push_back(p == c.ell ? 0 : ThetaDelta(p, e.j, c.ell));
        Check(c.primes == std::vector<uint64_t>({2, 3, 5, 7, c.ell}), tag + ": the default prime set {2,3,5,7,ell}");
        const SatBlock* h = Find(c, sys);
        Check(h != nullptr, tag + ": the L-F1 block at the system p^" + std::to_string(e.j) + " tau(p) mod ell, a_ell = 0, exists " +
                                Txt(std::vector<uint32_t>(sys.begin(), sys.end())));
        if (h) {
            const DefectReading& r = h->at;
            Check(h->status == "ok" && h->D == 4 && h->rank_T == 2 && h->dim_A1 == 1 && h->K_stab == 2 && r.hilbert == H({1, 1}),
                  tag + ": D = 4, rank_T = 2, A_1 = F, T / ell = F[x] / x^2, K_stab = 2");
            Check(r.dim_I == 1 && r.I_is_m && r.principal && r.I_loewy == 1 && r.I_ann == 1,
                  tag + ": I = m, dim 1, principal, Loewy 1, Ann I = m (dim 1) (note Sec. 4 prediction)");
            Check(h->sat_final && h->sat_structure == H({1}) && h->sat_length == 1, tag + ": T^sat / T = Z / ell (cyclic, delta = 1)");
        }
        for (const SatBlock& b : c.blocks)
            if (&b != h && b.status == "ok") Check(b.at.dim_I == 0, tag + ": every other ok block has I = 0");
    } else {
        uint32_t kummer = 0;
        for (const SatBlock& b : c.blocks) {
            Check(b.status == "ok", tag + ": every block ok (torsion-free, stable by KMAX), got " + b.status);
            if (b.status != "ok") continue;
            Check(b.at.dim_I == 0, tag + ": I = 0 (the note's ell = 37 prediction: the Loewy-4 algebra is saturated)");
            if (b.A1_hilbert == H({1, 1, 1, 1})) {
                ++kummer;
                Check(b.sat_final && b.sat_length == 0, tag + ": the Kummer block is saturated (delta = 0, final)");
            }
        }
        Check(kummer >= 1, tag + ": a Kummer block (A_1 = F_37[x] / x^4) is present");
    }
    const bool held = Check.fails == g0;
    std::cout << "SAT_DEFECT_EXPECT n=" << c.n << " ell=" << c.ell << " kind=" << e.kind << " j=" << e.j << " arm=" << arm << " "
              << (held ? "held" : "FAILED") << "\n";
    return held;
}

void ArmH() {
    const Expect* e = FindExpect(168, 13);
    const SatCell& c = Cell(168, 13, 2);   // K_stab = 2 and the plateau at K = 2: KMAX = 2 suffices
    GateExpectation(c, *e, "(H)");
}

// ---- (ENV) ---------------------------------------------------------------------------------
// (slot-16 WP-5 audit repairs, findings 2, 3, 5)
//   * a SET but empty HECKE_SAT_DEFECT_CELLS (or one with no valid cell) FAILS: 0 cells is never a pass;
//   * every field is parsed by a checked helper naming the cell: n even, 2 <= n <= 2^20; ell prime,
//     5 <= ell < 2^31; KMAX in [2, 64] with ell^KMAX < 2^64 -- a bad cell is a failing check, the others run;
//   * refused blocks are reported (one SAT_DEFECT_REFUSED line each) and counted on the EFFECTIVE line, which
//     is printed AFTER the run.  Gating, per the report's intent (torsion blocks are refused by design, the
//     "separate treatment"; not_stable_at_KMAX is a refusal of a reading that KMAX cut short): a cell with no
//     ok block fails unless every block is a torsion refusal; HECKE_SAT_DEFECT_STRICT=1 fails every
//     not_stable_at_KMAX block as well (a census shard should set it, so a too-small KMAX is loud);
//   * a listed cell with an expectation (Expectations()) is gated against it.
bool ParseField(const std::string& x, uint64_t lo, uint64_t hi, uint64_t* out) {
    if (x.empty() || x.size() > 12) return false;
    for (char ch : x)
        if (ch < '0' || ch > '9') return false;
    const uint64_t v = std::stoull(x);
    if (v < lo || v > hi) return false;
    *out = v;
    return true;
}

bool FitsU64(uint64_t ell, uint32_t k) {
    unsigned __int128 r = 1;
    for (uint32_t i = 0; i < k; ++i) {
        r *= ell;
        if (r >> 64) return false;
    }
    return true;
}

void ArmEnv(const char* env) {
    std::cout << "-- (ENV) HECKE_SAT_DEFECT_CELLS=\"" << env << "\"\n";
    const bool strict = [] {
        const char* s = std::getenv("HECKE_SAT_DEFECT_STRICT");
        return s && std::string(s) == "1";
    }();
    std::vector<std::vector<uint64_t>> cells;
    uint32_t bad = 0;
    for (const oc_util::cells::Token& t : oc_util::cells::Tokenize(env)) {
        const std::string name = "HECKE_SAT_DEFECT_CELLS cell " + std::to_string(t.pos) + " '" + t.text + "'";
        const std::vector<std::string>& f = t.raw;
        uint64_t n = 0, ell = 0, kmax = 0;
        std::string why;
        if (f.size() != 2 && f.size() != 3) why = "is not n:ell[:KMAX]";
        else if (!ParseField(f[0], 2, 1u << 20, &n) || n % 2 != 0) why = "n is not an even integer in [2, 2^20]";
        else if (!ParseField(f[1], 5, (1ull << 31) - 1, &ell) || !hecke::IsPrimeU64(ell)) why = "ell is not a prime in [5, 2^31)";
        else if (f.size() == 3 && (!ParseField(f[2], 2, 64, &kmax) || !FitsU64(ell, static_cast<uint32_t>(kmax))))
            why = "KMAX is not an integer in [2, 64] with ell^KMAX < 2^64";
        else if (n + 2 >= 12 && (ell >= hecke_ext::forms::kFormsMaxEllExclusive ||   // (slot-24 ELL-W audit F6; FNTT range)
                                 !hecke_ext::forms::FormsSpaceSupported(
                                     static_cast<uint32_t>(ell),
                                     hecke_ext::forms::SatFormsPrecision(static_cast<uint32_t>(n + 2), static_cast<uint32_t>(ell)))))
            why = "the forms cross-route cannot build M_k at this (n, ell) (forms::FormsSpaceSupported: ell >= 2^16 needs the "
                  "FNTT route within 2N - 1 <= 2^25 and N (ell-1)^2 < P1 P2, or the three-prime route (HECKE_FORMS_NTT3 not 0) "
                  "within nb B (ell-1)^2 < P1 P2 P3)";
        if (!why.empty()) {
            ++bad;
            Check(false, name + ": " + why + " -- refused");
            continue;
        }
        if (f.size() == 2) {   // KMAX defaults to 6, lowered while ell^KMAX >= 2^64 (the native module route's range)
            uint32_t k = 6;
            while (k > 2 && !FitsU64(ell, k)) --k;
            kmax = k;
        }
        cells.push_back({n, ell, kmax});
    }
    if (cells.empty()) {
        Check(false, "(ENV) HECKE_SAT_DEFECT_CELLS is set but names no valid cell (" + std::to_string(bad) + " refused): 0 cells is not a pass");
        std::cout << "EFFECTIVE HeckeSaturationDefectValidation arm=CELLS cells=0 refused_cells=" << bad << " (FAILED: no cell)\n";
        return;
    }
    std::cout << hecke_ext::ringalg::PrecisionReuseEffectiveLine() << "\n";   // (slot-23 G1) the level route of these cells
    Tally t;
    uint32_t expected = 0, expected_held = 0;
    for (const auto& f : cells) {
        SatOptions o;
        o.kmax = static_cast<uint32_t>(f[2]);
        const hecke_ext::ringalg::SatPerfCounters sp0 = hecke_ext::ringalg::SatPerfCounts();   // (S24 SATPERF audit F9)
        const SatCell c = ComputeSaturationDefectCell(static_cast<uint32_t>(f[0]), f[1], o);
        const hecke_ext::ringalg::SatPerfCounters sp1 = hecke_ext::ringalg::SatPerfCounts();
        uint32_t ok = 0, torsion = 0;
        for (const SatBlock& b : c.blocks) {
            std::cout << SatDefectLine(c, b) << "\n";
            GateBlock(c, b, "(ENV)", t);
            if (b.status == "ok") ++ok;
            else {
                if (b.status == "refused:torsion_block") ++torsion;
                std::cout << "SAT_DEFECT_REFUSED n=" << c.n << " ell=" << c.ell << " system="
                          << Txt(std::vector<uint32_t>(b.system.begin(), b.system.end())) << " reason=" << b.status << "\n";
                if (strict && b.status == "refused:not_stable_at_KMAX")
                    Check(false, "(ENV) " + S(c.n, c.ell) + " a block is not stable by KMAX = " + std::to_string(c.kmax) +
                                     " (HECKE_SAT_DEFECT_STRICT=1)");
            }
        }
        if (ok == 0 && torsion != c.blocks.size())
            Check(false, "(ENV) " + S(c.n, c.ell) + ": no block was read (every block refused, not all as torsion) -- raise KMAX");
        if (const Expect* e = FindExpect(c.n, c.ell)) {
            ++expected;
            expected_held += GateExpectation(c, *e, "(ENV)") ? 1 : 0;
        }
        std::cout << "SAT_DEFECT_CELL n=" << c.n << " ell=" << c.ell << " kmax=" << c.kmax << " e_max=" << c.e_max << " blocks=" << c.blocks.size()
                  << " ok=" << ok << " seconds=" << c.seconds << " module_seconds=" << c.module_seconds << "\n";
        // (slot-23 G1, [C, obl:precisionreuse](vi)) the levels read (K_run) and run (Lv_run); a line of its own, so the
        // SAT_DEFECT_CELL line keeps its bytes
        std::cout << "SAT_DEFECT_LEVELS n=" << c.n << " ell=" << c.ell << " K_run=" << Txt(c.K_run) << " Lv_run=" << Txt(c.Lv_run)
                  << " mode=" << c.level_mode << "\n";
        {   // (S24 SATPERF) the post-module stage clocks, diagnostic (a line of its own, so no other line moves)
            std::ostringstream os;
            double staged = 0.0;
            for (const auto& kv : c.stage_seconds) {
                os << " " << kv.first << "=" << kv.second;
                staged += kv.second;
            }
            std::cout << "SAT_DEFECT_TIMES n=" << c.n << " ell=" << c.ell << " seconds=" << c.seconds << " module=" << c.module_seconds
                      << os.str() << " other=" << (c.seconds - c.module_seconds - staged) << " sat_perf="
                      << (hecke_ext::ringalg::SatPerfEnabled() ? "on" : "off") << " memo_hits=" << (sp1.memo_hits - sp0.memo_hits)
                      << " memo_misses=" << (sp1.memo_misses - sp0.memo_misses) << " mul_narrow=" << (sp1.mul_narrow - sp0.mul_narrow)
                      << " mul_wide=" << (sp1.mul_wide - sp0.mul_wide) << " mul_parallel=" << (sp1.mul_parallel - sp0.mul_parallel) << "\n";
        }
    }
    std::cout << "SAT_DEFECT_SUMMARY arm=CELLS blocks=" << t.blocks << " ok=" << t.ok << " torsion_refused=" << t.torsion
              << " not_stable=" << t.not_stable << " nonzero_I=" << t.nonzero_I << " hidden=" << t.hidden << " forms_eq=" << t.forms_eq
              << " forms_le=" << t.forms_le << "\n";
    std::cout << "EFFECTIVE HeckeSaturationDefectValidation arm=CELLS cells=" << cells.size() << " refused_cells=" << bad
              << " blocks=" << t.blocks << " ok=" << t.ok << " torsion_refused=" << t.torsion << " not_stable=" << t.not_stable
              << " strict=" << (strict ? 1 : 0) << " expectations=" << expected_held << "/" << expected << "\n";
}

} // namespace

uint32_t HeckeSaturationDefectValidation() {
    Check.reset();
    g_cells.clear();
    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "HeckeSaturationDefectValidation (slot-16 WP-5: I_m = Tor_1(T^sat / T_m, F_ell) on torsion-free blocks)\n";
    if (const char* env = std::getenv("HECKE_SAT_DEFECT_CELLS")) {
        ArmEnv(env);
    } else {
        ArmP();
        ArmA();
        ArmD();
        ArmC();
        ArmH();   // (slot-16 WP-5 audit repair, finding 1) before (B), so (B) gates its blocks too
        ArmB();
    }
    g_cells.clear();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "HeckeSaturationDefectValidation: " << Check.checks << " checks, " << Check.fails << " failing, " << s << " s\n";
    return Check.fails ? 1u : 0u;
}
