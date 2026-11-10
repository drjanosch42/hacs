// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E1a (slot 2, WP-5).  See
// HeckeGenerationCertificates.h for the mathematics, the [E] / [X] / [C]
// citations, the socle bound, the cost model and the R3 contract.
//
// INCLUDE DISCIPLINE ([X] Sec. 1.1, enforced by
// docs/matrix-engine/tools/check_heckeext_includes.sh rules 1, 2 and 4): this
// TU includes its own header (hence HeckeLocalAlgebra.h, Hecke/SubspaceOps.h,
// Hecke/FpPoly.h, Hecke/HeckeModuleDecomposition.h and ModArithmetic's
// Matrix.h), ModArithmetic's EngineCheck.h / Fnv.h, and C++ standard headers.
// No Console/, no Cuda/, no Engine/, no BlockFlatMatrix.h, no `unordered_`.

#include "HeckeGenerationCertificates.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../Util/TextIo.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hecke_ext {

// CommutantRoute::kCommutantBudget is a `static constexpr` alias of
// hecke::kDefaultCommutantBudget since slot-4 WP-1 (E2a) -- implicitly inline
// in C++17, so it needs (and may have) no out-of-line definition here.

namespace {

using hecke::FpPoly;
using hecke::RrefRowSpan;
using hecke::VecOfMatrix;

// --- small deterministic helpers -------------------------------------------

// Byte-wise FNV-1a-64 for the R3.1 item 6 integrity field only (never on a
// result path): the ONE definition in src/ModArithmetic/Fnv.h, from
// its standard offset basis -- the same function, seed included, as every
// golden / campaign / benchmark hash in the tree (WP-2, and WP-7's decision
// recorded in HeckeSturmBound.cpp).
uint64_t Fnv1a64(const std::string& s) {
    return engine::FnvBytes(engine::kFnvOffset64, s);
}

using oc_util::HexU64;   // "0x" + 16 digits

// Saturating multiply / add on the modmul cost model, so a pathological shape
// can never wrap the budget comparison into "affordable".
uint64_t SatMul(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > UINT64_MAX / b) return UINT64_MAX;
    return a * b;
}
uint64_t SatAdd(uint64_t a, uint64_t b) {
    if (a > UINT64_MAX - b) return UINT64_MAX;
    return a + b;
}

// The companion matrix of a monic q of degree k, in the convention of the
// primary rational canonical form the tree already uses
// (HeckeModuleDecomposition.cpp's CompanionOfPoly): the basis of the cyclic
// summand is v, a v, ..., a^{k-1} v, so the sub-diagonal is 1 and the last
// column holds -q_i.  Negation is `ell - c` on a canonical residue, which is
// exact and needs no raw mod_* call.
ExtMatrix CompanionOf(const FpPoly& q) {
    ENGINE_CHECK_MSG(q.is_monic() && q.degree() >= 1,
                     "CompanionOf: needs a monic polynomial of degree >= 1");
    const uint32_t k = static_cast<uint32_t>(q.degree());
    ExtMatrix C(k, k, q.ell);
    for (uint32_t i = 1; i < k; ++i) C.data[i][i - 1] = 1u;
    for (uint32_t i = 0; i < k; ++i) {
        const uint32_t c = q.coeffs[i];
        C.data[i][k - 1] = (c == 0u) ? 0u : (q.ell - c);
    }
    return C;
}

// dim_{F_ell} soc V, soc V = { v : m_S v = 0 } = INTERSECT_p ker n_p.  Since
// m_S = (n_p)_p as an ideal of A_S, killing every generator kills the ideal.
uint32_t SocleOfModuleDim(const std::vector<ExtMatrix>& n, uint32_t D, uint32_t ell) {
    if (n.empty()) return D;
    ExtMatrix stacked = n[0];
    for (size_t j = 1; j < n.size(); ++j) stacked = hecke::VStack(stacked, n[j]);
    (void)ell;
    return hecke::KernelAsMatrix(stacked).cols;
}

// dim_{F_ell} soc A_S = dim Ann_{A_S}(m_S) = { b in A_S : b n_p = 0 for all p }
// -- the kernel of the (k D^2) x dim_A matrix whose column j stacks
// vec(b_j n_p) over p.  This is the same recipe ComputeLocalAlgebra uses, and
// here it is a RE-DERIVATION: the certificate never reads `dim_soc_A`.
uint32_t SocleOfAlgebraDim(const std::vector<ExtMatrix>& a_basis,
                           const std::vector<ExtMatrix>& n,
                           uint32_t D, uint32_t ell) {
    const uint32_t dim_A = static_cast<uint32_t>(a_basis.size());
    if (dim_A == 0) return 0;
    const uint32_t k = static_cast<uint32_t>(n.size());
    if (k == 0) return dim_A;
    ExtMatrix sys(k * D * D, dim_A, ell);
    for (uint32_t j = 0; j < dim_A; ++j) {
        for (uint32_t p = 0; p < k; ++p) {
            const std::vector<uint32_t> v = VecOfMatrix(a_basis[j] * n[p]);
            for (uint32_t r = 0; r < D * D; ++r) sys.data[p * D * D + r][j] = v[r];
        }
    }
    return hecke::KernelAsMatrix(sys).cols;
}

// n_p = t_p - P_p(w_hat), recomputed from a certificate's own stored data.
std::vector<ExtMatrix> RederiveGenerators(const CyclicityCertificate& c) {
    std::vector<ExtMatrix> n;
    n.reserve(c.t.size());
    for (size_t j = 0; j < c.t.size(); ++j) {
        n.push_back(c.t[j] - hecke::EvalPolyAtMatrix(c.character[j], c.w_hat));
    }
    return n;
}

// The FIXED candidate list of [X] Sec. 2.2 Stage 4(b) step 2: the RREF basis
// elements b_1 .. b_{dim A_S}, then the prefix sums b_1+b_2, b_1+b_2+b_3, ...
std::vector<ExtMatrix> CandidateList(const std::vector<ExtMatrix>& a_basis,
                                     uint32_t D, uint32_t ell) {
    std::vector<ExtMatrix> out = a_basis;
    if (a_basis.size() >= 2) {
        ExtMatrix acc = a_basis[0];
        for (size_t j = 1; j < a_basis.size(); ++j) {
            acc = acc + a_basis[j];
            out.push_back(acc);
        }
    }
    (void)D; (void)ell;
    return out;
}

// The first candidate whose minimal polynomial on V has MAXIMAL degree.  The
// choice affects the cost, never the verdict ([X] Stage 4(b) step 2) -- but it
// is deterministic, and the certificate's clause 10 enforces exactly it.
uint32_t ChooseCandidate(const std::vector<ExtMatrix>& cands) {
    int64_t best_deg = -1;
    uint32_t best = 0;
    for (size_t j = 0; j < cands.size(); ++j) {
        const int64_t d = hecke::MinPolyKrylov(cands[j]).degree();
        if (d > best_deg) { best_deg = d; best = static_cast<uint32_t>(j); }
    }
    return best;
}

// pi_a: the irreducible whose powers are the invariant factors of `a`.  A_S is
// local, so a = (a polynomial in w_hat) + nilpotent and its minimal polynomial
// is pi_a^e for ONE irreducible pi_a -- certified here by the shipped,
// deterministic FactorPoly certificate rather than assumed.
FpPoly PrimaryIrreducibleOf(const ExtMatrix& a) {
    const FpPoly mp = hecke::MinPolyKrylov(a);
    ENGINE_CHECK_MSG(mp.degree() >= 1, "CommutantRoute: a has no minimal polynomial");
    const hecke::FpPolyFactorization fac = hecke::FactorPoly(mp);
    ENGINE_CHECK_MSG(fac.factors.size() == 1u,
                     "CommutantRoute: the chosen element is not primary on a local block");
    return fac.factors[0].pi;
}

// The explicit basis of End_{F_ell[a]}(V), [E] `cert:comm` / [X] Stage 4(b)
// step 2.  With V = (+)_i F_ell[X]/(g_i) in the RCF basis (columns of R),
// g_i = pi^{s_i}, a homomorphism is determined by the image of each generator
// 1_i, which must be annihilated by g_i; the basis of
// Hom(F[X]/g_i, F[X]/g_j) is { X^r * (g_j / gcd(g_i, g_j)) : 0 <= r <
// deg gcd(g_i, g_j) }.  The order is FIXED: i ascending, then j ascending,
// then r ascending.
std::vector<ExtMatrix> EndOfCyclicBasis(const ExtMatrix& R_in, const FpPoly& pi,
                                        const std::vector<uint32_t>& sizes,
                                        uint32_t D, uint32_t ell) {
    const uint32_t deg = static_cast<uint32_t>(pi.degree());
    const size_t m = sizes.size();
    std::vector<uint32_t> off(m, 0u);
    uint32_t acc = 0;
    for (size_t i = 0; i < m; ++i) { off[i] = acc; acc += sizes[i] * deg; }
    ENGINE_CHECK_MSG(acc == D, "EndOfCyclicBasis: the invariant factors do not fill V");

    ExtMatrix R = R_in;
    ExtMatrix Rinv = R.inverse();

    std::vector<ExtMatrix> out;
    for (size_t i = 0; i < m; ++i) {
        const uint32_t ki = sizes[i] * deg;
        for (size_t j = 0; j < m; ++j) {
            const uint32_t kj = sizes[j] * deg;
            const uint32_t d = std::min(sizes[i], sizes[j]);
            const FpPoly gj = FpPoly::pow(pi, sizes[j]);
            const FpPoly X  = FpPoly::X(ell);
            // base_r = X^r * (g_j / gcd(g_i, g_j)) mod g_j, built incrementally
            // so the whole family costs O(k_j^2) coefficient operations rather
            // than a fresh FpPoly::pow per (r, col).
            FpPoly base = FpPoly::mod(FpPoly::pow(pi, sizes[j] - d), gj);
            for (uint32_t r = 0; r < d * deg; ++r) {
                ExtMatrix E(D, D, ell);
                FpPoly img = base;                 // phi(X^col 1_i) = X^col * base
                for (uint32_t col = 0; col < ki; ++col) {
                    for (int64_t e = 0; e <= img.degree(); ++e) {
                        ENGINE_CHECK_MSG(static_cast<uint32_t>(e) < kj,
                                         "EndOfCyclicBasis: image leaves its block");
                        const uint32_t row = off[j] + static_cast<uint32_t>(e);
                        E.data[row][off[i] + col] = img.coeffs[static_cast<size_t>(e)];
                    }
                    if (col + 1 < ki) img = FpPoly::mod(img * X, gj);
                }
                out.push_back(R * (E * Rinv));
                base = FpPoly::mod(base * X, gj);
            }
        }
    }
    return out;
}

uint32_t EndOfCyclicDim(const FpPoly& pi, const std::vector<uint32_t>& sizes) {
    const uint32_t deg = static_cast<uint32_t>(pi.degree());
    uint64_t total = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        for (size_t j = 0; j < sizes.size(); ++j) total += std::min(sizes[i], sizes[j]);
    }
    return static_cast<uint32_t>(total * deg);
}

// The rank of { vec(E) : E in `mats` } inside F_ell^{D^2}.
uint32_t RankOfMatrixSet(const std::vector<ExtMatrix>& mats, uint32_t D, uint32_t ell) {
    RrefRowSpan span(D * D, ell);
    for (const ExtMatrix& E : mats) span.insert(VecOfMatrix(E));
    return span.dim();
}

// The structured system, reduced: rows in the fixed order "operators ascending
// p, then the commutator entry (i, j) row-major", streamed into a canonical
// incremental RREF.  `rows` says how many of the k*D^2 rows to consume.
RrefRowSpan ReduceStructuredSystem(const std::vector<std::vector<ExtMatrix>>& comm,
                                   uint64_t rows, uint32_t u, uint32_t D, uint32_t ell) {
    RrefRowSpan span(u, ell);
    const uint64_t per_op = static_cast<uint64_t>(D) * static_cast<uint64_t>(D);
    for (uint64_t r = 0; r < rows; ++r) {
        const size_t p = static_cast<size_t>(r / per_op);
        const uint64_t idx = r % per_op;
        const uint32_t i = static_cast<uint32_t>(idx / D);
        const uint32_t j = static_cast<uint32_t>(idx % D);
        std::vector<uint32_t> row(u, 0u);
        for (uint32_t c = 0; c < u; ++c) row[c] = comm[p][c].data[i][j];
        span.insert(std::move(row));
    }
    return span;
}

std::vector<std::vector<ExtMatrix>> FormCommutators(const std::vector<ExtMatrix>& end_basis,
                                                    const std::vector<ExtMatrix>& t) {
    std::vector<std::vector<ExtMatrix>> comm(t.size());
    for (size_t p = 0; p < t.size(); ++p) {
        comm[p].reserve(end_basis.size());
        for (const ExtMatrix& E : end_basis) comm[p].push_back(E * t[p] - t[p] * E);
    }
    return comm;
}

// --- JSON helpers (fixed key order, no pointers, no hash container) --------

void AppendU32List(std::ostringstream& os, const std::vector<uint32_t>& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) { if (i) os << ','; os << v[i]; }
    os << ']';
}
void AppendU64List(std::ostringstream& os, const std::vector<uint64_t>& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) { if (i) os << ','; os << v[i]; }
    os << ']';
}
void AppendU32Table(std::ostringstream& os, const std::vector<std::vector<uint32_t>>& t) {
    os << '[';
    for (size_t i = 0; i < t.size(); ++i) { if (i) os << ','; AppendU32List(os, t[i]); }
    os << ']';
}
void AppendMatrix(std::ostringstream& os, const ExtMatrix& M) {
    os << '[';
    for (uint32_t i = 0; i < M.rows; ++i) {
        if (i) os << ',';
        os << '[';
        for (uint32_t j = 0; j < M.cols; ++j) { if (j) os << ','; os << M.data[i][j]; }
        os << ']';
    }
    os << ']';
}

bool SameMatrix(const ExtMatrix& a, const ExtMatrix& b) {
    return a.rows == b.rows && a.cols == b.cols && a.p == b.p && a.data == b.data;
}

} // namespace

// ===========================================================================
// enum -> string
// ===========================================================================

const char* ToString(CommutantOutcome outcome) {
    switch (outcome) {
        case CommutantOutcome::InProgress:           return "in_progress";
        case CommutantOutcome::NotNeeded:            return "not_needed";
        case CommutantOutcome::ExcludedBySocleBound: return "excluded_by_socle_bound";
        case CommutantOutcome::OverBudget:           return "over_budget";
        case CommutantOutcome::Passed:               return "passed";
        case CommutantOutcome::Failed:               return "failed";
    }
    return "unknown";
}

const char* ToString(GenerationKind kind) {
    switch (kind) {
        case GenerationKind::Unavailable: return "unavailable";
        case GenerationKind::Cyclic:      return "cyclic";
        case GenerationKind::Commutant:   return "commutant";
        case GenerationKind::Sturm:       return "sturm(B')";
    }
    return "unknown";
}

// ===========================================================================
// CommutantState
// ===========================================================================

bool CommutantState::equals(const CommutantState& o, bool ignore_budget_history) const {
    if (ell != o.ell || D != o.D || f != o.f || k != o.k) return false;
    if (budget != o.budget) return false;
    if (!ignore_budget_history && budget_history != o.budget_history) return false;
    if (dim_A != o.dim_A || t_top != o.t_top || s_soc != o.s_soc) return false;
    if (dim_soc_A != o.dim_soc_A || socle_bound != o.socle_bound || admits != o.admits) return false;
    if (prepared != o.prepared || candidate_index != o.candidate_index) return false;
    if (!SameMatrix(a, o.a) || !SameMatrix(R, o.R)) return false;
    if (!(pi_a == o.pi_a)) return false;
    if (inv_sizes != o.inv_sizes || u != o.u) return false;
    if (total_rows != o.total_rows || rows_done != o.rows_done) return false;
    if (modmul_spent != o.modmul_spent) return false;
    if (system_rref != o.system_rref || system_pivots != o.system_pivots) return false;
    if (rank != o.rank || outcome != o.outcome || dim_C != o.dim_C) return false;
    return true;
}

std::string CommutantState::to_json() const {
    std::ostringstream os;
    os << "{\"kind\":\"commutant\"";
    os << ",\"ell\":" << ell;
    os << ",\"D\":" << D;
    os << ",\"f\":" << f;
    os << ",\"k\":" << k;
    os << ",\"budget\":" << budget;
    os << ",\"budget_history\":"; AppendU64List(os, budget_history);
    os << ",\"dim_A_S\":" << dim_A;
    os << ",\"t\":" << t_top;
    os << ",\"s\":" << s_soc;
    os << ",\"dim_soc_A_S\":" << dim_soc_A;
    os << ",\"socle_bound\":" << socle_bound;
    os << ",\"admits\":" << (admits ? "true" : "false");
    os << ",\"prepared\":" << (prepared ? "true" : "false");
    os << ",\"candidate_index\":" << candidate_index;
    os << ",\"a\":"; AppendMatrix(os, a);
    os << ",\"pi_a\":\"" << pi_a.to_string() << "\"";
    os << ",\"invariant_factor_exponents\":"; AppendU32List(os, inv_sizes);
    os << ",\"R\":"; AppendMatrix(os, R);
    os << ",\"u\":" << u;
    os << ",\"total_rows\":" << total_rows;
    os << ",\"rows_done\":" << rows_done;
    os << ",\"modmul_spent\":" << modmul_spent;
    os << ",\"rank\":" << rank;
    os << ",\"system_pivots\":"; AppendU32List(os, system_pivots);
    os << ",\"system_rref\":"; AppendU32Table(os, system_rref);
    os << ",\"outcome\":\"" << ToString(outcome) << "\"";
    os << ",\"dim_C\":";
    if (outcome == CommutantOutcome::Passed || outcome == CommutantOutcome::Failed) {
        os << dim_C;
    } else {
        os << "null";
    }
    os << '}';
    return os.str();
}

// ===========================================================================
// CommutantCertificate
// ===========================================================================

const char* CommutantCertificate::verify_message(uint32_t code) {
    switch (code) {
        case 0:  return "ok";
        case 1:  return "malformed: ell, D or f out of range, f does not divide D, or the "
                        "embedded (a)-certificate disagrees on them";
        case 2:  return "malformed: dim A_S does not match the stored A_S basis";
        case 3:  return "malformed: the outcome is in_progress, or `forced` contradicts it";
        case 4:  return "the embedded cyclicity certificate does not verify";
        case 5:  return "t = dim_F(V / m_S V) disagrees with the re-derived Nakayama quotient";
        case 6:  return "s = dim_F soc V disagrees with the re-derived intersection of ker n_p";
        case 7:  return "dim_{F_ell} soc A_S disagrees with the re-derived annihilator";
        case 8:  return "the socle bound or the `admits` gate disagrees with the re-derived numbers";
        case 9:  return "excluded_by_socle_bound claimed where the gate admits (b), or vice versa";
        case 10: return "the chosen element is not the first candidate of maximal minimal-"
                        "polynomial degree in the fixed candidate list";
        case 11: return "pi_a is not monic irreducible, or the invariant factors do not "
                        "reproduce the minimal polynomial and dim V";
        case 12: return "R is not invertible, or R^{-1} a R is not the companion form of (g_i)";
        case 13: return "u, or the rebuilt End_{F_ell[a]}(V) basis (commutation, independence), "
                        "disagrees with the stored data";
        case 14: return "the A_S basis does not lie in End_{F_ell[a]}(V), or does not commute "
                        "with every t_p";
        case 15: return "the partial reduction stored for an over_budget state is not the RREF "
                        "of the first rows_done rows";
        case 16: return "the re-formed and re-reduced system has a different rank, or dim C "
                        "disagrees with u - rank";
        case 17: return "the computed dim C violates the socle bound";
        case 18: return "the outcome disagrees with dim C == dim A_S";
        default: return "unknown code";
    }
}

uint32_t CommutantCertificate::verify() const {
    // --- 1-3: shapes and the outcome's own consistency. -------------------
    if (ell < 2 || D == 0 || f == 0 || (D % f) != 0) return 1;
    if (cyclicity.ell != ell || cyclicity.D != D || cyclicity.f != f) return 1;
    if (dim_A == 0 || dim_A != static_cast<uint32_t>(cyclicity.a_basis.size())) return 2;
    if (outcome == CommutantOutcome::InProgress) return 3;
    if (outcome == CommutantOutcome::NotNeeded && (forced || !cyclicity.cyclic)) return 3;
    if (!forced && cyclicity.cyclic && outcome != CommutantOutcome::NotNeeded) return 3;

    // --- 4: route (a)'s data, re-derived by its own verify(). -------------
    if (cyclicity.verify() != 0) return 4;
    if (outcome == CommutantOutcome::NotNeeded) return 0;   // Nakayama; nothing further

    // --- 5-8: the socle-bound numbers, all RE-DERIVED. --------------------
    const std::vector<ExtMatrix> n = RederiveGenerators(cyclicity);
    if ((cyclicity.dim_quotient % f) != 0) return 5;
    if (t_top != cyclicity.dim_quotient / f) return 5;

    const uint32_t soc_V = SocleOfModuleDim(n, D, ell);
    if ((soc_V % f) != 0 || s_soc != soc_V / f) return 6;

    if (dim_soc_A != SocleOfAlgebraDim(cyclicity.a_basis, n, D, ell)) return 7;

    {
        const uint64_t hom = static_cast<uint64_t>(f) * t_top * s_soc;
        if (dim_soc_A > dim_A) return 8;
        const uint64_t bound = static_cast<uint64_t>(dim_A) + hom - dim_soc_A;
        if (socle_bound != bound) return 8;
        if (admits != (hom <= static_cast<uint64_t>(dim_soc_A))) return 8;
    }

    // --- 9: the gate, and the proof-of-failure outcome. -------------------
    if (outcome == CommutantOutcome::ExcludedBySocleBound) {
        if (admits || cyclicity.cyclic) return 9;
        return 0;   // the claim is exactly "(b) provably fails"; it is proved
    }
    if (!forced && !admits) return 9;

    // --- 10: the cyclic-most element, re-chosen from the fixed list. ------
    {
        const std::vector<ExtMatrix> cands = CandidateList(cyclicity.a_basis, D, ell);
        if (candidate_index >= cands.size()) return 10;
        if (ChooseCandidate(cands) != candidate_index) return 10;
        if (!SameMatrix(a, cands[candidate_index])) return 10;
    }

    // --- 11: pi_a and the invariant factors. -------------------------------
    if (pi_a.ell != ell || pi_a.degree() < 1 || !pi_a.is_monic()) return 11;
    if (hecke::CertifyIrreducible(pi_a) != 0) return 11;
    if (inv_sizes.empty()) return 11;
    {
        const uint32_t deg = static_cast<uint32_t>(pi_a.degree());
        uint64_t total = 0;
        for (size_t i = 0; i < inv_sizes.size(); ++i) {
            if (inv_sizes[i] == 0) return 11;
            if (i && inv_sizes[i] > inv_sizes[i - 1]) return 11;   // non-increasing
            total += static_cast<uint64_t>(inv_sizes[i]) * deg;
        }
        if (total != D) return 11;
        // The minimal polynomial is pi_a^{s_1}: the largest invariant factor.
        const FpPoly mp = hecke::MinPolyKrylov(a);
        if (!(mp == FpPoly::pow(pi_a, inv_sizes[0]))) return 11;
    }

    // --- 12: the rational canonical form, entrywise. ----------------------
    {
        if (R.rows != D || R.cols != D || R.p != ell) return 12;
        ExtMatrix Rcopy = R;
        if (!Rcopy.is_invertible()) return 12;
        ExtMatrix Rinv = Rcopy.inverse();
        const ExtMatrix conj = Rinv * (a * R);
        const uint32_t deg = static_cast<uint32_t>(pi_a.degree());
        uint32_t off = 0;
        for (size_t i = 0; i < inv_sizes.size(); ++i) {
            const ExtMatrix comp = CompanionOf(FpPoly::pow(pi_a, inv_sizes[i]));
            for (uint32_t r = 0; r < comp.rows; ++r) {
                for (uint32_t c = 0; c < comp.cols; ++c) {
                    if (conj.data[off + r][off + c] != comp.data[r][c]) return 12;
                }
            }
            off += inv_sizes[i] * deg;
        }
        // And nothing outside the diagonal blocks.
        off = 0;
        std::vector<uint32_t> block_of(D, 0u);
        for (size_t i = 0; i < inv_sizes.size(); ++i) {
            for (uint32_t r = 0; r < inv_sizes[i] * deg; ++r) {
                block_of[off + r] = static_cast<uint32_t>(i);
            }
            off += inv_sizes[i] * deg;
        }
        for (uint32_t r = 0; r < D; ++r) {
            for (uint32_t c = 0; c < D; ++c) {
                if (block_of[r] != block_of[c] && conj.data[r][c] != 0u) return 12;
            }
        }
    }

    // --- 13: the End_{F_ell[a]}(V) basis, REBUILT by the fixed formula. ----
    std::vector<ExtMatrix> end_basis;
    {
        if (u != EndOfCyclicDim(pi_a, inv_sizes)) return 13;
        end_basis = EndOfCyclicBasis(R, pi_a, inv_sizes, D, ell);
        if (static_cast<uint32_t>(end_basis.size()) != u) return 13;
        for (const ExtMatrix& E : end_basis) {
            if (!((E * a) == (a * E))) return 13;
        }
        if (RankOfMatrixSet(end_basis, D, ell) != u) return 13;
    }

    // --- 14: A_S lies in C -- inside End_{F_ell[a]}(V), and commuting. -----
    {
        RrefRowSpan span(D * D, ell);
        for (const ExtMatrix& E : end_basis) span.insert(VecOfMatrix(E));
        for (const ExtMatrix& b : cyclicity.a_basis) {
            if (!span.contains(VecOfMatrix(b))) return 14;
            for (const ExtMatrix& tp : cyclicity.t) {
                if (!((b * tp) == (tp * b))) return 14;
            }
        }
    }

    // --- 15-18: the system itself. ----------------------------------------
    const std::vector<std::vector<ExtMatrix>> comm = FormCommutators(end_basis, cyclicity.t);
    const uint64_t total_rows =
        static_cast<uint64_t>(cyclicity.t.size()) * D * D;

    if (outcome == CommutantOutcome::OverBudget) {
        if (state.rows_done > total_rows) return 15;
        const RrefRowSpan part = ReduceStructuredSystem(comm, state.rows_done, u, D, ell);
        if (part.rows() != state.system_rref) return 15;
        if (part.pivots() != state.system_pivots) return 15;
        if (part.dim() != state.rank) return 15;
        return 0;   // the partial statement, and only it (R3.1 item 5)
    }

    const RrefRowSpan full = ReduceStructuredSystem(comm, total_rows, u, D, ell);
    if (full.dim() != rank) return 16;
    if (u < rank || dim_C != u - rank) return 16;
    if (dim_C < socle_bound) return 17;
    if ((dim_C == dim_A) != (outcome == CommutantOutcome::Passed)) return 18;
    if (outcome != CommutantOutcome::Passed && outcome != CommutantOutcome::Failed) return 18;
    return 0;
}

std::string CommutantCertificate::to_json() const {
    std::ostringstream os;
    os << "{\"kind\":\"commutant\"";
    os << ",\"ell\":" << ell;
    os << ",\"D\":" << D;
    os << ",\"f\":" << f;
    os << ",\"dim_A_S\":" << dim_A;
    os << ",\"t\":" << t_top;
    os << ",\"s\":" << s_soc;
    os << ",\"dim_soc_A_S\":" << dim_soc_A;
    os << ",\"socle_bound\":" << socle_bound;
    os << ",\"admits\":" << (admits ? "true" : "false");
    os << ",\"forced\":" << (forced ? "true" : "false");
    os << ",\"outcome\":\"" << ToString(outcome) << "\"";
    os << ",\"candidate_index\":" << candidate_index;
    os << ",\"pi_a\":\"" << pi_a.to_string() << "\"";
    os << ",\"invariant_factor_exponents\":"; AppendU32List(os, inv_sizes);
    os << ",\"u\":" << u;
    os << ",\"rank\":" << rank;
    os << ",\"dim_C\":";
    if (outcome == CommutantOutcome::Passed || outcome == CommutantOutcome::Failed) {
        os << dim_C;
    } else {
        os << "null";
    }
    os << ",\"certificate_a\":" << cyclicity.to_json();
    os << ",\"resumable\":" << resumable_json();
    os << ",\"verify\":" << verify();
    os << '}';
    return os.str();
}

std::string CommutantCertificate::resumable_json() const {
    const std::string state_json = state.to_json();
    std::ostringstream os;
    os << "{\"kind\":\"commutant\"";
    os << ",\"complete\":" << (outcome == CommutantOutcome::OverBudget ? "false" : "true");
    os << ",\"budget\":{\"modmul\":" << state.budget << "}";
    os << ",\"reached\":{\"rows_done\":" << state.rows_done
       << ",\"total_rows\":" << state.total_rows
       << ",\"modmul_spent\":" << state.modmul_spent
       << ",\"rank\":" << state.rank
       << ",\"u\":" << state.u << "}";
    os << ",\"budget_history\":"; AppendU64List(os, state.budget_history);

    // R3.1 item 6: inline up to 64 KiB, sidecar beyond it.  Only the record
    // layer (E2) knows the record stem, so without one we inline anyway and
    // say so rather than emitting a path that names no file.
    const bool oversize = state_json.size() > 65536u;
    if (!oversize) {
        os << ",\"state\":" << state_json;
    } else if (!state_sidecar_stem.empty()) {
        const std::string path = state_sidecar_stem + ".state.commutant.json";
        std::ofstream out(path.c_str(), std::ios::binary);
        ENGINE_CHECK_MSG(out.good(), "CommutantCertificate: cannot write the state sidecar");
        out << state_json;
        out.close();
        os << ",\"state_ref\":{\"path\":\"" << path
           << "\",\"fnv64\":\"" << HexU64(Fnv1a64(state_json))
           << "\",\"bytes\":" << state_json.size() << "}";
    } else {
        os << ",\"state_oversize\":true";
        os << ",\"state_fnv64\":\"" << HexU64(Fnv1a64(state_json)) << "\"";
        os << ",\"state\":" << state_json;
    }
    os << '}';
    return os.str();
}

// ===========================================================================
// CommutantRoute
// ===========================================================================

CommutantRoute::CommutantRoute(const LocalAlgebra& L, uint64_t budget, bool force) {
    if (!L.ok) throw std::invalid_argument("CommutantRoute: the block did not certify local");
    // (slot-4 WP-1 / E2a) 0 = the `commutant_budget` mathematical setting in
    // force; at its default that is kCommutantBudget, so the pre-E2a
    // behaviour of `CommutantRoute(L)` is unchanged.  The setting is
    // thread-local and propagates onto run_concurrently() task threads.
    if (budget == 0) budget = hecke::math_settings().commutant_budget.budget;
    if (budget == 0) throw std::invalid_argument("CommutantRoute: budget must be positive");
    ell_ = L.ell;
    D_ = L.D;
    f_ = L.f;
    t_ = L.t;
    n_ = L.n;
    a_basis_ = L.basis;
    cyc_ = L.cyclicity;
    force_ = force;

    st_.ell = ell_;
    st_.D = D_;
    st_.f = f_;
    st_.k = static_cast<uint32_t>(t_.size());
    st_.budget = budget;
    st_.dim_A = L.dim_A;
    st_.total_rows = static_cast<uint64_t>(st_.k) * D_ * D_;
}

void CommutantRoute::gates_() {
    ENGINE_CHECK_MSG((cyc_.dim_quotient % f_) == 0,
                     "CommutantRoute: dim V/m_S V is not a multiple of f");
    st_.t_top = cyc_.dim_quotient / f_;
    const uint32_t soc_V = SocleOfModuleDim(n_, D_, ell_);
    ENGINE_CHECK_MSG((soc_V % f_) == 0,
                     "CommutantRoute: dim soc V is not a multiple of f");
    st_.s_soc = soc_V / f_;
    st_.dim_soc_A = SocleOfAlgebraDim(a_basis_, n_, D_, ell_);
    ENGINE_CHECK_MSG(st_.dim_soc_A <= st_.dim_A,
                     "CommutantRoute: soc A_S is larger than A_S");
    const uint64_t hom = static_cast<uint64_t>(f_) * st_.t_top * st_.s_soc;
    st_.socle_bound = static_cast<uint32_t>(static_cast<uint64_t>(st_.dim_A) + hom
                                            - st_.dim_soc_A);
    st_.admits = (hom <= static_cast<uint64_t>(st_.dim_soc_A));
}

void CommutantRoute::prepare_() {
    const std::vector<ExtMatrix> cands = CandidateList(a_basis_, D_, ell_);
    st_.candidate_index = ChooseCandidate(cands);
    st_.a = cands[st_.candidate_index];
    st_.pi_a = PrimaryIrreducibleOf(st_.a);

    std::vector<uint32_t> sizes;
    st_.R = hecke::PrimaryRCFBasis(st_.a, st_.pi_a, sizes);
    st_.inv_sizes = sizes;
    st_.u = EndOfCyclicDim(st_.pi_a, st_.inv_sizes);
    ENGINE_CHECK_MSG(st_.u >= 1, "CommutantRoute: End_{F_ell[a]}(V) is trivial");

    end_basis_ = EndOfCyclicBasis(st_.R, st_.pi_a, st_.inv_sizes, D_, ell_);
    ENGINE_CHECK_MSG(static_cast<uint32_t>(end_basis_.size()) == st_.u,
                     "CommutantRoute: the End basis has the wrong size");
    st_.prepared = true;

    // The cost model (see the header): the formation of the k*u commutators,
    // and u^2 per reduced row.  [X] Stage 4(b)'s `k*D^2*u^2` is the second
    // term summed over the k*D^2 rows.
    const uint64_t D3 = SatMul(SatMul(D_, D_), D_);
    cost_formation_ = SatMul(SatMul(2ull, static_cast<uint64_t>(st_.k)),
                             SatMul(static_cast<uint64_t>(st_.u), D3));
    cost_row_ = SatMul(st_.u, st_.u);
    if (cost_row_ == 0) cost_row_ = 1;
}

uint64_t CommutantRoute::rows_for_budget_(uint64_t budget) const {
    if (cost_formation_ > budget) return 0;
    const uint64_t left = budget - cost_formation_;
    const uint64_t affordable = left / cost_row_;
    return (affordable < st_.total_rows) ? affordable : st_.total_rows;
}

uint64_t CommutantRoute::charge_(uint64_t rows) const {
    if (cost_formation_ > st_.budget) return 0;
    return SatAdd(cost_formation_, SatMul(rows, cost_row_));
}

void CommutantRoute::advance_() {
    const uint64_t target = rows_for_budget_(st_.budget);
    if (target > st_.rows_done) {
        if (!comm_formed_) {
            comm_ = FormCommutators(end_basis_, t_);
            comm_formed_ = true;
        }
        // Restore the canonical RREF of the prefix already consumed, then
        // extend it.  Re-inserting the rows of an RREF reproduces it exactly
        // (the RREF of a span is unique), so this is a true continuation.
        RrefRowSpan span(st_.u, ell_);
        for (const std::vector<uint32_t>& row : st_.system_rref) span.insert(row);
        ENGINE_CHECK_MSG(span.dim() == st_.rank,
                         "CommutantRoute: the stored partial reduction is not an RREF");
        const uint64_t per_op = static_cast<uint64_t>(D_) * D_;
        for (uint64_t r = st_.rows_done; r < target; ++r) {
            const size_t p = static_cast<size_t>(r / per_op);
            const uint64_t idx = r % per_op;
            const uint32_t i = static_cast<uint32_t>(idx / D_);
            const uint32_t j = static_cast<uint32_t>(idx % D_);
            std::vector<uint32_t> row(st_.u, 0u);
            for (uint32_t c = 0; c < st_.u; ++c) row[c] = comm_[p][c].data[i][j];
            span.insert(std::move(row));
        }
        st_.rows_done = target;
        st_.system_rref = span.rows();
        st_.system_pivots = span.pivots();
        st_.rank = span.dim();
    }
    st_.modmul_spent = charge_(st_.rows_done);

    if (st_.rows_done == st_.total_rows) {
        ENGINE_CHECK_MSG(st_.u >= st_.rank, "CommutantRoute: rank exceeds the unknown count");
        st_.dim_C = st_.u - st_.rank;
        ENGINE_CHECK_MSG(st_.dim_C >= st_.socle_bound,
                         "CommutantRoute: the computed dim C violates the socle bound");
        st_.outcome = (st_.dim_C == st_.dim_A) ? CommutantOutcome::Passed
                                               : CommutantOutcome::Failed;
    } else {
        st_.outcome = CommutantOutcome::OverBudget;
    }
}

CommutantOutcome CommutantRoute::run() {
    ENGINE_CHECK_MSG(st_.budget_history.empty(),
                     "CommutantRoute::run: already run -- use resume(new_budget)");
    st_.budget_history.push_back(st_.budget);
    gates_();
    if (!force_ && cyc_.cyclic) {
        st_.outcome = CommutantOutcome::NotNeeded;
        return st_.outcome;
    }
    if (!force_ && !st_.admits) {
        st_.outcome = CommutantOutcome::ExcludedBySocleBound;
        return st_.outcome;
    }
    prepare_();
    advance_();
    return st_.outcome;
}

CommutantOutcome CommutantRoute::resume(uint64_t new_budget) {
    ENGINE_CHECK_MSG(!st_.budget_history.empty(),
                     "CommutantRoute::resume: nothing to resume -- run() first");
    // The budget IS the reached state here: rows_done is a monotone function
    // of it, so a lower budget would un-do work (R3.1 item 2).
    ENGINE_CHECK_MSG(new_budget >= st_.budget, "resume: budget below reached state");
    // R3.3 item 3: on a complete state resume is a strict no-op -- it returns
    // the same outcome AND the same certificate, so it must move neither the
    // budget nor the history.
    if (st_.outcome != CommutantOutcome::OverBudget) return st_.outcome;
    st_.budget = new_budget;
    st_.budget_history.push_back(new_budget);
    advance_();
    return st_.outcome;
}

CommutantCertificate CommutantRoute::certificate() const {
    CommutantCertificate c;
    c.cyclicity = cyc_;
    c.ell = ell_;
    c.D = D_;
    c.f = f_;
    c.dim_A = st_.dim_A;
    c.t_top = st_.t_top;
    c.s_soc = st_.s_soc;
    c.dim_soc_A = st_.dim_soc_A;
    c.socle_bound = st_.socle_bound;
    c.admits = st_.admits;
    c.forced = force_;
    c.outcome = st_.outcome;
    c.candidate_index = st_.candidate_index;
    c.a = st_.a;
    c.pi_a = st_.pi_a;
    c.inv_sizes = st_.inv_sizes;
    c.R = st_.R;
    c.u = st_.u;
    c.rank = st_.rank;
    c.dim_C = st_.dim_C;
    c.state = st_;
    return c;
}

// ===========================================================================
// The naive oracle
// ===========================================================================

uint32_t NaiveCommutantDim(const std::vector<ExtMatrix>& t, uint32_t D, uint32_t ell) {
    if (t.empty() || D == 0) return 0;
    const uint32_t k = static_cast<uint32_t>(t.size());
    // Row (p, i, j), column (r, c) -> the coefficient of X_{rc} in
    // (X t_p - t_p X)_{ij} = sum_c X_{ic} (t_p)_{cj} - sum_r (t_p)_{ir} X_{rj}.
    ExtMatrix sys(k * D * D, D * D, ell);
    for (uint32_t p = 0; p < k; ++p) {
        for (uint32_t i = 0; i < D; ++i) {
            for (uint32_t j = 0; j < D; ++j) {
                const uint32_t row = p * D * D + i * D + j;
                for (uint32_t c = 0; c < D; ++c) {
                    const uint32_t col = i * D + c;
                    const uint64_t acc =
                        static_cast<uint64_t>(sys.data[row][col]) + t[p].data[c][j];
                    sys.data[row][col] = static_cast<uint32_t>(acc % ell);
                }
                for (uint32_t r = 0; r < D; ++r) {
                    const uint32_t col = r * D + j;
                    const uint32_t v = t[p].data[i][r];
                    const uint64_t acc = static_cast<uint64_t>(sys.data[row][col])
                                       + ((v == 0u) ? 0u : (ell - v));
                    sys.data[row][col] = static_cast<uint32_t>(acc % ell);
                }
            }
        }
    }
    return hecke::KernelAsMatrix(sys).cols;
}

// ===========================================================================
// The route-combining verdict
// ===========================================================================

GenerationVerdict generation_verdict(bool cyclic_passed,
                                     CommutantOutcome b,
                                     SturmRouteStatus c) {
    GenerationVerdict v;
    if (cyclic_passed) {
        v.kind = GenerationKind::Cyclic;
        v.final_value = SturmFinal::True;
        return v;
    }
    if (b == CommutantOutcome::Passed) {
        v.kind = GenerationKind::Commutant;
        v.final_value = SturmFinal::True;
        return v;
    }
    if (c == SturmRouteStatus::Passed) {
        v.kind = GenerationKind::Sturm;
        v.final_value = SturmFinal::True;
        return v;
    }
    // Nothing certified.  OQ8 / [OD]: the ideal is EMITTED with `final = false`
    // and a reason from R3.2's fixed list, never withheld.
    v.kind = GenerationKind::Unavailable;
    v.final_value = SturmFinal::False;
    if (c == SturmRouteStatus::Failed) {
        v.reason = "certificate_failed:sturm";
    } else if (b == CommutantOutcome::OverBudget) {
        v.reason = "budget_exhausted:commutant";
    } else if (b == CommutantOutcome::ExcludedBySocleBound || b == CommutantOutcome::Failed) {
        // The socle-bound proof that (b) fails, or a computed dim C > dim A_S:
        // either way the ideal CANNOT reach cyclic/commutant closure, and (c)
        // is not on its proof range.
        v.reason = "closure_unreachable";
    } else {
        v.reason = "sturm_unavailable";
    }
    return v;
}

} // namespace hecke_ext
