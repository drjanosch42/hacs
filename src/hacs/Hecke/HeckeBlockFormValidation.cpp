// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S10V -- see HeckeBlockFormValidation.h for the independence
// statement and the full oracle list (O6, O8).
//
// =====================================================================
// INDEPENDENT DERIVATION OF THE EXPECTED VALUES (read before the numbers
// below: none of them are read off HeckeModuleDecomposition's own smoke
// test HeckeDecompositionSmokeValidation.cpp -- only the PUBLIC headers
// HeckeModuleDecomposition.h / FpPoly.h / HeckeOperatorOnK.h /
// HeckeCosetTables.h / HeckeTauTable.h / EquivariantCohomologyOnK.h /
// LocalSystemSL2Z.h / Matrix.h were consulted for API signatures).
//
// (1) The n=10 (weight 12) Hecke datum: S_12 = C*Delta, a_2(Delta) = -24
//     (classical, Ramanujan's tau function). The Eisenstein eigenvalue in
//     weight k = n+2 is 1+p^{k-1} = 1+p^{n+1} (Diamond-Shurman / Serre); at
//     p=2, n=10 this is 1+2^11 = 2049. Char-0 Eichler-Shimura (reduced mod
//     ell, no Tor obstruction for ell >= 5, [C, obl:oracle]) therefore gives
//     charpoly(A^(1)(T_2)) = (X+24)^2 (X-2049) mod ell for ell != 691 (the
//     square because a cuspidal eigenform contributes a
//     holomorphic+antiholomorphic pair sharing the SAME T_2-eigenvalue).
//
// (2) The Ramanujan congruence tau(p) === p^11+1 (mod 691) (proved
//     classically): 2049 = 1+2^11 and -24 = a_2(Delta); 2049-(-24) = 2073 =
//     3*691, i.e. 2049 === -24 (mod 691) -- re-verified by direct integer
//     arithmetic in RunO6() below, not merely asserted. Hence
//     charpoly(A^(1)(T_2)) === (X+24)^3 (mod 691): the Eisenstein root and
//     the (doubled) cuspidal root collide. The task's O6 (restating
//     amendment A-F2 / Deng, JNT 2023) asserts this collision is a
//     NONSPLIT extension: the cuspidal H^1_! carries T_2 as an honest
//     integral SCALAR -24*I_2 (mult 2, holo+antiholo -- a_2(Delta) has no
//     room for anything else, being a single rational integer), so ALL of
//     the non-scalar defect of A^(1)(T_2) (which must be nonzero, since
//     A^(1)(T_2) != -24*I by O6's own hard clause, already checked
//     independently by Stage S7V) is carried by the Eisenstein line gluing
//     nonsplit onto the cuspidal block -- i.e. rank(A+24I) is forced to be
//     EXACTLY 1 (a single Jordan block of size 2 glued to a size-1 scalar
//     block: partition (2,1) of the multiplicity-3 root -24, not (1,1,1)
//     [fully split/scalar, excluded by A != -24I] and not (3) [a single
//     Jordan block of size 3, which 2-dimensional scalar cuspidal data
//     cannot produce]). This file's job (A-F2) is to make this an
//     UNCONDITIONAL, machine-checked assertion, not merely a rationale.
//
// (3) The n=22 (weight 24) Hecke datum: dim S_24 = 2, spanned by a SINGLE
//     Galois orbit of size 2 (no rational eigenform at level 1, weight 24 --
//     classical fact, LMFDB label 1.24.a.a has Hecke field degree 2). The
//     task's O8 supplies the trace/norm of T_2 on S_24 directly:
//     tr(T_2|S_24) = 1080, det(T_2|S_24) = -20468736, so the minimal
//     polynomial of T_2 on S_24 is X^2-1080X-20468736 (irreducible over Q);
//     Eichler-Shimura doubles this (holo+antiholo) to get a degree-4
//     factor (X^2-1080X-20468736)^2 in charpoly(A^(1)(T_2)), and the
//     Eisenstein eigenvalue in weight 24 is 1+2^23 = 8388609. Hence
//     charpoly(A^(1)(T_2)) = (X^2-1080X-20468736)^2 (X-8388609) mod ell,
//     dim H^1 = 2*2+1 = 5 -- exactly the task's O8 restatement, itself
//     re-derivable from Eichler-Shimura + the classical weight-24 Hecke
//     data (not copied from any file in this repository).
//
// (4) The mod-ell SPLITTING of the degree-2 factor is governed by the
//     Legendre symbol (144169|ell): the task states the discriminant of
//     X^2-1080X-20468736 is 1080^2+4*20468736 = 24^2*144169 (re-verified by
//     direct integer arithmetic below: 1080^2 = 1166400,
//     4*20468736 = 81874944, sum = 83041344 = 576*144169 = 24^2*144169 --
//     yes, matches). So the quadratic's discriminant is a square times
//     144169: the quadratic splits over F_ell (two distinct F_ell-rational
//     roots, residue degree 1 each) iff 144169 is a nonzero square mod ell,
//     i.e. the Legendre symbol (144169|ell) = +1, and stays irreducible
//     (residue degree 2) iff (144169|ell) = -1 (144169 is never 0 mod 101
//     or mod 10007: 144169 = 101*1427+42 and 144169 = 10007*14+27, both
//     nonzero remainders, re-verified below). This file computes the
//     Legendre symbol AT RUNTIME via Euler's criterion
//     (ModArithmetic::mod_pow(144169 mod ell, (ell-1)/2, ell)) and BRANCHES
//     on the result -- the branch actually taken by ell=101 and ell=10007
//     is discovered by this file's own code, never hard-coded.
//
// KEY SIMPLIFICATION (avoids needing a modular square root): rather than
// solving X^2-1080X-20468736 = 0 mod ell to name the two roots r1, r2 when
// the quadratic splits, this file instead (a) reads off each dim-2 block's
// reported char_poly as a monic quadratic (X-r)^2 = X^2-2rX+r^2, solves for
// r algebraically from the linear coefficient (r = -coeff[1]/2), (b)
// double-checks self-consistency (coeff[0] == r^2), and (c) substitutes r
// into the THEORETICAL polynomial X^2-1080X-20468736 (this file's own
// PolyEvalAt, on the embedded, independently-derived coefficients of (3)
// above) to confirm r is a genuine root of the theoretical Hecke
// polynomial -- and separately confirms the two blocks' extracted roots are
// pairwise distinct. This sidesteps Tonelli-Shanks entirely while still
// being a from-scratch, independent verification (not trusting
// HeckeModuleDecomposition's internal factorization machinery for
// anything beyond the raw reported coefficient lists, which are then
// cross-checked against independently-derived theory).
// =====================================================================

#include "HeckeBlockFormValidation.h"

#include "HeckeCosetTables.h"
#include "HeckeModuleDecomposition.h"
#include "HeckeOperatorOnK.h"
#include "HeckeTauTable.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// ===========================================================================
// A from-scratch F_ell[X] toolkit, LOCAL to this validation file and
// DELIBERATELY INDEPENDENT of hecke::FpPoly (Stage S10's own toolkit): this
// file must re-derive gcd/irreducibility/charpoly facts with its OWN code,
// never by calling hecke::CertifyIrreducible or
// HeckeModuleDecomposition::certify(). Polynomials are stored low -> high
// degree (constant term first), trimmed (highest stored coefficient
// nonzero; the zero polynomial is {}).
// ===========================================================================

using Poly = std::vector<uint32_t>;

uint32_t ModOfInt64(int64_t x, uint32_t ell) {
    int64_t m = x % static_cast<int64_t>(ell);
    if (m < 0) m += static_cast<int64_t>(ell);
    return static_cast<uint32_t>(m);
}

Poly PolyTrim(Poly p) {
    while (!p.empty() && p.back() == 0u) p.pop_back();
    return p;
}

int64_t PolyDegree(const Poly& p) { return static_cast<int64_t>(p.size()) - 1; }

Poly PolyAdd(const Poly& a, const Poly& b, uint32_t ell) {
    Poly r(std::max(a.size(), b.size()), 0u);
    for (size_t i = 0; i < a.size(); ++i) r[i] = ModArithmetic::mod_add(r[i], a[i], ell);
    for (size_t i = 0; i < b.size(); ++i) r[i] = ModArithmetic::mod_add(r[i], b[i], ell);
    return PolyTrim(r);
}

Poly PolySub(const Poly& a, const Poly& b, uint32_t ell) {
    Poly r(std::max(a.size(), b.size()), 0u);
    for (size_t i = 0; i < a.size(); ++i) r[i] = ModArithmetic::mod_add(r[i], a[i], ell);
    for (size_t i = 0; i < b.size(); ++i) r[i] = ModArithmetic::mod_sub(r[i], b[i], ell);
    return PolyTrim(r);
}

Poly PolyScale(const Poly& a, uint32_t c, uint32_t ell) {
    Poly r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = ModArithmetic::mod_mul(a[i], c, ell);
    return PolyTrim(r);
}

Poly PolyMul(const Poly& a, const Poly& b, uint32_t ell) {
    if (a.empty() || b.empty()) return {};
    Poly r(a.size() + b.size() - 1, 0u);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0u) continue;
        for (size_t j = 0; j < b.size(); ++j) {
            if (b[j] == 0u) continue;
            r[i + j] = ModArithmetic::mod_add(r[i + j], ModArithmetic::mod_mul(a[i], b[j], ell), ell);
        }
    }
    return PolyTrim(r);
}

Poly PolyPow(const Poly& base, uint32_t e, uint32_t ell) {
    Poly result{1u};
    Poly b = PolyTrim(base);
    uint32_t exp = e;
    while (exp > 0) {
        if (exp & 1u) result = PolyMul(result, b, ell);
        b = PolyMul(b, b, ell);
        exp >>= 1;
    }
    return PolyTrim(result);
}

bool PolyEqual(const Poly& a, const Poly& b) { return PolyTrim(a) == PolyTrim(b); }

// (X - root)^mult mod ell.
Poly PolyLinearFactorPow(int64_t root, uint32_t mult, uint32_t ell) {
    const Poly factor = PolyTrim(Poly{ModOfInt64(-root, ell), 1u});
    return PolyPow(factor, mult, ell);
}

uint32_t PolyEvalAt(const Poly& p, uint32_t x, uint32_t ell) {
    uint32_t acc = 0;
    for (size_t i = p.size(); i-- > 0; ) {
        acc = ModArithmetic::mod_add(ModArithmetic::mod_mul(acc, x, ell), p[i], ell);
    }
    return acc;
}

std::string PolyToString(const Poly& p) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < p.size(); ++i) {
        if (i) oss << ", ";
        oss << p[i];
    }
    oss << "] (coeffs low->high)";
    return oss.str();
}

// General polynomial division a = q*b + r, deg r < deg b, b != 0. Field
// division via mod_inv on b's leading coefficient (ell prime).
void PolyDivRem(const Poly& a_in, const Poly& b_in, uint32_t ell, Poly& q, Poly& r) {
    Poly a = PolyTrim(a_in);
    Poly b = PolyTrim(b_in);
    if (b.empty()) throw std::invalid_argument("PolyDivRem: division by the zero polynomial");
    const int64_t db = PolyDegree(b);
    const uint32_t lb_inv = ModArithmetic::mod_inv(b.back(), ell);
    r = a;
    q.assign(r.size() > b.size() ? r.size() - b.size() + 1 : 0, 0u);
    while (!r.empty() && PolyDegree(r) >= db) {
        const int64_t dr = PolyDegree(r);
        const uint32_t c = ModArithmetic::mod_mul(r.back(), lb_inv, ell);
        const size_t shift = static_cast<size_t>(dr - db);
        if (shift >= q.size()) q.resize(shift + 1, 0u);
        q[shift] = ModArithmetic::mod_add(q[shift], c, ell);
        for (size_t i = 0; i < b.size(); ++i) {
            r[shift + i] = ModArithmetic::mod_sub(r[shift + i], ModArithmetic::mod_mul(c, b[i], ell), ell);
        }
        r = PolyTrim(r);
    }
    q = PolyTrim(q);
}

Poly PolyMod(const Poly& a, const Poly& b, uint32_t ell) {
    Poly q, r;
    PolyDivRem(a, b, ell, q, r);
    return r;
}

Poly PolyGCD(Poly a, Poly b, uint32_t ell) {
    a = PolyTrim(a);
    b = PolyTrim(b);
    while (!b.empty()) {
        Poly r = PolyMod(a, b, ell);
        a = b;
        b = r;
    }
    if (a.empty()) return a;
    const uint32_t inv_lead = ModArithmetic::mod_inv(a.back(), ell);
    return PolyScale(a, inv_lead, ell);
}

// g^e mod f, square-and-multiply, from scratch (deliberately not
// hecke::FpPoly::powmod).
Poly PolyPowMod(const Poly& base, uint64_t e, const Poly& modulus, uint32_t ell) {
    Poly result{1u};
    Poly b = PolyMod(base, modulus, ell);
    uint64_t exp = e;
    while (exp > 0) {
        if (exp & 1ull) result = PolyMod(PolyMul(result, b, ell), modulus, ell);
        b = PolyMod(PolyMul(b, b, ell), modulus, ell);
        exp >>= 1;
    }
    return result;
}

// X^{ell^k} mod modulus via k successive ell-th-power steps -- a from-
// scratch Frobenius ladder (deliberately not hecke::FpPoly::frobenius_powmod).
Poly FrobeniusPower(uint32_t k, const Poly& modulus, uint32_t ell) {
    Poly cur = PolyMod(Poly{0u, 1u}, modulus, ell); // X mod modulus
    for (uint32_t i = 0; i < k; ++i) {
        cur = PolyPowMod(cur, ell, modulus, ell);
    }
    return cur;
}

// Deterministic, unconditional Rabin irreducibility test, FROM SCRATCH
// (never calls hecke::CertifyIrreducible): pi (monic, deg >= 1) is
// irreducible over F_ell iff gcd(pi, X^{ell^i}-X) == 1 for
// 1 <= i <= floor(deg/2), and X^{ell^deg} == X (mod pi).
bool IrreducibleTestMine(const Poly& pi_in, uint32_t ell) {
    const Poly pi = PolyTrim(pi_in);
    const int64_t deg = PolyDegree(pi);
    if (deg < 1) throw std::invalid_argument("IrreducibleTestMine: need deg(pi) >= 1");
    if (deg == 1) return true; // linear polynomials are trivially irreducible
    const int64_t half = deg / 2;
    for (int64_t i = 1; i <= half; ++i) {
        const Poly fi = FrobeniusPower(static_cast<uint32_t>(i), pi, ell);
        const Poly diff = PolySub(fi, Poly{0u, 1u}, ell); // X^{ell^i} - X
        const Poly g = PolyGCD(diff, pi, ell);
        if (!(g.size() == 1 && g[0] == 1u)) return false; // gcd != 1 => reducible
    }
    const Poly fd = FrobeniusPower(static_cast<uint32_t>(deg), pi, ell);
    const Poly xmodpi = PolyMod(Poly{0u, 1u}, pi, ell);
    return PolyEqual(fd, xmodpi);
}

// Recursive Leibniz/cofactor-expansion determinant of a matrix of
// polynomials over F_ell (division-free; valid for every prime ell,
// including ell <= matrix dimension). A from-scratch helper, independent
// of hecke::CharPolyBerkowitz.
Poly PolyMatDet(const std::vector<std::vector<Poly>>& M, uint32_t ell) {
    const size_t n = M.size();
    if (n == 0) return Poly{1u};
    if (n == 1) return M[0][0];
    Poly result;
    for (size_t j = 0; j < n; ++j) {
        std::vector<std::vector<Poly>> minor(n - 1, std::vector<Poly>(n - 1));
        for (size_t r = 1; r < n; ++r) {
            size_t cc = 0;
            for (size_t c = 0; c < n; ++c) {
                if (c == j) continue;
                minor[r - 1][cc++] = M[r][c];
            }
        }
        Poly term = PolyMul(M[0][j], PolyMatDet(minor, ell), ell);
        if (j % 2 == 1) term = PolyScale(term, ell - 1u, ell); // negate mod ell
        result = PolyAdd(result, term, ell);
    }
    return result;
}

// charpoly(A) = det(X*I - A), division-free, from scratch (independent of
// hecke::CharPolyBerkowitz).
Poly CharPolyMine(const Matrix& A, uint32_t ell) {
    const uint32_t n = A.rows;
    std::vector<std::vector<Poly>> M(n, std::vector<Poly>(n));
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t aij = A.data[i][j];
            const uint32_t neg_aij = (aij == 0u) ? 0u : (ell - aij);
            M[i][j] = (i == j) ? PolyTrim(Poly{neg_aij, 1u}) : PolyTrim(Poly{neg_aij});
        }
    }
    return PolyMatDet(M, ell);
}

bool IsBlockDiagonalMine(const Matrix& C, const std::vector<uint32_t>& dims) {
    uint32_t off = 0;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    for (uint32_t s : dims) {
        ranges.emplace_back(off, off + s);
        off += s;
    }
    if (off != C.rows || C.rows != C.cols) return false;
    for (uint32_t i = 0; i < C.rows; ++i) {
        uint32_t blo = 0, bhi = 0;
        for (const auto& rg : ranges) {
            if (i >= rg.first && i < rg.second) { blo = rg.first; bhi = rg.second; break; }
        }
        for (uint32_t j = 0; j < C.cols; ++j) {
            if ((j < blo || j >= bhi) && C.data[i][j] != 0) return false;
        }
    }
    return true;
}

// Legendre symbol (a|ell) via Euler's criterion, computed AT RUNTIME
// (ModArithmetic::mod_pow), never hard-coded. Returns +1 or -1; throws if
// a === 0 (mod ell) (not expected for the settings used here -- checked).
int LegendreSymbolMine(uint32_t a, uint32_t ell) {
    const uint32_t a_mod = a % ell;
    if (a_mod == 0) throw std::invalid_argument("LegendreSymbolMine: a === 0 (mod ell)");
    const uint32_t r = ModArithmetic::mod_pow(a_mod, (ell - 1) / 2, ell);
    if (r == 1u) return 1;
    if (r == ell - 1u) return -1;
    throw std::runtime_error("LegendreSymbolMine: Euler's criterion returned neither 1 nor ell-1 "
                             "-- ell is not an odd prime, or an internal error occurred");
}

// ===========================================================================
// Pipeline fixtures, EXACTLY the construction the task prescribes: K =
// CreateSL2ZFundamentalDomainTriangulation(); base L = Sym^n(F_ell^2) via
// GenerateSL2ZRelationsWithGroupData(K,n,ell); EquivariantCohomologyOnK;
// HeckeCosetTables(2,1,ell,n); HeckeTauTable; HeckeOperatorOnK::compute();
// A = get_H_matrices()[1].
// ===========================================================================

struct CohomologyFixture {
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;

    CohomologyFixture(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell)
        : lssc(K, MakeRelations(K, n_sym, ell), n_sym + 1, ell), coh(&lssc) {
        coh.compute_cohomology();
    }

private:
    static std::vector<LocalSystemSimplicialComplex::SimplexRelation> MakeRelations(
            const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
            rels.push_back(datum.rel);
        }
        return rels;
    }
};

struct HeckeFixture {
    hecke::HeckeCosetTables cosets;
    hecke::HeckeTauTable tau;
    hecke::HeckeOperatorOnK op;

    HeckeFixture(const SimplicialComplex& K, const hecke::SL2ZTessellationTables& model,
                 EquivariantCohomologyOnK& coh, uint64_t p_hecke, uint32_t ell, uint32_t n_sym)
        : cosets(p_hecke, /*r=*/1, ell, n_sym),
          tau(K, model, cosets, ell, n_sym),
          op(&coh, &cosets, &tau) {
        op.compute();
    }
};

// ===========================================================================
// Shared diagnostic + reconstruction/certificate-re-verification machinery.
// ===========================================================================

class BlockFormContext {
public:
    uint32_t failures() const { return failures_; }

    void Fail(const std::string& msg) {
        ++failures_;
        std::cout << "HeckeBlockFormValidation: FAIL: " << msg << std::endl;
    }

    void Ok(const std::string& msg) {
        std::cout << "HeckeBlockFormValidation: OK: " << msg << std::endl;
    }

    // Independently re-verifies the reconstruction identity and the
    // block-diagonal shape for a computed decomposition, WITHOUT calling
    // decomp.certify(): fresh Matrix::inverse() of U (not the class's
    // cached U_inv_), U * conjugated(p) * U^{-1} == A recomputed here,
    // conjugated(p) exactly block-diagonal at the reported sizes (this
    // file's own IsBlockDiagonalMine), and sum(block dims) == d.
    // Returns false (and calls Fail()) on any discrepancy.
    bool ReconstructAndCheck(const hecke::HeckeModuleDecomposition& decomp, uint64_t p,
                              const Matrix& A, const std::string& tag) {
        const auto& blocks = decomp.blocks();
        const uint32_t d = A.rows;

        uint32_t total = 0;
        std::vector<uint32_t> dims;
        for (const auto& b : blocks) { total += b.dim; dims.push_back(b.dim); }
        if (total != d) {
            std::ostringstream oss;
            oss << "[" << tag << "] sum of block dims (" << total << ") != dim H^q (" << d << ")";
            Fail(oss.str());
            return false;
        }

        Matrix U = decomp.change_of_basis();
        if (U.rows != d || U.cols != d) {
            Fail(tag + ": change_of_basis() has unexpected shape");
            return false;
        }
        Matrix Ucopy = U;
        if (!Ucopy.is_invertible()) {
            Fail(tag + ": change_of_basis() U is not invertible (det U == 0)");
            return false;
        }
        // Fresh inverse -- independent of HeckeModuleDecomposition's
        // internally cached U_inv_.
        Matrix Uinv = Ucopy.inverse();
        Matrix identity(d, A.p);
        if (!((U * Uinv) == identity)) {
            Fail(tag + ": freshly recomputed U * U^{-1} != I");
            return false;
        }

        // Recompute U^{-1} A U ourselves and cross-check against
        // conjugated(p) (which the class computes with its OWN cached
        // inverse) -- this is a genuine independent cross-check, not a
        // tautology.
        Matrix C_mine = Uinv * A * U;
        Matrix C_reported = decomp.conjugated(p);
        if (!(C_mine == C_reported)) {
            std::ostringstream oss;
            oss << "[" << tag << "] independently recomputed U^{-1} A U != conjugated(p) "
                << "(reported)\n  mine:\n" << C_mine.to_string()
                << "  reported:\n" << C_reported.to_string();
            Fail(oss.str());
            return false;
        }

        if (!IsBlockDiagonalMine(C_mine, dims)) {
            Fail(tag + ": U^{-1} A U (recomputed independently) is NOT exactly block-diagonal "
                       "at the reported block sizes");
            return false;
        }

        // Reconstruction: U * conjugated(p) * U^{-1} == A EXACTLY.
        Matrix recon = U * C_mine * Uinv;
        if (!(recon == A)) {
            std::ostringstream oss;
            oss << "[" << tag << "] reconstruction FAILED: U * conjugated(p) * U^{-1} != A\n"
                << "  A =\n" << A.to_string() << "  U*C*U^-1 =\n" << recon.to_string();
            Fail(oss.str());
            return false;
        }

        Ok(tag + ": reconstruction exact (det U != 0, U*conjugated(p)*U^{-1} == A, "
                  "block-diagonal shape confirmed independently)");
        return true;
    }

private:
    uint32_t failures_ = 0;
};

} // namespace

// ===========================================================================
// O6 -- n=10, ell=691, p=2 (Ramanujan-congruence non-semisimple case;
// amendment A-F2, HARD).
// ===========================================================================

namespace {

void RunO6(BlockFormContext& ctx) {
    const uint32_t n = 10, ell = 691;
    const uint64_t p_hecke = 2;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        CohomologyFixture cf(K, n, ell);
        const auto& dims = cf.coh.get_H_eq_dimensions();
        if (dims.size() < 2 || dims[1] != 3) {
            std::ostringstream oss;
            oss << "O6: dim H^1 mismatch at n=10, ell=691: got "
                << (dims.size() >= 2 ? std::to_string(dims[1]) : std::string("<missing>"))
                << ", expected 3";
            ctx.Fail(oss.str());
            return;
        }

        const hecke::SL2ZTessellationTables model(K);
        HeckeFixture hf(K, model, cf.coh, p_hecke, ell, n);
        const auto& H = hf.op.get_H_matrices();
        if (H.size() < 2 || H[1].rows != 3 || H[1].cols != 3) {
            ctx.Fail("O6: A^(1)(T_2) has unexpected shape");
            return;
        }
        const Matrix A = H[1];

        // Independent re-derivation check: 2049 - (-24) must be an exact
        // multiple of 691 (the Ramanujan congruence itself), re-verified by
        // direct integer arithmetic, not merely quoted.
        if ((2049 - (-24)) % 691 != 0) {
            ctx.Fail("O6: internal derivation error -- 2049+24 is not a multiple of 691");
            return;
        }

        // Independent charpoly(A) via this file's OWN division-free
        // cofactor expansion (never hecke::CharPolyBerkowitz).
        const Poly cpA = CharPolyMine(A, ell);
        const Poly expected_cp = PolyLinearFactorPow(-24, 3, ell); // (X+24)^3 mod 691
        if (!PolyEqual(cpA, expected_cp)) {
            std::ostringstream oss;
            oss << "O6: charpoly(A^(1)(T_2)) mismatch. A =\n" << A.to_string()
                << "computed charpoly = " << PolyToString(cpA)
                << "\nexpected (X+24)^3   = " << PolyToString(expected_cp);
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok("O6: charpoly(A^(1)(T_2)) == (X+24)^3 mod 691, independently recomputed");

        // HARD (amendment A-F2): rank(A + 24*I) == 1, computed directly on
        // A (not through HeckeModuleDecomposition). diff = A - (-24*I).
        Matrix twentyFourI = Matrix(3, ell) * ModOfInt64(24, ell); // 24*I
        Matrix diff = A + twentyFourI; // A + 24I
        if (diff.is_zero()) {
            ctx.Fail("O6: HARD FAILURE: A^(1)(T_2) == -24*I mod 691 (rank(A+24I) would be 0; "
                     "the Eisenstein/cuspidal extension is expected NONSPLIT). A =\n" + A.to_string());
            return;
        }
        Matrix diff_copy = diff;
        const uint32_t rk = diff_copy.rank();
        if (rk != 1) {
            std::ostringstream oss;
            oss << "O6: HARD FAILURE (amendment A-F2): rank(A+24I) = " << rk
                << ", expected EXACTLY 1 (Jordan partition (2,1)). This is a GENUINE "
                   "MATHEMATICAL SURPRISE per the task's instructions -- reporting, not "
                   "patching. A =\n" << A.to_string() << "A+24I =\n" << diff.to_string();
            ctx.Fail(oss.str());
            return;
        }
        // Direct nilpotency cross-check: (A+24I)^2 == 0 confirms minpoly is
        // (X+24)^2 and not (X+24)^3 (independent of the rank computation).
        Matrix diff2 = diff * diff;
        if (!diff2.is_zero()) {
            ctx.Fail("O6: (A+24I)^2 != 0 mod 691 -- contradicts rank(A+24I)=1 / partition (2,1). "
                     "A+24I =\n" + diff.to_string());
            return;
        }
        ctx.Ok("O6 HARD PASS: rank(A+24I) = 1 and (A+24I)^2 == 0, (A+24I) != 0 -- "
               "Jordan partition (2,1) confirmed by direct matrix computation on A alone.");

        // Now exercise HeckeModuleDecomposition and RE-VERIFY every one of
        // its reported claims with this file's own code (never certify()).
        std::vector<std::pair<uint64_t, Matrix>> ops = {{p_hecke, A}};
        hecke::HeckeModuleDecomposition decomp(ops, ell);
        decomp.compute();
        const auto& blocks = decomp.blocks();
        if (blocks.size() != 1) {
            std::ostringstream oss;
            oss << "O6: expected EXACTLY ONE primary block, got " << blocks.size();
            ctx.Fail(oss.str());
            return;
        }
        const hecke::PrimaryBlock& blk = blocks[0];
        if (blk.dim != 3) { ctx.Fail("O6: block dim != 3, got " + std::to_string(blk.dim)); return; }
        if (blk.residue_degree != 1) {
            ctx.Fail("O6: block residue_degree != 1, got " + std::to_string(blk.residue_degree));
            return;
        }

        const Poly reported_cp = blk.char_poly.at(p_hecke).coeffs;
        if (!PolyEqual(reported_cp, expected_cp)) {
            ctx.Fail("O6: reported block char_poly != (X+24)^3: " + PolyToString(reported_cp));
            return;
        }
        const Poly expected_mp = PolyLinearFactorPow(-24, 2, ell); // (X+24)^2
        const Poly reported_mp = blk.min_poly.at(p_hecke).coeffs;
        if (!PolyEqual(reported_mp, expected_mp)) {
            std::ostringstream oss;
            oss << "O6: HARD FAILURE (A-F2): reported min_poly != (X+24)^2, got "
                << PolyToString(reported_mp);
            ctx.Fail(oss.str());
            return;
        }
        const auto& part = blk.jordan_partition.at(p_hecke);
        if (!(part.size() == 2 && part[0] == 2 && part[1] == 1)) {
            std::ostringstream oss;
            oss << "O6: HARD FAILURE (A-F2): reported Jordan partition != (2,1), got (";
            for (size_t t = 0; t < part.size(); ++t) { if (t) oss << ","; oss << part[t]; }
            oss << ")";
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok("O6: HeckeModuleDecomposition reports dim=3, residue_degree=1, "
               "char_poly=(X+24)^3, min_poly=(X+24)^2, partition=(2,1) -- all independently "
               "cross-checked against the direct rank/nilpotency computation above.");

        // Re-verify irreducibility of pi = X+24 with THIS FILE'S OWN
        // gcd/Frobenius test (never hecke::CertifyIrreducible).
        const Poly pi = PolyTrim(Poly{ModOfInt64(24, ell), 1u}); // X+24
        if (!IrreducibleTestMine(pi, ell)) {
            ctx.Fail("O6: OWN irreducibility test failed on pi=X+24 (should be trivially "
                     "irreducible, deg 1)");
            return;
        }
        ctx.Ok("O6: pi = X+24 independently re-verified irreducible (own gcd/Frobenius test).");

        // Reconstruction + block-diagonal shape, independent of certify().
        ctx.ReconstructAndCheck(decomp, p_hecke, A, "O6");

    } catch (const std::exception& e) {
        ctx.Fail(std::string("O6: EXCEPTION: ") + e.what());
    }
}

} // namespace

// ===========================================================================
// O8 -- n=22, ell in {101, 10007}, p=2 (weight-24 Galois-conjugate-pair
// split-vs-nonsplit branch).
// ===========================================================================

namespace {

void RunO8ForEll(BlockFormContext& ctx, uint32_t ell) {
    const uint32_t n = 22;
    const uint64_t p_hecke = 2;
    const std::string tag = "O8(ell=" + std::to_string(ell) + ")";
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        CohomologyFixture cf(K, n, ell);
        const auto& dims = cf.coh.get_H_eq_dimensions();
        if (dims.size() < 2 || dims[1] != 5) {
            std::ostringstream oss;
            oss << tag << ": dim H^1 mismatch: got "
                << (dims.size() >= 2 ? std::to_string(dims[1]) : std::string("<missing>"))
                << ", expected 5";
            ctx.Fail(oss.str());
            return;
        }

        const hecke::SL2ZTessellationTables model(K);
        HeckeFixture hf(K, model, cf.coh, p_hecke, ell, n);
        const auto& H = hf.op.get_H_matrices();
        if (H.size() < 2 || H[1].rows != 5 || H[1].cols != 5) {
            ctx.Fail(tag + ": A^(1)(T_2) has unexpected shape");
            return;
        }
        const Matrix A = H[1];

        // Independent re-derivation checks (see the file header (3)/(4)):
        // discriminant identity and non-vanishing of 144169 mod ell.
        const int64_t disc = 1080LL * 1080LL + 4LL * 20468736LL;
        if (disc != 576LL * 144169LL) {
            ctx.Fail(tag + ": internal derivation error -- discriminant identity 1080^2+4*20468736 "
                            "!= 24^2*144169 does not hold");
            return;
        }
        if (144169u % ell == 0u) {
            ctx.Fail(tag + ": internal derivation error -- 144169 === 0 (mod ell), Legendre "
                            "symbol undefined at this ell");
            return;
        }

        // Independent theoretical charpoly: (X^2-1080X-20468736)^2 (X-8388609).
        const Poly quad = PolyTrim(Poly{ModOfInt64(-20468736, ell), ModOfInt64(-1080, ell), 1u});
        const Poly quad_sq = PolyMul(quad, quad, ell);
        const Poly eisenstein = PolyTrim(Poly{ModOfInt64(-8388609, ell), 1u}); // X-8388609
        const Poly expected_cp_full = PolyMul(quad_sq, eisenstein, ell);

        const Poly cpA = CharPolyMine(A, ell);
        if (!PolyEqual(cpA, expected_cp_full)) {
            std::ostringstream oss;
            oss << tag << ": charpoly(A^(1)(T_2)) mismatch. A =\n" << A.to_string()
                << "computed = " << PolyToString(cpA)
                << "\nexpected  = " << PolyToString(expected_cp_full);
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok(tag + ": charpoly(A^(1)(T_2)) == (X^2-1080X-20468736)^2 (X-8388609), "
                     "independently recomputed");

        // Runtime Legendre symbol computation -- SELECTS the branch, never
        // hard-coded.
        const int legendre = LegendreSymbolMine(144169u, ell);
        ctx.Ok(tag + ": Legendre symbol (144169|" + std::to_string(ell) + ") = " +
               std::to_string(legendre) + " (computed via Euler's criterion at runtime)");

        // Re-verify irreducibility/reducibility of the theoretical quadratic
        // with THIS FILE'S OWN gcd/Frobenius test, and cross-check it agrees
        // with the Legendre-symbol prediction (irreducible iff Legendre==-1).
        const bool quad_irreducible = IrreducibleTestMine(quad, ell);
        if (legendre == -1 && !quad_irreducible) {
            ctx.Fail(tag + ": Legendre symbol says -1 (should be irreducible) but OWN "
                            "irreducibility test says REDUCIBLE -- contradiction");
            return;
        }
        if (legendre == 1 && quad_irreducible) {
            ctx.Fail(tag + ": Legendre symbol says +1 (should split) but OWN irreducibility "
                            "test says IRREDUCIBLE -- contradiction");
            return;
        }
        ctx.Ok(tag + ": OWN gcd/Frobenius irreducibility test on X^2-1080X-20468736 agrees "
                     "with the Legendre-symbol prediction (irreducible == " +
               std::string(quad_irreducible ? "true" : "false") + ")");

        // Decompose and independently re-verify every reported claim.
        std::vector<std::pair<uint64_t, Matrix>> ops = {{p_hecke, A}};
        hecke::HeckeModuleDecomposition decomp(ops, ell);
        decomp.compute();
        const auto& blocks = decomp.blocks();

        uint32_t total_dim = 0;
        for (const auto& b : blocks) total_dim += b.dim;
        if (total_dim != 5) {
            ctx.Fail(tag + ": sum of reported block dims != 5, got " + std::to_string(total_dim));
            return;
        }

        // Locate the Eisenstein block (dim 1) -- expected in BOTH branches.
        const hecke::PrimaryBlock* eisen_blk = nullptr;
        std::vector<const hecke::PrimaryBlock*> others;
        for (const auto& b : blocks) {
            if (b.dim == 1) eisen_blk = &b; else others.push_back(&b);
        }
        if (eisen_blk == nullptr) {
            ctx.Fail(tag + ": no dim-1 (Eisenstein) block found among " +
                     std::to_string(blocks.size()) + " reported block(s)");
            return;
        }
        if (eisen_blk->residue_degree != 1) {
            ctx.Fail(tag + ": Eisenstein block residue_degree != 1, got " +
                     std::to_string(eisen_blk->residue_degree));
            return;
        }
        const Poly eisen_reported_cp = eisen_blk->char_poly.at(p_hecke).coeffs;
        const Poly eisen_reported_mp = eisen_blk->min_poly.at(p_hecke).coeffs;
        if (!PolyEqual(eisen_reported_cp, eisenstein) || !PolyEqual(eisen_reported_mp, eisenstein)) {
            std::ostringstream oss;
            oss << tag << ": Eisenstein block char/min-poly mismatch. char=" << PolyToString(eisen_reported_cp)
                << " min=" << PolyToString(eisen_reported_mp) << " expected=" << PolyToString(eisenstein);
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok(tag + ": Eisenstein block (dim 1, residue_degree 1, char=min=X-8388609) confirmed.");

        if (legendre == -1) {
            // --- Branch (144169|ell) = -1: one dim-4 block, residue 2. ---
            if (others.size() != 1) {
                std::ostringstream oss;
                oss << tag << ": Legendre=-1 branch: expected exactly one non-Eisenstein block, "
                            "got " << others.size();
                ctx.Fail(oss.str());
                return;
            }
            const hecke::PrimaryBlock& cusp_blk = *others[0];
            if (cusp_blk.dim != 4) {
                ctx.Fail(tag + ": Legendre=-1 branch: non-Eisenstein block dim != 4, got " +
                         std::to_string(cusp_blk.dim));
                return;
            }
            if (cusp_blk.residue_degree != 2) {
                ctx.Fail(tag + ": Legendre=-1 branch: non-Eisenstein block residue_degree != 2, "
                               "got " + std::to_string(cusp_blk.residue_degree));
                return;
            }
            const Poly cusp_cp = cusp_blk.char_poly.at(p_hecke).coeffs;
            if (!PolyEqual(cusp_cp, quad_sq)) {
                std::ostringstream oss;
                oss << tag << ": Legendre=-1 branch: dim-4 block char_poly != "
                            "(X^2-1080X-20468736)^2, got " << PolyToString(cusp_cp);
                ctx.Fail(oss.str());
                return;
            }
            // min_poly must be quad^e for e in {1,2} -- find which (RECORD,
            // per the task, not assert the specific e: semisimplicity of
            // the mod-ell reduction is not contract-backed).
            const Poly cusp_mp = cusp_blk.min_poly.at(p_hecke).coeffs;
            int found_e = -1;
            for (uint32_t e = 1; e <= 2; ++e) {
                if (PolyEqual(cusp_mp, PolyPow(quad, e, ell))) { found_e = static_cast<int>(e); break; }
            }
            if (found_e < 0) {
                std::ostringstream oss;
                oss << tag << ": Legendre=-1 branch: dim-4 block min_poly is not a power of "
                            "the theoretical quadratic (found neither quad^1 nor quad^2): "
                        << PolyToString(cusp_mp);
                ctx.Fail(oss.str());
                return;
            }
            ctx.Ok(tag + ": Legendre=-1 branch: dim-4 block, residue_degree=2, "
                         "char_poly=quad^2, min_poly=quad^" + std::to_string(found_e) +
                         " (Jordan structure RECORDED, not asserted, per the task).");

            // Re-verify the quadratic's irreducibility with OUR OWN test
            // (already done above as quad_irreducible == true, re-assert here
            // in context).
            if (!quad_irreducible) {
                ctx.Fail(tag + ": Legendre=-1 branch requires the quadratic irreducible, but "
                                "OWN test says reducible (contradiction, already flagged above)");
                return;
            }

            ctx.ReconstructAndCheck(decomp, p_hecke, A, tag);

        } else {
            // --- Branch (144169|ell) = +1: two dim-2 blocks, residue 1. ---
            if (others.size() != 2) {
                std::ostringstream oss;
                oss << tag << ": Legendre=+1 branch: expected exactly two non-Eisenstein blocks, "
                            "got " << others.size();
                ctx.Fail(oss.str());
                return;
            }
            std::vector<uint32_t> roots;
            for (const hecke::PrimaryBlock* bp : others) {
                const hecke::PrimaryBlock& b = *bp;
                if (b.dim != 2) {
                    ctx.Fail(tag + ": Legendre=+1 branch: non-Eisenstein block dim != 2, got " +
                             std::to_string(b.dim));
                    return;
                }
                if (b.residue_degree != 1) {
                    ctx.Fail(tag + ": Legendre=+1 branch: non-Eisenstein block residue_degree != 1, "
                                   "got " + std::to_string(b.residue_degree));
                    return;
                }
                const Poly cp = b.char_poly.at(p_hecke).coeffs;
                // Must be a monic quadratic (X-r)^2 = X^2 - 2r X + r^2.
                if (PolyDegree(cp) != 2 || cp.back() != 1u) {
                    ctx.Fail(tag + ": Legendre=+1 branch: dim-2 block char_poly is not a monic "
                                    "quadratic: " + PolyToString(cp));
                    return;
                }
                const uint32_t two_inv = ModArithmetic::mod_inv(2u, ell);
                const uint32_t r = ModArithmetic::mod_mul(
                    ModArithmetic::mod_sub(0u, cp[1], ell), two_inv, ell);
                const uint32_t r_sq = ModArithmetic::mod_mul(r, r, ell);
                if (cp[0] != r_sq) {
                    std::ostringstream oss;
                    oss << tag << ": Legendre=+1 branch: dim-2 block char_poly is not a perfect "
                                  "square (X-r)^2 for the r extracted from its linear "
                                  "coefficient: " << PolyToString(cp);
                    ctx.Fail(oss.str());
                    return;
                }
                // Confirm r is genuinely a root of the THEORETICAL quadratic
                // (independent re-derivation, sidesteps needing a modular
                // square root -- see file header).
                if (PolyEvalAt(quad, r, ell) != 0u) {
                    std::ostringstream oss;
                    oss << tag << ": Legendre=+1 branch: extracted root r=" << r
                        << " does NOT satisfy the theoretical quadratic "
                           "X^2-1080X-20468736 mod " << ell;
                    ctx.Fail(oss.str());
                    return;
                }
                roots.push_back(r);

                // Min poly must be (X-r)^e for e in {1,2}; record which.
                const Poly linear_r = PolyTrim(Poly{ModOfInt64(-static_cast<int64_t>(r), ell), 1u});
                const Poly mp = b.min_poly.at(p_hecke).coeffs;
                int found_e = -1;
                for (uint32_t e = 1; e <= 2; ++e) {
                    if (PolyEqual(mp, PolyPow(linear_r, e, ell))) { found_e = static_cast<int>(e); break; }
                }
                if (found_e < 0) {
                    ctx.Fail(tag + ": Legendre=+1 branch: dim-2 block min_poly is not a power of "
                                    "(X-r): " + PolyToString(mp));
                    return;
                }
                ctx.Ok(tag + ": Legendre=+1 branch: dim-2 block, residue_degree=1, root r=" +
                       std::to_string(r) + ", char_poly=(X-r)^2, min_poly=(X-r)^" +
                       std::to_string(found_e) + " (Jordan structure RECORDED, not asserted).");
            }
            if (roots.size() != 2 || roots[0] == roots[1]) {
                ctx.Fail(tag + ": Legendre=+1 branch: the two dim-2 blocks' extracted roots are "
                                "not distinct (expected the quadratic to split into two DISTINCT "
                                "F_ell-roots)");
                return;
            }
            // Product identity: (X-r1)^2 (X-r2)^2 == quad^2 (independent
            // cross-check that the two blocks together reconstitute the
            // theoretical squared quadratic).
            const Poly cp0 = others[0]->char_poly.at(p_hecke).coeffs;
            const Poly cp1 = others[1]->char_poly.at(p_hecke).coeffs;
            const Poly product = PolyMul(cp0, cp1, ell);
            if (!PolyEqual(product, quad_sq)) {
                std::ostringstream oss;
                oss << tag << ": Legendre=+1 branch: product of the two dim-2 blocks' char_polys "
                            "!= (X^2-1080X-20468736)^2. product=" << PolyToString(product);
                ctx.Fail(oss.str());
                return;
            }
            ctx.Ok(tag + ": Legendre=+1 branch: the two dim-2 blocks' char_polys multiply to "
                         "the theoretical (quad)^2, roots distinct and each independently "
                         "verified to satisfy the theoretical quadratic.");

            if (quad_irreducible) {
                ctx.Fail(tag + ": Legendre=+1 branch requires the quadratic reducible, but OWN "
                                "test says irreducible (contradiction, already flagged above)");
                return;
            }

            ctx.ReconstructAndCheck(decomp, p_hecke, A, tag);
        }

    } catch (const std::exception& e) {
        ctx.Fail(tag + ": EXCEPTION: " + e.what());
    }
}

void RunO8(BlockFormContext& ctx) {
    RunO8ForEll(ctx, 101u);
    RunO8ForEll(ctx, 10007u);
}

} // namespace

uint32_t HeckeBlockFormValidation() {
    try {
        BlockFormContext ctx;
        RunO6(ctx);
        RunO8(ctx);
        std::cout << "HeckeBlockFormValidation: " << ctx.failures() << " failing case(s)." << std::endl;
        return ctx.failures();
    } catch (const std::exception& e) {
        std::cout << "HeckeBlockFormValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
