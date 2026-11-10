// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 5, WP-8 -- see HeckeDicksonMultiplication.h for the
// binding contract (the object, the det-twist (*), and why it is FlatMatrix).

#include "HeckeDicksonMultiplication.h"

#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/ModCtx.h"

#include <sstream>

namespace hecke_ext {

namespace {

// The coefficient vector of a homogeneous polynomial of degree d: entry k is
// the coefficient of x^{d-k} y^k, i.e. the coordinate of u^{(d)}_k.
using Homog = std::vector<uint32_t>;
using Ctx = engine::ModCtx<uint32_t>;

// L = x^ell y - x y^ell as a degree-(ell+1) coefficient vector over F_ell.
Homog DicksonCoefficients(const Ctx& m) {
    const uint32_t ell = m.p;
    Homog c(ell + 2, 0);
    c[1] = 1 % ell;              // + x^ell y
    c[ell] = m.neg(1 % ell);     // - x y^ell
    return c;
}

// The GENERIC product of two homogeneous coefficient vectors (a plain
// convolution).  This is verify()'s independent route to a column of L: the
// constructor writes two indices, this multiplies the whole polynomial.
Homog HomogeneousProduct(const Homog& f, const Homog& g, const Ctx& m) {
    if (f.empty() || g.empty()) return Homog();
    Homog out(f.size() + g.size() - 1, 0);
    for (size_t i = 0; i < f.size(); ++i) {
        if (f[i] == 0) continue;
        for (size_t j = 0; j < g.size(); ++j) {
            if (g[j] == 0) continue;
            out[i + j] = m.add(out[i + j], m.mul(f[i], g[j]));
        }
    }
    return out;
}

// A deterministic 64-bit mixer (splitmix64), so that the sample list is a pure
// function of (n, ell, num_samples) and two runs certify the same matrices.
uint64_t Mix(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

FlatMatrix Two(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t ell) {
    return FlatMatrix(2, 2, ell, std::vector<uint32_t>{a, b, c, d});
}

} // namespace

// ===========================================================================
// The matrix.
// ===========================================================================
FlatMatrix DicksonMultiplicationMatrix(uint32_t n, uint32_t ell) {
    if (!DicksonMultiplicationApplicable(n, ell)) return FlatMatrix();
    ENGINE_CHECK_MSG(ell >= 2, "DicksonMultiplicationMatrix: ell must be >= 2");

    const Ctx m(ell);
    const uint32_t rows = n + 1;
    const uint32_t cols = n - ell;          // = m + 1, m = n - ell - 1
    const uint32_t one = 1 % ell;
    const uint32_t minus_one = m.neg(one);

    std::vector<uint32_t> flat(static_cast<size_t>(rows) * cols, 0);
    for (uint32_t j = 0; j < cols; ++j) {
        // column j = u^{(n)}_{j+1} - u^{(n)}_{j+ell}
        const uint32_t r_plus = j + 1;
        const uint32_t r_minus = j + ell;
        ENGINE_CHECK(r_plus < rows && r_minus < rows && r_plus != r_minus);
        flat[static_cast<size_t>(r_plus) * cols + j] =
            m.add(flat[static_cast<size_t>(r_plus) * cols + j], one);
        flat[static_cast<size_t>(r_minus) * cols + j] =
            m.add(flat[static_cast<size_t>(r_minus) * cols + j], minus_one);
    }
    return FlatMatrix(rows, cols, ell, flat);
}

// ===========================================================================
// The certificate.
// ===========================================================================
DicksonMultiplicationCertificate CertifyDicksonMultiplication(uint32_t n, uint32_t ell,
                                                              uint32_t num_samples) {
    DicksonMultiplicationCertificate cert;
    cert.n = n;
    cert.ell = ell;
    cert.applicable = DicksonMultiplicationApplicable(n, ell);
    cert.m = DicksonSourceDegree(n, ell);
    cert.rows = cert.applicable ? n + 1 : 0;
    cert.cols = cert.applicable ? n - ell : 0;
    cert.L = DicksonMultiplicationMatrix(n, ell);
    if (!cert.applicable) return cert;

    const Ctx mc(ell);
    const uint32_t one = 1 % ell;
    const uint32_t minus_one = mc.neg(one);

    // The six FIXED samples, in this order:
    //   I, -I, S = [[0,-1],[1,0]], T = [[1,1],[0,1]],
    //   and two SINGULAR ones -- [[1,1],[1,1]] (det 0) and [[0,1],[0,0]]
    //   (nilpotent) -- because (*) is proved with no determinant hypothesis
    //   and must therefore hold there too.
    const uint32_t fixed[6][4] = {
        {one, 0, 0, one},
        {minus_one, 0, 0, minus_one},
        {0, minus_one, one, 0},
        {one, one, 0, one},
        {one, one, one, one},
        {0, one, 0, 0},
    };
    for (const auto& f : fixed) {
        for (uint32_t k = 0; k < 4; ++k) cert.samples.push_back(f[k]);
    }

    // The seed is a pure function of (n, ell), so two runs -- and two
    // processes -- certify the SAME samples.
    uint64_t state = (static_cast<uint64_t>(n) << 32) ^ static_cast<uint64_t>(ell)
                     ^ 0x9e3779b97f4a7c15ull;
    for (uint32_t s = 0; s < num_samples; ++s) {
        for (uint32_t k = 0; k < 4; ++k) {
            cert.samples.push_back(static_cast<uint32_t>(Mix(state) % ell));
        }
    }
    return cert;
}

uint32_t DicksonMultiplicationCertificate::verify() const {
    // --- 1: applicability re-derived from (n, ell) alone. -----------------
    if (ell < 2) return 1;
    if (applicable != DicksonMultiplicationApplicable(n, ell)) return 1;
    if (m != DicksonSourceDegree(n, ell)) return 1;
    if (!applicable) {
        return (L.get_rows() == 0 && L.get_cols() == 0 && samples.empty()) ? 0u : 1u;
    }

    // --- 2: the shape and the modulus. ------------------------------------
    if (rows != n + 1 || cols != n - ell) return 2;
    if (L.get_rows() != rows || L.get_cols() != cols || L.get_p() != ell) return 2;

    // --- 3: every entry, by an independent convolution. --------------------
    //
    // Column j of L. is the coefficient vector of  L . u^{(m)}_j, which
    // HomogeneousProduct computes from the FULL coefficient vectors -- never
    // from the "+1 at j+1, -1 at j+ell" rule the constructor used.
    {
        const Ctx mc(ell);
        const Homog Lc = DicksonCoefficients(mc);
        for (uint32_t j = 0; j < cols; ++j) {
            Homog uj(m + 1, 0);
            uj[j] = 1 % ell;
            const Homog prod = HomogeneousProduct(Lc, uj, mc);
            if (prod.size() != rows) return 3;
            for (uint32_t r = 0; r < rows; ++r) {
                if (L.get_entry(r, j) != prod[r]) return 3;
            }
        }
    }

    // --- 4: injectivity. ---------------------------------------------------
    if (L.rank() != cols) return 4;

    // --- 5: the sample list. -----------------------------------------------
    if (samples.empty() || samples.size() % 4 != 0) return 5;
    for (uint32_t v : samples) {
        if (v >= ell) return 5;
    }

    // --- 6: the det-twisted intertwining (*), entrywise. -------------------
    //
    //     rho_n(A) . L  ==  det(A) . L . rho_m(A)
    //
    // with rho computed by the pipeline's own ComputeSymmetricPowerMatrixFM,
    // i.e. in the action convention of [C, def:rhon].
    for (uint32_t s = 0; s + 3 < samples.size(); s += 4) {
        const uint32_t a = samples[s], b = samples[s + 1];
        const uint32_t c = samples[s + 2], d = samples[s + 3];
        const FlatMatrix A = Two(a, b, c, d, ell);
        const Ctx mc(ell);
        const uint32_t det = mc.sub(mc.mul(a, d), mc.mul(b, c));
        const FlatMatrix rho_n = ComputeSymmetricPowerMatrixFM(A, n);
        const FlatMatrix rho_m = ComputeSymmetricPowerMatrixFM(A, m);
        const FlatMatrix lhs = rho_n * L;
        const FlatMatrix rhs = (L * rho_m) * det;
        if (lhs.get_rows() != rhs.get_rows() || lhs.get_cols() != rhs.get_cols()) return 6;
        for (uint32_t r = 0; r < lhs.get_rows(); ++r) {
            for (uint32_t cc = 0; cc < lhs.get_cols(); ++cc) {
                if (lhs.get_entry(r, cc) != rhs.get_entry(r, cc)) return 6;
            }
        }
    }
    return 0;
}

const char* DicksonMultiplicationCertificate::verify_message(uint32_t code) {
    switch (code) {
        case 0: return "ok";
        case 1: return "applicability disagrees with (n, ell)";
        case 2: return "the shape or the modulus is not (n+1) x (n-ell) over F_ell";
        case 3: return "an entry differs from the independently convolved column";
        case 4: return "L. is not injective (rank != n - ell)";
        case 5: return "no samples, or a malformed sample list";
        case 6: return "rho_n(A) L != det(A) L rho_m(A) on some sample";
        default: return "unknown code";
    }
}

std::string DicksonMultiplicationCertificate::to_string_offset(uint32_t offset) const {
    const std::string ind(offset, ' ');
    std::ostringstream os;
    os << ind << "DicksonMultiplicationCertificate (n=" << n << ", ell=" << ell
       << ", m=" << m << ")\n";
    os << ind << "  applicable: " << (applicable ? "yes" : "no")
       << "  shape: " << rows << " x " << cols
       << "  samples: " << num_samples() << "\n";
    return os.str();
}

} // namespace hecke_ext
