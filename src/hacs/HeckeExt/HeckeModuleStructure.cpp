// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E2a (slot 4, WP-1) -- Stage 5, the module structure
// at m.  See HeckeModuleStructure.h for the mathematics and the [E] / [X]
// citations.
//
// INCLUDE DISCIPLINE ([X] Sec. 1.1; docs/matrix-engine/tools/
// check_heckeext_includes.sh rules 1, 2 and 4): its own header (hence
// HeckeLocalAlgebra.h, Hecke/SubspaceOps.h, Hecke/FpPoly.h,
// Hecke/HeckeModuleDecomposition.h and ModArithmetic's Matrix.h),
// ModArithmetic's EngineCheck.h, and C++ standard headers.  No Console/, no
// Cuda/, no Engine/, no BlockFlatMatrix.h, no `unordered_`.

#include "HeckeModuleStructure.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {

namespace {

using hecke::RrefRowSpan;

// --- small deterministic helpers -------------------------------------------

// The columns of `rows` (each of length D) as a D x rows.size() matrix -- i.e.
// a row-held subspace basis re-read as column vectors of V.
ExtMatrix ColumnsOfRows(const std::vector<std::vector<uint32_t>>& rows, uint32_t D, uint32_t ell) {
    const uint32_t k = static_cast<uint32_t>(rows.size());
    ExtMatrix out(D, k, ell);
    for (uint32_t j = 0; j < k; ++j) {
        ENGINE_CHECK_MSG(rows[j].size() == D, "HeckeModuleStructure: row of the wrong length");
        for (uint32_t i = 0; i < D; ++i) out.data[i][j] = rows[j][i];
    }
    return out;
}

// B * r, with `r` read as a column vector of V and the result returned as a
// row again.
std::vector<uint32_t> ApplyToRow(const ExtMatrix& B, const std::vector<uint32_t>& r) {
    const uint32_t D = B.rows;
    std::vector<uint32_t> out(D, 0);
    const uint64_t ell = B.p;
    for (uint32_t i = 0; i < D; ++i) {
        uint64_t acc = 0;
        for (uint32_t j = 0; j < B.cols; ++j) {
            if (B.data[i][j] == 0 || r[j] == 0) continue;
            acc = (acc + static_cast<uint64_t>(B.data[i][j]) * r[j]) % ell;
        }
        out[i] = static_cast<uint32_t>(acc);
    }
    return out;
}

ExtMatrix Transposed(const ExtMatrix& M) {
    ExtMatrix out(M.cols, M.rows, M.p);
    for (uint32_t i = 0; i < M.rows; ++i)
        for (uint32_t j = 0; j < M.cols; ++j) out.data[j][i] = M.data[i][j];
    return out;
}

// Horizontal concatenation of two matrices with the same row count.  (Either
// may have zero columns.)
ExtMatrix HCat(const ExtMatrix& A, const ExtMatrix& B, uint32_t rows, uint32_t ell) {
    ExtMatrix out(rows, A.cols + B.cols, ell);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < A.cols; ++j) out.data[i][j] = A.data[i][j];
        for (uint32_t j = 0; j < B.cols; ++j) out.data[i][A.cols + j] = B.data[i][j];
    }
    return out;
}

// The generalized-Jordan partition of a NILPOTENT N from its nullity sequence
// in units of f: n_r = (null_r - null_{r-1}) / f counts the parts >= r, and the
// partition is that sequence's conjugate.  This is the arithmetic of
// HeckeModuleDecomposition.cpp's file-local PartitionFromNullities ([X]
// Sec. 2.2 Stage 5, "fed to PartitionFromNullities (reused)"); it is
// re-derived here rather than shared because that function is file-local in a
// pipeline TU this layer must not reach into.
std::vector<uint32_t> PartitionFromNullitySequence(const std::vector<uint32_t>& nullities,
                                                   uint32_t f) {
    ENGINE_CHECK_MSG(!nullities.empty() && nullities[0] == 0 && f != 0,
                     "HeckeModuleStructure: malformed nullity sequence");
    const uint32_t e = static_cast<uint32_t>(nullities.size()) - 1;
    std::vector<uint32_t> n_r(e + 2, 0);
    for (uint32_t r = 1; r <= e; ++r) {
        ENGINE_CHECK_MSG(nullities[r] >= nullities[r - 1],
                         "HeckeModuleStructure: non-monotone nullity sequence");
        const uint32_t drop = nullities[r] - nullities[r - 1];
        ENGINE_CHECK_MSG(drop % f == 0,
                         "HeckeModuleStructure: a nullity drop is not a multiple of f");
        n_r[r] = drop / f;
        ENGINE_CHECK_MSG(r == 1 || n_r[r] <= n_r[r - 1],
                         "HeckeModuleStructure: non-monotone Jordan filtration");
    }
    std::vector<uint32_t> partition;
    for (uint32_t r = e; r >= 1; --r) {
        const uint32_t count = n_r[r] - n_r[r + 1];
        for (uint32_t c = 0; c < count; ++c) partition.push_back(r);
    }
    std::sort(partition.begin(), partition.end(), std::greater<uint32_t>());
    return partition;
}

bool SameMatrix(const ExtMatrix& a, const ExtMatrix& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.p != b.p) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j] != b.data[i][j]) return false;
    return true;
}

void AppendU32List(std::ostringstream& os, const std::vector<uint32_t>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) os << ","; os << v[i]; }
    os << "]";
}

void AppendU64List(std::ostringstream& os, const std::vector<uint64_t>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) os << ","; os << v[i]; }
    os << "]";
}

void AppendMatrix(std::ostringstream& os, const ExtMatrix& M) {
    os << "[";
    for (uint32_t r = 0; r < M.rows; ++r) {
        if (r) os << ",";
        os << "[";
        for (uint32_t c = 0; c < M.cols; ++c) { if (c) os << ","; os << M.data[r][c]; }
        os << "]";
    }
    os << "]";
}

// ---------------------------------------------------------------------------
// The whole of Stage 5, as a pure function of the CERTIFICATE (a) data.
//
// This is what makes verify() a genuine re-derivation ([X] Sec. 1.3 R17):
// ComputeModuleStructure() and ModuleStructure::verify() call the SAME
// function on the same input, and verify() then compares entrywise.  Nothing
// it concludes from is read back out of the stored object.
// ---------------------------------------------------------------------------
ModuleStructure Rebuild(const CyclicityCertificate& cyc) {
    ModuleStructure out;
    const uint32_t D = cyc.D, f = cyc.f, ell = cyc.ell;
    ENGINE_CHECK_MSG(D > 0 && f > 0 && ell >= 2,
                     "HeckeModuleStructure: malformed certificate dimensions");
    ENGINE_CHECK_MSG(cyc.t.size() == cyc.primes.size() &&
                     cyc.character.size() == cyc.primes.size(),
                     "HeckeModuleStructure: the certificate's operator lists disagree");

    out.ok = true;
    out.ell = ell;
    out.D = D;
    out.f = f;
    out.primes = cyc.primes;
    out.dim_A = static_cast<uint32_t>(cyc.a_basis.size());
    out.cyclicity = cyc;

    // The generators n_p = t_p - P_p(w_hat), re-derived, never read.
    const size_t k = cyc.primes.size();
    std::vector<ExtMatrix> n;
    n.reserve(k);
    for (size_t j = 0; j < k; ++j) {
        n.push_back(cyc.t[j] - hecke::EvalPolyAtMatrix(cyc.character[j], cyc.w_hat));
    }

    // --- (1) the Loewy filtration m^j V ----------------------------------
    std::vector<std::vector<std::vector<uint32_t>>> m_rows;   // per j, canonical rows
    {
        RrefRowSpan v_span(D, ell);
        for (uint32_t i = 0; i < D; ++i) {
            std::vector<uint32_t> e(D, 0);
            e[i] = 1;
            v_span.insert(e);
        }
        m_rows.push_back(v_span.rows());
    }
    while (!m_rows.back().empty()) {
        RrefRowSpan next(D, ell);
        for (const ExtMatrix& b : cyc.m_basis) {
            for (const std::vector<uint32_t>& r : m_rows.back()) next.insert(ApplyToRow(b, r));
        }
        // m_S is nilpotent on V, so this terminates; the guard is the
        // dimension, which must strictly drop once it is below D.
        ENGINE_CHECK_MSG(next.dim() < m_rows.back().size() || m_rows.back().empty(),
                         "HeckeModuleStructure: the Loewy filtration does not descend");
        m_rows.push_back(next.rows());
    }
    for (const auto& rows : m_rows) out.m_dims.push_back(static_cast<uint32_t>(rows.size()));
    for (size_t j = 0; j + 1 < m_rows.size(); ++j) {
        const uint32_t drop = out.m_dims[j] - out.m_dims[j + 1];
        ENGINE_CHECK_MSG(drop % f == 0,
                         "HeckeModuleStructure: a Loewy layer is not a K-subspace");
        out.loewy.push_back(drop / f);
    }

    // The canonical layer bases: the rows of m^j V, IN THE CANONICAL RREF
    // ORDER, that are independent modulo m^{j+1} V.
    for (size_t j = 0; j + 1 < m_rows.size(); ++j) {
        RrefRowSpan probe(D, ell);
        for (const std::vector<uint32_t>& r : m_rows[j + 1]) probe.insert(r);
        std::vector<std::vector<uint32_t>> reps;
        for (const std::vector<uint32_t>& r : m_rows[j]) {
            if (probe.insert(r)) reps.push_back(r);
        }
        ENGINE_CHECK_MSG(reps.size() == out.loewy[j] * f,
                         "HeckeModuleStructure: the layer basis has the wrong size");
        out.layer_basis.push_back(ColumnsOfRows(reps, D, ell));
    }
    for (const auto& rows : m_rows) out.m_basis_cols.push_back(ColumnsOfRows(rows, D, ell));

    // --- (1) the socle filtration ----------------------------------------
    // soc^{j+1} V = {x : n_a x in soc^j V for every a}, which is the preimage
    // of soc(V / soc^j V).  Held as a D x dim column matrix; soc^0 V = 0.
    out.socle_dims.push_back(0);
    {
        ExtMatrix s_cols(D, 0, ell);
        while (s_cols.cols < D) {
            // Q: rows spanning the annihilator of soc^j V, so that
            // `y in soc^j V <=> Q y = 0`.  (Once soc^j V is all of V the
            // filtration has stopped and Q would have no rows at all, so the
            // loop condition exits first.)
            ExtMatrix Q;
            if (s_cols.cols == 0) {
                Q = ExtMatrix(D, ell);          // the identity: only 0 is in 0
            } else {
                Q = Transposed(hecke::KernelAsMatrix(Transposed(s_cols)));
            }
            ExtMatrix stacked(0, D, ell);
            bool first = true;
            for (size_t a = 0; a < k; ++a) {
                ExtMatrix piece = Q * n[a];
                stacked = first ? piece : hecke::VStack(stacked, piece);
                first = false;
            }
            ENGINE_CHECK_MSG(!first, "HeckeModuleStructure: an empty operator family");
            ExtMatrix next = hecke::KernelAsMatrix(stacked);
            const uint32_t d_next = next.cols;
            ENGINE_CHECK_MSG(d_next >= s_cols.cols,
                             "HeckeModuleStructure: the socle filtration is not increasing");
            if (d_next == s_cols.cols) break;
            out.socle_dims.push_back(d_next);
            s_cols = next;
        }
    }
    for (size_t j = 0; j + 1 < out.socle_dims.size(); ++j) {
        const uint32_t grow = out.socle_dims[j + 1] - out.socle_dims[j];
        ENGINE_CHECK_MSG(grow % f == 0,
                         "HeckeModuleStructure: a socle layer is not a K-subspace");
        out.socle.push_back(grow / f);
    }
    ENGINE_CHECK_MSG(out.socle_dims.empty() || out.socle_dims.back() == D,
                     "HeckeModuleStructure: the socle filtration does not exhaust V");

    // --- (2) Jordan type of each n_p on V, in units of f ------------------
    for (size_t a = 0; a < k; ++a) {
        std::vector<uint32_t> nullities;
        nullities.push_back(0);
        ExtMatrix power = n[a];
        for (uint32_t s = 1;; ++s) {
            const uint32_t nullity = D - hecke::RankOf(power);
            nullities.push_back(nullity);
            if (nullity == nullities[s - 1]) break;   // stabilised
            if (nullity == D) break;                  // n^s = 0
            ENGINE_CHECK_MSG(s <= D, "HeckeModuleStructure: n_p is not nilpotent");
            power = power * n[a];
        }
        // A stabilised-but-not-full sequence would mean n_p is not nilpotent,
        // which certificate (a) has already excluded.
        ENGINE_CHECK_MSG(nullities.back() == D, "HeckeModuleStructure: n_p is not nilpotent");
        out.jordan.push_back(PartitionFromNullitySequence(nullities, f));
    }

    // --- (3) the structure maps -------------------------------------------
    for (size_t j = 0; j + 1 < out.layer_basis.size(); ++j) {
        StructureMap sm;
        sm.top_layer = static_cast<uint32_t>(j);
        sm.bottom_layer = static_cast<uint32_t>(j + 1);
        sm.top_dim_fl = out.layer_basis[j].cols;
        sm.bottom_dim_fl = out.layer_basis[j + 1].cols;
        sm.top_dim = out.loewy[j];
        sm.bottom_dim = out.loewy[j + 1];

        // [bottom layer basis | m^{j+2} V] is a basis of m^{j+1} V, so the
        // exact solve below splits n_p * (top vector) into its bottom-layer
        // coordinates and its m^{j+2} V part.
        const ExtMatrix& deeper = out.m_basis_cols[j + 2];
        const ExtMatrix split = HCat(out.layer_basis[j + 1], deeper, D, ell);
        ENGINE_CHECK_MSG(split.cols == out.m_dims[j + 1],
                         "HeckeModuleStructure: the layer split is not a basis of m^{j+1} V");

        ExtMatrix stacked(0, sm.top_dim_fl, ell);
        bool first = true;
        for (size_t a = 0; a < k; ++a) {
            const ExtMatrix images = n[a] * out.layer_basis[j];
            const ExtMatrix coords =
                hecke::SolveColumnsExact(split, images, "HeckeModuleStructure(structure map)");
            ExtMatrix theta(sm.bottom_dim_fl, sm.top_dim_fl, ell);
            for (uint32_t r = 0; r < sm.bottom_dim_fl; ++r)
                for (uint32_t c = 0; c < sm.top_dim_fl; ++c) theta.data[r][c] = coords.data[r][c];
            const uint32_t rk = hecke::RankOf(theta);
            sm.rank_p.push_back(rk);
            if (rk > 0) sm.active.push_back(cyc.primes[a]);
            sm.matrices.push_back(theta);
            stacked = first ? theta : hecke::VStack(stacked, theta);
            first = false;
        }
        sm.rank = (sm.top_dim_fl == 0 || sm.bottom_dim_fl == 0) ? 0u : hecke::RankOf(stacked);
        ENGINE_CHECK_MSG(sm.rank % f == 0,
                         "HeckeModuleStructure: a structure map's rank is not a multiple of f");
        sm.rank_over_f = sm.rank / f;
        out.structure_maps.push_back(sm);
    }

    // --- (5) cyclic / free -------------------------------------------------
    ENGINE_CHECK_MSG(cyc.dim_quotient % f == 0,
                     "HeckeModuleStructure: dim V/m_S V is not a multiple of f");
    out.min_generators = cyc.dim_quotient / f;
    out.cyclic = (cyc.dim_quotient == f);
    out.free_rank_one = out.cyclic && (D == out.dim_A);
    return out;
}

} // namespace

// ---------------------------------------------------------------------------

ModuleStructure ComputeModuleStructure(const LocalAlgebra& L) {
    if (!L.ok) {
        ModuleStructure out;
        out.ok = false;
        out.ell = L.ell;
        out.D = L.D;
        return out;
    }
    if (L.cyclicity.D != L.D || L.cyclicity.ell != L.ell || L.cyclicity.f != L.f) {
        throw std::invalid_argument(
            "ComputeModuleStructure: the LocalAlgebra and its certificate (a) disagree");
    }
    ModuleStructure out = Rebuild(L.cyclicity);
    ENGINE_CHECK_MSG(out.dim_A == L.dim_A,
                     "ComputeModuleStructure: dim A_S disagrees with the LocalAlgebra");
    ENGINE_CHECK_MSG(out.min_generators == L.min_generators,
                     "ComputeModuleStructure: min_generators disagrees with the LocalAlgebra");
    return out;
}

// ---------------------------------------------------------------------------
// verify()
// ---------------------------------------------------------------------------

const char* ModuleStructure::verify_message(uint32_t code) {
    switch (code) {
        case 0:  return "ok";
        case 1:  return "the embedded certificate (a) does not verify";
        case 2:  return "ell / D / f / dim A_S disagree with certificate (a)";
        case 3:  return "the primes are not certificate (a)'s";
        case 4:  return "the Loewy dimensions are not the re-derived ones";
        case 5:  return "the Loewy layer dimensions (in units of f) are not the re-derived ones";
        case 6:  return "a stored m^j V basis is not the re-derived canonical one";
        case 7:  return "a stored Loewy layer basis is not the re-derived canonical one";
        case 8:  return "the socle dimensions are not the re-derived ones";
        case 9:  return "the socle layer dimensions are not the re-derived ones";
        case 10: return "a Jordan type is not the re-derived one";
        case 11: return "the number of structure-map pieces is wrong";
        case 12: return "a structure-map piece's layer indices or dimensions are wrong";
        case 13: return "a structure-map matrix is not the re-derived one";
        case 14: return "a structure-map rank, per-generator rank or active list is wrong";
        case 15: return "cyclic / free / min_generators are not the re-derived ones";
        default: return "unknown code";
    }
}

uint32_t ModuleStructure::verify() const {
    if (!ok) return 0;   // a non-local block carries no claim
    if (cyclicity.verify() != 0) return 1;

    ModuleStructure re;
    try {
        re = Rebuild(cyclicity);
    } catch (const std::exception&) {
        return 1;
    }

    if (re.ell != ell || re.D != D || re.f != f || re.dim_A != dim_A) return 2;
    if (re.primes != primes) return 3;
    if (re.m_dims != m_dims) return 4;
    if (re.loewy != loewy) return 5;

    if (re.m_basis_cols.size() != m_basis_cols.size()) return 6;
    for (size_t j = 0; j < m_basis_cols.size(); ++j)
        if (!SameMatrix(re.m_basis_cols[j], m_basis_cols[j])) return 6;

    if (re.layer_basis.size() != layer_basis.size()) return 7;
    for (size_t j = 0; j < layer_basis.size(); ++j)
        if (!SameMatrix(re.layer_basis[j], layer_basis[j])) return 7;

    if (re.socle_dims != socle_dims) return 8;
    if (re.socle != socle) return 9;
    if (re.jordan != jordan) return 10;

    if (re.structure_maps.size() != structure_maps.size()) return 11;
    for (size_t j = 0; j < structure_maps.size(); ++j) {
        const StructureMap& a = re.structure_maps[j];
        const StructureMap& b = structure_maps[j];
        if (a.top_layer != b.top_layer || a.bottom_layer != b.bottom_layer ||
            a.top_dim != b.top_dim || a.bottom_dim != b.bottom_dim ||
            a.top_dim_fl != b.top_dim_fl || a.bottom_dim_fl != b.bottom_dim_fl) {
            return 12;
        }
        if (a.matrices.size() != b.matrices.size()) return 13;
        for (size_t i = 0; i < a.matrices.size(); ++i)
            if (!SameMatrix(a.matrices[i], b.matrices[i])) return 13;
        if (a.rank_p != b.rank_p || a.active != b.active || a.rank != b.rank ||
            a.rank_over_f != b.rank_over_f) {
            return 14;
        }
    }

    if (re.cyclic != cyclic || re.free_rank_one != free_rank_one ||
        re.min_generators != min_generators) {
        return 15;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// to_json()
// ---------------------------------------------------------------------------

std::string ModuleStructure::to_json() const {
    std::ostringstream os;
    os << "{\"stage\":\"module_structure\"";
    os << ",\"ok\":" << (ok ? "true" : "false");
    os << ",\"ell\":" << ell << ",\"D\":" << D << ",\"f\":" << f
       << ",\"dim_A\":" << dim_A;
    os << ",\"primes\":";
    AppendU64List(os, primes);
    os << ",\"loewy\":";
    AppendU32List(os, loewy);
    os << ",\"loewy_dims_fl\":";
    AppendU32List(os, m_dims);
    os << ",\"socle\":";
    AppendU32List(os, socle);
    os << ",\"socle_dims_fl\":";
    AppendU32List(os, socle_dims);
    os << ",\"jordan\":{";
    for (size_t i = 0; i < jordan.size(); ++i) {
        if (i) os << ",";
        os << "\"" << primes[i] << "\":";
        AppendU32List(os, jordan[i]);
    }
    os << "}";
    os << ",\"cyclic\":" << (cyclic ? "true" : "false")
       << ",\"free\":" << (free_rank_one ? "true" : "false")
       << ",\"min_generators\":" << min_generators;
    os << ",\"structure_maps\":[";
    for (size_t j = 0; j < structure_maps.size(); ++j) {
        const StructureMap& sm = structure_maps[j];
        if (j) os << ",";
        os << "{\"piece\":[" << sm.top_layer << "," << sm.bottom_layer << "]"
           << ",\"top_dim\":" << sm.top_dim << ",\"bottom_dim\":" << sm.bottom_dim
           << ",\"rank\":" << sm.rank_over_f << ",\"rank_fl\":" << sm.rank
           << ",\"active\":";
        AppendU64List(os, sm.active);
        os << ",\"matrices\":{";
        for (size_t i = 0; i < sm.matrices.size(); ++i) {
            if (i) os << ",";
            os << "\"" << primes[i] << "\":";
            AppendMatrix(os, sm.matrices[i]);
        }
        os << "}}";
    }
    os << "]";
    os << ",\"verify\":" << verify();
    os << "}";
    return os.str();
}

} // namespace hecke_ext
