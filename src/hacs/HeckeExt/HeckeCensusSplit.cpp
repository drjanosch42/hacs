// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// hecke-extensions slot 11, WP-7 -- HeckeExt/HeckeCensusSplit.  See the header.
// ===========================================================================

#include "HeckeCensusSplit.h"

#include "HeckeAtlas.h"
#include "HeckeExtDriver.h"
#include "HeckeLocalAlgebra.h"

#include "../Hecke/HeckeInvolution.h"
#include "../Hecke/HeckeModuleDecomposition.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SL2ZBoundaryPair.h"
#include "../Hecke/SubspaceOps.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/FlatMatrix.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>

namespace hecke_ext {

namespace {

using Clock = std::chrono::steady_clock;
using Orbit = std::vector<std::vector<std::vector<uint32_t>>>;
using ClassKey = std::pair<uint32_t, Orbit>;

ExtMatrix ToExt(const FlatMatrix& M) {
    ExtMatrix out(M.get_rows(), M.get_cols(), M.get_p());
    for (uint32_t i = 0; i < M.get_rows(); ++i)
        for (uint32_t j = 0; j < M.get_cols(); ++j) out.data[i][j] = M.get_entry(i, j);
    return out;
}

bool SameEntries(const ExtMatrix& a, const ExtMatrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (uint32_t i = 0; i < a.rows; ++i)
        for (uint32_t j = 0; j < a.cols; ++j)
            if (a.data[i][j] != b.data[i][j]) return false;
    return true;
}

std::string OwnGroup(const IdealRecord& id) {
    for (const IdealGroupMatch& g : id.groups)
        if (g.own) return g.group;
    return std::string();
}

// The options every census driver run uses: the published campaign's
// attachments (`ext_groups = all`, `sign = all` -> with_cone, with_sign_split)
// and the default family.  The generation routes (b), (c) are off: they run
// after Stage 2 and change no ideal, witness or restricted operator -- only
// `final`, which the census reads from the PUBLISHED record.
ExtDriverOptions CellOptions(bool attachments) {
    ExtDriverOptions o;
    o.with_cone = attachments;
    o.with_sign_split = attachments;
    o.run_commutant = false;
    o.run_sturm = false;
    return o;
}

bool IsGloverEll(uint32_t ell) { return ell == 5 || ell == 7 || ell == 11 || ell == 13; }

std::map<ClassKey, int64_t> AsMap(const std::vector<ResidueDifference>& v) {
    std::map<ClassKey, int64_t> m;
    for (const ResidueDifference& d : v) m[ClassKey(d.cls.r, d.cls.orbit)] += d.delta;
    return m;
}

int64_t At(const std::map<ClassKey, int64_t>& m, const ClassKey& k) {
    auto it = m.find(k);
    return it == m.end() ? 0 : it->second;
}

} // namespace

uint64_t CensusTpFingerprint(const std::vector<uint64_t>& primes, const std::vector<ExtMatrix>& t) {
    ENGINE_CHECK_MSG(primes.size() == t.size(), "CensusTpFingerprint: primes and t not aligned");
    std::ostringstream s;
    for (size_t k = 0; k < primes.size(); ++k) {
        if (k) s << ";";
        s << primes[k] << ":[";
        for (uint32_t i = 0; i < t[k].rows; ++i) {
            if (i) s << ",";
            s << "[";
            for (uint32_t j = 0; j < t[k].cols; ++j) s << (j ? "," : "") << t[k].data[i][j];
            s << "]";
        }
        s << "]";
    }
    return engine::FnvBytes(engine::kFnvOffset64, s.str());
}

bool CensusGloverWeights(uint32_t ell, std::vector<std::vector<ResidueCharacterMultiset>>& out,
                         std::vector<uint64_t>& primes_out, std::string& why) {
    ENGINE_CHECK_MSG(IsGloverEll(ell), "CensusGloverWeights: ell outside {5, 7, 11, 13}");
    out.assign(ell, std::vector<ResidueCharacterMultiset>());
    primes_out.clear();
    for (uint32_t m = 0; m < ell; m += 2) {
        HeckeExtDriver d(m, ell, CellOptions(false));
        d.run();
        const std::vector<AtlasIdealCharacter> chars = AtlasIdealCharacters(d);
        std::string w;
        if (!AtlasResidueMultisets(chars, out[m], w)) {
            why = "weight cell (" + std::to_string(m) + "," + std::to_string(ell) + "): " + w;
            return false;
        }
        for (const AtlasIdealCharacter& c : chars) {
            if (!c.local || c.primes.empty()) continue;
            if (primes_out.empty()) primes_out = c.primes;
            if (c.primes != primes_out) {
                why = "weight cell (" + std::to_string(m) + "," + std::to_string(ell)
                      + "): anemic primes differ between ideals";
                return false;
            }
        }
    }
    return true;
}

CensusSplitCell ComputeCensusSplit(uint32_t n, uint32_t ell, const CensusSplitOptions& opts,
                                   const std::vector<std::vector<ResidueCharacterMultiset>>* weights,
                                   const std::vector<uint64_t>* weight_primes) {
    const Clock::time_point t0 = Clock::now();
    CensusSplitCell cell;
    cell.n = n;
    cell.ell = ell;

    std::unique_ptr<hecke::ScopedUEll> u_guard;
    if (opts.u_ell_budget_mb != 0) {
        hecke::UEllSetting u;
        u.mode = hecke::UEllMode::BudgetMb;
        u.budget_mb = opts.u_ell_budget_mb;
        u_guard.reset(new hecke::ScopedUEll(u, hecke::SettingSource::Spec));
    }

    HeckeExtDriver d(n, ell, CellOptions(true));
    d.run();
    const HeckeExtRecord& rec = d.record();
    const std::vector<uint32_t>& dims = d.H_dimensions();
    cell.dim_H1 = dims.size() > 1 ? dims[1] : 0;

    std::vector<uint32_t> h1_ideals;   // indices into rec.ideals
    for (uint32_t i = 0; i < rec.ideals.size(); ++i)
        if (OwnGroup(rec.ideals[i]) == "H1") h1_ideals.push_back(i);

    if (cell.dim_H1 == 0) {
        if (!h1_ideals.empty()) cell.failures.push_back("H1 ideals on a zero H^1");
        cell.glover_status = "not_applicable:dim_H1=0";
        cell.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
        return cell;
    }

    // --- Stage 2 again, on the driver's own operators ---------------------
    std::vector<std::pair<uint64_t, ExtMatrix>> ops = d.cell_operators(1);
    std::sort(ops.begin(), ops.end(),
              [](const std::pair<uint64_t, ExtMatrix>& a, const std::pair<uint64_t, ExtMatrix>& b) {
                  return a.first < b.first;
              });
    hecke::HeckeModuleDecomposition decomp(ops, ell);
    decomp.compute();
    ENGINE_CHECK_MSG(decomp.certify() == 0, "ComputeCensusSplit: the re-run decomposition does not certify");
    const std::vector<hecke::PrimaryBlock>& blocks = decomp.blocks();
    ENGINE_CHECK_MSG(blocks.size() == h1_ideals.size(),
                     "ComputeCensusSplit: the re-run decomposition has a different number of H^1 blocks");

    // --- r^1 and the sign bases (the driver's own objects) ----------------
    const hecke::SL2ZBoundaryPair* pair = d.boundary_pair();
    ENGINE_CHECK_MSG(pair != nullptr, "ComputeCensusSplit: the driver built no boundary pair");
    const uint32_t dim_bd1 = pair->dim_H_bd(1);
    ExtMatrix r1(dim_bd1, cell.dim_H1, ell);
    {
        const std::vector<FlatMatrix>& maps = pair->restriction().get_induced_H_eq_maps();
        if (maps.size() > 1 && maps[1].get_rows() == dim_bd1 && maps[1].get_cols() == cell.dim_H1) {
            r1 = ToExt(maps[1]);
        } else {
            ENGINE_CHECK_MSG(dim_bd1 == 0 && maps.size() > 1 && maps[1].get_rows() == 0,
                             "ComputeCensusSplit: r^1 has an unexpected shape");
        }
    }
    cell.rank_r1 = (dim_bd1 == 0) ? 0 : hecke::RankOf(r1);

    const hecke::HeckeInvolution* inv = d.involution();
    ENGINE_CHECK_MSG(inv != nullptr, "ComputeCensusSplit: the driver built no involution");
    const ExtMatrix Hp = ToExt(inv->H_plus_basis(1));
    const ExtMatrix Hm = ToExt(inv->H_minus_basis(1));
    ENGINE_CHECK_MSG(Hp.rows == cell.dim_H1 && Hm.rows == cell.dim_H1,
                     "ComputeCensusSplit: sign bases of the wrong height");
    cell.dim_H1_plus = Hp.cols;
    cell.dim_H1_minus = Hm.cols;

    auto meet = [&](const ExtMatrix& B, const ExtMatrix& U) -> uint32_t {
        if (U.cols == 0 || B.cols == 0) return 0;
        const uint32_t r = hecke::RankOf(hecke::HStack({B, U}, cell.dim_H1, ell));
        return B.cols + U.cols - r;
    };

    // --- per block ---------------------------------------------------------
    uint32_t sum_dim = 0, sum_rank = 0, sum_plus = 0, sum_minus = 0;
    for (size_t k = 0; k < blocks.size(); ++k) {
        const hecke::PrimaryBlock& b = blocks[k];
        const IdealRecord& id = rec.ideals[h1_ideals[k]];
        CensusSplitRow row;
        row.ideal_id = id.id;
        row.dim_E = b.dim;
        ENGINE_CHECK_MSG(b.basis_columns.cols == b.dim && b.basis_columns.rows == cell.dim_H1,
                         "ComputeCensusSplit: a block basis of the wrong shape");

        // The matching rule of [E] sec:census (e), between the re-run block
        // and the driver's record ideal: witness_poly AND every restricted
        // t_p, exactly.  A miss is structural (the decomposition is
        // deterministic), so it stops the cell.
        std::vector<ExtMatrix> t;
        for (const auto& op : ops) {
            row.operator_primes.push_back(op.first);
            t.push_back(hecke::SolveColumnsExact(b.basis_columns, op.second * b.basis_columns,
                                                 "ComputeCensusSplit: block restriction"));
        }
        ENGINE_CHECK_MSG(b.witness_poly.coeffs == id.witness_poly,
                         "ComputeCensusSplit: a re-run block's witness_poly is not the record ideal's");
        ENGINE_CHECK_MSG(row.operator_primes == id.operator_primes && t.size() == id.t_p.size(),
                         "ComputeCensusSplit: a re-run block's operator primes are not the record ideal's");
        for (size_t j = 0; j < t.size(); ++j)
            ENGINE_CHECK_MSG(SameEntries(t[j], id.t_p[j]),
                             "ComputeCensusSplit: a re-run block's restricted t_p is not the record ideal's");
        row.witness = id.witness_poly;
        row.t_p_fnv = CensusTpFingerprint(row.operator_primes, t);

        // (d1)
        row.rank_r1 = (dim_bd1 == 0) ? 0 : hecke::RankOf(r1 * b.basis_columns);
        row.verdict = (row.rank_r1 == row.dim_E) ? "eisenstein"
                      : (row.rank_r1 == 0)      ? "cuspidal"
                                                : "mixed";
        // (d2)
        row.dim_plus = meet(b.basis_columns, Hp);
        row.dim_minus = meet(b.basis_columns, Hm);
        row.sign_computed = true;
        if (row.dim_plus + row.dim_minus != row.dim_E)
            cell.failures.push_back("ideal " + std::to_string(row.ideal_id)
                                    + ": dim_plus + dim_minus != dim_E");

        sum_dim += row.dim_E;
        sum_rank += row.rank_r1;
        sum_plus += row.dim_plus;
        sum_minus += row.dim_minus;
        row.glover_token = "not_evaluated";
        cell.rows.push_back(std::move(row));
    }
    if (sum_dim != cell.dim_H1) cell.failures.push_back("sum dim_E != dim H^1");
    if (sum_rank != cell.rank_r1) cell.failures.push_back("sum rank_r1 != rank r^1");
    if (sum_plus != cell.dim_H1_plus) cell.failures.push_back("sum dim_plus != dim H^1_+");
    if (sum_minus != cell.dim_H1_minus) cell.failures.push_back("sum dim_minus != dim H^1_-");
    if (rec.has_dim_H1_shriek && rec.dim_H1_shriek + cell.rank_r1 != cell.dim_H1)
        cell.failures.push_back("dim H^1_! + rank r^1 != dim H^1 (the record's dim_H1_shriek)");

    // --- (d6) Glover ------------------------------------------------------
    if (!opts.glover || !IsGloverEll(ell)) {
        cell.glover_status = "not_evaluated:ell";
    } else if (weights == nullptr || weight_primes == nullptr) {
        cell.glover_status = "not_evaluated:no_weight_cells";
    } else {
        const std::vector<AtlasIdealCharacter> chars = AtlasIdealCharacters(d);
        std::vector<ResidueCharacterMultiset> ms;
        std::string why;
        bool ok = AtlasResidueMultisets(chars, ms, why);
        std::vector<uint64_t> primes;
        for (const AtlasIdealCharacter& c : chars)
            if (c.local && !c.primes.empty()) { primes = c.primes; break; }
        if (ok && primes != *weight_primes) {
            ok = false;
            why = "the cell's anemic primes differ from the weight cells'";
        }
        if (!ok) {
            cell.glover_status = "not_evaluated:" + why;
        } else {
            IdentityCertificate cert;
            cert.n = n;
            cert.ell = ell;
            cert.dim_H0 = dims.size() > 0 ? dims[0] : 0;
            cert.dim_H1 = cell.dim_H1;
            cert.dim_H2 = dims.size() > 2 ? dims[2] : 0;
            cert.p = 2;
            cert.p_prime = 3;
            for (const auto& op : ops) {
                if (op.first == 2) cert.A_p = op.second;
                if (op.first == 3) cert.A_p_prime = op.second;
            }
            cert.residue_primes = primes;
            cert.residue_cell = ms;
            cert.residue_weights = *weights;
            cell.identity_code = cert.verify();
            cell.identity_detail = cert.residue_atlas.detail;
            if (cert.residue_atlas.status != IdentityStatus::kPassed)
                cell.failures.push_back("identity 6 not kPassed: " + cert.residue_atlas.detail);

            // The main term alone: the same evaluation with every H^0 / H^2
            // class removed (the cell's own and the weight cells').
            IdentityCertificate main_only = cert;
            main_only.residue_cell[0].clear();
            main_only.residue_cell[2].clear();
            for (auto& w : main_only.residue_weights)
                if (w.size() == 3) { w[0].clear(); w[2].clear(); }
            const std::map<ClassKey, int64_t> full =
                AsMap(IdentityResidueRhs(cert, GloverConvention::kStandard));
            const std::map<ClassKey, int64_t> main =
                AsMap(IdentityResidueRhs(main_only, GloverConvention::kStandard));
            std::map<ClassKey, int64_t> lhs;
            for (const ResidueCharacterClass& c : ms[1]) lhs[ClassKey(c.r, c.orbit)] += c.multiplicity;

            // Row -> class, and the rows sharing a class.
            std::map<uint32_t, ClassKey> key_of;
            for (const AtlasIdealCharacter& c : chars)
                if (c.degree == 1) key_of[c.ideal] = ClassKey(c.r, c.orbit);
            std::map<ClassKey, std::vector<uint32_t>> rows_of;
            std::map<uint32_t, uint32_t> dim_e_of;
            for (const CensusSplitRow& r : cell.rows) {
                ENGINE_CHECK_MSG(key_of.count(r.ideal_id) == 1, "ComputeCensusSplit: an H1 row without a class");
                rows_of[key_of[r.ideal_id]].push_back(r.ideal_id);
                dim_e_of[r.ideal_id] = r.dim_E;
            }
            for (CensusSplitRow& r : cell.rows) {
                const ClassKey& k = key_of[r.ideal_id];
                r.glover_evaluated = true;
                r.glover_mult = At(lhs, k);
                r.glover_main = At(main, k);
                r.glover_corr = At(full, k) - r.glover_main;
                r.identity6_mismatch = (r.glover_mult - r.glover_main != r.glover_corr);
                const std::vector<uint32_t>& ids = rows_of[k];
                if (ids.size() > 1) {
                    std::string s = "class_level:";
                    for (size_t i = 0; i < ids.size(); ++i) s += (i ? "+" : "") + std::to_string(ids[i]);
                    r.glover_token = s;
                    // (slot-14 WP-14b, pass-2 m-6): the class-level Glover sum
                    // Sum_{rows of the class} dim_E == glover_mult * r, a
                    // per-cell invariant like the single-row check below --
                    // previously checked only by hand (the pass-2 review, all
                    // 298 classes of edition 1). Checked once per class (on
                    // its first row in cell.rows order) so a failure is one
                    // message, not one per row.
                    if (r.ideal_id == ids.front()) {
                        int64_t sum_dim_e = 0;
                        for (uint32_t id : ids) sum_dim_e += static_cast<int64_t>(dim_e_of.at(id));
                        if (r.glover_mult * static_cast<int64_t>(k.first) != sum_dim_e)
                            cell.failures.push_back("class " + s + ": class-level glover multiplicity != sum dim_E / r (m-6)");
                    }
                } else {
                    if (r.glover_mult * static_cast<int64_t>(k.first) != static_cast<int64_t>(r.dim_E))
                        cell.failures.push_back("ideal " + std::to_string(r.ideal_id)
                                                + ": glover multiplicity != dim_E / r");
                    const int64_t dd = r.glover_mult - r.glover_main;
                    r.glover_token = (dd == 0) ? "equal" : "differs_by:" + std::to_string(dd);
                }
                if (r.identity6_mismatch)
                    cell.failures.push_back("ideal " + std::to_string(r.ideal_id) + ": identity_6_mismatch");
            }
            cell.glover_status = "evaluated";
        }
    }
    cell.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    return cell;
}

std::string CensusSplitLine(const CensusSplitCell& cell, const CensusSplitRow& row) {
    std::ostringstream s;
    s << "CENSUS_SPLIT n=" << cell.n << " ell=" << cell.ell << " ideal=" << row.ideal_id << " witness=";
    for (size_t i = 0; i < row.witness.size(); ++i) s << (i ? "," : "") << row.witness[i];
    s << " dim_E=" << row.dim_E << " rank_r1=" << row.rank_r1 << " verdict=" << row.verdict
      << " dim_plus=" << row.dim_plus << " dim_minus=" << row.dim_minus;
    if (row.glover_evaluated) {
        s << " glover_mult=" << row.glover_mult << " glover_main=" << row.glover_main
          << " glover_corr=" << row.glover_corr << " glover=" << row.glover_token;
    } else {
        s << " glover_mult=not_evaluated glover_main=not_evaluated glover_corr=not_evaluated glover="
          << row.glover_token;
    }
    std::ostringstream h;
    h << std::hex << row.t_p_fnv;
    std::string hx = h.str();
    while (hx.size() < 16) hx = "0" + hx;
    s << " t_p=0x" << hx;
    return s.str();
}

} // namespace hecke_ext
