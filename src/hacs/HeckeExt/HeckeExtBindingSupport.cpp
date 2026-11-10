// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// HeckeExtBindingSupport.cpp -- see HeckeExtBindingSupport.h.
// (hecke-extensions slot 9, WP-13 / console phase C5b.)
// ===========================================================================

#include "HeckeExtBindingSupport.h"

#include "HeckeAtlas.h"
#include "HeckeDicksonMultiplication.h"
#include "HeckeExtDriver.h"
#include "HeckeExtRecord.h"

#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../Hecke/HeckeContragredient.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckePairing.h"
#include "../Hecke/HeckeSettings.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"

#include "../../ModArithmetic/EngineCheck.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hecke_ext {

namespace {

ExtMatrix ToExt(const FlatMatrix& f) { return f.to_matrix(); }

// One owned (local system, cohomology) of Sym^m over F_ell on K -- the
// source cell of L_* (HeckeExtEndToEndValidation.cpp's `Stack`, minus the
// complex, which the caller's K supplies).
struct OwnedCohomology {
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    OwnedCohomology(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& d : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) rels.push_back(d.rel);
        lssc.reset(new LocalSystemSimplicialComplex(K, std::move(rels), n_sym + 1, ell));
        coh.reset(new EquivariantCohomologyOnK(lssc.get()));
        coh->compute_cohomology();
    }
};

std::unique_ptr<HeckeExtDriver> RunCell(uint32_t n, uint32_t ell, const std::vector<uint64_t>& T) {
    ExtDriverOptions o;
    o.proof_range = true;
    o.good_primes = T;
    o.s_sep = std::vector<uint64_t>();
    o.run_commutant = false;
    o.run_sturm = false;
    std::unique_ptr<HeckeExtDriver> d(new HeckeExtDriver(n, ell, o));
    d->run();
    return d;
}

} // namespace

// ---------------------------------------------------------------------------

std::unique_ptr<BoundaryActionAtPrime> BuildBoundaryActionAtPrime(
    const hecke::SL2ZBoundaryPair& pair, const hecke::SL2ZTessellationTables& model, uint64_t p) {
    ENGINE_CHECK_MSG(p >= 2, "BuildBoundaryActionAtPrime: p < 2");
    std::unique_ptr<BoundaryActionAtPrime> out(new BoundaryActionAtPrime());
    out->p = p;
    {
        std::lock_guard<std::mutex> guard(hecke::tau_construction_mutex());
        out->cosets.reset(new hecke::HeckeCosetTables(p, 1, pair.ell(), pair.n_sym()));
        out->taub.reset(new hecke::HeckeBoundaryTauTable(pair.K(), model, *out->cosets));
    }
    out->bd.reset(new hecke::HeckeBoundaryOperator(&pair.boundary_cohomology(), out->cosets.get(),
                                                   out->taub.get()));
    out->bd->compute();
    return out;
}

std::unique_ptr<ConeActionAtPrime> BuildConeActionAtPrime(
    const hecke::SL2ZBoundaryPair& pair, const EquivariantMappingCone& cone,
    const hecke::SL2ZTessellationTables& model, uint64_t p) {
    ENGINE_CHECK_MSG(p >= 2, "BuildConeActionAtPrime: p < 2");
    std::unique_ptr<ConeActionAtPrime> out(new ConeActionAtPrime());
    out->p = p;
    {
        std::lock_guard<std::mutex> guard(hecke::tau_construction_mutex());
        out->cosets.reset(new hecke::HeckeCosetTables(p, 1, pair.ell(), pair.n_sym()));
        out->tau.reset(new hecke::HeckeTauTable(pair.K(), model, *out->cosets, pair.ell(),
                                                pair.n_sym()));
        out->taub.reset(new hecke::HeckeBoundaryTauTable(pair.K(), model, *out->cosets));
    }
    out->op.reset(new hecke::HeckeOperatorOnK(&pair.bulk_cohomology(), out->cosets.get(),
                                              out->tau.get()));
    out->op->compute();
    out->eta.reset(new hecke::HeckePairHomotopy(*out->tau, *out->taub));
    out->bd.reset(new hecke::HeckeBoundaryOperator(&pair.boundary_cohomology(), out->cosets.get(),
                                                   out->taub.get()));
    out->bd->compute();
    out->act.reset(new hecke::HeckeConeAction(&cone, out->op.get(), out->bd.get(), out->eta.get()));
    out->act->compute();
    return out;
}

// ---------------------------------------------------------------------------

std::unique_ptr<hecke::HeckeInvolution> BuildProductionInvolution(
    const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
    const hecke::SL2ZTessellationTables& model, uint32_t n_sym, uint32_t ell,
    const std::string& label) {
    const hecke::IotaSimplicialMap iota_K = hecke::BuildIotaSimplicialMapOnModel(K, model);
    std::unique_ptr<hecke::HeckeInvolution> inv(new hecke::HeckeInvolution(&coh, iota_K, label));
    inv->set_exact_relation_data(GenerateSL2ZRelationsWithGroupData(K, n_sym, ell));
    inv->compute();
    return inv;
}

std::unique_ptr<HeckeSignSplit> BuildSignSplit(
    const EquivariantCohomologyOnK& coh, const SimplicialComplex& K,
    const hecke::SL2ZTessellationTables& model, const hecke::HeckeInvolution& iota,
    uint32_t n_sym, uint32_t ell, uint32_t q, const std::vector<uint64_t>& primes) {
    ENGINE_CHECK_MSG(q < iota.num_degrees(), "BuildSignSplit: degree out of range");
    if (iota.dim_H(q) == 0) return nullptr;
    hecke::HeckeOperatorFamily fam(&coh, K, model, ell, n_sym);
    std::vector<std::pair<uint64_t, ExtMatrix>> ops;
    for (uint64_t p : primes) {
        if (p % ell == 0) throw std::invalid_argument("sign split: the operator primes must be good (p != ell)");
        ops.emplace_back(p, fam.T_p(p)[q]);
    }
    std::unique_ptr<HeckeSignSplit> s(new HeckeSignSplit(ell, n_sym, q, ToExt(iota.matrix(q)),
                                                         ToExt(iota.H_plus_basis(q)),
                                                         ToExt(iota.H_minus_basis(q)),
                                                         std::move(ops)));
    s->compute();
    return s;
}

// ---------------------------------------------------------------------------

std::vector<FlatMatrix> DicksonInducedMaps(const EquivariantCohomologyOnK& target,
                                           const EquivariantCohomologyOnK& source,
                                           const SimplicialComplex& K, uint32_t n, uint32_t ell) {
    if (!DicksonMultiplicationApplicable(n, ell))
        throw std::invalid_argument("L_map: needs n >= ell + 1 (the source weight m = n - ell - 1)");
    const FlatMatrix Lmat = DicksonMultiplicationMatrix(n, ell);
    FiberBlockTable table;
    table.resize(K.get_dimension() + 1);
    for (uint32_t q = 0; q < table.size(); ++q) {
        const uint32_t N_q = K.get_num_q_simplices(q);
        table[q].resize(N_q);
        for (uint32_t s = 0; s < N_q; ++s) {
            PerSimplexBlock blk;
            blk.s2_idx_in_K2 = s;
            blk.F = Lmat;
            table[q][s].push_back(blk);
        }
    }
    EquivariantCochainComplexMorphismOnK Lmap(&target, &source, table, "DicksonMultiplication");
    Lmap.compute_induced_maps_on_cohomology();
    return Lmap.get_induced_H_eq_maps();
}

// ---------------------------------------------------------------------------

const char* ToString(IdentityStatus s) {
    switch (s) {
        case IdentityStatus::kNotApplicable: return "not_applicable";
        case IdentityStatus::kPassed:        return "passed";
        case IdentityStatus::kFailed:        return "failed";
    }
    return "unknown";
}

std::vector<std::pair<const char*, const IdentityItem*>> IdentityItems(const IdentityCertificate& c) {
    // verify()'s fixed order: 1 = eichler_shimura, ..., 7 = socle_loewy_pd.
    return {{"eichler_shimura", &c.eichler_shimura},
            {"commutation", &c.commutation},
            {"d2_vanishes", &c.d2_vanishes},
            {"p2_recursion", &c.p2_recursion},
            {"dickson_multiplication", &c.dickson_multiplication},
            {"residue_atlas", &c.residue_atlas},
            {"socle_loewy_pd", &c.socle_loewy_pd}};
}

IdentityAssembly AssembleIdentityCertificate(const EquivariantCohomologyOnK& coh,
                                             const SimplicialComplex& K,
                                             const hecke::SL2ZTessellationTables& model,
                                             uint32_t n, uint32_t ell, uint64_t p,
                                             uint64_t p_prime,
                                             const std::vector<bool>& want_item) {
    if (p == p_prime) throw std::invalid_argument("check_identities: the two primes must differ");
    if (p % ell == 0 || p_prime % ell == 0)
        throw std::invalid_argument("check_identities: the two primes must be good (!= ell)");
    auto want = [&](uint32_t k) { return k < want_item.size() && want_item[k]; };

    IdentityAssembly out;
    out.supplied.assign(kIdentityItems + 1, false);
    out.why.assign(kIdentityItems + 1, std::string());
    IdentityCertificate& c = out.cert;
    const std::vector<uint32_t>& dims = coh.get_H_eq_dimensions();
    ENGINE_CHECK_MSG(dims.size() == 3, "check_identities: the 7-vertex model has three degrees");
    c.n = n;
    c.ell = ell;
    c.dim_H0 = dims[0];
    c.dim_H1 = dims[1];
    c.dim_H2 = dims[2];
    c.p = p;
    c.p_prime = p_prime;
    // Items 1 and 3 read the dimensions alone.
    out.supplied[1] = true;
    out.supplied[3] = true;

    hecke::HeckeOperatorFamily fam(&coh, K, model, ell, n);
    if (dims[1] > 0) {
        c.A_p = fam.T_p(p)[1];
        c.A_p_prime = fam.T_p(p_prime)[1];
        c.A_p2 = fam.T_p_pow_direct(p, 2)[1];   // the INDEPENDENT double-coset T_{p^2}
    }
    out.supplied[2] = true;
    out.supplied[4] = true;

    // Item 5: the Dickson multiplication from the source cell m = n - ell - 1.
    if (DicksonMultiplicationApplicable(n, ell) && dims[1] > 0) {
        const uint32_t m = DicksonSourceDegree(n, ell);
        OwnedCohomology source(K, m, ell);
        const auto& dm = source.coh->get_H_eq_dimensions();
        if (dm.size() > 1 && dm[1] > 0) {
            hecke::HeckeOperatorFamily fam_m(source.coh.get(), K, model, ell, m);
            c.n_source = m;
            c.A_p_source = fam_m.T_p(p)[1];
            const std::vector<FlatMatrix> induced = DicksonInducedMaps(coh, *source.coh, K, n, ell);
            if (induced.size() > 1) c.L_star = induced[1].to_matrix();
        }
        out.supplied[5] = true;
    } else {
        // Inapplicable at the cell: the certificate reports not_applicable.
        out.supplied[5] = true;
    }

    // Item 6: the residue atlas, on request, at the in-process atlas range.
    if (want(6)) {
        if (!(ell >= 5 && ell <= 13 && n % 2 == 0)) {
            out.why[6] = "the weight cells m < ell are run in process only at 5 <= ell <= 13, n even";
        } else {
            hecke::UEllSetting off;
            off.mode = hecke::UEllMode::Off;
            const hecke::ScopedUEll u_guard(off);
            const std::vector<uint64_t> T = AnemicPool(ell);
            std::vector<std::vector<ResidueCharacterMultiset>> weights(ell);
            bool ok = true;
            std::string why;
            for (uint32_t m = 0; m <= ell - 1 && ok; m += 2) {
                std::unique_ptr<HeckeExtDriver> d = RunCell(m, ell, T);
                ok = AtlasResidueMultisets(AtlasIdealCharacters(*d), weights[m], why);
            }
            std::vector<ResidueCharacterMultiset> cell;
            if (ok) {
                std::unique_ptr<HeckeExtDriver> d = RunCell(n, ell, T);
                ok = AtlasResidueMultisets(AtlasIdealCharacters(*d), cell, why);
            }
            if (ok) {
                c.residue_primes = T;
                c.residue_cell = cell;
                c.residue_weights = weights;
                out.supplied[6] = true;
            } else {
                out.why[6] = "the residue multisets are not available: " + why;
            }
        }
    } else {
        out.why[6] = "not requested (the weight cells m < ell are run on request only)";
    }

    // Item 7: socle <-> Loewy under PD, on request.
    if (want(7)) {
        if (!(ell >= 5 && n % 2 == 0)) {
            out.why[7] = "needs ell >= 5 and n even";
        } else {
            const hecke::ContragredientCell cell(n, ell);
            const hecke::PoincareDualityPairing pd(cell);
            const FlatMatrix& G = pd.gram(1);
            if (G.get_rows() == 0) {
                out.supplied[7] = true;   // X_3 = 0: the certificate says not_applicable
            } else {
                for (uint32_t q : {2u, 3u, 5u, 7u}) {
                    if (q % ell == 0) continue;
                    hecke::ContragredientCell::HeckePair hp = cell.hecke_pair(q);
                    hecke::HeckeBoundaryTauTable taub(cell.K(), cell.model(), *hp.cosets);
                    hecke::HeckePairHomotopy eta(*hp.tau, taub);
                    hecke::HeckeBoundaryOperator bd(&pd.pair_M().boundary_cohomology(),
                                                    hp.cosets.get(), &taub);
                    bd.compute();
                    hecke::HeckeConeAction act(&pd.cone(), hp.op_M.get(), &bd, &eta);
                    act.compute();
                    c.pd_ops_V.push_back(act.A_c(1));
                    c.pd_ops_W.push_back(hp.op_D->get_H_matrices()[1]);
                }
                c.pd_pairing = G.to_matrix();
                out.supplied[7] = true;
            }
        }
    } else {
        out.why[7] = "not requested (the contragredient cell and the cone action are built on request only)";
    }

    out.code = c.verify();
    return out;
}

} // namespace hecke_ext
