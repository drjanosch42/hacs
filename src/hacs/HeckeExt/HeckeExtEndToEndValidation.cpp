// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// (hecke-extensions slot 6, WP-9 / phase E6) HeckeExtEndToEndValidation --
// see the header for the arm list.
// ===========================================================================

#include "HeckeExtEndToEndValidation.h"

#include "GloverFactors.h"
#include "HeckeAtlas.h"
#include "HeckeDicksonMultiplication.h"
#include "HeckeExtDriver.h"
#include "HeckeExtReport.h"
#include "HeckeIdentityCertificate.h"

#include "../Hecke/HeckeBoundaryDatum.h"
#include "../Hecke/HeckeBoundaryOperator.h"
#include "../Hecke/HeckeConeAction.h"
#include "../Hecke/HeckeContragredient.h"
#include "../Hecke/HeckeOperatorFamily.h"
#include "../Hecke/HeckePairHomotopy.h"
#include "../Hecke/HeckePairing.h"
#include "../Hecke/HeckeSettings.h"
#include "../Hecke/SL2ZTessellationTables.h"
#include "../EquivariantCochainComplexMorphismOnK.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace hecke_ext;

namespace {

oc_util::Checker Check("  *** FAIL [e2e] ");

using Clock = std::chrono::steady_clock;
double Since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string Tag(uint32_t n, uint64_t ell) {
    return "(" + std::to_string(n) + "," + std::to_string(ell) + ")";
}

std::string Primes(const std::vector<uint64_t>& v) {
    std::string s = "{";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "}";
}

bool IsPrime(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

// The atlas's anemic pool T = {q <= 31} \ {ell}.
std::vector<uint64_t> Pool(uint32_t ell) {
    std::vector<uint64_t> T;
    for (uint64_t q = 2; q <= 31; ++q)
        if (IsPrime(q) && q != ell) T.push_back(q);
    return T;
}

uint32_t PowMod(uint64_t b, uint64_t e, uint32_t m) {
    uint64_t r = 1 % m;
    b %= m;
    while (e) {
        if (e & 1u) r = (r * b) % m;
        b = (b * b) % m;
        e >>= 1u;
    }
    return static_cast<uint32_t>(r);
}

std::string ClassText(const ResidueCharacterClass& c) {
    // r = 1 throughout at ell <= 13 ([WP8] Sec. 2); print the first conjugate.
    std::string s = "r=" + std::to_string(c.r) + ":(";
    if (!c.orbit.empty()) {
        const auto& v = c.orbit[0];
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s += ",";
            if (v[i].size() == 1) {
                s += std::to_string(v[i][0]);
            } else {
                s += "[";
                for (size_t k = 0; k < v[i].size(); ++k)
                    s += (k ? "," : "") + std::to_string(v[i][k]);
                s += "]";
            }
        }
    }
    return s + ")";
}

std::string FactorText(const std::vector<IdentityGloverFactor>& f) {
    std::string s;
    for (size_t i = 0; i < f.size(); ++i) {
        if (i) s += " ";
        s += "Sym^" + std::to_string(f[i].sym) + "(x)det^" + std::to_string(f[i].det_twist);
        if (f[i].multiplicity > 1) s += "x" + std::to_string(f[i].multiplicity);
    }
    return s;
}

// One driver run, owned.
std::unique_ptr<HeckeExtDriver> Run(uint32_t n, uint32_t ell, const std::vector<uint64_t>& good,
                                    const std::vector<uint64_t>& s_sep, bool gen_routes,
                                    bool proof_range = true) {
    ExtDriverOptions o;
    o.proof_range = proof_range;
    o.good_primes = good;
    o.s_sep = s_sep;
    o.run_commutant = gen_routes;
    o.run_sturm = gen_routes;
    std::unique_ptr<HeckeExtDriver> d(new HeckeExtDriver(n, ell, o));
    d->run();
    return d;
}

// The pipeline stack for items 4 and 5 (the pattern of HeckeIdentityValidation).
struct Stack {
    SimplicialComplex K;
    hecke::SL2ZTessellationTables model;
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;

    Stack(uint32_t n_sym, uint32_t ell)
        : K(CreateSL2ZFundamentalDomainTriangulation()),
          model(K),
          lssc(K, BuildRelations(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc) {
        coh.compute_cohomology();
    }
    static std::vector<LocalSystemSimplicialComplex::SimplexRelation>
    BuildRelations(const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell))
            rels.push_back(datum.rel);
        return rels;
    }
};

// L_* : H^1(M_m) -> H^1(M_n), m = n - ell - 1 (HeckeIdentityValidation arm 3).
MatrixT<uint32_t> InducedDickson(Stack& target, Stack& source, uint32_t n, uint32_t ell) {
    const FlatMatrix Lmat = DicksonMultiplicationMatrix(n, ell);
    const SimplicialComplex& K = target.K;
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
    EquivariantCochainComplexMorphismOnK Lmap(&target.coh, &source.coh, table,
                                              "DicksonMultiplication");
    Lmap.compute_induced_maps_on_cohomology();
    const std::vector<FlatMatrix>& induced = Lmap.get_induced_H_eq_maps();
    if (induced.size() < 2) return MatrixT<uint32_t>();
    return induced[1].to_matrix();
}

// Item 7's inputs (slot-6 WP-3, [C] cl:pd(3)): A_c^{(1)}(T_p) on X_3 =
// H^1_c(M_n), A(D_n)^{(1)}(T_p) on H^1(D_n) and the Gram matrix of the PD
// pairing between them -- built exactly as HeckeDualityValidation's arm A13
// builds them (the same public objects; that TU's helpers are file-local).
// Returns false when X_3 = 0 (the item is then not applicable).
bool FillItem7(IdentityCertificate& cert, uint32_t n, uint32_t ell) {
    const hecke::ContragredientCell cell(n, ell);
    const hecke::PoincareDualityPairing pd(cell);
    const FlatMatrix& G = pd.gram(1);
    if (G.get_rows() == 0) return false;
    MatrixT<uint32_t> Gm(G.get_rows(), G.get_cols(), G.get_p());
    for (uint32_t i = 0; i < G.get_rows(); ++i)
        for (uint32_t j = 0; j < G.get_cols(); ++j) Gm.data[i][j] = G.get_entry(i, j);
    for (uint32_t p : {2u, 3u, 5u, 7u}) {
        if (p % ell == 0) continue;   // the good-prime algebra
        hecke::ContragredientCell::HeckePair hp = cell.hecke_pair(p);
        hecke::HeckeBoundaryTauTable taub(cell.K(), cell.model(), *hp.cosets);
        hecke::HeckePairHomotopy eta(*hp.tau, taub);
        hecke::HeckeBoundaryOperator bd(&pd.pair_M().boundary_cohomology(), hp.cosets.get(), &taub);
        bd.compute();
        hecke::HeckeConeAction act(&pd.cone(), hp.op_M.get(), &bd, &eta);
        act.compute();
        cert.pd_ops_V.push_back(act.A_c(1));
        cert.pd_ops_W.push_back(hp.op_D->get_H_matrices()[1]);
    }
    cert.pd_pairing = Gm;
    return true;
}

bool IsItem7Cell(uint32_t n, uint32_t ell) {
    // The cells of HeckeDualityValidation's list at ell in {5, 7, 11}, plus
    // (22, 5) (the negative-control cell) and a Dickson-locus cell per ell.
    static const std::pair<uint32_t, uint32_t> cells[] = {
        {6, 5}, {12, 5}, {22, 5}, {4, 7}, {8, 7}, {10, 7}, {12, 11}, {14, 11}};
    for (const auto& c : cells)
        if (c.first == n && c.second == ell) return true;
    return false;
}

// ===========================================================================
// [A] The Glover factor multiset, two implementations.
// ===========================================================================
void RunGloverArm() {
    std::cout << "-- [A] the Glover factors: the certificate's recursion vs GloverFactors\n";
    uint32_t cells = 0;
    for (uint32_t ell : {5u, 7u, 11u, 13u}) {
        for (uint32_t n = 0; n <= 200; n += 2) {
            const GloverDecomposition g = GloverFactors(n, ell);
            const std::vector<IdentityGloverFactor> s =
                IdentityGloverFactors(n, ell, GloverConvention::kStandard);
            const std::vector<IdentityGloverFactor> c =
                IdentityGloverFactors(n, ell, GloverConvention::kContragredient);
            bool same = (g.factors.size() == s.size());
            uint32_t dim = 0;
            for (size_t i = 0; same && i < s.size(); ++i) {
                same = g.factors[i].sym == s[i].sym && g.factors[i].det_twist == s[i].det_twist
                       && g.factors[i].multiplicity == s[i].multiplicity;
                dim += (s[i].sym + 1) * s[i].multiplicity;
            }
            Check(same, Tag(n, ell) + ": IdentityGloverFactors == GloverFactors");
            Check(dim == n + 1, Tag(n, ell) + ": sum (n_i + 1) = n + 1");
            // The contragredient reading: every j_i shifted by -n.
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> want, got;
            for (const auto& f : s)
                want[{f.sym, (f.det_twist + (ell - 1) - n % (ell - 1)) % (ell - 1)}] +=
                    f.multiplicity;
            for (const auto& f : c) got[{f.sym, f.det_twist}] += f.multiplicity;
            Check(want == got, Tag(n, ell) + ": contragredient = standard shifted by -n");
            // The central character: n_i + 2 j_i == n (standard), == -n (dual).
            for (const auto& f : s)
                Check((f.sym + 2 * f.det_twist) % (ell - 1) == n % (ell - 1),
                      Tag(n, ell) + ": n_i + 2 j_i == n mod (ell - 1)");
            ++cells;
        }
    }
    std::cout << "    " << cells << " (n, ell) cells compared\n";
}

// ===========================================================================
// [B] + [C] Identity 6 with items 1-5, and the negative control.
// ===========================================================================

struct IdentityTotals {
    uint32_t cells = 0, item_pass[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t item7_cells = 0;
    uint32_t nonvacuous = 0, control_failed = 0;
    bool named_seen = false, named_nonvacuous = false;
};

const uint32_t kNamedN = 22, kNamedEll = 5;

void RunIdentityArm(IdentityTotals& tot) {
    std::cout << "-- [B] identity 6 (certificate item 6) with items 1-5; [C] the negative control\n";
    std::cout << "   convention (quoted verbatim from [E] cl:glover): \"Sym^n is the symmetric power"
                 " of the standard representation -- the module rho_n of def:rhon acts on, in which"
                 " diag(a,d) has entry(t,t)=a^{n-t}d^t -- and not of its dual; the dual would shift"
                 " every det-twist by -n.\"\n";

    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off);

    for (uint32_t ell : {5u, 7u, 11u}) {
        const Clock::time_point t_ell = Clock::now();
        const std::vector<uint64_t> T = Pool(ell);

        // The atlas Sigma(ell), for locating the classes.
        const Atlas atlas = BuildAtlas(ell);
        const AtlasStore& st = atlas.store;
        Check(st.primes == T, "ell=" + std::to_string(ell) + ": the atlas pool is T");

        // The weight cells m = 0, 2, ..., ell - 1.
        std::vector<std::vector<ResidueCharacterMultiset>> weights(ell);
        for (uint32_t m = 0; m + 1 < ell + 1; m += 2) {
            if (m > ell - 1) break;
            std::unique_ptr<HeckeExtDriver> d = Run(m, ell, T, {}, false);
            std::string why;
            const bool ok = AtlasResidueMultisets(AtlasIdealCharacters(*d), weights[m], why);
            Check(ok, "weight cell " + Tag(m, ell) + ": multisets (" + why + ")");
            // The atlas's own cell agrees in dimensions.
            for (const AtlasCell& c : st.cells)
                if (c.n == m)
                    Check(c.dims == d->H_dimensions() && c.local_ideals == c.ideals,
                          "weight cell " + Tag(m, ell) + ": atlas cell dims and locality");
        }

        std::vector<uint32_t> ns;
        for (uint32_t n = 0; n <= 60; n += 2) ns.push_back(n);
        for (uint32_t n : {80u, 100u, 120u, 160u, 200u}) ns.push_back(n);

        std::map<uint32_t, CanonicalField> fields;
        auto field = [&](uint32_t r) -> const CanonicalField& {
            auto it = fields.find(r);
            if (it == fields.end()) it = fields.emplace(r, CanonicalField(ell, r)).first;
            return it->second;
        };

        uint32_t in_sigma = 0, as_chi = 0, h1_classes = 0;
        for (uint32_t n : ns) {
            const std::string tag = Tag(n, ell);
            std::unique_ptr<HeckeExtDriver> d = Run(n, ell, T, {}, false);
            const std::vector<uint32_t>& dims = d->H_dimensions();
            std::vector<ResidueCharacterMultiset> cell;
            std::string why;
            const bool ok = AtlasResidueMultisets(AtlasIdealCharacters(*d), cell, why);
            Check(ok, tag + ": the cell's multisets (" + why + ")");
            if (!ok || dims.size() != 3) continue;

            IdentityCertificate cert;
            cert.n = n;
            cert.ell = ell;
            cert.dim_H0 = dims[0];
            cert.dim_H1 = dims[1];
            cert.dim_H2 = dims[2];
            cert.p = 2;
            cert.p_prime = 3;
            const bool full_items = (n <= 40) && dims[1] > 0;
            std::unique_ptr<Stack> stack;
            if (full_items) {
                stack.reset(new Stack(n, ell));
                hecke::HeckeOperatorFamily fam(&stack->coh, stack->K, stack->model, ell, n);
                cert.A_p = fam.T_p(2)[1];
                cert.A_p_prime = fam.T_p(3)[1];
                cert.A_p2 = fam.T_p_pow_direct(2, 2)[1];
                if (DicksonMultiplicationApplicable(n, ell)) {
                    const uint32_t m = DicksonSourceDegree(n, ell);
                    Stack source(m, ell);
                    const auto& dm = source.coh.get_H_eq_dimensions();
                    if (dm.size() > 1 && dm[1] > 0) {
                        hecke::HeckeOperatorFamily fam_m(&source.coh, source.K, source.model, ell, m);
                        cert.n_source = m;
                        cert.A_p_source = fam_m.T_p(2)[1];
                        cert.L_star = InducedDickson(*stack, source, n, ell);
                    }
                }
            } else if (dims[1] > 0) {
                for (const auto& op : d->cell_operators(1)) {
                    if (op.first == 2) cert.A_p = op.second;
                    if (op.first == 3) cert.A_p_prime = op.second;
                }
            }
            if (IsItem7Cell(n, ell)) {
                const bool nontrivial = FillItem7(cert, n, ell);
                Check(nontrivial, tag + ": item 7 cell has X_3 != 0");
                if (nontrivial) ++tot.item7_cells;
            }
            cert.residue_primes = T;
            cert.residue_cell = cell;
            cert.residue_weights = weights;

            const uint32_t code = cert.verify();
            Check(code == 0, tag + ": IdentityCertificate::verify() = " + std::to_string(code)
                                 + "\n" + cert.to_string_offset(4));
            Check(cert.residue_atlas.status == IdentityStatus::kPassed,
                  tag + ": item 6 is kPassed (" + cert.residue_atlas.detail + ")");
            const IdentityItem* items[7] = {&cert.eichler_shimura, &cert.commutation,
                                            &cert.d2_vanishes, &cert.p2_recursion,
                                            &cert.dickson_multiplication, &cert.residue_atlas,
                                            &cert.socle_loewy_pd};
            if (IsItem7Cell(n, ell))
                Check(cert.socle_loewy_pd.status == IdentityStatus::kPassed,
                      tag + ": item 7 (socle <-> Loewy under PD) is kPassed ("
                          + cert.socle_loewy_pd.detail + ")");
            for (int k = 0; k < 7; ++k)
                if (items[k]->status == IdentityStatus::kPassed) ++tot.item_pass[k + 1];
            ++tot.cells;

            // Every H^1 class is an atlas system (or a chi_{a,b}).
            for (const ResidueCharacterClass& c : cell[1]) {
                ++h1_classes;
                std::vector<FieldElt> v(c.orbit[0].begin(), c.orbit[0].end());
                const std::vector<FieldElt> key = AtlasOrbitKey(field(c.r), v);
                if (AtlasFind(st, c.r, key) >= 0) { ++in_sigma; continue; }
                bool chi = false;
                if (c.r == 1) {
                    for (uint32_t a = 0; a + 1 < ell && !chi; ++a)
                        for (uint32_t b = a + 1; b + 1 < ell && !chi; ++b) {
                            bool eq = true;
                            for (size_t i = 0; i < T.size() && eq; ++i)
                                eq = key[i][0] == (PowMod(T[i], a, ell) + PowMod(T[i], b, ell)) % ell;
                            chi = eq;
                        }
                }
                if (chi) ++as_chi;
                Check(chi, tag + ": an H^1 class outside Sigma(ell) u chi: " + ClassText(c));
            }

            // [C] the negative control.
            const std::vector<ResidueDifference> rs =
                IdentityResidueRhs(cert, GloverConvention::kStandard);
            const std::vector<ResidueDifference> rc =
                IdentityResidueRhs(cert, GloverConvention::kContragredient);
            bool differ = rs.size() != rc.size();
            for (size_t i = 0; !differ && i < rs.size(); ++i)
                differ = rs[i].delta != rc[i].delta || rs[i].cls.r != rc[i].cls.r
                         || rs[i].cls.orbit != rc[i].cls.orbit;
            IdentityCertificate wrong = cert;
            wrong.residue_convention = GloverConvention::kContragredient;
            const uint32_t wcode = wrong.verify();
            if (differ) {
                ++tot.nonvacuous;
                Check(wcode == 6 && wrong.residue_atlas.status == IdentityStatus::kFailed,
                      tag + ": the contragredient reading FAILS item 6 (verify() = "
                          + std::to_string(wcode) + ")");
                if (wcode == 6) ++tot.control_failed;
            } else {
                Check(wcode == 0, tag + ": vacuous cell -- the two readings agree, so both pass");
            }
            if (n == kNamedN && ell == kNamedEll) {
                tot.named_seen = true;
                tot.named_nonvacuous = differ;
                std::cout << "    NEGATIVE_CONTROL cell=" << tag << " dims=" << dims[0] << ","
                          << dims[1] << "," << dims[2] << "\n      standard factors:       "
                          << FactorText(IdentityGloverFactors(n, ell, GloverConvention::kStandard))
                          << "\n      contragredient factors: "
                          << FactorText(IdentityGloverFactors(n, ell,
                                                              GloverConvention::kContragredient))
                          << "\n      H^1 classes (LHS): ";
                for (const ResidueCharacterClass& c : cell[1])
                    std::cout << ClassText(c) << "x" << c.multiplicity << " ";
                std::cout << "\n      contragredient RHS - LHS:";
                for (const ResidueDifference& x :
                     IdentityResidueDifference(cert, GloverConvention::kContragredient))
                    std::cout << " " << (x.delta > 0 ? "+" : "") << x.delta << "*"
                              << ClassText(x.cls);
                std::cout << "\n      standard verify()=" << code
                          << " contragredient verify()=" << wcode << "\n";
            }
        }
        std::cout << "    ell=" << ell << ": " << ns.size() << " cells, H^1 classes " << h1_classes
                  << " (in Sigma " << in_sigma << ", chi " << as_chi << "), atlas |Sigma|="
                  << st.systems.size() << ", " << std::fixed << std::setprecision(1)
                  << Since(t_ell) << " s\n";
    }
}

// ===========================================================================
// [E] + [F] S_sep / Stage 2(3), and the proof_range comparison.
// ===========================================================================

struct Atlases {
    std::map<uint32_t, Atlas> by_ell;
    const Atlas& get(uint32_t ell) {
        auto it = by_ell.find(ell);
        if (it == by_ell.end()) it = by_ell.emplace(ell, BuildAtlas(ell)).first;
        return it->second;
    }
};

struct SsepTotals {
    uint32_t cells = 0, vacuous = 0, split = 0, unavailable = 0;
};
struct ProofTotals {
    uint32_t ideals = 0, not_dickson = 0, dickson = 0, mixed = 0, undecided = 0,
             lowered = 0, final_flips = 0, unavailable_cells = 0, h0_ideals = 0,
             cell_checked = 0, cell_flips = 0;   // (slot-7 WP-3) the default run
};

std::vector<uint32_t> CountsByDegree(const HeckeExtRecord& r) {
    std::vector<uint32_t> c(3, 0);
    for (const IdealRecord& id : r.ideals)
        for (const IdealGroupMatch& g : id.groups)
            if (g.own && g.group.size() == 2 && g.group[1] >= '0' && g.group[1] <= '2')
                ++c[g.group[1] - '0'];
    return c;
}

std::string CountsText(const std::vector<uint32_t>& c) {
    return std::to_string(c[0]) + "/" + std::to_string(c[1]) + "/" + std::to_string(c[2]);
}

// The restriction of a full-pool character to the primes S (both sorted).
std::vector<std::vector<FieldElt>> Restrict(const AtlasIdealCharacter& t,
                                            const std::vector<uint64_t>& S) {
    std::vector<size_t> idx;
    for (uint64_t q : S) {
        const auto it = std::find(t.primes.begin(), t.primes.end(), q);
        ENGINE_CHECK_MSG(it != t.primes.end(), "Restrict: a prime outside the pool");
        idx.push_back(static_cast<size_t>(it - t.primes.begin()));
    }
    std::vector<std::vector<FieldElt>> out;
    for (const auto& v : t.orbit) {
        std::vector<FieldElt> w;
        for (size_t i : idx) w.push_back(v[i]);
        out.push_back(std::move(w));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<std::vector<FieldElt>> Dedup(std::vector<std::vector<FieldElt>> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// One compared cell: `before` = the caller's run, `after` = + S_sep, `full`
// = the full anemic pool (u_ell off), which the atlas certifies separates.
void CompareCell(const std::string& set, uint32_t n, uint32_t ell,
                 const std::vector<uint64_t>& good, bool harness_u_off, Atlases& atlases,
                 SsepTotals& st, ProofTotals& pt) {
    const std::string tag = Tag(n, ell);
    const Atlas& atlas = atlases.get(ell);
    const std::vector<uint64_t>& s_sep = atlas.store.s_sep.primes;
    const std::vector<uint64_t> T = Pool(ell);

    std::unique_ptr<HeckeExtDriver> before, after, full;
    {
        hecke::UEllSetting off;
        off.mode = hecke::UEllMode::Off;
        std::unique_ptr<hecke::ScopedUEll> g;
        if (harness_u_off) g.reset(new hecke::ScopedUEll(off));
        before = Run(n, ell, good, {}, true);
        after = Run(n, ell, good, s_sep, true);
    }
    {
        hecke::UEllSetting off;
        off.mode = hecke::UEllMode::Off;
        const hecke::ScopedUEll g(off);
        full = Run(n, ell, T, {}, false);
    }
    const HeckeExtRecord& rb = before->record();
    const HeckeExtRecord& ra = after->record();
    Check(rb.verify() == 0 && ra.verify() == 0, tag + ": both records verify()");

    // S_0 as run (ell removed, U_ell appended iff admitted).
    std::vector<uint64_t> s0 = rb.operators_computed;
    std::vector<uint64_t> s0_anemic;
    for (uint64_t q : s0)
        if (q != ell) s0_anemic.push_back(q);
    bool vacuous = true;
    for (uint64_t q : s_sep)
        if (std::find(s0_anemic.begin(), s0_anemic.end(), q) == s0_anemic.end()) vacuous = false;

    const std::vector<uint32_t> cb = CountsByDegree(rb), ca = CountsByDegree(ra),
                                cf = CountsByDegree(full->record());
    const bool splits = (cb != ca);
    ++st.cells;
    if (vacuous) {
        ++st.vacuous;
        Check(BuildHeckeExtReportJson(rb) == BuildHeckeExtReportJson(ra),
              tag + ": S_sep subset of S_0 -- the record is BYTE-IDENTICAL with the option set");
    }
    if (splits) ++st.split;
    // Completeness on the anemic side ([E] alg:decomp Stage 2(3)): S_0 u S_sep
    // decomposes exactly as the full pool, degree by degree.
    const bool after_local = std::all_of(ra.ideals.begin(), ra.ideals.end(),
                                         [](const IdealRecord& i) { return i.local; });
    if (after_local && !ra.u_ell.included)
        Check(ca == cf, tag + ": S_0 u S_sep decomposes as the full pool (" + CountsText(ca)
                            + " vs " + CountsText(cf) + ")");
    const AtlasSeparation sep0 = AtlasSeparationBy(atlas.store, s0_anemic);
    std::cout << "    SSEP set=" << set << " cell=" << tag << " S0=" << Primes(s0)
              << " S_sep=" << Primes(s_sep) << " S0_separates=" << (sep0.separates ? 1 : 0)
              << " ideals(H0/H1/H2) " << CountsText(cb) << " -> " << CountsText(ca)
              << " full=" << CountsText(cf) << " vacuous=" << (vacuous ? 1 : 0)
              << " split=" << (splits ? 1 : 0) << "\n";

    // [F] proof_range, H^1 ideals of `before`.
    const std::vector<AtlasIdealCharacter> cb_chars = AtlasIdealCharacters(*before);
    const std::vector<AtlasIdealCharacter> cf_chars = AtlasIdealCharacters(*full);
    // What the atlas's decision would do to `final`: the SAME run with the
    // caller's proof_range lowered (it enters through the Sturm sweep's
    // outcome, not through is_final alone), compared ideal by ideal.  A
    // diagnostic run only -- nothing is fed back.
    std::unique_ptr<HeckeExtDriver> lowered;
    {
        hecke::UEllSetting off;
        off.mode = hecke::UEllMode::Off;
        std::unique_ptr<hecke::ScopedUEll> g;
        if (harness_u_off) g.reset(new hecke::ScopedUEll(off));
        lowered = Run(n, ell, good, {}, true, /*proof_range=*/false);
    }
    const HeckeExtRecord& rl = lowered->record();
    Check(rl.ideals.size() == rb.ideals.size(),
          tag + ": proof_range = false decomposes identically (it enters Stage 4 only)");
    // (slot-7 WP-3, stage 2 item 3) The DEFAULT run decides the range at the
    // cell (no atlas: the joint pool decomposition of V (+) H^0).  The atlas
    // verdict below is its independent oracle: `cell_local` true exactly on
    // `not_dickson_type`, and the default `final` is the caller run's where
    // the range holds and the lowered run's where it does not.
    std::unique_ptr<HeckeExtDriver> cell;
    {
        hecke::UEllSetting off;
        off.mode = hecke::UEllMode::Off;
        std::unique_ptr<hecke::ScopedUEll> g;
        if (harness_u_off) g.reset(new hecke::ScopedUEll(off));
        ExtDriverOptions o;
        o.good_primes = good;
        cell.reset(new HeckeExtDriver(n, ell, o));
        cell->run();
    }
    const HeckeExtRecord& rc = cell->record();
    Check(rc.verify() == 0 && rc.ideals.size() == rb.ideals.size(),
          tag + ": the cell-local run verifies and decomposes identically");
    for (size_t i = 0; i < rb.ideals.size(); ++i) {
        const IdealRecord& id = rb.ideals[i];
        const AtlasIdealCharacter& m = cb_chars[i];
        if (m.degree == 0) { ++pt.h0_ideals; continue; }
        if (m.degree != 1) continue;
        ++pt.ideals;
        std::string verdict;
        if (!m.local) {
            verdict = "undecided(not_local)";
        } else {
            const std::vector<std::vector<FieldElt>> mo = Dedup(m.orbit);
            // The full-pool H^1 ideals refining m, and whether each is Dickson-type.
            uint32_t refine_dim = 0, dick = 0, nondick = 0;
            bool field_mismatch = false;
            for (const AtlasIdealCharacter& t : cf_chars) {
                if (t.degree != 1 || !t.local) continue;
                if (Restrict(t, m.primes) != mo) continue;
                if (t.r != m.r) field_mismatch = true;
                refine_dim += t.dim;
                bool is_d = false;
                for (const AtlasIdealCharacter& h : cf_chars)
                    if (h.degree == 0 && h.local && h.r == t.r && h.orbit == t.orbit) is_d = true;
                (is_d ? dick : nondick) += 1;
            }
            // Every before-ideal with the same anemic restriction, summed.
            uint32_t same_dim = 0;
            for (const AtlasIdealCharacter& o : cb_chars)
                if (o.degree == 1 && o.local && Dedup(o.orbit) == mo) same_dim += o.dim;
            Check(refine_dim == same_dim, tag + ": ideal " + std::to_string(id.id)
                                              + " refines to the full pool dimension-exactly ("
                                              + std::to_string(same_dim) + " vs "
                                              + std::to_string(refine_dim) + ")");
            if (field_mismatch) verdict = "undecided(field)";
            else if (dick > 0 && nondick == 0) verdict = "dickson_type";
            else if (dick == 0 && nondick > 0) verdict = "not_dickson_type";
            else if (dick > 0) verdict = "mixed";
            else verdict = "undecided(no_refinement)";
        }
        const bool caller = id.sturm.present ? id.sturm.proof_range : true;
        const bool lower = caller && verdict != "not_dickson_type";
        bool flip = false;
        std::string f_after = ToString(id.final.value);
        if (lower && i < rl.ideals.size()) {
            const Finality& f2 = rl.ideals[i].final;
            Check(rl.ideals[i].id == id.id && rl.ideals[i].dim == id.dim,
                  tag + ": the lowered run's ideal " + std::to_string(id.id) + " aligns");
            flip = (f2 != id.final);
            f_after = ToString(f2.value) + (f2.reason.empty() ? std::string() : "(" + f2.reason + ")");
        }
        // (slot-7 WP-3) the default run's cell-local decision against the atlas.
        {
            const IdealRecord& ic = rc.ideals[i];
            const bool decided_by_atlas = (verdict == "not_dickson_type" ||
                                           verdict == "dickson_type" || verdict == "mixed");
            if (ic.sturm.present) {
                const std::string& src = ic.sturm.proof_range_source;
                Check(src == kProofRangeCellLocal || src == kProofRangeMethodUndefined,
                      tag + ": ideal " + std::to_string(id.id)
                          + ": the default run's source is cell_local or method_undefined");
                if (src == kProofRangeCellLocal && decided_by_atlas) {
                    ++pt.cell_checked;
                    Check(ic.sturm.proof_range == (verdict == "not_dickson_type"),
                          tag + ": ideal " + std::to_string(id.id)
                              + ": cell_local proof_range agrees with the atlas (" + verdict
                              + ")");
                    const Finality& want = ic.sturm.proof_range ? id.final : rl.ideals[i].final;
                    Check(ic.final == want,
                          tag + ": ideal " + std::to_string(id.id)
                              + ": the default final is the re-run Sturm route's");
                }
            }
            const bool cflip = !(ic.final == id.final);
            if (cflip) ++pt.cell_flips;
            std::cout << "    CELL_LOCAL set=" << set << " cell=" << tag << " ideal=" << id.id
                      << " source="
                      << (ic.sturm.present ? ic.sturm.proof_range_source : std::string("-"))
                      << " proof_range="
                      << (!ic.sturm.present ? "-"
                                            : (!ic.sturm.has_proof_range
                                                       ? "null"
                                                       : (ic.sturm.proof_range ? "true"
                                                                               : "false")))
                      << " final " << ToString(id.final.value) << " -> "
                      << ToString(ic.final.value)
                      << (ic.final.reason.empty() ? std::string() : "(" + ic.final.reason + ")")
                      << " flip=" << (cflip ? 1 : 0) << "\n";
        }
        if (verdict == "not_dickson_type") ++pt.not_dickson;
        else if (verdict == "dickson_type") ++pt.dickson;
        else if (verdict == "mixed") ++pt.mixed;
        else ++pt.undecided;
        if (lower) ++pt.lowered;
        if (flip) ++pt.final_flips;
        std::cout << "    PROOF_RANGE set=" << set << " cell=" << tag << " ideal=" << id.id
                  << " dim=" << id.dim << " cert=" << id.certificate << " routes(a,b,c)="
                  << id.certificate_routes.a << "," << id.certificate_routes.b << ","
                  << id.certificate_routes.c
                  << " caller=" << (caller ? "true" : "false") << " atlas=" << verdict
                  << " lower=" << (lower ? 1 : 0) << " final=" << ToString(id.final.value)
                  << (lower ? " -> " + f_after : std::string()) << " flip=" << (flip ? 1 : 0)
                  << "\n";
    }
}

// (slot-7 WP-3, stage 2 item 3; owner ruling 3a "measured first") A cell with
// no atlas: the old default (caller_input true) against the new one
// (cell_local), by RE-RUNNING the driver -- the flip is what the Sturm route
// does with the decided range, never a flag flipped on `is_final`.
void MeasureCellLocalOnly(const std::string& set, uint32_t n, uint32_t ell,
                          const std::vector<uint64_t>& good, bool harness_u_off,
                          ProofTotals& pt) {
    const std::string tag = Tag(n, ell);
    std::unique_ptr<HeckeExtDriver> caller, cell;
    {
        hecke::UEllSetting off;
        off.mode = hecke::UEllMode::Off;
        std::unique_ptr<hecke::ScopedUEll> g;
        if (harness_u_off) g.reset(new hecke::ScopedUEll(off));
        caller = Run(n, ell, good, {}, true, /*proof_range=*/true);
        ExtDriverOptions o;
        o.good_primes = good;
        cell.reset(new HeckeExtDriver(n, ell, o));
        cell->run();
    }
    const HeckeExtRecord& ra = caller->record();
    const HeckeExtRecord& rc = cell->record();
    Check(rc.verify() == 0 && rc.ideals.size() == ra.ideals.size(),
          tag + ": the cell-local run verifies and decomposes identically");
    for (size_t i = 0; i < rc.ideals.size() && i < ra.ideals.size(); ++i) {
        const IdealRecord& ic = rc.ideals[i];
        const IdealRecord& ia = ra.ideals[i];
        const bool cflip = !(ic.final == ia.final);
        if (cflip) ++pt.cell_flips;
        if (ic.sturm.present && ic.sturm.proof_range_source == kProofRangeCellLocal) {
            ++pt.cell_checked;
            // H^0 = 0 here (n < ell + 1 has no Dickson invariant), so no
            // system can be shared and the range must hold.
            if (cell->H_dimensions().empty() || cell->H_dimensions()[0] == 0u) {
                Check(ic.sturm.proof_range,
                      tag + ": H^0 = 0 -- cell_local proof_range is true");
            }
        }
        std::cout << "    CELL_LOCAL set=" << set << " cell=" << tag << " ideal=" << ic.id
                  << " source="
                  << (ic.sturm.present ? ic.sturm.proof_range_source : std::string("-"))
                  << " proof_range="
                  << (!ic.sturm.present
                              ? "-"
                              : (!ic.sturm.has_proof_range ? "null"
                                                           : (ic.sturm.proof_range ? "true"
                                                                                   : "false")))
                  << " H0=" << (cell->H_dimensions().empty() ? 0u : cell->H_dimensions()[0])
                  << " final " << ToString(ia.final.value) << " -> " << ToString(ic.final.value)
                  << (ic.final.reason.empty() ? std::string() : "(" + ic.final.reason + ")")
                  << " flip=" << (cflip ? 1 : 0) << " (no atlas)\n";
    }
}

void RunSsepArm(Atlases& atlases) {
    std::cout << "-- [E] S_sep / Stage 2(3) (ExtDriverOptions::s_sep) and [F] the proof_range "
                 "comparison\n";
    for (uint32_t ell : {5u, 7u, 11u, 13u}) {
        const Atlas& a = atlases.get(ell);
        std::cout << "    atlas ell=" << ell << " S_sep=" << Primes(a.store.s_sep.primes)
                  << " separates=" << (a.store.s_sep.separates ? 1 : 0) << "\n";
        Check(a.store.s_sep.separates, "ell=" + std::to_string(ell) + ": S_sep separates");
    }

    SsepTotals st_sub, st_m1;
    ProofTotals pt_sub, pt_m1;

    // The 31 golden ext-subgrid cells ([X] Sec. 1.6; GoldenHarness kGridNSym x
    // kGridEll in the subgrid, plus the Dickson cell (32, 5)), harness settings.
    std::vector<std::pair<uint32_t, uint32_t>> sub;
    for (uint32_t n : {2u, 4u, 10u, 16u, 22u, 24u})
        for (uint32_t ell : {5u, 13u, 691u, 10007u, 134217757u}) sub.push_back({n, ell});
    sub.push_back({32, 5});
    Check(sub.size() == 31, "the ext subgrid has 31 cells");
    for (const auto& c : sub) {
        if (c.second > 13) {
            ++st_sub.unavailable;
            ++pt_sub.unavailable_cells;
            std::cout << "    SSEP set=subgrid cell=" << Tag(c.first, c.second)
                      << " atlas unavailable (ell > 13; R12 caps the atlas at ell <= 1000, and"
                         " 691 is slot 7's)\n";
            MeasureCellLocalOnly("subgrid", c.first, c.second, {2, 3, 5}, true, pt_sub);
            continue;
        }
        CompareCell("subgrid", c.first, c.second, {2, 3, 5}, true, atlases, st_sub, pt_sub);
    }

    // The M1 case list (HeckeExtSmallEll / Separation / Kummer), default settings.
    const std::vector<std::pair<uint32_t, uint64_t>> m1 = {
        {6, 5},    {8, 7},    {12, 11},  {12, 5},    {34, 5},       {58, 7},     {58, 5},
        {82, 7},   {22, 5},   {26, 7},   {32, 131},  {160, 691},    {198, 691},  {10, 691},
        {14, 3617}, {16, 43867}, {18, 283}, {18, 617}, {20, 131}, {20, 593}, {22, 103},
        {22, 2294797}, {24, 657931},
    };
    for (const auto& c : m1) {
        if (c.second > 13) {
            ++st_m1.unavailable;
            ++pt_m1.unavailable_cells;
            std::cout << "    SSEP set=m1 cell=" << Tag(c.first, c.second)
                      << " atlas unavailable (ell > 13)\n";
            continue;
        }
        CompareCell("m1", c.first, static_cast<uint32_t>(c.second), {}, false, atlases, st_m1,
                    pt_m1);
    }

    auto print_s = [](const char* set, const SsepTotals& s) {
        std::cout << "    SSEP_SUMMARY set=" << set << " compared=" << s.cells
                  << " vacuous=" << s.vacuous << " split=" << s.split
                  << " atlas_unavailable=" << s.unavailable << "\n";
    };
    auto print_p = [](const char* set, const ProofTotals& p) {
        std::cout << "    PROOF_RANGE_SUMMARY set=" << set << " H1_ideals=" << p.ideals
                  << " not_dickson=" << p.not_dickson << " dickson=" << p.dickson
                  << " mixed=" << p.mixed << " undecided=" << p.undecided
                  << " atlas_would_lower=" << p.lowered << " final_flips=" << p.final_flips
                  << " H0_ideals_not_compared=" << p.h0_ideals
                  << " atlas_unavailable_cells=" << p.unavailable_cells
                  << " cell_local_checked=" << p.cell_checked
                  << " cell_local_final_flips=" << p.cell_flips << "\n";
    };
    print_s("subgrid", st_sub);
    print_s("m1", st_m1);
    print_p("subgrid", pt_sub);
    print_p("m1", pt_m1);
    Check(st_sub.cells + st_sub.unavailable == 31, "subgrid: every cell has a row");
    Check(st_sub.unavailable == 18, "subgrid: 18 cells at ell in {691, 10007, 134217757}");
}
// The constructed arm: one-prime S_0' = {q} that the atlas says does NOT
// separate Sigma u chi, so that Stage 2(3) has something to do.  Over
// n <= 30 and every such q outside S_sep, a cell where {q} merges ideals
// that the full pool separates (a pre-check against the one full-pool run
// per cell) is re-run as {q} u S_sep, which must decompose as the full pool.
void RunConstructedSsepArm(Atlases& atlases) {
    std::cout << "-- [E'] Stage 2(3) where it can act: one-prime S_0' = {q}\n";
    hecke::UEllSetting off;
    off.mode = hecke::UEllMode::Off;
    const hecke::ScopedUEll u_guard(off);
    uint32_t acted = 0, tried = 0;
    for (uint32_t ell : {5u, 7u, 11u, 13u}) {
        const Atlas& a = atlases.get(ell);
        const std::vector<uint64_t>& s_sep = a.store.s_sep.primes;
        std::vector<uint64_t> qs;
        for (uint64_t q : Pool(ell)) {
            if (q > 13) break;
            if (std::find(s_sep.begin(), s_sep.end(), q) != s_sep.end()) continue;
            if (!AtlasSeparationBy(a.store, {q}).separates) qs.push_back(q);
        }
        std::cout << "    ell=" << ell << " non-separating one-prime S_0' outside S_sep: "
                  << Primes(qs) << "\n";
        for (uint32_t n = 2; n <= 30; n += 2) {
            std::unique_ptr<HeckeExtDriver> f = Run(n, ell, Pool(ell), {}, false);
            const std::vector<uint32_t> cf = CountsByDegree(f->record());
            for (uint64_t q : qs) {
                std::unique_ptr<HeckeExtDriver> b = Run(n, ell, {q}, {}, false);
                const std::vector<uint32_t> cb = CountsByDegree(b->record());
                ++tried;
                if (cb == cf) continue;   // {q} already separates this cell
                std::unique_ptr<HeckeExtDriver> s = Run(n, ell, {q}, s_sep, false);
                const std::vector<uint32_t> cs = CountsByDegree(s->record());
                const bool all_local =
                    std::all_of(s->record().ideals.begin(), s->record().ideals.end(),
                                [](const IdealRecord& i) { return i.local; });
                Check(all_local && cs == cf,
                      Tag(n, ell) + ": {" + std::to_string(q)
                          + "} u S_sep decomposes as the full pool");
                if (cb != cs) ++acted;
                std::cout << "    SSEP set=constructed cell=" << Tag(n, ell) << " S0'={" << q
                          << "} S_sep=" << Primes(s_sep) << " ideals(H0/H1/H2) "
                          << CountsText(cb) << " -> " << CountsText(cs)
                          << " full=" << CountsText(cf) << " split=" << (cb != cs ? 1 : 0)
                          << "\n";
            }
        }
    }
    std::cout << "    constructed: " << acted << " (cell, q) pairs split by Stage 2(3), of "
              << tried << " tried\n";
    Check(acted > 0, "the constructed arm: Stage 2(3) splits at least one cell (non-vacuous)");
}
} // namespace

uint32_t HeckeExtEndToEndValidation() {
    Check.reset();
    const Clock::time_point t0 = Clock::now();
    std::cout << "=== HeckeExtEndToEndValidation (slot 6, WP-9: identity 6, S_sep, proof_range) ===\n";
    try {
        RunGloverArm();

        IdentityTotals it;
        RunIdentityArm(it);
        std::cout << "    identity cells=" << it.cells << " passed: item1=" << it.item_pass[1]
                  << " item2=" << it.item_pass[2] << " item3=" << it.item_pass[3]
                  << " item4=" << it.item_pass[4] << " item5=" << it.item_pass[5]
                  << " item6=" << it.item_pass[6] << " item7=" << it.item_pass[7] << " (of "
                  << it.item7_cells << " item-7 cells)\n";
        std::cout << "    negative control: " << it.nonvacuous << " non-vacuous cells, "
                  << it.control_failed << " of them fail under the contragredient reading\n";
        Check(it.item_pass[6] == it.cells, "item 6 passed on every identity cell");
        Check(it.nonvacuous > 0 && it.control_failed == it.nonvacuous,
              "the negative control fails on every non-vacuous cell");
        Check(it.named_seen && it.named_nonvacuous,
              "the named negative-control cell " + Tag(kNamedN, kNamedEll) + " is non-vacuous");

        // [D] item 7 (WP-3 merged first; slot-6 plan Sec. 4.2 rule 9: asserted).
        Check(it.item7_cells == 8 && it.item_pass[7] == it.item7_cells,
              "[D] item 7 asserted and passed on every item-7 cell");

        Atlases atlases;
        RunSsepArm(atlases);
        RunConstructedSsepArm(atlases);
    } catch (const std::exception& e) {
        Check(false, std::string("exception: ") + e.what());
    }
    std::cout << "HeckeExtEndToEndValidation " << (Check.fails == 0 ? "PASSED" : "FAILED") << " ("
              << Check.checks << " checks, " << Check.fails << " failing, " << std::fixed
              << std::setprecision(1) << Since(t0) << " s)\n";
    return Check.fails;
}
