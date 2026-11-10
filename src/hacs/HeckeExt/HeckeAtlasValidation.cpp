// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// (hecke-extensions slot 6, WP-8 / phase E6) HeckeAtlasSkeletonValidation and
// HeckeAtlasValidation -- see the header for the arm list.
// ===========================================================================

#include "HeckeAtlasValidation.h"

#include "HeckeAtlas.h"
#include "HeckeTangentPair.h"
#include "GloverFactors.h"
#include "HeckePredictions.h"
#include "HeckePredictionsValidation.h"
#include "../Util/ValidationCheck.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hecke_ext;

namespace {

oc_util::Checker Check("  *** FAIL [atlas] ");

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

std::string KeyText(const std::vector<FieldElt>& k) {
    std::string s = "(";
    for (size_t i = 0; i < k.size(); ++i) {
        if (i) s += ",";
        if (k[i].size() == 1u) {
            s += std::to_string(k[i][0]);
        } else {
            s += "[";
            for (size_t j = 0; j < k[i].size(); ++j) s += (j ? "," : "") + std::to_string(k[i][j]);
            s += "]";
        }
    }
    return s + ")";
}

// Ramanujan's tau(q) for the pool's primes q <= 31 (OEIS A000594).  The table
// is self-checked below against tau(p) == 1 + p^11 (mod 691).
struct TauEntry { uint64_t q; int64_t tau; };
const TauEntry kTau[] = {
    {2, -24},        {3, 252},        {5, 4830},       {7, -16744},
    {11, 534612},    {13, -577738},   {17, -6905934},  {19, 10661420},
    {23, 18643272},  {29, 128406630}, {31, -52843168},
};

int64_t TauOf(uint64_t q) {
    for (const TauEntry& e : kTau)
        if (e.q == q) return e.tau;
    throw std::logic_error("tau table: no entry for q = " + std::to_string(q));
}

uint32_t TauMod(uint64_t q, uint32_t ell) {
    const int64_t t = TauOf(q) % static_cast<int64_t>(ell);
    return static_cast<uint32_t>(t < 0 ? t + static_cast<int64_t>(ell) : t);
}

void ArmTauTable() {
    for (const TauEntry& e : kTau) {
        const uint32_t lhs = TauMod(e.q, 691u);
        const uint32_t rhs = (1u + PowMod(e.q, 11u, 691u)) % 691u;
        Check(lhs == rhs, "tau table self-check: tau(" + std::to_string(e.q)
                              + ") != 1 + q^11 mod 691");
    }
}

// ---------------------------------------------------------------------------
// (A) closure, (B) separation, (C) round trip -- the per-ell structural arms.
// ---------------------------------------------------------------------------
void ArmClosure(const Atlas& a) {
    const AtlasStore& s = a.store;
    const uint32_t ell = s.ell;
    uint32_t key_checks = 0, poly_checks = 0, certs = 0;

    // On the keys: theta^j of every system is a system.
    for (const AtlasSystem& sys : s.systems) {
        const CanonicalField F(ell, sys.r);
        for (uint32_t j = 0; j + 1u < ell; ++j) {
            const std::vector<FieldElt> t = AtlasTwist(F, s.primes, sys.eigenvalues, j);
            Check(AtlasFind(s, sys.r, t) >= 0,
                  "Sigma(" + std::to_string(ell) + ") not closed: theta^" + std::to_string(j)
                      + " of system " + std::to_string(sys.id));
            ++key_checks;
        }
        // The recorded construction re-derives.
        const AtlasSystem& base = s.systems[sys.base];
        Check(base.collected && base.r == sys.r
                  && AtlasTwist(F, s.primes, base.eigenvalues, sys.twist) == sys.eigenvalues,
              "system " + std::to_string(sys.id) + " is not theta^twist of its base");
        Check(sys.collected == (sys.twist == 0u), "collected <=> twist 0");
    }

    // On the polynomial data, P_p |-> p^j P_p mod pi_w.
    for (const AtlasSource& src : a.sources) {
        Check(CertifyCanonicalSplit(src.split, src.character) == 0u,
              "CertifyCanonicalSplit on the source at n = " + std::to_string(src.n));
        ++certs;
        const AtlasSystem& own = s.systems[src.system];
        const uint32_t r = static_cast<uint32_t>(src.witness_poly.degree());
        Check(r == own.r, "source degree == system degree");
        const CanonicalField F(ell, r);

        std::vector<uint64_t> anemic;
        std::vector<hecke::FpPoly> chars;
        for (size_t i = 0; i < src.primes.size(); ++i) {
            if (src.primes[i] == ell) continue;
            anemic.push_back(src.primes[i]);
            chars.push_back(src.character[i]);
        }
        Check(anemic == s.primes, "the source's anemic primes are the pool");
        for (uint32_t j = 0; j + 1u < ell; ++j) {
            std::vector<hecke::FpPoly> tw;
            for (size_t i = 0; i < anemic.size(); ++i)
                tw.push_back(chars[i].scalar_mul(PowMod(anemic[i], j, ell)));
            const CanonicalSplit sp = SplitInCanonicalField(F, src.witness_poly, anemic, tw);
            Check(sp.ok, "twisted split failed");
            if (!sp.ok) continue;
            Check(CertifyCanonicalSplit(sp, tw) == 0u, "CertifyCanonicalSplit on a twist");
            ++certs;
            const std::vector<FieldElt> k_poly = AtlasOrbitKey(F, sp.eigenvalues);
            const std::vector<FieldElt> k_key = AtlasTwist(F, s.primes, own.eigenvalues, j);
            Check(k_poly == k_key, "twist on the polynomial data != twist on the key at n = "
                                       + std::to_string(src.n) + ", j = " + std::to_string(j));
            Check(AtlasFind(s, r, k_poly) >= 0, "a polynomial-data twist is not in Sigma");
            ++poly_checks;
        }
    }
    std::cout << "    (A) closure: " << key_checks << " key twists, " << poly_checks
              << " polynomial-data twists, " << certs << " canonical-split certificates\n";
}

void ArmSeparation(const Atlas& a) {
    const AtlasStore& s = a.store;
    Check(s.s_sep.separates, "S_sep(" + std::to_string(s.ell) + ") does not separate");
    const AtlasSeparation re = AtlasSeparationBy(s, s.s_sep.greedy_order);
    Check(re.separates && re.unseparated == 0u && re.undecided == 0u,
          "S_sep re-derived does not separate");
    Check(re.targets == s.s_sep.targets && re.pairs == s.s_sep.pairs, "targets / pairs re-derive");
    // Each greedy step strictly reduced the unseparated count.
    std::vector<uint64_t> prefix;
    uint64_t prev = AtlasSeparationBy(s, prefix).unseparated;
    for (uint64_t q : s.s_sep.greedy_order) {
        prefix.push_back(q);
        const uint64_t now = AtlasSeparationBy(s, prefix).unseparated;
        Check(now < prev, "a greedy step did not help");
        prev = now;
    }
    // Irredundancy is RECORDED (greedy does not promise it).
    bool irredundant = true;
    for (size_t drop = 0; drop < s.s_sep.primes.size(); ++drop) {
        std::vector<uint64_t> S;
        for (size_t i = 0; i < s.s_sep.primes.size(); ++i)
            if (i != drop) S.push_back(s.s_sep.primes[i]);
        if (AtlasSeparationBy(s, S).separates) irredundant = false;
    }
    // A non-separating control: the empty set separates nothing when there
    // are two targets.
    if (s.s_sep.targets >= 2u) Check(!AtlasSeparationBy(s, {}).separates, "empty set control");
    std::cout << "    (B) S_sep(" << s.ell << ") = {";
    for (size_t i = 0; i < s.s_sep.primes.size(); ++i) std::cout << (i ? "," : "") << s.s_sep.primes[i];
    std::cout << "} (greedy order";
    for (uint64_t q : s.s_sep.greedy_order) std::cout << " " << q;
    std::cout << ") separates " << s.s_sep.targets << " targets / " << s.s_sep.pairs
              << " pairs; irredundant = " << (irredundant ? "yes" : "no")
              << " [recorded]; anemic S_0 = {";
    for (size_t i = 0; i < s.s0_anemic.primes.size(); ++i)
        std::cout << (i ? "," : "") << s.s0_anemic.primes[i];
    std::cout << "} separates: " << (s.s0_anemic.separates ? "yes" : "no") << " ("
              << s.s0_anemic.unseparated << " pairs left, " << s.s0_anemic.undecided
              << " undecided) [recorded]\n";
}

void ArmRoundTrip(const Atlas& a) {
    const std::string j1 = a.store.to_json();
    AtlasStore back;
    bool parsed = true;
    try {
        back = ParseAtlasStore(j1);
    } catch (const std::exception& e) {
        parsed = false;
        std::cout << "    parse error: " << e.what() << "\n";
    }
    Check(parsed, "the store parses");
    if (!parsed) return;
    const std::string j2 = back.to_json();
    Check(j1 == j2, "to_json(parse(to_json(x))) != to_json(x)");
    Check(back.ell == a.store.ell && back.systems.size() == a.store.systems.size()
              && back.cells.size() == a.store.cells.size()
              && back.s_sep.primes == a.store.s_sep.primes,
          "parsed fields");
    for (size_t i = 0; i < back.systems.size() && i < a.store.systems.size(); ++i)
        Check(back.systems[i].eigenvalues == a.store.systems[i].eigenvalues
                  && back.systems[i].occurrences.size() == a.store.systems[i].occurrences.size(),
              "parsed system " + std::to_string(i));
    // A corrupted schema and a truncated text are refused, not half-read.
    std::string bad = j1;
    const size_t at = bad.find("hecke_atlas/v1");
    if (at != std::string::npos) bad[at + 13] = '0';
    bool refused = false;
    try { (void)ParseAtlasStore(bad); } catch (const std::exception&) { refused = true; }
    Check(refused, "a wrong schema string is refused");
    refused = false;
    try { (void)ParseAtlasStore(j1.substr(0, j1.size() / 2)); } catch (const std::exception&) { refused = true; }
    Check(refused, "a truncated store is refused");
    std::cout << "    (C) round trip: " << j1.size() << " bytes, byte-identical after parse\n";
}

void PrintSummary(const Atlas& a) {
    const AtlasStore& s = a.store;
    uint32_t eis = 0, cusp = 0, collected = 0;
    for (const AtlasSystem& sys : s.systems) {
        if (sys.type == "eisenstein") ++eis; else ++cusp;
        if (sys.collected) ++collected;
    }
    std::cout << "  Sigma(" << s.ell << "): " << s.systems.size() << " systems (" << collected
              << " collected, " << eis << " eisenstein, " << cusp << " cuspidal); chi "
              << s.chi_in_sigma << "/" << s.chi_count << " in Sigma; pool T = {";
    for (size_t i = 0; i < s.primes.size(); ++i) std::cout << (i ? "," : "") << s.primes[i];
    std::cout << "}; fields r =";
    for (const AtlasField& f : s.fields) std::cout << " " << f.r;
    std::cout << "; build " << std::fixed << std::setprecision(3) << a.build_s << " s\n";
    std::cout.unsetf(std::ios::floatfield);
    for (const AtlasSystem& sys : s.systems) {
        if (!sys.collected) continue;
        std::cout << "    system " << sys.id << " r=" << sys.r << " " << sys.type;
        if (sys.has_chi) std::cout << " chi(" << sys.chi_a << "," << sys.chi_b << ")";
        std::cout << " a_T=" << KeyText(sys.eigenvalues) << " at";
        for (const AtlasOccurrence& o : sys.occurrences) {
            std::cout << " n=" << o.n << "/" << o.group << "/dim" << o.dim;
            if (o.a_ell.known) {
                std::cout << "/a_ell=[";
                for (size_t i = 0; i < o.a_ell.value.size(); ++i)
                    std::cout << (i ? "," : "") << o.a_ell.value[i];
                std::cout << "]";
            } else {
                std::cout << "/a_ell=unknown(" << o.a_ell.reason << ")";
            }
        }
        std::cout << "\n";
    }
}

// (I) the per-cell column.
void PrintCellColumn(const Atlas& a) {
    std::cout << "    (I) per-cell  n  dims      ideals  r_max  total_s    canonical_field_s  cf/total\n";
    for (size_t i = 0; i < a.timings.size(); ++i) {
        const AtlasCellTiming& t = a.timings[i];
        const AtlasCell& c = a.store.cells[i];
        std::string dims;
        for (size_t k = 0; k < c.dims.size(); ++k) dims += (k ? "," : "") + std::to_string(c.dims[k]);
        std::cout << "        ell=" << std::setw(2) << a.store.ell << " " << std::setw(3) << t.n
                  << "  " << std::left << std::setw(9) << dims << std::right << " "
                  << std::setw(6) << c.local_ideals << "  " << std::setw(5) << t.r_max << "  "
                  << std::scientific << std::setprecision(3) << t.total_s << "  "
                  << t.canonical_field_s << "          "
                  << (t.total_s > 0.0 ? t.canonical_field_s / t.total_s : 0.0) << "\n";
        std::cout.unsetf(std::ios::floatfield);
    }
}

// ---------------------------------------------------------------------------
// (E)-(G): the mathematical content at ell in {5, 7, 11, 13}.
// ---------------------------------------------------------------------------

// The system with a_q = q^a + q^b on the pool, or null.
const AtlasSystem* FindChi(const AtlasStore& s, uint32_t a, uint32_t b) {
    std::vector<FieldElt> k;
    for (uint64_t q : s.primes) k.push_back(FieldElt{(PowMod(q, a, s.ell) + PowMod(q, b, s.ell)) % s.ell});
    const int64_t id = AtlasFind(s, 1u, k);
    return id < 0 ? nullptr : &s.systems[static_cast<size_t>(id)];
}

void ArmEisenstein(const Atlas& a) {
    const AtlasStore& s = a.store;
    const uint32_t ell = s.ell, m = ell - 1u;
    uint32_t cells = 0;
    for (const AtlasCell& c : s.cells) {
        // dim H^1 = 2 dim S_k + 1 for even n >= 2 (k = n + 2 <= ell + 1 <= 14:
        // dim S_k = 1 iff k = 12), and H^1 = 0 at n = 0 for ell >= 5.
        const uint32_t k = c.n + 2u;
        const uint32_t dimS = (k == 12u) ? 1u : 0u;
        Check(c.dims.size() >= 2u, "dims recorded");
        if (c.dims.size() < 2u) continue;
        const uint32_t expect = (c.n == 0u) ? 0u : 2u * dimS + 1u;
        Check(c.dims[1] == expect, "dim H^1 at (n, ell) = (" + std::to_string(c.n) + ", "
                                       + std::to_string(ell) + ") is " + std::to_string(c.dims[1])
                                       + ", expected " + std::to_string(expect));
        Check(c.local_ideals == c.ideals, "every ideal certified local at n = " + std::to_string(c.n));
        if (c.n == 0u) continue;
        const AtlasSystem* e = FindChi(s, 0u, (c.n + 1u) % m);
        Check(e != nullptr && e->type == "eisenstein", "the Eisenstein system 1 + q^{n+1} is in Sigma");
        if (e == nullptr) continue;
        bool at_n = false;
        uint32_t dim = 0;
        for (const AtlasOccurrence& o : e->occurrences)
            if (o.n == c.n && o.group == "H1") { at_n = true; dim = o.dim; }
        Check(at_n, "1 + q^{n+1} occurs in H^1 at n = " + std::to_string(c.n));
        // (G) the Kummer/Ribet dichotomy on the atlas, k <= ell - 1.
        if (k <= ell - 1u) {
            const KummerVerdict v = kummer_verdict(k, ell);
            Check(v.applicable, "kummer_verdict applicable");
            Check((dim == 1u) == !v.divides,
                  "Eisenstein block dim " + std::to_string(dim) + " at k = " + std::to_string(k)
                      + " vs ell | num(B_k/k) = " + (v.divides ? "yes" : "no"));
        }
        ++cells;
    }
    // Every non-H^1 occurrence is Eisenstein-type.
    for (const AtlasSystem& sys : s.systems)
        for (const AtlasOccurrence& o : sys.occurrences)
            if (o.group != "H1")
                Check(sys.type == "eisenstein", "a non-H^1 system that is not Eisenstein-type");
    Check(s.chi_in_sigma == s.chi_count, "every chi_{a,b} is in Sigma");
    Check(s.chi_count == (m * m) / 4u, "the chi family has (ell-1)^2/4 distinct members on T");
    std::cout << "    (E) Eisenstein: 1 + q^{n+1} in H^1 on " << cells << " cells; chi "
              << s.chi_in_sigma << "/" << s.chi_count << "; dim H^1 = 2 dim S_k + 1 throughout\n";
}

void ArmCuspidal(const Atlas& a) {
    const AtlasStore& s = a.store;
    const uint32_t ell = s.ell;
    std::vector<const AtlasSystem*> cusp_bases;
    for (const AtlasSystem& sys : s.systems)
        if (sys.collected && sys.type == "cuspidal") cusp_bases.push_back(&sys);
    const uint32_t expect = (ell >= 11u) ? 1u : 0u;
    Check(cusp_bases.size() == expect, "Sigma(" + std::to_string(ell) + ") has "
                                           + std::to_string(cusp_bases.size())
                                           + " collected cuspidal systems, expected "
                                           + std::to_string(expect));
    if (expect == 0u || cusp_bases.size() != 1u) {
        std::cout << "    (F) cuspidal: " << cusp_bases.size() << " collected (level 1, k <= "
                  << ell + 1u << ": none expected)\n";
        return;
    }
    const AtlasSystem& D = *cusp_bases[0];
    Check(D.r == 1u, "Delta has residue degree 1");
    std::vector<FieldElt> tau;
    for (uint64_t q : s.primes) tau.push_back(FieldElt{TauMod(q, ell)});
    Check(D.eigenvalues == tau, "Delta: a_q == tau(q) mod " + std::to_string(ell) + " on the pool; got "
                                    + KeyText(D.eigenvalues) + " vs " + KeyText(tau));
    bool only10 = !D.occurrences.empty();
    for (const AtlasOccurrence& o : D.occurrences) {
        if (o.n != 10u || o.group != "H1") only10 = false;
        Check(o.dim == 2u, "Delta's H^1 block has dimension 2 (the +/- pair)");
        Check(o.a_ell.known, "a_ell of Delta is computed at ell <= 13");
        if (o.a_ell.known)
            Check(o.a_ell.value == FieldElt{TauMod(ell, ell)},
                  "a_ell(Delta) == tau(ell) mod ell");
    }
    Check(only10, "Delta occurs at n = 10 in H^1 only");
    // Not a twist of any lower-weight collected system ([E] Stage 6).
    bool lower_twist = false;
    for (const AtlasSystem& sys : s.systems) {
        if (!sys.collected || sys.r != D.r || &sys == &D) continue;
        uint32_t lowest = UINT32_MAX;
        for (const AtlasOccurrence& o : sys.occurrences) lowest = std::min(lowest, o.n);
        if (lowest >= 10u) continue;
        const CanonicalField F(ell, sys.r);
        for (uint32_t j = 0; j + 1u < ell; ++j)
            if (AtlasTwist(F, s.primes, sys.eigenvalues, j) == D.eigenvalues) lower_twist = true;
    }
    Check(!lower_twist, "Delta mod " + std::to_string(ell) + " is a twist of a lower weight");
    uint32_t orbit = 0;
    for (const AtlasSystem& sys : s.systems)
        if (sys.base == D.id) ++orbit;
    std::cout << "    (F) cuspidal: Delta mod " << ell << " = " << KeyText(D.eigenvalues)
              << " at n = 10 only, a_ell = tau(" << ell << ") mod " << ell << " = "
              << TauMod(ell, ell) << "; theta-orbit size " << orbit
              << "; not a twist of a lower weight\n";
}

void ArmKummerTable() {
    Check(kummer_divides(12, 691), "691 | num(B_12/12)");
    Check(kummer_divides(16, 3617), "3617 | num(B_16/16)");
    Check(kummer_divides(24, 103), "103 | num(B_24/24)");
    Check(kummer_divides(24, 2294797), "2294797 | num(B_24/24)");
    Check(kummer_divides(26, 657931), "657931 | num(B_26/26)");
    Check(bernoulli_mod(12, 691) == 0u, "bernoulli_mod(12, 691) == 0");
    for (uint32_t ell : {5u, 7u, 11u, 13u})
        for (uint32_t k = 2; k + 1u < ell; k += 2)
            Check(!kummer_divides(k, ell), "regular prime " + std::to_string(ell));
    std::cout << "  (G) Kummer table: 691|B_12, 3617|B_16, 103*2294797|B_24, 657931|num(B_26/26); "
                 "5, 7, 11, 13 regular\n";
}

void ArmBrauer(const std::vector<Atlas>& atlases) {
    for (uint32_t ell : {5u, 7u}) {
        uint32_t cells = 0, sl2 = 0, checks = 0;
        const uint32_t fails = GloverBrauerCrossCheck(ell, 3u * ell, cells, sl2, checks);
        Check.checks += checks;
        Check.fails += fails;
        if (fails) std::cout << "  *** FAIL [atlas] Brauer arm at ell = " << ell << ": " << fails << "\n";
        // Every Glover factor's weight is an atlas cell, and its theta^{j_i}
        // twists of the cell's H^1 systems are in Sigma (what identity 6,
        // WP-9, will quantify over).
        const Atlas* at = nullptr;
        for (const Atlas& a : atlases)
            if (a.store.ell == ell) at = &a;
        Check(at != nullptr, "atlas present for the Brauer arm");
        uint32_t factors = 0;
        if (at != nullptr) {
            const AtlasStore& s = at->store;
            for (uint32_t n = 0; n <= 3u * ell; n += 2u) {
                for (const GloverFactor& f : GloverFactors(n, ell).factors) {
                    Check((f.sym % 2u) == 0u, "even n has even Glover factors");
                    bool cell = false;
                    for (const AtlasCell& c : s.cells) cell = cell || (c.n == f.sym);
                    Check(cell, "Glover factor weight " + std::to_string(f.sym + 2u) + " is an atlas cell");
                    for (const AtlasSystem& sys : s.systems) {
                        bool here = false;
                        for (const AtlasOccurrence& o : sys.occurrences)
                            here = here || (o.n == f.sym && o.group == "H1");
                        if (!here) continue;
                        const CanonicalField F(ell, sys.r);
                        Check(AtlasFind(s, sys.r, AtlasTwist(F, s.primes, sys.eigenvalues, f.det_twist)) >= 0,
                              "a theta^{j_i}-twist over a Glover factor is in Sigma");
                    }
                    ++factors;
                }
            }
        }
        std::cout << "  (H) Brauer, ell = " << ell << ", n <= " << 3u * ell << ": " << cells
                  << " (class, n) cells (" << sl2 << " in SL_2), " << checks
                  << " comparisons; " << factors << " Glover factors, every weight an atlas cell\n";
    }
}

// ---------------------------------------------------------------------------
// (slot-7 WP-5) (L) AtlasSeparationBy against the slot-6 PAIRWISE loop, kept
// here verbatim (only its helpers are spelled out locally): WP-5 replaced the
// loop by group counting so that it scales to ell = 691, and this is the
// before/after comparison of that rewrite -- every counter, every subset S.
// ---------------------------------------------------------------------------
using RefKey = std::vector<FieldElt>;

std::vector<RefKey> RefConjugates(const CanonicalField& F, const RefKey& v) {
    std::vector<RefKey> out;
    RefKey cur = v;
    for (uint32_t i = 0; i < F.degree(); ++i) {
        out.push_back(cur);
        for (FieldElt& x : cur) x = F.frobenius(x);
    }
    return out;
}

std::vector<RefKey> RefRestrictedOrbit(const std::vector<RefKey>& orbit, const std::vector<size_t>& idx) {
    std::vector<RefKey> out;
    for (const RefKey& v : orbit) {
        RefKey w;
        for (size_t i : idx) w.push_back(v[i]);
        out.push_back(std::move(w));
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::map<RefKey, std::pair<uint32_t, uint32_t>> RefChiFamily(uint32_t ell, const std::vector<uint64_t>& T) {
    std::map<RefKey, std::pair<uint32_t, uint32_t>> out;
    const uint32_t m = ell - 1u;
    for (uint32_t a = 0; a < m; ++a)
        for (uint32_t b = a + 1u; b < m; ++b) {
            if (((a + b) & 1u) == 0u) continue;
            RefKey k;
            for (uint64_t q : T) k.push_back(FieldElt{(PowMod(q, a, ell) + PowMod(q, b, ell)) % ell});
            if (out.find(k) == out.end()) out.emplace(k, std::make_pair(a, b));
        }
    return out;
}

// The slot-6 AtlasSeparationBy body (HeckeAtlas.cpp at 3a40f68), verbatim.
AtlasSeparation ReferenceSeparationBy(const AtlasStore& s, const std::vector<uint64_t>& S) {
    struct Target { uint32_t r; std::vector<RefKey> orbit; };
    std::vector<Target> targets;
    for (const AtlasSystem& sys : s.systems)
        targets.push_back({sys.r, RefConjugates(CanonicalField(s.ell, sys.r), sys.eigenvalues)});
    for (const auto& kv : RefChiFamily(s.ell, s.primes))
        if (AtlasFind(s, 1u, kv.first) < 0) targets.push_back({1u, {kv.first}});

    std::vector<size_t> idx;
    for (uint64_t q : S) {
        const auto it = std::find(s.primes.begin(), s.primes.end(), q);
        idx.push_back(static_cast<size_t>(it - s.primes.begin()));
    }
    std::vector<std::vector<RefKey>> restricted;
    for (const Target& t : targets) restricted.push_back(RefRestrictedOrbit(t.orbit, idx));

    AtlasSeparation out;
    out.primes = S;
    std::sort(out.primes.begin(), out.primes.end());
    out.targets = static_cast<uint32_t>(targets.size());
    for (size_t a = 0; a < targets.size(); ++a) {
        for (size_t b = a + 1; b < targets.size(); ++b) {
            ++out.pairs;
            const std::vector<RefKey>& A = restricted[a];
            const std::vector<RefKey>& B = restricted[b];
            bool separated;
            if (A.size() != B.size()) {
                separated = true;
            } else if (targets[a].r == targets[b].r) {
                separated = (A != B);
            } else if (A.size() == 1u) {
                bool differ = false;
                for (size_t i = 0; i < A[0].size(); ++i) {
                    const uint32_t ca = A[0][i].empty() ? 0u : A[0][i][0];
                    const uint32_t cb = B[0][i].empty() ? 0u : B[0][i][0];
                    if (ca != cb) differ = true;
                }
                separated = differ;
            } else {
                separated = false;
                ++out.undecided;
            }
            if (!separated) ++out.unseparated;
        }
    }
    out.separates = (out.unseparated == 0u);
    return out;
}

bool SameSeparation(const AtlasSeparation& a, const AtlasSeparation& b) {
    return a.primes == b.primes && a.separates == b.separates && a.targets == b.targets
           && a.pairs == b.pairs && a.unseparated == b.unseparated && a.undecided == b.undecided;
}

// Every subset of the pool when it has <= max_bits primes, else every subset
// of the first max_bits primes.
void ArmSeparationVsReference(const AtlasStore& s, uint32_t max_bits = 10u) {
    const size_t m = std::min<size_t>(s.primes.size(), max_bits);
    uint32_t subsets = 0, mismatches = 0;
    uint64_t undecided_seen = 0;
    for (uint32_t mask = 0; mask < (1u << m); ++mask) {
        std::vector<uint64_t> S;
        for (size_t i = 0; i < m; ++i)
            if (mask & (1u << i)) S.push_back(s.primes[i]);
        const AtlasSeparation now = AtlasSeparationBy(s, S);
        const AtlasSeparation ref = ReferenceSeparationBy(s, S);
        if (!SameSeparation(now, ref)) ++mismatches;
        undecided_seen += ref.undecided;
        ++subsets;
    }
    Check(mismatches == 0u, "AtlasSeparationBy != the slot-6 pairwise loop on "
                                + std::to_string(mismatches) + " subsets at ell = "
                                + std::to_string(s.ell));
    std::cout << "    (L) separation == slot-6 pairwise loop on " << subsets
              << " subsets S (ell = " << s.ell << ", " << s.systems.size() << " systems, "
              << undecided_seen << " undecided pairs seen)\n";
}

// (slot-7 WP-5) (M) BuildAtlas == CollectAtlas + CloseAtlas, and a store
// re-closes to itself from its collected prefix (what `--atlas --verify`
// relies on for the ell = 691 store).
void ArmCollectClose(uint32_t ell, const Atlas& built) {
    const std::string j = built.store.to_json();
    Atlas c = CollectAtlas(ell);
    for (const AtlasSystem& sys : c.store.systems)
        Check(sys.collected && sys.twist == 0u, "CollectAtlas returns collected systems only");
    CloseAtlas(c.store);
    Check(c.store.to_json() == j, "CollectAtlas + CloseAtlas != BuildAtlas at ell = " + std::to_string(ell));
    AtlasStore again = ParseAtlasStore(j);
    CloseAtlas(again);
    Check(again.to_json() == j, "CloseAtlas(ParseAtlasStore(json)) != json at ell = " + std::to_string(ell));
    std::cout << "    (M) CollectAtlas + CloseAtlas == BuildAtlas; the parsed store re-closes to "
                 "itself byte for byte\n";
}

// (slot-7 WP-5) (K) THE ell = 691 STORE'S INTEGRITY, without rebuilding it
// (the build is ~an hour; docs/matrix-engine/tools/atlas_build.sh).  The store
// is read from HECKE_ATLAS_IN=<dir>, else from its publication path
// docs/matrix-engine/results/ext_first/hecke_atlas_ell691.json (slot-7
// confirmation 14: WP-7 copies it there); with neither present the arm
// prints SKIP and asserts nothing.  Asserted: it parses; it round-trips byte
// for byte; its shape (ell 691, n_max 690, the 346 cells, the pool); the
// (10, 691) system is Delta's -- a_q = tau(q) mod 691 on the whole pool, an H1
// occurrence at n = 10, and (Ramanujan) the Eisenstein chi(0, 11) -- ; a_691
// is known exactly at the cells whose u_ell verdict admitted U_691, and
// `unknown` with the verdict's reason elsewhere; S_sep separates.
void ArmStore691() {
    std::string path;
    if (const char* in = std::getenv("HECKE_ATLAS_IN")) {
        if (std::string(in).size() > 0) path = std::string(in) + "/hecke_atlas_ell691.json";
    }
#ifdef OC_SOURCE_DIR
    if (path.empty())
        path = std::string(OC_SOURCE_DIR) + "/docs/matrix-engine/results/ext_first/hecke_atlas_ell691.json";
#endif
    std::ifstream f(path, std::ios::binary);
    if (path.empty() || !f) {
        std::cout << "    (K) SKIP: no ell = 691 store at '" << path
                  << "' (set HECKE_ATLAS_IN=<dir>; atlas_build.sh writes one)\n";
        return;
    }
    std::ostringstream o;
    o << f.rdbuf();
    const std::string text = o.str();
    AtlasStore s;
    bool parsed = true;
    try {
        s = ParseAtlasStore(text);
    } catch (const std::exception& e) {
        parsed = false;
        std::cout << "    parse error: " << e.what() << "\n";
    }
    Check(parsed, "the ell = 691 store parses");
    if (!parsed) return;
    Check(s.to_json() == text, "the ell = 691 store does not round-trip byte for byte");
    Check(s.schema == kAtlasSchema && s.ell == 691u && s.n_max == 690u && s.cells.size() == 346u,
          "the ell = 691 store's shape (schema, ell, n_max, 346 cells)");
    for (size_t i = 0; i < s.cells.size(); ++i)
        Check(s.cells[i].n == 2u * i, "cell " + std::to_string(i) + " is n = " + std::to_string(2 * i));
    const std::vector<uint64_t> pool = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
    Check(s.primes == pool, "the ell = 691 pool is {q <= 31}");

    // Delta mod 691.
    std::vector<FieldElt> delta;
    for (uint64_t q : s.primes) delta.push_back(FieldElt{TauMod(q, 691u)});
    const int64_t id = AtlasFind(s, 1u, delta);
    Check(id >= 0, "Delta's eigenvalues mod 691 are a system of Sigma(691)");
    std::string a691 = "?";
    if (id >= 0) {
        const AtlasSystem& d = s.systems[static_cast<size_t>(id)];
        bool at10 = false;
        for (const AtlasOccurrence& oc : d.occurrences)
            if (oc.n == 10u && oc.group == "H1") {
                at10 = true;
                a691 = oc.a_ell.known ? std::to_string(oc.a_ell.value.empty() ? 0u : oc.a_ell.value[0])
                                      : "unknown(" + oc.a_ell.reason + ")";
            }
        Check(d.collected && at10, "Delta's system occurs in H1 at n = 10");
        Check(d.has_chi && d.chi_a == 0u && d.chi_b == 11u && d.type == "eisenstein",
              "Delta mod 691 is chi(0, 11) (Ramanujan's congruence)");
    }
    // a_691 known exactly where U_691 was admitted.
    uint64_t known = 0, unknown = 0, bad = 0;
    for (const AtlasSystem& sys : s.systems)
        for (const AtlasOccurrence& oc : sys.occurrences) {
            const AtlasCell& c = s.cells[oc.n / 2u];
            if (oc.a_ell.known != c.u_ell_included) ++bad;
            if (!oc.a_ell.known
                && oc.a_ell.reason != (c.u_ell_reason.empty() ? std::string("unavailable") : c.u_ell_reason))
                ++bad;
            (oc.a_ell.known ? known : unknown) += 1;
        }
    Check(bad == 0u, "a_691 known <=> the cell's u_ell verdict admitted U_691 (" + std::to_string(bad) + ")");
    Check(s.s_sep.separates, "S_sep(691) separates");
    uint64_t cusp = 0, deferred = 0;
    for (const AtlasSystem& sys : s.systems)
        if (sys.type == "cuspidal") ++cusp;
    for (const TangentPair& t : ComputeTangentPairs(s))
        if (t.status == "deferred") ++deferred;
    Check(deferred == cusp, "ell = 691: every cuspidal tangent pair is deferred (n > 1000)");
    uint32_t last_u = 0;
    for (const AtlasCell& c : s.cells)
        if (c.u_ell_included) last_u = c.n;
    std::cout << "    (K) ell = 691 store: " << text.size() << " bytes, " << s.systems.size()
              << " systems, parses and round-trips; Delta = chi(0,11) at n = 10, a_691 = " << a691
              << "; a_691 known on " << known << " occurrences (U_691 admitted up to n = " << last_u
              << "), unknown on " << unknown << "; tangent pairs deferred on all " << cusp
              << " cuspidal systems; S_sep = {";
    for (size_t i = 0; i < s.s_sep.primes.size(); ++i) std::cout << (i ? "," : "") << s.s_sep.primes[i];
    std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// (slot-7 WP-5) (N) the per-rho-bar tangent pair (HeckeExt/HeckeTangentPair.h
// states what is computed).  On a real Sigma(ell): no cuspidal system at
// ell in {5, 7}; at ell in {11, 13} exactly Delta is computed, at
// n = 10 + ell^2 - 1, with multiplicity >= 2 (the reason for the weight) and
// its generating set K-spanning m/m^2 (asserted inside) with exactly embdim
// primes, and every theta-twist inherits it.  (Measured: at ell = 11 the
// anemic algebra at Delta is the residue field -- embdim 0, a split block of
// multiplicity 2; at ell = 13, multiplicity 3, embdim 1, generated by T_2.)  Printed: D, r, multiplicity, embdim, active, generating, pair.
// ---------------------------------------------------------------------------
void ArmTangentPairs(const Atlas& a) {
    const AtlasStore& s = a.store;
    const std::vector<TangentPair> tp = ComputeTangentPairs(s);
    uint32_t cusp = 0, computed = 0, inherited = 0;
    for (const AtlasSystem& sys : s.systems)
        if (sys.type == "cuspidal") ++cusp;
    for (const TangentPair& t : tp) {
        if (t.status == "computed") {
            ++computed;
            std::cout << "    (N) " << t.to_json() << "\n";
            Check(t.weight == 12u && t.n == 10u + s.ell * s.ell - 1u,
                  "the computed tangent pair is Delta's, at n = 10 + ell^2 - 1");
            Check(t.multiplicity >= 2u, "multiplicity >= 2 at n = w - 2 + ell^2 - 1 (ell = "
                                            + std::to_string(s.ell) + ")");
            // Each greedy prime adds one K-dimension of m/m^2, so the
            // generating set has exactly embdim primes; embdim = 0 (the
            // anemic algebra is the residue field: a split block) is a
            // legitimate outcome, not a failure.
            Check(t.generating.size() == t.embdim && t.pair == (t.embdim <= 2u),
                  "|generating| == embdim, and a pair exists iff embdim <= 2");
        } else if (t.status == "inherited") {
            ++inherited;
        } else {
            Check(false, "tangent pair at ell = " + std::to_string(s.ell) + " system "
                             + std::to_string(t.system) + ": " + t.status + " (" + t.reason + ")");
        }
    }
    Check(tp.size() == cusp, "one tangent entry per cuspidal system");
    if (s.ell == 5u || s.ell == 7u) Check(cusp == 0u, "no cuspidal system at ell in {5, 7}");
    if (s.ell == 11u || s.ell == 13u)
        Check(computed == 1u && inherited + 1u == cusp,
              "Delta computed, its theta-twists inherited, at ell = " + std::to_string(s.ell));
    std::cout << "    (N) tangent pairs at ell = " << s.ell << ": " << cusp << " cuspidal, "
              << computed << " computed, " << inherited << " inherited\n";
}

// (N') the deferral path, on a constructed store at ell = 691: a collected
// cuspidal system of weight 12 is `deferred` with n = 10 + 691^2 - 1 =
// 477490 > 1000, WITHOUT running the pipeline, and a twist of it inherits
// the deferral.
void ArmTangentDeferred() {
    AtlasStore st;
    st.ell = 691u;
    st.primes = {2u, 3u};
    AtlasSystem a, b;
    a.id = 0; a.r = 1; a.type = "cuspidal"; a.base = 0; a.twist = 0; a.collected = true;
    a.eigenvalues = {FieldElt{1u}, FieldElt{2u}};
    AtlasOccurrence oc;
    oc.n = 10u;
    oc.group = "H1";
    a.occurrences = {oc};
    b = a;
    b.id = 1; b.twist = 5; b.collected = false; b.occurrences.clear();
    st.systems = {a, b};
    const std::vector<TangentPair> tp = ComputeTangentPairs(st);
    Check(tp.size() == 2u && tp[0].status == "deferred" && tp[0].n == 477490u
              && tp[0].reason == "n = 477490 > 1000" && tp[1].status == "deferred",
          "ell = 691: the tangent pair is deferred (n = 477490 > 1000), twists too");
    std::cout << "    (N') deferral: " << (tp.empty() ? std::string("-") : tp[0].to_json()) << "\n";
}

void ArmDeterminismAndGate(uint32_t ell, const Atlas& first) {
    const Atlas again = BuildAtlas(ell);
    Check(again.store.to_json() == first.store.to_json(),
          "the store is NOT byte-deterministic across two builds at ell = " + std::to_string(ell));
    std::cout << "    (D) byte-deterministic across two builds (" << first.store.to_json().size()
              << " bytes)\n";
}

// ---------------------------------------------------------------------------
// (J) the r >= 2 machinery, on constructed data.  Every system of Sigma(ell)
// at ell <= 13 is F_ell-rational (level 1, weight <= 14: Eisenstein and
// Delta), so the atlas's own cells never reach the canonical split's r >= 2
// path; this arm drives the same functions -- AtlasOrbitKey, AtlasTwist,
// SplitInCanonicalField on twisted polynomial data, AtlasSeparationBy's three
// decision paths -- where r = 2, 3, 4.
// ---------------------------------------------------------------------------
void ArmHigherDegree() {
    const uint32_t checks0 = Check.checks;
    for (uint32_t ell : {5u, 7u}) {
        const std::vector<uint64_t> T = {2u, 3u, 11u};
        for (uint32_t r : {2u, 3u}) {
            const CanonicalField F(ell, r);
            const hecke::FpPoly pi = F.defining_poly();   // its roots: Y and conjugates
            std::vector<hecke::FpPoly> P;
            uint32_t seed = 7u * ell + r;
            for (size_t i = 0; i < T.size(); ++i) {
                std::vector<uint32_t> c(r);
                for (uint32_t k = 0; k < r; ++k) {
                    seed = seed * 1103515245u + 12345u;
                    c[k] = (seed >> 8) % ell;
                }
                c[r - 1] = 1u + c[r - 1] % (ell - 1u);    // deg P = r - 1
                P.push_back(hecke::FpPoly(ell, c));
            }
            const CanonicalSplit sp = SplitInCanonicalField(F, pi, T, P);
            Check(sp.ok && CertifyCanonicalSplit(sp, P) == 0u, "constructed split certifies");
            if (!sp.ok) continue;
            const std::vector<FieldElt> key = AtlasOrbitKey(F, sp.eigenvalues);
            std::vector<FieldElt> v = sp.eigenvalues;
            bool in_orbit = false;
            for (uint32_t i = 0; i < r; ++i) {
                Check(AtlasOrbitKey(F, v) == key, "the key is conjugation-invariant");
                in_orbit = in_orbit || (v == key);
                for (FieldElt& x : v) x = F.frobenius(x);
            }
            Check(in_orbit, "the key is a conjugate");
            for (uint32_t j = 0; j + 1u < ell; ++j) {
                std::vector<hecke::FpPoly> tw;
                for (size_t i = 0; i < T.size(); ++i)
                    tw.push_back(P[i].scalar_mul(PowMod(T[i], j, ell)));
                const CanonicalSplit st = SplitInCanonicalField(F, pi, T, tw);
                Check(st.ok && CertifyCanonicalSplit(st, tw) == 0u,
                      "twisted constructed split certifies");
                Check(AtlasOrbitKey(F, st.eigenvalues) == AtlasTwist(F, T, key, j),
                      "r >= 2: twist on polynomial data == twist on the key");
                for (uint32_t j2 : {1u, 3u})
                    Check(AtlasTwist(F, T, AtlasTwist(F, T, key, j), j2)
                              == AtlasTwist(F, T, key, (j + j2) % (ell - 1u)),
                          "theta^j theta^j' == theta^{j+j'}");
            }
            Check(AtlasTwist(F, T, key, ell - 1u) == key, "theta^{ell-1} is the identity");
        }
    }

    // AtlasSeparationBy's three paths on a constructed store at ell = 5,
    // T = {2, 3, 7}: A (r = 2), B (r = 1), C (r = 4), A and C restricting to
    // {3} inside F_25 -- the conservative case.
    const uint32_t ell = 5u;
    const CanonicalField F2(ell, 2), F4(ell, 4);
    FieldElt z;
    bool found = false;
    for (uint64_t i = 2; i < 625u && !found; ++i) {
        const FieldElt c = F4.pow(F4.element_at(i), 26u);   // lies in F_25
        const FieldElt f1 = F4.frobenius(c);
        if (f1 != c && F4.frobenius(f1) == c) { z = c; found = true; }
    }
    Check(found, "an element of F_25 \\ F_5 inside F_625");
    AtlasStore st;
    st.ell = ell;
    st.primes = {2u, 3u, 7u};
    AtlasSystem A, B, C;
    A.id = 0; A.r = 2; A.eigenvalues = AtlasOrbitKey(F2, {F2.zero(), F2.generator(), F2.zero()});
    B.id = 1; B.r = 1; B.eigenvalues = {FieldElt{0u}, FieldElt{0u}, FieldElt{0u}};
    C.id = 2; C.r = 4; C.eigenvalues = AtlasOrbitKey(F4, {F4.zero(), z, F4.generator()});
    st.systems = {A, B, C};
    const AtlasSeparation s2 = AtlasSeparationBy(st, {2u});
    const AtlasSeparation s3 = AtlasSeparationBy(st, {3u});
    const AtlasSeparation sT = AtlasSeparationBy(st, {2u, 3u, 7u});
    Check(s2.undecided == 0u && s2.unseparated >= 3u,
          "{2}: A, B, C all restrict to the rational 0 -- unseparated, not undecided");
    Check(s3.undecided == 1u && s3.unseparated == 1u,
          "{3}: exactly the (r = 2, r = 4) pair is undecided and counted unseparated");
    Check(sT.separates && sT.undecided == 0u, "{2, 3, 7} separates by orbit size");
    // (slot-7 WP-5) the rewritten AtlasSeparationBy against the slot-6 loop
    // on this constructed store: all three decision paths, every subset.
    ArmSeparationVsReference(st);
    std::cout << "    (J) r >= 2 machinery: " << (Check.checks - checks0)
              << " checks at r in {2, 3} (ell = 5, 7) and the three separation paths "
                 "(rational / same r / undecided) at r in {1, 2, 4}\n";
}

uint32_t Finish(const char* name) {
    std::cout << name << " checks=" << Check.checks << " failures=" << Check.fails << std::endl;
    return Check.fails;
}

} // namespace

// ===========================================================================
uint32_t HeckeAtlasSkeletonValidation() {
    Check.reset();
    std::cout << "================= HeckeAtlasSkeletonValidation (slot 6 WP-8, E6) =================\n";
    try {
        // The write gate is closed unless HECKE_ATLAS_OUT is set; under --all
        // it is not, and this validation never sets it.
        const char* env = std::getenv("HECKE_ATLAS_OUT");
        const bool gate_open = env != nullptr && std::string(env).size() > 0;
        for (uint32_t ell : {5u, 7u}) {
            const Atlas a = BuildAtlas(ell);
            PrintSummary(a);
            ArmClosure(a);
            ArmSeparation(a);
            ArmRoundTrip(a);
            ArmDeterminismAndGate(ell, a);
            ArmSeparationVsReference(a.store);
            ArmCollectClose(ell, a);
            if (!gate_open) Check(WriteAtlasStoreIfRequested(a.store).empty(),
                                  "the write gate is closed without HECKE_ATLAS_OUT");
        }
        ArmHigherDegree();
        ArmTangentDeferred();
    } catch (const std::exception& e) {
        std::cout << "HeckeAtlasSkeletonValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        ++Check.fails;
    }
    return Finish("ATLAS_SKELETON");
}

uint32_t HeckeAtlasValidation() {
    Check.reset();
    std::cout << "================= HeckeAtlasValidation (slot 6 WP-8, E6) =================\n";
    try {
        ArmTauTable();
        std::vector<Atlas> atlases;
        for (uint32_t ell : {5u, 7u, 11u, 13u}) {
            atlases.push_back(BuildAtlas(ell));
            const Atlas& a = atlases.back();
            PrintSummary(a);
            ArmClosure(a);
            ArmSeparation(a);
            ArmRoundTrip(a);
            ArmEisenstein(a);
            ArmCuspidal(a);
            PrintCellColumn(a);
            ArmSeparationVsReference(a.store);
            ArmCollectClose(ell, a);
            ArmTangentPairs(a);
            const std::string path = WriteAtlasStoreIfRequested(a.store);
            if (!path.empty()) std::cout << "    store written: " << path << "\n";
        }
        ArmKummerTable();
        ArmBrauer(atlases);
        ArmStore691();
    } catch (const std::exception& e) {
        std::cout << "HeckeAtlasValidation: TOP-LEVEL EXCEPTION: " << e.what() << std::endl;
        ++Check.fails;
    }
    return Finish("ATLAS");
}
