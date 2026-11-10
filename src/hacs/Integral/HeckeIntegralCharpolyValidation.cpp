// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeIntegralCharpolyValidation -- slot 8, WP-7 (Line 2, M-0). See the header.

#include "HeckeIntegralCharpolyValidation.h"
#include "IntegralCharpolyCrt.h"

#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using integral::O5Row;

struct Fails {
    uint32_t count = 0;
    uint32_t checks = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++count;
            std::cout << "  *** FAIL: " << what << std::endl;
        }
    }
};

std::string PolyVec(const hecke::FpPoly& f) {
    std::ostringstream o;
    o << "[";
    for (int64_t i = 0; i <= f.degree(); ++i) o << (i ? "," : "") << f.coeff(static_cast<uint32_t>(i));
    o << "]";
    return o.str();
}

std::string PrimeList(const std::vector<uint32_t>& v, uint32_t count) {
    std::ostringstream o;
    o << "{";
    for (uint32_t i = 0; i < count; ++i) o << (i ? ", " : "") << v[i];
    o << "}";
    return o.str();
}

// Oracle-free checks at one prime (M-0 item 3): O3 degree, Eisenstein factor,
// square cofactor. Theorems, so they count.
void OracleFreeChecks(const hecke::FpPoly& chi, uint32_t n, uint32_t p, uint32_t ell,
                      const std::string& tag, Fails& F) {
    F.check(chi.degree() == static_cast<int64_t>(integral::RankLambda1(n)),
            tag + ": deg charpoly != 2 d_k - 1 (O3)");
    hecke::FpPoly cof(ell);
    F.check(integral::EisensteinFactorDivides(chi, n, p, ell, &cof),
            tag + ": (x - 1 - p^{k-1}) does not divide charpoly");
    F.check(integral::IsSquareMonic(cof), tag + ": cofactor is not a square");
}

// (0) The O5 transcription checks.
void CheckO5Rows(Fails& F) {
    std::cout << "--- (0) O5 transcription checks ([IB] Sec. 3.4; probe ranks) ---\n";
    for (const O5Row& r : integral::O5Table()) {
        std::ostringstream tag;
        tag << "O5 n=" << r.n << " p=" << r.p;
        const uint32_t deg = 1u + (r.cusp_root ? r.cusp_mult : 0u);
        F.check(deg == integral::RankLambda1(r.n), tag.str() + ": deg P != 2 d_k - 1");
        F.check(deg == r.probe_rank_h1, tag.str() + ": deg P != probe rank of H^1");
        if (r.cusp_root) {
            // Deligne: a_p^2 <= 4 p^{k-1} (n <= 20: p^{21} < 2^64 for p <= 3).
            const std::string s = r.cusp_root;
            const long long a = std::strtoll(s.c_str(), nullptr, 10);
            unsigned __int128 pk = 1;
            for (uint32_t i = 0; i < r.n + 1u; ++i) pk *= r.p;
            const unsigned __int128 a2 = static_cast<unsigned __int128>(a < 0 ? -a : a) *
                                         static_cast<unsigned __int128>(a < 0 ? -a : a);
            F.check(a2 <= 4 * pk, tag.str() + ": cusp root violates the Deligne bound");
        }
        const integral::DeligneBitBound B = integral::CharpolyCoeffBitBound(r.n, r.p);
        const uint64_t LP = integral::OracleCoeffBitBound(r);
        std::cout << "  " << tag.str() << ": " << r.verbatim << "  deg " << deg
                  << " = 2d_k-1 = probe rank " << r.probe_rank_h1
                  << "; bits: Deligne L=" << B.bits << " oracle L_P=" << LP
                  << (LP <= B.bits ? " (L_P <= L)" : " (L_P > L: the certificate uses L_P)") << "\n";
    }
}

// (1) + (2) One n: all oracle rows at that n share the F_ell runs.
void RunOracleN(uint32_t n, const std::vector<const O5Row*>& rows,
                const std::vector<uint32_t>& primes, Fails& F) {
    struct Cell { const O5Row* row; uint64_t L, LP, need, covered; uint32_t m; };
    std::vector<Cell> cells;
    uint32_t runs = 0;
    std::vector<uint32_t> ps;
    for (const O5Row* r : rows) {
        Cell c{r, integral::CharpolyCoeffBitBound(n, r->p).bits, integral::OracleCoeffBitBound(*r), 0, 0, 0};
        c.need = std::max(c.L, c.LP) + 2u;           // M >= 2^{need} > 2^{max + 1} >= |P| + |Q|
        c.m = integral::CertificatePrefix(primes, c.need, &c.covered);
        runs = std::max(runs, c.m + 1u);             // + the held-out prime
        cells.push_back(c);
        ps.push_back(r->p);
    }
    std::map<uint32_t, std::vector<bool>> agree;     // p -> per prime index
    for (uint32_t i = 0; i < runs; ++i) {
        const uint32_t ell = primes[i];
        const integral::FellRun run = integral::RunPipelineModEll(n, ell, ps);
        std::ostringstream t0;
        t0 << "n=" << n << " ell=" << ell;
        F.check(run.dims.size() >= 3 && run.dims[0] == 0 && run.dims[1] == integral::RankLambda1(n) &&
                    run.dims[2] == 0,
                t0.str() + ": dim H^q != (0, 2 d_k - 1, 0)");
        for (const Cell& c : cells) {
            const uint32_t p = c.row->p;
            const hecke::FpPoly& chi = run.charpoly.at(p);
            const hecke::FpPoly P = integral::OracleCharpolyMod(*c.row, ell);
            const bool eq = (chi == P);
            std::ostringstream tag;
            tag << t0.str() << " p=" << p << (i < c.m ? " [certificate]" : (i == c.m ? " [held-out]" : " [extra]"));
            F.check(eq, tag.str() + ": charpoly(A^(1)(T_p)) = " + PolyVec(chi) + " != O5 mod ell = " + PolyVec(P));
            OracleFreeChecks(chi, n, p, ell, tag.str(), F);
            agree[p].push_back(eq);
            // Negative control (certificate primes and held-out): a tampered oracle -- the
            // Eisenstein root with a digit appended, a different integer below 2^30 -- must
            // disagree, so the comparison is not vacuous.
            const std::string tampered_eis = std::string(c.row->eis_root) + "1";
            O5Row bad = *c.row;
            bad.eis_root = tampered_eis.c_str();
            F.check(integral::OracleCharpolyMod(bad, ell) != chi,
                    tag.str() + ": negative control -- a tampered O5 row was accepted");
        }
    }
    for (const Cell& c : cells) {
        const std::vector<bool>& a = agree[c.row->p];
        const bool cert_ok = std::all_of(a.begin(), a.begin() + c.m, [](bool b) { return b; });
        const bool held_ok = a[c.m];
        std::cout << "  CELL n=" << n << " p=" << c.row->p << ": Deligne L=" << c.L << " bits, oracle L_P="
                  << c.LP << ", need sum(bitlen-1) >= " << c.need << "; certificate primes "
                  << PrimeList(primes, c.m) << " cover " << c.covered << " bits; held-out prime "
                  << primes[c.m] << " " << (held_ok ? "agrees" : "DISAGREES") << "\n"
                  << "    => char_Z(T_" << c.row->p << " | Lambda^1) = " << c.row->verbatim
                  << (cert_ok ? "  CERTIFIED over Z" : "  NOT CERTIFIED") << "\n";
    }
}

// (3) Campaign n: residue vectors + certified bound + oracle-free checks.
void RunCampaignN(uint32_t n, const std::vector<uint32_t>& primes, Fails& F) {
    const std::vector<uint32_t> ps = {2u, 3u};
    std::map<uint32_t, uint32_t> m;
    std::map<uint32_t, uint64_t> L, covered, digest;
    uint32_t runs = 0;
    for (uint32_t p : ps) {
        L[p] = integral::CharpolyCoeffBitBound(n, p).bits;
        // M >= 2^{L + 2} > 2 * 2^L >= 2 |Q|: Q is the unique vector of the symmetric range.
        m[p] = integral::CertificatePrefix(primes, L[p] + 2u, &covered[p]);
        runs = std::max(runs, m[p]);
        digest[p] = engine::kFnvOffset64;
    }
    for (uint32_t i = 0; i < runs; ++i) {
        const uint32_t ell = primes[i];
        std::vector<uint32_t> ps_here;
        for (uint32_t p : ps) if (i < m[p]) ps_here.push_back(p);
        const integral::FellRun run = integral::RunPipelineModEll(n, ell, ps_here);
        std::ostringstream t0;
        t0 << "n=" << n << " ell=" << ell;
        F.check(run.dims.size() >= 2 && run.dims[1] == integral::RankLambda1(n),
                t0.str() + ": dim H^1 != 2 d_k - 1");
        for (uint32_t p : ps_here) {
            const hecke::FpPoly& chi = run.charpoly.at(p);
            std::ostringstream tag;
            tag << t0.str() << " p=" << p;
            OracleFreeChecks(chi, n, p, ell, tag.str(), F);
            std::cout << "  RES n=" << n << " p=" << p << " ell=" << ell << " chi=" << PolyVec(chi) << "\n";
            digest[p] = engine::FnvU64(digest[p], ell);
            for (int64_t j = 0; j <= chi.degree(); ++j)
                digest[p] = engine::FnvU64(digest[p], chi.coeff(static_cast<uint32_t>(j)));
        }
    }
    for (uint32_t p : ps) {
        std::cout << "  CAMPAIGN n=" << n << " p=" << p << ": rank Lambda^1 = " << integral::RankLambda1(n)
                  << ", Deligne L=" << L[p] << " bits, " << m[p] << " primes " << primes[0] << " .. "
                  << primes[m[p] - 1] << " cover " << covered[p] << " bits (>= L + 2): residue vector "
                  << "determines char_Z(T_" << p << " | Lambda^1) uniquely; FNV-1a-64 of (ell, chi) = 0x"
                  << std::hex << digest[p] << std::dec << " (recorded, not gated)\n";
    }
}

} // namespace

uint32_t HeckeIntegralCharpolyValidation() {
    try {
        std::cout << "============ HeckeIntegralCharpolyValidation (Line 2, M-0) ============\n";
        Fails F;
        CheckO5Rows(F);

        const bool with128 = [] {
            const char* e = std::getenv("HECKE_INTEGRAL_CAMPAIGN");
            return e != nullptr && std::strcmp(e, "1") == 0;
        }();
        // Enough primes for every cell run below (n = 128, p = 3 needs ~ 80).
        const std::vector<uint32_t> primes = integral::CrtPrimes(with128 ? 128u : 48u);
        for (uint32_t ell : primes) F.check(integral::IsPrimeU32(ell), "CRT prime not prime");

        std::cout << "--- (1)+(2) certified equality over Z against O5, one held-out prime per cell ---\n";
        std::map<uint32_t, std::vector<const O5Row*>> by_n;
        for (const O5Row& r : integral::O5Table()) by_n[r.n].push_back(&r);
        for (const auto& kv : by_n) RunOracleN(kv.first, kv.second, primes, F);

        std::cout << "--- (3) campaign n: residue vectors, certified bound, oracle-free checks ---\n";
        RunCampaignN(64u, primes, F);
        if (with128) {
            RunCampaignN(128u, primes, F);
            std::cout << "EFFECTIVE HeckeIntegralCharpolyValidation arm=n128 cells=" << primes.size() << "\n";
        } else {
            std::cout << "  (n = 128 skipped: set HECKE_INTEGRAL_CAMPAIGN=1)\n";
        }

        std::cout << "HeckeIntegralCharpolyValidation: " << F.checks << " checks, " << F.count << " failing\n";
        return F.count;
    } catch (const std::exception& e) {
        std::cout << "HeckeIntegralCharpolyValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
