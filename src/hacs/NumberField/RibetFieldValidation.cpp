// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// NumberFieldRibetFieldValidation (standard) -- the degree-ell subfield F of the Ribet /
// Kummer field H = Q(zeta_ell, eta_i^{1/ell}) as a polynomial over Z (hecke-extensions
// slot 21, T2; owner_q2_q4.md Q2 (b); the claims C1 - C8 are stated for the referee in
// docs/implementation/slot21/T2_claim_for_F4.md and are NOT asserted as proven here).
//
// Default cell (37, 32), about 1 min:
//   (1) the field data (C1, C3): the gamma exponents integral and gamma_R real (exact
//       congruences, in ribet_field_spec), the complement relation b_s^e sigma_s(gamma_R) =
//       b_{sg} in double precision, n_0 = eta mod pi against the exact power-basis eta
//       (Kummer.h), the layer-1 hypotheses (unramified, lambda split, not an ell-th power);
//       the same field data at (691, 12) and (691, 200);
//   (2) P = charpoly(theta) numerically at two precisions (equal; rounding margin);
//   (3) P exactly by CRT at split primes (C5): equal to (2), no chain failure; a negative
//       control (the non-real twist m + 1, another complement) gives a different polynomial;
//   (4) Frobenius (C4): the number of roots of P mod q for every prime q < 20000 with P mod q
//       squarefree against the Kummer prediction (eta(r) by the product form); a negative
//       control (P + 1) fails;
//   (5) disc P: sign (-1)^{(ell-1)/2}, odd ell-valuation >= ell - 2, the cofactor a square (C2);
//   (6) the maximal order (C6): Tr(O_K beta^kappa) of rank |<e>| per class, M_beta with
//       disc 37^{35 + 2v}, cofactor 1, and the order with disc exactly +37^35;
//   (7) POLRED (C7): the reduced basis over alpha with the exact trace form, Q(theta) =
//       Q(alpha) exactly, T1's polred; the chosen polynomial pinned, its disc 37^35 times a
//       square, its Frobenius counts again those of (4).
// Read-only env arm NUMBERFIELD_RIBET_CELLS="691:12,691:200" (EFFECTIVE line): per cell the
// polynomial by both routes (equal), digests, and Frobenius counts at 10 + 10 + 10 primes
// (q = 1 mod ell split in H, q = 1 mod ell not split, q != 1 mod ell). (691, 12) about 10 min.
// Threads: ENGINE_THREADS (default 8). No golden object, writes nothing.
#include "RibetFieldValidation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../../ModArithmetic/Fnv.h"
#include "BigInt.h"
#include "Cyclotomic.h"
#include "Kummer.h"
#include "ModUtil.h"
#include "MpReal.h"
#include "RibetField.h"
#include "Sha256.h"

using namespace nf;

namespace {

uint32_t g_fail = 0, g_checks = 0;

void Check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        std::cout << "  [FAIL] " << what << std::endl;
    }
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string PolyStr(const IntPoly& p) {
    std::ostringstream os;
    for (std::size_t i = p.size(); i-- > 0;) os << p[i].to_string() << (i ? " " : "");
    return os.str();
}

uint64_t PolyFnv(const IntPoly& p) {
    uint64_t h = engine::kFnvOffset64;
    for (const auto& x : p) h = engine::FnvBytes(h, x.to_string() + "\n");
    return h;
}

std::string PolySha(const IntPoly& p) {
    Sha256 sh;
    for (const auto& x : p) {
        const std::string s = x.to_string() + "\n";
        sh.update(s.data(), s.size());
    }
    return sh.hex();
}

std::size_t MaxDigits(const IntPoly& p) {
    std::size_t m = 0;
    for (const auto& x : p) m = std::max(m, x.decimal_digits());
    return m;
}

unsigned Threads() {
    if (const char* e = std::getenv("ENGINE_THREADS")) {
        const int v = std::atoi(e);
        if (v > 0) return static_cast<unsigned>(v);
    }
    return 8;
}

unsigned NumericPrec(const RibetFieldSpec& S) {
    return static_cast<unsigned>(std::ceil(S.bound_bits + 2.0 * std::log2(static_cast<double>(S.ell)) + 64.0));
}

struct FrobTally {
    std::size_t one = 0, split = 0, inert = 0, skipped = 0, mismatch = 0;
};
FrobTally Frobenius(const RibetFieldSpec& S, const IntPoly& P, uint32_t qmax) {
    FrobTally t;
    for (uint32_t q = 2; q < qmax; ++q) {
        if (q == S.ell || !is_prime_u32(q)) continue;
        const int r = poly_root_count_mod(P, q);
        if (r < 0) {
            ++t.skipped;
            continue;
        }
        const unsigned pr = ribet_predicted_roots(S, q);
        if (static_cast<unsigned>(r) != pr) ++t.mismatch;
        if (q % S.ell != 1) ++t.one;
        else if (pr == S.ell) ++t.split;
        else ++t.inert;
    }
    return t;
}

// The pinned POLRED output at (37, 32) (Cohen Alg. 4.4.12 on O_F with the rounded T2 form),
// constant term first.
const char* const kBest37[] = {
    "-1437499034", "-6646273463", "-12830602958", "-14856759165", "-16606853156", "-24955137198",
    "-35409312832", "-38430652267", "-35126020266", "-34071010718", "-35698146968", "-33060984680",
    "-25098426194", "-16882268703", "-10878267396", "-6053372105", "-2016981632", "608353759", "1941642824",
    "2441218431", "2055892750", "976459507", "15071176", "-301403905", "-181756694", "-28255328", "18828674",
    "10820632", "1222440", "-674529", "-234429", "-2201", "10394", "1498", "-108", "-55", "-3", "1",
};

void SpecChecks(uint32_t ell, uint32_t i, RibetFieldSpec* out) {
    const RibetFieldSpec S = ribet_field_spec(ell, i);
    bool signs = false;
    const double cc = ribet_complement_check(S, &signs);
    Check(cc < 1e-6 && signs, "complement relation b_s^e sigma_s(gamma_R) = b_{sg} at (" + std::to_string(ell) + ", " +
                                  std::to_string(i) + ")");
    std::cout << "RIBET_SPEC ell=" << ell << " i=" << i << " g=" << S.g << " e=" << S.e << " twist=" << S.twist
              << " n0=" << S.n0 << " bound_bits=" << S.bound_bits << " complement_logdiff=" << cc
              << " signs=" << (signs ? "ok" : "BAD") << std::endl;
    if (out) *out = S;
}

}  // namespace

uint32_t NumberFieldRibetFieldValidation() {
    g_fail = g_checks = 0;
    const double t_start = Now();
    const unsigned th = Threads();
    std::cout << "NumberFieldRibetFieldValidation: the degree-ell Ribet field polynomial (slot-21 T2; claims in "
                 "docs/implementation/slot21/T2_claim_for_F4.md), threads="
              << th << std::endl;

    // (1) field data
    RibetFieldSpec S;
    SpecChecks(37, 32, &S);
    SpecChecks(691, 12, nullptr);
    SpecChecks(691, 200, nullptr);
    {
        const ExactEta eta = exact_eta_layer1(37, 32);
        BigInt sum(0);
        for (const auto& a : eta.coeffs) sum += a;  // eta mod (1 - zeta) = eta(1) mod 37
        Check(static_cast<int64_t>(sum.floor_mod(BigInt(37)).to_int64()) == (S.n0 % 37 + 37) % 37,
              "n_0 = eta mod pi (product form vs the exact power-basis eta)");
        const Layer1Result L1 = kummer_layer1(37, 32);
        Check(L1.unramified && L1.lambda_split && L1.not_lth_power, "layer-1 hypotheses at (37, 32)");
        std::cout << "RIBET_LAYER1 ell=37 i=32 v_pi=" << L1.v_pi << " unramified=" << L1.unramified
                  << " lambda_split=" << L1.lambda_split << " not_lth_power=" << L1.not_lth_power << std::endl;
        // eta(r) by the product form (the Frobenius oracle) against the exact power basis
        bool ok = true;
        for (uint32_t q : split_primes(37, 5)) {
            const uint32_t r = root_of_order(q, 37);
            ok = ok && eval_at(eta.coeffs, r, q) == eval_product_unit(S.f, r, q);
        }
        Check(ok, "eta(r) product form == power basis at 5 primes q = 1 mod 37");
    }

    // (2) numerical, two precisions
    const unsigned p1 = NumericPrec(S);
    double t0 = Now();
    const RibetNumeric N1 = ribet_numeric(S, p1, th);
    double d1 = 0, d2 = 0;
    const IntPoly P = poly_from_conjugates(N1.theta, p1, th, &d1);
    const RibetNumeric N2 = ribet_numeric(S, p1 + 64, th);
    const IntPoly P2 = poly_from_conjugates(N2.theta, p1 + 64, th, &d2);
    Check(P == P2, "numerical P at two precisions");
    Check(d1 < -32 && d2 < -64, "rounding margins");
    Check(P.size() == 38 && P[37] == BigInt(1) && P[36].is_zero() && P[35].is_zero(), "P monic of degree 37, a_36 = a_35 = 0");
    std::cout << "RIBET_NUMERIC ell=37 i=32 prec=" << p1 << "," << p1 + 64 << " margin_log2=" << d1 << "," << d2
              << " max_digits=" << MaxDigits(P) << " fnv=0x" << std::hex << PolyFnv(P) << std::dec
              << " theta0=" << mp_to_string(N1.theta[0].re, 12) << " secs=" << Now() - t0 << std::endl;

    // (3) exact modular route, and the non-real twist as a negative control
    t0 = Now();
    const RibetModular M = ribet_poly_modular(S, th);
    Check(M.poly == P, "modular P == numerical P");
    Check(M.chain_failures == 0, "tau_R^{ell-1} = 1 at every prime (chain check)");
    Check(M.modulus_bits > S.bound_bits + 1.0, "CRT modulus > 2 * coefficient bound");
    std::cout << "RIBET_MODULAR ell=37 i=32 primes=" << M.primes << " modulus_bits=" << M.modulus_bits
              << " q_range=[" << M.q_min << "," << M.q_max << "] candidates=" << M.candidates_tested
              << " chain_failures=" << M.chain_failures << " equal=" << (M.poly == P) << " secs=" << Now() - t0 << std::endl;
    {
        RibetFieldSpec Sbad = S;
        Sbad.twist = (S.twist + 1) % S.ell;
        const RibetModular Mb = ribet_poly_modular(Sbad, th);
        Check(Mb.chain_failures == 0 && Mb.poly != P, "negative control: the twist m + 1 (another complement) gives another polynomial");
        Check(Frobenius(S, Mb.poly, 3000).mismatch == 0, "  ... of a conjugate field (same Frobenius counts)");
    }

    // (4) Frobenius
    t0 = Now();
    const FrobTally F = Frobenius(S, P, 20000);
    Check(F.mismatch == 0, "Frobenius root counts of P == Kummer prediction, q < 20000");
    std::cout << "RIBET_FROB ell=37 i=32 q<20000 one_root=" << F.one << " split_ell_roots=" << F.split
              << " inert_zero_roots=" << F.inert << " skipped_nonsquarefree=" << F.skipped << " mismatches=" << F.mismatch
              << " secs=" << Now() - t0 << std::endl;
    {
        IntPoly Pbad = P;
        Pbad[0] += BigInt(1);
        Check(Frobenius(S, Pbad, 3000).mismatch > 0, "negative control: P + 1 violates the Frobenius counts");
    }

    // (5) disc P
    t0 = Now();
    const BigInt DP = poly_discriminant(P);
    const unsigned vP = DP.valuation(37);
    const BigInt cofP = DP / BigInt::pow(BigInt(37), vP);
    const BigInt mP = bigint_isqrt(cofP.abs());
    Check(DP.sign() > 0 && vP % 2 == 1 && vP >= 35 && mP * mP == cofP, "disc P = +37^odd * square");
    const BigInt D0 = bigint_isqrt(DP / BigInt::pow(BigInt(37), 35));
    Check(D0 * D0 * BigInt::pow(BigInt(37), 35) == DP, "index candidate D = sqrt(disc P / 37^35)");
    std::cout << "RIBET_DISC ell=37 i=32 sign=+ v37=" << vP << " cofactor_square_digits=" << mP.decimal_digits()
              << " index_bound_bits=" << D0.bit_length() << " secs=" << Now() - t0 << std::endl;

    // (6) the maximal order, at a precision covering D and |x|^36
    t0 = Now();
    double l2b = 0;
    for (uint32_t s = 1; s < 37; ++s) l2b = std::max(l2b, std::fabs(S.log_b[s]) / std::log(2.0));
    const unsigned po = static_cast<unsigned>(D0.bit_length() + 36.0 * (l2b + 8.0) + 400.0);
    const RibetNumeric NO = ribet_numeric(S, po, th);
    const RibetMaxOrder MO = ribet_maximal_order(S, NO, P, D0, th);
    const unsigned vb = MO.beta.disc.valuation(37), vo = MO.order.disc.valuation(37);
    bool ranks_ok = MO.ranks.size() == MO.kappas.size();
    for (std::size_t r : MO.ranks) ranks_ok = ranks_ok && r == 36 / MO.kappas.size();
    Check(ranks_ok && MO.kappas.size() == 4, "Tr(O_K beta^kappa) has rank |<e>| = 9 for each of the 4 classes");
    Check(MO.beta.disc.sign() > 0 && MO.beta.disc / BigInt::pow(BigInt(37), vb) == BigInt(1),
          "disc M_beta is a power of 37 (M_beta p-maximal for p != 37)");
    Check(MO.x_log2_dist < -32, "x-generators: rounding margin of the 37^v-scaled coordinates");
    Check(MO.order.disc == BigInt::pow(BigInt(37), 35), "disc of the constructed order == +37^35");
    std::cout << "RIBET_ORDER ell=37 i=32 prec=" << po << " kappas=" << MO.kappas.size() << " beta_shift=" << MO.beta_shift
              << " disc_M_beta=37^" << vb << " index_v=" << MO.v << " x_margin_log2=" << MO.x_log2_dist
              << " disc_O=37^" << vo << " secs=" << Now() - t0 << std::endl;

    // (7) POLRED
    t0 = Now();
    const RibetPolred R = ribet_polred(P, NO.theta, MO.order, po, th);
    Check(R.trace_exact, "the alpha-coordinates reproduce the exact trace form of O_F");
    Check(R.theta_exact, "charpoly(theta over alpha) == P: Q(theta) = Q(alpha)");
    Check(R.h_log2_dist < -32 && R.f_log2_dist < -32, "alpha: rounding margins");
    Check(!R.best.empty(), "POLRED chose a generating candidate");
    const unsigned vbest = R.best.empty() ? 0 : R.best_disc.valuation(37);
    const BigInt cofb = R.best.empty() ? BigInt(0) : R.best_disc / BigInt::pow(BigInt(37), vbest);
    const BigInt mb = bigint_isqrt(cofb.abs());
    Check(vbest >= 35 && (vbest - 35) % 2 == 0 && cofb.sign() > 0 && mb * mb == cofb, "disc(best) = 37^35 * square");
    {
        IntPoly pin;
        for (const char* c : kBest37) pin.push_back(BigInt::from_string(c));
        Check(R.best == pin, "the POLRED polynomial at (37, 32) equals the pinned one");
    }
    Check(Frobenius(S, R.best, 5000).mismatch == 0, "Frobenius counts of the POLRED polynomial == prediction, q < 5000");
    std::cout << "RIBET_POLRED ell=37 i=32 lll_passes=" << R.passes << " index_alpha_digits=" << R.d.decimal_digits()
              << " best_index_digits=" << mb.decimal_digits() << " trace_exact=" << R.trace_exact
              << " theta_exact=" << R.theta_exact << " secs=" << Now() - t0 << std::endl;
    std::cout << "RIBET_POLRED_POLY ell=37 i=32 " << PolyStr(R.best) << std::endl;

    // env arm: the 691 cells
    if (const char* cells = std::getenv("NUMBERFIELD_RIBET_CELLS")) {
        std::vector<std::pair<uint32_t, uint32_t>> list;
        std::stringstream ss(cells);
        std::string tok;
        bool bad = false;
        while (std::getline(ss, tok, ',')) {
            const std::size_t c = tok.find(':');
            if (c == std::string::npos) {
                bad = true;
                break;
            }
            list.emplace_back(static_cast<uint32_t>(std::stoul(tok.substr(0, c))), static_cast<uint32_t>(std::stoul(tok.substr(c + 1))));
        }
        Check(!bad, "NUMBERFIELD_RIBET_CELLS parses as ell:i,...");
        std::cout << "EFFECTIVE NumberFieldRibetFieldValidation arm=CELLS cells=" << (bad ? 0 : list.size()) << std::endl;
        for (const auto& cell : bad ? decltype(list)() : list) {
            const uint32_t ell = cell.first, i = cell.second;
            t0 = Now();
            const RibetFieldSpec T = ribet_field_spec(ell, i);
            const unsigned pp = NumericPrec(T);
            const RibetNumeric NN = ribet_numeric(T, pp, th);
            double dd = 0;
            const IntPoly Q = poly_from_conjugates(NN.theta, pp, th, &dd);
            const double tn = Now() - t0;
            t0 = Now();
            const RibetModular MM = ribet_poly_modular(T, th);
            const double tm = Now() - t0;
            const std::string cs = "(" + std::to_string(ell) + ", " + std::to_string(i) + ")";
            Check(MM.poly == Q && MM.chain_failures == 0 && dd < -32, "numerical == modular at " + cs);
            // Frobenius at 10 + 10 + 10 primes
            t0 = Now();
            std::size_t ns = 0, ni = 0, no = 0, mism = 0, skip = 0;
            auto test = [&](uint32_t q, std::size_t& slot) {
                const int r = poly_root_count_mod(Q, q);
                if (r < 0) {
                    ++skip;
                    return;
                }
                ++slot;
                if (static_cast<unsigned>(r) != ribet_predicted_roots(T, q)) ++mism;
            };
            // q = 1 mod ell: the first 10 split in H (ell roots) and the first 10 not split (0 roots)
            for (uint64_t q = 2 * static_cast<uint64_t>(ell) + 1; q < 0x7fffffffu && (ns < 10 || ni < 10); q += 2 * ell) {
                if (!is_prime_u32(static_cast<uint32_t>(q))) continue;
                const bool split = ribet_predicted_roots(T, static_cast<uint32_t>(q)) == ell;
                if (split && ns < 10) test(static_cast<uint32_t>(q), ns);
                if (!split && ni < 10) test(static_cast<uint32_t>(q), ni);
            }
            // q != 1 mod ell (one root): the first 10 primes above ell
            for (uint32_t q = ell + 1; no < 10; ++q)
                if (is_prime_u32(q) && q % ell != 1) test(q, no);
            Check(mism == 0, "Frobenius counts at " + cs);
            std::cout << "RIBET_CELL ell=" << ell << " i=" << i << " prec=" << pp << " margin_log2=" << dd
                      << " max_digits=" << MaxDigits(Q) << " fnv=0x" << std::hex << PolyFnv(Q) << std::dec
                      << " sha256=" << PolySha(Q) << " modular_primes=" << MM.primes << " equal=" << (MM.poly == Q)
                      << " frob_split=" << ns << " frob_inert=" << ni << " frob_one=" << no << " frob_skipped=" << skip
                      << " frob_mismatch=" << mism << " secs_numeric=" << tn << " secs_modular=" << tm
                      << " secs_frob=" << Now() - t0 << std::endl;
        }
    }

    std::cout << "NumberFieldRibetFieldValidation: " << g_checks << " checks, " << g_fail << " failing, "
              << Now() - t_start << " s" << std::endl;
    return g_fail;
}
