// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZFreeBlockCrtValidation: EXPORT-Z step 2, package EXZ2-P2 (HeckeZFreeBlockCrtValidation.h).

#include "HeckeZFreeBlockCrtValidation.h"

#include "DescentFreeBlockCrt.h"
#include "DescentHecke.h"
#include "IsoTreeCarrierZ.h"

#include "../../../Cuda/CudaCtx.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Kernels.h"
#include "../../../ModArithmetic/RnsKernels.h"

#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <set>
#include <thread>
#include <vector>

// TEST ONLY hook (N3), defined in DescentFreeBlockCrt.cpp and deliberately not declared in the public header
namespace integral_descent {
namespace freez {
FreeBlockResult FreeBlockCrtTestCorruptLane(const FreeBlockInput& in, const FreeBlockOptions& opt, int32_t lane);
}
}  // namespace integral_descent

namespace {

namespace fz = integral_descent::freez;
namespace h3 = integral_descent::hecke3;
using fz::BigInt;
using clk = std::chrono::steady_clock;
using Isa = engine::cpu::IsaPath;

uint32_t g_fail = 0;
uint64_t g_checks = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string Cell(uint32_t N, uint32_t n) { return "(" + std::to_string(N) + ", " + std::to_string(n) + ")"; }
std::string Hex(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return o.str();
}

bool Exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}
const char* kFixRel = "docs/implementation/slot25/exz2p2/fixtures/";
std::string FixtureRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string r(OC_SOURCE_DIR);
        if (!r.empty() && Exists(r + "/" + kFixRel + "c1_10.txt")) return r + "/" + kFixRel;
    }
#endif
    char buf[4096];
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (Exists(dir + "/" + kFixRel + "c1_10.txt")) return dir + "/" + kFixRel;
            const size_t s = dir.find_last_of('/');
            if (s == std::string::npos || s == 0) break;
            dir = dir.substr(0, s);
        }
    }
    return std::string();
}

const std::vector<std::pair<uint32_t, uint32_t>> kCells = {{1, 10}, {11, 0}, {2, 6}, {11, 2}, {1, 30}, {5, 12}, {9, 8}, {8, 10}, {1, 60}};

struct Loaded {
    fz::FreeBlockFixture fx;
    std::unique_ptr<fz::IsoTreeCarrierZ> car;
    bool certified = false;
    // (EXZ2-P3) the production column format of FreeBlockInput (PackedZ), packed once from the fixture's SparseColsZ
    fz::PackedZ Bp, Phi0p, PhiCertp;
    void pack() {
        Bp = fx.B.to_packed();
        Phi0p = fx.Phi0.to_packed();
        PhiCertp = fx.PhiCert.to_packed();
    }
};
std::map<std::pair<uint32_t, uint32_t>, Loaded> g_cells;
std::string g_root;

fz::FreeBlockInput Input(const Loaded& L, const fz::PackedZ* B = nullptr) {
    fz::FreeBlockInput in;
    in.car = L.car.get();
    in.B = B ? B : &L.Bp;
    in.Phi0 = &L.Phi0p;
    in.PhiCert = L.fx.PhiCert.ncols() > 0 ? &L.PhiCertp : nullptr;   // (fix round 1, audit F3) re-certified by FreeBlockCrt itself
    in.R0 = L.fx.R0;
    in.P = L.fx.P;
    in.ps = L.fx.ps;
    return in;
}

uint64_t Mod64(const BigInt& x) {
    const BigInt two64 = BigInt::pow(BigInt(2), 64);
    const BigInt m = x.floor_mod(two64);
    const auto& mg = m.magnitude();
    uint64_t v = 0;
    if (mg.size() > 0) v = mg[0];
    if (mg.size() > 1) v |= uint64_t(mg[1]) << 32;
    return v;
}

// (fix round 1, audit F11) a tooth checks the SPECIFIC refusal, not "something threw"
bool Refused(const fz::FreeBlockResult& R, fz::FreeBlockRefusal want) {
    const bool ok = !R.ok && R.refused && R.code == want &&
                    R.refusal.find(std::string("[") + fz::FreeBlockRefusalName(want) + "]") != std::string::npos;
    if (!ok)
        std::cout << "  (refusal) wanted [" << fz::FreeBlockRefusalName(want) << "], got ok=" << R.ok << " refused=" << R.refused << " code=["
                  << fz::FreeBlockRefusalName(R.code) << "] " << R.refusal << std::endl;
    return ok;
}

// (fix round 1, audit F4 a) the bound against the exact H = G' F (F = the model's exact F, checked separately): |H_ij| <= ceil(beta_H,ij)
// and, with the inverse, |det G'| |F_ij| <= ceil(sum_k |adj_ik| beta_H,kj) on every entry.  A majorant that is too small fails here.
uint64_t g_bound_entries = 0;
void BoundTooth(const fz::FreeBlockResult& R, const fz::FreeBlockOp& op, const std::string& t) {
    const uint32_t r = R.r;
    if (!check(R.Gp.size() == size_t(r) * r && op.beta_H_ceil.size() == size_t(r) * r && op.F.size() == size_t(r) * r,
               t + ": G', ceil(beta_H) exposed (F4)"))
        return;
    const std::vector<BigInt> H = fz::MatMulZ(R.Gp, op.F, r);
    uint32_t bad_h = 0, bad_f = 0;
    for (size_t e = 0; e < H.size(); ++e) {
        if (H[e].abs() > op.beta_H_ceil[e]) ++bad_h;
        if (!op.apriori_rhs.empty() && op.F[e].abs() * R.detG.abs() > op.apriori_rhs[e]) ++bad_f;
        ++g_bound_entries;
    }
    check(bad_h == 0, t + ": |H| = |G' F| <= ceil(beta_H) entrywise (violations " + std::to_string(bad_h) + ")");
    check(!op.apriori_rhs.empty() && bad_f == 0, t + ": |det G'| |F| <= ceil(|adj G'| beta_H) entrywise (violations " + std::to_string(bad_f) + ")");
}

// (fix round 1, audit F4 b) the a-priori prefix and the bits of beta_H, beta_F pinned per cell and operator (= the P1a model's majorant
// figures, report Sec. 5); a majorant scaled by 2^-8 moves bits(beta_H) by 8.
struct BoundPin {
    uint32_t apriori, bits_beta_H, bits_beta_F;
};
const std::map<std::pair<uint32_t, uint32_t>, std::vector<BoundPin>> kBoundPins = {
    {{1, 10}, {{1, 32, 29}, {2, 44, 42}}},   {{11, 0}, {{1, 4, 4}, {1, 5, 5}}},       {{2, 6}, {{1, 24, 24}, {1, 28, 28}}},
    {{11, 2}, {{1, 24, 22}, {2, 33, 32}}},   {{1, 30}, {{4, 112, 100}, {5, 153, 141}}}, {{5, 12}, {{4, 139, 117}, {5, 158, 136}}},
    {{9, 8}, {{4, 120, 117}, {5, 152, 147}}}, {{8, 10}, {{6, 180, 168}, {6, 184, 168}}}, {{1, 60}, {{8, 251, 233}, {11, 337, 319}}}};
void PinBound(std::pair<uint32_t, uint32_t> cell, const fz::FreeBlockOp& op, const std::string& t) {
    auto it = kBoundPins.find(cell);
    const auto& fx = g_cells.at(cell).fx;
    const size_t o = static_cast<size_t>(std::find(fx.ps.begin(), fx.ps.end(), op.p) - fx.ps.begin());
    if (!check(it != kBoundPins.end() && o < it->second.size(), t + ": a pinned bound exists")) return;
    const BoundPin& w = it->second[o];
    check(op.primes_apriori == w.apriori && op.bits_beta_H == w.bits_beta_H && op.bits_beta_F == w.bits_beta_F,
          t + ": the pinned a-priori prefix / bits(beta_H) / bits(beta_F) = " + std::to_string(w.apriori) + " / " + std::to_string(w.bits_beta_H) +
              " / " + std::to_string(w.bits_beta_F) + " (got " + std::to_string(op.primes_apriori) + " / " + std::to_string(op.bits_beta_H) +
              " / " + std::to_string(op.bits_beta_F) + ")");
}

void PremiseTeeth();
void InadmissibleTooth();

// ---- (X) the fixtures, re-certified ----------------------------------------------------------------------------------------------
void ArmFixtures() {
    const uint64_t c0 = g_checks;
    for (const auto& [N, n] : kCells) {
        Loaded L;
        std::string why;
        const std::string path = g_root + "c" + std::to_string(N) + "_" + std::to_string(n) + ".txt";
        const bool loaded = fz::LoadFreeBlockFixture(path, L.fx, why);   // (N2) sequenced: the reason is read after the load ran
        if (!check(loaded, "(X) fixture " + path + " loads: " + why)) continue;
        const std::string tag = "(X) " + Cell(N, n);
        check(L.fx.N == N && L.fx.n == n, tag + ": the fixture's cell");
        L.car = std::make_unique<fz::IsoTreeCarrierZ>(N, n);
        const fz::IsoTreeCarrierZ& car = *L.car;
        check(L.fx.r == car.b1() && L.fx.dim1 == car.dim(1) && L.fx.B.ncols() == car.b1() && L.fx.B.rows == car.dim(1), tag + ": r, dim C^1");
        const auto Bd = L.fx.B.dense(), P0 = L.fx.Phi0.dense(), PC = L.fx.PhiCert.dense();
        const bool cyc = fz::ExactCocycles(car, Bd);
        const bool k0 = fz::ExactKillsD0(car, P0), kc = fz::ExactKillsD0(car, PC);
        const BigInt det = fz::DetBareiss(fz::PairingMatrix(PC, Bd), car.b1());
        const bool unit = det == BigInt(1) || det == BigInt(-1);
        check(cyc, tag + ": d^1 B = 0 over Z");
        check(k0, tag + ": Phi_0^T d^0 = 0 over Z");
        check(kc, tag + ": Phi_cert^T d^0 = 0 over Z");
        check(unit, tag + ": det(Phi_cert^T B) = +-1 (B is a Z-basis of L at every prime) -- det = " + det.to_string());
        // R0, P = BuildPairs' pair 0
        fz::PairsOptions po;
        po.K = 1;
        const fz::PairsResult pr = fz::BuildPairs(car, po);
        check(!pr.refused && !pr.pairs.empty() && pr.pairs[0].R0 == L.fx.R0 && pr.pairs[0].P == L.fx.P,
              tag + ": R0, P = BuildPairs' pair 0 (" + fz::kBasisRule + ")");
        L.certified = cyc && k0 && kc && unit;
        std::cout << "  (X) " << Cell(N, n) << " r=" << car.b1() << " dimC1=" << car.dim(1) << " bits B/Phi0/Phi_cert=" << L.fx.B.max_bits() << "/"
                  << L.fx.Phi0.max_bits() << "/" << L.fx.PhiCert.max_bits() << " det=" << det.to_string() << " fnv(B)=" << Hex(L.fx.B.fnv())
                  << " fnv(Phi_cert)=" << Hex(L.fx.PhiCert.fnv()) << std::endl;
        L.pack();
        g_cells[{N, n}] = std::move(L);
    }
    std::cout << "ARM (X) checks=" << g_checks - c0 << std::endl;
}

// ---- (L) single lanes against the model's F mod q ----------------------------------------------------------------------------------
void ArmLanes() {
    const uint64_t c0 = g_checks;
    for (const auto& cell : std::vector<std::pair<uint32_t, uint32_t>>{{1, 10}, {11, 2}, {9, 8}}) {
        auto it = g_cells.find(cell);
        if (!check(it != g_cells.end(), "(L) fixture " + Cell(cell.first, cell.second) + " present")) continue;
        const Loaded& L = it->second;
        const uint32_t r = L.fx.r;
        const std::vector<uint32_t> cand = fz::CachedCrtPrimes(40);
        uint32_t ran = 0;
        for (uint32_t q : {cand[0], cand[7], cand[19], cand[33], 1000003u, 65537u}) {
            if ((6ull * cell.first) % q == 0) continue;
            std::vector<uint32_t> out;
            std::string why;
            const bool adm = fz::FreeBlockLane(Input(L), q, fz::FreeBlockOptions{}, out, &why);
            if (!adm) {
                std::cout << "  (L) " << Cell(cell.first, cell.second) << " q=" << q << " inadmissible: " << why << std::endl;
                continue;
            }
            ++ran;
            bool eq = true;
            for (size_t o = 0; o < L.fx.ps.size(); ++o) {
                const auto& Fm = L.fx.F_model.at(L.fx.ps[o]);
                for (uint32_t i = 0; i < r * r; ++i) eq = eq && out[o * r * r + i] == Fm[i].floor_mod(BigInt(int64_t(q))).mod_u32(q);
            }
            check(eq, "(L) " + Cell(cell.first, cell.second) + " q=" + std::to_string(q) + ": the lane's F_p mod q = the model's F_p mod q");
        }
        check(ran >= 4, "(L) " + Cell(cell.first, cell.second) + ": at least 4 admissible lanes compared");
    }
    std::cout << "ARM (L) checks=" << g_checks - c0 << std::endl;
}

// ---- (A) F proven = the model's F; char_Z against CharpolyOverZ; (C) commutation and the pinned charpolys ---------------------------
std::map<std::pair<uint32_t, uint32_t>, fz::FreeBlockResult> g_res;
void ArmProven() {
    const uint64_t c0 = g_checks, cA = g_checks;
    for (auto& [cell, L] : g_cells) {
        const std::string tag = Cell(cell.first, cell.second);
        if (!L.certified) continue;
        const fz::FreeBlockResult R = fz::FreeBlockCrt(Input(L));
        if (!check(R.ok && !R.refused, "(A) " + tag + ": the CRT ran: " + R.refusal)) continue;
        const uint32_t r = R.r;
        check(R.certified && R.nonsingular_by == "inverse", "(A) " + tag + ": the a-priori premise certified and det G' != 0 by the inverse (F2, F3)");
        for (const fz::FreeBlockOp& op : R.ops) {
            const std::string t = "(A) " + tag + " T_" + std::to_string(op.p);
            if (!check(op.proven, t + ": proven")) continue;
            check(op.F == L.fx.F_model.at(op.p), t + ": F_p = the model's exact F_p (fmpq, independent route)");
            check(R.carrier_lanes_left == 0, t + ": no lane TreeModule left in the carrier (F7)");
            BoundTooth(R, op, t);
            PinBound(cell, op, t);
            std::cout << "  (A) " << tag << " T_" << op.p << " proof=" << op.proof << " primes=" << op.primes_used << " a-priori=" << op.primes_apriori
                      << " bits F/beta_H/beta_F=" << op.bits_F << "/" << op.bits_beta_H << "/" << op.bits_beta_F << " fnv(F)=" << Hex(op.fnv)
                      << std::endl;
        }
        // char_Z against the record's route (D3 CharpolyOverZ), on the cells of cost <= a second
        if (cell.second <= 30 && r <= 12) {
            for (const fz::FreeBlockOp& op : R.ops) {
                if (!op.proven) continue;
                const std::vector<BigInt> cp = fz::CharpolyBerkowitz(op.F, r);
                const h3::CharpolyZ cz = h3::CharpolyOverZ(cell.first, cell.second, op.p);
                const std::string t = "(A) " + tag + " T_" + std::to_string(op.p);
                if (!check(!cz.refused && cz.determined && cz.r == r && cz.c.size() == r + 1, t + ": CharpolyOverZ determined, r agrees")) continue;
                bool eq = true;
                for (uint32_t i = 0; i <= r; ++i) {
                    const h3::CrtCoefficient& c = cz.c[i];
                    if (c.fits) eq = eq && cp[i] == BigInt(c.value);
                    else eq = eq && c.certified && Mod64(cp[i]) == c.mod_2_64 && cp[i].is_neg() == c.negative;
                }
                check(eq, t + ": char(F_p) over Z = CharpolyOverZ's char_Z(T_p | H^1/tors)");
            }
        }
        // (C) commutation
        if (R.ops.size() == 2 && R.ops[0].proven && R.ops[1].proven)
            check(fz::MatMulZ(R.ops[0].F, R.ops[1].F, r) == fz::MatMulZ(R.ops[1].F, R.ops[0].F, r), "(C) " + tag + ": F_p F_q = F_q F_p over Z");
        g_res[cell] = R;
    }
    std::cout << "ARM (A) checks=" << g_checks - cA << std::endl;
    const uint64_t cC = g_checks;
    auto pinned = [&](std::pair<uint32_t, uint32_t> cell, uint32_t p, const std::vector<int64_t>& want) {
        auto it = g_res.find(cell);
        if (!check(it != g_res.end(), "(C) " + Cell(cell.first, cell.second) + " ran")) return;
        for (const auto& op : it->second.ops)
            if (op.p == p && op.proven) {
                const std::vector<BigInt> cp = fz::CharpolyBerkowitz(op.F, it->second.r);
                std::vector<BigInt> w;
                for (int64_t x : want) w.push_back(BigInt(x));
                check(cp == w, "(C) " + Cell(cell.first, cell.second) + ": the pinned char F_" + std::to_string(p));
                return;
            }
        check(false, "(C) " + Cell(cell.first, cell.second) + ": T_" + std::to_string(p) + " proven");
    };
    // (x - 3)(x + 2)^2 = x^3 + x^2 - 8x - 12;  (x - 2049)(x + 24)^2 = x^3 - 2001 x^2 - 97776 x - 1180224
    pinned({11, 0}, 2, {1, 1, -8, -12});
    pinned({1, 10}, 2, {1, -2001, -97776, -1180224});
    std::cout << "ARM (C) checks=" << g_checks - cC << std::endl;
    (void)c0;
}

// ---- (P) the proofs, independence, teeth ------------------------------------------------------------------------------------------
void ArmProofs() {
    const uint64_t c0 = g_checks;
    for (const auto& cell : std::vector<std::pair<uint32_t, uint32_t>>{{1, 30}, {9, 8}, {5, 12}}) {
        auto it = g_cells.find(cell);
        auto jt = g_res.find(cell);
        if (!check(it != g_cells.end() && jt != g_res.end(), "(P) " + Cell(cell.first, cell.second) + " present")) continue;
        const Loaded& L = it->second;
        const std::string tag = "(P) " + Cell(cell.first, cell.second);
        fz::FreeBlockOptions a, b, c;
        a.use_apost = false;
        b.use_apriori = false;
        c.skip_first = 7;
        const fz::FreeBlockResult Ra = fz::FreeBlockCrt(Input(L), a), Rb = fz::FreeBlockCrt(Input(L), b), Rc = fz::FreeBlockCrt(Input(L), c);
        for (size_t o = 0; o < L.fx.ps.size(); ++o) {
            const auto& ref = jt->second.ops[o];
            check(Ra.ok && Ra.ops[o].proven && Ra.ops[o].proof == "apriori" && Ra.ops[o].F == ref.F, tag + ": a-priori alone proves the same F");
            check(Rb.ok && Rb.ops[o].proven && Rb.ops[o].proof == "apost" && Rb.ops[o].F == ref.F, tag + ": a-posteriori alone proves the same F");
            check(Rc.ok && Rc.ops[o].proven && Rc.ops[o].F == ref.F && Rc.primes.front() != jt->second.primes.front(),
                  tag + ": a later prime window (skip 7) gives the same F");
            std::cout << "  (P) " << Cell(cell.first, cell.second) << " T_" << ref.p << " lanes: both " << ref.primes_used << " a-priori-only "
                      << Ra.ops[o].primes_used << " a-post-only " << Rb.ops[o].primes_used << " exact-need-bits " << ref.bits_F + 1 << std::endl;
        }
        // tooth 1: B' = B with column k scaled by a prime s (index s).  F' = D^{-1} F D, D = diag(1, .., s (at k), .., 1): row k of F' is
        // row k of F divided by s off the diagonal, column k multiplied by s.  (k, s) with an off-diagonal entry of row k prime to s in
        // some F_p: F' is NOT integral and the unconditional a-posteriori test can never close (max_primes refuses).  (k', s') with
        // every off-diagonal entry of row k' divisible by s' in every F_p: B' spans a T-stable sublattice and the proof must return
        // exactly D^{-1} F D (the positive twin).
        const uint32_t r = L.fx.r;
        int32_t kbad = -1, kgood = -1;
        uint32_t sbad = 0, sgood = 0;
        for (uint32_t sp : {2u, 3u, 5u, 7u, 11u, 13u})
            for (uint32_t k = 0; k < r; ++k) {
                bool prime_to = false;
                for (const auto& op : jt->second.ops)
                    for (uint32_t j = 0; j < r; ++j)
                        if (j != k && op.F[size_t(k) * r + j].mod_u32(sp) != 0) prime_to = true;
                if (prime_to && kbad < 0) {
                    kbad = static_cast<int32_t>(k);
                    sbad = sp;
                }
                if (!prime_to && kgood < 0) {
                    kgood = static_cast<int32_t>(k);
                    sgood = sp;
                }
            }
        if (check(kbad >= 0, tag + ": a column whose scaling makes F non-integral exists")) {
            fz::SparseColsZ B2 = L.fx.B;
            for (auto& e : B2.cols[kbad]) e.second = e.second * BigInt(int64_t(sbad));
            fz::FreeBlockOptions t;
            t.use_apriori = false;
            t.max_primes = 3 * std::max<uint32_t>(4, jt->second.ops[0].primes_used);
            const fz::PackedZ B2p = B2.to_packed();
            const fz::FreeBlockResult Rt = fz::FreeBlockCrt(Input(L, &B2p), t);
            check(Refused(Rt, fz::FreeBlockRefusal::NoProof), tag + ": tooth " + std::to_string(sbad) + " b_" + std::to_string(kbad + 1) +
                                                                 " (F not integral): the a-posteriori proof refuses [NoProof] within 3x the lanes");
        }
        if (kgood >= 0) {
            fz::SparseColsZ B2 = L.fx.B;
            for (auto& e : B2.cols[kgood]) e.second = e.second * BigInt(int64_t(sgood));
            fz::FreeBlockOptions t;
            t.use_apriori = false;
            const fz::PackedZ B2p = B2.to_packed();
            const fz::FreeBlockResult R2 = fz::FreeBlockCrt(Input(L, &B2p), t);
            bool ok = R2.ok;
            const BigInt S{int64_t(sgood)};
            for (size_t o = 0; ok && o < R2.ops.size(); ++o) {
                const auto& F = jt->second.ops[o].F;
                for (uint32_t i = 0; ok && i < r; ++i)
                    for (uint32_t j = 0; ok && j < r; ++j) {
                        BigInt x = F[size_t(i) * r + j];
                        if (j == static_cast<uint32_t>(kgood) && i != j) x = x * S;
                        if (i == static_cast<uint32_t>(kgood) && i != j) x = x / S;
                        ok = R2.ops[o].proven && R2.ops[o].F[size_t(i) * r + j] == x;
                    }
            }
            check(ok, tag + ": " + std::to_string(sgood) + " b_" + std::to_string(kgood + 1) +
                          " spans a T-stable sublattice: proven F' = D^{-1} F D exactly");
            std::cout << "  (P) " << Cell(cell.first, cell.second) << " T-stable index-" << sgood << " sublattice at b_" << kgood + 1 << " proven" << std::endl;
        }
        // tooth 2: one lane only
        fz::FreeBlockOptions t1;
        t1.max_primes = 1;
        t1.batch = 1;
        check(Refused(fz::FreeBlockCrt(Input(L), t1), fz::FreeBlockRefusal::NoProof), tag + ": a 1-lane cap refuses [NoProof]");
    }
    PremiseTeeth();
    InadmissibleTooth();
    std::cout << "ARM (P) checks=" << g_checks - c0 << std::endl;
}

// ---- (D) bit identity across routes, paths, threads, tiles --------------------------------------------------------------------------
void ArmIdentity() {
    const uint64_t c0 = g_checks;
    for (const auto& cell : std::vector<std::pair<uint32_t, uint32_t>>{{9, 8}, {1, 30}}) {
        auto it = g_cells.find(cell);
        auto jt = g_res.find(cell);
        if (!check(it != g_cells.end() && jt != g_res.end(), "(D) " + Cell(cell.first, cell.second) + " present")) continue;
        const Loaded& L = it->second;
        const fz::FreeBlockResult& ref = jt->second;
        const std::string tag = "(D) " + Cell(cell.first, cell.second);
        // lanes_only residues of 6 lanes at the default settings = the reference for the residue comparison
        fz::FreeBlockOptions lo;
        lo.lanes_only = 6;
        lo.threads = 1;
        const fz::FreeBlockResult Rl = fz::FreeBlockCrt(Input(L), lo);
        auto same = [&](const fz::FreeBlockResult& R, const std::string& what) {
            bool ok = R.ok && R.ops.size() == ref.ops.size();
            for (size_t o = 0; ok && o < R.ops.size(); ++o) ok = R.ops[o].proven && R.ops[o].fnv == ref.ops[o].fnv && R.ops[o].F == ref.ops[o].F;
            check(ok, tag + ": FNV(F) identical, " + what);
        };
        auto same_lanes = [&](fz::FreeBlockOptions o, const std::string& what) {
            o.lanes_only = 6;
            const fz::FreeBlockResult R = fz::FreeBlockCrt(Input(L), o);
            check(R.ok && R.primes == Rl.primes && R.lane_residues == Rl.lane_residues, tag + ": lane residues identical, " + what);
        };
        uint32_t routes = 0;
        for (Isa gp : {Isa::Portable, Isa::Avx2, Isa::Avx512}) {
            if (gp == Isa::Avx2 && !engine::kernels::avx2_kernel_available()) continue;
            if (gp == Isa::Avx512 && !engine::kernels::avx512_kernel_available()) continue;
            fz::FreeBlockOptions o;
            o.gemm_path = gp;
            same(fz::FreeBlockCrt(Input(L), o), std::string("gemm ") + engine::cpu::isa_path_name(gp));
            same_lanes(o, std::string("gemm ") + engine::cpu::isa_path_name(gp));
            ++routes;
        }
        for (Isa rp : {Isa::Portable, Isa::Avx2, Isa::Avx512}) {
            if (rp == Isa::Avx2 && !engine::rns::garner_avx2_available()) continue;
            if (rp == Isa::Avx512 && !engine::rns::garner_avx512_available()) continue;
            fz::FreeBlockOptions o;
            o.garner_path = rp;
            same(fz::FreeBlockCrt(Input(L), o), std::string("garner ") + engine::cpu::isa_path_name(rp));
            ++routes;
        }
        for (uint32_t th : {1u, 4u, 16u}) {
            fz::FreeBlockOptions o;
            o.threads = th;
            same(fz::FreeBlockCrt(Input(L), o), "threads " + std::to_string(th));
            same_lanes(o, "threads " + std::to_string(th));
            ++routes;
        }
        for (uint32_t w : {1u, 3u, 128u}) {
            fz::FreeBlockOptions o;
            o.tile = w;
            same(fz::FreeBlockCrt(Input(L), o), "tile " + std::to_string(w));
            ++routes;
        }
        if (engine::cuda::device_enabled()) {
            fz::FreeBlockOptions o;
            o.route = fz::LaneRoute::Gpu;
            same(fz::FreeBlockCrt(Input(L), o), "GPU lane route (one stream per prime, batched u32 GEMM)");
            same_lanes(o, "GPU lane route");
            ++routes;
            std::cout << "  (D) " << Cell(cell.first, cell.second) << " GPU route compared" << std::endl;
        } else {
            std::cout << "  [SKIP] (D) " << Cell(cell.first, cell.second) << " GPU lane route: "
                      << (engine::cuda::available() ? "ENGINE_DEVICE != gpu (opt-in)" : "no device") << std::endl;
        }
        std::cout << "  (D) " << Cell(cell.first, cell.second) << " routes compared=" << routes;
        for (const auto& op : ref.ops) std::cout << " fnv(F_" << op.p << ")=" << Hex(op.fnv);
        std::cout << std::endl;
    }
    std::cout << "ARM (D) checks=" << g_checks - c0 << std::endl;
}

// ---- (T) thread safety of the shared read paths (audit F11), lane caches (F7) ----------------------------------------------------
void ArmThreads() {
    const uint64_t c0 = g_checks;
    auto it = g_cells.find({9, 8});
    if (!check(it != g_cells.end(), "(T) (9, 8) present")) return;
    const fz::IsoTreeCarrierZ& car = *it->second.car;
    const fz::SparseZ& d0 = car.d(0);
    const uint32_t q = fz::kQ0;
    fz::EchelonModQ E(q, d0.cols);
    std::vector<uint32_t> rows;
    for (uint32_t i : it->second.fx.R0) {
        fz::SpVec v;
        for (uint64_t k = d0.rowptr[i]; k < d0.rowptr[i + 1]; ++k) {
            const uint32_t x = d0.val[k].mod_u32(q);
            if (x) {
                v.idx.push_back(d0.col[k]);
                v.val.push_back(x);
            }
        }
        if (E.insert(v)) rows.push_back(i);
    }
    check(rows.size() == it->second.fx.R0.size(), "(T) the rows R0 independent mod q0");
    // reference answers single-threaded
    const uint32_t K = E.rank();
    std::vector<std::vector<uint32_t>> rhs(16), ref_x(16), ref_z(16);
    for (uint32_t t = 0; t < 16; ++t) {
        // b = sum_k (t + k + 1) v_k  (in the span); u = (t * 7 + k) for the transposed solve
        std::vector<uint32_t> u(K);
        for (uint32_t k = 0; k < K; ++k) u[k] = (t * 7 + k) % q;
        rhs[t] = u;
        E.solve_transposed(u, ref_z[t]);
    }
    // the transposed solve is right: v_k . z = u_k
    bool tz = true;
    for (uint32_t t = 0; t < 4; ++t)
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t i = rows[k];
            uint64_t s = 0;
            for (uint64_t kk = d0.rowptr[i]; kk < d0.rowptr[i + 1]; ++kk) s = (s + uint64_t(d0.val[kk].mod_u32(q)) * ref_z[t][d0.col[kk]]) % q;
            tz = tz && s == rhs[t][k];
        }
    check(tz, "(T) solve_transposed: v_k . z = u_k for every pivot row");
    // solve: b = rows combination
    std::vector<std::vector<uint32_t>> bs(16);
    for (uint32_t t = 0; t < 16; ++t) {
        std::vector<uint32_t> b(d0.cols, 0);
        for (uint32_t k = 0; k < K; k += 3) {
            const uint32_t i = rows[k];
            for (uint64_t kk = d0.rowptr[i]; kk < d0.rowptr[i + 1]; ++kk)
                b[d0.col[kk]] = static_cast<uint32_t>((b[d0.col[kk]] + uint64_t(t + k + 1) * d0.val[kk].mod_u32(q)) % q);
        }
        bs[t] = b;
        check(E.solve(b, ref_x[t]), "(T) solve: b in the span");
    }
    const uint32_t bits_ref = car.d(1).max_row_abs_bits(), bits0_ref = car.d(0).max_row_abs_bits();
    fz::SparseZ unsealed = car.d(1), unsealed0 = car.d(0);
    unsealed.row_abs_bits_ = -1;   // the unsealed path: computed per call, nothing written
    unsealed0.row_abs_bits_ = -1;
    check(unsealed0.max_row_abs_bits() == bits0_ref, "(T) d^0: the unsealed value = the sealed one");
    std::atomic<uint32_t> bad{0};
    std::vector<std::thread> th;
    for (uint32_t w = 0; w < 8; ++w)
        th.emplace_back([&, w]() {
            for (uint32_t rep = 0; rep < 4; ++rep)
                for (uint32_t t = w % 2; t < 16; t += 2) {
                    std::vector<uint32_t> x, z;
                    if (!E.solve(bs[t], x) || x != ref_x[t]) bad.fetch_add(1);
                    E.solve_transposed(rhs[t], z);
                    if (z != ref_z[t]) bad.fetch_add(1);
                }
            if (unsealed.max_row_abs_bits() != bits_ref || unsealed0.max_row_abs_bits() != bits0_ref || car.d(0).max_row_abs_bits() != bits0_ref)
                bad.fetch_add(1);
        });
    for (auto& x : th) x.join();
    check(bad.load() == 0, "(T) 8 threads: EchelonModQ::solve / solve_transposed and SparseZ::max_row_abs_bits agree with the single-thread answers");
    // a 16-thread CRT leaves no lane TreeModule behind (F7)
    fz::FreeBlockOptions o;
    o.threads = 16;
    o.batch = 16;
    const fz::FreeBlockResult R = fz::FreeBlockCrt(Input(it->second), o);
    check(R.ok && R.carrier_lanes_left == 0 && car.lane_count() == 0, "(T) a 16-thread run leaves no lane TreeModule in the carrier (F7)");
    std::cout << "  (T) threads=8 solve/solve_transposed/max_row_abs_bits, 16-lane run: lanes_left=" << R.carrier_lanes_left << " rho_memo_peak="
              << R.lane_rho_memo_peak << " B" << std::endl;
    std::cout << "ARM (T) checks=" << g_checks - c0 << std::endl;
}

// ---- (R) the size cap -------------------------------------------------------------------------------------------------------------
void ArmRss() {
    const uint64_t c0 = g_checks;
    auto it = g_cells.find({1, 10});
    if (check(it != g_cells.end(), "(R) (1, 10) present")) {
        fz::FreeBlockOptions o;
        o.rss_cap_bytes = 1;
        const fz::FreeBlockResult R = fz::FreeBlockCrt(Input(it->second), o);
        check(Refused(R, fz::FreeBlockRefusal::RssCeiling) && R.primes.empty(), "(R) a 1-byte ceiling refuses [RssCeiling] before any lane");
        std::cout << "  (R) " << R.refusal << std::endl;
    }
    std::cout << "ARM (R) checks=" << g_checks - c0 << std::endl;
}

// ---- (P, fix round 1) the premises of the two proofs (audit F2, F3) and the inadmissible-prime path (audit F7) -------------------
void PremiseTeeth() {
    // F2: a singular G' = Phi_0^T B is refused [SingularG] before any lane, on both routes (the inverse; the witness search)
    if (auto it = g_cells.find({1, 60}); check(it != g_cells.end(), "(P) (1, 60) present for the G' teeth")) {
        const Loaded& L = it->second;
        fz::SparseColsZ Z0 = L.fx.Phi0, Z1 = L.fx.Phi0, Z2 = L.fx.Phi0;
        for (auto& c : Z0.cols) c.clear();
        Z1.cols[0].clear();
        Z2.cols[0] = Z2.cols[1];   // two equal rows of G': singular with a non-zero Hadamard bound (the witness search must PROVE det = 0)
        const fz::PackedZ Z0p = Z0.to_packed(), Z1p = Z1.to_packed(), Z2p = Z2.to_packed();
        const std::vector<std::pair<std::string, const fz::PackedZ*>> bad = {
            {"Phi_0 := 0", &Z0p}, {"Phi_0 column 1 := 0", &Z1p}, {"Phi_0 column 1 := column 2", &Z2p}};
        for (const auto& [what, M] : bad) {
            fz::FreeBlockInput in = Input(L);
            in.Phi0 = M;
            fz::FreeBlockOptions apost_only;
            apost_only.use_apriori = false;
            const fz::FreeBlockResult Ra = fz::FreeBlockCrt(in), Rb = fz::FreeBlockCrt(in, apost_only);
            check(Refused(Ra, fz::FreeBlockRefusal::SingularG) && Ra.primes.empty(), "(P) (1, 60) " + what + ": refused [SingularG] (default), no lane run");
            check(Refused(Rb, fz::FreeBlockRefusal::SingularG) && Rb.primes.empty(),
                  "(P) (1, 60) " + what + ": refused [SingularG] on the a-posteriori route (det G' = 0 proven by the witness search)");
        }
        // the witness on a nonsingular G': the a-posteriori route records it
        fz::FreeBlockOptions apost_only;
        apost_only.use_apriori = false;
        const fz::FreeBlockResult Rw = fz::FreeBlockCrt(Input(L), apost_only);
        check(Rw.ok && Rw.nonsingular_by == "witness" && Rw.det_witness != 0, "(P) (1, 60) a-posteriori only: det G' != 0 by a witness prime");
        std::cout << "  (P) (1, 60) G' teeth: 3 singular Phi_0 refused on both routes; witness q=" << Rw.det_witness << std::endl;
    }
    // F3: the a-priori premise.  7 b_k (k = 1, 2, 3) is NOT a Z-basis of L (det(Phi_cert^T B') = +-343): refused [Uncertified]; no
    // certificate at all: refused [Uncertified]; a faulty lane: the a-priori post-check refuses [BoundViolated], and the a-posteriori
    // route does not close on it either ([NoProof]).
    if (auto it = g_cells.find({1, 30}); check(it != g_cells.end(), "(P) (1, 30) present for the premise teeth")) {
        const Loaded& L = it->second;
        fz::SparseColsZ B7 = L.fx.B;
        for (uint32_t k = 0; k < 3 && k < B7.ncols(); ++k)
            for (auto& e : B7.cols[k]) e.second = e.second * BigInt(7);
        const fz::PackedZ B7p = B7.to_packed();
        check(Refused(fz::FreeBlockCrt(Input(L, &B7p)), fz::FreeBlockRefusal::Uncertified), "(P) (1, 30) 7 b_1, 7 b_2, 7 b_3 (index 343): refused [Uncertified]");
        fz::FreeBlockInput nc = Input(L);
        nc.PhiCert = nullptr;
        check(Refused(fz::FreeBlockCrt(nc), fz::FreeBlockRefusal::Uncertified), "(P) (1, 30) use_apriori without a certificate: refused [Uncertified]");
        fz::FreeBlockOptions bad;
        const fz::FreeBlockResult Rc = fz::FreeBlockCrtTestCorruptLane(Input(L), bad, 0);
        check(Refused(Rc, fz::FreeBlockRefusal::BoundViolated), "(P) (1, 30) a corrupted lane residue: the a-priori post-check refuses [BoundViolated]");
        std::cout << "  (P) (1, 30) " << Rc.refusal << std::endl;
        fz::FreeBlockOptions bad2 = bad;
        bad2.use_apriori = false;
        bad2.max_primes = 24;
        check(Refused(fz::FreeBlockCrtTestCorruptLane(Input(L), bad2, 0), fz::FreeBlockRefusal::NoProof),
              "(P) (1, 30) a corrupted lane residue, a-posteriori only: no proof ([NoProof] within 24 lanes), never a wrong F");
    }
    // N1 (re-audit): the premises d^1 B = 0 and Phi_0^T d^0 = 0 are checked on EVERY proof route, before any lane.  A B that is not a set
    // of cocycles (column 1 + e_0) and a Phi_0 that does not kill d^0 (Phi_0 := +-B; Phi_0 column 1 + 1000 e_i with i in R0 outside
    // supp B, which leaves G' unchanged) are refused [Uncertified] on the default and on the a-posteriori-only route, with no lane run.
    for (const auto& cell : std::vector<std::pair<uint32_t, uint32_t>>{{1, 30}, {9, 8}}) {
        auto it2 = g_cells.find(cell);
        if (!check(it2 != g_cells.end(), "(P) " + Cell(cell.first, cell.second) + " present for the N1 teeth")) continue;
        const Loaded& L = it2->second;
        fz::SparseColsZ Bn = L.fx.B, Pm = L.fx.B, Pe = L.fx.Phi0;
        {
            auto& c = Bn.cols[1];
            if (!c.empty() && c[0].first == 0) {
                c[0].second += BigInt(1);
                if (c[0].second.is_zero()) c.erase(c.begin());
            } else c.insert(c.begin(), std::make_pair(0u, BigInt(1)));
        }
        for (auto& col : Pm.cols)
            for (auto& e : col) e.second = -e.second;
        std::set<uint32_t> inB;
        for (const auto& col : L.fx.B.cols)
            for (const auto& e : col) inB.insert(e.first);
        uint32_t ri = ~0u;
        for (uint32_t i : L.fx.R0)
            if (!inB.count(i)) {
                ri = i;
                break;
            }
        if (!check(ri != ~0u, "(P) " + Cell(cell.first, cell.second) + ": a row of R0 outside supp B exists")) continue;
        {
            auto& c = Pe.cols[1];
            auto pos = std::lower_bound(c.begin(), c.end(), ri, [](const std::pair<uint32_t, BigInt>& a, uint32_t v) { return a.first < v; });
            if (pos != c.end() && pos->first == ri) pos->second += BigInt(1000);
            else c.insert(pos, std::make_pair(ri, BigInt(1000)));
        }
        const fz::PackedZ Bnp = Bn.to_packed(), Pmp = Pm.to_packed(), Pep = Pe.to_packed();
        fz::FreeBlockOptions deflt, apost_only;
        apost_only.use_apriori = false;
        for (int route = 0; route < 2; ++route) {
            const fz::FreeBlockOptions& o = route ? apost_only : deflt;
            const std::string t = "(P) " + Cell(cell.first, cell.second) + (route ? " (a-posteriori only)" : " (default)");
            const fz::FreeBlockResult Rn = fz::FreeBlockCrt(Input(L, &Bnp), o);
            check(Refused(Rn, fz::FreeBlockRefusal::Uncertified) && Rn.primes.empty(), t + ": B column 1 + e_0 (d^1 B != 0): refused [Uncertified], no lane");
            fz::FreeBlockInput in = Input(L);
            in.Phi0 = &L.Bp;
            const fz::FreeBlockResult Rb = fz::FreeBlockCrt(in, o);
            check(Refused(Rb, fz::FreeBlockRefusal::Uncertified) && Rb.primes.empty(), t + ": Phi_0 := B (does not kill d^0): refused [Uncertified], no lane");
            in.Phi0 = &Pmp;
            const fz::FreeBlockResult Rm = fz::FreeBlockCrt(in, o);
            check(Refused(Rm, fz::FreeBlockRefusal::Uncertified) && Rm.primes.empty(), t + ": Phi_0 := -B: refused [Uncertified], no lane");
            in.Phi0 = &Pep;
            const fz::FreeBlockResult Re = fz::FreeBlockCrt(in, o);
            check(Refused(Re, fz::FreeBlockRefusal::Uncertified) && Re.primes.empty(),
                  t + ": Phi_0 column 1 + 1000 e_" + std::to_string(ri) + " (G' unchanged, Phi_0^T d^0 != 0): refused [Uncertified], no lane");
            if (route == 0 && cell.first == 1) std::cout << "  (P) N1 " << Rn.refusal << "\n  (P) N1 " << Rb.refusal << std::endl;
        }
    }
}

uint32_t InvMod(uint32_t a, uint32_t q) {
    uint64_t r = 1, b = a % q, e = q - 2;
    while (e) {
        if (e & 1) r = r * b % q;
        b = b * b % q;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

void InadmissibleTooth() {
    // F7: B with column 1 multiplied by the first lane prime q_1: lane q_1 has G'' = Psi B' singular mod q_1 and is skipped; every
    // other lane returns F' = D^{-1} F D mod q, D = diag(q_1, 1, .., 1)
    auto it = g_cells.find({1, 30});
    if (!check(it != g_cells.end(), "(P) (1, 30) present for the inadmissible-prime tooth")) return;
    const Loaded& L = it->second;
    const uint32_t r = L.fx.r;
    uint32_t q1 = 0;
    for (uint32_t q : fz::CachedCrtPrimes(16))
        if ((6ull * L.fx.N) % q != 0) {
            q1 = q;
            break;
        }
    fz::SparseColsZ Bq = L.fx.B;
    for (auto& e : Bq.cols[0]) e.second = e.second * BigInt(int64_t(q1));
    fz::FreeBlockOptions o;
    o.lanes_only = 6;
    o.threads = 1;
    const fz::PackedZ Bqp = Bq.to_packed();
    const fz::FreeBlockResult R = fz::FreeBlockCrt(Input(L, &Bqp), o);
    const bool skipped = R.ok && R.skipped.size() == 1 && R.skipped[0] == q1 && R.primes.size() == 6 && R.lane_residues.size() == 6 &&
                         std::find(R.primes.begin(), R.primes.end(), q1) == R.primes.end();
    check(skipped, "(P) (1, 30) column 1 x q_1 = " + std::to_string(q1) + ": q_1 is skipped (inadmissible), six later lanes run");
    bool eq = skipped;
    for (size_t l = 0; eq && l < R.primes.size(); ++l) {
        const uint32_t q = R.primes[l], d = q1 % q, di = InvMod(d, q);
        for (size_t o2 = 0; eq && o2 < L.fx.ps.size(); ++o2) {
            const auto& F = L.fx.F_model.at(L.fx.ps[o2]);
            for (uint32_t i = 0; eq && i < r; ++i)
                for (uint32_t j = 0; eq && j < r; ++j) {
                    uint64_t f = F[size_t(i) * r + j].mod_u32(q);
                    if (j == 0) f = f * d % q;
                    if (i == 0) f = f * di % q;
                    eq = R.lane_residues[l][o2 * r * r + size_t(i) * r + j] == f;
                }
        }
    }
    check(eq, "(P) (1, 30) column 1 x q_1: every lane's residues = D^{-1} F D mod q");
    std::cout << "  (P) (1, 30) inadmissible q_1=" << q1 << ": skipped=" << R.skipped.size() << " lanes=" << R.primes.size() << std::endl;
}

// ---- (U) UpReal rounds upward (audit F4 c), against exact BigInt arithmetic --------------------------------------------------------
// value(u) * 2^S as an exact integer (S >= -u.e)
BigInt Scaled(const fz::UpReal& u, int64_t S) {
    if (u.m == 0) return BigInt(0);
    ENGINE_CHECK_MSG(int64_t(u.e) + S >= 0 && int64_t(u.e) + S < 4096, "(U) Scaled: the shift must be a small non-negative exponent");
    return BigInt::from_u64(u.m) * BigInt::pow(BigInt(2), static_cast<uint64_t>(int64_t(u.e) + S));
}
void ArmUpReal() {
    const uint64_t c0 = g_checks;
    std::mt19937_64 rng(0x5eed2025u);
    uint32_t bad_add = 0, bad_mul = 0, bad_big = 0, bad_ceil = 0, bad_norm = 0;
    auto rnd_up = [&](int k) {
        fz::UpReal u;
        const int kind = k % 8;
        u.m = kind == 0 ? ~0ull : kind == 1 ? (1ull << 63) : (rng() | (1ull << 63));
        u.e = static_cast<int32_t>(int64_t(rng() % 181) - 90);
        return u;
    };
    auto normal = [](const fz::UpReal& u) { return u.m == 0 || (u.m >> 63) == 1; };
    const int kCases = 20000;
    for (int k = 0; k < kCases; ++k) {
        fz::UpReal a = rnd_up(k), b = rnd_up(k / 8);
        if (k % 5 == 0) b.e = a.e - static_cast<int32_t>(rng() % 130);   // shifts 0 .. 129: the >= 64 edge and the carry edge
        // UpAdd: exact <= result <= exact + 2 ulp (S: every exponent met shifted to >= 0)
        const fz::UpReal s = fz::UpAdd(a, b);
        const int64_t S = std::max<int64_t>(0, -std::min<int64_t>({int64_t(a.e), int64_t(b.e), int64_t(s.e)}));
        const BigInt ex = Scaled(a, S) + Scaled(b, S), got = Scaled(s, S), ulp2 = BigInt::pow(BigInt(2), static_cast<uint64_t>(int64_t(s.e) + S + 1));
        if (!(got >= ex) || got - ex > ulp2) ++bad_add;
        // UpMul
        const fz::UpReal p = fz::UpMul(a, b);
        const int64_t S2 = std::max<int64_t>(0, -std::min<int64_t>(int64_t(a.e) + b.e, int64_t(p.e)));
        const BigInt exm = BigInt::from_u64(a.m) * BigInt::from_u64(b.m) * BigInt::pow(BigInt(2), static_cast<uint64_t>(int64_t(a.e) + b.e + S2));
        const BigInt gotm = Scaled(p, S2), ulpm = BigInt::pow(BigInt(2), static_cast<uint64_t>(int64_t(p.e) + S2 + 1));
        if (!(gotm >= exm) || gotm - exm > ulpm) ++bad_mul;
        if (!normal(s) || !normal(p)) ++bad_norm;
    }
    for (int k = 0; k < kCases; ++k) {
        // of_big: 1 .. 7 limbs (all-ones, sparse low limbs, every leading-zero count), >= |x| within one ulp
        const int limbs = 1 + k % 7;
        BigInt x(0);
        const BigInt two32 = BigInt::from_u64(1ull << 32);
        for (int l = 0; l < limbs; ++l) {
            uint32_t v = static_cast<uint32_t>(rng());
            if (k % 11 == 0) v = 0xffffffffu;
            if (k % 13 == 0 && l > 0 && l + 1 < limbs) v = 0;
            if (l == 0 && k % 17 == 0) v = 1;
            if (l + 1 == limbs) v >>= (rng() % 32);
            x = x * two32 + BigInt::from_u64(v);
        }
        if (k % 2) x = -x;
        const fz::UpReal u = fz::UpReal::of_big(x);
        if (x.is_zero()) {
            if (!u.is_zero()) ++bad_big;
            continue;
        }
        const int64_t S = u.e < 0 ? -int64_t(u.e) : 0;
        const BigInt ax = x.abs() * BigInt::pow(BigInt(2), static_cast<uint64_t>(S)), got = Scaled(u, S);
        if (!(got >= ax) || got - ax > BigInt::pow(BigInt(2), static_cast<uint64_t>(int64_t(u.e) + S)) || !normal(u)) ++bad_big;
        // ceil_big: the least integer >= the value (e from -70 to 10)
        fz::UpReal w = rnd_up(k);
        w.e = static_cast<int32_t>(int64_t(rng() % 81) - 70);
        const BigInt c = w.ceil_big();
        const int64_t T = w.e < 0 ? -int64_t(w.e) : 0;
        const BigInt cv = c * BigInt::pow(BigInt(2), static_cast<uint64_t>(T)), wv = Scaled(w, T);
        if (!(cv >= wv) || !((c - BigInt(1)) * BigInt::pow(BigInt(2), static_cast<uint64_t>(T)) < wv)) ++bad_ceil;
    }
    check(bad_add == 0, "(U) UpAdd >= the exact sum, within 2 ulp (" + std::to_string(kCases) + " cases; failures " + std::to_string(bad_add) + ")");
    check(bad_mul == 0, "(U) UpMul >= the exact product, within 2 ulp (failures " + std::to_string(bad_mul) + ")");
    check(bad_norm == 0, "(U) UpAdd / UpMul results normalised (failures " + std::to_string(bad_norm) + ")");
    check(bad_big == 0, "(U) UpReal::of_big >= |x| within one ulp, normalised (failures " + std::to_string(bad_big) + ")");
    check(bad_ceil == 0, "(U) UpReal::ceil_big = the least integer >= the value (failures " + std::to_string(bad_ceil) + ")");
    std::cout << "  (U) UpReal battery: " << 2 * kCases << " add/mul + " << kCases << " of_big + " << kCases << " ceil_big cases; bound entries checked in (A): "
              << g_bound_entries << std::endl;
    std::cout << "ARM (U) checks=" << g_checks - c0 << std::endl;
}

// ---- (Z) rank 0 (audit F1): b_1 = 0 at (1, 0) and at an odd n -------------------------------------------------------------------
void ArmRankZero() {
    const uint64_t c0 = g_checks;
    for (const auto& [N, n] : std::vector<std::pair<uint32_t, uint32_t>>{{1, 0}, {1, 7}}) {
        const std::string tag = "(Z) " + Cell(N, n);
        Loaded L;
        L.car = std::make_unique<fz::IsoTreeCarrierZ>(N, n);
        const fz::IsoTreeCarrierZ& car = *L.car;
        if (!check(car.b1() == 0, tag + ": b_1 = 0")) continue;
        // R0: a maximal independent set of rows of d^0 (greedy mod q0; its size is checked = rank d^0 = dim C^0 - b_0 by FreeBlockCrt)
        const fz::SparseZ& d0 = car.d(0);
        fz::EchelonModQ E(fz::kQ0, d0.cols);
        for (uint32_t i = 0; i < d0.rows; ++i) {
            fz::SpVec v;
            for (uint64_t k = d0.rowptr[i]; k < d0.rowptr[i + 1]; ++k)
                if (const uint32_t x = d0.val[k].mod_u32(fz::kQ0)) {
                    v.idx.push_back(d0.col[k]);
                    v.val.push_back(x);
                }
            if (E.insert(v)) L.fx.R0.push_back(i);
        }
        if (!check(L.fx.R0.size() == car.dim(0) - car.b0(), tag + ": |R0| = rank d^0")) continue;
        L.fx.N = N;
        L.fx.n = n;
        L.fx.r = 0;
        L.fx.dim1 = car.dim(1);
        L.fx.ps = {2, 3};
        L.fx.B.rows = L.fx.Phi0.rows = L.fx.PhiCert.rows = car.dim(1);
        L.pack();
        fz::FreeBlockOptions apost_only, lanes;
        apost_only.use_apriori = false;
        lanes.lanes_only = 2;
        const fz::FreeBlockResult Rd = fz::FreeBlockCrt(Input(L)), Ra = fz::FreeBlockCrt(Input(L), apost_only), Rl = fz::FreeBlockCrt(Input(L), lanes);
        for (const auto* R : {&Rd, &Ra}) {
            bool ok = R->ok && !R->refused && R->r == 0 && R->ops.size() == 2 && R->primes.empty();
            for (const auto& op : R->ops) ok = ok && op.proven && op.proof == "rank0" && op.F.empty();
            check(ok, tag + ": ok, both operators proven (rank0) with the empty F, no lane (" + (R == &Rd ? "default" : "a-posteriori only") + ")");
        }
        check(Rl.ok && Rl.primes.empty() && Rl.lane_residues.empty(), tag + ": lanes_only at rank 0: ok, nothing to run");
        std::cout << "  (Z) " << Cell(N, n) << " r=0 dimC1=" << car.dim(1) << " |R0|=" << L.fx.R0.size() << ": proven (rank0), no crash" << std::endl;
    }
    std::cout << "ARM (Z) checks=" << g_checks - c0 << std::endl;
}

// ---- (M) measurement ------------------------------------------------------------------------------------------------------------
uint32_t ArmMeasure() {
    const uint64_t c0 = g_checks;
    std::vector<std::string> files;
    std::string arm = "default";
    if (const char* e = std::getenv("HECKE_EXZ2P2_FIXTURES")) {
        arm = "env";
        std::string s(e), cur;
        for (char ch : s + ",") {
            if (ch == ',') {
                if (!cur.empty()) files.push_back(cur);
                cur.clear();
            } else cur += ch;
        }
    }
    std::cout << "EFFECTIVE HeckeZFreeBlockCrtValidation arm=" << arm << " cells=" << (arm == "env" ? files.size() : g_res.size()) << std::endl;
    uint32_t refused = 0;
    auto line = [](const fz::FreeBlockResult& R, const std::string& mode) {
        std::cout << "EXZ2_P2 cell=(" << R.N << "," << R.n << ") r=" << R.r << " mode=" << mode << " threads=" << R.threads << " route=" << R.route
                  << " gemm=" << R.gemm_path << " garner=" << R.garner_path << " lanes=" << R.primes.size() << " skipped=" << R.skipped.size()
                  << " rounds=" << R.rounds << std::fixed << std::setprecision(4) << " t_bound=" << R.t_bound << " t_lanes=" << R.t_lanes
                  << " lane_max=" << R.lane_seconds_max << " lane_sum=" << R.lane_seconds_sum << " t_garner=" << R.t_garner << " t_lift=" << R.t_lift
                  << " t_total=" << R.t_total << " rss_pred_MiB=" << (R.rss_pred >> 20) << " rho_memo_peak=" << R.lane_rho_memo_peak;
        for (const auto& op : R.ops)
            std::cout << " | T_" << op.p << " proof=" << (op.proof.empty() ? "-" : op.proof) << " used=" << op.primes_used << " apriori=" << op.primes_apriori
                      << " bits_F=" << op.bits_F << " bits_beta_H=" << op.bits_beta_H << " bits_beta_F=" << op.bits_beta_F;
        std::cout << std::endl;
    };
    if (arm == "default") {
        for (const auto& [cell, R] : g_res) line(R, "proven");
    } else {
        for (const std::string& f : files) {
            Loaded L;
            std::string why;
            const bool loaded = fz::LoadFreeBlockFixture(f, L.fx, why);   // (N2)
            if (!check(loaded, "(M) " + f + ": " + why)) {
                ++refused;
                continue;
            }
            L.car = std::make_unique<fz::IsoTreeCarrierZ>(L.fx.N, L.fx.n);
            L.pack();
            fz::FreeBlockOptions o;
            const bool cert = L.fx.PhiCert.ncols() > 0;
            if (!cert) o.lanes_only = 8;
            // (fix round 1, audit F3 c) mode=proven only through FreeBlockCrt's own re-certification of Phi_cert (refused [Uncertified]
            // otherwise -- an env fixture is never trusted on its word)
            const fz::FreeBlockResult R = fz::FreeBlockCrt(Input(L), o);
            if (cert && !R.refused) check(R.certified, "(M) " + Cell(L.fx.N, L.fx.n) + ": mode=proven ran on a re-certified basis");
            if (R.refused) {
                std::cout << "  (M) " << R.refusal << std::endl;
                ++refused;
                continue;
            }
            line(R, cert ? "proven" : "lanes_only");
            if (!cert && engine::cuda::device_enabled()) {   // the GPU lane route on the same lanes: timings + the residues compared
                fz::FreeBlockOptions g = o;
                g.route = fz::LaneRoute::Gpu;
                const fz::FreeBlockResult Rg = fz::FreeBlockCrt(Input(L), g);
                check(Rg.ok && Rg.primes == R.primes && Rg.lane_residues == R.lane_residues,
                      "(M) " + Cell(L.fx.N, L.fx.n) + ": the GPU lane route's residues = the CPU route's");
                line(Rg, "lanes_only");
            }
        }
    }
    std::cout << "ARM (M) checks=" << g_checks - c0 << std::endl;
    return refused;
}

}  // namespace

uint32_t HeckeZFreeBlockCrtValidation() {
    std::cout << "HeckeZFreeBlockCrtValidation: EXPORT-Z step 2 P2 -- the free block F_p over Z by RNS lanes and the proven CRT (fixture bases)"
              << std::endl;
    const auto t0 = clk::now();
    g_fail = 0;
    g_checks = 0;
    g_cells.clear();
    g_res.clear();
    g_root = FixtureRoot();
    if (!check(!g_root.empty(), "the fixture directory " + std::string(kFixRel) + " found (OC_SOURCE_DIR / cwd)")) {
        std::cout << "HECKEZ_FREEBLOCK_CRT arms=11 failing=" << g_fail << " checks=" << g_checks << " cells=0 verdict=FAIL" << std::endl;
        return 1;
    }
    ArmFixtures();
    ArmLanes();
    ArmProven();
    ArmProofs();
    ArmIdentity();
    ArmThreads();
    ArmRss();
    ArmUpReal();
    ArmRankZero();
    const uint32_t refused = ArmMeasure();
    const bool ok = g_fail == 0 && refused == 0;
    std::cout << "HECKEZ_FREEBLOCK_CRT arms=11 failing=" << g_fail << " checks=" << g_checks << " cells=" << g_res.size()
              << " verdict=" << (ok ? "OK" : "FAIL") << " seconds=" << std::fixed << std::setprecision(2) << since(t0) << std::endl;
    g_cells.clear();
    g_res.clear();
    return ok ? 0 : 1;
}
