// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EXPORT-Z step 2, package EXZ2-P3: HeckeZExportZ2Validation (standard) and HeckeZExportZ2SlowValidation (slow).  See the header.
#include "HeckeZExportZ2Validation.h"

#include "DescentFreeBasisZ.h"
#include "DescentFreeBlockCrt.h"
#include "DescentHecke.h"
#include "DescentModuleExport.h"
#include "DescentModuleExportZ.h"
#include "DescentRecord.h"
#include "IsoTreeCarrierZ.h"
#include "../HeckeZTor.h"
#include "../../Hecke/CoefficientRing.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../Cuda/CudaCtx.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace mx = integral_descent::modexport;
namespace fz = integral_descent::freez;
namespace h3 = integral_descent::hecke3;
namespace rec = integral_descent::record;
namespace tor = heckez::tor;
using fz::BigInt;
using clk = std::chrono::steady_clock;

uint32_t g_fail = 0;
uint64_t g_checks = 0, g_skips = 0;
bool check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_fail;
        if (g_fail <= 60) std::cout << "  [FAIL] " << what << std::endl;
    }
    return ok;
}
void Skip(const std::string& what, const std::string& why) {
    ++g_skips;
    std::cout << "  SKIP " << what << ": " << why << std::endl;
}
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
std::string Hex(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return o.str();
}
uint64_t Pow(uint64_t ell, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) x *= ell;
    return x;
}

// a record coefficient (decimal, or the FMTMP exact hex "0x.." / "-0x..") as an exact integer; false for "?" or the digit form
bool ParseCoefficient(const std::string& s, BigInt& v) {
    if (s.empty() || s == "?" || s[0] == '[') return false;
    const bool neg = s[0] == '-';
    size_t i = neg ? 1 : 0;
    if (s.compare(i, 2, "0x") == 0) {
        i += 2;
        if (i >= s.size()) return false;
        BigInt x(0);
        const BigInt sixteen(16);
        for (; i < s.size(); ++i) {
            const char ch = s[i];
            int dg;
            if (ch >= '0' && ch <= '9') dg = ch - '0';
            else if (ch >= 'a' && ch <= 'f') dg = ch - 'a' + 10;
            else return false;
            x = x * sixteen + BigInt(int64_t(dg));
        }
        v = neg ? -x : x;
        return true;
    }
    for (size_t k = i; k < s.size(); ++k)
        if (s[k] < '0' || s[k] > '9') return false;
    v = BigInt::from_string(s);
    return true;
}

struct Cell {
    uint32_t N = 1, n = 0;
    std::vector<uint32_t> ps;
    bool want_record = true;
    std::unique_ptr<fz::IsoTreeCarrierZ> car;
    mx::FreeBlockRun run;
    mx::ExportCache cache;
    mx::ModuleZ Z2, Z1;   // step 2, and step 1 on the SAME exports
    bool have_record = false;
    rec::DescentRecordZ R;
    double t_carrier = 0, t_total = 0, t_record = 0;
    std::string tag() const { return "(" + std::to_string(N) + ", " + std::to_string(n) + ")"; }
};

void BuildCell(Cell& c, bool verbose) {
    const auto t0 = clk::now();
    const auto tc = clk::now();
    c.car = std::make_unique<fz::IsoTreeCarrierZ>(c.N, c.n);
    c.t_carrier = since(tc);
    mx::ModuleZ2Options o;
    o.verbose = verbose;
    c.run = mx::RunFreeBlock(*c.car, c.ps, o);
    c.run.G.t_carrier = c.t_carrier;
    c.Z2 = mx::AssembleModuleZ2(c.N, c.n, c.ps, c.run, o, &c.cache);
    c.Z1 = mx::ExportModuleZ(c.N, c.n, c.ps, {}, [&c](uint32_t ell) { return c.cache.at(ell); });
    c.t_total = since(t0);
    if (c.want_record) {
        const auto tr = clk::now();
        const hecke::ScopedCoefficientRing z(0);
        rec::RecordOptions ro;
        ro.ps = c.ps;
        c.R = rec::BuildDescentRecord(c.N, c.n, ro);
        c.have_record = true;
        c.t_record = since(tr);
    }
    const mx::FreeBlockZ& G = c.Z2.global;
    std::cout << "  CELL " << c.tag() << " r=" << c.Z2.free_rank << " t=" << c.Z2.torsion_rank() << " status=" << c.Z2.status()
              << " free_block=" << c.Z2.free_block << " M_basis=" << c.Z2.M_basis << (c.Z2.free_refusal.empty() ? "" : " " + c.Z2.free_refusal)
              << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "EXZ2_P3 cell=" << c.tag() << " r=" << c.Z2.free_rank << " rule=" << G.basis_rule << " "
              << G.q0_key << " saturation=" << G.saturation << " basis_fnv=" << Hex(G.basis_fnv) << " phi_fnv=" << Hex(G.phi_fnv)
              << " bits_B=" << G.bits_B << " bits_Phi=" << G.bits_Phi << " K=" << G.K_pairs << " bound=" << G.bound_basis << " lanes=" << G.lanes
              << " t_carrier=" << G.t_carrier << " t_pairs=" << G.t_pairs << " t_sums=" << G.t_sums << " t_sat=" << G.t_sat << " t_cert=" << G.t_cert
              << " t_basis=" << G.t_basis << " t_crt=" << G.t_crt << " (t_bound=" << G.t_bound << " t_inverse=" << G.t_inverse << " t_lanes=" << G.t_lanes
              << " t_lift=" << G.t_lift << ") t_exports=" << G.t_exports << " t_assembly=" << G.t_assembly << " t_total=" << c.t_total;
    for (size_t o = 0; o < G.F.size(); ++o)
        std::cout << " | T_" << c.ps[o] << " proof=" << G.proof[o] << " primes=" << G.primes_used[o] << "/" << G.primes_apriori[o]
                  << " bits_F=" << G.bits_F[o] << " fnv=" << Hex(G.F_fnv[o]) << " M_fnv=" << (o < c.Z2.M_fnv.size() ? Hex(c.Z2.M_fnv[o]) : "-");
    std::cout << " | lanes_route=" << G.lane_route << " pi_B:";
    for (const mx::LocalPartZ& P : c.Z2.local) {
        if (P.t == 0) continue;
        uint32_t nz = 0;
        for (uint64_t v : P.pi_B) nz += v != 0;
        std::cout << " ell=" << P.ell << ":nz=" << nz << "/" << P.pi_B.size() << ":fnv=" << Hex(P.pi_B_fnv);
    }
    std::cout << std::endl;
}

// ---- (R) P2's lanes on P1b's real basis -------------------------------------------------------------------------------------------
struct FnvPin {
    uint64_t B, Phi;
    std::vector<uint64_t> F;
    std::vector<uint64_t> M = {}, piB = {};   // (fix round 1) M_fnv per op (MixedBlockFnv) and PiBFnv per torsion prime with t > 0
};
// pinned on the binary of this package (deterministic: iso_dixon/v3, the proven CRT); B / Phi_cert of (8, 10), (1, 200), (23, 20),
// (311, 2), (9, 8), (5, 12), (1, 60), (1, 30) are P1b's figures of record (EXZ2P1b_report.md, fix round 1)
std::map<std::pair<uint32_t, uint32_t>, FnvPin>& Pins() {
    static std::map<std::pair<uint32_t, uint32_t>, FnvPin> p = {
        {{1, 10}, {0x0255fefc8633f257ull, 0xf47074c0d45aca6cull, {0xb69e2fb38c33b922ull, 0x1d6f006cfdd59e32ull}, {0x4d757b562b22b23bull, 0xfc382b8f12d93c68ull}, {0x57b91eb02019b90eull, 0x114376c56538a6c9ull}}},
        {{2, 6}, {0x7452d339bd2a059full, 0xbd25f5d66d14ec1bull, {0x5a962d632790e086ull, 0xc7e27adae03a131eull}, {0xf6257e6c54713c24ull, 0x5e79022999c83376ull}, {0x75492c72c0d9da12ull, 0x4b0bd4505812903full, 0x4f5beb9eb9b0fb2full}}},
        {{9, 8}, {0x015cdfd0c7e89ff9ull, 0xe2794b92cdbd6d5eull, {0x5e85783d9bfc4fdaull, 0x8bc4dc8dd022030dull}, {0xeb226afe7f9810e2ull, 0xd1c1b91bd503cd07ull}, {0x5668e1ef879a5cd5ull, 0x1f1aa07cb306f3ebull, 0x68b51241f6ace4eeull}}},
        {{11, 0}, {0xffd1b2f569decd23ull, 0x21b108edabcff66cull, {0x8ef3bae552efe0e5ull, 0x7260db88f3363f5aull}, {0x0b841277bf2429acull, 0x53af7973f8fee117ull}, {}}},
        {{11, 2}, {0xd1e86c42335955adull, 0x7d098f7bfc5c478bull, {0xd9fd900fdab5cc84ull, 0x1451856612481f05ull}, {0x69c3f08c30a10e2eull, 0x26199dd54b4b6c73ull}, {0xd0378ec412a06e5eull}}},
        {{1, 30}, {0xe3c0fe41e3e1a148ull, 0x9f5440700a6c94beull, {0xbd7fa9de794036d7ull, 0x854e34e23f08596dull}, {0xbb06e16d86960b1dull, 0xc90655e9b42f13b3ull}, {0x95439f06807eb7e5ull, 0xca40e87c6e4616c4ull, 0x4a4d16700999ef90ull, 0x68ee70e96ff3ff35ull}}},
        {{5, 12}, {0x5f4fb2768aecf020ull, 0x454c2932552ff8c0ull, {0xdb5b0d4e02a10378ull, 0x6435cfe4c5eac3a9ull}, {0xb4e1e5be0b8e9819ull, 0x124e0104542778a1ull}, {0xe0b0572e0ffd3eb6ull, 0xa7080cf433a618bdull, 0x321500da2b68c71dull, 0xc0cdd577f8598022ull}}},
        {{1, 60}, {0x058d14a6944cf5f6ull, 0x8834536476449338ull, {0x69c9fd4fcf6f181dull, 0x0d02fb2ea2b36682ull}, {0xfa7ae63417cf0f64ull, 0x47381499d6bc547cull}, {0x27b20c778062de29ull, 0xdfad16121e895406ull, 0x86826b862ddded8dull, 0x5bd6ba0a05df592full, 0x3c0ca0308ec44d9cull, 0x393834acd3df0d6cull, 0xb8712562123144c1ull}}},
        {{8, 10}, {0x856a51817eee0ea4ull, 0x8f86c5f1fbc24865ull, {0x52351a862177f133ull, 0xa22dfad15ada9738ull}, {0x89f5f0d42b048632ull, 0x7bc778ad7bdded2dull}, {0x3dabf6103c7bce02ull, 0x2e82bdee26e4e4bbull}}},
        {{1, 200}, {0xaf2e9497b0cfdcc3ull, 0x2a92da56a377d513ull, {0xb0ec3aca48139578ull, 0x73b60c9baa956ddeull}, {0x59a39acd8257f6b9ull, 0x1b07acc63aff9847ull}, {0x1c1d1072b0a74c2bull, 0x47eef534e70a3877ull, 0x97a72e10e812ca73ull, 0xe6e49bde96c422a0ull, 0x2e156985d2bb6d3eull, 0x8194bd525af648beull}}},
        {{23, 20}, {0x8a0638648acb2b72ull, 0x3483cd183cf5e264ull, {0xdebbb9afb6dc918full, 0x279647a83e92dda9ull}, {0x87ae4730993dc573ull, 0x5c7261426b01e2a6ull}, {0xd09450e3da8b638cull, 0x73bb06d2da285228ull, 0x23b65033589c599aull, 0xdf038e5bc6b21ad1ull}}},
        {{311, 2}, {0x52c5e3c6694b0ea8ull, 0x455e7145603178a5ull, {0x0c64770798b731b4ull, 0xab5996a3898da5ebull}, {0xd6e6ea33c8cb7e8eull, 0xaa28e2c87db6e660ull}, {0x0b3497929391b85full}}},
    };
    return p;
}

void ArmR(std::vector<Cell>& cells, bool heavy) {
    for (Cell& c : cells) {
        const std::string t = "(R) " + c.tag();
        const mx::FreeBlockRun& run = c.run;
        if (!check(run.refusal.empty() && c.Z2.ok && c.Z2.free_block == "exact", t + ": the free block is exact (" + run.refusal + ")")) continue;
        const uint32_t r = c.Z2.free_rank;
        check(c.Z2.global.basis_rule == "iso_dixon/v3" && (r == 0 ? c.Z2.global.saturation == "rank0" : c.Z2.global.saturation == "det_pm1"),
              t + ": basis_rule iso_dixon/v3, saturation det_pm1");
        bool proven = true;
        for (const auto& pr : c.Z2.global.proof) proven = proven && (pr == "apriori" || pr == "apost" || pr == "rank0");
        check(proven && c.Z2.global.F.size() == c.ps.size(), t + ": every T_p proven");
        if (r == 0) continue;
        {
            const bool want_cert = uint64_t(r) * (uint64_t(run.FB.pairs.pairs.at(0).Phi.max_bits()) + run.FB.B.max_bits()) > fz::kAutoBoundBits;
            check(run.CR.certified && run.CR.bound_basis == (want_cert ? "phi_cert" : "phi0"),
                  t + ": certified by P1b's CertifyBasis; the Auto bound is " + (want_cert ? "Phi_cert" : "Phi_0") + " by the size rule");
        }
        // (fix round 1, audit F5) the lanes' route follows ENGINE_DEVICE (FromDevice): gpu iff device_enabled()
        check(c.Z2.global.lane_route == (engine::cuda::device_enabled() ? "gpu" : "cpu") && run.CR.route == c.Z2.global.lane_route,
              t + ": the lanes ran on the ENGINE_DEVICE route (" + c.Z2.global.lane_route + ")");
        // (Z3a fix1, audit F1) the production path REUSES BuildFreeBasis' certificate (stored fingerprints + carrier key match) ...
        check(run.FB.cert.ok && run.CR.cert_reused && run.CR.certified && run.CR.cert_rank_q == run.FB.cert.rank_q && run.CR.cert_primes == run.FB.cert.primes,
              t + ": the production path reuses BuildFreeBasis' certificate (cert_reused)");
        const fz::Pair& P0 = run.FB.pairs.pairs.at(0);
        fz::FreeBlockInput in;
        in.car = c.car.get();
        in.B = &run.FB.B;
        in.Phi0 = &P0.Phi;
        in.PhiCert = &run.FB.Phi_cert;
        in.R0 = P0.R0;
        in.P = P0.P;
        in.ps = c.ps;
        auto same = [&](const fz::FreeBlockResult& X, const std::string& what, bool primes = false) {
            bool eq = X.ok && X.ops.size() == c.ps.size();
            for (size_t o = 0; eq && o < X.ops.size(); ++o) eq = X.ops[o].proven && X.ops[o].fnv == c.Z2.global.F_fnv[o] && X.ops[o].F == c.Z2.global.F[o];
            for (size_t o = 0; eq && primes && o < X.ops.size(); ++o) eq = X.ops[o].primes_used == c.Z2.global.primes_used[o] && X.ops[o].proof == c.Z2.global.proof[o];
            check(eq, t + ": " + what + ": the same F (FNV and entries)" + (primes ? ", the same proof and primes_used" : ""));
        };
        fz::FreeBlockOptions o0;
        o0.bound_with = fz::BoundBasis::Phi0;
        const fz::FreeBlockResult R0 = fz::FreeBlockCrt(in, o0);
        fz::FreeBlockOptions oc;
        oc.bound_with = fz::BoundBasis::PhiCert;
        const fz::FreeBlockResult RC = fz::FreeBlockCrt(in, oc);
        same(RC, "the Phi_cert bound (G_c unimodular, with the Phi_0 a-posteriori test)");
        {   // (Z3a fix1, audit F1) FreeBlockCrt run FRESH (no PriorCert) re-proves the certificate and agrees with BuildFreeBasis' one
            check(!RC.cert_reused && RC.certified && RC.cert_primes == run.FB.cert.primes && (run.FB.pairs.q0 != fz::kQ0 || RC.cert_rank_q == run.FB.cert.rank_q),
                  t + ": the fresh FreeBlockCrt certificate (no PriorCert) agrees with BuildFreeBasis' (det primes, rank witness)");
            fz::FreeBlockInput im = in;   // a mismatched stored fingerprint: must re-run and certify, same F
            fz::Certificate bad = run.FB.cert;
            bad.fnv_B ^= 1;
            im.PriorCert = &bad;
            const fz::FreeBlockResult RM = fz::FreeBlockCrt(im, oc);
            check(!RM.cert_reused && RM.certified, t + ": mismatched fingerprint -> the certificate is re-run (cert_reused=false) and holds");
            same(RM, "mismatched-fingerprint arm");
            fz::FreeBlockInput ic = in;   // a mismatched carrier key
            fz::Certificate badc = run.FB.cert;
            badc.car_dim1 += 1;
            ic.PriorCert = &badc;
            const fz::FreeBlockResult RK = fz::FreeBlockCrt(ic, oc);
            check(!RK.cert_reused && RK.certified, t + ": mismatched carrier key -> the certificate is re-run and holds");
            // forged B (column 0 doubled) with the genuine PriorCert: the stored fingerprint no longer matches -> re-run -> det != +-1 -> refused
            std::vector<std::vector<BigInt>> Bd = run.FB.B.dense();
            for (BigInt& e : Bd.at(0)) e = e + e;
            const fz::PackedZ Bbad = fz::PackedZ::FromDense(Bd, run.FB.B.rows);
            fz::FreeBlockInput ib = in;
            ib.B = &Bbad;
            ib.PriorCert = &run.FB.cert;
            const fz::FreeBlockResult RB = fz::FreeBlockCrt(ib, oc);
            check(!RB.ok && !RB.cert_reused && RB.refusal.find("[Uncertified]") != std::string::npos,
                  t + ": forged B with the genuine PriorCert -> refused [Uncertified]");
        }
        same(R0, "the Phi_0 bound (P2)");
        std::cout << std::fixed << std::setprecision(4) << "EXZ2_P3_BOUND cell=" << c.tag() << " r=" << r;
        for (const fz::FreeBlockResult* X : {&RC, &R0}) {
            std::cout << " | bound=" << X->bound_basis << " t_bound=" << X->t_bound << " t_inverse=" << X->t_inverse << " steps=" << X->inverse_steps
                      << " t_cert=" << X->t_cert << " t_lanes=" << X->t_lanes << " t_total=" << X->t_total;
            for (const auto& op : X->ops)
                std::cout << " T_" << op.p << ":" << op.proof << (op.proof_detail.empty() ? "" : "(" + op.proof_detail + ")") << "/" << op.primes_used << "/" << op.primes_apriori << " bF=" << op.bits_F
                          << " bH=" << op.bits_beta_H << " bbF=" << op.bits_beta_F;
        }
        std::cout << std::endl;
        // the bound with Phi_cert holds entrywise: |G_c F| <= ceil(beta_H), |F| <= ceil(|G_c^{-1}| beta_H) (det +-1)
        for (size_t o = 0; o < RC.ops.size(); ++o) {
            const fz::FreeBlockOp& op = RC.ops[o];
            const std::vector<BigInt> H = fz::MatMulZ(RC.Gp, op.F, r);
            uint32_t bad = 0;
            for (size_t e = 0; e < H.size(); ++e) {
                if (H[e].abs() > op.beta_H_ceil[e]) ++bad;
                if (!op.apriori_rhs.empty() && op.F[e].abs() > op.apriori_rhs[e]) ++bad;
            }
            check(bad == 0 && RC.detG.abs() == BigInt(1) && RC.bound_basis == "phi_cert",
                  t + " T_" + std::to_string(op.p) + ": the Phi_cert bound holds entrywise (det G_c = +-1)");
        }
        // threads 1 vs 4, tile 1 (small r) -- the lanes are bit-identical
        fz::FreeBlockOptions o1;
        o1.bound_with = fz::BoundBasis::Auto;
        o1.threads = 1;
        same(fz::FreeBlockCrt(in, o1), "threads 1", true);
        o1.threads = 4;
        same(fz::FreeBlockCrt(in, o1), "threads 4", true);
        if (!heavy) {
            o1.threads = 16;
            same(fz::FreeBlockCrt(in, o1), "threads 16", true);
            o1.tile = 1;
            same(fz::FreeBlockCrt(in, o1), "tile 1", true);
        }
        // pins
        auto it = Pins().find({c.N, c.n});
        if (it == Pins().end()) {
            Skip(t + ": pins", "no pin for this cell");
            continue;
        }
        bool pf = it->second.F.size() == c.Z2.global.F_fnv.size();
        for (size_t o = 0; pf && o < it->second.F.size(); ++o) pf = it->second.F[o] == c.Z2.global.F_fnv[o];
        check(c.Z2.global.basis_fnv == it->second.B && c.Z2.global.phi_fnv == it->second.Phi, t + ": FNV(B), FNV(Phi_cert) = the pins");
        check(pf, t + ": FNV(F_p) = the pins");
        // (fix round 1) the span convention's FNVs: M per op, pi_B per torsion prime
        std::vector<uint64_t> pb;
        for (const mx::LocalPartZ& P : c.Z2.local)
            if (P.t > 0) pb.push_back(P.pi_B_fnv);
        check(it->second.M == c.Z2.M_fnv && it->second.piB == pb, t + ": FNV(M_p) (span convention) and FNV(pi_B) = the pins");
    }
}

// det(x - A) mod m by Berkowitz (division-free; high -> low), A r x r row-major residues mod m; m = 0 means 2^64 (wrap-around)
std::vector<uint64_t> CharpolyMod(const std::vector<uint64_t>& A, uint32_t r, uint64_t m) {
    auto mul = [m](uint64_t a, uint64_t b) -> uint64_t { return m ? static_cast<uint64_t>((unsigned __int128)a * b % m) : a * b; };
    auto add = [m](uint64_t a, uint64_t b) -> uint64_t { return m ? static_cast<uint64_t>(((unsigned __int128)a + b) % m) : a + b; };
    auto neg = [m](uint64_t a) -> uint64_t { return m ? (a ? m - a : 0) : uint64_t(0) - a; };
    std::vector<uint64_t> c = {1};
    if (r == 0) return c;
    c = {1, neg(A[0])};
    for (uint32_t k = 1; k < r; ++k) {
        std::vector<uint64_t> t(k + 2, 0), v(k);
        t[0] = 1;
        t[1] = neg(A[size_t(k) * r + k]);
        for (uint32_t i = 0; i < k; ++i) v[i] = A[size_t(i) * r + k];
        for (uint32_t i = 0; i < k; ++i) {
            uint64_t sv = 0;
            for (uint32_t j = 0; j < k; ++j) sv = add(sv, mul(A[size_t(k) * r + j], v[j]));
            t[i + 2] = neg(sv);
            if (i + 1 < k) {
                std::vector<uint64_t> nv(k, 0);
                for (uint32_t x = 0; x < k; ++x)
                    for (uint32_t y = 0; y < k; ++y) nv[x] = add(nv[x], mul(A[size_t(x) * r + y], v[y]));
                v = std::move(nv);
            }
        }
        std::vector<uint64_t> nc(k + 2, 0);
        for (uint32_t i = 0; i < k + 2; ++i)
            for (uint32_t j = 0; j <= std::min<uint32_t>(i, k); ++j) nc[i] = add(nc[i], mul(t[i - j], c[j]));
        c = std::move(nc);
    }
    return c;
}

// (A, slow cells) char(F) against D3's char_Z route itself (CharpolyOverZ, the shipped level-N operator, prop:crtcharpolyN): every
// certified coefficient mod 2^64 and mod 3^40 (F reduced; Berkowitz over the two rings), and the exact value where it fits 63 bits
void ArmACrt(std::vector<Cell>& cells) {
    const BigInt two64 = BigInt::pow(BigInt(2), 64);
    const uint64_t three40 = 12157665459056928801ull;   // 3^40
    const BigInt b3 = BigInt::from_u64(three40);
    auto low64 = [](const BigInt& m) {
        const auto& mg = m.magnitude();
        uint64_t v = 0;
        if (mg.size() > 0) v = mg[0];
        if (mg.size() > 1) v |= uint64_t(mg[1]) << 32;
        return v;
    };
    for (Cell& c : cells) {
        if (c.Z2.free_block != "exact") continue;
        const uint32_t r = c.Z2.free_rank;
        for (size_t o = 0; o < c.ps.size(); ++o) {
            const std::string t = "(A) " + c.tag() + " T_" + std::to_string(c.ps[o]);
            const auto tc = clk::now();
            const h3::CharpolyZ X = h3::CharpolyOverZ(c.N, c.n, c.ps[o]);
            if (X.refused || !X.determined || X.r != r) {
                Skip(t, "CharpolyOverZ is refused or undetermined: " + X.refusal);
                continue;
            }
            std::vector<uint64_t> A2(size_t(r) * r), A3(size_t(r) * r);
            for (size_t e = 0; e < A2.size(); ++e) {
                A2[e] = low64(c.Z2.global.F[o][e].floor_mod(two64));
                A3[e] = low64(c.Z2.global.F[o][e].floor_mod(b3));
            }
            const auto p2 = CharpolyMod(A2, r, 0), p3 = CharpolyMod(A3, r, three40);
            bool eq = p2.size() == X.c.size() && p3.size() == X.c.size();
            uint32_t exact = 0;
            for (size_t i = 0; eq && i < X.c.size(); ++i) {
                const h3::CrtCoefficient& k = X.c[i];
                eq = k.certified && p2[i] == k.mod_2_64 && p3[i] == k.mod_3_40;
                if (eq && k.fits) {
                    eq = BigInt(k.value).floor_mod(two64) == BigInt::from_u64(p2[i]);
                    ++exact;
                }
            }
            check(eq, t + ": char(F_p) = CharpolyOverZ's certified char_Z mod 2^64 and mod 3^40 (degree " + std::to_string(r) + ", " +
                          std::to_string(exact) + " coefficients < 2^63; " + std::to_string(since(tc)) + " s)");
        }
    }
}

// ---- (A) char(F) = the record's char_Z ---------------------------------------------------------------------------------------------
void ArmA(std::vector<Cell>& cells) {
    for (Cell& c : cells) {
        if (!c.have_record) {
            Skip("(A) " + c.tag(), "no record built for this cell");
            continue;
        }
        if (c.Z2.free_block != "exact") continue;
        const uint32_t r = c.Z2.free_rank;
        for (size_t o = 0; o < c.ps.size(); ++o) {
            const std::string t = "(A) " + c.tag() + " T_" + std::to_string(c.ps[o]);
            const rec::CharpolyEntry* ce = nullptr;
            for (const auto& e : c.R.charpoly)
                if (e.p == c.ps[o]) ce = &e;
            if (!ce || ce->refused || !ce->determined) {
                Skip(t, "the record's char_Z is not determined");
                continue;
            }
            std::vector<BigInt> rz;
            bool parsed = true;
            for (size_t i = 0; i < ce->coefficients.size(); ++i) {
                BigInt v;
                parsed = parsed && (ParseCoefficient(ce->coefficients[i], v) || (i < ce->exact.size() && ParseCoefficient(ce->exact[i], v)));
                rz.push_back(v);
            }
            const std::vector<BigInt> mine = fz::CharpolyBerkowitz(c.Z2.global.F[o], r);
            check(parsed && mine == rz, t + ": char(F_p) over Z = the record's char_Z(T_p | H^1/tors) (degree " + std::to_string(r) + ")");
        }
    }
}

// ---- (C) commutation over Z and on T (+) Z^r; the toys -------------------------------------------------------------------------
void ArmC(std::vector<Cell>& cells) {
    for (Cell& c : cells) {
        if (c.Z2.free_block != "exact") continue;
        const uint32_t r = c.Z2.free_rank, t = c.Z2.torsion_rank(), D = t + r;
        for (size_t a = 0; a < c.ps.size(); ++a)
            for (size_t b = a + 1; b < c.ps.size(); ++b) {
                const std::string tg = "(C) " + c.tag() + " T_" + std::to_string(c.ps[a]) + " T_" + std::to_string(c.ps[b]);
                check(fz::MatMulZ(c.Z2.global.F[a], c.Z2.global.F[b], r) == fz::MatMulZ(c.Z2.global.F[b], c.Z2.global.F[a], r),
                      tg + ": F_p F_q = F_q F_p over Z");
                const std::vector<BigInt> A = c.Z2.full_matrix(a), B = c.Z2.full_matrix(b);
                const std::vector<BigInt> AB = fz::MatMulZ(A, B, D), BA = fz::MatMulZ(B, A, D);
                uint32_t bad = 0;
                for (uint32_t i = 0; i < D; ++i) {
                    const BigInt m = i < t ? BigInt::from_u64(c.Z2.d[i]) : BigInt(0);
                    for (uint32_t j = 0; j < D; ++j) {
                        const BigInt x = AB[size_t(i) * D + j] - BA[size_t(i) * D + j];
                        if (i < t ? !x.floor_mod(m).is_zero() : !x.is_zero()) ++bad;
                    }
                }
                check(bad == 0, tg + ": the full T_p on T (+) Z^r commute (rows < t mod d_i; " + std::to_string(D) + " x " + std::to_string(D) + ")");
            }
    }
    auto toy = [&](uint32_t N, uint32_t n, uint32_t p, const std::vector<int64_t>& want) {
        for (Cell& c : cells) {
            if (c.N != N || c.n != n) continue;
            const size_t o = static_cast<size_t>(std::find(c.ps.begin(), c.ps.end(), p) - c.ps.begin());
            if (o >= c.ps.size() || c.Z2.free_block != "exact") continue;
            std::vector<BigInt> w;
            for (int64_t x : want) w.push_back(BigInt(x));
            check(fz::CharpolyBerkowitz(c.Z2.global.F[o], c.Z2.free_rank) == w, "(C) " + c.tag() + " T_" + std::to_string(p) + ": the hand-checked charpoly");
            return;
        }
    };
    toy(11, 0, 2, {1, 1, -8, -12});              // (x - 3)(x + 2)^2
    toy(1, 10, 2, {1, -2001, -97776, -1180224});  // (x - 2049)(x + 24)^2
}

// ---- (G) HZ-TOR on free_view local vs global --------------------------------------------------------------------------------------
void ArmG(std::vector<Cell>& cells, uint32_t& units) {
    for (Cell& c : cells) {
        if (c.Z2.free_block != "exact" || !c.Z1.ok) continue;
        for (size_t l = 0; l < c.Z2.local.size(); ++l) {
            if (c.Z2.local[l].t == 0) continue;
            ++units;
            const std::string t = "(G) " + c.tag() + " ell=" + std::to_string(c.Z2.local[l].ell);
            const mx::FreeView vg = c.Z2.free_view(0, l), vl = c.Z1.free_view(0, l);
            check(vg.basis_tag == "global" && vl.basis_tag == "local", t + ": free_view is tagged global (step 2) / local (step 1)");
            const tor::TorRead A = tor::SplitFromModuleZ(c.Z1, l), B = tor::SplitFromModuleZ(c.Z2, l);
            const bool words = A.status == B.status && A.split == B.split && A.jprime_length == B.jprime_length && A.algebra_length == B.algebra_length &&
                               A.blocks.size() == B.blocks.size();
            check(words, t + ": the HZ-TOR words (status, split, j', |A'|, blocks) are the same on both bases (" + A.split + ")");
            if (A.text() != B.text()) std::cout << "  (G) " << t << ": the read texts differ beyond the words (basis-dependent detail)" << std::endl;
        }
        check(tor::ModuleZSplitWord(c.Z1) == tor::ModuleZSplitWord(c.Z2), "(G) " + c.tag() + ": ModuleZSplitWord local == global");
    }
}

// C F == F_local C mod QK (r x r)
bool CommutesWithC(const std::vector<uint64_t>& C, const std::vector<uint64_t>& Fk, const std::vector<uint64_t>& Fl, uint32_t r, uint64_t QK) {
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            unsigned __int128 a = 0, b = 0;
            for (uint32_t k = 0; k < r; ++k) {
                a += (unsigned __int128)(C[size_t(i) * r + k] % QK) * (Fk[size_t(k) * r + j] % QK);
                b += (unsigned __int128)(Fl[size_t(i) * r + k] % QK) * (C[size_t(k) * r + j] % QK);
            }
            if (static_cast<uint64_t>(a % QK) != static_cast<uint64_t>(b % QK)) return false;
        }
    return true;
}
uint32_t g_units_H = 0, g_units_piB = 0;

// ---- (H) the global oracle -------------------------------------------------------------------------------------------------------
void ArmH(std::vector<Cell>& cells, bool heavy) {
    for (Cell& c : cells) {
        if (c.Z2.free_block != "exact") continue;
        const uint32_t r = c.Z2.free_rank;
        std::vector<uint32_t> done;
        for (size_t l = 0; l < c.Z2.local.size(); ++l) {
            const mx::LocalPartZ& P = c.Z2.local[l];
            done.push_back(P.ell);
            const std::string t = "(H) " + c.tag() + " ell=" + std::to_string(P.ell);
            for (size_t o = 0; o < c.ps.size(); ++o)
                check(c.Z2.free_mod(o, P.ell, P.K_C) == P.F_basis[o], t + " T_" + std::to_string(c.ps[o]) + ": F mod ell^" + std::to_string(P.K_C) +
                                                                          " = the route's T_p in B (" + P.E->route + ")");
            // prefix consistency of free_mod
            if (P.K_C >= 2) {
                const uint64_t q1 = Pow(P.ell, P.K_C - 1);
                const auto a = c.Z2.free_mod(0, P.ell, P.K_C), b = c.Z2.free_mod(0, P.ell, P.K_C - 1);
                bool eq = a.size() == b.size();
                for (size_t e = 0; eq && e < a.size(); ++e) eq = a[e] % q1 == b[e];
                check(eq, t + ": free_mod(ell^K) reduces to free_mod(ell^{K-1})");
            }
            if (P.t == 0) continue;
            const uint64_t QK = Pow(P.ell, P.K);
            // pi_B kept, its FNV recomputed; pi_B != 0 counted (audit F2: the units where the ker pi lifts and the classes [b_j] differ)
            check(P.pi_B.size() == size_t(P.t) * r && P.pi_TB.size() == c.ps.size() && P.pi_B_fnv == mx::PiBFnv(c.Z2, P) && P.pi_B_fnv != 0,
                  t + ": pi_ell(B) kept (t x r) with its FNV, pi_ell(T_p B) per op");
            bool piz = true;
            for (uint64_t v : P.pi_B) piz = piz && v == 0;
            ++g_units_H;
            if (!piz) ++g_units_piB;
            for (size_t o = 0; o < c.ps.size(); ++o) {
                const auto Ml = c.Z2.M_block(o, l), Fl = c.Z2.F_local(o, l), Fk = c.Z2.free_mod(o, P.ell, P.K);
                const std::string to = t + " T_" + std::to_string(c.ps[o]);
                // (fix round 1, audit F4 (ii)) THE M ORACLE, independent of M_local and C: the torsion coordinates of
                //   T_p [b_j] = sum_i F_ij [b_i] + sum_a M_aj g_a,   i.e.  pi(T_p B) == M + pi(B) F  (row a mod d_a),
                // pi(T_p B) by the route's own operator on the transported B (ell = 2, 3: the tree transport of T~_Iso; ell >= 5: the
                // Heilbronn operator), F the exact global F
                uint32_t bad = 0;
                for (uint32_t i = 0; i < P.t; ++i) {
                    const BigInt di = BigInt::from_u64(c.Z2.d[P.t0 + i]);
                    for (uint32_t j = 0; j < r; ++j) {
                        BigInt s = BigInt::from_u64(P.M_global[o][size_t(i) * r + j]);
                        for (uint32_t k = 0; k < r; ++k) s = s + BigInt::from_u64(P.pi_B[size_t(i) * r + k]) * c.Z2.global.F[o][size_t(k) * r + j];
                        if (s.floor_mod(di) != BigInt::from_u64(P.pi_TB[o][size_t(i) * r + j]).floor_mod(di)) ++bad;
                    }
                }
                check(bad == 0, to + ": the M oracle pi(T_p B) = M + pi(B) F (rows mod d_a; the route's operator on B, exact F; " + P.E->route + ")");
                check(mx::MixedOracleHolds(c.Z2.d, P.t0, P.t, r, P.pi_TB[o], P.M_global[o], P.pi_B, Fk), to + ": the production M oracle agrees");
                // the assembly's arithmetic (not an oracle: the production formula on the production's inputs)
                bool mok = true;
                for (uint32_t i = 0; i < P.t; ++i) {
                    const uint64_t di = c.Z2.d[P.t0 + i];
                    for (uint32_t j = 0; j < r; ++j) {
                        uint64_t s = 0;
                        for (uint32_t k = 0; k < r; ++k) s = (s + (Ml[size_t(i) * r + k] % di) * (P.C[size_t(k) * r + j] % di)) % di;
                        for (uint32_t k = 0; k < P.t; ++k)
                            s = (s + (c.Z2.tau[o][size_t(P.t0 + i) * c.Z2.torsion_rank() + P.t0 + k] % di) * (P.pi_B[size_t(k) * r + j] % di)) % di;
                        for (uint32_t k = 0; k < r; ++k) s = (s + di - (P.pi_B[size_t(i) * r + k] % di) * (Fk[size_t(k) * r + j] % di) % di) % di;
                        mok = mok && s == P.M_global[o][size_t(i) * r + j];
                    }
                }
                check(mok, to + ": M = M_local C - pi(B) F + tau pi(B) (rows mod d_i; the assembly's arithmetic)");
                check(CommutesWithC(P.C, Fk, Fl, r, QK), to + ": C_ell F = F_local C_ell mod ell^" + std::to_string(P.K));
            }
        }
        // the oracle at primes without torsion (the cokernel route; the field at ell >= 46341)
        const std::vector<uint32_t> extra = heavy ? std::vector<uint32_t>{7, 1000003} : std::vector<uint32_t>{5, 7, 13, 1000003, 2147483647u};
        for (uint32_t ell : extra) {
            if (c.N % ell == 0 || std::find(done.begin(), done.end(), ell) != done.end()) continue;
            const std::string t = "(H) " + c.tag() + " ell=" + std::to_string(ell);
            const mx::ModuleExport E = mx::ExportInBasis(c.N, c.n, ell, c.ps, c.run.FB.B);
            if (!check(E.ok && E.basis_ok, t + ": the export in B (" + E.refusal + E.basis_why + ")")) continue;
            if (!E.sigma.empty()) {
                Skip(t, "ell-torsion at a prime outside RecordTorsionPrimes");
                continue;
            }
            for (size_t o = 0; o < c.ps.size(); ++o) {
                const auto Fm = c.Z2.free_mod(o, ell, E.K_max);
                bool eq = true;
                for (uint32_t i = 0; i < r; ++i)
                    for (uint32_t j = 0; j < r; ++j) eq = eq && Fm[size_t(i) * r + j] == E.F_basis[o].at(i, j);
                check(eq, t + " T_" + std::to_string(c.ps[o]) + ": F mod ell^" + std::to_string(E.K_max) + " = the cokernel route's T_p in B (no torsion)");
            }
        }
    }
}

// ---- (L) module_Z("local") is step 1 ----------------------------------------------------------------------------------------------
void ArmL(std::vector<Cell>& cells) {
    for (Cell& c : cells) {
        if (!(c.N == 1 && c.n == 10) && !(c.N == 2 && c.n == 6) && !(c.N == 11 && c.n == 2) && !(c.N == 1 && c.n == 30)) continue;
        const mx::ModuleZ P = mx::ExportModuleZ(c.N, c.n, c.ps);   // plain exports
        const std::string t = "(L) " + c.tag();
        bool eq = P.ok && c.Z1.ok && P.tau == c.Z1.tau && P.d == c.Z1.d && P.tor_ell == c.Z1.tor_ell && P.tor_sigma == c.Z1.tor_sigma &&
                  P.free_rank == c.Z1.free_rank && P.local.size() == c.Z1.local.size() && P.free_block == "pending(step 2)" && P.M_basis == "local";
        for (size_t l = 0; eq && l < P.local.size(); ++l) {
            eq = P.local[l].retraction_fnv == c.Z1.local[l].retraction_fnv && P.local[l].K == c.Z1.local[l].K;
            for (size_t o = 0; eq && o < c.ps.size(); ++o)
                eq = P.local[l].t == 0 || (P.M_block(o, l) == c.Z1.M_block(o, l) && P.F_local(o, l) == c.Z1.F_local(o, l));
        }
        check(eq, t + ": the step-1 fields of the exports in B equal a plain step-1 run (tau, d, M_block, F_local, retractions)");
        bool eq2 = c.Z2.tau == P.tau && c.Z2.d == P.d;
        for (size_t l = 0; eq2 && l < P.local.size(); ++l)
            for (size_t o = 0; eq2 && o < c.ps.size(); ++o) eq2 = P.local[l].t == 0 || c.Z2.M_block(o, l) == P.M_block(o, l);
        check(eq2, t + ": step 2 keeps tau and the local blocks of step 1");
    }
}

// ---- (T) teeth --------------------------------------------------------------------------------------------------------------------
void ArmT(std::vector<Cell>& cells) {
    {
        std::vector<BigInt> X;
        uint32_t st = 0;
        check(!fz::InverseUnimodular({BigInt(2), BigInt(0), BigInt(0), BigInt(1)}, 2, X, &st, 64), "(T) InverseUnimodular refuses diag(2, 1)");
        const bool ok = fz::InverseUnimodular({BigInt(2), BigInt(3), BigInt(1), BigInt(2)}, 2, X, &st);
        check(ok && X == std::vector<BigInt>{BigInt(2), BigInt(-3), BigInt(-1), BigInt(2)}, "(T) InverseUnimodular([[2,3],[1,2]]) = [[2,-3],[-1,2]]");
        // a large unimodular matrix: U = L R with unit triangular factors of 40-bit entries
        const uint32_t r = 6;
        std::vector<BigInt> L(r * r, BigInt(0)), R(r * r, BigInt(0));
        uint64_t s = 0x9e3779b97f4a7c15ull;
        for (uint32_t i = 0; i < r; ++i) {
            L[i * r + i] = BigInt(1);
            R[i * r + i] = BigInt(i % 2 ? -1 : 1);
            for (uint32_t j = 0; j < r; ++j) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                const int64_t v = int64_t(s >> 24) - (int64_t(1) << 39);
                if (j < i) L[i * r + j] = BigInt(v);
                if (j > i) R[i * r + j] = BigInt(v / 3);
            }
        }
        const std::vector<BigInt> U = fz::MatMulZ(L, R, r);
        const bool ok2 = fz::InverseUnimodular(U, r, X, &st);
        std::vector<BigInt> I(r * r, BigInt(0));
        for (uint32_t i = 0; i < r; ++i) I[i * r + i] = BigInt(1);
        check(ok2 && fz::MatMulZ(U, X, r) == I && st > 1, "(T) InverseUnimodular on a 6 x 6 unimodular matrix with 80-bit entries: U X = I (" +
                                                             std::to_string(st) + " q-adic steps)");
    }
    for (Cell& c : cells) {
        if (!(c.N == 1 && c.n == 30)) continue;
        if (c.Z2.free_block != "exact") continue;
        const uint32_t r = c.Z2.free_rank;
        // a perturbed F fails the oracle at every torsion prime
        mx::ModuleZ Zp = c.Z2;
        Zp.global.F[0][0] += BigInt(1);
        bool caught = true;
        for (size_t l = 0; l < Zp.local.size(); ++l) caught = caught && Zp.free_mod(0, Zp.local[l].ell, Zp.local[l].K_C) != Zp.local[l].F_basis[0];
        check(caught, "(T) (1, 30): F_11 + 1 fails the global oracle at every torsion prime");
        // (fix round 1, audit F4 (i)) C = I in place of C_ell: at a unit where F_local != F mod ell^K the commutation check must fail
        // (a real tooth: CommutesWithC is the arm-(H) predicate), and such a unit must exist at (1, 30)
        bool found = false;
        for (size_t l = 0; l < c.Z2.local.size() && !found; ++l) {
            const mx::LocalPartZ& P = c.Z2.local[l];
            if (P.t == 0) continue;
            const uint64_t QK = Pow(P.ell, P.K);
            const auto Fl = c.Z2.F_local(0, l), Fk = c.Z2.free_mod(0, P.ell, P.K);
            bool differs = false;
            for (size_t e = 0; e < Fl.size(); ++e) differs = differs || Fl[e] % QK != Fk[e] % QK;
            if (!differs) continue;
            found = true;
            std::vector<uint64_t> I(size_t(r) * r, 0);
            for (uint32_t i = 0; i < r; ++i) I[size_t(i) * r + i] = 1;
            check(CommutesWithC(P.C, Fk, Fl, r, QK) && !CommutesWithC(I, Fk, Fl, r, QK),
                  "(T) (1, 30) ell=" + std::to_string(P.ell) + ": C_ell passes and C = I fails C F = F_local C mod ell^" + std::to_string(P.K));
        }
        check(found, "(T) (1, 30): a torsion unit with F_local != F mod ell^K exists (C_ell != 1 there)");
        // a basis column times 2 is not a basis of L / 2^K
        fz::PackedZ B2(c.run.FB.B.rows);
        for (uint32_t j = 0; j < r; ++j) {
            auto col = c.run.FB.B.col_sparse(j);
            if (j == 0)
                for (auto& e : col) e.second = e.second * BigInt(2);
            B2.append_col(col);
        }
        const mx::ModuleExport E = mx::ExportInBasis(c.N, c.n, 2, c.ps, B2);
        check(E.ok && !E.basis_ok, "(T) (1, 30): 2 b_1 is not a basis of L / 2^K (basis_ok false: " + E.basis_why + ")");
        // the Phi_cert bound refuses a Phi_cert that is not a certificate (7 phi_1)
        fz::PackedZ P7(c.run.FB.Phi_cert.rows);
        for (uint32_t j = 0; j < r; ++j) {
            auto col = c.run.FB.Phi_cert.col_sparse(j);
            if (j == 0)
                for (auto& e : col) e.second = e.second * BigInt(7);
            P7.append_col(col);
        }
        const fz::Pair& P0 = c.run.FB.pairs.pairs.at(0);
        fz::FreeBlockInput in;
        in.car = c.car.get();
        in.B = &c.run.FB.B;
        in.Phi0 = &P0.Phi;
        in.PhiCert = &P7;
        in.R0 = P0.R0;
        in.P = P0.P;
        in.ps = c.ps;
        fz::FreeBlockOptions o;
        o.bound_with = fz::BoundBasis::PhiCert;
        const fz::FreeBlockResult R = fz::FreeBlockCrt(in, o);
        check(!R.ok && R.code == fz::FreeBlockRefusal::Uncertified, "(T) (1, 30): 7 phi_1 is refused [Uncertified] before any lane (" + R.refusal + ")");
    }
    // (fix round 1, audit F2 / F4 (ii)) teeth of the M oracle: M + 1 fails it; the ker pi block M_kerpi = M_local C (the pre-fix
    // served block) fails it wherever pi(B) F - tau pi(B) != 0 (the audit's refuted units (1, 30) ell = 5, (11, 2) ell = 2, (9, 8)
    // ell = 2, 3, 7); and such a unit exists at each of the three cells
    for (Cell& c : cells) {
        if (!((c.N == 1 && c.n == 30) || (c.N == 11 && c.n == 2) || (c.N == 9 && c.n == 8))) continue;
        if (c.Z2.free_block != "exact") continue;
        const uint32_t r = c.Z2.free_rank;
        uint32_t refuted = 0;
        for (size_t l = 0; l < c.Z2.local.size(); ++l) {
            const mx::LocalPartZ& P = c.Z2.local[l];
            if (P.t == 0) continue;
            for (size_t o = 0; o < c.ps.size(); ++o) {
                const std::string to = "(T) " + c.tag() + " ell=" + std::to_string(P.ell) + " T_" + std::to_string(c.ps[o]);
                const auto Ml = c.Z2.M_block(o, l), Fk = c.Z2.free_mod(o, P.ell, P.K);
                std::vector<uint64_t> Mp = P.M_global[o], Mk(size_t(P.t) * r, 0);
                Mp[0] = (Mp[0] + 1) % c.Z2.d[P.t0];
                check(!mx::MixedOracleHolds(c.Z2.d, P.t0, P.t, r, P.pi_TB[o], Mp, P.pi_B, Fk), to + ": M + 1 fails the M oracle");
                for (uint32_t i = 0; i < P.t; ++i) {
                    const uint64_t di = c.Z2.d[P.t0 + i];
                    for (uint32_t j = 0; j < r; ++j) {
                        uint64_t s = 0;
                        for (uint32_t q = 0; q < r; ++q) s = (s + (Ml[size_t(i) * r + q] % di) * (P.C[size_t(q) * r + j] % di)) % di;
                        Mk[size_t(i) * r + j] = s;
                    }
                }
                if (Mk == P.M_global[o]) continue;   // pi(B) F = tau pi(B) here: both conventions agree
                ++refuted;
                check(!mx::MixedOracleHolds(c.Z2.d, P.t0, P.t, r, P.pi_TB[o], Mk, P.pi_B, Fk),
                      to + ": the ker pi block M_local C (the pre-fix M) fails the M oracle for the classes [b_j]");
            }
        }
        check(refuted > 0, "(T) " + c.tag() + ": a unit where the ker pi block and the span block differ exists (" + std::to_string(refuted) + " ops)");
    }
}

// ---- (D) the GPU lane route on the real bases (fix round 1, audit F5; slow tier; [SKIP] without ENGINE_DEVICE=gpu and a device) ----
// FreeBlockCrt with route = Cpu and route = Gpu (the production Auto bound) on each cell's certified basis: F (entries), FNV, proof and
// primes_used identical, and both equal to the served F; then ExportModuleZ2 itself with lanes = Gpu (the ModuleZ2Options plumbing the
// ENGINE_DEVICE switch selects) on (8, 10) against lanes = Cpu: the same F_fnv, M_fnv and pi_B FNVs.
void ArmD(std::vector<Cell>& cells) {
    if (!engine::cuda::device_enabled()) {
        std::cout << "  [SKIP] (D) GPU lane route: " << (engine::cuda::available() ? "ENGINE_DEVICE != gpu (opt-in)" : "no device") << std::endl;
        Skip("(D)", "the GPU lane route needs ENGINE_DEVICE=gpu and a device");
        return;
    }
    for (Cell& c : cells) {
        if (c.Z2.free_block != "exact" || c.Z2.free_rank == 0) continue;
        const std::string t = "(D) " + c.tag();
        const fz::Pair& P0 = c.run.FB.pairs.pairs.at(0);
        fz::FreeBlockInput in;
        in.car = c.car.get();
        in.B = &c.run.FB.B;
        in.Phi0 = &P0.Phi;
        in.PhiCert = &c.run.FB.Phi_cert;
        in.R0 = P0.R0;
        in.P = P0.P;
        in.ps = c.ps;
        fz::FreeBlockOptions oc, og;
        oc.bound_with = og.bound_with = fz::BoundBasis::Auto;
        oc.route = fz::LaneRoute::Cpu;
        og.route = fz::LaneRoute::Gpu;
        const auto t1 = clk::now();
        const fz::FreeBlockResult RC = fz::FreeBlockCrt(in, oc);
        const double tc = since(t1);
        const auto t2 = clk::now();
        const fz::FreeBlockResult RG = fz::FreeBlockCrt(in, og);
        const double tg = since(t2);
        bool eq = RC.ok && RG.ok && RG.route == "gpu" && RC.route == "cpu" && RC.ops.size() == c.ps.size() && RG.ops.size() == c.ps.size() &&
                  RG.primes == RC.primes;
        for (size_t o = 0; eq && o < c.ps.size(); ++o)
            eq = RG.ops[o].proven && RC.ops[o].proven && RG.ops[o].F == RC.ops[o].F && RG.ops[o].fnv == RC.ops[o].fnv &&
                 RG.ops[o].proof == RC.ops[o].proof && RG.ops[o].primes_used == RC.ops[o].primes_used && RC.ops[o].F == c.Z2.global.F[o];
        check(eq, t + ": the GPU lane route = the CPU lane route on the certified basis (F, FNV, proof, primes) = the served F (" + RG.refusal + ")");
        std::cout << std::fixed << std::setprecision(3) << "  (D) " << c.tag() << " r=" << c.Z2.free_rank << " lanes=" << RG.primes.size()
                  << " t_cpu_crt=" << tc << " t_gpu_crt=" << tg << " t_gpu_lanes=" << RG.t_lanes << " identical=" << (eq ? "yes" : "NO") << std::endl;
        if (c.N == 8 && c.n == 10) {
            mx::ModuleZ2Options a, b;
            a.lanes = mx::LaneChoice::Gpu;
            b.lanes = mx::LaneChoice::Cpu;
            const mx::ModuleZ Zg = mx::ExportModuleZ2(c.N, c.n, c.ps, a), Zc = mx::ExportModuleZ2(c.N, c.n, c.ps, b);
            std::vector<uint64_t> pg, pc;
            for (const auto& P : Zg.local) pg.push_back(P.pi_B_fnv);
            for (const auto& P : Zc.local) pc.push_back(P.pi_B_fnv);
            check(Zg.free_block == "exact" && Zc.free_block == "exact" && Zg.global.lane_route == "gpu" && Zc.global.lane_route == "cpu" &&
                      Zg.global.F_fnv == Zc.global.F_fnv && Zg.global.F == Zc.global.F && Zg.M_fnv == Zc.M_fnv && pg == pc &&
                      Zc.global.F_fnv == c.Z2.global.F_fnv && Zc.M_fnv == c.Z2.M_fnv,
                  t + ": ExportModuleZ2 with lanes = Gpu equals lanes = Cpu (F, F_fnv, M_fnv, pi_B FNVs) and the cell's object");
        }
    }
}

struct ArmDef {
    const char* name;
    std::function<void()> fn;
    uint64_t floor;
};

uint32_t RunArms(const char* title, const char* key, std::vector<Cell>& cells, const std::vector<ArmDef>& arms, clk::time_point t0, const uint32_t& units) {
    uint32_t failing = 0;
    for (const auto& [name, fn, floor] : arms) {
        g_fail = 0;
        g_skips = 0;
        const uint64_t c0 = g_checks;
        const auto ta = clk::now();
        try {
            fn();
        } catch (const std::exception& e) {
            ++g_fail;
            std::cout << "  [EXCEPTION] arm (" << name << "): " << e.what() << std::endl;
        }
        const uint64_t nc = g_checks - c0;
        if (nc < floor) {
            ++g_fail;
            std::cout << "  [FAIL] arm (" << name << "): " << nc << " checks, below the floor " << floor << std::endl;
        }
        if (g_fail) ++failing;
        std::cout << "ARM (" << name << ") checks=" << nc << " floor=" << floor << " skipped=" << g_skips << " failed=" << g_fail << " seconds=" << std::fixed
                  << std::setprecision(2) << since(ta) << std::endl;
    }
    (void)title;
    const bool ok = failing == 0;
    std::cout << key << " arms=" << arms.size() << " failing=" << failing << " cells=" << cells.size() << " checks=" << g_checks << " tor_units=" << units
              << " free_block=exact M_basis=global rule=iso_dixon/v3 verdict=" << (ok ? "OK" : "FAIL") << " seconds=" << std::fixed << std::setprecision(2)
              << since(t0) << std::endl;
    return ok ? 0 : 1;
}

uint32_t Run(bool slow) {
    const auto t0 = clk::now();
    g_checks = 0;
    std::vector<Cell> cells;
    const std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>> list =
        slow ? std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>>{{{8, 10}, {3, 5}}, {{1, 200}, {2, 3}}, {{23, 20}, {2, 3}}, {{311, 2}, {2, 3}}}
             : std::vector<std::pair<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>>>{{{1, 10}, {2, 3}}, {{2, 6}, {3, 5}}, {{9, 8}, {2, 5}},
                                                                                            {{11, 0}, {2, 3}}, {{11, 2}, {2, 3}}, {{1, 30}, {2, 3}},
                                                                                            {{5, 12}, {2, 3}}, {{1, 60}, {2, 3}}};
    try {
        for (const auto& [Nn, ps] : list) {
            Cell c;
            c.N = Nn.first;
            c.n = Nn.second;
            c.ps = ps;
            c.want_record = !slow;   // the slow cells: D3's char_Z route directly (ArmACrt), not the whole record
            BuildCell(c, slow);
            cells.push_back(std::move(c));
        }
    } catch (const std::exception& e) {
        std::cout << "  [EXCEPTION] building the cells: " << e.what() << std::endl;
        std::cout << (slow ? "HECKEZ_EXPORTZ2_SLOW" : "HECKEZ_EXPORTZ2") << " arms=0 failing=1 verdict=FAIL" << std::endl;
        return 1;
    }
    std::cout << "  cells built in " << std::fixed << std::setprecision(2) << since(t0) << " s" << std::endl;
    uint32_t units = 0;
    if (slow) {
        const uint64_t dfloor = engine::cuda::device_enabled() ? 5 : 0;   // (D) prints [SKIP] without a device
        const std::vector<ArmDef> arms = {{"R", [&] { ArmR(cells, true); }, 30}, {"A", [&] { ArmACrt(cells); }, 8}, {"C", [&] { ArmC(cells); }, 8},
                                          {"H", [&] { ArmH(cells, true); }, 20}, {"D", [&] { ArmD(cells); }, dfloor}};
        return RunArms("slow", "HECKEZ_EXPORTZ2_SLOW", cells, arms, t0, units);
    }
    const std::vector<ArmDef> arms = {{"R", [&] { ArmR(cells, false); }, 60}, {"A", [&] { ArmA(cells); }, 14}, {"C", [&] { ArmC(cells); }, 18},
                                      {"G", [&] { ArmG(cells, units); }, 20}, {"H", [&] { ArmH(cells, false); }, 80}, {"L", [&] { ArmL(cells); }, 8},
                                      {"T", [&] { ArmT(cells); }, 20}};
    return RunArms("standard", "HECKEZ_EXPORTZ2", cells, arms, t0, units);
}

}  // namespace

uint32_t HeckeZExportZ2Validation() {
    std::cout << "HeckeZExportZ2Validation: T_p = [[tau, M], [0, F]] on H^1 = T (+) Z^r with F exact over Z (slot-25 EXPORT-Z step 2, EXZ2-P3)" << std::endl;
    return Run(false);
}

uint32_t HeckeZExportZ2SlowValidation() {
    std::cout << "HeckeZExportZ2SlowValidation: EXPORT-Z step 2 on (8, 10), (1, 200), (23, 20), (311, 2) with stage timings (EXZ2-P3)" << std::endl;
    return Run(true);
}
