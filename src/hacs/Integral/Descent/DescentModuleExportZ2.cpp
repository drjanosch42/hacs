// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZ package EXPORT-Z, STEP 2 (slot 25, package EXZ2-P3): the free block F over Z in one picked basis (P1b's BuildFreeBasis + P2's
// FreeBlockCrt), the global M, the accessors free_view / free_mod / full_matrix.  See DescentModuleExportZ.h ("STEP 2") and
// docs/implementation/slot25/EXPORTZ2_design.md Sec. 5, 7.5; EXZ2P3_report.md.
#include "DescentModuleExportZ.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>

#include "DescentAssembly.h"
#include "DescentFreeBasisZ.h"
#include "DescentFreeBlockCrt.h"
#include "DescentHecke.h"
#include "../Model/IsotropyModelTree.h"
#include "../Model/TreeCochains.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Fnv.h"
#include "../../../Cuda/CudaCtx.h"

namespace integral_descent {
namespace modexport {

namespace {
using clk = std::chrono::steady_clock;
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
uint64_t PowU(uint32_t ell, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) {
        ENGINE_CHECK_MSG(x <= (~0ull) / ell, "ExportModuleZ2: ell^e overflows 64 bits");
        x *= ell;
    }
    return x;
}
// a^{-1} mod m (gcd(a, m) = 1), extended Euclid
uint64_t InvMod(uint64_t a, uint64_t m) {
    int64_t t = 0, nt = 1;
    int64_t r = static_cast<int64_t>(m), nr = static_cast<int64_t>(a % m);
    while (nr) {
        const int64_t q = r / nr;
        t -= q * nt;
        std::swap(t, nt);
        r -= q * nr;
        std::swap(r, nr);
    }
    ENGINE_CHECK_MSG(r == 1, "RouteBasis: a denominator of the comparison map is not a unit mod ell^K");
    return static_cast<uint64_t>(t < 0 ? t + static_cast<int64_t>(m) : t);
}
uint64_t RedRatMod(const integral_model::Rat& q, uint64_t M) {
    auto red = [M](int64_t v) -> uint64_t {
        int64_t r = v % static_cast<int64_t>(M);
        return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(M) : r);
    };
    const uint64_t a = red(q.num);
    if (q.den == 1) return a;
    return static_cast<uint64_t>((unsigned __int128)a * InvMod(red(q.den), M) % M);
}
// out[block j] += coef * rho(g) in, over Z / M (k columns row-major; the block-monomial action of TreeCochains.h)
void AddQRho(uint64_t M, const integral_model::TreeModule<uint32_t>& TM, const integral_model::QGroupRingElt& a, const uint32_t* in,
             uint32_t* out, uint32_t k) {
    for (const auto& [g, rq] : a.terms) {
        const uint64_t coef = RedRatMod(rq, M);
        if (!coef) continue;
        const integral_model::BlockMonomial<uint32_t>& B = TM.rho(integral_model::mat_of(g));
        const uint32_t d = B.d;
        for (uint32_t j = 0; j < B.m; ++j) {
            const std::vector<uint32_t>& blk = B.blocks[B.blk[j]];
            const uint32_t* src = in + size_t(B.col[j]) * d * k;
            uint32_t* dst = out + size_t(j) * d * k;
            for (uint32_t x = 0; x < d; ++x)
                for (uint32_t y = 0; y < d; ++y) {
                    const uint64_t f = coef * blk[size_t(x) * d + y] % M;
                    if (!f) continue;
                    uint32_t* yr = dst + size_t(x) * k;
                    const uint32_t* xr = src + size_t(y) * k;
                    for (uint32_t c = 0; c < k; ++c)
                        if (xr[c]) yr[c] = static_cast<uint32_t>((yr[c] + f * xr[c]) % M);
                }
        }
    }
}
// Psi^T_ell (degrees 0 .. 2), built once per ell
const integral_model::TreeMap& PsiTreeEllShared(uint32_t ell) {
    static std::mutex mu;
    static std::map<uint32_t, std::unique_ptr<integral_model::TreeMap>> memo;
    std::lock_guard<std::mutex> lk(mu);
    auto& e = memo[ell];
    if (!e) e = std::make_unique<integral_model::TreeMap>(integral_model::PsiTreeEll(ell, 2));
    return *e;
}
uint64_t VmKb(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    const std::string k(key);
    while (std::getline(f, line))
        if (line.rfind(k, 0) == 0) {
            std::istringstream is(line.substr(k.size()));
            uint64_t v = 0;
            is >> v;
            return v;
        }
    return 0;
}
}  // namespace

// =================================================================================================================================
// The transport of the global basis to an export route
// =================================================================================================================================
Mat RouteBasis(uint32_t N, uint32_t n, const freez::PackedZ& B, uint32_t ell, uint32_t& K_run) {
    ENGINE_CHECK_MSG(ell == 2 || ell == 3 || ell >= 5, "RouteBasis: ell");
    const uint32_t r = B.cols;
    K_run = ell <= 3 ? LocalWidth(ell) : RouteMaxK(ell);
    ENGINE_CHECK_MSG(K_run >= 1, "RouteBasis: no word ring at ell");
    const uint64_t M = PowU(ell, K_run);
    ENGINE_CHECK_MSG(M < (1ull << 32), "RouteBasis: ell^K_run >= 2^32");
    const engine::ModCtx<uint32_t> ctx = K_run == 1 ? engine::ModCtx<uint32_t>(ell) : engine::ModCtx<uint32_t>::prime_power(ell, K_run);
    const integral_model::TreeModule<uint32_t> TM = integral_model::TreeInducedSymModule(N, n, ctx);
    const uint32_t D = TM.dim();
    const integral_model::TreeModelChain T0(0, 2);
    ENGINE_CHECK_MSG(B.rows == T0.rank(1) * D, "RouteBasis: B is not a cochain of the free tree carrier Iso_T (dim C^1 = 3 D)");
    // B mod ell^{K_run} on C^1_{Iso_T}
    std::vector<uint32_t> X(size_t(B.rows) * r, 0);
    for (uint32_t j = 0; j < r; ++j)
        for (uint64_t e = B.colptr[j]; e < B.colptr[j + 1]; ++e) X[size_t(B.row[e]) * r + j] = B.mod_u32(e, static_cast<uint32_t>(M));
    // the pullback along Psi^T_ell : Iso^(ell)_T -> Iso_T in degree 1 (value[1][ell-cell x] = a chain on Iso_T)
    const integral_model::TreeMap& Psi = PsiTreeEllShared(ell);
    const auto& rows = Psi.value.at(1);
    const integral_model::TreeModelChain Tl(ell, 2);
    ENGINE_CHECK_MSG(rows.size() == Tl.rank(1), "RouteBasis: Psi^T_ell's source cells are not the ell-local tree's");
    std::vector<uint32_t> V(size_t(rows.size()) * D * r, 0);
    for (uint32_t x = 0; x < rows.size(); ++x)
        for (const auto& [y, b] : rows[x]) {
            ENGINE_CHECK_MSG(y < T0.rank(1), "RouteBasis: Psi^T_ell's target cell");
            AddQRho(M, TM, b, X.data() + size_t(y) * D * r, V.data() + size_t(x) * D * r, r);
        }
    if (ell <= 3) {
        // the J-coordinates of the tree route (the export's TreeTransport, rebuilt deterministically at the same width)
        const hecke3::TreeTransport Xt(ell, N, n, 1, K_run);
        ENGINE_CHECK_MSG(Xt.full(1) == V.size() / std::max<uint32_t>(1, r) || r == 0, "RouteBasis: the tree route's layout");
        std::vector<uint32_t> J;
        if (r > 0) Xt.from_full(1, V, r, J);
        Mat out(Xt.rank(1), r);
        if (r > 0) out.a = std::move(J);
        return out;
    }
    // ell >= 5: one ell-local edge cell, its cochains = M_N (n even: -I acts trivially, e_c = 1), the cokernel route's carrier
    ENGINE_CHECK_MSG(rows.size() == 1, "RouteBasis: the ell-local tree at ell >= 5 has one cell in degree 1");
    Mat out(D, r);
    out.a = std::move(V);
    return out;
}

ModuleExport ExportInBasis(uint32_t N, uint32_t n, uint32_t ell, const std::vector<uint32_t>& ps, const freez::PackedZ& B, const ExportOptions& eo0) {
    uint32_t K = 0;
    const Mat Bm = RouteBasis(N, n, B, ell, K);
    ExportOptions eo = eo0;
    eo.keep_cocycles = true;
    eo.basis = &Bm;
    eo.K_run = K;
    return ExportModule(N, n, ell, ps, eo);
}

// =================================================================================================================================
// The M oracle and the FNVs of the span convention (fix round 1, audit F2 / F4)
// =================================================================================================================================
bool MixedOracleHolds(const std::vector<uint64_t>& d, uint32_t t0, uint32_t t, uint32_t r, const std::vector<uint64_t>& piTB,
                      const std::vector<uint64_t>& M, const std::vector<uint64_t>& piB, const std::vector<uint64_t>& Fk) {
    ENGINE_CHECK_MSG(piTB.size() == size_t(t) * r && M.size() == size_t(t) * r && piB.size() == size_t(t) * r && Fk.size() == size_t(r) * r &&
                         d.size() >= size_t(t0) + t,
                     "MixedOracleHolds: shapes");
    for (uint32_t i = 0; i < t; ++i) {
        const uint64_t di = d[t0 + i];
        for (uint32_t j = 0; j < r; ++j) {
            unsigned __int128 s = M[size_t(i) * r + j] % di;
            for (uint32_t k = 0; k < r; ++k) s += (unsigned __int128)(piB[size_t(i) * r + k] % di) * (Fk[size_t(k) * r + j] % di);
            if (static_cast<uint64_t>(s % di) != piTB[size_t(i) * r + j] % di) return false;
        }
    }
    return true;
}

namespace {
std::string HexFnv(uint64_t v) {
    static const char* x = "0123456789abcdef";
    std::string s = "0x";
    for (int i = 15; i >= 0; --i) s += x[(v >> (4 * i)) & 15];
    return s;
}
void RowsText(std::ostringstream& o, const std::vector<uint64_t>& A, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) o << (j ? "," : "") << A[size_t(i) * cols + j];
        o << ";";
    }
}
}  // namespace

uint64_t PiBFnv(const ModuleZ& Z, const LocalPartZ& P) {
    ENGINE_CHECK_MSG(Z.ok && Z.free_block == "exact" && P.pi_B.size() == size_t(P.t) * Z.free_rank, "PiBFnv: step 2 with pi_B kept");
    std::ostringstream o;
    o << "pi_B/v1|N=" << Z.N << "|n=" << Z.n << "|ell=" << P.ell << "|basis_fnv=" << HexFnv(Z.global.basis_fnv) << "|d=";
    for (uint32_t i = 0; i < P.t; ++i) o << (i ? "," : "") << Z.d[P.t0 + i];
    o << "|";
    RowsText(o, P.pi_B, P.t, Z.free_rank);
    return engine::FnvBytes(engine::kFnvOffset64, o.str());
}

uint64_t MixedBlockFnv(const ModuleZ& Z, size_t o) {
    ENGINE_CHECK_MSG(Z.ok && Z.free_block == "exact" && Z.M_basis == "global" && o < Z.ps.size(), "MixedBlockFnv: step 2 only");
    std::ostringstream s;
    s << "M_span/v1|N=" << Z.N << "|n=" << Z.n << "|p=" << Z.ps[o] << "|basis_fnv=" << HexFnv(Z.global.basis_fnv) << "|";
    for (const LocalPartZ& P : Z.local) {
        if (P.t == 0) continue;
        ENGINE_CHECK_MSG(o < P.M_global.size(), "MixedBlockFnv: M is missing");
        s << "ell=" << P.ell << "|d=";
        for (uint32_t i = 0; i < P.t; ++i) s << (i ? "," : "") << Z.d[P.t0 + i];
        s << "|";
        RowsText(s, P.M_global[o], P.t, Z.free_rank);
        s << "|";
    }
    return engine::FnvBytes(engine::kFnvOffset64, s.str());
}

// =================================================================================================================================
// The accessors of step 2
// =================================================================================================================================
FreeView ModuleZ::free_view_local(size_t o, size_t l) const {
    ENGINE_CHECK_MSG(ok && l < local.size() && o < ps.size(), "ModuleZ::free_view_local: index");
    FreeView v;
    v.basis_tag = "local";
    v.ell = local[l].ell;
    v.K = local[l].K;
    v.M_block = M_block(o, l);
    v.F_block = F_local(o, l);
    return v;
}
FreeView ModuleZ::free_view(size_t o, size_t l) const {
    if (free_block != "exact" || M_basis != "global") return free_view_local(o, l);
    ENGINE_CHECK_MSG(ok && l < local.size() && o < ps.size(), "ModuleZ::free_view: index");
    const LocalPartZ& P = local[l];
    FreeView v;
    v.basis_tag = "global";
    v.ell = P.ell;
    v.K = P.K;
    if (P.t == 0) {   // no ell-torsion: nothing is read at ell (as F_local / M_block)
        v.M_block.assign(0, 0);
        v.F_block.assign(size_t(free_rank) * free_rank, 0);
        return v;
    }
    ENGINE_CHECK_MSG(o < P.M_global.size(), "ModuleZ::free_view: the global M is missing");
    v.M_block = P.M_global[o];
    v.F_block = free_mod(o, P.ell, P.K);
    return v;
}
std::vector<uint64_t> FreeBlockModQ(const FreeBlockZ& G, size_t o, uint64_t q) {
    ENGINE_CHECK_MSG(o < G.F.size(), "FreeBlockModQ: operator index");
    ENGINE_CHECK_MSG(q >= 2 && q < (uint64_t(1) << 63), "FreeBlockModQ: 2 <= q < 2^63");
    const nf::BigInt QB = nf::BigInt::from_u64(q);
    std::vector<uint64_t> out(G.F[o].size());
    for (size_t e = 0; e < out.size(); ++e) {
        const nf::BigInt m = G.F[o][e].floor_mod(QB);
        const auto& mg = m.magnitude();
        ENGINE_CHECK_MSG(mg.size() <= 2, "FreeBlockModQ: a residue mod q < 2^63 has at most two limbs");
        uint64_t v = 0;
        if (mg.size() > 0) v = mg[0];
        if (mg.size() > 1) v |= uint64_t(mg[1]) << 32;
        ENGINE_CHECK(v < q);
        out[e] = v;
    }
    return out;
}
std::vector<uint64_t> ModuleZ::free_mod(size_t o, uint64_t ell, uint32_t K) const {
    ENGINE_CHECK_MSG(ok && free_block == "exact" && o < global.F.size(), "ModuleZ::free_mod: F over Z is not computed (free_block != exact)");
    ENGINE_CHECK_MSG(ell >= 2 && K >= 1, "ModuleZ::free_mod: ell >= 2, K >= 1");
    uint64_t Q = 1;
    for (uint32_t i = 0; i < K; ++i) {
        ENGINE_CHECK_MSG(Q <= ((1ull << 63) - 1) / ell, "ModuleZ::free_mod: ell^K must be < 2^63");
        Q *= ell;
    }
    return FreeBlockModQ(global, o, Q);
}
std::vector<nf::BigInt> ModuleZ::full_matrix(size_t o) const {
    ENGINE_CHECK_MSG(ok && free_block == "exact" && M_basis == "global" && o < ps.size(), "ModuleZ::full_matrix: step 2 only");
    const uint32_t t = torsion_rank(), r = free_rank, D = t + r;
    std::vector<nf::BigInt> A(size_t(D) * D, nf::BigInt(0));
    for (uint32_t i = 0; i < t; ++i)
        for (uint32_t j = 0; j < t; ++j) A[size_t(i) * D + j] = nf::BigInt::from_u64(tau[o][size_t(i) * t + j]);
    for (const LocalPartZ& P : local) {
        if (P.t == 0) continue;
        for (uint32_t i = 0; i < P.t; ++i)
            for (uint32_t j = 0; j < r; ++j) A[size_t(P.t0 + i) * D + t + j] = nf::BigInt::from_u64(P.M_global[o][size_t(i) * r + j]);
    }
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) A[size_t(t + i) * D + t + j] = global.F[o][size_t(i) * r + j];
    return A;
}

// =================================================================================================================================
// Step 2
// =================================================================================================================================
FreeBlockRun RunFreeBlock(const freez::IsoTreeCarrierZ& car, const std::vector<uint32_t>& ps, const ModuleZ2Options& opt) {
    FreeBlockRun run;
    FreeBlockZ& G = run.G;
    const uint32_t N = car.N(), n = car.n();
    auto stage = [&](const char* what, double s) {
        if (opt.verbose)
            std::cout << "EXZ2_P3_STAGE cell=(" << N << "," << n << ") stage=" << what << " seconds=" << s << " rss_kb=" << VmKb("VmRSS:")
                      << " hwm_kb=" << VmKb("VmHWM:") << std::endl;
    };
    const uint32_t r = car.b1();
    G.basis_rule = freez::kBasisRule;
    if (r == 0) {
        run.FB.B = freez::PackedZ(car.dim(1));
        run.FB.Phi_cert = freez::PackedZ(car.dim(1));
        G.saturation = "rank0";
        G.basis_fnv = freez::PackedFnv(run.FB.B);
        G.phi_fnv = freez::PackedFnv(run.FB.Phi_cert);
        G.q0_key = "-";
        G.bound_basis = "-";
        for (size_t o = 0; o < ps.size(); ++o) {
            G.F.emplace_back();
            G.F_fnv.push_back(freez::MatrixFnv({}, 0));
            G.proof.push_back("rank0");
            G.primes_used.push_back(0);
            G.primes_apriori.push_back(0);
            G.bits_F.push_back(0);
        }
        return run;
    }
    freez::FreeBasisOptions fo;
    fo.pairs.rss_cap_bytes = opt.rss_cap_bytes;
    const auto tb = clk::now();
    run.FB = freez::BuildFreeBasis(car, fo);
    const freez::FreeBasisResult& FB = run.FB;
    G.t_basis = since(tb);
    G.t_pairs = FB.t_pairs;
    G.t_sums = FB.t_sums;
    G.t_sat = FB.t_sat;
    G.t_cert = FB.t_cert;
    stage("basis", G.t_basis);
    if (FB.refused || !FB.cert.ok) {
        run.refusal = FB.refused ? FB.refusal : "REFUSED [Uncertified] the free basis' certificate does not hold";
        return run;
    }
    G.saturation = FB.saturation;
    G.q0_key = freez::Q0Key(FB.pairs);
    G.basis_fnv = FB.fnv_B;
    G.phi_fnv = FB.fnv_Phi;
    G.bits_B = FB.bits_B;
    G.bits_Phi = FB.bits_Phi;
    G.K_pairs = FB.K_used;
    const freez::Pair& P0 = FB.pairs.pairs.at(0);
    freez::FreeBlockInput in;
    in.car = &car;
    in.B = &FB.B;
    in.Phi0 = &P0.Phi;
    in.PhiCert = &FB.Phi_cert;
    in.PriorCert = &FB.cert;   // (Z3a, F10) the certificate runs once; FreeBlockCrt re-runs it iff the fingerprints/carrier key stored in it do not match
    in.R0 = P0.R0;
    in.P = P0.P;
    in.ps = ps;
    freez::FreeBlockOptions bo;
    bo.threads = opt.threads;
    bo.rss_cap_bytes = opt.rss_cap_bytes;
    bo.rss_lanes = opt.rss_lanes;
    bo.bound_with = opt.bound;
    bo.allow_bad_primes = opt.allow_bad_primes;   // (S26 Z1) default false
    // (fix round 1, audit F5) the lanes' route by the existing ENGINE_DEVICE switch (opt-in; bit-identical routes)
    const bool gpu = opt.lanes == LaneChoice::Gpu || (opt.lanes == LaneChoice::FromDevice && engine::cuda::device_enabled());
    bo.route = gpu ? freez::LaneRoute::Gpu : freez::LaneRoute::Cpu;
    G.lane_route = freez::LaneRouteName(bo.route);
    const auto tl = clk::now();
    run.CR = freez::FreeBlockCrt(in, bo);
    const freez::FreeBlockResult& CR = run.CR;
    G.t_crt = since(tl);
    G.t_bound = CR.t_bound;
    G.t_inverse = CR.t_inverse;
    G.t_lanes = CR.t_lanes;
    G.t_lift = CR.t_garner + CR.t_lift;
    G.lanes = static_cast<uint32_t>(CR.primes.size());
    G.bound_basis = CR.bound_basis;
    stage("crt", G.t_crt);
    if (!CR.ok) {
        run.refusal = CR.refusal;
        return run;
    }
    for (const freez::FreeBlockOp& op : CR.ops) {
        ENGINE_CHECK_MSG(op.proven, "RunFreeBlock: an operator of an ok FreeBlockCrt is not proven");
        G.F.push_back(op.F);
        G.F_fnv.push_back(op.fnv);
        G.proof.push_back(op.proof);
        G.primes_used.push_back(op.primes_used);
        G.primes_apriori.push_back(op.primes_apriori);
        G.bits_F.push_back(op.bits_F);
    }
    return run;
}

ModuleZ AssembleModuleZ2(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const FreeBlockRun& run, const ModuleZ2Options& opt,
                         ExportCache* cache) {
    const auto t0 = clk::now();
    auto stage = [&](const char* what, double s) {
        if (opt.verbose)
            std::cout << "EXZ2_P3_STAGE cell=(" << N << "," << n << ") stage=" << what << " seconds=" << s << " rss_kb=" << VmKb("VmRSS:")
                      << " hwm_kb=" << VmKb("VmHWM:") << std::endl;
    };
    const bool have_F = run.refusal.empty();
    const uint32_t r = run.FB.B.cols;
    FreeBlockZ G = run.G;
    // the exports, each ONCE, in the global basis when F exists (the same step-1 fields: ExportOptions::basis adds F_basis only)
    const auto te = clk::now();
    ExportCache local_cache;
    ExportCache& ex = cache ? *cache : local_cache;
    ExportOptions eo = opt.step1.export_opt;
    eo.keep_cocycles = true;
    if (opt.allow_bad_primes) eo.allow_bad_primes = true;   // (S26 Z1) default false: eo unchanged
    for (uint32_t ell : hecke3::RecordTorsionPrimes(N, n)) {
        if (ex.count(ell)) continue;
        if (have_F) ex[ell] = std::make_shared<const ModuleExport>(ExportInBasis(N, n, ell, ps, run.FB.B, eo));
        else ex[ell] = std::make_shared<const ModuleExport>(ExportModule(N, n, ell, ps, eo));
    }
    G.t_exports = since(te);
    stage("exports", G.t_exports);
    ModuleZ Z = ExportModuleZ(N, n, ps, opt.step1, [&ex](uint32_t ell) { return ex.at(ell); });
    if (!Z.ok) {
        Z.seconds = since(t0);
        return Z;   // blocked(export): nothing of step 2 is served
    }
    if (!have_F) {
        Z.free_block = "refused";
        Z.free_refusal = run.refusal;
        Z.seconds = since(t0);
        return Z;
    }
    ENGINE_CHECK_MSG(Z.free_rank == r, "ExportModuleZ2: the exports' rank L != b_1 of the carrier");
    // ---- the change of basis C_ell and the route's own T_p in B, at every torsion prime
    const auto ta = clk::now();
    for (LocalPartZ& P : Z.local) {
        const ModuleExport& E = *P.E;
        ENGINE_CHECK_MSG(E.basis_given && E.basis_ok,
                         ("ExportModuleZ2: the certified Z-basis B is not a basis of L / ell^K at ell = " + std::to_string(P.ell) + ": " + E.basis_why).c_str());
        P.K_C = E.K_max;
        P.C.assign(size_t(r) * r, 0);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) P.C[size_t(i) * r + j] = E.basis_coords.at(i, j);
        P.F_basis.clear();
        for (size_t o = 0; o < ps.size(); ++o) {
            std::vector<uint64_t> Fb(size_t(r) * r);
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) Fb[size_t(i) * r + j] = E.F_basis[o].at(i, j);
            P.F_basis.push_back(std::move(Fb));
        }
    }
    Z.global = G;
    Z.free_block = "exact";
    for (size_t l = 0; l < Z.local.size(); ++l) {
        LocalPartZ& P = Z.local[l];
        // the global oracle: F mod ell^{K_C} == the route's own T_p in B (an independent operator: the transported tree / Heilbronn)
        for (size_t o = 0; o < ps.size(); ++o) {
            const std::vector<uint64_t> Fm = Z.free_mod(o, P.ell, P.K_C);
            ENGINE_CHECK_MSG(Fm == P.F_basis[o], ("ExportModuleZ2: F_" + std::to_string(ps[o]) + " mod " + std::to_string(P.ell) + "^" +
                                                  std::to_string(P.K_C) + " != the export's T_p in B (the global oracle)").c_str());
        }
        if (P.t == 0) continue;
        // M_kerpi = M_local C (row i mod d_i, the block against the ker pi lifts of B); C F == F_local C mod ell^{sigma_max}; then the
        // served block against the classes [b_j] (fix round 1, audit F2): M_span = M_kerpi - pi_B F + tau|_ell pi_B (row a mod d_a)
        const uint64_t QK = PowU(P.ell, P.K);
        const ModuleExport& E = *P.E;
        ENGINE_CHECK_MSG(E.pi_B.r == P.t && E.pi_B.c == r && E.pi_TB.size() == ps.size(),
                         "ExportModuleZ2: the export in B did not keep pi_ell(B) / pi_ell(T_p B)");
        P.pi_B.assign(size_t(P.t) * r, 0);
        for (uint32_t i = 0; i < P.t; ++i)
            for (uint32_t j = 0; j < r; ++j) P.pi_B[size_t(i) * r + j] = E.pi_B.at(i, j);
        P.M_global.clear();
        P.pi_TB.clear();
        for (size_t o = 0; o < ps.size(); ++o) {
            const std::vector<uint64_t> Ml = Z.M_block(o, l), Fl = Z.F_local(o, l), Fk = Z.free_mod(o, P.ell, P.K);
            std::vector<uint64_t> Mg(size_t(P.t) * r, 0), PT(size_t(P.t) * r, 0);
            for (uint32_t i = 0; i < P.t; ++i) {
                const uint64_t di = Z.d[P.t0 + i];
                for (uint32_t j = 0; j < r; ++j) {
                    unsigned __int128 s = 0;   // M_kerpi + tau pi_B, then - pi_B F  (all terms reduced mod d_i)
                    for (uint32_t k = 0; k < r; ++k) s += (unsigned __int128)Ml[size_t(i) * r + k] * (P.C[size_t(k) * r + j] % di);
                    for (uint32_t c = 0; c < P.t; ++c)
                        s += (unsigned __int128)(Z.tau[o][size_t(P.t0 + i) * Z.torsion_rank() + P.t0 + c] % di) * (P.pi_B[size_t(c) * r + j] % di);
                    unsigned __int128 m = 0;
                    for (uint32_t k = 0; k < r; ++k) m += (unsigned __int128)(P.pi_B[size_t(i) * r + k] % di) * (Fk[size_t(k) * r + j] % di);
                    const uint64_t a = static_cast<uint64_t>(s % di), b = static_cast<uint64_t>(m % di);
                    Mg[size_t(i) * r + j] = (a + di - b) % di;
                    PT[size_t(i) * r + j] = E.pi_TB[o].at(i, j);
                }
            }
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) {
                    unsigned __int128 a = 0, b = 0;
                    for (uint32_t k = 0; k < r; ++k) {
                        a += (unsigned __int128)(P.C[size_t(i) * r + k] % QK) * Fk[size_t(k) * r + j];
                        b += (unsigned __int128)Fl[size_t(i) * r + k] * (P.C[size_t(k) * r + j] % QK);
                    }
                    ENGINE_CHECK_MSG(static_cast<uint64_t>(a % QK) == static_cast<uint64_t>(b % QK),
                                     ("ExportModuleZ2: C F != F_local C mod " + std::to_string(P.ell) + "^" + std::to_string(P.K)).c_str());
                }
            // the M oracle (fix round 1, audit F4 (ii)): pi_ell(T_p B) == M_span + pi_B F (row a mod d_a) -- the route's own operator on
            // B and the retraction, no M_local and no C
            ENGINE_CHECK_MSG(MixedOracleHolds(Z.d, P.t0, P.t, r, PT, Mg, P.pi_B, Fk),
                             ("ExportModuleZ2: pi_" + std::to_string(P.ell) + "(T_" + std::to_string(ps[o]) +
                              " B) != M + pi_B F (the M oracle)").c_str());
            P.M_global.push_back(std::move(Mg));
            P.pi_TB.push_back(std::move(PT));
        }
    }
    Z.M_basis = "global";
    Z.entry_encoding = "exact";
    for (LocalPartZ& P : Z.local)
        if (P.t > 0) P.pi_B_fnv = PiBFnv(Z, P);
    Z.M_fnv.clear();
    for (size_t o = 0; o < ps.size(); ++o) Z.M_fnv.push_back(MixedBlockFnv(Z, o));
    Z.global.t_assembly = since(ta);
    stage("assembly", Z.global.t_assembly);
    Z.seconds = since(t0);
    return Z;
}

ModuleZ ExportModuleZ2(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const ModuleZ2Options& opt) {
    const auto t0 = clk::now();
    const auto tc = clk::now();
    const freez::IsoTreeCarrierZ car(N, n);
    const double t_car = since(tc);
    if (opt.verbose)
        std::cout << "EXZ2_P3_STAGE cell=(" << N << "," << n << ") stage=carrier seconds=" << t_car << " rss_kb=" << VmKb("VmRSS:")
                  << " hwm_kb=" << VmKb("VmHWM:") << std::endl;
    FreeBlockRun run = RunFreeBlock(car, ps, opt);
    run.G.t_carrier = t_car;
    ModuleZ2Options o2 = opt;   // (S26 Z1) the step-1 exports with U_p when asked; default: opt unchanged
    if (opt.allow_bad_primes) o2.step1.export_opt.allow_bad_primes = true;
    ModuleZ Z = AssembleModuleZ2(N, n, ps, run, o2);
    Z.seconds = since(t0);
    return Z;
}

}  // namespace modexport
}  // namespace integral_descent
