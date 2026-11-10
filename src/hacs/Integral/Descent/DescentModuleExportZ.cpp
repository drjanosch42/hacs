// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZ package EXPORT-Z, step 1 (slot 25): tau and M of H^1 over Z from the per-ell exports.  See DescentModuleExportZ.h.
#include "DescentModuleExportZ.h"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <exception>
#include <sstream>
#include <thread>

#include "DescentHecke.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Fnv.h"

namespace integral_descent {
namespace modexport {

namespace {
uint64_t Pow(uint32_t ell, uint32_t e) {
    uint64_t x = 1;
    for (uint32_t i = 0; i < e; ++i) {
        ENGINE_CHECK_MSG(x < (uint64_t(1) << 32), "ExportModuleZ: ell^e overflow");
        x *= ell;
    }
    return x;
}
}  // namespace

uint64_t RetractionFnv(const ModuleExport& E) {
    ENGINE_CHECK_MSG(E.ok, "RetractionFnv: the export was refused");
    ENGINE_CHECK_MSG(E.retraction.r == E.sigma.size() && E.retraction.c == E.cochain_dim,
                     "RetractionFnv: the export's retraction was not kept");
    std::ostringstream o;
    o << kSplittingV1 << "|N=" << E.N << "|n=" << E.n << "|ell=" << E.ell << "|route=" << E.route << "|K_run=" << E.K_run
      << "|c=" << E.cochain_dim << "|sigma=";
    for (size_t i = 0; i < E.sigma.size(); ++i) o << (i ? "," : "") << E.sigma[i];
    o << "|";
    for (uint32_t a = 0; a < E.retraction.r; ++a) {
        for (uint32_t r = 0; r < E.retraction.c; ++r) o << (r ? "," : "") << E.retraction.at(a, r);
        o << ";";
    }
    return engine::FnvBytes(engine::kFnvOffset64, o.str());
}

uint32_t ModuleZ::largest_torsion_prime() const {
    uint32_t m = 0;
    for (const LocalPartZ& P : local)
        if (P.t > 0) m = std::max(m, P.ell);
    return m;
}

std::vector<uint64_t> ModuleZ::M_block(size_t o, size_t l) const {
    ENGINE_CHECK_MSG(ok && l < local.size() && o < ps.size(), "ModuleZ::M_block: index");
    const LocalPartZ& P = local[l];
    std::vector<uint64_t> B(size_t(P.t) * free_rank, 0);
    if (P.t == 0) return B;
    const uint32_t rk = P.at.rank();
    for (uint32_t i = 0; i < P.t; ++i)
        for (uint32_t j = 0; j < free_rank; ++j) B[size_t(i) * free_rank + j] = P.at.ops[o][size_t(i) * rk + P.t + j];
    return B;
}

std::vector<uint64_t> ModuleZ::F_local(size_t o, size_t l) const {
    ENGINE_CHECK_MSG(ok && l < local.size() && o < ps.size(), "ModuleZ::F_local: index");
    const LocalPartZ& P = local[l];
    std::vector<uint64_t> B(size_t(free_rank) * free_rank, 0);
    if (P.t == 0) return B;
    const uint32_t rk = P.at.rank();
    for (uint32_t i = 0; i < free_rank; ++i)
        for (uint32_t j = 0; j < free_rank; ++j) B[size_t(i) * free_rank + j] = P.at.ops[o][size_t(P.t + i) * rk + P.t + j];
    return B;
}

ModuleZ ExportModuleZ(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const ModuleZOptions& opt, const ExportProvider& provider) {
    const auto t0 = std::chrono::steady_clock::now();
    ModuleZ Z;
    Z.N = N;
    Z.n = n;
    Z.ps = ps;
    Z.torsion_primes = hecke3::RecordTorsionPrimes(N, n);
    ExportOptions eo = opt.export_opt;
    eo.keep_cocycles = true;
    bool have_rank = false;
    for (uint32_t ell : Z.torsion_primes) {
        std::shared_ptr<const ModuleExport> E = provider ? provider(ell) : std::make_shared<const ModuleExport>(ExportModule(N, n, ell, ps, eo));
        ENGINE_CHECK_MSG(E && E->N == N && E->n == n && E->ell == ell && E->ps == ps, "ExportModuleZ: the provider served another export");
        if (!E->ok) {
            Z.ok = false;
            Z.code = E->code;
            Z.refusal = "ell = " + std::to_string(ell) + ": " + E->refusal;
            Z.local.clear();
            Z.d.clear();
            Z.tor_ell.clear();
            Z.tor_sigma.clear();
            Z.tau.clear();
            Z.free_rank = 0;   // audit F1: a blocked value keeps no rank of an earlier, accepted export
            Z.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return Z;
        }
        // every export reads the same L: its free rank is b_1 (EXPORT certifies it against LevelNClosedForm)
        if (!have_rank) Z.free_rank = E->free_rank;
        ENGINE_CHECK_MSG(E->free_rank == Z.free_rank, "ExportModuleZ: the exports disagree on rank L");
        have_rank = true;
        LocalPartZ P;
        P.ell = ell;
        P.E = E;
        P.K = E->sigma_max;
        P.t0 = Z.torsion_rank();
        P.t = static_cast<uint32_t>(E->sigma.size());
        P.retraction_fnv = RetractionFnv(*E);
        if (P.t > 0) {
            ENGINE_CHECK_MSG(P.K >= 1 && P.K <= E->K_max, "ExportModuleZ: sigma_max(ell) <= K_max (EXPORT serves it)");
            P.at = E->at(P.K);
            ENGINE_CHECK(P.at.torsion_count == P.t && P.at.free_rank == Z.free_rank);
            for (uint32_t i = 0; i < P.t; ++i) {
                ENGINE_CHECK(P.at.exps[i] == E->sigma[i]);
                Z.tor_ell.push_back(ell);
                Z.tor_sigma.push_back(E->sigma[i]);
                Z.d.push_back(Pow(ell, E->sigma[i]));
            }
        }
        Z.local.push_back(std::move(P));
    }
    ENGINE_CHECK_MSG(have_rank, "ExportModuleZ: no torsion prime (2 is always one)");
    // tau: block-diagonal
    const uint32_t t = Z.torsion_rank();
    for (size_t o = 0; o < ps.size(); ++o) {
        std::vector<uint64_t> A(size_t(t) * t, 0);
        for (const LocalPartZ& P : Z.local) {
            if (P.t == 0) continue;
            const uint32_t rk = P.at.rank();
            for (uint32_t i = 0; i < P.t; ++i)
                for (uint32_t j = 0; j < P.t; ++j) {
                    const uint64_t v = P.at.ops[o][size_t(i) * rk + j];
                    ENGINE_CHECK(v < Z.d[P.t0 + i]);
                    A[size_t(P.t0 + i) * t + (P.t0 + j)] = v;
                }
        }
        Z.tau.push_back(std::move(A));
    }
    Z.ok = true;
    Z.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return Z;
}

std::vector<BasisBatchEntry> MatricesInBasisBatch(uint32_t N, uint32_t n, const std::vector<uint32_t>& ps, const std::vector<uint32_t>& qs,
                                                  const std::vector<int64_t>& B, uint32_t rows, uint32_t r, uint32_t threads,
                                                  const LaneProbe& lane_probe) {
    ENGINE_CHECK_MSG(B.size() == size_t(rows) * r, "MatricesInBasisBatch: B must be rows x r");
    std::vector<BasisBatchEntry> out(qs.size());
    auto one = [&](size_t s) {
        BasisBatchEntry& o = out[s];
        o.q = qs[s];
        if (lane_probe) lane_probe(o.q);
        if (o.q < 5) {
            o.why = "q = " + std::to_string(o.q) + ": the batch uses the cokernel carrier (q >= 5)";
            return;
        }
        if (N % o.q == 0) {   // audit F9: the documented precondition q !| N, refused by name
            o.why = "q = " + std::to_string(o.q) + " divides N = " + std::to_string(N) + ": the batch serves q !| N only";
            return;
        }
        // q^{K_run} of the cokernel route: the export's own width (RouteMaxK), passed explicitly; B reduced once (audit F9)
        const uint32_t Krun = RouteMaxK(o.q);
        if (Krun == 0) {
            o.why = "q = " + std::to_string(o.q) + ": q >= 2^31 (no word ring)";
            return;
        }
        const uint64_t Q = Pow(o.q, Krun);
        Mat Bq(rows, r);
        for (size_t i = 0; i < B.size(); ++i) {
            const int64_t m = static_cast<int64_t>(Q), v = B[i] % m;
            Bq.a[i] = static_cast<uint32_t>(v < 0 ? v + m : v);
        }
        ExportOptions eo;
        eo.keep_cocycles = false;
        eo.basis = &Bq;
        eo.K_run = Krun;
        ModuleExport E = ExportModule(N, n, o.q, ps, eo);
        ENGINE_CHECK_MSG(!E.ok || (E.route == "cokernel" && E.K_run == Krun), "MatricesInBasisBatch: the export ran at another width");
        o.K = E.K_max;
        if (!E.ok) o.why = E.refusal;
        else if (!E.basis_ok) o.why = E.basis_why;
        else {
            o.ok = true;
            o.F = std::move(E.F_basis);
        }
    };
    // audit F2: a defect (ENGINE_CHECK, bad_alloc) in a lane must not escape a std::thread (std::terminate).  Each lane stores its
    // exception; after the join the first one in qs order is rethrown, so the caller (the validation harness, the console) sees a
    // defect exactly as on the 1-thread path, which runs the same loop.
    std::vector<std::exception_ptr> err(qs.size());
    auto lane = [&](size_t s) {
        try {
            one(s);
        } catch (...) {
            err[s] = std::current_exception();
        }
    };
    const uint32_t T = std::max<uint32_t>(1, std::min<uint32_t>(threads, static_cast<uint32_t>(qs.size())));
    if (T == 1) {
        for (size_t s = 0; s < qs.size(); ++s) lane(s);
    } else {
        std::atomic<size_t> next{0};
        std::vector<std::thread> pool;
        for (uint32_t w = 0; w < T; ++w)
            pool.emplace_back([&] {
                for (size_t s; (s = next.fetch_add(1)) < qs.size();) lane(s);
            });
        for (auto& th : pool) th.join();
    }
    for (const std::exception_ptr& e : err)
        if (e) std::rethrow_exception(e);
    return out;
}

}  // namespace modexport
}  // namespace integral_descent
