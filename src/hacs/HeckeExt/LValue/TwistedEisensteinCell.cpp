// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LV2) THE TWISTED EISENSTEIN INSTANCE ([E] cl:lvtwist) on the character model
// H^1(Gamma_0(ell); V_{Z/ell^K}(chi)), chi = omega^j -- see EisensteinBernoulliRoute.h (RunTwistedEisensteinCell).
//
// The model route of slot-18 WP-6 ((W), (GT) of HeckeNebentypusValidation): Z/ell^K = W_K(F_ell) (chi = omega^j takes
// values in mu_{ell-1}, which lies in Z_ell, so no extension is needed); the local system L_{Gamma_0(ell)} with
// eq:MHchi and the Teichmuller table a |-> omega(a)^j mod ell^K (TableScalar), T_p (p != ell) and U_ell by eq:nebcorr
// (BuildLevelCorrespondenceChi + InducedOnH, the pinned r_t of def:cosetrepsN (6)).  iota^* is new: the level-one
// cochain formula (iota^* f)(sigma) = -eps_sigma rho_n(iota) f(iota_K sigma) of [C, def:iota](iv) with rho_n(iota)
// replaced by the action of iota on M = Ind_{Gamma_0(ell)}^{Gamma} V(chi):
//     (iota_M f)_j = chi(a_h) rho_n(iota) rho_n(h) f_{j'},   iota w_j iota = h w_{j'},  h in Gamma_0(ell)
// (iota normalises Gamma_0(ell) and a_{iota h iota} = a_h, so iota_M intertwines M(gamma) with M(iota gamma iota)).
// Every iota^* column is checked to be an equivariant cocycle (HFrame::require_equivariant_cocycles); iota^2 = 1 and
// iota T_p = T_p iota on H follow in FinishEisensteinCell.  Validation / env-hook only: no record, no golden object.
#include "EisensteinBernoulliRoute.h"
#include "Bernoulli.h"

#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeInvolution.h"       // BuildIotaSimplicialMapOnModel
#include "../../Hecke/HeckeLevelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../Hecke/HeckeNebentypus.h"
#include "../../Hecke/HeckeOperatorFamily.h"   // tau_construction_mutex
#include "../../Hecke/HeckeSettings.h"         // ScopedCoefficientRing
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../EquivariantCohomologyOnK.h"
#include "../../LocalSystemSL2Z.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ModCtx.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hecke_ext {
namespace lvalue {

using hecke::Mat22;

namespace {
using u128 = unsigned __int128;

uint64_t Pow32(uint64_t ell, uint32_t K) {
    uint64_t r = 1;
    for (uint32_t i = 0; i < K; ++i) {
        r *= ell;
        ENGINE_CHECK_MSG(r < (uint64_t(1) << 32), "HeckeLValue (twisted): ell^K must stay below 2^32 (the word-sized model route)");
    }
    return r;
}

// chi = omega^j as a complex Dirichlet character mod ell (for the characteristic-0 dimension formula only:
// chi(g^i) = e(j i / (ell - 1)), g the least primitive root -- a Galois conjugate of omega^j, same dimensions).
hecke::neb::DirichletCharacter ComplexOmegaPower(uint64_t ell, uint32_t j) {
    hecke::neb::DirichletCharacter chi;
    chi.N = ell;
    chi.conrey = 0;
    chi.L = ell - 1;
    chi.expo.assign(ell, -1);
    const uint64_t g = PrimitiveRootModPrime(ell);
    uint64_t x = 1;
    for (uint64_t i = 0; i + 1 < ell; ++i) {
        chi.expo[x] = static_cast<int64_t>((static_cast<uint64_t>(j) * i) % (ell - 1));
        x = x * g % ell;
    }
    return chi;
}
} // namespace

EisensteinCell RunTwistedEisensteinCell(uint32_t n, uint64_t ell, uint32_t j, const EisensteinOptions& opt) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(n >= 1, "HeckeLValue (twisted): n >= 1");
    ENGINE_CHECK_MSG(ell >= 5 && PrimitiveRootModPrime(ell) > 0, "HeckeLValue (twisted): ell >= 5 prime");
    ENGINE_CHECK_MSG(j % (ell - 1) != 0, "HeckeLValue (twisted): chi = omega^j must be non-trivial");
    ENGINE_CHECK_MSG((j + n) % 2 == 0, "HeckeLValue (twisted): chi(-1) = (-1)^n (else H^1 = 0, prop:neborbit(2))");
    ENGINE_CHECK_MSG(opt.m_max >= 1, "HeckeLValue (twisted): m_max >= 1");
    const uint32_t k = n + 2;
    EisensteinCell c;
    c.n = n;
    c.ell = ell;
    c.level = ell;
    c.twist_j = j % static_cast<uint32_t>(ell - 1);
    c.K = std::max<uint32_t>(2, opt.m_max + 1);
    const uint32_t K = c.K;
    const uint64_t q = Pow32(ell, K);
    c.primes = opt.primes.empty() ? std::vector<uint64_t>{2, 3, 5, 7, ell} : opt.primes;
    std::sort(c.primes.begin(), c.primes.end());
    c.primes.erase(std::unique(c.primes.begin(), c.primes.end()), c.primes.end());

    std::unique_ptr<SimplicialComplex> Kx;
    std::unique_ptr<hecke::SL2ZTessellationTables> tess;
    {
        std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
        Kx.reset(new SimplicialComplex(CreateSL2ZFundamentalDomainTriangulation()));
        tess.reset(new hecke::SL2ZTessellationTables(*Kx));
    }
    const hecke::level::Gamma0LevelN lvl(ell);
    const uint32_t m = lvl.m(), d = n + 1, D = m * d;
    const hecke::neb::EulerPrediction ep = hecke::neb::EulerCharacteristicFormula(ell, n, ComplexOmegaPower(ell, c.twist_j));
    ENGINE_CHECK_MSG(ep.parity_ok && ep.h1_minus_h0 > 0, "HeckeLValue (twisted): the character model has H^1 = 0");
    const uint32_t r = static_cast<uint32_t>(ep.h1_minus_h0);   // rank over Z_ell (H^0 = 0 in characteristic 0, n >= 1)

    // the free part at finite precision (RunEisensteinCell's note)
    uint32_t K_run = K + opt.min_extra_precision;
    std::unique_ptr<hecke::ScopedCoefficientRing> ring;
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    std::unique_ptr<engine::ModCtx<uint32_t>> ctx;
    std::vector<uint32_t> table;
    std::vector<uint32_t> F;
    for (;;) {
        const uint64_t qr = Pow32(ell, K_run);
        coh.reset();
        lssc.reset();
        ring.reset();
        ring.reset(new hecke::ScopedCoefficientRing(K_run));
        ctx.reset(new engine::ModCtx<uint32_t>(engine::ModCtx<uint32_t>::prime_power(static_cast<uint32_t>(ell), K_run)));
        table.assign(ell, 0);
        for (uint64_t a = 1; a < ell; ++a) table[a] = static_cast<uint32_t>(TeichmullerPowMod(a, c.twist_j, ell, K_run) % qr);
        lssc.reset(new LocalSystemSimplicialComplex(
            *Kx, hecke::neb::InducedRelationsChi(*Kx, lvl, n, *ctx, hecke::neb::TableScalar(ell, table)), D, ctx->p));
        coh.reset(new EquivariantCohomologyOnK(lssc.get()));
        coh->compute_cohomology();
        const hecke::level::HFrame fr(coh.get(), 1);
        c.exps = fr.exps();
        F.clear();
        uint32_t t = 0;
        for (uint32_t i = 0; i < c.exps.size(); ++i) {
            if (c.exps[i] == K_run) F.push_back(i);
            else t = std::max(t, c.exps[i]);
        }
        const bool ok = F.size() == r && K_run >= K + t;
        if (ok && (t == 0 || opt.torsion_aware)) {
            c.torsion_t = t;
            break;
        }
        if (!opt.torsion_aware)
            throw hecke::NotImplementedInThisBuild("HeckeLValue (twisted): H^1(Gamma_0(" + std::to_string(ell) + "); V(omega^" +
                                                   std::to_string(c.twist_j) + ")) has ell-torsion; the torsion-aware reading is opt-in");
        ENGINE_CHECK_MSG(K_run < K + opt.max_extra_precision,
                         "HeckeLValue (twisted): the free part was not isolated within K + max_extra_precision");
        K_run = std::max(K_run + 1, K + t);
    }
    c.K_run = K_run;
    c.torsion_summands = static_cast<uint32_t>(c.exps.size()) - r;
    c.rank = r;
    const uint64_t qr = Pow32(ell, K_run);
    const hecke::level::HFrame frame(coh.get(), 1);
    const uint32_t tt = frame.t();
    const hecke::neb::BlockScalar s = hecke::neb::TableScalar(ell, table);

    auto ToFree = [&](const hecke::level::HMat& A) {
        ENGINE_CHECK_MSG(A.rows == tt && A.cols == tt, "HeckeLValue (twisted): operator shape");
        FreeMat X;
        X.ell = ell;
        X.K = K;
        X.r = r;
        X.a.assign(static_cast<size_t>(r) * r, 0);
        for (uint32_t a = 0; a < r; ++a)
            for (uint32_t b = 0; b < r; ++b) X.a[static_cast<size_t>(a) * r + b] = A.at(F[a], F[b]) % qr % q;
        return X;
    };

    // (a) T_p (p != ell) and U_ell by eq:nebcorr on the character model
    for (uint64_t p : c.primes) {
        std::unique_ptr<hecke::HeckeCosetTables> cosets;
        std::unique_ptr<hecke::HeckeTauTable> tau;
        {
            std::lock_guard<std::mutex> lk(hecke::tau_construction_mutex());
            cosets.reset(new hecke::HeckeCosetTables(p, 1, static_cast<uint32_t>(ell), n));
            tau.reset(new hecke::HeckeTauTable(*Kx, *tess, *cosets, static_cast<uint32_t>(ell), n));
        }
        const hecke::level::Gamma0LevelN lvlNp(ell * p);
        hecke::level::CorrespondenceSpec spec{&lvl, &lvl, cosets.get(), tau.get(), hecke::level::Gamma0SubTransversalN(lvl, lvlNp),
                                              std::string(p == ell ? "U_" : "T_") + std::to_string(p)};
        hecke::level::VerifyTransversal(spec, p == ell ? static_cast<uint32_t>(p) : static_cast<uint32_t>(p + 1));
        c.T.push_back(ToFree(hecke::level::InducedOnH(frame, frame, hecke::neb::BuildLevelCorrespondenceChi(*ctx, spec, *Kx, n, 1, s),
                                                      spec.label)));
    }

    // (b) iota^* on the character model
    {
        const hecke::IotaSimplicialMap im = hecke::BuildIotaSimplicialMapOnModel(*Kx, *tess);
        ENGINE_CHECK_MSG(im.num_degrees() >= 2, "HeckeLValue (twisted): iota_K has no degree 1");
        const uint32_t N1 = static_cast<uint32_t>(im.image_index[1].size());
        ENGINE_CHECK_MSG(frame.flat_dim() == N1 * D, "HeckeLValue (twisted): the frame is not in flat 1-cochain coordinates");
        hecke::level::RhoCache rho(*ctx, n);
        std::vector<hecke::level::Block> store;
        // the induced iota: block j -> (j', chi(a_h) rho(iota) rho(h))
        std::vector<uint32_t> jp(m + 1, 0);
        std::vector<std::vector<uint64_t>> blk(m + 1);
        auto Iota = [](const Mat22& x) { return Mat22{x.a, -x.b, -x.c, x.d}; };
        for (uint32_t jj = 1; jj <= m; ++jj) {
            const Mat22 x = Iota(lvl.rep(jj));
            const uint32_t j2 = lvl.class_of(x);
            const Mat22 h = x * lvl.rep(j2).inv_unimodular();
            ENGINE_CHECK_MSG(lvl.contains(h), "HeckeLValue (twisted): iota w_j iota w_{j'}^{-1} is not in Gamma_0(ell)");
            jp[jj] = j2;
            const std::vector<uint32_t> R = store[rho.block_of(h, store)].dense;
            const std::vector<uint32_t> Rc = store[rho.block_of(Iota(h), store)].dense;
            // rho_n(iota) = diag((-1)^{n-i}); check rho(iota) rho(h) rho(iota) == rho(iota h iota)
            for (uint32_t a = 0; a < d; ++a)
                for (uint32_t b = 0; b < d; ++b) {
                    const bool neg = ((n - a) + (n - b)) % 2u == 1u;
                    const uint64_t v = R[static_cast<size_t>(a) * d + b] % qr;
                    ENGINE_CHECK_MSG((neg ? (qr - v) % qr : v) == Rc[static_cast<size_t>(a) * d + b] % qr,
                                     "HeckeLValue (twisted): rho_n(iota) rho_n(h) rho_n(iota) != rho_n(iota h iota)");
                }
            const uint64_t sc = s(h) % qr;
            blk[jj].assign(static_cast<size_t>(d) * d, 0);
            for (uint32_t a = 0; a < d; ++a)
                for (uint32_t b = 0; b < d; ++b) {
                    const uint64_t v = static_cast<uint64_t>(static_cast<u128>(sc) * (R[static_cast<size_t>(a) * d + b] % qr) % qr);
                    blk[jj][static_cast<size_t>(a) * d + b] = ((n - a) % 2u == 1u) ? (qr - v) % qr : v;
                }
        }
        const std::vector<uint32_t>& B = frame.basis();   // flat_dim x tt, row-major
        std::vector<uint32_t> Y(static_cast<size_t>(frame.flat_dim()) * tt, 0);
        for (uint32_t sg = 0; sg < N1; ++sg) {
            const uint32_t tg = im.image_index[1][sg];
            const bool eps_pos = im.image_sign[1][sg] > 0;   // overall sign -eps
            for (uint32_t jj = 1; jj <= m; ++jj)
                for (uint32_t a = 0; a < d; ++a) {
                    const size_t row = static_cast<size_t>(sg) * D + (jj - 1) * d + a;
                    for (uint32_t col = 0; col < tt; ++col) {
                        u128 acc = 0;
                        for (uint32_t b = 0; b < d; ++b) {
                            const size_t src = static_cast<size_t>(tg) * D + (jp[jj] - 1) * d + b;
                            acc += static_cast<u128>(blk[jj][static_cast<size_t>(a) * d + b]) * B[src * tt + col];
                        }
                        uint64_t v = static_cast<uint64_t>(acc % qr);
                        if (eps_pos) v = (qr - v) % qr;
                        Y[row * tt + col] = static_cast<uint32_t>(v);
                    }
                }
        }
        frame.require_equivariant_cocycles(Y, tt, "iota^* on the character model");
        c.iota = ToFree(frame.coords(Y, tt));
    }

    // (c), (d): e_p = 1 + chi(p) p^{k-1} (E_k(1, chi)), e_ell = 1 (chi(ell) = 0)
    std::vector<uint64_t> e_values;
    for (uint64_t p : c.primes) {
        if (p == ell) {
            e_values.push_back(1);
            continue;
        }
        const uint64_t chip = TeichmullerPowMod(p, c.twist_j, ell, K);
        e_values.push_back((1 + static_cast<uint64_t>(static_cast<u128>(chip) * PowModU64(p, k - 1, q) % q)) % q);
    }
    FinishEisensteinCell(c, e_values, -1, -1);
    c.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return c;
}

} // namespace lvalue
} // namespace hecke_ext
