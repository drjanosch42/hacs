// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D3, parts (B) and (C): the level-N operator T~^{(N)}_{Iso,p} on the free model ([C] prop:strictliftN(2), (3),
// eq:levelcorr with the bulk strict lift) applied sparsely, and its transport T'_{T,p} = Psi_T^* T~ Phi_T^* to the ell-local tree
// model ([C] prop:strictliftN(5), prop:isomodeltree(4)(d), (6)).  See DescentHecke.h.
#include "DescentHecke.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>

#include "DescentLevelN.h"
#include "../Model/IsotropyModelTree.h"
#include "../Model/StrictLift.h"
#include "../Model/TreeCochains.h"
#include "../../Hecke/ExactInt.h"
#include "../../Hecke/HeckeLevelEll.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/ScalarMod.h"

namespace integral_descent {
namespace hecke3 {

using hecke::Mat22;
using integral_model::GKey;
using integral_model::QGroupRingElt;
using integral_model::Rat;

namespace {
double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
uint64_t InvMod(uint64_t a, uint64_t m) {   // (S24 CL-G3, C-07) ScalarMod.h's extended Euclid
    uint64_t x = 0;
    ENGINE_CHECK_MSG(engine::scalar::try_invmod<uint64_t>(a, m, x), "InvMod: not a unit (a denominator divisible by ell)");
    return x;
}
uint32_t ReduceRat(const U32Ctx& ctx, const Rat& r) {
    const uint64_t m = ctx.p;
    int64_t a = r.num % static_cast<int64_t>(m);
    if (a < 0) a += static_cast<int64_t>(m);
    return static_cast<uint32_t>(static_cast<uint64_t>(a) * InvMod(static_cast<uint64_t>(r.den), m) % m);
}
uint32_t ReduceInt(const U32Ctx& ctx, int64_t v) {
    const int64_t m = static_cast<int64_t>(ctx.p);
    int64_t r = v % m;
    if (r < 0) r += m;
    return static_cast<uint32_t>(r);
}
// y[0..k) += f * x[0..k) mod M
inline void Axpy(uint32_t* y, const uint32_t* x, uint64_t f, uint32_t k, uint64_t M) {
    if (!f) return;
    if ((M & (M - 1)) == 0) {
        // M = 2^K, K <= 31: native 32-bit wrap-around is exact mod 2^32, hence mod M (callers mask once at the end)
        const uint32_t g = static_cast<uint32_t>(f);
        for (uint32_t c = 0; c < k; ++c) y[c] += g * x[c];
        return;
    }
    for (uint32_t c = 0; c < k; ++c)
        if (x[c]) y[c] = static_cast<uint32_t>((y[c] + f * x[c]) % M);
}
inline void MaskPow2(std::vector<uint32_t>& Y, uint64_t M) {
    if ((M & (M - 1)) == 0)
        for (auto& y : Y) y &= static_cast<uint32_t>(M - 1);
}
}  // namespace

// =================================================================================================================================
// (B) IsoHeckeN
// =================================================================================================================================
uint32_t IsoHeckeN::cells(uint32_t q) const { return cells_.at(q); }

const std::vector<IsoHeckeN::TermZ>& IsoHeckeN::terms_Z(uint32_t q) const {
    ENGINE_CHECK_MSG(with_terms_z_, "IsoHeckeN::terms_Z: the integer term list was not requested (construct with with_terms_z = true)");
    return terms_z_.at(q);
}

const char* IsoRouteName(IsoRoute r) {
    switch (r) {
        case IsoRoute::kAuto: return "auto";
        case IsoRoute::kMaterialised: return "materialised";
        case IsoRoute::kCountDense: return "count_dense";
        case IsoRoute::kStreamed: return "streamed";
    }
    return "?";
}

// (S26 Z2) the data of the streamed / count routes: the coset items grouped by destination coset i, the Gamma_0(N) coset data, the lift.
struct IsoHeckeN::Stream {
    struct It {
        uint32_t j, a;
        Mat22 vd;
    };
    hecke::level::Gamma0LevelN A;
    const integral_model::StrictLift* lift = nullptr;
    std::vector<std::vector<It>> by_i;            // [i - 1]
    std::vector<Mat22> w, w_inv;                  // [1 .. m]
    std::vector<std::vector<uint32_t>> count;     // [q] (kCountDense) W x W, built once, from the worker pool of the first apply
    std::vector<std::unique_ptr<std::once_flag>> once;
    explicit Stream(uint32_t N) : A(N) {}
};

IsoHeckeN::IsoHeckeN(uint32_t N, uint32_t n, uint32_t p, const U32Ctx& ctx, uint32_t q_top, bool with_terms_z, bool allow_bad_primes, IsoRoute route)
    : N_(N), n_(n), p_(p), q_top_(q_top), ctx_(ctx), with_terms_z_(with_terms_z) {
    const auto t0 = std::chrono::steady_clock::now();
    auto refuse = [&](D3Refusal c, const std::string& w) {
        code_ = c;
        why_ = "REFUSED [" + std::string(D3RefusalName(c)) + "] T~_" + std::to_string(p) + " on Iso at (N, n) = (" + std::to_string(N) + ", " +
               std::to_string(n) + "): " + w;
    };
    if (p < 2 || !IsPrime(p)) { refuse(D3Refusal::NotPrime, "p is not a prime"); return; }
    if (N % p == 0 && !allow_bad_primes) { refuse(D3Refusal::PDividesN, "p | N: U_p is deferred ([R-Q4])"); return; }
    if (q_top > pair::kPairLiftTop) { refuse(D3Refusal::DegreeRange, "q_top above the lift's degree 6"); return; }
    // the bulk strict lift of prop:strictliftN(2): D4's certified lift, built once per p on {s_j e_c} (independent of r_t, N)
    const pair::PairLiftCost& cost = pair::PairLiftCostOf(p);
    if (!cost.ok) {
        lift_code_ = cost.code;
        refuse(D3Refusal::Lift, std::string("the strict lift is refused [") + pair::PairRefusalName(cost.code) + "]: " + cost.why);
        return;
    }
    const integral_model::StrictLift& lift = pair::PairLift(p);
    const integral_model::LocalModel& L = pair::FreeIsoModel();
    ENGINE_CHECK_MSG(q_top <= lift.q_top() && q_top <= L.q_top(), "IsoHeckeN: q_top beyond the built lift");
    for (uint32_t q = 0; q <= q_top; ++q) {
        const auto& a = lift.model().cells(q);
        const auto& b = L.cells(q);
        ENGINE_CHECK_MSG(a.size() == b.size(), "IsoHeckeN: the cell lists differ");
        for (size_t i = 0; i < a.size(); ++i)
            ENGINE_CHECK_MSG(a[i].s == b[i].s && a[i].rep == b[i].rep && a[i].t == b[i].t, "IsoHeckeN: the cell lists differ");
        cells_.push_back(static_cast<uint32_t>(b.size()));
    }
    const D3Fault fault = ActiveD3Fault();
    try {
        const hecke::level::Gamma0LevelN A(N);
        m_ = A.m();
        d_ = n + 1;
        D_ = m_ * d_;
        const std::vector<Mat22> r = hecke::level::Gamma0SubTransversal(N, p);
        // [A : A''] = p + 1 for p !| N, = p for p | N (S26 Z1; UPZ_design.md Sec. 7.1 (1): A'' = Gamma_0(pN) of index p)
        ENGINE_CHECK_MSG(r.size() == (N % p == 0 ? p : p + 1), "IsoHeckeN: the transversal of Gamma_0(pN) \\ Gamma_0(N) has the wrong size");
        std::vector<Mat22> w(m_ + 1), w_inv(m_ + 1);
        for (uint32_t a = 1; a <= m_; ++a) {
            w[a] = A.rep(a);
            w_inv[a] = w[a].inv_unimodular();
        }
        struct It {
            uint32_t i, j, a;
            Mat22 vd;
        };
        std::vector<It> its;
        for (uint32_t i = 1; i <= m_; ++i)
            for (size_t t = 0; t < r.size(); ++t) {
                if (fault == D3Fault::DropCoset && t + 1 == r.size()) continue;   // VALIDATION ONLY
                const Mat22 x = r[t] * w[i];
                uint32_t j = 0;
                const Mat22 k = lift.twisted_decomposition(x, j);   // x = h s_j, k = g^{-1} h g
                ENGINE_CHECK_MSG(k.det() == 1, "IsoHeckeN: g^{-1} h g is not in Gamma (prop:levelcorrespondence(1))");
                const uint32_t a = A.class_of(k);
                ENGINE_CHECK_MSG(A.contains(k * w_inv[a]), "IsoHeckeN: k w_a^{-1} not in A");
                its.push_back(It{i, j, a, w[i] * lift.coset_delta_matrix(j)});
            }
        // (S26 Z2) the term count of each degree and the route, p | N with allow_bad_primes only (no cost on the shipped p !| N path)
        term_count_.assign(q_top + 1, 0);
        IsoRoute chosen = IsoRoute::kMaterialised;
        if (N % p == 0 && allow_bad_primes && !with_terms_z_ && fault == D3Fault::None) {
            for (uint32_t q = 0; q <= q_top; ++q)
                for (uint32_t c = 0; c < cells_[q]; ++c)
                    for (const It& it : its) {
                        const integral_model::IsoChain& v = lift.value(q, it.j, c);
                        for (uint32_t c2 = 0; c2 < v.c.size(); ++c2) term_count_[q] += v.c[c2].terms.size();
                    }
            chosen = route;
            if (chosen == IsoRoute::kAuto) {
                uint64_t tmax = 0, wmax = 0;
                for (uint32_t q = 0; q <= q_top; ++q) {
                    tmax = std::max<uint64_t>(tmax, term_count_[q]);
                    wmax = std::max<uint64_t>(wmax, uint64_t(cells_[q]) * m_);
                }
                if (n == 0 && wmax * wmax * sizeof(uint32_t) <= kIsoCountDenseBudget) chosen = IsoRoute::kCountDense;
                else if (tmax * sizeof(Term) <= kIsoTermBudget) chosen = IsoRoute::kMaterialised;
                else chosen = IsoRoute::kStreamed;
            }
            ENGINE_CHECK_MSG(chosen != IsoRoute::kCountDense || n == 0, "IsoHeckeN: the count route needs n = 0 (rho_0 = 1)");
        }
        route_ = chosen;
        if (route_ != IsoRoute::kMaterialised) {
            auto st = std::make_shared<Stream>(N);
            st->lift = &lift;
            st->w = w;
            st->w_inv = w_inv;
            st->by_i.resize(m_);
            for (const It& it : its) st->by_i[it.i - 1].push_back(Stream::It{it.j, it.a, it.vd});
            st->count.resize(q_top + 1);
            for (uint32_t q = 0; q <= q_top; ++q) st->once.push_back(std::make_unique<std::once_flag>());
            stream_ = std::move(st);
            build_seconds = since(t0);
            return;
        }
        terms_.resize(q_top + 1);
        if (with_terms_z_) terms_z_.resize(q_top + 1);
        for (uint32_t q = 0; q <= q_top; ++q) {
            auto& Tq = terms_[q];
            std::vector<TermZ>* Zq = with_terms_z_ ? &terms_z_[q] : nullptr;
            for (uint32_t c = 0; c < cells_[q]; ++c)
                for (const It& it : its) {
                    const integral_model::IsoChain& v = lift.value(q, it.j, c);
                    for (uint32_t c2 = 0; c2 < v.c.size(); ++c2)
                        for (const auto& [gk, mul] : v.c[c2].terms) {
                            const Mat22 gam = integral_model::mat_of(gk);
                            const uint32_t a2 = A.class_of(w[it.a] * gam);
                            const Mat22 E = it.vd * gam * w_inv[a2];   // v_i s_j^{-1} g gamma w_{a.gamma}^{-1}
                            ENGINE_CHECK_MSG(E.det() == static_cast<int64_t>(p), "IsoHeckeN: a block matrix has det != p");
                            Tq.push_back(Term{c * m_ + (it.i - 1), c2 * m_ + (a2 - 1), ReduceInt(ctx, mul), E});
                            if (Zq) Zq->push_back(TermZ{c * m_ + (it.i - 1), c2 * m_ + (a2 - 1), E, mul});
                        }
                }
            if (fault == D3Fault::BlockSign)   // VALIDATION ONLY: every second block term (even index) with its multiplicity + 1
                for (size_t t = 0; t < Tq.size(); t += 2) Tq[t].mul = static_cast<uint32_t>((uint64_t(Tq[t].mul) + 1) % ctx.p);
            term_count_[q] = Tq.size();
        }
    } catch (const std::exception& ex) {
        const D3Refusal c = ClassifyD3Exception(ex);
        if (c == D3Refusal::None) throw;
        terms_.clear();
        terms_z_.clear();
        refuse(c, std::string("the operator's assembly stopped: ") + ex.what());
        return;
    }
    build_seconds = since(t0);
}

namespace {
// run f(w, task) for tasks 0 .. n-1 on `threads` workers (the first exception is rethrown); no value may depend on the assignment
template <class F>
void ParallelTasks(uint32_t n, uint32_t threads, const F& f) {
    const uint32_t T = std::max<uint32_t>(1, std::min<uint32_t>(threads, std::max<uint32_t>(n, 1)));
    std::atomic<uint32_t> next(0);
    std::vector<std::exception_ptr> errs(T);
    auto work = [&](uint32_t w) {
        try {
            for (;;) {
                const uint32_t t = next.fetch_add(1);
                if (t >= n) break;
                f(w, t);
            }
        } catch (...) {
            errs[w] = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < T; ++w) {
        try {
            pool.emplace_back(work, w);
        } catch (const std::system_error&) {
            break;
        }
    }
    work(0);
    for (auto& th : pool) th.join();
    for (auto& e : errs)
        if (e) std::rethrow_exception(e);
}
}  // namespace

void IsoHeckeN::apply(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y, uint32_t threads) const {
    ENGINE_CHECK_MSG(!refused(), "IsoHeckeN::apply on a refused operator");
    ENGINE_CHECK_MSG(q <= q_top_ && X.size() == size_t(dim(q)) * k, "IsoHeckeN::apply: shapes");
    if (route_ == IsoRoute::kStreamed) return apply_streamed(q, X, k, Y, threads);
    if (route_ == IsoRoute::kCountDense) return apply_count(q, X, k, Y, threads);
    Y.assign(size_t(dim(q)) * k, 0);
    const uint64_t M = ctx_.p;
    for (const Term& t : terms_[q]) {
        if (!t.mul) continue;
        const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx_, t.E, n_);
        for (uint32_t x = 0; x < d_; ++x) {
            uint32_t* yr = Y.data() + (size_t(t.rb) * d_ + x) * k;
            for (uint32_t y = 0; y < d_; ++y) {
                const uint64_t f = uint64_t(t.mul) * R.at(x, y) % M;
                Axpy(yr, X.data() + (size_t(t.cb) * d_ + y) * k, f, k, M);
            }
        }
    }
    MaskPow2(Y, M);
}

// (S26 Z2) the block terms of row block (c, i - 1), regenerated: as the materialised loop, one coset block at a time.  Row blocks are
// disjoint (rb = c m + i - 1), so distinct (c, i) never write one row: no accumulator, no synchronisation.
void IsoHeckeN::apply_streamed(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y, uint32_t threads) const {
    ENGINE_CHECK_MSG(static_cast<bool>(stream_), "IsoHeckeN::apply_streamed: no stream");
    Y.assign(size_t(dim(q)) * k, 0);
    const uint64_t M = ctx_.p;
    const Stream& S = *stream_;
    const uint32_t C = cells_[q];
    ParallelTasks(C * m_, threads, [&](uint32_t, uint32_t task) {
        const uint32_t c = task / m_, i0 = task % m_;
        uint32_t* yb = Y.data() + size_t(task) * d_ * k;
        for (const Stream::It& it : S.by_i[i0]) {
            const integral_model::IsoChain& v = S.lift->value(q, it.j, c);
            for (uint32_t c2 = 0; c2 < v.c.size(); ++c2)
                for (const auto& [gk, mul] : v.c[c2].terms) {
                    const Mat22 gam = integral_model::mat_of(gk);
                    const uint32_t a2 = S.A.class_of(S.w[it.a] * gam);
                    const uint32_t cb = c2 * m_ + (a2 - 1);
                    const uint32_t m = ReduceInt(ctx_, mul);
                    if (!m) continue;
                    if (n_ == 0) {
                        Axpy(yb, X.data() + size_t(cb) * k, m, k, M);
                        continue;
                    }
                    const Mat22 E = it.vd * gam * S.w_inv[a2];
                    ENGINE_DEBUG_CHECK_MSG(E.det() == static_cast<int64_t>(p_), "IsoHeckeN: a block matrix has det != p");
                    const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx_, E, n_);
                    for (uint32_t x = 0; x < d_; ++x) {
                        uint32_t* yr = yb + size_t(x) * k;
                        for (uint32_t y = 0; y < d_; ++y) {
                            const uint64_t f = uint64_t(m) * R.at(x, y) % M;
                            Axpy(yr, X.data() + (size_t(cb) * d_ + y) * k, f, k, M);
                        }
                    }
                }
        }
    });
    MaskPow2(Y, M);
}

// (S26 Z2) n = 0: rho_0 = 1, the operator is the W x W matrix counting the signed multiplicities; built once per degree from a parallel
// stream (rows are disjoint), then a dense product.  Entries are canonical residues mod ctx.p, so every order of summation is identical.
void IsoHeckeN::apply_count(uint32_t q, const std::vector<uint32_t>& X, uint32_t k, std::vector<uint32_t>& Y, uint32_t threads) const {
    ENGINE_CHECK_MSG(static_cast<bool>(stream_) && n_ == 0 && d_ == 1, "IsoHeckeN::apply_count: the count route needs n = 0");
    Stream& S = *stream_;
    const uint32_t W = cells_[q] * m_;
    const uint64_t M = ctx_.p;
    std::call_once(*S.once[q], [&] {
        std::vector<uint32_t> A(size_t(W) * W, 0);
        ParallelTasks(W, threads, [&](uint32_t, uint32_t task) {
            const uint32_t c = task / m_, i0 = task % m_;
            uint32_t* row = A.data() + size_t(task) * W;
            for (const Stream::It& it : S.by_i[i0]) {
                const integral_model::IsoChain& v = S.lift->value(q, it.j, c);
                for (uint32_t c2 = 0; c2 < v.c.size(); ++c2)
                    for (const auto& [gk, mul] : v.c[c2].terms) {
                        const Mat22 gam = integral_model::mat_of(gk);
                        const uint32_t a2 = S.A.class_of(S.w[it.a] * gam);
                        uint32_t& e = row[c2 * m_ + (a2 - 1)];
                        const uint64_t t = uint64_t(e) + ReduceInt(ctx_, mul);
                        e = static_cast<uint32_t>(t >= M ? t - M : t);
                    }
            }
        });
        S.count[q] = std::move(A);
    });
    const std::vector<uint32_t>& A = S.count[q];
    Y.assign(size_t(dim(q)) * k, 0);
    ParallelTasks(W, threads, [&](uint32_t, uint32_t r) {
        uint32_t* yr = Y.data() + size_t(r) * k;
        const uint32_t* ar = A.data() + size_t(r) * W;
        for (uint32_t c = 0; c < W; ++c)
            if (ar[c]) Axpy(yr, X.data() + size_t(c) * k, ar[c], k, M);
    });
    MaskPow2(Y, M);
}

bool IsoHeckeN::apply_boundary(uint32_t q, const std::vector<int32_t>& bpos, const std::vector<uint32_t>& X, uint32_t k,
                               std::vector<uint32_t>& Y) const {
    ENGINE_CHECK_MSG(route_ == IsoRoute::kMaterialised, "IsoHeckeN::apply_boundary: the materialised route only");
    ENGINE_CHECK_MSG(!refused(), "IsoHeckeN::apply_boundary on a refused operator");
    ENGINE_CHECK_MSG(q <= q_top_ && bpos.size() == cells_.at(q), "IsoHeckeN::apply_boundary: shapes");
    uint32_t nb = 0;
    for (int32_t b : bpos) nb += b >= 0 ? 1u : 0u;
    ENGINE_CHECK_MSG(X.size() == size_t(nb) * D_ * k, "IsoHeckeN::apply_boundary: the input is not on Isob's layout");
    Y.assign(size_t(nb) * D_ * k, 0);
    const uint64_t M = ctx_.p;
    bool closed = true;
    for (const Term& t : terms_[q]) {
        const int32_t br = bpos[t.rb / m_];
        if (br < 0) continue;
        const int32_t bc = bpos[t.cb / m_];
        if (bc < 0) {   // a term of tau_Iso(Isob) off Isob: the pair datum is not strict on this layout
            if (t.mul) closed = false;
            continue;
        }
        if (!t.mul) continue;
        const uint32_t rb = uint32_t(br) * m_ + t.rb % m_, cb = uint32_t(bc) * m_ + t.cb % m_;
        const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx_, t.E, n_);
        for (uint32_t x = 0; x < d_; ++x) {
            uint32_t* yr = Y.data() + (size_t(rb) * d_ + x) * k;
            for (uint32_t y = 0; y < d_; ++y) {
                const uint64_t f = uint64_t(t.mul) * R.at(x, y) % M;
                Axpy(yr, X.data() + (size_t(cb) * d_ + y) * k, f, k, M);
            }
        }
    }
    MaskPow2(Y, M);
    return closed;
}

engine::DenseBuf<uint32_t> IsoHeckeN::dense(uint32_t q) const {
    ENGINE_CHECK_MSG(!refused(), "IsoHeckeN::dense on a refused operator");
    const uint32_t W = dim(q);
    engine::DenseBuf<uint32_t> T(W, W);
    const uint64_t M = ctx_.p;
    if (route_ != IsoRoute::kMaterialised) {   // (S26 Z2) validation only: the columns against the identity
        std::vector<uint32_t> I(size_t(W) * W, 0), Y;
        for (uint32_t i = 0; i < W; ++i) I[size_t(i) * W + i] = 1 % static_cast<uint32_t>(M);
        apply(q, I, W, Y, 1);
        for (uint32_t x = 0; x < W; ++x)
            for (uint32_t y = 0; y < W; ++y) T.at(x, y) = Y[size_t(x) * W + y];
        return T;
    }
    for (const Term& t : terms_[q]) {
        const engine::DenseBuf<uint32_t> R = integral_model::rho_n(ctx_, t.E, n_);
        for (uint32_t x = 0; x < d_; ++x)
            for (uint32_t y = 0; y < d_; ++y) {
                uint32_t& e = T.at(t.rb * d_ + x, t.cb * d_ + y);
                e = static_cast<uint32_t>((e + uint64_t(t.mul) * R.at(x, y)) % M);
            }
    }
    return T;
}

// =================================================================================================================================
// (C) TreeTransport
// =================================================================================================================================
// The comparison of prop:isomodeltree(4) on the code's Iso (Descriptor5's cells), built once over Z (level- and ell-independent), to
// degree 5 (D5c: the tree cusp pair needs Phi_T, Psi_T to the pair's cochain degree q_max + 1 = 5; the maps in degrees <= 4 are those
// of the degree-4 build, the construction being degree by degree).
const integral_model::IntegralTreeComparison& SharedTreeComparison() {
    static std::once_flag once;
    static std::unique_ptr<integral_model::IntegralTreeComparison> cmp;
    std::call_once(once, [] {
        cmp = std::make_unique<integral_model::IntegralTreeComparison>(Descriptor5(), kTreeComparisonTop);
        const integral_model::IsoTreeCertificates c = cmp->certify();
        ENGINE_CHECK_MSG(c.all(), ("TreeTransport: the comparison of prop:isomodeltree(4) is not certified: " + c.first_failure).c_str());
    });
    return *cmp;
}

namespace {
const integral_model::IntegralTreeComparison& TreeCmp() { return SharedTreeComparison(); }
struct EllMaps {
    integral_model::TreeMap phi, psi;
};
const EllMaps& TreeMapsOf(uint32_t ell) {
    static std::mutex mu;
    static std::map<uint32_t, std::unique_ptr<EllMaps>> memo;
    std::lock_guard<std::mutex> lk(mu);
    auto it = memo.find(ell);
    if (it == memo.end()) {
        const auto& cmp = TreeCmp();
        const integral_model::IntegralTreeComparison::EllCheck ck = cmp.check_ell(ell);
        ENGINE_CHECK_MSG(ck.all(), ("TreeTransport: (4)(c), (d) not certified at ell = " + std::to_string(ell) + ": " + ck.first_failure).c_str());
        // build into a local; publish only after success (an exception leaves no half-filled entry)
        auto m = std::make_unique<EllMaps>();
        m->phi = cmp.phi_T(ell);
        m->psi = cmp.psi_T(ell);
        it = memo.emplace(ell, std::move(m)).first;
    }
    return *it->second;
}
}  // namespace

struct TreeTransport::Impl {
    struct Comp {
        std::vector<uint32_t> pos;     // global (cell-local) positions of the component, ascending
        std::vector<uint32_t> Jt;      // J-coordinate indices (into the cell's J order) of the component's J entries
        std::vector<uint32_t> Jl, Rl;  // local positions (into pos) of the J and R entries
        std::vector<uint32_t> B;       // |pos| x |Jl| : P_c[:, J_c]
        std::vector<uint32_t> Ginv;    // |Jl| x |Jl| : (P_c[R_c, J_c])^{-1}
    };
    struct Cell {
        bool all = false;
        uint32_t off = 0;              // J-coordinate offset in the degree
        integral_model::ImageSelection sel;
        std::vector<Comp> comps;
        QGroupRingElt e;
    };
    std::unique_ptr<integral_model::TreeModelChain> L;
    integral_model::TreeModule<uint32_t> TM;
    std::vector<std::vector<Cell>> cells;   // [q][x]
    std::vector<uint32_t> rank;
    uint32_t b0 = 0, b1 = 0;
    const EllMaps* maps = nullptr;
    std::vector<uint32_t> iso_cells;
};

namespace {
uint32_t uf_find(std::vector<uint32_t>& p, uint32_t x) {
    while (p[x] != x) x = p[x] = p[p[x]];
    return x;
}
// inverse of a square matrix invertible mod ell (unit pivots), over Z/ell^k
std::vector<uint32_t> InverseUnitZ(const ZMod& Z, std::vector<uint32_t> G, uint32_t n) {
    std::vector<uint32_t> I(size_t(n) * n, 0);
    for (uint32_t i = 0; i < n; ++i) I[size_t(i) * n + i] = 1;
    for (uint32_t t = 0; t < n; ++t) {
        uint32_t p = t;
        while (p < n && Z.val(G[size_t(p) * n + t]) != 0) ++p;
        ENGINE_CHECK_MSG(p < n, "TreeTransport: the block P[R, J] is not invertible mod ell");
        if (p != t)
            for (uint32_t j = 0; j < n; ++j) {
                std::swap(G[size_t(t) * n + j], G[size_t(p) * n + j]);
                std::swap(I[size_t(t) * n + j], I[size_t(p) * n + j]);
            }
        const uint32_t u = Z.inv(G[size_t(t) * n + t]);
        for (uint32_t j = 0; j < n; ++j) {
            G[size_t(t) * n + j] = Z.mul(G[size_t(t) * n + j], u);
            I[size_t(t) * n + j] = Z.mul(I[size_t(t) * n + j], u);
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (i == t) continue;
            const uint32_t f = G[size_t(i) * n + t];
            if (!f) continue;
            for (uint32_t j = 0; j < n; ++j) {
                G[size_t(i) * n + j] = Z.sub(G[size_t(i) * n + j], Z.mul(f, G[size_t(t) * n + j]));
                I[size_t(i) * n + j] = Z.sub(I[size_t(i) * n + j], Z.mul(f, I[size_t(t) * n + j]));
            }
        }
    }
    return I;
}
// out[block j] += coef * rho(g) in[...]: rho(g) block-monomial, (rho(g) v)_j = B_j v_{col[j]}; k columns, row-major blocks of d rows
void AddRhoApply(const U32Ctx& ctx, const integral_model::BlockMonomial<uint32_t>& B, uint32_t coef, const uint32_t* in, uint32_t* out,
                 uint32_t k) {
    const uint32_t d = B.d;
    const uint64_t M = ctx.p;
    for (uint32_t j = 0; j < B.m; ++j) {
        const std::vector<uint32_t>& blk = B.blocks[B.blk[j]];
        const uint32_t* src = in + size_t(B.col[j]) * d * k;
        uint32_t* dst = out + size_t(j) * d * k;
        for (uint32_t x = 0; x < d; ++x)
            for (uint32_t y = 0; y < d; ++y) {
                const uint64_t f = uint64_t(coef) * blk[size_t(x) * d + y] % M;
                Axpy(dst + size_t(x) * k, src + size_t(y) * k, f, k, M);
            }
    }
}
void AddQRhoApply(const U32Ctx& ctx, const integral_model::TreeModule<uint32_t>& TM, const QGroupRingElt& a, const uint32_t* in,
                  uint32_t* out, uint32_t k) {
    for (const auto& [g, r] : a.terms) {
        const uint32_t coef = ReduceRat(ctx, r);
        if (!coef) continue;
        AddRhoApply(ctx, TM.rho(integral_model::mat_of(g)), coef, in, out, k);
    }
}
}  // namespace

TreeTransport::~TreeTransport() = default;

TreeTransport::TreeTransport(uint32_t ell, uint32_t N, uint32_t n, uint32_t q_top, uint32_t K)
    : ell_(ell), N_(N), n_(n), q_top_(q_top), impl_(std::make_unique<Impl>()) {
    const auto t0 = std::chrono::steady_clock::now();
    ENGINE_CHECK_MSG(ell == 2 || ell == 3, "TreeTransport: ell in {2, 3}");
    ENGINE_CHECK_MSG(q_top <= kTreeComparisonTop, "TreeTransport: q_top <= 5 (the comparison of prop:isomodeltree(4) is built to degree 5)");
    const uint32_t W = K ? K : LocalWidth(ell);
    Z_ = ZMod(ell, W);
    ctx_ = U32Ctx::prime_power(ell, W);
    Impl& I = *impl_;
    I.L = std::make_unique<integral_model::TreeModelChain>(ell, q_top + 1);
    I.TM = integral_model::TreeInducedSymModule(N, n, ctx_);
    D_ = I.TM.dim();
    const LevelNCounts cnt = LevelNClosedForm(N, n);
    I.b0 = n % 2 ? 0u : cnt.b0;
    I.b1 = n % 2 ? 0u : cnt.b1;
    I.maps = &TreeMapsOf(ell);
    const auto& cmp = TreeCmp();
    for (uint32_t q = 0; q <= q_top; ++q) {
        const auto& a = cmp.iso().cells(q);
        const auto& b = pair::FreeIsoModel().cells(q);
        ENGINE_CHECK_MSG(a.size() == b.size(), "TreeTransport: the Iso of (4) and the free model differ");
        for (size_t i = 0; i < a.size(); ++i)
            ENGINE_CHECK_MSG(a[i].s == b[i].s && a[i].rep == b[i].rep && a[i].t == b[i].t, "TreeTransport: the Iso of (4) and the free model differ");
        I.iso_cells.push_back(static_cast<uint32_t>(a.size()));
    }
    const uint32_t m = I.TM.m, d = I.TM.d;
    I.cells.resize(q_top + 2);
    for (uint32_t q = 0; q <= q_top + 1; ++q) {
        uint32_t off = 0;
        for (uint32_t x = 0; x < I.L->rank(q); ++x) {
            Impl::Cell C;
            C.e = I.L->idem(q, x);
            C.off = off;
            C.sel = integral_model::SelectImage(ctx_, I.TM, C.e);
            C.all = C.sel.all;
            if (!C.all) {
                // the components of rho(e)'s block pattern, with P_c, B = P_c[:, J_c], G_c^{-1}
                std::vector<std::pair<uint32_t, const integral_model::BlockMonomial<uint32_t>*>> terms;
                for (const auto& [g, r] : C.e.terms) terms.push_back({ReduceRat(ctx_, r), &I.TM.rho(integral_model::mat_of(g))});
                std::vector<uint32_t> par(m);
                std::iota(par.begin(), par.end(), 0u);
                for (const auto& t : terms)
                    for (uint32_t j = 0; j < m; ++j) {
                        const uint32_t aa = uf_find(par, j), bb = uf_find(par, t.second->col[j]);
                        if (aa != bb) par[std::max(aa, bb)] = std::min(aa, bb);
                    }
                std::map<uint32_t, std::vector<uint32_t>> comps;
                for (uint32_t j = 0; j < m; ++j) comps[uf_find(par, j)].push_back(j);
                for (const auto& kv : comps) {
                    Impl::Comp cp;
                    const std::vector<uint32_t>& blks = kv.second;
                    std::map<uint32_t, uint32_t> where;
                    for (uint32_t aa = 0; aa < blks.size(); ++aa) where[blks[aa]] = aa;
                    for (uint32_t b : blks)
                        for (uint32_t r = 0; r < d; ++r) cp.pos.push_back(b * d + r);
                    const uint32_t Lc = static_cast<uint32_t>(cp.pos.size());
                    std::vector<uint32_t> P(size_t(Lc) * Lc, 0);
                    for (const auto& t : terms) {
                        const auto& B = *t.second;
                        for (uint32_t aa = 0; aa < blks.size(); ++aa) {
                            const uint32_t bb = where.at(B.col[blks[aa]]);
                            const std::vector<uint32_t>& blk = B.blocks[B.blk[blks[aa]]];
                            for (uint32_t r = 0; r < d; ++r)
                                for (uint32_t c = 0; c < d; ++c) {
                                    uint32_t& e = P[size_t(aa * d + r) * Lc + bb * d + c];
                                    e = Z_.add(e, Z_.mul(t.first, blk[size_t(r) * d + c]));
                                }
                        }
                    }
                    for (uint32_t l = 0; l < Lc; ++l) {
                        const int32_t jp = C.sel.jpos[cp.pos[l]], rp = C.sel.rpos[cp.pos[l]];
                        if (jp >= 0) {
                            cp.Jl.push_back(l);
                            cp.Jt.push_back(static_cast<uint32_t>(jp));
                        }
                        if (rp >= 0) cp.Rl.push_back(l);
                    }
                    ENGINE_CHECK_MSG(cp.Jl.size() == cp.Rl.size(), "TreeTransport: |J_c| != |R_c| on a component");
                    const uint32_t r = static_cast<uint32_t>(cp.Jl.size());
                    if (r == 0) continue;
                    cp.B.assign(size_t(Lc) * r, 0);
                    std::vector<uint32_t> G(size_t(r) * r, 0);
                    for (uint32_t l = 0; l < Lc; ++l)
                        for (uint32_t t = 0; t < r; ++t) cp.B[size_t(l) * r + t] = P[size_t(l) * Lc + cp.Jl[t]];
                    for (uint32_t s = 0; s < r; ++s)
                        for (uint32_t t = 0; t < r; ++t) G[size_t(s) * r + t] = P[size_t(cp.Rl[s]) * Lc + cp.Jl[t]];
                    cp.Ginv = InverseUnitZ(Z_, std::move(G), r);
                    C.comps.push_back(std::move(cp));
                }
            }
            off += C.sel.rank();
            I.cells[q].push_back(std::move(C));
        }
        I.rank.push_back(off);
    }
    build_seconds = since(t0);
}

uint32_t TreeTransport::rank(uint32_t q) const { return impl_->rank.at(q); }
uint32_t TreeTransport::full(uint32_t q) const { return impl_->L->rank(q) * D_; }
uint32_t TreeTransport::iso_full(uint32_t q) const { return impl_->iso_cells.at(q) * D_; }

uint32_t TreeTransport::rank_Q(uint32_t q) const {
    // rank d^0 = c_0 - b_0; rank d^q = c_q - b_q - rank d^{q-1}
    uint32_t prev = 0;
    for (uint32_t t = 0; t <= q; ++t) {
        const uint32_t b = t == 0 ? impl_->b0 : (t == 1 ? impl_->b1 : 0u);
        ENGINE_CHECK_MSG(rank(t) >= b + prev, "TreeTransport::rank_Q: Betti numbers too large");
        prev = rank(t) - b - prev;
    }
    return prev;
}

Mat TreeTransport::differential(uint32_t q) const {
    const Impl& I = *impl_;
    ENGINE_CHECK_MSG(q + 1 < I.cells.size(), "TreeTransport::differential: q");
    const uint32_t rows = rank(q + 1), cols = rank(q);
    std::vector<uint32_t> out(size_t(rows) * cols, 0);   // R-coordinates on the target first
    for (const integral_model::LEntry& e : I.L->differential(q + 1)) {
        const Impl::Cell& src = I.cells[q][e.row];
        const Impl::Cell& dst = I.cells[q + 1][e.col];
        if (src.sel.rank() == 0 || dst.sel.rank() == 0) continue;
        integral_model::AddRhoBlock(ctx_, out, cols, dst.off, src.off, I.TM, e.coeff * src.e, dst.sel, src.sel);
    }
    // R -> J on every target cell with a non-trivial selection: the rows of a component through G_c^{-1} (into a fresh store: the R
    // and J index sets of different components overlap)
    std::vector<uint32_t> outJ(size_t(rows) * cols, 0);
    for (const Impl::Cell& C : I.cells[q + 1]) {
        if (C.all) {
            std::copy(out.begin() + size_t(C.off) * cols, out.begin() + size_t(C.off + C.sel.rank()) * cols, outJ.begin() + size_t(C.off) * cols);
            continue;
        }
        for (const Impl::Comp& cp : C.comps) {
            const uint32_t r = static_cast<uint32_t>(cp.Jl.size());
            for (uint32_t t = 0; t < r; ++t) {
                uint32_t* dstrow = outJ.data() + size_t(C.off + cp.Jt[t]) * cols;
                for (uint32_t s = 0; s < r; ++s) {
                    const uint32_t f = cp.Ginv[size_t(t) * r + s];
                    if (!f) continue;
                    const uint32_t rrow = C.off + static_cast<uint32_t>(C.sel.rpos[cp.pos[cp.Rl[s]]]);
                    const uint32_t* src = out.data() + size_t(rrow) * cols;
                    for (uint32_t j = 0; j < cols; ++j)
                        if (src[j]) dstrow[j] = Z_.add(dstrow[j], Z_.mul(f, src[j]));
                }
            }
        }
    }
    out = std::move(outJ);
    Mat M(rows, cols);
    M.a = std::move(out);
    return M;
}

void TreeTransport::to_full(uint32_t q, const std::vector<uint32_t>& U, uint32_t k, std::vector<uint32_t>& V) const {
    const Impl& I = *impl_;
    ENGINE_CHECK_MSG(U.size() == size_t(rank(q)) * k, "TreeTransport::to_full: shape");
    V.assign(size_t(full(q)) * k, 0);
    for (uint32_t x = 0; x < I.cells[q].size(); ++x) {
        const Impl::Cell& C = I.cells[q][x];
        uint32_t* base = V.data() + size_t(x) * D_ * k;
        if (C.all) {
            std::copy(U.begin() + size_t(C.off) * k, U.begin() + size_t(C.off + D_) * k, base);
            continue;
        }
        for (const Impl::Comp& cp : C.comps) {
            const uint32_t r = static_cast<uint32_t>(cp.Jl.size());
            for (uint32_t l = 0; l < cp.pos.size(); ++l) {
                uint32_t* dst = base + size_t(cp.pos[l]) * k;
                for (uint32_t t = 0; t < r; ++t) {
                    const uint32_t f = cp.B[size_t(l) * r + t];
                    if (!f) continue;
                    const uint32_t* src = U.data() + size_t(C.off + cp.Jt[t]) * k;
                    for (uint32_t c = 0; c < k; ++c) dst[c] = Z_.add(dst[c], Z_.mul(f, src[c]));
                }
            }
        }
    }
}

void TreeTransport::from_full(uint32_t q, const std::vector<uint32_t>& V, uint32_t k, std::vector<uint32_t>& U) const {
    const Impl& I = *impl_;
    ENGINE_CHECK_MSG(V.size() == size_t(full(q)) * k, "TreeTransport::from_full: shape");
    U.assign(size_t(rank(q)) * k, 0);
    for (uint32_t x = 0; x < I.cells[q].size(); ++x) {
        const Impl::Cell& C = I.cells[q][x];
        const uint32_t* base = V.data() + size_t(x) * D_ * k;
        if (C.all) {
            std::copy(base, base + size_t(D_) * k, U.begin() + size_t(C.off) * k);
            continue;
        }
        for (const Impl::Comp& cp : C.comps) {
            const uint32_t r = static_cast<uint32_t>(cp.Jl.size());
            for (uint32_t t = 0; t < r; ++t) {
                uint32_t* dst = U.data() + size_t(C.off + cp.Jt[t]) * k;
                for (uint32_t s = 0; s < r; ++s) {
                    const uint32_t f = cp.Ginv[size_t(t) * r + s];
                    if (!f) continue;
                    const uint32_t* src = base + size_t(cp.pos[cp.Rl[s]]) * k;
                    for (uint32_t c = 0; c < k; ++c) dst[c] = Z_.add(dst[c], Z_.mul(f, src[c]));
                }
            }
        }
    }
}

void TreeTransport::phi_star(uint32_t q, const std::vector<uint32_t>& V, uint32_t k, std::vector<uint32_t>& W) const {
    const Impl& I = *impl_;
    ENGINE_CHECK_MSG(V.size() == size_t(full(q)) * k, "TreeTransport::phi_star: shape");
    W.assign(size_t(iso_full(q)) * k, 0);
    const auto& rows = I.maps->phi.value.at(q);   // Iso cell y -> chain on the tree
    ENGINE_CHECK_MSG(rows.size() == I.iso_cells[q], "TreeTransport::phi_star: the map's source cells");
    for (uint32_t y = 0; y < rows.size(); ++y)
        for (const auto& [x, b] : rows[y])
            AddQRhoApply(ctx_, I.TM, b, V.data() + size_t(x) * D_ * k, W.data() + size_t(y) * D_ * k, k);
    MaskPow2(W, ctx_.p);
}

void TreeTransport::psi_star(uint32_t q, const std::vector<uint32_t>& W, uint32_t k, std::vector<uint32_t>& V) const {
    const Impl& I = *impl_;
    ENGINE_CHECK_MSG(W.size() == size_t(iso_full(q)) * k, "TreeTransport::psi_star: shape");
    V.assign(size_t(full(q)) * k, 0);
    const auto& rows = I.maps->psi.value.at(q);   // tree cell x -> chain on Iso
    ENGINE_CHECK_MSG(rows.size() == I.L->rank(q), "TreeTransport::psi_star: the map's source cells");
    const bool drop = ActiveD3Fault() == D3Fault::PsiDropTerm;   // VALIDATION ONLY
    bool dropped = false;
    for (uint32_t x = 0; x < rows.size(); ++x)
        for (const auto& [y, b] : rows[x]) {
            if (drop && !dropped) {
                dropped = true;
                continue;
            }
            AddQRhoApply(ctx_, I.TM, b, W.data() + size_t(y) * D_ * k, V.data() + size_t(x) * D_ * k, k);
        }
    MaskPow2(V, ctx_.p);
}

void TreeTransport::apply_hecke(const IsoHeckeN& T, uint32_t q, const std::vector<uint32_t>& U, uint32_t k,
                                std::vector<uint32_t>& Out, uint32_t threads) const {
    ENGINE_CHECK_MSG(T.D() == D_ && T.q_top() >= q, "TreeTransport::apply_hecke: the operator's module / degrees");
    std::vector<uint32_t> V, W, W2, V2;
    to_full(q, U, k, V);
    phi_star(q, V, k, W);
    V.clear();
    V.shrink_to_fit();
    T.apply(q, W, k, W2, threads);
    W.clear();
    W.shrink_to_fit();
    psi_star(q, W2, k, V2);
    from_full(q, V2, k, Out);
}

uint64_t PredictTreeReadRss(uint32_t ell, uint32_t N, uint32_t n, uint32_t q) {
    const uint64_t D = uint64_t(PsiIndex(N)) * (n + 1);
    // cochain ranks (upper bounds, n even): ell = 2: c_0 = 4D/3, c_q = 7D/3; ell = 3: c_0 = 3D/2, c_1 = 2D, c_q = D
    auto c = [&](uint32_t t) -> uint64_t {
        if (ell == 2) return t == 0 ? (4 * D + 2) / 3 : (7 * D + 2) / 3;
        return t == 0 ? (3 * D + 1) / 2 : (t == 1 ? 2 * D : D);
    };
    const uint64_t cq = c(q), cp = c(q ? q - 1 : 0), cn = c(q + 1);
    // the caller's d^{q-1}, the Smith form's copy of it, L and L^{-1}T (4 B each) + the next differential + the batch through Iso (<= 40
    // cells, 256 columns, three copies)
    const uint64_t smith = 4 * (2 * cq * cp + 2 * cq * cq) + 4 * cn * cq;
    const uint64_t batch = 3ull * 40 * D * 256 * 4;
    return smith + batch + (256ull << 20);
}

}  // namespace hecke3
}  // namespace integral_descent
