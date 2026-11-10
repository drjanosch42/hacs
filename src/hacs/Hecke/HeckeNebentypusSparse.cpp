// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 17, WP-6) The sparse cokernel with a character -- see HeckeNebentypusSparse.h
// ([C] sec:nebentypus: prop:neborbit (1), (3), rmk:nebheilbronn, rmk:nebcost (1), (4)).
#include "HeckeNebentypusSparse.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

#include "SparseRref.h"
#include "../../ModArithmetic/EngineCheck.h"

namespace hecke {
namespace neb {

namespace {

double Seconds(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// (S24 CL-G2, B1) F_q as the field policy of the shared sparse kernel (SparseRref.h): the Fq codes are reduced already.
struct FqOps {
    const Fq* F;
    uint32_t add(uint32_t a, uint32_t b) const { return b ? F->add(a, b) : a; }
    uint32_t mul(uint32_t a, uint32_t b) const { return F->mul(a, b); }
    uint32_t muladd(uint32_t a, uint32_t b, uint32_t c) const { return F->add(a, F->mul(b, c)); }
    uint32_t neg(uint32_t a) const { return F->neg(a); }
    uint32_t inv(uint32_t a) const { return F->inv(a); }
    uint32_t reduce(uint32_t a) const { return a; }
    uint32_t one() const { return F->one(); }
};

// s_j B_j v on one block (the TwistedMonomial's block action).
std::vector<uint32_t> BlockMul(const Fq& F, const TwistedMonomial& M, uint32_t j, const std::vector<uint32_t>& v) {
    const uint32_t d = M.d;
    std::vector<uint32_t> w(d, 0);
    const std::vector<uint32_t>& B = M.blocks[M.blk[j]];
    for (uint32_t r = 0; r < d; ++r) {
        uint32_t acc = 0;
        for (uint32_t c = 0; c < d; ++c)
            if (B[static_cast<size_t>(r) * d + c] && v[c]) acc = F.add(acc, F.mul(B[static_cast<size_t>(r) * d + c], v[c]));
        w[r] = F.mul(M.scal[j], acc);
    }
    return w;
}

// The spanning vectors of M^<gamma> orbit by orbit (prop:neborbit (1)), sparse: exactly NebCokernel's
// OrbitFixedVectors (orbit representatives ascending, r = 0 .. d - 1; (1/o) sum_{i<o} M^i e_{j0,r}), zero vectors
// dropped.  Certifies first that gamma^o acts as the identity.
std::vector<SparseVec> OrbitFixedSparse(const TwistedMonomial& M, uint32_t order, const Fq& F, uint32_t& orbits) {
    const uint32_t m = M.m, d = M.d;
    std::vector<uint32_t> inv_perm(m + 1, 0);
    for (uint32_t j = 1; j <= m; ++j) inv_perm[M.perm[j]] = j;
    for (uint32_t j = 1; j <= m; ++j) {
        uint32_t k = j;
        for (uint32_t i = 0; i < order; ++i) k = M.perm[k];
        ENGINE_CHECK_MSG(k == j, "NebSparseCokernel: gamma^o does not fix the coset");
    }
    // gamma^o acts by the identity on every block (the orbit's block product), checked once per orbit start j
    // (each block j is the start of its own product).
    for (uint32_t j = 1; j <= m; ++j) {
        for (uint32_t r = 0; r < d; ++r) {
            std::vector<uint32_t> v(d, 0);
            v[r] = 1;
            uint32_t b = j;
            for (uint32_t i = 0; i < order; ++i) { b = inv_perm[b]; v = BlockMul(F, M, b, v); }
            for (uint32_t c = 0; c < d; ++c)
                ENGINE_CHECK_MSG(v[c] == (c == r ? 1u : 0u),
                                 "NebSparseCokernel: gamma^o acts non-trivially (-I acts by chi(-1)(-1)^n != 1)");
        }
    }
    const uint32_t inv_o = F.inv(F.from_int(order));
    std::vector<uint8_t> seen(m + 1, 0);
    std::vector<SparseVec> out;
    orbits = 0;
    for (uint32_t j0 = 1; j0 <= m; ++j0) {
        if (seen[j0]) continue;
        ++orbits;
        uint32_t k = j0;
        for (uint32_t i = 0; i < order; ++i) { seen[k] = 1; k = M.perm[k]; }
        for (uint32_t r = 0; r < d; ++r) {
            std::vector<std::pair<uint32_t, std::vector<uint32_t>>> acc;   // block -> d-vector
            std::vector<uint32_t> v(d, 0);
            v[r] = 1;
            uint32_t b = j0;
            for (uint32_t i = 0; i < order; ++i) {
                auto it = std::find_if(acc.begin(), acc.end(), [&](const auto& e) { return e.first == b; });
                if (it == acc.end()) { acc.emplace_back(b, std::vector<uint32_t>(d, 0)); it = acc.end() - 1; }
                for (uint32_t c = 0; c < d; ++c) it->second[c] = F.add(it->second[c], F.mul(inv_o, v[c]));
                b = inv_perm[b];
                v = BlockMul(F, M, b, v);
            }
            std::sort(acc.begin(), acc.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
            SparseVec f;
            for (const auto& e : acc)
                for (uint32_t c = 0; c < d; ++c)
                    if (e.second[c]) { f.col.push_back((e.first - 1) * d + c); f.val.push_back(e.second[c]); }
            if (f.nnz()) out.push_back(std::move(f));
        }
    }
    return out;
}

} // namespace

// ---- the sparse rank of orbit-local vectors -----------------------------------------------------------
uint32_t FqSparseRank(const Fq& F, uint32_t D, const std::vector<SparseVec>& vecs) {
    return sparse::EchelonRank(FqOps{&F}, D, vecs);
}

// ---- the sparse RREF over F_q -----------------------------------------------------------------------------
NebSparseCokernel::NebSparseCokernel(const level::Gamma0LevelN& lvl, uint32_t n, const Fq& F, const CharacterValues& chi,
                                     bool keep_generators)
    : F_(&F) {
    if (F.ell() < 5)
        throw NebentypusRefusal("nebentypus_small_ell", "ell in {2, 3}: no cokernel and no comparison with group cohomology");
    ENGINE_CHECK_MSG(chi.N == std::max<uint64_t>(lvl.N(), 1), "NebSparseCokernel: chi has another modulus");
    const uint64_t D64 = static_cast<uint64_t>(lvl.m()) * (static_cast<uint64_t>(n) + 1);
    ENGINE_CHECK_MSG(D64 <= UINT32_MAX / 2, "NebSparseCokernel: D = m_N (n + 1) exceeds the uint32 column-index domain");
    D_ = static_cast<uint32_t>(D64);
    const Fq::E chim1 = lvl.N() <= 2 ? 1u : chi.at(static_cast<int64_t>(lvl.N()) - 1);
    const Fq::E eps = (n % 2 == 0) ? chim1 : F.neg(chim1);
    if (eps != 1) {
        ENGINE_CHECK_MSG(eps == F.neg(1), "NebSparseCokernel: chi(-1)^2 != 1");
        parity_zero_ = true;   // prop:neborbit (2)
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const BlockScalar s = CharacterScalar(chi);
    std::vector<SparseVec> gS, gU;
    {
        const TwistedMonomial MS = TwistedMonomialOf(lvl, Mat22::S(), n, F, s);
        gS = OrbitFixedSparse(MS, 2, F, orbits_S_);
    }
    {
        const TwistedMonomial MU = TwistedMonomialOf(lvl, Mat22::ST(), n, F, s);
        gU = OrbitFixedSparse(MU, 3, F, orbits_U_);
    }
    stats_.generators_s = Seconds(t0);
    const auto t1 = std::chrono::steady_clock::now();
    dim_S_ = FqSparseRank(F, D_, gS);
    dim_U_ = FqSparseRank(F, D_, gU);
    std::vector<SparseVec> all;
    all.reserve(gS.size() + gU.size());
    for (auto& v : gS) all.push_back(std::move(v));
    for (auto& v : gU) all.push_back(std::move(v));
    eliminate(all);
    stats_.eliminate_s = Seconds(t1);
    if (keep_generators) gens_ = std::move(all);
}

void NebSparseCokernel::eliminate(const std::vector<SparseVec>& gens) {
    // the generators as built (S then ST); the result is the unique RREF in any order
    std::vector<uint32_t> seq(gens.size());
    for (uint32_t i = 0; i < seq.size(); ++i) seq[i] = i;
    sparse::Counters st;
    sparse::Eliminate(FqOps{F_}, D_, gens, seq, "NebSparseCokernel", R_, st);
    stats_.generators += st.generators;
    stats_.generator_nnz += st.generator_nnz;
    stats_.dependent += st.dependent;
    stats_.final_nnz += st.final_nnz;
    stats_.peak_nnz = std::max(stats_.peak_nnz, st.peak_nnz);
    stats_.merge_ops += st.merge_ops;
}

std::vector<uint32_t> NebSparseCokernel::coords(const SparseVec& x) const {
    ENGINE_CHECK_MSG(!parity_zero_, "NebSparseCokernel::coords: H^1 = 0 (parity)");
    return sparse::Coords(FqOps{F_}, R_, x, "NebSparseCokernel");
}

std::vector<uint32_t> NebSparseCokernel::coords_dense(const std::vector<uint32_t>& x) const {
    ENGINE_CHECK_MSG(x.size() == D_, "NebSparseCokernel::coords_dense: operand length");
    return coords(sparse::FromDense(FqOps{F_}, x));
}

bool NebSparseCokernel::in_rel(const SparseVec& x) const {
    const std::vector<uint32_t> c = coords(x);
    return std::all_of(c.begin(), c.end(), [](uint32_t v) { return v == 0; });
}

HeilbronnChiSparse::HeilbronnChiSparse(const HeilbronnChi& H, uint32_t D) : H_(&H), D_(D), d_(1) {
    ENGINE_CHECK_MSG(H.D() == D, "HeilbronnChiSparse: fibre ranks differ");
    // d from the blocks (every block is d x d)
    if (!H.blocks().empty()) {
        const size_t sz = H.blocks()[0].size();
        uint32_t d = 1;
        while (static_cast<size_t>(d) * d < sz) ++d;
        ENGINE_CHECK_MSG(static_cast<size_t>(d) * d == sz && D % d == 0, "HeilbronnChiSparse: a block is not d x d");
        d_ = d;
    }
    by_src_.assign(D / d_, {});
    const auto& T = H.terms();
    for (uint32_t i = 0; i < T.size(); ++i) {
        ENGINE_CHECK_MSG(T[i].src >= 1 && T[i].src <= D / d_ && T[i].dst >= 1 && T[i].dst <= D / d_,
                         "HeilbronnChiSparse: a term's block out of range");
        by_src_[T[i].src - 1].push_back(i);
    }
}

SparseVec HeilbronnChiSparse::apply(const Fq& F, const SparseVec& x) const {
    const uint32_t d = d_;
    sparse::Acc<FqOps> acc(FqOps{&F}, D_);
    const auto& T = H_->terms();
    const auto& blocks = H_->blocks();
    size_t k = 0;
    while (k < x.col.size()) {
        const uint32_t b = x.col[k] / d;
        std::vector<uint32_t> xs(d, 0);
        while (k < x.col.size() && x.col[k] / d == b) { xs[x.col[k] % d] = x.val[k]; ++k; }
        for (uint32_t ti : by_src_[b]) {
            const HeilbronnChi::Term& t = T[ti];
            const std::vector<uint32_t>& B = blocks[t.block];
            const uint32_t base = (t.dst - 1) * d;
            for (uint32_t r = 0; r < d; ++r) {
                uint32_t a = 0;
                for (uint32_t c = 0; c < d; ++c)
                    if (B[static_cast<size_t>(r) * d + c] && xs[c]) a = F.add(a, F.mul(B[static_cast<size_t>(r) * d + c], xs[c]));
                if (a) acc.add(base + r, F.mul(t.scal, a));
            }
        }
    }
    return acc.take();
}

uint32_t NebSparseCokernel::RelEscapes(const HeilbronnChi& H) const {
    ENGINE_CHECK_MSG(!parity_zero_ && H.D() == D_, "NebSparseCokernel::RelEscapes: operand");
    ENGINE_CHECK_MSG(!gens_.empty() || dim_rel() == 0, "NebSparseCokernel::RelEscapes: the generators were not kept");
    const HeilbronnChiSparse A(H, D_);
    uint32_t bad = 0;
    for (const SparseVec& g : gens_) bad += in_rel(A.apply(*F_, g)) ? 0 : 1;
    return bad;
}

FqMat NebSparseCokernel::OnH1(const HeilbronnChi& H) const {
    ENGINE_CHECK_MSG(!parity_zero_, "NebSparseCokernel::OnH1: H^1 = 0 (parity)");
    if (!gens_.empty())
        ENGINE_CHECK_MSG(RelEscapes(H) == 0, "NebSparseCokernel::OnH1: H_{N,p,chi}(Rel) is not in Rel (rmk:nebheilbronn)");
    const HeilbronnChiSparse A(H, D_);
    const uint32_t h = h1();
    FqMat M(h, h);
    for (uint32_t j = 0; j < h; ++j) {
        SparseVec e;
        e.col.push_back(R_.free[j]);
        e.val.push_back(1);
        const std::vector<uint32_t> c = coords(A.apply(*F_, e));
        for (uint32_t i = 0; i < h; ++i) M.at(i, j) = c[i];
    }
    return M;
}

// ---- the route ------------------------------------------------------------------------------------------
std::string NebRouteName(NebRoute r) {
    return r == NebRoute::kDense ? "dense" : r == NebRoute::kSparse ? "sparse" : "auto";
}

NebRoute ParseNebRoute(const std::string& s, const std::string& what) {
    if (s == "dense") return NebRoute::kDense;
    if (s == "sparse") return NebRoute::kSparse;
    if (s == "auto") return NebRoute::kAuto;
    throw std::invalid_argument(what + ": unknown route \"" + s + "\" (dense | sparse | auto)");
}

NebRoute ResolveNebRoute(NebRoute r, uint64_t m_N, uint32_t n_sym) {
    if (r != NebRoute::kAuto) return r;
    return m_N * (static_cast<uint64_t>(n_sym) + 1) <= kNebDenseCap ? NebRoute::kDense : NebRoute::kSparse;
}

NebPresentation::NebPresentation(const level::Gamma0LevelN& lvl, uint32_t n, const Fq& F, const CharacterValues& chi,
                                 NebRoute route)
    : route_(ResolveNebRoute(route, lvl.m(), n)) {
    if (route_ == NebRoute::kDense) dense_.reset(new NebCokernel(lvl, n, F, chi));
    else sparse_.reset(new NebSparseCokernel(lvl, n, F, chi));
}

bool NebPresentation::parity_zero() const { return dense_ ? dense_->parity_zero() : sparse_->parity_zero(); }
uint32_t NebPresentation::D() const { return dense_ ? dense_->D() : sparse_->D(); }
uint32_t NebPresentation::h0() const { return dense_ ? dense_->h0() : sparse_->h0(); }
uint32_t NebPresentation::h1() const { return dense_ ? dense_->h1() : sparse_->h1(); }
uint32_t NebPresentation::dim_S() const { return dense_ ? dense_->dim_S() : sparse_->dim_S(); }
uint32_t NebPresentation::dim_U() const { return dense_ ? dense_->dim_U() : sparse_->dim_U(); }
FqMat NebPresentation::OnH1(const HeilbronnChi& H) const { return dense_ ? dense_->OnH1(H) : sparse_->OnH1(H); }

} // namespace neb
} // namespace hecke
