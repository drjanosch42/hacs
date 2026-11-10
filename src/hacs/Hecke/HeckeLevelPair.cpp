// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 22, LES-1) Level pairs and the fibre maps of the level-change cone -- see HeckeLevelPair.h and
// [C] v1.22.0-pre.3 sec:nebles.
#include "HeckeLevelPair.h"
#include "HeckeGammaH.h"     // (slot 23, G-EVT23) GammaHLevelN for the LES-1 readings
#include "HeckeLevelN.h"     // Gamma0LevelN

#include <algorithm>
#include <unordered_set>

#include "../../ModArithmetic/EngineCheck.h"

namespace hecke {
namespace level {
namespace les {

namespace {

uint64_t ModNonneg(int64_t x, uint64_t m) {
    const int64_t r = x % static_cast<int64_t>(m);
    return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(m) : r);
}

uint64_t Gcd64(uint64_t a, uint64_t b) {
    while (b) { const uint64_t t = a % b; a = b; b = t; }
    return a;
}

Mat22W W(const Mat22& g) { return Mat22W::from(g); }

const Mat22 kMinusI{-1, 0, 0, -1};

// The residues of the exact product X Y (Z), modulo `mod` (< 2^32).
std::array<uint32_t, 4> Residues(const Mat22W& h, uint64_t mod) {
    ENGINE_CHECK_MSG(mod >= 1 && mod < (1ull << 32), "HeckeLevelPair: residue modulus ell * N_chi must be < 2^32");
    return h.reduced_mod(static_cast<uint32_t>(mod));
}

Mat22 FromResidues(const std::array<uint32_t, 4>& r) { return Mat22{r[0], r[1], r[2], r[3]}; }

std::vector<uint32_t> IdentityBlock(uint32_t d, uint32_t p) {
    std::vector<uint32_t> I(static_cast<size_t>(d) * d, 0);
    for (uint32_t r = 0; r < d; ++r) I[static_cast<size_t>(r) * d + r] = 1 % p;
    return I;
}

// C += X Y (d x d, mod p).
void MulAdd(std::vector<uint32_t>& C, const std::vector<uint32_t>& X, const std::vector<uint32_t>& Y, uint32_t d, uint32_t p) {
    if (static_cast<unsigned __int128>(d) * p * p < (static_cast<unsigned __int128>(1) << 63)) {
        // d (p - 1)^2 < 2^63: one reduction per entry.
        for (uint32_t i = 0; i < d; ++i)
            for (uint32_t j = 0; j < d; ++j) {
                uint64_t acc = C[static_cast<size_t>(i) * d + j];
                for (uint32_t k = 0; k < d; ++k)
                    acc += static_cast<uint64_t>(X[static_cast<size_t>(i) * d + k]) * Y[static_cast<size_t>(k) * d + j];
                C[static_cast<size_t>(i) * d + j] = static_cast<uint32_t>(acc % p);
            }
        return;
    }
    for (uint32_t i = 0; i < d; ++i)
        for (uint32_t k = 0; k < d; ++k) {
            const uint64_t x = X[static_cast<size_t>(i) * d + k];
            if (!x) continue;
            for (uint32_t j = 0; j < d; ++j) {
                uint32_t& c = C[static_cast<size_t>(i) * d + j];
                c = static_cast<uint32_t>((c + x * Y[static_cast<size_t>(k) * d + j]) % p);
            }
        }
}

bool IsZero(const std::vector<uint32_t>& b) {
    for (uint32_t x : b) if (x) return false;
    return true;
}

// The Schreier generators w_j s w_{j.s}^{-1} (s = S, T) of A, exactly (Mat22W).
std::vector<Mat22W> SchreierGenerators(const CosetLevel& A, const CosetPermutation& P) {
    std::vector<Mat22W> out;
    out.reserve(2 * static_cast<size_t>(P.m));
    const Mat22 S = Mat22::S(), T = Mat22::T();
    for (uint32_t j = 1; j <= P.m; ++j) {
        out.push_back(W(A.rep(j)) * W(S) * W(A.rep(P.S[j]).inv_unimodular()));
        out.push_back(W(A.rep(j)) * W(T) * W(A.rep(P.T[j]).inv_unimodular()));
    }
    return out;
}

// B <= A by the equivariance of the fibre; fills `fibre` (1-based).
bool FibreOf(const CosetLevel& B, const CosetPermutation& pB, const CosetLevel& A, const CosetPermutation& pA,
             std::vector<uint32_t>& fibre) {
    fibre.assign(static_cast<size_t>(pB.m) + 1, 0);
    for (uint32_t i = 1; i <= pB.m; ++i) fibre[i] = A.class_of(B.rep(i));
    if (fibre[1] != 1) return false;
    for (uint32_t i = 1; i <= pB.m; ++i)
        if (fibre[pB.S[i]] != pA.S[fibre[i]] || fibre[pB.T[i]] != pA.T[fibre[i]]) return false;
    return true;
}

} // namespace

std::vector<uint32_t> RhoBlocks::rho(const Mat22& g) {
    const uint32_t idx = cache_.block_of(g, store_);
    return store_[idx].dense;
}

// ============================================================================
// CongruenceLevel
// ============================================================================
CongruenceLevel::CongruenceLevel(uint64_t M, const std::vector<Mat22>& K_generators, std::string name)
    : M_(M), name_(std::move(name)) {
    ENGINE_CHECK_MSG(M >= 1 && M <= 1000, "CongruenceLevel: M in [1, 1000]");
    auto red = [&](const Mat22& g) {
        return key(ModNonneg(g.a, M), ModNonneg(g.b, M), ModNonneg(g.c, M), ModNonneg(g.d, M));
    };
    // SL_2(Z/M) by BFS on right multiplication by S, T (they generate SL_2(Z), which maps onto SL_2(Z/M)).
    elems_.push_back(red(Mat22::I()));
    lifts_.push_back(Mat22::I());
    index_.emplace(elems_[0], 0);
    const Mat22 gens[2] = {Mat22::S(), Mat22::T()};
    for (size_t q = 0; q < elems_.size(); ++q)
        for (const Mat22& s : gens) {
            const Mat22 h = lifts_[q] * s;
            const uint64_t k = red(h);
            if (index_.count(k)) continue;
            if (elems_.size() >= kMaxGroup)
                throw LevelPairRefusal("les_congruence_too_large", "|SL_2(Z/" + std::to_string(M) + ")| > " + std::to_string(kMaxGroup));
            index_.emplace(k, static_cast<uint32_t>(elems_.size()));
            elems_.push_back(k);
            lifts_.push_back(h);
        }
    auto unpack = [&](uint64_t k, uint64_t e[4]) {
        e[3] = k % M; k /= M; e[2] = k % M; k /= M; e[1] = k % M; k /= M; e[0] = k;
    };
    auto mul = [&](uint64_t x, uint64_t y) {
        uint64_t a[4], b[4];
        unpack(x, a);
        unpack(y, b);
        return key((a[0] * b[0] + a[1] * b[2]) % M, (a[0] * b[1] + a[1] * b[3]) % M, (a[2] * b[0] + a[3] * b[2]) % M,
                   (a[2] * b[1] + a[3] * b[3]) % M);
    };
    // K: the closure of the generators' residues.
    std::vector<uint64_t> kg;
    for (const Mat22& g : K_generators) {
        ENGINE_CHECK_MSG(ModNonneg(g.det(), M) == 1 % M, "CongruenceLevel: a generator of K is not in SL_2(Z/M)");
        kg.push_back(red(g));
    }
    std::vector<uint64_t> K{elems_[0]};
    std::unordered_set<uint64_t> inK{elems_[0]};
    for (size_t q = 0; q < K.size(); ++q)
        for (uint64_t g : kg) {
            const uint64_t y = mul(K[q], g);
            if (inK.insert(y).second) K.push_back(y);
        }
    K_order_ = K.size();
    ENGINE_CHECK_MSG(elems_.size() % K_order_ == 0, "CongruenceLevel: |K| does not divide |SL_2(Z/M)|");
    // Right cosets K g, g in BFS order (so class 1 = K, represented by I).
    cls_.assign(elems_.size(), 0);
    for (size_t q = 0; q < elems_.size(); ++q) {
        if (cls_[q]) continue;
        reps_.push_back(lifts_[q]);
        const uint32_t c = static_cast<uint32_t>(reps_.size());
        for (uint64_t k : K) {
            const auto it = index_.find(mul(k, elems_[q]));
            ENGINE_CHECK_MSG(it != index_.end(), "CongruenceLevel: K g left SL_2(Z/M)");
            ENGINE_CHECK_MSG(cls_[it->second] == 0 || cls_[it->second] == c, "CongruenceLevel: right cosets overlap");
            cls_[it->second] = c;
        }
    }
    ENGINE_CHECK_MSG(static_cast<uint64_t>(reps_.size()) * K_order_ == elems_.size(), "CongruenceLevel: [SL_2 : K] mismatch");
    ENGINE_CHECK_MSG(reps_[0] == Mat22::I(), "CongruenceLevel: w_1 != I");
    for (uint32_t a = 1; a <= m(); ++a) ENGINE_CHECK_MSG(class_of(rep(a)) == a, "CongruenceLevel: class_of(w_a) != a");
}

CongruenceLevel CongruenceLevel::Principal(uint64_t M) {
    return CongruenceLevel(M, {}, "Gamma(" + std::to_string(M) + ")");
}

CongruenceLevel CongruenceLevel::Commutator(uint64_t M) {
    const CongruenceLevel G = Principal(M);
    ENGINE_CHECK_MSG(G.group_order() <= 2000, "CongruenceLevel::Commutator: |SL_2(Z/M)| <= 2000");
    std::vector<Mat22> comm;
    std::unordered_set<uint64_t> seen;
    for (const Mat22& x : G.lifts_)
        for (const Mat22& y : G.lifts_) {
            const Mat22 c = x * y * x.inv_unimodular() * y.inv_unimodular();
            const uint64_t k = G.key(ModNonneg(c.a, M), ModNonneg(c.b, M), ModNonneg(c.c, M), ModNonneg(c.d, M));
            if (seen.insert(k).second) comm.push_back(c);
        }
    std::string nm = M == 2 ? "Gamma^2" : M == 3 ? "Gamma^3" : "Gamma^{[" + std::to_string(M) + "]}";
    return CongruenceLevel(M, comm, nm);
}

Mat22 CongruenceLevel::rep(uint32_t a) const {
    ENGINE_CHECK_MSG(a >= 1 && a <= m(), "CongruenceLevel::rep: index out of range");
    return reps_[a - 1];
}

uint32_t CongruenceLevel::class_of_key(uint64_t k) const {
    const auto it = index_.find(k);
    ENGINE_CHECK_MSG(it != index_.end(), "CongruenceLevel::class_of: not in SL_2(Z/M)");
    return cls_[it->second];
}

uint32_t CongruenceLevel::class_of(const Mat22& g) const {
    ENGINE_CHECK_MSG(g.det() == 1, "CongruenceLevel::class_of: gamma not in SL_2(Z)");
    return class_of_key(key(ModNonneg(g.a, M_), ModNonneg(g.b, M_), ModNonneg(g.c, M_), ModNonneg(g.d, M_)));
}

bool CongruenceLevel::contains(const Mat22& g) const { return g.det() == 1 && class_of(g) == 1; }

bool CongruenceLevel::contains_wide(const Mat22W& g) const {
    if (g.det() != 1) return false;
    const std::array<uint32_t, 4> r = g.reduced_mod(static_cast<uint32_t>(M_));
    return class_of_key(key(r[0], r[1], r[2], r[3])) == 1;
}

// ============================================================================
// the Gamma-set A \ Gamma
// ============================================================================
CosetPermutation PermutationOf(const CosetLevel& A) {
    CosetPermutation P;
    P.m = A.m();
    P.S.assign(static_cast<size_t>(P.m) + 1, 0);
    P.T.assign(static_cast<size_t>(P.m) + 1, 0);
    const Mat22 S = Mat22::S(), T = Mat22::T();
    for (uint32_t i = 1; i <= P.m; ++i) {
        const Mat22 w = A.rep(i);
        P.S[i] = A.class_of(w * S);
        P.T[i] = A.class_of(w * T);
    }
    P.minus_I_in = A.contains(kMinusI);
    return P;
}

TorsionContent TorsionOf(const CosetPermutation& P) {
    TorsionContent t;
    t.minus_I = P.minus_I_in;
    for (uint32_t i = 1; i <= P.m; ++i) {
        if (P.S[i] == i) ++t.fixed_S;
        const uint32_t u = P.U(i);
        if (P.U(u) == i) ++t.fixed_U2;
        if (u == i) t.order6 = true;
    }
    t.order4 = t.fixed_S > 0;
    t.order3 = t.fixed_U2 > 0;
    return t;
}

ModelTag ModelTagOf(const TorsionContent& t, uint64_t ell) {
    // The finite subgroups have orders 1, 2, 3, 4, 6; every even-order one contains -I; an order-6 one an order-3 element.
    if (ell == 2) return t.minus_I ? ModelTag::kBredon : ModelTag::kGroup;
    if (ell == 3) return (t.order3 || t.order6) ? ModelTag::kBredon : ModelTag::kGroup;
    return ModelTag::kGroup;
}

const char* ModelTagName(ModelTag t) { return t == ModelTag::kGroup ? "group" : "bredon"; }

bool GammaSetAutomorphism(const CosetPermutation& P, uint32_t c, std::vector<uint32_t>& phi) {
    ENGINE_CHECK_MSG(c >= 1 && c <= P.m, "GammaSetAutomorphism: class out of range");
    phi.assign(static_cast<size_t>(P.m) + 1, 0);
    std::vector<uint32_t> queue;
    queue.reserve(P.m);
    phi[1] = c;
    queue.push_back(1);
    for (size_t q = 0; q < queue.size(); ++q) {
        const uint32_t x = queue[q];
        const uint32_t px = phi[x];
        const uint32_t ys[2] = {P.S[x], P.T[x]}, zs[2] = {P.S[px], P.T[px]};
        for (int e = 0; e < 2; ++e) {
            if (phi[ys[e]] == 0) {
                phi[ys[e]] = zs[e];
                queue.push_back(ys[e]);
            } else if (phi[ys[e]] != zs[e]) {
                return false;
            }
        }
    }
    ENGINE_CHECK_MSG(queue.size() == P.m, "GammaSetAutomorphism: the S, T Schreier graph is not connected");
    return true;
}

uint32_t QuotientGroup::pos_of(uint32_t cls) const {
    const auto it = std::lower_bound(classes.begin(), classes.end(), cls);
    ENGINE_CHECK_MSG(it != classes.end() && *it == cls, "QuotientGroup::pos_of: not a B-coset inside A");
    return static_cast<uint32_t>(it - classes.begin());
}

uint32_t QuotientGroup::order_of(uint32_t pos) const {
    ENGINE_CHECK_MSG(normal && pos < index, "QuotientGroup::order_of: normal pairs only");
    uint32_t x = pos, t = 1;
    while (x != 0) {
        x = mult[static_cast<size_t>(pos) * index + x];
        ++t;
        ENGINE_CHECK_MSG(t <= index, "QuotientGroup::order_of: the table is not a group");
    }
    return t;
}

bool QuotientGroup::abelian() const {
    ENGINE_CHECK_MSG(normal, "QuotientGroup::abelian: normal pairs only");
    for (uint32_t i = 0; i < index; ++i)
        for (uint32_t j = i + 1; j < index; ++j)
            if (mult[static_cast<size_t>(i) * index + j] != mult[static_cast<size_t>(j) * index + i]) return false;
    return true;
}

bool QuotientGroup::cyclic() const {
    for (uint32_t i = 0; i < index; ++i)
        if (order_of(i) == index) return true;
    return false;
}

QuotientGroup QuotientOf(const CosetLevel& B, const CosetPermutation& pB, const CosetLevel& A, const CosetPermutation& pA) {
    QuotientGroup Q;
    Q.contained = FibreOf(B, pB, A, pA, Q.fibre);
    if (!Q.contained) return Q;
    for (uint32_t i = 1; i <= pB.m; ++i)
        if (Q.fibre[i] == 1) Q.classes.push_back(i);
    Q.index = static_cast<uint32_t>(Q.classes.size());
    ENGINE_CHECK_MSG(static_cast<uint64_t>(Q.index) * pA.m == pB.m, "QuotientOf: [A : B] m_A != m_B");
    ENGINE_CHECK_MSG(Q.classes[0] == 1, "QuotientOf: class 1 is not over A");
    const uint32_t k = Q.index;
    Q.mult.assign(static_cast<size_t>(k) * k, 0);
    for (uint32_t j = 0; j < k; ++j) Q.mult[j] = j;
    std::vector<uint32_t> phi;
    for (uint32_t pos = 1; pos < k; ++pos) {
        if (!GammaSetAutomorphism(pB, Q.classes[pos], phi)) {
            Q.normal = false;
            Q.witness = Q.classes[pos];
            Q.mult.clear();
            return Q;
        }
        for (uint32_t j = 0; j < k; ++j) Q.mult[static_cast<size_t>(pos) * k + j] = Q.pos_of(phi[Q.classes[j]]);
    }
    Q.normal = true;
    return Q;
}

uint64_t IntersectionIndex(const CosetPermutation& pA, const CosetPermutation& pB) {
    auto pack = [&](uint32_t x, uint32_t y) { return static_cast<uint64_t>(x) * (static_cast<uint64_t>(pB.m) + 1) + y; };
    std::unordered_set<uint64_t> seen{pack(1, 1)};
    std::vector<std::pair<uint32_t, uint32_t>> queue{{1, 1}};
    for (size_t q = 0; q < queue.size(); ++q) {
        const auto [x, y] = queue[q];
        const std::pair<uint32_t, uint32_t> nx[2] = {{pA.S[x], pB.S[y]}, {pA.T[x], pB.T[y]}};
        for (const auto& z : nx)
            if (seen.insert(pack(z.first, z.second)).second) queue.push_back(z);
    }
    return queue.size();
}

// ============================================================================
// characters, FibreMatrix
// ============================================================================
CharacterDatum CharacterDatum::Trivial(uint32_t p) {
    CharacterDatum c;
    c.N = 1;
    c.table = {1 % p};
    c.label = "1";
    return c;
}

uint32_t CharacterDatum::value(uint64_t a) const {
    ENGINE_CHECK_MSG(table.size() == N && N >= 1, "CharacterDatum: table size != N");
    const uint32_t v = table[a % N];
    ENGINE_CHECK_MSG(v != 0 && (N == 1 || Gcd64(a % N, N) == 1), "CharacterDatum: a_h is not a unit mod N_chi (h not in Gamma_0(N_chi))");
    return v;
}

FibreMatrix::FibreMatrix(uint32_t m_dst, uint32_t m_src, uint32_t d, uint32_t p)
    : m_dst_(m_dst), m_src_(m_src), d_(d), p_(p), rows_(m_dst) {}

FibreMatrix FibreMatrix::Identity(uint32_t m, uint32_t d, uint32_t p, uint32_t scalar) {
    FibreMatrix I(m, m, d, p);
    const std::vector<uint32_t> blk = IdentityBlock(d, p);
    for (uint32_t i = 1; i <= m; ++i) I.add(i, i, blk, scalar % p);
    I.canonical();
    return I;
}

void FibreMatrix::add(uint32_t i, uint32_t j, const std::vector<uint32_t>& blk, uint32_t s) {
    ENGINE_CHECK_MSG(i >= 1 && i <= m_dst_ && j >= 1 && j <= m_src_ && blk.size() == static_cast<size_t>(d_) * d_,
                     "FibreMatrix::add: block out of range");
    std::vector<uint32_t> b(blk.size());
    for (size_t k = 0; k < blk.size(); ++k) b[k] = static_cast<uint32_t>(static_cast<uint64_t>(blk[k]) * s % p_);
    rows_[i - 1].emplace_back(j, std::move(b));
    canon_ = false;
}

void FibreMatrix::canonical() {
    if (canon_) return;
    canon_ = true;
    for (auto& row : rows_) {
        std::sort(row.begin(), row.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        std::vector<std::pair<uint32_t, std::vector<uint32_t>>> out;
        for (auto& e : row) {
            if (!out.empty() && out.back().first == e.first) {
                for (size_t k = 0; k < e.second.size(); ++k)
                    out.back().second[k] = static_cast<uint32_t>((static_cast<uint64_t>(out.back().second[k]) + e.second[k]) % p_);
            } else {
                out.push_back(std::move(e));
            }
        }
        row.clear();
        for (auto& e : out)
            if (!IsZero(e.second)) row.push_back(std::move(e));
    }
}

size_t FibreMatrix::num_blocks() const {
    size_t s = 0;
    for (const auto& r : rows_) s += r.size();
    return s;
}

bool FibreMatrix::monomial() const {
    for (const auto& r : rows_)
        if (r.size() > 1) return false;
    return true;
}

FibreMatrix FibreMatrix::operator*(const FibreMatrix& rhs) const {
    ENGINE_CHECK_MSG(m_src_ == rhs.m_dst_ && d_ == rhs.d_ && p_ == rhs.p_, "FibreMatrix::operator*: shapes");
    FibreMatrix C(m_dst_, rhs.m_src_, d_, p_);
    const size_t dd = static_cast<size_t>(d_) * d_;
    std::vector<int64_t> slot(static_cast<size_t>(rhs.m_src_) + 1, -1);   // column j -> position in the row being built
    for (uint32_t i = 0; i < m_dst_; ++i) {
        auto& out = C.rows_[i];
        for (const auto& [k, X] : rows_[i])
            for (const auto& [j, Y] : rhs.rows_[k - 1]) {
                if (slot[j] < 0) {
                    slot[j] = static_cast<int64_t>(out.size());
                    out.emplace_back(j, std::vector<uint32_t>(dd, 0));
                }
                MulAdd(out[static_cast<size_t>(slot[j])].second, X, Y, d_, p_);
            }
        for (const auto& e : out) slot[e.first] = -1;
        std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        out.erase(std::remove_if(out.begin(), out.end(), [](const auto& e) { return IsZero(e.second); }), out.end());
    }
    C.canon_ = true;
    return C;
}

FibreMatrix FibreMatrix::operator+(const FibreMatrix& rhs) const {
    ENGINE_CHECK_MSG(m_dst_ == rhs.m_dst_ && m_src_ == rhs.m_src_ && d_ == rhs.d_ && p_ == rhs.p_, "FibreMatrix::operator+: shapes");
    FibreMatrix C = *this;
    for (uint32_t i = 0; i < m_dst_; ++i)
        for (const auto& e : rhs.rows_[i]) C.rows_[i].push_back(e);
    C.canon_ = false;
    C.canonical();
    return C;
}

FibreMatrix FibreMatrix::scaled(uint32_t s) const {
    FibreMatrix C = *this;
    for (auto& r : C.rows_)
        for (auto& e : r)
            for (auto& x : e.second) x = static_cast<uint32_t>(static_cast<uint64_t>(x) * s % p_);
    C.canon_ = false;
    C.canonical();
    return C;
}

uint64_t FibreMatrix::differing_entries(const FibreMatrix& rhs) const {
    if (m_dst_ != rhs.m_dst_ || m_src_ != rhs.m_src_ || d_ != rhs.d_ || p_ != rhs.p_) return UINT64_MAX;
    FibreMatrix Xc, Yc;
    const FibreMatrix* Xp = this;
    const FibreMatrix* Yp = &rhs;
    if (!canon_) { Xc = *this; Xc.canonical(); Xp = &Xc; }
    if (!rhs.canon_) { Yc = rhs; Yc.canonical(); Yp = &Yc; }
    const FibreMatrix& X = *Xp;
    const FibreMatrix& Y = *Yp;
    uint64_t diff = 0;
    auto nnz = [](const std::vector<uint32_t>& b) {
        uint64_t s = 0;
        for (uint32_t x : b) s += x != 0;
        return s;
    };
    for (uint32_t i = 0; i < m_dst_; ++i) {
        const auto& a = X.rows_[i];
        const auto& b = Y.rows_[i];
        size_t u = 0, v = 0;
        while (u < a.size() || v < b.size()) {
            if (v == b.size() || (u < a.size() && a[u].first < b[v].first)) { diff += nnz(a[u].second); ++u; }
            else if (u == a.size() || b[v].first < a[u].first) { diff += nnz(b[v].second); ++v; }
            else {
                for (size_t k = 0; k < a[u].second.size(); ++k) diff += a[u].second[k] != b[v].second[k];
                ++u;
                ++v;
            }
        }
    }
    return diff;
}

std::vector<uint32_t> FibreMatrix::dense() const {
    const size_t R = static_cast<size_t>(m_dst_) * d_, C = static_cast<size_t>(m_src_) * d_;
    std::vector<uint32_t> out(R * C, 0);
    for (uint32_t i = 0; i < m_dst_; ++i)
        for (const auto& [j, blk] : rows_[i])
            for (uint32_t r = 0; r < d_; ++r)
                for (uint32_t c = 0; c < d_; ++c) {
                    uint32_t& o = out[(static_cast<size_t>(i) * d_ + r) * C + static_cast<size_t>(j - 1) * d_ + c];
                    o = static_cast<uint32_t>((static_cast<uint64_t>(o) + blk[static_cast<size_t>(r) * d_ + c]) % p_);
                }
    return out;
}

// ============================================================================
// the twisted monomial action, the restriction and the trace, the conjugation
// ============================================================================
FibreMatrix TwistedMonomialMatrix(const CosetLevel& A, const CharacterDatum& chi, const Mat22& gamma, uint32_t n_sym,
                                  const engine::ModCtx<uint32_t>& ctx, RhoBlocks* blocks) {
    ENGINE_CHECK_MSG(gamma.det() == 1, "TwistedMonomialMatrix: gamma must lie in SL_2(Z)");
    const uint64_t mod = static_cast<uint64_t>(ctx.p) * chi.N;
    RhoBlocks local(ctx, n_sym);
    RhoBlocks& bm = blocks ? *blocks : local;
    ENGINE_CHECK_MSG(bm.p() == ctx.p && bm.n() == n_sym, "TwistedMonomialMatrix: the shared rho cache is for another ring or n");
    FibreMatrix M(A.m(), A.m(), n_sym + 1, ctx.p);
    for (uint32_t j = 1; j <= A.m(); ++j) {
        const Mat22 w = A.rep(j);
        const uint32_t jj = A.class_of(w * gamma);
        const Mat22W h = W(w) * W(gamma) * W(A.rep(jj).inv_unimodular());
        ENGINE_CHECK_MSG(A.contains_wide(h), "TwistedMonomialMatrix: h_j(gamma) is not in A");
        const std::array<uint32_t, 4> r = Residues(h, mod);
        ENGINE_CHECK_MSG(r[2] % chi.N == 0, "TwistedMonomialMatrix: h_j(gamma) not in Gamma_0(N_chi)");
        M.add(j, jj, bm.rho(FromResidues(r)), chi.value(r[0] % chi.N));
    }
    M.canonical();
    return M;
}

RestrictionFibreMap::RestrictionFibreMap(const CosetLevel& A, const CharacterDatum& chiA, const CosetLevel& B,
                                         const CharacterDatum& chiB, uint32_t n_sym, const engine::ModCtx<uint32_t>& ctx,
                                         RestrictionOptions opts, RhoBlocks* blocks) {
    const CosetPermutation pA = PermutationOf(A), pB = PermutationOf(B);
    // def:levelpair: A in Gamma_0(N_A), B in Gamma_0(N_B), on Schreier generators (both are groups).
    const std::vector<Mat22W> genA = SchreierGenerators(A, pA), genB = SchreierGenerators(B, pB);
    const uint64_t L = chiA.N / Gcd64(chiA.N, chiB.N) * chiB.N;
    for (const Mat22W& g : genA)
        if (Residues(g, chiA.N)[2] != 0)
            throw LevelPairRefusal("les_character_level", A.name() + " is not contained in Gamma_0(" + std::to_string(chiA.N) + ")");
    for (const Mat22W& g : genB)
        if (Residues(g, chiB.N)[2] != 0)
            throw LevelPairRefusal("les_character_level", B.name() + " is not contained in Gamma_0(" + std::to_string(chiB.N) + ")");
    if (!FibreOf(B, pB, A, pA, fibre_))
        throw LevelPairRefusal("les_not_contained", B.name() + " is not contained in " + A.name());
    // lem:nebcompat on the Schreier generators of B (both chi(a_.) and chi'(a_.) are homomorphisms on B).
    for (const Mat22W& g : genB) {
        const std::array<uint32_t, 4> r = Residues(g, L);
        ++schreier_;
        if (chiA.value(r[0] % chiA.N) != chiB.value(r[0] % chiB.N))
            throw LevelPairRefusal("les_incompatible_character",
                                   "chi = " + chiA.label + " and chi' = " + chiB.label + " differ on a_gamma, gamma in " + B.name());
    }
    index_ = B.m() / A.m();
    std::vector<uint32_t> count(static_cast<size_t>(A.m()) + 1, 0);
    const uint64_t mod = static_cast<uint64_t>(ctx.p) * chiA.N;
    const uint32_t d = n_sym + 1;
    RhoBlocks local(ctx, n_sym);
    RhoBlocks& bm = blocks ? *blocks : local;
    ENGINE_CHECK_MSG(bm.p() == ctx.p && bm.n() == n_sym, "RestrictionFibreMap: the shared rho cache is for another ring or n");
    const std::vector<uint32_t> Id = IdentityBlock(d, ctx.p);
    alpha_ = FibreMatrix(B.m(), A.m(), d, ctx.p);
    omega_ = FibreMatrix(A.m(), B.m(), d, ctx.p);
    for (uint32_t i = 1; i <= B.m(); ++i) {
        const uint32_t a = fibre_[i];
        ++count[a];
        const Mat22W h = W(B.rep(i)) * W(A.rep(a).inv_unimodular());
        ENGINE_CHECK_MSG(A.contains_wide(h), "RestrictionFibreMap: h_i = v_i w_{a(i)}^{-1} is not in A");
        const std::array<uint32_t, 4> r = Residues(h, mod);
        const Mat22W hi{h.d, -h.b, -h.c, h.a};
        const std::array<uint32_t, 4> ri = Residues(hi, mod);
        const uint32_t s = chiA.value(r[0] % chiA.N), si = chiA.value(ri[0] % chiA.N);   // chi(a_h), chi(a_h)^{-1} = chi(d_h)
        alpha_.add(i, a, opts.neg_identity_block ? Id : bm.rho(FromResidues(r)), opts.neg_inverse_scalar ? si : s);
        omega_.add(a, i, bm.rho(FromResidues(ri)), si);
    }
    for (uint32_t a = 1; a <= A.m(); ++a)
        ENGINE_CHECK_MSG(count[a] == index_, "RestrictionFibreMap: the fibre B \\ Gamma -> A \\ Gamma is not [A : B]-to-one");
    alpha_.canonical();
    omega_.canonical();
}

ConjugationFibreMap::ConjugationFibreMap(const CosetLevel& A, const CharacterDatum& chiA, const CosetLevel& B,
                                         const CosetPermutation& pB, const Mat22& sigma, uint32_t n_sym,
                                         const engine::ModCtx<uint32_t>& ctx, bool skip_normaliser_check, RhoBlocks* blocks) {
    ENGINE_CHECK_MSG(sigma.det() == 1, "ConjugationFibreMap: sigma must lie in SL_2(Z)");
    if (!skip_normaliser_check) {
        if (!A.contains(sigma)) throw LevelPairRefusal("les_not_normal", "sigma is not in " + A.name());
        std::vector<uint32_t> phi;
        if (!GammaSetAutomorphism(pB, B.class_of(sigma), phi))
            throw LevelPairRefusal("les_not_normal", "sigma does not normalise " + B.name());
    }
    const Mat22 sinv = sigma.inv_unimodular();
    const uint64_t mod = static_cast<uint64_t>(ctx.p) * chiA.N;
    RhoBlocks local(ctx, n_sym);
    RhoBlocks& bm = blocks ? *blocks : local;
    ENGINE_CHECK_MSG(bm.p() == ctx.p && bm.n() == n_sym, "ConjugationFibreMap: the shared rho cache is for another ring or n");
    R_ = FibreMatrix(B.m(), B.m(), n_sym + 1, ctx.p);
    for (uint32_t J = 1; J <= B.m(); ++J) {
        const Mat22 vJ = B.rep(J);
        const uint32_t Jp = B.class_of(sinv * vJ);
        const Mat22W g = W(vJ) * W(B.rep(Jp).inv_unimodular());   // = sigma h, h in B
        if (!skip_normaliser_check) ENGINE_CHECK_MSG(A.contains_wide(g), "ConjugationFibreMap: v_J v_{J'}^{-1} is not in A");
        const std::array<uint32_t, 4> r = Residues(g, mod);
        R_.add(J, Jp, bm.rho(FromResidues(r)), skip_normaliser_check && r[2] % chiA.N != 0 ? 1u : chiA.value(r[0] % chiA.N));
    }
    R_.canonical();
}

// ---- (slot 23, G-EVT23) the LES-1 level readings -- moved verbatim from HeckeLevelNormalityValidation.cpp ----------------
LevelHandle MakeLevel(std::unique_ptr<CosetLevel> l) {
    LevelHandle h;
    h.perm = PermutationOf(*l);
    h.lvl = std::move(l);
    return h;
}
LevelHandle Gamma0Handle(uint64_t N) { return MakeLevel(std::make_unique<Gamma0LevelN>(N)); }
LevelHandle GammaHHandle(uint64_t N, const std::function<bool(uint64_t)>& pred) {
    std::vector<uint64_t> elems;
    for (uint64_t u = 1; u < N; ++u)
        if (Gcd64(u, N) == 1 && pred(u)) elems.push_back(u);
    auto G = std::make_unique<GammaHLevelN>(N, elems);
    ENGINE_CHECK_MSG(N == 1 || G->H_elements().size() == elems.size(), "GH: the predicate set is not a subgroup of G_N");
    return MakeLevel(std::move(G));
}
LevelHandle Gamma1Handle(uint64_t N) { return GammaHHandle(N, [N](uint64_t u) { return u % N == 1 % N; }); }
LevelHandle Gamma1Cap0Handle(uint64_t Np, uint64_t L) { return GammaHHandle(L, [Np](uint64_t u) { return u % Np == 1 % Np; }); }

bool QuotientMapIso(const QuotientGroup& Q, const CosetLevel& B, uint64_t M, const std::function<uint64_t(const Mat22&)>& value) {
    if (!Q.normal || Q.index != M) return false;
    std::vector<uint64_t> val(Q.index);
    std::vector<uint8_t> hit(M, 0);
    for (uint32_t pos = 0; pos < Q.index; ++pos) {
        val[pos] = value(B.rep(Q.classes[pos])) % M;
        if (hit[val[pos]]) return false;
        hit[val[pos]] = 1;
    }
    for (uint32_t i = 0; i < Q.index; ++i)
        for (uint32_t j = 0; j < Q.index; ++j)
            if (val[Q.mult[static_cast<size_t>(i) * Q.index + j]] != (val[i] + val[j]) % M) return false;
    return true;
}

std::vector<Gamma0NormalRow> Gamma0NormalityTable() {
    std::map<uint64_t, LevelHandle> cache;
    auto get = [&](uint64_t N) -> const LevelHandle& {
        auto it = cache.find(N);
        if (it == cache.end()) it = cache.emplace(N, Gamma0Handle(N)).first;
        return it->second;
    };
    std::vector<Gamma0NormalRow> rows;
    for (uint64_t N = 1; N <= 60; ++N)
        for (uint64_t M = 2; M <= 24; ++M) {
            const LevelHandle& A = get(N);
            const LevelHandle& B = get(M * N);
            const QuotientGroup Q = QuotientOf(*B, B.perm, *A, A.perm);
            Gamma0NormalRow r;
            r.N = N;
            r.M = M;
            r.contained = Q.contained;
            r.normal = Q.normal;
            if (Q.normal)
                r.iso = Q.cyclic() && QuotientMapIso(Q, *B, M, [N, M](const Mat22& v) {
                            const uint64_t a = ModNonneg(v.a, M), c = ModNonneg(v.c / static_cast<int64_t>(N), M);
                            return a * c % M;
                        });
            rows.push_back(r);
        }
    return rows;
}

std::vector<Gamma1StepRow> Gamma1StepsTable() {
    std::vector<Gamma1StepRow> rows;
    for (uint64_t Np = 2; Np <= 24; ++Np) {
        const LevelHandle A1 = Gamma1Handle(Np);
        for (uint64_t M = 2; M <= Np; ++M) {
            if (Np % M) continue;
            const uint64_t L = M * Np;
            const LevelHandle C = Gamma1Cap0Handle(Np, L);
            const LevelHandle B = Gamma1Handle(L);
            Gamma1StepRow r;
            r.Np = Np;
            r.M = M;
            // (a) Gamma_1(N') cap Gamma_0(MN') normal in Gamma_1(N'), Z/M via c / N' mod M.
            const QuotientGroup Qa = QuotientOf(*C, C.perm, *A1, A1.perm);
            r.a = Qa.contained && Qa.normal && Qa.index == M && Qa.cyclic() &&
                  QuotientMapIso(Qa, *C, M, [Np, M](const Mat22& v) { return ModNonneg(v.c / static_cast<int64_t>(Np), M); });
            // (b) Gamma_1(MN') normal in Gamma_1(N') cap Gamma_0(MN'), Z/M via d = 1 + N' k mod MN' -> k mod M.
            const QuotientGroup Qb = QuotientOf(*B, B.perm, *C, C.perm);
            r.b = Qb.contained && Qb.normal && Qb.index == M && Qb.cyclic() &&
                  QuotientMapIso(Qb, *B, M, [Np, L, M](const Mat22& v) { return ((ModNonneg(v.d, L) + L - 1) % L) / Np % M; });
            // (c) Gamma_1(MN') not normal in Gamma_1(N'), index M^2.
            const QuotientGroup Qc = QuotientOf(*B, B.perm, *A1, A1.perm);
            r.c = Qc.contained && !Qc.normal && Qc.index == M * M;
            rows.push_back(r);
        }
    }
    return rows;
}

std::vector<ModelTagRow> ModelTagTable() {
    std::vector<std::pair<std::string, LevelHandle>> lv;
    lv.emplace_back("Gamma", Gamma0Handle(1));
    lv.emplace_back("Gamma_0(2)", Gamma0Handle(2));
    lv.emplace_back("Gamma_0(4)", Gamma0Handle(4));
    lv.emplace_back("Gamma_0(3)", Gamma0Handle(3));
    lv.emplace_back("Gamma_0(9)", Gamma0Handle(9));
    lv.emplace_back("Gamma_1(4)", Gamma1Handle(4));
    lv.emplace_back("Gamma^2", MakeLevel(std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(2))));
    lv.emplace_back("Gamma^3", MakeLevel(std::make_unique<CongruenceLevel>(CongruenceLevel::Commutator(3))));
    std::vector<ModelTagRow> rows;
    for (const auto& nl : lv) {
        ModelTagRow r;
        r.name = nl.first;
        r.t = TorsionOf(nl.second.perm);
        r.t2 = ModelTagOf(r.t, 2);
        r.t3 = ModelTagOf(r.t, 3);
        r.t5 = ModelTagOf(r.t, 5);
        rows.push_back(r);
    }
    return rows;
}

} // namespace les
} // namespace level
} // namespace hecke
