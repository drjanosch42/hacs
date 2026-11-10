// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EXPORT-Z step 2, package EXZ2-P2: RNS lanes, the Sec. 1.6 bounds, the proven CRT and the BigInt output of the free block F_p
// (DescentFreeBlockCrt.h; docs/implementation/slot25/EXPORTZ2_design.md v2 Sec. 1.5, 1.6, 2; EXZ2P2_report.md).

#include "DescentFreeBlockCrt.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "DescentFreeBasisZ.h"
#include "DescentHecke.h"
#include "../Model/IsotropyModel.h"
#include "../Model/IsotropyModelTree.h"
#include "../../Hecke/HeckeLevelN.h"
#include "../../../Cuda/CudaCtx.h"
#include "../../../Cuda/CudaKernels.h"
#include "../../../Cuda/DeviceBuf.h"
#include "../../../Engine/Threads.h"
#include "../../../ModArithmetic/DenseBuf.h"
#include "../../../ModArithmetic/DenseView.h"
#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Kernels.h"
#include "../../../ModArithmetic/RnsKernels.h"

namespace integral_descent {
namespace freez {

using hecke::Mat22;
using clk = std::chrono::steady_clock;
using U32Buf = engine::DenseBuf<uint32_t>;

namespace {
double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
uint32_t RedI64(int64_t v, uint32_t q) {
    int64_t r = v % static_cast<int64_t>(q);
    if (r < 0) r += q;
    return static_cast<uint32_t>(r);
}
uint32_t PowModU(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1) r = r * a % m;
        a = a * a % m;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}
std::vector<std::vector<BigInt>> ColsOfRowMajor(const std::vector<BigInt>& F, uint32_t r) {
    std::vector<std::vector<BigInt>> c(r, std::vector<BigInt>(r));
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) c[j][i] = F[size_t(i) * r + j];
    return c;
}
}  // namespace

// =================================================================================================================================
// SparseColsZ
// =================================================================================================================================
uint64_t SparseColsZ::nnz() const {
    uint64_t s = 0;
    for (const auto& c : cols) s += c.size();
    return s;
}
uint32_t SparseColsZ::max_bits() const {
    uint32_t b = 0;
    for (const auto& c : cols)
        for (const auto& e : c) b = std::max(b, BitsOf(e.second));
    return b;
}
std::vector<std::vector<BigInt>> SparseColsZ::dense() const {
    std::vector<std::vector<BigInt>> out(cols.size(), std::vector<BigInt>(rows, BigInt(0)));
    for (size_t j = 0; j < cols.size(); ++j)
        for (const auto& [i, v] : cols[j]) out[j][i] = v;
    return out;
}
uint64_t SparseColsZ::fnv() const { return MatrixFnv(dense(), rows); }
SparseColsZ SparseColsZ::from_dense(const std::vector<std::vector<BigInt>>& c, uint32_t rows) {
    SparseColsZ S;
    S.rows = rows;
    S.cols.resize(c.size());
    for (size_t j = 0; j < c.size(); ++j) {
        ENGINE_CHECK_MSG(c[j].size() == rows, "SparseColsZ::from_dense: column length");
        for (uint32_t i = 0; i < rows; ++i)
            if (!c[j][i].is_zero()) S.cols[j].push_back({i, c[j][i]});
    }
    return S;
}
PackedZ SparseColsZ::to_packed() const {
    PackedZ A(rows);
    for (const auto& c : cols) A.append_col(c);
    return A;
}
SparseColsZ SparseColsZ::from_packed(const PackedZ& A) {
    SparseColsZ S;
    S.rows = A.rows;
    S.cols.resize(A.cols);
    for (uint32_t j = 0; j < A.cols; ++j) S.cols[j] = A.col_sparse(j);
    return S;
}

// =================================================================================================================================
// UpReal: m * 2^e, every operation rounded UP (a sound majorant arithmetic; no floating point)
// =================================================================================================================================
namespace {
inline int Clz64(uint64_t x) { return __builtin_clzll(x); }
UpReal Norm(uint64_t m, int64_t e, bool sticky) {
    // value >= m * 2^e (+ a positive remainder when sticky); returns a normalised upward representation
    UpReal r;
    if (m == 0) {
        if (!sticky) return r;
        m = 1;
    } else if (sticky) {
        if (m == ~0ull) {
            m = 1ull << 63;
            ++e;
            r.m = m;
            r.e = static_cast<int32_t>(e);
            return r;
        }
        ++m;
    }
    const int s = Clz64(m);
    // shifting left is exact
    r.m = m << s;
    r.e = static_cast<int32_t>(e - s);
    return r;
}
}  // namespace

UpReal UpReal::of_u64(uint64_t v) { return Norm(v, 0, false); }

UpReal UpReal::of_big(const BigInt& x) {
    const auto& mg = x.magnitude();
    size_t L = mg.size();
    while (L > 0 && mg[L - 1] == 0) --L;
    if (L == 0) return UpReal{};
    if (L <= 2) {
        uint64_t v = mg[0];
        if (L == 2) v |= uint64_t(mg[1]) << 32;
        return of_u64(v);
    }
    // top 64 bits of the top three limbs, sticky for the rest
    const uint64_t hi = (uint64_t(mg[L - 1]) << 32) | mg[L - 2];
    const uint32_t lo = mg[L - 3];
    const int s = Clz64(hi);   // hi != 0
    uint64_t m = s ? (hi << s) | (uint64_t(lo) >> (32 - s)) : hi;
    bool sticky = s ? (uint64_t(lo) << (32 + s)) != 0 : lo != 0;   // the bits of lo below the window
    for (size_t i = 0; i + 3 < L && !sticky; ++i) sticky = mg[i] != 0;
    const int64_t e = int64_t(32) * int64_t(L - 2) - s;
    // value = m * 2^(e) + (sticky rest); normalised m already
    if (!sticky) return UpReal{m, static_cast<int32_t>(e)};
    return Norm(m, e, true);
}

BigInt UpReal::ceil_big() const {
    if (m == 0) return BigInt(0);
    if (e >= 0) return BigInt::from_u64(m) * BigInt::pow(BigInt(2), static_cast<uint64_t>(e));
    if (e <= -64) return BigInt(1);
    const int s = -e;
    uint64_t v = m >> s;
    if ((m & ((1ull << s) - 1)) != 0) ++v;
    return BigInt::from_u64(v);
}

UpReal UpAdd(const UpReal& a0, const UpReal& b0) {
    if (a0.m == 0) return b0;
    if (b0.m == 0) return a0;
    const UpReal& a = a0.e >= b0.e ? a0 : b0;
    const UpReal& b = a0.e >= b0.e ? b0 : a0;
    const int64_t sh = int64_t(a.e) - b.e;
    uint64_t bm;
    bool sticky = false;
    if (sh >= 64) {
        bm = 0;
        sticky = true;
    } else {
        bm = sh ? b.m >> sh : b.m;
        sticky = sh ? (b.m & ((1ull << sh) - 1)) != 0 : false;
    }
    uint64_t s = a.m + bm;
    int64_t e = a.e;
    if (s < a.m) {   // carry: (2^64 + s) / 2
        sticky = sticky || (s & 1);
        s = (s >> 1) | (1ull << 63);
        ++e;
    }
    return Norm(s, e, sticky);
}

UpReal UpMul(const UpReal& a, const UpReal& b) {
    if (a.m == 0 || b.m == 0) return UpReal{};
    // 64 x 64 -> 128 by 32-bit halves (no __int128 on this path)
    const uint64_t a1 = a.m >> 32, a0 = a.m & 0xffffffffull, b1 = b.m >> 32, b0 = b.m & 0xffffffffull;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffull) + (p10 & 0xffffffffull);
    const uint64_t lo = (mid << 32) | (p00 & 0xffffffffull);
    const uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    // hi != 0 (both mantissas >= 2^63); take the top 64 bits of hi:lo
    const int s = Clz64(hi);
    const uint64_t m = s ? (hi << s) | (lo >> (64 - s)) : hi;
    const bool sticky = s ? (lo << s) != 0 : lo != 0;
    return Norm(m, int64_t(a.e) + b.e + 64 - s, sticky);
}

// =================================================================================================================================
// The fixture
// =================================================================================================================================
namespace {
bool IsIntText(const std::string& v) {
    size_t i = (!v.empty() && (v[0] == '-' || v[0] == '+')) ? 1 : 0;
    if (i >= v.size()) return false;
    for (; i < v.size(); ++i)
        if (v[i] < '0' || v[i] > '9') return false;
    return true;
}
bool IsUIntText(const std::string& v) {
    if (v.empty() || v.size() > 10) return false;
    for (char c : v)
        if (c < '0' || c > '9') return false;
    return std::stoull(v) <= 0xffffffffull;
}
// whitespace-separated uint32 list; false on any token that is not one
bool ReadUInts(std::istringstream& is, std::vector<uint32_t>& v) {
    std::string t;
    while (is >> t) {
        if (!IsUIntText(t)) return false;
        v.push_back(static_cast<uint32_t>(std::stoul(t)));
    }
    return true;
}
}  // namespace

bool LoadFreeBlockFixture(const std::string& path, FreeBlockFixture& fx, std::string& why) {
    // (fix round 1, audit F6) every read is tested; the env hook HECKE_EXZ2P2_FIXTURES feeds arbitrary files to this loader
    std::ifstream f(path);
    if (!f) {
        why = "cannot open " + path;
        return false;
    }
    std::string line;
    if (!std::getline(f, line) || line != "EXZ2P2_FIXTURE v1") {
        why = "bad header in " + path;
        return false;
    }
    fx = FreeBlockFixture{};
    bool have_cell = false;
    std::vector<std::pair<std::string, SparseColsZ>> mats;
    uint64_t lineno = 1;
    auto fail = [&](const std::string& w) {
        why = path + ":" + std::to_string(lineno) + ": " + w;
        return false;
    };
    while (std::getline(f, line)) {
        ++lineno;
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "end") {
            if (!have_cell) return fail("no cell line");
            // shapes (B, Phi0, PhiCert: dim1 x r; F_p: r x r, one per p of ps and nothing else)
            for (auto& [name, S] : mats) {
                const bool fmat = name.rfind("F_", 0) == 0;
                const uint32_t R = fmat ? fx.r : fx.dim1;
                if (S.rows != R || S.ncols() != fx.r)
                    return fail("matrix " + name + " is " + std::to_string(S.rows) + " x " + std::to_string(S.ncols()) + ", expected " +
                                std::to_string(R) + " x " + std::to_string(fx.r));
                if (name == "B") fx.B = std::move(S);
                else if (name == "Phi0") fx.Phi0 = std::move(S);
                else if (name == "PhiCert") fx.PhiCert = std::move(S);
                else {
                    const std::string ptxt = name.substr(2);
                    if (!IsUIntText(ptxt)) return fail("bad operator name " + name);
                    const uint32_t p = static_cast<uint32_t>(std::stoul(ptxt));
                    if (std::find(fx.ps.begin(), fx.ps.end(), p) == fx.ps.end()) return fail(name + " is not in ps");
                    if (fx.F_model.count(p)) return fail("duplicate matrix " + name);
                    std::vector<BigInt> F(size_t(fx.r) * fx.r, BigInt(0));
                    for (uint32_t j = 0; j < fx.r; ++j)
                        for (const auto& [i, v] : S.cols[j]) F[size_t(i) * fx.r + j] = v;
                    fx.F_model[p] = std::move(F);
                }
            }
            if (fx.B.rows != fx.dim1 || fx.B.ncols() != fx.r) return fail("no matrix B");
            return true;
        }
        if (tag == "cell") {
            std::vector<uint32_t> c;
            if (!ReadUInts(is, c) || c.size() != 4) return fail("cell needs N n r dim1");
            fx.N = c[0];
            fx.n = c[1];
            fx.r = c[2];
            fx.dim1 = c[3];
            have_cell = true;
        } else if (tag == "ps") {
            if (!ReadUInts(is, fx.ps)) return fail("bad ps list");
        } else if (tag == "R0") {
            if (!ReadUInts(is, fx.R0)) return fail("bad R0 list");
        } else if (tag == "P") {
            if (!ReadUInts(is, fx.P)) return fail("bad P list");
        } else if (tag == "mat") {
            std::string name, tR, tC, tnz, extra;
            if (!(is >> name >> tR >> tC >> tnz) || (is >> extra) || !IsUIntText(tR) || !IsUIntText(tC) || !IsUIntText(tnz))
                return fail("mat needs <name> <rows> <cols> <nnz>");
            if (name != "B" && name != "Phi0" && name != "PhiCert" && name.rfind("F_", 0) != 0) return fail("unknown matrix " + name);
            for (const auto& m : mats)
                if (m.first == name) return fail("duplicate matrix " + name);
            const uint32_t R = static_cast<uint32_t>(std::stoul(tR)), C = static_cast<uint32_t>(std::stoul(tC));
            const uint64_t nz = std::stoull(tnz);
            if (nz > uint64_t(R) * C) return fail("matrix " + name + ": more entries than positions");
            SparseColsZ S;
            S.rows = R;
            S.cols.assign(C, {});
            for (uint64_t t = 0; t < nz; ++t) {
                if (!std::getline(f, line)) return fail("truncated matrix " + name);
                ++lineno;
                std::istringstream ls(line);
                std::string ti, tj, v, x2;
                if (!(ls >> ti >> tj >> v) || (ls >> x2) || !IsUIntText(ti) || !IsUIntText(tj) || !IsIntText(v))
                    return fail("matrix " + name + ": an entry line needs <i> <j> <integer>");
                const uint32_t i = static_cast<uint32_t>(std::stoul(ti)), j = static_cast<uint32_t>(std::stoul(tj));
                if (i >= R || j >= C) return fail("index out of range in " + name);
                BigInt bv = BigInt::from_string(v);
                if (bv.is_zero()) return fail("matrix " + name + ": a stored zero");
                S.cols[j].push_back({i, std::move(bv)});
            }
            for (auto& c : S.cols) {
                std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                for (size_t k = 1; k < c.size(); ++k)
                    if (c[k].first == c[k - 1].first) return fail("matrix " + name + ": duplicate entry (" + std::to_string(c[k].first) + ", .)");
            }
            mats.emplace_back(name, std::move(S));
        } else if (!tag.empty()) {
            return fail("unknown line '" + tag + "'");
        }
    }
    why = path + ": no end marker";
    return false;
}

const char* LaneRouteName(LaneRoute r) { return r == LaneRoute::Gpu ? "gpu" : "cpu"; }

// =================================================================================================================================
// The shared (read-only during the lanes) state
// =================================================================================================================================
namespace {

struct Shared {
    const IsoTreeCarrierZ* car = nullptr;
    const PackedZ* B = nullptr;
    std::vector<uint32_t> R0, P, ps;
    uint32_t r = 0, d = 0, D = 0, dim0 = 0, dim1 = 0, iso1 = 0;
    std::vector<std::unique_ptr<hecke3::IsoHeckeN>> T;   // one per p (terms_Z only are read)
    FreeBlockOptions opt;
};

// The integer term list of T~_Iso on C^1 for each p (built once; IsoHeckeN's reduced terms live at an arbitrary word modulus and are
// never read here).  Returns the refusal text or "".
std::string BuildTerms(Shared& S) {
    for (uint32_t p : S.ps) {
        auto T = std::make_unique<hecke3::IsoHeckeN>(S.car->N(), S.car->n(), p, hecke3::U32Ctx(kQ0), 1, true, S.opt.allow_bad_primes);
        if (T->refused()) return T->why();
        S.T.push_back(std::move(T));
    }
    return "";
}

// ---- one lane -------------------------------------------------------------------------------------------------------------------
struct LaneOut {
    bool admissible = false;
    std::string why;
    std::vector<uint32_t> F;   // ops x r x r row-major
    double seconds = 0;
    uint64_t rho_bytes = 0;
};

// rows of a CSR integer matrix mod q
SpVec RowMod(const SparseZ& A, uint32_t i, uint32_t q) {
    SpVec v;
    for (uint64_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k) {
        const uint32_t x = A.val[k].mod_u32(q);
        if (!x) continue;
        v.idx.push_back(A.col[k]);
        v.val.push_back(x);
    }
    return v;
}

// out (r x k) = Psi Y, Y row-major dim1 x k
void ApplyPsi(const Shared& S, const EchelonModQ& E0, const std::vector<SpVec>& dP, uint32_t q, const std::vector<uint32_t>& Y, uint32_t k,
              uint32_t* out, uint32_t ld_out) {
    const uint32_t r = S.r;
    std::vector<uint32_t> u(S.R0.size()), z;
    for (uint32_t c = 0; c < k; ++c) {
        for (size_t t = 0; t < S.R0.size(); ++t) u[t] = Y[size_t(S.R0[t]) * k + c];
        E0.solve_transposed(u, z);
        for (uint32_t i = 0; i < r; ++i) {
            uint64_t s = Y[size_t(S.P[i]) * k + c];
            const SpVec& w = dP[i];
            for (size_t t = 0; t < w.idx.size(); ++t) {
                const uint32_t zv = z[w.idx[t]];
                if (zv) s = (s + uint64_t(q - w.val[t]) * zv) % q;
            }
            out[size_t(i) * ld_out + c] = static_cast<uint32_t>(s);
        }
    }
}

// T~_Iso on C^1 mod q by its integer terms: W2 (iso1 x k) = sum_t (mul_t rho_n(E_t)) W[cb_t] into block rb_t.  CPU: one
// engine::kernels::gemm (Add) per term.  GPU: one batched GEMM per chunk on `stream`, the per-term products summed on the host in
// term order (exact canonical residues: the same bytes).
struct RhoMemo {
    std::map<integral_model::GKey, U32Buf> m;
    uint64_t bytes = 0, cap = 0;
    const U32Buf& get(const engine::ModCtx<uint32_t>& ctx, const Mat22& E, uint32_t n, U32Buf& scratch) {
        const integral_model::GKey k = integral_model::key_of(E);
        auto it = m.find(k);
        if (it != m.end()) return it->second;
        U32Buf R = integral_model::rho_n(ctx, E, n);
        const uint64_t b = uint64_t(R.rows) * R.cols * 4 + 64;
        if (bytes + b <= cap) {
            bytes += b;
            return m.emplace(k, std::move(R)).first->second;
        }
        scratch = std::move(R);
        return scratch;
    }
};

void ApplyIsoTermsCpu(const Shared& S, const hecke3::IsoHeckeN& T, const engine::ModCtx<uint32_t>& ctx, RhoMemo& memo,
                      std::vector<uint32_t>& W, std::vector<uint32_t>& W2, uint32_t k) {
    const uint32_t d = S.d, q = ctx.p;
    U32Buf Wb(W.data(), W.size(), S.iso1, k, engine::Layout::RowMajor);
    W2.assign(size_t(S.iso1) * k, 0);
    U32Buf W2b(W2.data(), W2.size(), S.iso1, k, engine::Layout::RowMajor);
    U32Buf A(d, d), scratch;
    for (const auto& t : T.terms_Z(1)) {
        const uint32_t mul = RedI64(t.mul, q);
        if (!mul) continue;
        const U32Buf& R = memo.get(ctx, t.E, S.car->n(), scratch);
        for (uint32_t x = 0; x < d; ++x)
            for (uint32_t y = 0; y < d; ++y) A.at(x, y) = static_cast<uint32_t>(uint64_t(mul) * R.at(x, y) % q);
        engine::kernels::gemm<uint32_t>(ctx, engine::cview(A), engine::cview(Wb).sub(t.cb * d, d, 0, k), engine::view(W2b).sub(t.rb * d, d, 0, k),
                                        engine::kernels::GemmOp::Add);
    }
}

void ApplyIsoTermsGpu(const Shared& S, const hecke3::IsoHeckeN& T, const engine::ModCtx<uint32_t>& ctx, RhoMemo& memo,
                      std::vector<uint32_t>& W, std::vector<uint32_t>& W2, uint32_t k, uint32_t lane_index) {
    namespace cu = engine::cuda;
    const uint32_t d = S.d, q = ctx.p;
    const auto& terms = T.terms_Z(1);
    std::vector<uint32_t> live;
    for (uint32_t i = 0; i < terms.size(); ++i)
        if (RedI64(terms[i].mul, q)) live.push_back(i);
    W2.assign(size_t(S.iso1) * k, 0);
    if (live.empty() || k == 0) return;
    cu::CudaCtx& cx = cu::CudaCtx::get();
    const cu::Stream s = cx.stream(static_cast<int>(lane_index % static_cast<uint32_t>(cx.num_streams())));   // one stream per prime
    U32Buf Wb(W.data(), W.size(), S.iso1, k, engine::Layout::RowMajor);
    cu::DeviceBuf<uint32_t> dW(Wb);
    dW.upload(s);
    // chunks of terms with A (d x d) + Z (d x k) <= 256 MiB on the device
    const uint64_t per = (uint64_t(d) * d + uint64_t(d) * k) * 4;
    const uint32_t chunk = static_cast<uint32_t>(std::max<uint64_t>(1, (256ull << 20) / std::max<uint64_t>(per, 1)));
    U32Buf scratch;
    for (size_t c0 = 0; c0 < live.size(); c0 += chunk) {
        const uint32_t nt = static_cast<uint32_t>(std::min<size_t>(chunk, live.size() - c0));
        U32Buf A(nt * d, d), Z(nt * d, k);
        for (uint32_t u = 0; u < nt; ++u) {
            const auto& t = terms[live[c0 + u]];
            const uint32_t mul = RedI64(t.mul, q);
            const U32Buf& R = memo.get(ctx, t.E, S.car->n(), scratch);
            for (uint32_t x = 0; x < d; ++x)
                for (uint32_t y = 0; y < d; ++y) A.at(u * d + x, y) = static_cast<uint32_t>(uint64_t(mul) * R.at(x, y) % q);
        }
        cu::DeviceBuf<uint32_t> dA(A), dZ(Z);
        dA.upload(s);
        dZ.upload(s);
        cu::GemmBatch<uint32_t> batch;
        batch.set_shape(dA.cview().sub(0, d, 0, d), dW.cview().sub(0, d, 0, k), dZ.view().sub(0, d, 0, k));
        for (uint32_t u = 0; u < nt; ++u) {
            const auto& t = terms[live[c0 + u]];
            batch.push(dA.cview().sub(u * d, d, 0, d), dW.cview().sub(t.cb * d, d, 0, k), dZ.view().sub(u * d, d, 0, k));
        }
        batch.upload(s);
        batch.run(ctx, engine::kernels::GemmOp::Set, s);
        dZ.mark_device_written();
        dZ.download(s);
        // the fixed-order host sum (term order): W2[rb] += Z_u
        for (uint32_t u = 0; u < nt; ++u) {
            const auto& t = terms[live[c0 + u]];
            for (uint32_t x = 0; x < d; ++x) {
                uint32_t* yr = W2.data() + (size_t(t.rb) * d + x) * k;
                for (uint32_t c = 0; c < k; ++c) {
                    const uint32_t v = yr[c] + Z.at(u * d + x, c);
                    yr[c] = v >= q ? v - q : v;
                }
            }
        }
    }
}

// G'' X = H'' over F_q: [G'' | H''] (r x (r + m)) by Gauss-Jordan; false iff G'' is singular mod q.
bool SolveModQ(std::vector<uint32_t>& G, std::vector<uint32_t>& H, uint32_t r, uint32_t m, uint32_t q) {
    for (uint32_t c = 0; c < r; ++c) {
        uint32_t piv = c;
        while (piv < r && !G[size_t(piv) * r + c]) ++piv;
        if (piv == r) return false;
        if (piv != c) {
            for (uint32_t j = 0; j < r; ++j) std::swap(G[size_t(piv) * r + j], G[size_t(c) * r + j]);
            for (uint32_t j = 0; j < m; ++j) std::swap(H[size_t(piv) * m + j], H[size_t(c) * m + j]);
        }
        const uint64_t inv = PowModU(G[size_t(c) * r + c], q - 2, q);
        for (uint32_t j = 0; j < r; ++j) G[size_t(c) * r + j] = static_cast<uint32_t>(G[size_t(c) * r + j] * inv % q);
        for (uint32_t j = 0; j < m; ++j) H[size_t(c) * m + j] = static_cast<uint32_t>(H[size_t(c) * m + j] * inv % q);
        for (uint32_t i = 0; i < r; ++i) {
            if (i == c) continue;
            const uint64_t f = G[size_t(i) * r + c];
            if (!f) continue;
            const uint64_t nf = q - f;
            for (uint32_t j = 0; j < r; ++j) G[size_t(i) * r + j] = static_cast<uint32_t>((G[size_t(i) * r + j] + nf * G[size_t(c) * r + j]) % q);
            for (uint32_t j = 0; j < m; ++j) H[size_t(i) * m + j] = static_cast<uint32_t>((H[size_t(i) * m + j] + nf * H[size_t(c) * m + j]) % q);
        }
    }
    return true;
}

LaneOut RunLane(const Shared& S, uint32_t q, uint32_t lane_index) {
    const auto t0 = clk::now();
    LaneOut L;
    const IsoTreeCarrierZ& car = *S.car;
    if ((6ull * car.N()) % q == 0) {
        L.why = "q | 6N";
        return L;
    }
    const engine::ModCtx<uint32_t> ctx(q);
    // (fix round 1, audit F5 d) the lane's carrier TreeModule is released on EVERY exit of the lane, a throw included
    struct LaneRelease {
        const IsoTreeCarrierZ& c;
        uint32_t q;
        ~LaneRelease() { c.release_lane(q); }
    } release_guard{car, q};
    // the per-lane sparse elimination of the rows R0 of d^0 (the Psi route; Phi_0 is never reduced)
    const SparseZ& d0 = car.d(0);
    EchelonModQ E0(q, d0.cols);
    for (uint32_t i : S.R0)
        if (!E0.insert(RowMod(d0, i, q))) {
            L.why = "the rows R0 of d^0 are dependent mod q";
            return L;
        }
    std::vector<SpVec> dP(S.r);
    for (uint32_t i = 0; i < S.r; ++i) dP[i] = RowMod(d0, S.P[i], q);
    const uint32_t r = S.r, nops = static_cast<uint32_t>(S.ps.size()), w = std::max<uint32_t>(1, S.opt.tile);
    std::vector<uint32_t> G(size_t(r) * r, 0), H(size_t(r) * r * nops, 0);   // H: r x (nops * r), op-major column blocks
    RhoMemo memo;
    memo.cap = S.opt.rho_memo_cap_bytes;
    std::vector<uint32_t> X, Wv, W2v, Y, tmp(size_t(r) * w);
    for (uint32_t c0 = 0; c0 < r; c0 += w) {
        const uint32_t k = std::min(w, r - c0);
        // the tile of B mod q, reduced on the fly from the BigInt master
        X.assign(size_t(S.dim1) * k, 0);
        for (uint32_t c = 0; c < k; ++c)
            for (uint64_t e = S.B->colptr[c0 + c]; e < S.B->colptr[c0 + c + 1]; ++e) X[size_t(S.B->row[e]) * k + c] = S.B->mod_u32(e, q);
        ApplyPsi(S, E0, dP, q, X, k, tmp.data(), k);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t c = 0; c < k; ++c) G[size_t(i) * r + c0 + c] = tmp[size_t(i) * k + c];
        car.phi_star_mod(q, 1, X, k, Wv);   // (fix round 1, audit F8 d) Phi^* X does not depend on p: once per tile
        for (uint32_t o = 0; o < nops; ++o) {
            if (S.opt.route == LaneRoute::Gpu) ApplyIsoTermsGpu(S, *S.T[o], ctx, memo, Wv, W2v, k, lane_index);
            else ApplyIsoTermsCpu(S, *S.T[o], ctx, memo, Wv, W2v, k);
            car.psi_star_mod(q, 1, W2v, k, Y);
            ApplyPsi(S, E0, dP, q, Y, k, tmp.data(), k);
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t c = 0; c < k; ++c) H[size_t(i) * (r * nops) + o * r + c0 + c] = tmp[size_t(i) * k + c];
        }
    }
    L.rho_bytes = memo.bytes;
    if (!SolveModQ(G, H, r, r * nops, q)) {
        L.why = "G'' = Psi B is singular mod q";
        L.seconds = since(t0);
        return L;
    }
    L.F.assign(size_t(nops) * r * r, 0);
    for (uint32_t o = 0; o < nops; ++o)
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) L.F[(size_t(o) * r + i) * r + j] = H[size_t(i) * (r * nops) + o * r + j];
    L.admissible = true;
    L.seconds = since(t0);
    return L;
}

// ---- the bound (design Sec. 1.6), upward-rounded ---------------------------------------------------------------------------------
struct Bound {
    std::vector<std::vector<UpReal>> beta_H;   // per op: r x r row-major
    std::vector<std::vector<BigInt>> beta_H_ceil;   // per op: ceil(beta_H) (r x r), formed once (audit F8)
    std::vector<BigInt> beta_H_max_ceil;       // per op: max_ij ceil(beta_H,ij): Q <= this => no a-posteriori stop (no lift needed)
    std::vector<BigInt> Gp;                    // G' = Phi_0^T B (exact)
    // (fix round 1, audit F2) the inverse: NotTried (use_apriori off), Skipped (too large), Singular (det G' = 0, proven), Ok
    enum class Inv { NotTried, Skipped, Singular, Ok } inv = Inv::NotTried;
    bool have_inv = false;                     // inv == Ok
    std::vector<BigInt> adj;                   // d' G'^{-1} (exact), |d'| = |det G'|
    BigInt det;
    std::vector<BigInt> need_apriori;          // per op: 2 ceil(max_ij sum_k |adj_ik| beta_H,kj): a-priori iff Q |det| > this
    std::vector<std::vector<BigInt>> rhs;      // per op: ceil(sum_k |adj_ik| beta_H,kj) (r x r): the a-priori post-check (audit F3 b)
    std::vector<uint32_t> bits_beta_H, bits_beta_F;
    // (EXZ2-P3) a SECOND a-posteriori test with other functionals Phi_alt (Phi_0 beside the dual Xi), from the same majorant pass V:
    // G'_alt = Phi_alt^T B (det != 0 by a witness prime), ceil(beta_H,alt) = ceil(|Phi_alt|^T V)
    bool have_alt = false;
    std::vector<BigInt> Gp_alt;
    std::vector<std::vector<BigInt>> beta_H_ceil_alt;
    std::vector<BigInt> beta_H_max_ceil_alt;
};

// |rho_n(h)| of the block-monomial action of g on M_N (exact entries, upward) -- the transport maps' majorant blocks
struct AbsMono {
    std::vector<uint32_t> col;
    std::vector<const std::vector<UpReal>*> blk;
};
struct MajorantCtx {
    const IsoTreeCarrierZ* car;
    std::shared_ptr<const hecke::level::Gamma0LevelN> A;
    std::map<integral_model::GKey, std::vector<UpReal>> rho_abs;    // |rho_n(h)| exact -> up
    std::map<integral_model::GKey, AbsMono> mono;
    std::map<integral_model::GKey, std::vector<UpReal>> term_abs;   // rho_n(|E|)
    const std::vector<UpReal>& abs_rho(const Mat22& h, uint32_t n) {
        const integral_model::GKey k = integral_model::key_of(h);
        auto it = rho_abs.find(k);
        if (it != rho_abs.end()) return it->second;
        const std::vector<BigInt> Z = RhoNZ(h, n);
        std::vector<UpReal> v(Z.size());
        for (size_t i = 0; i < Z.size(); ++i) v[i] = UpReal::of_big(Z[i]);
        return rho_abs.emplace(k, std::move(v)).first->second;
    }
    const AbsMono& mono_of(const Mat22& g, uint32_t n) {
        const integral_model::GKey k = integral_model::key_of(g);
        auto it = mono.find(k);
        if (it != mono.end()) return it->second;
        AbsMono M;
        const uint32_t m = A->m();
        M.col.resize(m);
        M.blk.resize(m);
        for (uint32_t j = 1; j <= m; ++j) {
            const Mat22 x = A->rep(j) * g;
            const uint32_t jj = A->class_of(x);
            const Mat22 h = x * A->rep(jj).inv_unimodular();
            ENGINE_CHECK_MSG(A->contains(h), "FreeBlockCrt majorant: h_j(g) not in Gamma_0(N)");
            M.col[j - 1] = jj - 1;
            M.blk[j - 1] = &abs_rho(h, n);
        }
        return mono.emplace(k, std::move(M)).first->second;
    }
};

// out[block j] += |coef| |rho(g)| in[block col j] on k columns (UpReal)
void AddAbsRho(MajorantCtx& C, const integral_model::QGroupRingElt& a, uint32_t D, uint32_t d, const UpReal* in, UpReal* out, uint32_t k) {
    for (const auto& [gk, rr] : a.terms) {
        if (rr.num == 0) continue;
        ENGINE_CHECK_MSG(rr.den == 1, "FreeBlockCrt majorant: a non-integral coefficient in the integral comparison");
        const UpReal cf = UpReal::of_big(BigInt(rr.num));
        const AbsMono& M = C.mono_of(integral_model::mat_of(gk), C.car->n());
        for (uint32_t j = 0; j < M.col.size(); ++j) {
            const std::vector<UpReal>& blk = *M.blk[j];
            const UpReal* src = in + size_t(M.col[j]) * d * k;
            UpReal* dst = out + size_t(j) * d * k;
            for (uint32_t x = 0; x < d; ++x)
                for (uint32_t y = 0; y < d; ++y) {
                    const UpReal& b = blk[size_t(x) * d + y];
                    if (b.is_zero()) continue;
                    const UpReal f = UpMul(cf, b);
                    for (uint32_t c = 0; c < k; ++c)
                        if (!src[size_t(y) * k + c].is_zero()) dst[size_t(x) * k + c] = UpAdd(dst[size_t(x) * k + c], UpMul(f, src[size_t(y) * k + c]));
                }
        }
        (void)D;
    }
}

// Fraction-free Gauss-Jordan: [G | I] -> [d' I | d' G^{-1}] exactly (Bareiss divisions checked); false iff singular.
bool InverseFF(std::vector<BigInt> G, uint32_t r, std::vector<BigInt>& adj, BigInt& det) {
    if (r == 0) {   // (fix round 1, audit F1) the empty matrix: det = 1, adj empty (A[0] does not exist)
        adj.clear();
        det = BigInt(1);
        return true;
    }
    const uint32_t C = 2 * r;
    std::vector<BigInt> A(size_t(r) * C, BigInt(0));
    for (uint32_t i = 0; i < r; ++i) {
        for (uint32_t j = 0; j < r; ++j) A[size_t(i) * C + j] = G[size_t(i) * r + j];
        A[size_t(i) * C + r + i] = BigInt(1);
    }
    BigInt prev(1);
    for (uint32_t k = 0; k < r; ++k) {
        uint32_t p = k;
        while (p < r && A[size_t(p) * C + k].is_zero()) ++p;
        if (p == r) return false;
        if (p != k)
            for (uint32_t j = 0; j < C; ++j) std::swap(A[size_t(p) * C + j], A[size_t(k) * C + j]);
        const BigInt piv = A[size_t(k) * C + k];
        for (uint32_t i = 0; i < r; ++i) {
            if (i == k) continue;
            const BigInt aik = A[size_t(i) * C + k];
            for (uint32_t j = 0; j < C; ++j) {
                if (j == k) continue;
                BigInt v = piv * A[size_t(i) * C + j] - aik * A[size_t(k) * C + j];
                BigInt qq, rem;
                BigInt::divmod(v, prev, qq, rem);
                ENGINE_CHECK_MSG(rem.is_zero(), "InverseFF: a Bareiss division is not exact");
                A[size_t(i) * C + j] = std::move(qq);
            }
            A[size_t(i) * C + k] = BigInt(0);
        }
        // the pivot row is left as it is (fraction-free Gauss-Jordan: every other row is now scaled by piv / prev)
        prev = piv;
    }
    det = A[0];
    for (uint32_t i = 0; i < r; ++i)
        ENGINE_CHECK_MSG(A[size_t(i) * C + i] == det, "InverseFF: the diagonal is not d' I");
    adj.assign(size_t(r) * r, BigInt(0));
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) adj[size_t(i) * r + j] = A[size_t(i) * C + r + j];
    // check d' I = G * adj (exact)
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < r; ++j) {
            BigInt s(0);
            for (uint32_t t = 0; t < r; ++t) s += G[size_t(i) * r + t] * adj[size_t(t) * r + j];
            ENGINE_CHECK_MSG(s == (i == j ? det : BigInt(0)), "InverseFF: G adj != d' I");
        }
    return true;
}

// (fix round 2, N1) the two exact vanishing checks shared by every proof route and by the certificate, over Z on the packed columns.
// (EXZ2-P3) one logic: IsoTreeCarrierZ's FirstNonCocycle / FirstNotKillingD0 (P1b's ExactCocycles / ExactKillsD0 call the same).
//   CheckCocycles:  d^1 B = 0 (B is a set of cocycles);   CheckKillsD0:  Phi^T d^0 = 0 (Phi kills the coboundaries).
// "" iff it holds, else the reason.
std::string CheckCocycles(const IsoTreeCarrierZ& car, const PackedZ& B) {
    const int64_t j = FirstNonCocycle(car, B);
    return j < 0 ? std::string() : "d^1 b_" + std::to_string(j + 1) + " != 0 (B is not a set of cocycles)";
}
std::string CheckKillsD0(const IsoTreeCarrierZ& car, const PackedZ& Phi, const char* name) {
    const int64_t j = FirstNotKillingD0(car, Phi);
    return j < 0 ? std::string()
                 : std::string(name) + " column " + std::to_string(j + 1) + " does not kill d^0 (" + name + "^T d^0 != 0)";
}

// (fix round 1, audit F3 a; EXZ2-P3: P1b's CertifyBasis, not a second implementation) the Z-basis certificate: d^1 B = 0,
// Phi_cert^T d^0 = 0, det(Phi_cert^T B) = +-1 (residues beyond 2 x Hadamard) and the rank witness (rank L <= r).  "" iff it holds.
std::string CertificateReason(const IsoTreeCarrierZ& car, const PackedZ& B, const PackedZ& PhiCert, Certificate& C) {
    C = CertifyBasis(car, B, PhiCert);
    if (C.ok) return "";
    if (!C.cocycles) return CheckCocycles(car, B);
    if (!C.kills_d0) return CheckKillsD0(car, PhiCert, "Phi_cert");
    if (!C.rank_ok)
        return "the rank witness failed (rank_q d^1 + rank_q d^0 < dim C^1 - r at " + std::to_string(C.rank_tries) + " primes): rank L <= r unproven";
    return "det(Phi_cert^T B) != +-1 (B is not a Z-basis of L; " + std::to_string(C.primes) + " det primes, Hadamard " +
           std::to_string(C.hadamard_bits) + " bits)";
}

// (fix round 1, audit F2 b) det G' != 0 by a witness prime: det(G' mod q) != 0 for one q proves det G' != 0.  Returns that q, or 0 when
// det G' = 0 is PROVEN: every prime tried divides det G' and their product exceeds H = prod_i sum_j |G'_ij| >= |det G'| (Hadamard).
uint32_t DetWitness(const std::vector<BigInt>& G, uint32_t r) {
    BigInt H(1);
    for (uint32_t i = 0; i < r; ++i) {
        BigInt s(0);
        for (uint32_t j = 0; j < r; ++j) s += G[size_t(i) * r + j].abs();
        H = H * s;
    }
    if (H.is_zero()) return 0;   // a zero row
    BigInt Qacc(1);
    std::vector<uint32_t> cand;
    for (size_t pos = 0;; ++pos) {
        if (pos >= cand.size()) cand = CachedCrtPrimes(static_cast<uint32_t>(std::max<size_t>(64, 2 * cand.size())));
        const uint32_t q = cand[pos];
        std::vector<uint32_t> A(size_t(r) * r);
        for (size_t e = 0; e < A.size(); ++e) A[e] = G[e].mod_u32(q);
        bool nonsing = true;
        for (uint32_t c = 0; c < r; ++c) {
            uint32_t piv = c;
            while (piv < r && !A[size_t(piv) * r + c]) ++piv;
            if (piv == r) {
                nonsing = false;
                break;
            }
            if (piv != c)
                for (uint32_t j = 0; j < r; ++j) std::swap(A[size_t(piv) * r + j], A[size_t(c) * r + j]);
            const uint64_t inv = PowModU(A[size_t(c) * r + c], q - 2, q);
            for (uint32_t i = c + 1; i < r; ++i) {
                const uint64_t f = uint64_t(A[size_t(i) * r + c]) * inv % q;
                if (!f) continue;
                for (uint32_t j = c; j < r; ++j)
                    A[size_t(i) * r + j] = static_cast<uint32_t>((A[size_t(i) * r + j] + (q - f) * A[size_t(c) * r + j]) % q);
            }
        }
        if (nonsing) return q;
        Qacc = Qacc * BigInt(int64_t(q));
        if (Qacc > H) return 0;
    }
}

// The bound of design Sec. 1.6 with the functionals Phi (Phi_0, or with `unimodular` the certified Phi_cert: G = Phi^T B has det +-1,
// its inverse is integral and found by InverseUnimodular instead of the fraction-free inverse; EXZ2-P3).
Bound ComputeBound(const Shared& S, const PackedZ& Phi0, bool unimodular, double* t_inv, uint32_t* inv_steps, const PackedZ* PhiAlt = nullptr) {
    Bound Bd;
    const IsoTreeCarrierZ& car = *S.car;
    const uint32_t r = S.r, d = S.d, D = S.D;
    MajorantCtx C;
    C.car = &car;
    C.A = std::make_shared<const hecke::level::Gamma0LevelN>(car.N());
    const auto& cmp = hecke3::SharedTreeComparison();
    // |B| on C^1 (dim1 x r)
    std::vector<UpReal> X(size_t(S.dim1) * r);
    for (uint32_t j = 0; j < r; ++j)
        for (uint64_t e = S.B->colptr[j]; e < S.B->colptr[j + 1]; ++e) X[size_t(S.B->row[e]) * r + j] = UpReal::of_big(S.B->value(e));
    // |Phi^*| X
    std::vector<UpReal> W(size_t(S.iso1) * r);
    const auto& prow = cmp.phi().value.at(1);
    for (uint32_t y = 0; y < prow.size(); ++y)
        for (const auto& [x, b] : prow[y]) AddAbsRho(C, b, D, d, X.data() + size_t(x) * D * r, W.data() + size_t(y) * D * r, r);
    // G' = Phi_0^T B exactly
    Bd.Gp = PairingMatrix(Phi0, *S.B);
    // the a-priori bound needs d' G'^{-1}; skipped (a-posteriori only) when the fraction-free inverse would be too large.  (fix round 1,
    // audit F2) "singular" and "skipped" are distinct outcomes: a singular G' is refused by the caller, never silently a-posteriori.
    uint32_t gbits = 0;
    for (const auto& x : Bd.Gp) gbits = std::max(gbits, BitsOf(x));
    if (S.opt.use_apriori && unimodular) {
        // (EXZ2-P3) det(Phi_cert^T B) = +-1 is CERTIFIED (CertifyBasis): adj = det G^{-1}, |adj| = |G^{-1}|, integral -- by q-adic lifting
        // (output-sensitive, exact), not by the fraction-free inverse
        const auto ti = clk::now();
        std::vector<BigInt> X1;
        ENGINE_CHECK_MSG(InverseUnimodular(Bd.Gp, r, X1, inv_steps), "FreeBlockCrt: a certified G_c = Phi_cert^T B has no integral inverse");
        // the sign of det: det G mod the first CrtPrimes prime (+-1 there, as at every prime)
        const uint32_t q = CachedCrtPrimes(1)[0];
        std::vector<uint32_t> Gq(size_t(r) * r);
        for (size_t e = 0; e < Gq.size(); ++e) Gq[e] = Bd.Gp[e].mod_u32(q);
        const uint32_t dq = DetModPrime(std::move(Gq), r, q);
        ENGINE_CHECK_MSG(dq == 1 || dq == q - 1, "FreeBlockCrt: det G_c mod q is not +-1");
        Bd.det = BigInt(dq == 1 ? 1 : -1);
        Bd.adj = std::move(X1);
        if (dq != 1)
            for (auto& x : Bd.adj) x = -x;
        Bd.inv = Bound::Inv::Ok;
        if (t_inv) *t_inv = since(ti);
    } else if (S.opt.use_apriori) {
        const auto ti = clk::now();
        if (uint64_t(r) * (gbits + 8) > 400000) Bd.inv = Bound::Inv::Skipped;
        else Bd.inv = InverseFF(Bd.Gp, r, Bd.adj, Bd.det) ? Bound::Inv::Ok : Bound::Inv::Singular;
        if (t_inv) *t_inv = since(ti);
    }
    Bd.have_inv = Bd.inv == Bound::Inv::Ok;
    std::vector<UpReal> absPhi(Phi0.nnz());   // |Phi| upward, once (EXZ2-P3: packed values are decoded once, not per op x r)
    for (uint64_t e = 0; e < Phi0.nnz(); ++e) absPhi[e] = UpReal::of_big(Phi0.value(e));
    std::vector<UpReal> absAlt;
    if (PhiAlt) {
        Bd.Gp_alt = PairingMatrix(*PhiAlt, *S.B);
        Bd.have_alt = DetWitness(Bd.Gp_alt, r) != 0;   // det G'_alt != 0, else the second test is not used
        if (Bd.have_alt) {
            absAlt.resize(PhiAlt->nnz());
            for (uint64_t e = 0; e < PhiAlt->nnz(); ++e) absAlt[e] = UpReal::of_big(PhiAlt->value(e));
        }
    }
    for (size_t o = 0; o < S.ps.size(); ++o) {
        // sum_t |mul| rho_n(|E|) on W
        std::vector<UpReal> W2(size_t(S.iso1) * r);
        for (const auto& t : S.T[o]->terms_Z(1)) {
            if (t.mul == 0) continue;
            const integral_model::GKey ek = integral_model::key_of(t.E);
            auto it = C.term_abs.find(ek);
            if (it == C.term_abs.end()) {
                const std::vector<BigInt> Z = RhoNAbsZ(t.E, car.n());
                std::vector<UpReal> v(Z.size());
                for (size_t i = 0; i < Z.size(); ++i) v[i] = UpReal::of_big(Z[i]);
                it = C.term_abs.emplace(ek, std::move(v)).first;
            }
            const UpReal mul = UpReal::of_big(BigInt(t.mul < 0 ? -t.mul : t.mul));
            for (uint32_t x = 0; x < d; ++x)
                for (uint32_t y = 0; y < d; ++y) {
                    const UpReal& b = it->second[size_t(x) * d + y];
                    if (b.is_zero()) continue;
                    const UpReal f = UpMul(mul, b);
                    UpReal* yr = W2.data() + (size_t(t.rb) * d + x) * r;
                    const UpReal* xr = W.data() + (size_t(t.cb) * d + y) * r;
                    for (uint32_t c = 0; c < r; ++c)
                        if (!xr[c].is_zero()) yr[c] = UpAdd(yr[c], UpMul(f, xr[c]));
                }
        }
        // |Psi^*|
        std::vector<UpReal> V(size_t(S.dim1) * r);
        const auto& srow = cmp.psi().value.at(1);
        for (uint32_t x = 0; x < srow.size(); ++x)
            for (const auto& [y, b] : srow[x]) AddAbsRho(C, b, D, d, W2.data() + size_t(y) * D * r, V.data() + size_t(x) * D * r, r);
        // beta_H = |Phi_0|^T V
        std::vector<UpReal> bH(size_t(r) * r);
        uint32_t bh_bits = 0;
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                UpReal s;
                for (uint64_t e = Phi0.colptr[i]; e < Phi0.colptr[i + 1]; ++e) s = UpAdd(s, UpMul(absPhi[e], V[size_t(Phi0.row[e]) * r + j]));
                bH[size_t(i) * r + j] = s;
                bh_bits = std::max<uint32_t>(bh_bits, s.is_zero() ? 0u : static_cast<uint32_t>(std::max<int64_t>(0, int64_t(s.e) + 64)));
            }
        Bd.bits_beta_H.push_back(bh_bits);
        if (Bd.have_alt) {
            std::vector<BigInt> ac(size_t(r) * r);
            BigInt amax(0);
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) {
                    UpReal sa;
                    for (uint64_t e = PhiAlt->colptr[i]; e < PhiAlt->colptr[i + 1]; ++e) sa = UpAdd(sa, UpMul(absAlt[e], V[size_t(PhiAlt->row[e]) * r + j]));
                    ac[size_t(i) * r + j] = sa.ceil_big();
                    if (ac[size_t(i) * r + j] > amax) amax = ac[size_t(i) * r + j];
                }
            Bd.beta_H_ceil_alt.push_back(std::move(ac));
            Bd.beta_H_max_ceil_alt.push_back(std::move(amax));
        }
        std::vector<BigInt> bHc(size_t(r) * r);
        BigInt bhmax(0);
        for (size_t e = 0; e < bHc.size(); ++e) {
            bHc[e] = bH[e].ceil_big();
            if (bHc[e] > bhmax) bhmax = bHc[e];
        }
        Bd.beta_H_max_ceil.push_back(bhmax);
        Bd.beta_H_ceil.push_back(std::move(bHc));
        if (Bd.have_inv) {
            BigInt need(0);
            uint32_t bf_bits = 0;
            std::vector<BigInt> rhs(size_t(r) * r);
            for (uint32_t i = 0; i < r; ++i)
                for (uint32_t j = 0; j < r; ++j) {
                    UpReal s;
                    for (uint32_t k2 = 0; k2 < r; ++k2) s = UpAdd(s, UpMul(UpReal::of_big(Bd.adj[size_t(i) * r + k2]), bH[size_t(k2) * r + j]));
                    BigInt cb = s.ceil_big();
                    if (cb > need) need = cb;
                    rhs[size_t(i) * r + j] = std::move(cb);
                }
            Bd.rhs.push_back(std::move(rhs));
            need = need * BigInt(2);
            // bits of beta_F = need / (2 |det|), rounded up
            const BigInt bf = need / (BigInt(2) * Bd.det.abs()) + BigInt(1);
            bf_bits = static_cast<uint32_t>(bf.bit_length());
            Bd.need_apriori.push_back(need);
            Bd.bits_beta_F.push_back(bf_bits);
        } else {
            Bd.need_apriori.push_back(BigInt(0));
            Bd.bits_beta_F.push_back(0);
            Bd.rhs.emplace_back();
        }
        Bd.beta_H.push_back(std::move(bH));
    }
    return Bd;
}

uint32_t DefaultThreads() { return engine::threads::configured(engine::threads::kCap16); }

// (fix round 1, audit F6) the preconditions of the public entry points: shapes, index ranges, sortedness (ENGINE_CHECK: a violation
// is a programming error, not data)
void CheckColumns(const PackedZ& M, uint32_t rows, uint32_t r, const char* name) {
    // PackedZ::append_col already ENGINE_CHECKs strictly ascending rows and skips zeros; here the shape and the arena's consistency
    ENGINE_CHECK_MSG(M.rows == rows && M.cols == r && M.colptr.size() == size_t(r) + 1 && M.off.size() == M.row.size() &&
                         M.colptr.back() == M.row.size(),
                     (std::string("FreeBlockCrt: ") + name + " must be dim C^1 x r (packed)").c_str());
    for (uint32_t j = 0; j < r; ++j)
        for (uint64_t e = M.colptr[j]; e < M.colptr[j + 1]; ++e)
            ENGINE_CHECK_MSG(M.row[e] < rows && (e == M.colptr[j] || M.row[e - 1] < M.row[e]),
                             (std::string("FreeBlockCrt: ") + name + ": a column with an index out of range or unsorted rows").c_str());
}
void CheckIndexSet(const std::vector<uint32_t>& v, uint32_t size, uint32_t bound, const char* name) {
    ENGINE_CHECK_MSG(v.size() == size, (std::string("FreeBlockCrt: |") + name + "| is wrong").c_str());
    std::vector<uint32_t> s(v);
    std::sort(s.begin(), s.end());
    for (size_t t = 0; t < s.size(); ++t)
        ENGINE_CHECK_MSG(s[t] < bound && (t == 0 || s[t - 1] < s[t]), (std::string("FreeBlockCrt: ") + name + ": an index out of range or repeated").c_str());
}
void CheckInput(const FreeBlockInput& in) {
    ENGINE_CHECK_MSG(in.car && in.B, "FreeBlockCrt: carrier and basis required");
    const IsoTreeCarrierZ& car = *in.car;
    const uint32_t r = in.B->cols, dim1 = car.dim(1);
    ENGINE_CHECK_MSG(in.B->rows == dim1 && in.P.size() == r && r == car.b1(), "FreeBlockCrt: B must be r = b_1 columns on C^1");
    CheckColumns(*in.B, dim1, r, "B");
    if (in.Phi0) CheckColumns(*in.Phi0, dim1, r, "Phi_0");
    if (in.PhiCert) CheckColumns(*in.PhiCert, dim1, r, "Phi_cert");
    ENGINE_CHECK_MSG(car.d(0).rows == dim1, "FreeBlockCrt: d^0 has dim C^1 rows");
    CheckIndexSet(in.R0, car.dim(0) - car.b0(), dim1, "R0 (rank d^0 = dim C^0 - b_0 rows of d^0)");
    CheckIndexSet(in.P, r, dim1, "P");
}

void InitShared(Shared& S, const FreeBlockInput& in, const FreeBlockOptions& opt) {
    S.car = in.car;
    S.B = in.B;
    S.R0 = in.R0;
    S.P = in.P;
    S.ps = in.ps;
    S.r = in.B->cols;
    S.d = in.car->n() + 1;
    S.D = in.car->D();
    S.dim0 = in.car->dim(0);
    S.dim1 = in.car->dim(1);
    S.iso1 = in.car->iso_dim(1);
    S.opt = opt;
}

// (fix round 1, audit F5 a/b) the GEMM path override, restored on every exit; only a forced path touches the process-wide setting
struct KernelPathGuard {
    bool on = false;
    engine::kernels::KernelPath saved = engine::kernels::KernelPath::Auto;
    explicit KernelPathGuard(engine::cpu::IsaPath p) : on(p != engine::cpu::IsaPath::Auto) {
        if (!on) return;
        saved = engine::kernels::kernel_path();
        engine::kernels::set_kernel_path(p);
    }
    ~KernelPathGuard() {
        if (on) engine::kernels::set_kernel_path(saved);
    }
    KernelPathGuard(const KernelPathGuard&) = delete;
    KernelPathGuard& operator=(const KernelPathGuard&) = delete;
};

}  // namespace

const char* FreeBlockRefusalName(FreeBlockRefusal r) {
    switch (r) {
        case FreeBlockRefusal::None: return "None";
        case FreeBlockRefusal::NoDevice: return "NoDevice";
        case FreeBlockRefusal::Terms: return "Terms";
        case FreeBlockRefusal::RssCeiling: return "RssCeiling";
        case FreeBlockRefusal::Uncertified: return "Uncertified";
        case FreeBlockRefusal::SingularG: return "SingularG";
        case FreeBlockRefusal::NoProof: return "NoProof";
        case FreeBlockRefusal::Inadmissible: return "Inadmissible";
        case FreeBlockRefusal::BoundViolated: return "BoundViolated";
    }
    return "?";
}

// =================================================================================================================================
// The public entry points
// =================================================================================================================================
uint64_t PredictFreeBlockRss(const FreeBlockInput& in, const FreeBlockOptions& opt, uint64_t terms) {
    const IsoTreeCarrierZ& c = *in.car;
    const uint64_t D = c.D(), r = in.B ? in.B->cols : c.b1(), d = c.n() + 1, nops = in.ps.size();
    const uint64_t w = std::min<uint64_t>(std::max<uint32_t>(1, opt.tile), std::max<uint64_t>(r, 1));
    const uint64_t threads = opt.threads ? opt.threads : 16;
    auto big = [](uint64_t bits) -> uint64_t { return 48 + 4 * ((bits + 31) / 32); };
    uint64_t b = 32ull << 20;
    (void)big;
    // (EXZ2-P3) the packed masters are the caller's (resident already); the bound decodes |Phi| once (16 B per non-zero)
    if (in.B) b += in.B->bytes();
    if (in.Phi0) b += in.Phi0->nnz() * 16;
    if (in.PhiCert) b += in.PhiCert->nnz() * 16 + in.PhiCert->bytes();
    // per lane: X, Y on C^1 + W, W2 on Iso (136 D w), the echelon of R0 (<= 2 nnz d^0 entries of 8 B + the length-2D arrays), the
    // rho memo cap, the lane's TreeModule (dense d x d blocks per group element met: <= 64 d^2 B per term of the transport, bounded by
    // the memo cap again), G'' and H'' (r^2 (nops + 1) 4 B)
    const uint64_t memo = terms ? std::min<uint64_t>(opt.rho_memo_cap_bytes, terms * (d * d * 4 + 96)) : opt.rho_memo_cap_bytes;
    const uint64_t lane = 136 * D * w + 2 * c.d(0).nnz() * 16 + 2 * c.dim(0) * 16 + 2 * memo + r * r * (nops + 1) * 8 + d * d * 8;
    // (fix round 1, audit F8 a) a sweep runs up to `threads` lanes at once (the a-priori prefix or lanes_only in one sweep)
    b += std::max<uint64_t>(1, threads) * lane;
    // the bound: UpReal (16 B) X, W, W2, V on r columns + the exact majorant blocks; the exact r x r tables (G', adj, ceil beta_H and
    // the post-check bounds per op, ~ 48 B + the bits each)
    b += (3 * D + 2 * 14 * D + 3 * D) * r * 16;
    b += r * r * (2 + 2 * nops) * 96;
    // the residue planes and digits: nops r^2 x lanes x 2 x 4 B (lanes <= max_primes; sized at 512)
    b += nops * r * r * 512 * 8;
    return b;
}

bool FreeBlockLane(const FreeBlockInput& in, uint32_t q, const FreeBlockOptions& opt, std::vector<uint32_t>& out, std::string* why) {
    CheckInput(in);
    Shared S;
    InitShared(S, in, opt);
    const std::string w = BuildTerms(S);
    ENGINE_CHECK_MSG(w.empty(), ("FreeBlockLane: " + w).c_str());
    const LaneOut L = RunLane(S, q, 0);
    if (why) *why = L.why;
    out = L.F;
    return L.admissible;
}

namespace {
FreeBlockResult FreeBlockCrtImpl(const FreeBlockInput& in, const FreeBlockOptions& opt0, int32_t test_corrupt_lane);
}
FreeBlockResult FreeBlockCrt(const FreeBlockInput& in, const FreeBlockOptions& opt0) { return FreeBlockCrtImpl(in, opt0, -1); }
// TEST ONLY (N3): declared by the validation itself, not in the header: a faulty lane (+1 mod q on the first residue of op 0 in the
// `lane`-th admissible lane) for the a-priori post-check tooth.
FreeBlockResult FreeBlockCrtTestCorruptLane(const FreeBlockInput& in, const FreeBlockOptions& opt, int32_t lane) { return FreeBlockCrtImpl(in, opt, lane); }

namespace {
FreeBlockResult FreeBlockCrtImpl(const FreeBlockInput& in, const FreeBlockOptions& opt0, int32_t test_corrupt_lane) {
    const auto t0 = clk::now();
    FreeBlockOptions opt = opt0;
    FreeBlockResult res;
    CheckInput(in);
    const IsoTreeCarrierZ& car = *in.car;
    res.N = car.N();
    res.n = car.n();
    res.r = in.B->cols;
    ENGINE_CHECK_MSG(opt.lanes_only || in.Phi0 || (in.PhiCert && opt.use_apriori && opt.bound_with != BoundBasis::Phi0),
                     "FreeBlockCrt: Phi_0 (or, with the a-priori route, Phi_cert) is required for the bound");
    ENGINE_CHECK_MSG(opt.lanes_only || opt.use_apriori || opt.use_apost, "FreeBlockCrt: at least one proof must be enabled");
    auto refuse = [&](FreeBlockRefusal c, const std::string& text) -> FreeBlockResult {
        res.ok = false;
        res.refused = true;
        res.code = c;
        res.refusal = text.rfind("REFUSED", 0) == 0 ? text : std::string("REFUSED [") + FreeBlockRefusalName(c) + "] " + text;
        res.carrier_lanes_left = car.lane_count();
        res.t_total = since(t0);
        return res;
    };
    auto cell = [&]() { return "at (N, n) = (" + std::to_string(car.N()) + ", " + std::to_string(car.n()) + ")"; };
    if (opt.route == LaneRoute::Gpu && !engine::cuda::device_enabled())
        return refuse(FreeBlockRefusal::NoDevice, "the GPU lane route needs ENGINE_DEVICE=gpu and an available device");
    res.threads = opt.route == LaneRoute::Gpu ? 1u : (opt.threads ? std::min<uint32_t>(opt.threads, 64) : DefaultThreads());
    opt.threads = res.threads;
    opt.batch = std::max<uint32_t>(1, opt.batch);
    res.route = LaneRouteName(opt.route);
    const uint32_t r = res.r, nops = static_cast<uint32_t>(in.ps.size());
    // ---- (fix round 1, audit F1) rank 0: the empty free block, proven without a lane
    if (r == 0) {
        res.ops.resize(nops);
        for (uint32_t o = 0; o < nops; ++o) {
            FreeBlockOp& op = res.ops[o];
            op.p = in.ps[o];
            op.proven = !opt.lanes_only;
            op.proof = opt.lanes_only ? "" : "rank0";
            op.fnv = MatrixFnv({}, 0);
        }
        res.nonsingular_by = opt.lanes_only ? "" : "rank0";
        res.detG = BigInt(1);
        res.certified = !opt.lanes_only;
        res.gemm_path = engine::cpu::isa_path_name(engine::kernels::active_kernel_path());
        res.garner_path = engine::cpu::isa_path_name(engine::rns::garner_resolved_path(opt.garner_path));
        res.carrier_lanes_left = car.lane_count();
        res.ok = true;
        res.t_total = since(t0);
        return res;
    }
    // ---- (fix round 2, N1) the premises of BOTH proofs, on every proof route, exactly and before any lane: d^1 B = 0 (the lanes return
    // (Psi B)^{-1} Psi T~ B, which is the Hecke matrix F with T~ B = B F + d^0 x only for cocycles B) and Phi_0^T d^0 = 0 (so that
    // G' F = Phi_0^T T~ B and beta_H bounds G' F).  Phi_0 need NOT be the pair-0 functionals: the lanes use Psi from R0 / P alone, and any
    // Phi_0 killing d^0 with det G' != 0 gives the same exact F.  (The per-lane congruence G' F_q = Phi_0^T (T~ B mod q) was NOT added:
    // it costs nnz(Phi_0) r per operator per lane and follows from these two checks.)
    if (!opt.lanes_only) {
        if (std::string w = CheckCocycles(car, *in.B); !w.empty())
            return refuse(FreeBlockRefusal::Uncertified, "a premise fails " + cell() + ": " + w);
        if (in.Phi0)
            if (std::string w = CheckKillsD0(car, *in.Phi0, "Phi_0"); !w.empty())
                return refuse(FreeBlockRefusal::Uncertified, "a premise fails " + cell() + ": " + w);
    }
    // ---- (fix round 1, audit F3 a) the a-priori premise is part of the input and checked here
    if (!opt.lanes_only && opt.use_apriori) {
        if (!in.PhiCert)
            return refuse(FreeBlockRefusal::Uncertified, "the a-priori proof needs the Z-basis certificate Phi_cert (FreeBlockInput::PhiCert) " + cell() +
                                                             "; pass it, or set use_apriori = false");
        const auto tc = clk::now();
        Certificate C;
        // (Z3a, F10) reuse BuildFreeBasis' certificate when it is for exactly this (B, Phi_cert) (fingerprints), else run it here
        const Certificate* PC = in.PriorCert;
        const bool reuse = PC && PC->ok && PC->cocycles && PC->kills_d0 && PC->rank_ok && PC->car_N == car.N() && PC->car_n == car.n() &&
                           PC->car_dim0 == car.dim(0) && PC->car_dim1 == car.dim(1) && PC->car_dim2 == car.dim(2) &&
                           PC->fnv_B == PackedFnv(*in.B) && PC->fnv_Phi == PackedFnv(*in.PhiCert);
        std::string why;
        if (reuse) C = *PC;
        else why = CertificateReason(car, *in.B, *in.PhiCert, C);
        res.cert_reused = reuse;
        res.t_cert = since(tc);
        if (!why.empty()) return refuse(FreeBlockRefusal::Uncertified, "the certificate fails " + cell() + ": " + why);
        res.certified = true;
        res.cert_rank_q = C.rank_q;
        res.cert_primes = C.primes;
    }
    // (EXZ2-P3) the bound's functionals
    // Auto (EXZ2-P3, measured, EXZ2P3_report.md): Phi_0 while its fraction-free inverse is cheap -- r (bits Phi_0 + bits B) <= 20000,
    // i.e. the Bareiss entries stay below ~ 20000 bits -- else Phi_cert (no large inverse; the Phi_0 a-posteriori test beside it)
    const bool auto_cert = opt.bound_with == BoundBasis::Auto && in.Phi0 &&
                           uint64_t(r) * (uint64_t(in.Phi0->max_bits()) + in.B->max_bits()) > kAutoBoundBits;
    const bool bound_cert = !opt.lanes_only && opt.use_apriori && res.certified && in.PhiCert &&
                            (opt.bound_with == BoundBasis::PhiCert || auto_cert || !in.Phi0);
    const PackedZ* PhiB = bound_cert ? in.PhiCert : in.Phi0;
    res.bound_basis = PhiB ? (bound_cert ? "phi_cert" : "phi0") : "";
    Shared S;
    InitShared(S, in, opt);
    const std::string tw = BuildTerms(S);
    if (!tw.empty()) return refuse(FreeBlockRefusal::Terms, tw);
    uint64_t nterms = 0;
    for (const auto& T : S.T) nterms += T->terms_Z(1).size();
    res.rss_pred = PredictFreeBlockRss(in, opt, nterms);
    if (res.rss_pred > opt.rss_cap_bytes) {
        std::ostringstream o;
        o << "REFUSED [RssCeiling] the free-block lanes at (N, n) = (" << car.N() << ", " << car.n() << "): predicted " << res.rss_pred
          << " bytes > the ceiling " << opt.rss_cap_bytes << " bytes";
        return refuse(FreeBlockRefusal::RssCeiling, o.str());
    }
    const KernelPathGuard path_guard(opt.gemm_path);
    res.gemm_path = engine::cpu::isa_path_name(engine::kernels::active_kernel_path());
    res.garner_path = engine::cpu::isa_path_name(engine::rns::garner_resolved_path(opt.garner_path));
    const size_t E = size_t(nops) * r * r;
    // ---- the bound
    Bound Bd;
    if (PhiB) {   // lanes_only too (measurement: the bound's bits and the a-priori prime count without a proof)
        const auto tb = clk::now();
        Bd = ComputeBound(S, *PhiB, bound_cert, &res.t_inverse, &res.inverse_steps, bound_cert && opt.use_apost ? in.Phi0 : nullptr);
        car.release_exact_memo();   // (F7; deferred by the carrier while a majorant pass holds the memos, F9)
        res.t_bound = since(tb);
        res.Gp = Bd.Gp;
        if (Bd.have_inv) res.detG = Bd.det;
    }
    res.ops.resize(nops);
    for (uint32_t o = 0; o < nops; ++o) {
        res.ops[o].p = S.ps[o];
        if (PhiB) {
            res.ops[o].bits_beta_H = Bd.bits_beta_H[o];
            res.ops[o].bits_beta_F = Bd.bits_beta_F[o];
            res.ops[o].beta_H_ceil = Bd.beta_H_ceil[o];
            res.ops[o].apriori_rhs = Bd.rhs[o];
        }
    }
    // ---- (fix round 1, audit F2) det G' != 0 is established before a proof may run
    if (!opt.lanes_only) {
        if (Bd.inv == Bound::Inv::Singular)
            return refuse(FreeBlockRefusal::SingularG, "det G' = det(Phi_0^T B) = 0 (the fraction-free inverse) " + cell() +
                                                           ": Phi_0 is not a set of pair-0 functionals for B; no proof can hold");
        if (Bd.have_inv) {
            res.nonsingular_by = "inverse";
        } else {
            const uint32_t w = DetWitness(Bd.Gp, r);
            if (!w)
                return refuse(FreeBlockRefusal::SingularG, "det G' = det(Phi_0^T B) = 0, proven (every prime tried divides it and their product "
                                                           "exceeds the Hadamard bound) " + cell() + "; the a-posteriori stop would be vacuous");
            res.nonsingular_by = "witness";
            res.det_witness = w;
        }
    }
    const bool apriori_ok = !opt.lanes_only && opt.use_apriori && Bd.have_inv && res.certified;
    if (!opt.lanes_only && !apriori_ok && !opt.use_apost)
        return refuse(FreeBlockRefusal::NoProof, "the a-priori route is unavailable " + cell() +
                                                     " (the inverse of G' was skipped for size) and use_apost = false");
    // ---- the candidate order
    std::vector<uint32_t> cand;
    uint32_t cand_pos = 0, skipped_first = 0;
    auto next_candidate = [&]() -> uint32_t {
        while (true) {
            if (cand_pos >= cand.size()) cand = CachedCrtPrimes(static_cast<uint32_t>(std::max<size_t>(256, 2 * cand.size())));
            const uint32_t q = cand[cand_pos++];
            if ((6ull * car.N()) % q == 0) continue;
            if (skipped_first < opt.skip_first) {
                ++skipped_first;
                continue;
            }
            return q;
        }
    };
    // the a-priori prefix length per op, on the candidate order (an inadmissible prime only shifts the prefix by one)
    uint32_t ap_cap = 0, apost_first = 0;
    if (PhiB && Bd.have_inv && opt.use_apriori) {
        std::vector<uint32_t> cl = CachedCrtPrimes(4096), list;
        uint32_t sk = 0;
        for (uint32_t q : cl) {
            if ((6ull * car.N()) % q == 0) continue;
            if (sk < opt.skip_first) {
                ++sk;
                continue;
            }
            list.push_back(q);
        }
        const BigInt dabs = Bd.det.abs();
        for (uint32_t o = 0; o < nops; ++o) {
            BigInt Qp(1);
            uint32_t k = 0;
            while (k < list.size()) {
                Qp = Qp * BigInt(int64_t(list[k]));
                ++k;
                if (Qp * dabs > Bd.need_apriori[o]) break;
            }
            res.ops[o].primes_apriori = k;
            ap_cap = std::max(ap_cap, k);
        }
        // (EXZ2-P3) with the second a-posteriori test (Phi_0 beside Phi_cert) the a-priori prefix is no longer the expected stop: the first
        // sweep ends at the smallest L where an a-posteriori test can close for every operator (Q_L > min ceil beta_H of the two
        // functionals), then the sweeps grow by 4 batch up to the a-priori prefix.  Deterministic: every sweep end depends on the candidate
        // order and the bound only, never on `threads` (the checkpoints, hence primes_used, are the same at 1 and 16 threads).
        if (Bd.have_alt && opt.use_apost) {
            for (uint32_t o = 0; o < nops; ++o) {
                const BigInt& lim = Bd.beta_H_max_ceil_alt[o] < Bd.beta_H_max_ceil[o] ? Bd.beta_H_max_ceil_alt[o] : Bd.beta_H_max_ceil[o];
                BigInt Qp(1);
                uint32_t k = 0;
                while (k < list.size() && !(Qp > lim)) {
                    Qp = Qp * BigInt(int64_t(list[k]));
                    ++k;
                }
                apost_first = std::max(apost_first, k);
            }
        }
    }
    // ---- the lanes: sweeps (audit F8 a)
    std::vector<std::vector<uint32_t>> planes;   // per admissible lane: E residues (op-major)
    std::vector<BigInt> Qpref(1, BigInt(1));     // Qpref[L] = product of the first L lane primes
    uint32_t run_in_a_row_skipped = 0;
    FreeBlockRefusal sweep_refusal = FreeBlockRefusal::None;
    // run candidates until `target` admissible lanes are held (all of a sweep concurrently on `threads` workers); false: refused
    auto run_lanes = [&](uint32_t target) -> bool {
        while (res.primes.size() < target) {
            const size_t ran = res.primes.size() + res.skipped.size();
            if (ran >= opt.max_primes) {
                sweep_refusal = FreeBlockRefusal::NoProof;
                return false;
            }
            const uint32_t want = static_cast<uint32_t>(std::min<size_t>(target - res.primes.size(), opt.max_primes - ran));
            std::vector<uint32_t> qs(want);
            for (auto& q : qs) q = next_candidate();
            std::vector<LaneOut> outs(want);
            const auto tl = clk::now();
            const uint32_t base = static_cast<uint32_t>(ran);   // the global lane number (audit F11: the same on both paths)
            if (opt.threads <= 1 || want == 1) {
                for (uint32_t i = 0; i < want; ++i) outs[i] = RunLane(S, qs[i], base + i);
            } else {
                std::atomic<uint32_t> next{0};
                std::vector<std::string> errs(want);
                auto worker = [&]() {
                    while (true) {
                        const uint32_t i = next.fetch_add(1);
                        if (i >= want) return;
                        try {
                            outs[i] = RunLane(S, qs[i], base + i);
                        } catch (const std::exception& ex) {
                            errs[i] = ex.what();
                        }
                    }
                };
                std::vector<std::thread> pool;
                const uint32_t nt = std::min(opt.threads, want);
                for (uint32_t t = 0; t < nt; ++t) pool.emplace_back(worker);
                for (auto& th : pool) th.join();
                for (const auto& e : errs) ENGINE_CHECK_MSG(e.empty(), ("FreeBlockCrt: a lane threw: " + e).c_str());
            }
            res.t_lanes += since(tl);
            ++res.rounds;
            for (uint32_t i = 0; i < want; ++i) {
                res.lane_seconds_sum += outs[i].seconds;
                res.lane_seconds_max = std::max(res.lane_seconds_max, outs[i].seconds);
                res.lane_rho_memo_peak = std::max(res.lane_rho_memo_peak, outs[i].rho_bytes);
                if (!outs[i].admissible) {
                    res.skipped.push_back(qs[i]);
                    if (++run_in_a_row_skipped >= 64) {
                        sweep_refusal = FreeBlockRefusal::Inadmissible;
                        return false;
                    }
                    continue;
                }
                run_in_a_row_skipped = 0;
                if (test_corrupt_lane >= 0 && planes.size() == static_cast<size_t>(test_corrupt_lane) && E > 0)
                    outs[i].F[0] = (outs[i].F[0] + 1) % qs[i];   // TEST ONLY: a faulty lane
                res.primes.push_back(qs[i]);
                planes.push_back(std::move(outs[i].F));
                Qpref.push_back(Qpref.back() * BigInt(int64_t(qs[i])));
            }
        }
        return true;
    };
    auto sweep_refuse = [&]() {
        if (sweep_refusal == FreeBlockRefusal::Inadmissible)
            return refuse(FreeBlockRefusal::Inadmissible, "64 inadmissible primes in a row " + cell());
        return refuse(FreeBlockRefusal::NoProof, "no proof within max_primes = " + std::to_string(opt.max_primes) + " lanes " + cell() +
                                                     " (F is not integral in B, or the bound needs more lanes)");
    };
    if (opt.lanes_only) {
        if (!run_lanes(opt.lanes_only)) return sweep_refuse();
        res.lane_residues = std::move(planes);
        res.carrier_lanes_left = car.lane_count();
        res.ok = true;
        res.t_total = since(t0);
        return res;
    }
    // ---- the checkpoints: digits (once per plane), proofs at L = batch, 2 batch, .., and the a-priori prefix
    std::vector<std::vector<uint32_t>> digits;
    std::vector<bool> done(nops, false);
    auto lift = [&](uint32_t o, uint32_t L) {
        std::vector<BigInt> Fh(size_t(r) * r);
        for (uint32_t i = 0; i < r * r; ++i) {
            const size_t e = size_t(o) * r * r + i;
            BigInt x(0);
            for (size_t k = L; k-- > 0;) {
                x = x * BigInt(int64_t(res.primes[k]));
                if (digits[k][e]) x += BigInt(int64_t(digits[k][e]));
            }
            // the symmetric sign on the first L digit planes (the digits of (Q_L - 1)/2 are (q_k - 1)/2)
            bool neg = false;
            for (size_t k = L; k-- > 0;) {
                const uint32_t h = (res.primes[k] - 1) / 2;
                if (digits[k][e] != h) {
                    neg = digits[k][e] > h;
                    break;
                }
            }
            if (neg) x -= Qpref[L];
            Fh[i] = std::move(x);
        }
        return Fh;
    };
    uint32_t L_checked = 0;
    uint32_t target = apriori_ok ? std::max<uint32_t>(1, ap_cap) : opt.batch;
    const bool staged = apriori_ok && apost_first > 0 && apost_first < ap_cap;   // (EXZ2-P3) the Phi_cert route with the Phi_0 test
    if (staged) target = std::max<uint32_t>(1, apost_first);
    const BigInt dabs = Bd.have_inv ? Bd.det.abs() : BigInt(1);
    while (true) {
        if (!run_lanes(target)) return sweep_refuse();
        const uint32_t have = static_cast<uint32_t>(planes.size());
        // the new digit planes (audit F8 b: each plane once; the plan's row k depends on q_0 .. q_k only)
        const auto tg = clk::now();
        if (digits.size() < have) {
            const engine::rns::GarnerPlan plan(res.primes);
            while (digits.size() < have) digits.emplace_back(E, 0);
            std::vector<const uint32_t*> dp(have);
            for (uint32_t k = 0; k < have; ++k) dp[k] = digits[k].data();
            for (uint32_t k = L_checked; k < have; ++k)
                engine::rns::garner_step(plan, k, dp.data(), planes[k].data(), digits[k].data(), E, opt.garner_path);
        }
        res.t_garner += since(tg);
        std::vector<uint32_t> checkpoints;
        for (uint32_t c = (L_checked / opt.batch + 1) * opt.batch; c < have; c += opt.batch) checkpoints.push_back(c);
        checkpoints.push_back(have);
        const auto tf = clk::now();
        for (uint32_t L : checkpoints) {
            const BigInt& QL = Qpref[L];
            for (uint32_t o = 0; o < nops; ++o) {
                if (done[o]) continue;
                FreeBlockOp& op = res.ops[o];
                const bool apri = apriori_ok && QL * dabs > Bd.need_apriori[o];
                // audit F8 c: lift only when a test can close (Q_L <= max ceil beta_H excludes the a-posteriori stop)
                const bool apost_alt = !apri && opt.use_apost && Bd.have_alt && QL > Bd.beta_H_max_ceil_alt[o];
                const bool apost_possible = !apri && opt.use_apost && (QL > Bd.beta_H_max_ceil[o] || apost_alt);
                if (!apri && !apost_possible) continue;
                std::vector<BigInt> Fh = lift(o, L);
                bool apost = false;
                if (apri) {
                    // (fix round 1, audit F3 b) the post-check: |det G'| |F^_ij| <= ceil(sum_k |adj_ik| beta_H,kj) for every entry
                    for (uint32_t e = 0; e < r * r; ++e)
                        if (Fh[e].abs() * dabs > Bd.rhs[o][e]) {
                            std::ostringstream m;
                            m << "the a-priori stop at T_" << op.p << " " << cell() << " with " << L << " lanes: entry (" << e / r + 1 << ", " << e % r + 1
                              << ") has |det G'| |F^| of " << BitsOf(Fh[e].abs() * dabs) << " bits > the bound's " << BitsOf(Bd.rhs[o][e])
                              << " bits (a faulty lane or a false premise; not proven)";
                            return refuse(FreeBlockRefusal::BoundViolated, m.str());
                        }
                } else {
                    BigInt mx(0);
                    for (uint32_t i = 0; i < r; ++i)
                        for (uint32_t j = 0; j < r; ++j) {
                            BigInt s = Bd.beta_H_ceil[o][size_t(i) * r + j];
                            for (uint32_t k2 = 0; k2 < r; ++k2) {
                                const BigInt& g = Bd.Gp[size_t(i) * r + k2];
                                const BigInt& f = Fh[size_t(k2) * r + j];
                                if (!g.is_zero() && !f.is_zero()) s += g.abs() * f.abs();
                            }
                            if (s > mx) mx = s;
                        }
                    apost = QL > mx;
                    // (EXZ2-P3) the second test with Phi_0 (G'_alt F^ == H_alt mod Q, |G'_alt F^ - H_alt| < Q => G'_alt F^ = H_alt)
                    if (!apost && apost_alt) {
                        BigInt ma(0);
                        for (uint32_t i = 0; i < r; ++i)
                            for (uint32_t j = 0; j < r; ++j) {
                                BigInt s = Bd.beta_H_ceil_alt[o][size_t(i) * r + j];
                                for (uint32_t k2 = 0; k2 < r; ++k2) {
                                    const BigInt& g = Bd.Gp_alt[size_t(i) * r + k2];
                                    const BigInt& f = Fh[size_t(k2) * r + j];
                                    if (!g.is_zero() && !f.is_zero()) s += g.abs() * f.abs();
                                }
                                if (s > ma) ma = s;
                            }
                        apost = QL > ma;
                        if (apost) op.proof_detail = "apost_phi0";
                    }
                }
                if (apri || apost) {
                    done[o] = true;
                    op.proven = true;
                    op.proof = apri ? "apriori" : "apost";
                    op.primes_used = L;
                    op.F = std::move(Fh);
                    for (const auto& x : op.F) op.bits_F = std::max(op.bits_F, BitsOf(x));
                    op.fnv = MatrixFnv(ColsOfRowMajor(op.F, r), r);
                }
            }
        }
        res.t_lift += since(tf);
        L_checked = have;
        if (std::all_of(done.begin(), done.end(), [](bool b) { return b; })) break;
        target = have + opt.batch;
        if (staged && have < ap_cap) target = std::min<uint32_t>(ap_cap, have + 4 * opt.batch);   // a fixed step: independent of threads
    }
    res.carrier_lanes_left = car.lane_count();
    res.ok = true;
    res.t_total = since(t0);
    return res;
}
}  // namespace

// =================================================================================================================================
// small exact helpers
// =================================================================================================================================
const char* BoundBasisName(BoundBasis b) {
    switch (b) {
        case BoundBasis::Auto: return "auto";
        case BoundBasis::Phi0: return "phi0";
        case BoundBasis::PhiCert: return "phi_cert";
    }
    return "?";
}

bool InverseUnimodular(const std::vector<BigInt>& G, uint32_t r, std::vector<BigInt>& X, uint32_t* steps, uint32_t max_steps) {
    X.assign(size_t(r) * r, BigInt(0));
    if (steps) *steps = 0;
    if (r == 0) return true;
    ENGINE_CHECK_MSG(G.size() == size_t(r) * r, "InverseUnimodular: shape");
    if (max_steps == 0) {
        uint64_t hb = 0;
        for (uint32_t i = 0; i < r; ++i) {
            BigInt s(0);
            for (uint32_t j = 0; j < r; ++j) s += G[size_t(i) * r + j] * G[size_t(i) * r + j];
            hb += (s.bit_length() + 1) / 2;
        }
        max_steps = static_cast<uint32_t>(std::min<uint64_t>(hb / 29 + 4, 1u << 30));
    }
    // a prime with G invertible mod q, and G^{-1} mod q (Gauss-Jordan, row-major)
    uint32_t q = 0;
    std::vector<uint64_t> Gi;
    for (uint32_t cand : CachedCrtPrimes(8)) {
        std::vector<uint64_t> A(size_t(r) * r), I(size_t(r) * r, 0);
        for (size_t e = 0; e < A.size(); ++e) A[e] = G[e].mod_u32(cand);
        for (uint32_t i = 0; i < r; ++i) I[size_t(i) * r + i] = 1;
        bool ok = true;
        for (uint32_t c = 0; c < r && ok; ++c) {
            uint32_t piv = c;
            while (piv < r && !A[size_t(piv) * r + c]) ++piv;
            if (piv == r) {
                ok = false;
                break;
            }
            if (piv != c)
                for (uint32_t j = 0; j < r; ++j) {
                    std::swap(A[size_t(piv) * r + j], A[size_t(c) * r + j]);
                    std::swap(I[size_t(piv) * r + j], I[size_t(c) * r + j]);
                }
            const uint64_t inv = PowModU(A[size_t(c) * r + c], cand - 2, cand);
            for (uint32_t j = 0; j < r; ++j) {
                A[size_t(c) * r + j] = A[size_t(c) * r + j] * inv % cand;
                I[size_t(c) * r + j] = I[size_t(c) * r + j] * inv % cand;
            }
            for (uint32_t i = 0; i < r; ++i) {
                if (i == c) continue;
                const uint64_t f = A[size_t(i) * r + c];
                if (!f) continue;
                for (uint32_t j = 0; j < r; ++j) {
                    A[size_t(i) * r + j] = (A[size_t(i) * r + j] + (cand - f) * A[size_t(c) * r + j]) % cand;
                    I[size_t(i) * r + j] = (I[size_t(i) * r + j] + (cand - f) * I[size_t(c) * r + j]) % cand;
                }
            }
        }
        if (ok) {
            q = cand;
            Gi = std::move(I);
            break;
        }
    }
    if (!q) return false;
    // R = I (exact), X = 0; per step X_i = sym(G^{-1} R mod q), X += q^i X_i, R = (R - G X_i) / q (exact division, checked)
    std::vector<BigInt> R(size_t(r) * r, BigInt(0));
    for (uint32_t i = 0; i < r; ++i) R[size_t(i) * r + i] = BigInt(1);
    BigInt qi(1);
    const BigInt Q{int64_t(q)};
    const int64_t half = int64_t(q / 2);
    std::vector<int64_t> Xi(size_t(r) * r);
    std::vector<uint64_t> Rq(size_t(r) * r);
    for (uint32_t step = 0; step < max_steps; ++step) {
        bool zero = true;
        for (const auto& x : R)
            if (!x.is_zero()) {
                zero = false;
                break;
            }
        if (zero) {
            if (steps) *steps = step;
            return true;
        }
        for (size_t e = 0; e < R.size(); ++e) Rq[e] = R[e].mod_u32(q);
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                uint64_t s = 0;
                for (uint32_t k = 0; k < r; ++k) s = (s + Gi[size_t(i) * r + k] * Rq[size_t(k) * r + j]) % q;
                int64_t v = int64_t(s);
                if (v > half) v -= int64_t(q);
                Xi[size_t(i) * r + j] = v;
            }
        for (size_t e = 0; e < Xi.size(); ++e)
            if (Xi[e]) X[e] += qi * BigInt(Xi[e]);
        // R <- (R - G Xi) / q
        for (uint32_t i = 0; i < r; ++i)
            for (uint32_t j = 0; j < r; ++j) {
                BigInt s = R[size_t(i) * r + j];
                for (uint32_t k = 0; k < r; ++k) {
                    const int64_t x = Xi[size_t(k) * r + j];
                    if (x && !G[size_t(i) * r + k].is_zero()) s -= G[size_t(i) * r + k] * BigInt(x);
                }
                ENGINE_CHECK_MSG(s.mod_u32(q) == 0, "InverseUnimodular: the residual is not divisible by q");
                R[size_t(i) * r + j] = s / Q;
            }
        qi = qi * Q;
    }
    return false;
}
std::vector<BigInt> MatMulZ(const std::vector<BigInt>& A, const std::vector<BigInt>& B, uint32_t r) {
    std::vector<BigInt> C(size_t(r) * r, BigInt(0));
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t k = 0; k < r; ++k) {
            const BigInt& a = A[size_t(i) * r + k];
            if (a.is_zero()) continue;
            for (uint32_t j = 0; j < r; ++j)
                if (!B[size_t(k) * r + j].is_zero()) C[size_t(i) * r + j] += a * B[size_t(k) * r + j];
        }
    return C;
}

std::vector<BigInt> CharpolyBerkowitz(const std::vector<BigInt>& F, uint32_t r) {
    // Berkowitz: the characteristic polynomial det(x I - A) by Toeplitz products; vector high -> low
    std::vector<BigInt> c = {BigInt(1)};
    if (r == 0) return c;
    // c holds the charpoly of the leading k x k principal submatrix, high -> low; start with k = 1
    c = {BigInt(1), -F[0]};
    for (uint32_t k = 1; k < r; ++k) {
        // A_k = [[M, R], [S, a]] with M the leading k x k block, R = column k (rows < k), S = row k (cols < k), a = F[k][k]
        const BigInt a = F[size_t(k) * r + k];
        // Toeplitz column: t_0 = 1, t_1 = -a, t_{i+2} = -S M^i R (i = 0 .. k-1)
        std::vector<BigInt> t(k + 2, BigInt(0));
        t[0] = BigInt(1);
        t[1] = -a;
        std::vector<BigInt> v(k);
        for (uint32_t i = 0; i < k; ++i) v[i] = F[size_t(i) * r + k];   // R
        for (uint32_t i = 0; i < k; ++i) {
            BigInt s(0);
            for (uint32_t j = 0; j < k; ++j) s += F[size_t(k) * r + j] * v[j];   // S v
            t[i + 2] = -s;
            if (i + 1 < k) {
                std::vector<BigInt> nv(k, BigInt(0));
                for (uint32_t x = 0; x < k; ++x)
                    for (uint32_t y = 0; y < k; ++y) nv[x] += F[size_t(x) * r + y] * v[y];
                v = std::move(nv);
            }
        }
        // new c = T c, T lower-triangular Toeplitz (k + 2) x (k + 1) with first column t
        std::vector<BigInt> nc(k + 2, BigInt(0));
        for (uint32_t i = 0; i < k + 2; ++i)
            for (uint32_t j = 0; j <= std::min<uint32_t>(i, k); ++j) nc[i] += t[i - j] * c[j];
        c = std::move(nc);
    }
    return c;
}

}  // namespace freez
}  // namespace integral_descent
