// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZLocalSource -- slot-25 HZ-LOC: the level-1 lattice source and the census block invariants (see the header).

#include "HeckeZLocalSource.h"

#include "HeckeRingHeckeAlgebra.h"
#include "../../ModArithmetic/EngineCheck.h"

#include <cctype>
#include <algorithm>

namespace heckez {
namespace local {

namespace {

std::vector<uint64_t> Primes64(const std::vector<uint32_t>& P) { return std::vector<uint64_t>(P.begin(), P.end()); }

Mat FromRing(const hecke::ordinary::RingMat& A, uint64_t mod) {
    Mat M(A.a.size());
    for (size_t i = 0; i < A.a.size(); ++i) M[i] = A.a[i] % mod;
    return M;
}

}  // namespace

LatticeModule LatticeFromH1Level1(uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1) {
    LatticeModule M;
    M.N = 1;
    M.n = n;
    M.ell = ell;
    M.K = K;
    M.P = P;
    M.source = "h1_level1";
    const uint32_t Krun = std::max<uint32_t>(K, 2);
    if (!Zlk::Fits(ell, Krun)) {
        M.status = "refused(word_modulus)";
        M.detail = "ell^" + std::to_string(Krun) + " >= 2^63 (the module route runs at K >= 2)";
        return M;
    }
    ENGINE_CHECK_MSG(ell >= 5 && n >= 1, "HeckeZLocalSource: the level-1 H^1 route needs ell >= 5, n >= 1");
    const hecke_ext::ringalg::OpModule Om = hecke_ext::ringalg::RunModule(n, ell, Krun, Primes64(P));
    ENGINE_CHECK(Om.op.size() == P.size() && !Om.is_digit());
    M.h1_rank = Om.rank();
    M.K_max = Krun;
    M.word = Om.word;
    bool free = Om.rank() == b1;
    for (uint32_t a : Om.exps) free = free && a == Krun;
    const Zlk R = Zlk::Make(ell, K);
    if (free) {
        M.D = b1;
        for (const auto& A : Om.op) M.T.push_back(FromRing(A, R.q));
        M.is_lattice = true;
        M.regime = R.regime();
        M.ok = true;
        M.status = "ok";
        return M;
    }
    // a torsion prime: H^1 / ell over F_ell, A1_of_computed only
    const Zlk F = Zlk::Make(ell, 1);
    M.source = "h1_level1_torsion(A1_of_computed)";
    M.K = 1;
    M.D = Om.rank();
    for (const auto& A : Om.op) M.T.push_back(FromRing(A, F.q));
    M.is_lattice = false;
    M.regime = F.regime();
    M.ok = true;
    M.status = "torsion_prime";
    M.detail = "H^1 / ell has rank " + std::to_string(Om.rank()) + " > b1 = " + std::to_string(b1) + " (or a summand below K)";
    return M;
}

namespace {
LatticeModule LoadLatticeOn(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                            LatticeModule* h1_side, bool allow_bad_primes) {
    if (!Zlk::Fits(ell, K)) {
        LatticeModule M;
        M.N = N;
        M.n = n;
        M.ell = ell;
        M.K = K;
        M.P = P;
        M.source = N == 1 && ell >= 5 ? "h1_level1" : "export";
        M.status = "refused(word_modulus)";
        M.detail = "ell^K >= 2^63";
        return M;
    }
    if (N == 1 && ell >= 5 && n >= 1) {
        LatticeModule H = LatticeFromH1Level1(n, ell, K, P, b1);
        if (!H.ok || H.is_lattice) return H;
        if (h1_side) *h1_side = H;
        LatticeModule X = LatticeFromExport(N, n, ell, K, P);
        X.detail = (X.detail.empty() ? "" : X.detail + "; ") + "level-1 torsion prime: " + H.detail;
        return X;
    }
    return LatticeFromExport(N, n, ell, K, P, allow_bad_primes);
}
}  // namespace

LatticeModule LoadLattice(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                          LatticeModule* h1_side, bool allow_bad_primes) {
    std::string t_ell;
    const std::vector<uint32_t> Pr = ReadPrimes(N, ell, P, &t_ell);
    LatticeModule M = LoadLatticeOn(N, n, ell, K, Pr, b1, h1_side, allow_bad_primes);
    if (t_ell == "added" && M.status == "blocked(export)") {
        // the export does not lift T_ell at this (N, ell): the read runs on P, and says so
        // (HZ-REC schema audit F9) the refusal NAME only ("REFUSED [<Name>] ..."), never the free-text message
        std::string name;
        if (M.detail.compare(0, 9, "REFUSED [") == 0 && M.detail.find(']') != std::string::npos)
            name = M.detail.substr(9, M.detail.find(']') - 9);
        bool ok = !name.empty();
        for (char c : name) ok = ok && (std::isalnum(static_cast<unsigned char>(c)) || c == '_');
        M = LoadLatticeOn(N, n, ell, K, P, b1, h1_side, allow_bad_primes);
        t_ell = "omitted(export:" + (ok ? name : std::string("Unnamed")) + ")";
    }
    M.t_ell = t_ell;
    if (h1_side) h1_side->t_ell = t_ell;
    return M;
}

CellLocalRead ReadCellLocal(uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                            const ReadOptions& opt, uint32_t max_steps, LatticeModule* h1_side, const FreeSource* fz) {
    CellLocalRead C;
    if (fz) ENGINE_CHECK_MSG(fz->G != nullptr, "ReadCellLocal: the free source has no block");
    for (uint32_t step = 0; step <= max_steps; ++step) {
        if (step > 0 && !Zlk::Fits(ell, K)) {
            // (audit F2) the escalation stops at the word: the record word is the refusal, not open(precision)
            C.L.root = "refused(word_modulus)";
            C.L.C_T = "refused(word_modulus)";
            C.escalation = "refused(word_modulus) at K=" + std::to_string(K);
            break;
        }
        LatticeModule M;
        if (fz) {   // (EXW) the free route: F mod ell^K on the read set (T_ell by ReadPrimes, as LoadLattice)
            std::string t_ell;
            const std::vector<uint32_t> Pr = ReadPrimes(N, ell, P, &t_ell);
            M = LatticeFromFreeZ(N, n, ell, K, Pr, *fz->G, fz->ops);
            M.t_ell = t_ell;
            if (M.ok) ENGINE_CHECK_MSG(M.D == b1, "ReadCellLocal: the free block's rank differs from b1");
        } else {
            M = LoadLattice(N, n, ell, K, P, b1, h1_side, opt.allow_bad_primes);
        }
        if (!M.ok) {
            if (!C.read) {
                C.M = M;
            } else {
                // (audit F2) a refusal (or the export's open(precision) above its K_max) on the escalation path is carried into the
                // read: the K - 1 read keeps its other fields, its root and C_T are the module's word
                C.L.root = M.status;
                C.L.C_T = M.status;
                C.escalation = M.status + " at K=" + std::to_string(K) + " (" + M.detail + ")";
            }
            break;
        }
        C.M = M;
        C.L = ReadLocal(C.M, opt);
        C.read = true;
        if (C.L.root != "open(precision)") break;
        if (step == max_steps) {
            C.escalation = "open(precision) at the step cap K=" + std::to_string(K);
            break;
        }
        ++K;
    }
    return C;
}

Level1Blocks BlockInvariantsLevel1(uint32_t n, uint64_t ell, const std::vector<uint32_t>& P) {
    using namespace hecke_ext::ringalg;
    Level1Blocks out;
    const OpModule N2 = RunModule(n, ell, 2, Primes64(P));
    out.h1_rank = N2.rank();
    const Zlk F = Zlk::Make(ell, 1);
    for (const JointSystem& js : RationalSystems(N2)) {
        const Block B = Localize(N2, js.a);
        const OpModule loc = ReduceModEll(B.local);
        BlockInvariants bi;
        bi.system = js.a;
        bi.dim_E = loc.rank();
        const size_t D = loc.rank();
        if (D == 0) continue;
        std::vector<Mat> T;
        for (const auto& A : loc.op) T.push_back(FromRing(A, F.q));
        const Span A = AlgebraClosure(F, T, D);
        bi.length = A.length();
        // m = (T_p - a_p): m^1 = span{(T_p - a_p) b : b in A}, m^{i+1} = span{(T_p - a_p) x : x in m^i}
        std::vector<Mat> g;
        for (size_t i = 0; i < T.size(); ++i) g.push_back(MatMinusScalar(F, T[i], js.a[i] % ell, D));
        std::vector<std::vector<uint64_t>> cur = A.rows();
        uint32_t prev = A.length();
        for (uint32_t it = 0; prev > 0; ++it) {
            ENGINE_CHECK_MSG(it <= D * D + 1, "HeckeZLocalSource: the m-adic filtration did not terminate");
            Span nx(F, D * D);
            for (const auto& x : cur)
                for (const Mat& gi : g) nx.insert(MatMul(F, x, gi, D));
            const uint32_t len = nx.length();
            ENGINE_CHECK_MSG(len < prev, "HeckeZLocalSource: the block is not local at the system");
            bi.hilbert.push_back(prev - len);
            prev = len;
            cur = nx.rows();
        }
        bi.embdim = bi.hilbert.size() >= 2 ? bi.hilbert[1] : 0;
        // socle = {x in A : x (T_p - a_p) = 0 for every p}: dim A - rank of x -> (x g_p)_p
        const auto basis = A.rows();
        std::vector<uint64_t> big;
        const size_t cols = g.size() * D * D;
        for (const auto& b : basis) {
            for (const Mat& gi : g) {
                const Mat y = MatMul(F, b, gi, D);
                big.insert(big.end(), y.begin(), y.end());
            }
        }
        const uint32_t rk = static_cast<uint32_t>(SmithValuations(F, big, basis.size(), cols).size());
        bi.socle = static_cast<uint32_t>(basis.size()) - rk;
        bi.gorenstein = bi.socle == 1;
        out.blocks.push_back(bi);
    }
    return out;
}

}  // namespace local
}  // namespace heckez
