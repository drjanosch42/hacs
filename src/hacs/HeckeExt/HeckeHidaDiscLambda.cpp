// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-26 Z6 = H2) the per-block Fitting split at a non-Eisenstein residual system and the discriminant series D_F of [U] part 7
// (sec:u-H2: def:u-DF, lem:u-DF, lem:u-DFread, pr:u-H2); see HeckeHidaDiscLambda.h.
#include "HeckeHidaDiscLambda.h"

#include <algorithm>
#include <exception>
#include <map>
#include <utility>

#include "../../ModArithmetic/EngineCheck.h"
#include "../Util/ParallelFor.h"

namespace hecke_ext {
namespace disclambda {

namespace L = hecke_ext::lambda;
using L::ZMod;

namespace {

size_t Cols(const Mat& B) { return B.empty() ? 0 : B[0].size(); }

// The roots of f (coefficients low -> high, any modulus) in F_ell with their multiplicities, ascending (brute force; ell <= 10^4).
std::vector<std::pair<uint64_t, uint32_t>> RootsModEll(const Vec& f0, uint64_t ell) {
    std::vector<uint64_t> f;
    for (u128 c : f0) f.push_back(static_cast<uint64_t>(c % ell));
    while (!f.empty() && f.back() == 0) f.pop_back();
    std::vector<std::pair<uint64_t, uint32_t>> out;
    if (f.size() < 2) return out;
    for (uint64_t x = 0; x < ell; ++x) {
        std::vector<uint64_t> g = f;
        uint32_t mult = 0;
        while (g.size() >= 2) {
            // synthetic division by (Y - x): quotient q, remainder r
            std::vector<uint64_t> q(g.size() - 1);
            uint64_t acc = 0;
            for (size_t i = g.size(); i-- > 0;) {
                acc = static_cast<uint64_t>((static_cast<u128>(acc) * x + g[i]) % ell);
                if (i > 0) q[i - 1] = acc;
            }
            if (acc != 0) break;   // remainder g(x) != 0
            ++mult;
            g.swap(q);
        }
        if (mult) out.push_back({x, mult});
    }
    return out;
}

// The matrix of A (s x s on e S_k) on the A-stable summand with column basis B (s x r), mod R.M.
Mat RestrictTo(const Mat& A, const Mat& B, const ZMod& R) {
    const auto bd = L::BasisData(B, R);
    return L::Restrict(A, B, bd.first, bd.second, R);
}

// The generalised kernel of A - rho (A r x r on the current block): a basis (r x r') in the block's coordinates.
Mat GenKernel(const Mat& A, uint64_t rho, const ZMod& R) {
    return L::FittingSplit(L::MAddScaledIdent(A, static_cast<u128>(rho), R), R).first;
}

bool ScalarModEll(const Mat& A, uint64_t rho, uint64_t ell) {
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[i].size(); ++j)
            if (static_cast<uint64_t>(A[i][j] % ell) != (i == j ? rho % ell : 0)) return false;
    return true;
}

std::vector<std::string> CutOrder(const L::FormsSpace& S, const std::vector<uint64_t>& primes, bool need_two) {
    std::vector<uint64_t> ps(primes);
    std::sort(ps.begin(), ps.end());
    ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
    ENGINE_CHECK_MSG(!need_two || (!ps.empty() && ps.front() == 2), "HeckeHidaDiscLambda: the prime list must contain 2");
    std::vector<std::string> ops;
    for (uint64_t p : ps) ops.push_back("T" + std::to_string(p));
    ops.push_back("U");
    for (const auto& n : ops)
        ENGINE_CHECK_MSG(S.mats.count(n) == 1, ("HeckeHidaDiscLambda: operator " + n + " not in the forms data").c_str());
    return ops;
}

}  // namespace

uint64_t ResidualSystem::at(const std::string& op) const {
    for (size_t i = 0; i < ops.size(); ++i)
        if (ops[i] == op) return res[i];
    ENGINE_CHECK_MSG(false, ("ResidualSystem: no residue for " + op).c_str());
    return 0;
}

u128 XMod(uint64_t ell, uint64_t k, const ZMod& R) { return R.sub(R.pow(static_cast<u128>(ell + 1) % R.M, k), 1 % R.M); }

// ---------------------------------------------------------------------------
// step 2: the non-reduced residual systems at one weight
// ---------------------------------------------------------------------------
SystemSearch FindNonReducedSystems(const L::FormsWeight& w, const std::vector<uint64_t>& primes) {
    const uint64_t ell = w.ell;
    const ZMod R(ell, w.W), F(ell, 1);
    const L::FormsSpace& S = w.S;
    ENGINE_CHECK_MSG(S.s >= 1 && w.stable_ok && S.comm_ok, "FindNonReducedSystems: e S_k empty, or the forms data not stable / commuting");
    const std::vector<std::string> ops = CutOrder(S, primes, true);
    SystemSearch out;
    struct Piece {
        Mat B;
        std::vector<uint64_t> res;
    };
    std::vector<Piece> pieces;
    const Mat& T2 = S.mats.at("T2");
    for (const auto& rm : RootsModEll(L::Charpoly(T2, F), ell)) {
        if (rm.second < 2) continue;
        ++out.candidates;
        Mat B = GenKernel(T2, rm.first, R);
        ENGINE_CHECK_MSG(Cols(B) == rm.second, "FindNonReducedSystems: generalised T_2-eigenspace rank != multiplicity");
        pieces.push_back(Piece{B, {rm.first}});
    }
    for (size_t oi = 1; oi < ops.size(); ++oi) {
        std::vector<Piece> next;
        for (const Piece& pc : pieces) {
            const size_t r = Cols(pc.B);
            const Mat A = RestrictTo(S.mats.at(ops[oi]), pc.B, R);
            uint32_t seen = 0;
            for (const auto& rm : RootsModEll(L::Charpoly(A, F), ell)) {
                seen += rm.second;
                const Mat K = GenKernel(A, rm.first, R);
                const Mat Bn = L::MatMul(pc.B, K, R);
                ENGINE_CHECK_MSG(Cols(Bn) == rm.second, "FindNonReducedSystems: generalised eigenspace rank != multiplicity");
                if (rm.second < 2) {
                    ++out.refused;
                    out.notes.push_back("rbar2=" + std::to_string(pc.res[0]) + ": " + ops[oi] + " separates a rank-1 piece (reduced)");
                    continue;
                }
                std::vector<uint64_t> res = pc.res;
                res.push_back(rm.first);
                next.push_back(Piece{Bn, res});
            }
            if (seen < r) {
                ++out.refused;
                out.notes.push_back("rbar2=" + std::to_string(pc.res[0]) + ": " + ops[oi] + " has no F_ell eigenvalue on a rank-" +
                                    std::to_string(r - seen) + " part (conjugate over an extension of F_ell: reduced)");
            }
        }
        pieces.swap(next);
    }
    for (const Piece& pc : pieces) {
        bool nonscalar = false;
        for (size_t oi = 0; oi < ops.size(); ++oi)
            nonscalar = nonscalar || !ScalarModEll(RestrictTo(S.mats.at(ops[oi]), pc.B, R), pc.res[oi], ell);
        if (!nonscalar) {
            ++out.refused;
            out.notes.push_back("rbar2=" + std::to_string(pc.res[0]) + ": every operator scalar mod ell on a rank-" +
                                std::to_string(Cols(pc.B)) + " piece (reduced coincidence)");
            continue;
        }
        ResidualSystem sys;
        sys.ell = ell;
        sys.k = w.k;
        sys.ops = ops;
        sys.res = pc.res;
        sys.rank = static_cast<uint32_t>(Cols(pc.B));
        out.systems.push_back(sys);
    }
    std::sort(out.systems.begin(), out.systems.end(),
              [](const ResidualSystem& a, const ResidualSystem& b) { return a.res < b.res; });
    return out;
}

// ---------------------------------------------------------------------------
// step 3: the block at one weight (M3), certified
// ---------------------------------------------------------------------------
BlockAt ResidualBlock(const L::FormsWeight& w, const std::vector<uint64_t>& primes, const ResidualSystem& sys) {
    const uint64_t ell = w.ell;
    ENGINE_CHECK_MSG(sys.ell == ell, "ResidualBlock: the residual system is at another ell");
    const ZMod R(ell, w.W);
    const L::FormsSpace& S = w.S;
    ENGINE_CHECK_MSG(S.s >= 1 && w.stable_ok && S.comm_ok, "ResidualBlock: e S_k empty, or the forms data not stable / commuting");
    const std::vector<std::string> ops = CutOrder(S, primes, false);
    Mat B = L::Ident(S.s);
    for (const auto& op : ops) {
        if (Cols(B) == 0) break;
        const Mat A = RestrictTo(S.mats.at(op), B, R);
        B = L::MatMul(B, GenKernel(A, sys.at(op), R), R);
    }
    const size_t r = Cols(B);
    ENGINE_CHECK_MSG(r == 2, ("ResidualBlock REFUSED at (ell, k) = (" + std::to_string(ell) + ", " + std::to_string(w.k) + "): rank " +
                              std::to_string(r) + " != 2 (wrong residual system, or a collision on the cut primes: lem:u-DF(1))")
                                 .c_str());
    BlockAt b;
    b.k = w.k;
    b.rank = static_cast<uint32_t>(r);
    for (const auto& name : S.names) b.ops[name] = RestrictTo(S.mats.at(name), B, R);
    for (const auto& op : ops)
        ENGINE_CHECK_MSG(L::NilpotentModEll(L::MAddScaledIdent(b.ops.at(op), static_cast<u128>(sys.at(op)), R), ell),
                         ("ResidualBlock: " + op + " - lambda_bar not nilpotent mod ell on the block (certificate)").c_str());
    return b;
}

std::string SecondGenerator(const BlockAt& b, const std::vector<uint64_t>& primes, uint64_t ell) {
    uint64_t first = 0;
    std::vector<uint64_t> ps(primes);
    std::sort(ps.begin(), ps.end());
    for (uint64_t q : ps) {
        if (q == 2) continue;
        const auto it = b.ops.find("T" + std::to_string(q));
        if (it == b.ops.end()) continue;
        if (!first) first = q;
        if (L::GeneratesModEll(it->second, ell)) return "T" + std::to_string(q);
    }
    ENGINE_CHECK_MSG(first != 0, "SecondGenerator: no prime other than 2 in the forms data");
    return "T2+T" + std::to_string(first);
}

Mat GeneratorMatrix(const BlockAt& b, const std::string& spec, const ZMod& R) {
    const size_t plus = spec.find('+');
    if (plus == std::string::npos) return b.ops.at(spec);
    const Mat& A = b.ops.at(spec.substr(0, plus));
    const Mat& C = b.ops.at(spec.substr(plus + 1));
    Mat S = A;
    for (size_t i = 0; i < S.size(); ++i)
        for (size_t j = 0; j < S[i].size(); ++j) S[i][j] = R.add(A[i][j] % R.M, C[i][j] % R.M);
    return S;
}

Quad ReadQuad(const Mat& t, const ZMod& R) {
    ENGINE_CHECK_MSG(t.size() == 2, "ReadQuad: the block is not of rank 2");
    const Vec chi = L::Charpoly(t, R);   // [nu, -tau, 1]
    Quad q;
    q.nu = chi[0] % R.M;
    q.tau = R.neg(chi[1] % R.M);
    q.disc = R.sub(R.mul(q.tau, q.tau), R.mul(4 % R.M, q.nu));
    q.v_disc = R.val(q.disc, R.W);
    q.gen = L::GeneratesModEll(t, R.ell);
    return q;
}

// ---------------------------------------------------------------------------
// the Lambda stage (thm:u-interp, lem:u-DFread)
// ---------------------------------------------------------------------------
Vec Trunc33(uint64_t ell, uint32_t k0, const Vec& values) {
    ENGINE_CHECK_MSG(values.size() >= 5, "Trunc33: five values");
    std::vector<uint32_t> ks;
    for (uint32_t j = 0; j < 5; ++j) ks.push_back(k0 + j * static_cast<uint32_t>(ell - 1));
    return L::InterpolateTruncation(ell, ks, Vec(values.begin(), values.begin() + 5), 5, 3, 3, k0);
}

uint32_t PredictedV(const DiscTruncation& d, uint64_t ell, uint64_t k) {
    if (d.cas != 'A') return 1;
    const uint64_t l2 = ell * ell;
    if ((k % ell) != (d.sF_mod_l2 % ell)) return 1;
    if ((k % l2) != d.sF_mod_l2) return 2;
    return 3;
}

DiscTruncation ReadDisc(uint64_t ell, uint32_t k0, const Vec& v) {
    const size_t N = v.size();
    ENGINE_CHECK_MSG(N == 5 || N == 6, "ReadDisc: five values (plus the held-out sixth)");
    const ZMod R5(ell, 5), R3(ell, 3), R2(ell, 2);
    std::vector<uint32_t> ks;
    for (uint32_t j = 0; j < N; ++j) ks.push_back(k0 + j * static_cast<uint32_t>(ell - 1));
    DiscTruncation d;
    d.e = Trunc33(ell, k0, v);
    d.v_e0 = R3.val(d.e[0] % R3.M, 3);
    // the (2, 2) subset self-check: the truncation from j = 2, 3, 4 (N = 3 = K + m - 1, Kp = 5), centred at x_{k0}.  A code check,
    // not an error detector: once the five-node interpolant is ell-integral it agrees for ANY values (Z6_code_audit.md F1); node errors
    // are caught by the integrality refusal above and the held-out weight below.
    d.sub = L::InterpolateTruncation(ell, {ks[2], ks[3], ks[4]}, {v[2], v[3], v[4]}, 5, 2, 2, k0);
    d.subset_ok = d.sub.size() == 2 && d.sub[0] % R2.M == d.e[0] % R2.M && d.sub[1] % R2.M == d.e[1] % R2.M;
    // held-out (thm:u-interp(3)): the full Newton polynomial of the five, at x_{k5}, mod ell^min(K', N) = ell^5
    if (N == 6) {
        std::vector<nf::BigInt> xs;
        for (size_t j = 0; j < 5; ++j) xs.push_back(L::XOf(ell, ks[j], 5 + 40));
        const Vec r = L::NewtonMod(xs, Vec(v.begin(), v.begin() + 5), R5);
        d.heldout = L::PEval(r, XMod(ell, ks[5], R5), R5) == v[5] % R5.M ? 1 : 0;
    }
    const u128 e0 = d.e[0] % R3.M, e1 = d.e[1] % R3.M, e2 = d.e[2] % R3.M;
    if (e1 % ell != 0) {
        d.cas = 'A';
        d.lambda_word = "1";
        // Newton for the root in ell Z_ell (Hensel: e0 in ell Z, e1 a unit), mod ell^3
        u128 z = 0;
        for (int it = 0; it < 6; ++it) {
            const u128 f = R3.add(e0, R3.add(R3.mul(e1, z), R3.mul(e2, R3.mul(z, z))));
            const u128 fp = R3.add(e1, R3.mul(R3.mul(2 % R3.M, e2), z));
            z = R3.sub(z, R3.mul(f, R3.inv(fp)));
        }
        ENGINE_CHECK_MSG(z % ell == 0 && R3.add(e0, R3.add(R3.mul(e1, z), R3.mul(e2, R3.mul(z, z)))) == 0,
                         "ReadDisc: Newton did not converge to a root in ell Z_ell");
        d.zeta = z;
        d.xi = R3.add(XMod(ell, k0, R3), z);
        // s_F mod ell^2: u^s - 1 == xi mod ell^3 (u = 1 + ell has order ell^2 mod ell^3)
        const uint64_t l2 = ell * ell;
        u128 cur = 1 % R3.M;
        bool found = false;
        for (uint64_t s = 0; s < l2; ++s) {
            if (R3.sub(cur, 1 % R3.M) == d.xi) {
                d.sF_mod_l2 = s;
                found = true;
                break;
            }
            cur = R3.mul(cur, static_cast<u128>(ell + 1) % R3.M);
        }
        ENGINE_CHECK_MSG(found, "ReadDisc: xi is not of the form u^s - 1 mod ell^3 (xi not in ell Z_ell)");
        for (uint64_t j = 0; j < ell && !d.rung2; ++j) {
            const uint64_t k = k0 + j * (ell - 1);
            if (k >= 3 && k % ell == d.sF_mod_l2 % ell) d.rung2 = k;
        }
        for (uint64_t j = 0; j < l2 && !d.rung3; ++j) {
            const uint64_t k = k0 + j * (ell - 1);
            if (k >= 3 && k % l2 == d.sF_mod_l2) d.rung3 = k;
        }
        ENGINE_CHECK_MSG(d.rung2 && d.rung3, "ReadDisc: no rung weight (CRT)");
    } else {
        d.cas = 'B';
        d.lambda_word = e2 % ell != 0 ? "2" : "undecided";
    }
    // the internal v-reading at the nodes (a code check: at the nodes it follows from the interpolation)
    d.reading_ok = true;
    for (size_t j = 0; j < N; ++j) {
        const uint32_t vj = R5.val(v[j] % R5.M, 5);
        const uint32_t p = PredictedV(d, ell, ks[j]);
        d.reading_ok = d.reading_ok && (p < 3 ? vj == p : vj >= 3);
    }
    return d;
}

bool RingConsistent(uint64_t ell, const Vec& t, const Vec& n, const Vec& D) {
    const ZMod R3(ell, 3);
    if (t.size() < 3 || n.size() < 3 || D.size() < 3) return false;
    for (size_t i = 0; i < 3; ++i) {
        u128 c = 0;
        for (size_t a = 0; a <= i; ++a) c = R3.add(c, R3.mul(t[a] % R3.M, t[i - a] % R3.M));
        c = R3.sub(c, R3.mul(4 % R3.M, n[i] % R3.M));
        if (c != D[i] % R3.M) return false;
    }
    return true;
}

bool UnitSquareRatio(uint64_t ell, u128 e0, u128 e0p) {
    if (e0 % ell != 0 || e0p % ell != 0) return false;
    const uint64_t a = static_cast<uint64_t>((e0 / ell) % ell), b = static_cast<uint64_t>((e0p / ell) % ell);
    if (a == 0 || b == 0) return false;
    const ZMod F(ell, 1);
    const u128 ratio = F.mul(b, F.inv(a));
    return F.pow(ratio, (ell - 1) / 2) == 1;
}

// ---------------------------------------------------------------------------
// the parallel driver (index-addressed merge; never bound to a console verb)
// ---------------------------------------------------------------------------
std::vector<L::FormsWeight> ComputeFormsParallel(const std::vector<FormsJob>& jobs, unsigned threads) {
    std::vector<L::FormsWeight> out(jobs.size());
    std::vector<std::exception_ptr> err(jobs.size());
    oc_util::ParallelForBlocks(
        jobs.size(), threads == 0 ? 1u : threads,
        [&](size_t i) {
            try {
                out[i] = L::ComputeFormsWeight(jobs[i].ell, jobs[i].k, jobs[i].W, jobs[i].primes);
            } catch (...) {
                err[i] = std::current_exception();
            }
        },
        1, 16);
    for (const auto& e : err)
        if (e) std::rethrow_exception(e);
    return out;
}

}  // namespace disclambda
}  // namespace hecke_ext
