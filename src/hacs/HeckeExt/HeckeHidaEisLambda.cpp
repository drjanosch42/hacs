// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (slot-25 H1) the finite-weight readings of [U] part 6; see HeckeHidaEisLambda.h.
#include "HeckeHidaEisLambda.h"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>

#include "../../ModArithmetic/EngineCheck.h"

namespace hecke_ext {
namespace eislambda {

namespace L = hecke_ext::lambda;
using L::Mat;
using L::ZMod;

namespace {

bool IsPrime64(uint64_t p) {
    if (p < 2) return false;
    for (uint64_t d = 2; d * d <= p; ++d)
        if (p % d == 0) return false;
    return true;
}

Mat Reduced(const Mat& A, const ZMod& R) {
    Mat B = A;
    for (auto& row : B)
        for (auto& x : row) x %= R.M;
    return B;
}

Mat MinusScalar(const Mat& A, u128 c, const ZMod& R) {   // A - c I
    Mat B = Reduced(A, R);
    for (size_t i = 0; i < B.size(); ++i) B[i][i] = R.sub(B[i][i], c % R.M);
    return B;
}

// t generates the algebra Z/ell[t] = the block's algebra mod ell iff 1, t, ..., t^{r-1} are independent mod ell (dimension r = rank,
// the block's algebra mod ell being of dimension r: Lemma H0(3) asks for generation of R_k / ell R_k, which has dimension r)
bool GeneratesModEll(const Mat& t, uint64_t ell) {
    const size_t r = t.size();
    if (r == 0) return true;
    const ZMod F(ell, 1);
    Mat P = L::Ident(r);
    const Mat tl = Reduced(t, F);
    Mat V(r * r, Vec(r, 0));
    for (size_t i = 0; i < r; ++i) {
        for (size_t a = 0; a < r; ++a)
            for (size_t b = 0; b < r; ++b) V[a * r + b][i] = P[a][b] % ell;
        P = L::MatMul(P, tl, F);
    }
    return L::IndepColsModEll(V, r, ell).size() == r;
}

// G(x_k) = -(1 - ell^{k-1}) B_k / k mod ell^R.M ([U, lem:u-twistvalues])
u128 GOf(const ZMod& R, uint64_t ell, uint32_t k, u128 bk) {
    return R.neg(R.mul(R.sub(1 % R.M, R.pow(ell % R.M, k - 1)), bk % R.M));
}

double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

std::vector<uint64_t> SturmPrimes(uint64_t ell, uint32_t k, const std::vector<uint64_t>& extra) {
    const uint64_t bound = std::max<uint64_t>(k / 12 + 1, 7);
    std::vector<uint64_t> out;
    for (uint64_t p = 2; p <= bound; ++p)
        if (p != ell && IsPrime64(p)) out.push_back(p);
    for (uint64_t p : extra) {
        ENGINE_CHECK_MSG(p != ell && IsPrime64(p), "SturmPrimes: an extra operator is not a prime != ell");
        out.push_back(p);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

BlockReading ReadBlock(uint64_t ell, uint32_t k, uint32_t W, const std::vector<uint64_t>& extra_primes) {
    const double t0 = Now();
    ENGINE_CHECK_MSG(ell >= 5 && IsPrime64(ell) && k >= 4 && k % 2 == 0, "ReadBlock: ell >= 5 prime, k >= 4 even");
    ENGINE_CHECK_MSG(k % (ell - 1) != 0, "ReadBlock: (ell - 1) | k is the class a = 0 (no Eisenstein congruence, [U, rmk:u-classzero])");
    ENGINE_CHECK_MSG(W >= 2 && k - 1 >= W, "ReadBlock: 2 <= W <= k - 1");
    const std::vector<uint64_t> primes = SturmPrimes(ell, k, extra_primes);
    BlockReading b = ReadBlockFrom(L::ComputeFormsWeight(ell, k, W, primes), primes);   // ENGINE_CHECK: ell^W < 2^84
    b.secs = Now() - t0;
    return b;
}

BlockReading ReadBlockFrom(const lambda::FormsWeight& fw, const std::vector<uint64_t>& primes) {
    const double t0 = Now();
    const uint64_t ell = fw.ell;
    const uint32_t k = fw.k, W = fw.W;
    ENGINE_CHECK_MSG(k % (ell - 1) != 0 && W >= 2, "ReadBlockFrom: class a = 0 or W < 2");
    BlockReading b;
    b.ell = ell;
    b.k = k;
    b.W = W;
    b.primes = primes;
    b.fw = fw;
    ENGINE_CHECK_MSG(b.fw.stable_ok && b.fw.M.comm_ok && b.fw.S.comm_ok, "ReadBlock: the forms data are not stable / commuting");
    const ZMod R(ell, W);
    b.s_M = b.fw.M.s;
    const L::EisBlock eb = L::EisensteinBlock(b.fw, b.primes);
    b.g = eb.rank_S;
    b.rank_M = eb.rank_M;
    // [U, sec:u-eislambda]: rank of T(a)_m = 1 + g_a; on e M_k the block carries the Eisenstein series once more
    ENGINE_CHECK_MSG(b.rank_M == b.g + 1, "ReadBlock: rank of the Eisenstein block on e M_k is not 1 + its rank on e S_k");
    b.e2 = b.fw.eis.at("T2") % R.M;
    // the Bernoulli side: B_k / k mod ell^W, and B_k / k, B_{k+ell-1} / (k+ell-1) mod ell^2 (D3', K9)
    {
        const auto bk = L::BernoulliOverK(ell, W, {k});
        const u128 G = GOf(R, ell, k, bk.at(k));
        b.v_Bk = G == 0 ? -1 : static_cast<int32_t>(R.val(G, W));   // 1 - ell^{k-1} is a unit: v(G(x_k)) = v(B_k/k)
        const ZMod R2(ell, 2);
        const auto b2 = L::BernoulliOverK(ell, 2, {k, k + static_cast<uint32_t>(ell) - 1});
        b.Bk_l2 = b2.at(k) % R2.M;
        b.Gk_l2 = GOf(R2, ell, k, b2.at(k));
        b.Gnext_l2 = GOf(R2, ell, k + static_cast<uint32_t>(ell) - 1, b2.at(k + static_cast<uint32_t>(ell) - 1));
        const u128 d = R2.sub(b.Gk_l2, b.Gnext_l2);
        b.lambda_one = b.Gk_l2 % ell == 0 && d != 0 && R2.val(d, 2) == 1;
    }
    // per q: T_q - e_q on the cuspidal block
    for (uint64_t q : b.primes) {
        const std::string nm = "T" + std::to_string(q);
        if (b.g == 0) {   // no cuspidal block at m_Eis (a regular pair at this weight): nothing to read
            b.v_det.push_back(-1);
            b.gen.push_back(false);
            continue;
        }
        const Mat Y = MinusScalar(eb.S.at(nm), b.fw.eis.at(nm), R);
        const Vec cp = L::Charpoly(Y, R);   // det(Z - Y): its constant term is (-1)^g det(Y)
        b.v_det.push_back(cp[0] == 0 ? -1 : static_cast<int32_t>(R.val(cp[0], W)));
        b.gen.push_back(GeneratesModEll(Y, ell));
        if (q == 2) b.f2 = L::Charpoly(Reduced(eb.S.at(nm), R), R);
    }
    if (b.g >= 1) {
        if (b.g == 1) {
            int32_t mn = -1;
            for (int32_t v : b.v_det)
                if (v >= 0 && (mn < 0 || v < mn)) mn = v;
            b.L_forms = mn;
        }
        if (b.v_Bk >= 0)
            for (size_t i = 0; i < b.primes.size(); ++i)
                if (b.gen[i] && b.v_det[i] == b.v_Bk) {
                    b.I0_q = b.primes[i];
                    break;
                }
    }
    b.secs = Now() - t0;
    return b;
}

std::string K9Line(const BlockReading& b, bool* all_ok) {
    const uint64_t l = b.ell;
    const ZMod R2(l, 2), F(l, 1);
    std::vector<u128> f;
    for (u128 c : b.f2) f.push_back(c % R2.M);
    const u128 e = b.e2 % R2.M;
    std::vector<std::string> checks;
    bool ok = b.g >= 1 && f.size() == b.g + 1;
    // (K9-a) f_k == (x - e_k)^g mod ell
    if (ok) {
        Vec ref{1};
        for (uint32_t i = 0; i < b.g; ++i) {
            Vec nx(ref.size() + 1, 0);
            for (size_t j = 0; j < ref.size(); ++j) {
                nx[j + 1] = F.add(nx[j + 1], ref[j]);
                nx[j] = F.add(nx[j], F.mul(F.neg(e % l), ref[j]));
            }
            ref = nx;
        }
        bool a = true;
        for (size_t j = 0; j < f.size(); ++j) a = a && f[j] % l == ref[j] % l;
        if (a) checks.push_back("eis_root");
        ok = ok && a;
    }
    // (K9-b) v(f_k(e_k)) = v(B_k / k), both read mod ell^2
    u128 fe = 0;
    if (!f.empty()) {
        for (size_t j = f.size(); j-- > 0;) fe = R2.add(R2.mul(fe, e), f[j]);
        const uint32_t vf = R2.val(fe, 2), vb = R2.val(b.Bk_l2, 2);
        const bool bb = vf == vb && vb < 2;
        if (bb) checks.push_back("v_f_e=v_Bk");
        ok = ok && bb;
    }
    // (K9-c) g = 2: v(disc f_k) = 1
    if (b.g == 2 && f.size() == 3) {
        const u128 disc = R2.sub(R2.mul(f[1], f[1]), R2.mul(4 % R2.M, f[0]));
        const bool c = R2.val(disc, 2) == 1;
        if (c) checks.push_back("v_disc=1");
        ok = ok && c;
    }
    // (K9-d) the lambda-test residue
    const u128 d = R2.sub(b.Gk_l2, b.Gnext_l2);
    const u128 lres = d % l == 0 ? (d / l) % l : l;   // l = "not divisible": the test fails
    {
        const bool dd = lres != 0 && lres != l;
        if (dd) checks.push_back("lambda_test");
        ok = ok && dd;
    }
    std::ostringstream o;
    o << "HIDA_K9 ell=" << l << " k=" << b.k << " g=" << b.g << " fk_mod_l2=[";
    for (size_t j = 0; j < f.size(); ++j) o << (j ? "," : "") << L::U128Text(f[j]);
    o << "] ek_mod_l2=" << L::U128Text(e) << " Bk_over_k_mod_l2=" << L::U128Text(b.Bk_l2) << " G_k_mod_l2=" << L::U128Text(b.Gk_l2)
      << " G_next_mod_l2=" << L::U128Text(b.Gnext_l2) << " lambda_residue=" << L::U128Text(lres) << " checks=";
    for (size_t j = 0; j < checks.size(); ++j) o << (j ? "," : "") << checks[j];
    if (all_ok) *all_ok = ok;
    return o.str();
}

heckez::record::RecordSources::HidaBlockOut HidaBlockAt(uint64_t ell, uint32_t k, uint32_t W, const std::vector<uint32_t>& ops,
                                                         const std::vector<uint32_t>& theta) {
    heckez::record::RecordSources::HidaBlockOut out;
    ENGINE_CHECK_MSG(ops.size() == theta.size() && !ops.empty(), "HidaBlockAt: theta and its operators differ in length");
    if (ell < 5 || k % (ell - 1) == 0) {
        out.refusal = "class_zero_or_small_ell";
        return out;
    }
    std::vector<uint64_t> extra;
    for (uint32_t p : ops)
        if (p != ell) extra.push_back(p);
    // the count-based cap of the settings line (heckez.G8 = hida): q-expansion length max(ell, k/12 + 1, 7, theta's operators) (dim S_k + 2) + 1 <= 2^18
    {
        const uint64_t dimS = k / 12 - (k % 12 == 2 ? 1 : 0);
        uint64_t pmax = std::max<uint64_t>({ell, k / 12 + 1, 7});
        for (uint64_t p : extra) pmax = std::max(pmax, p);
        if (pmax * (dimS + 2) + 1 > (uint64_t(1) << 18)) {
            out.refusal = "forms_cap: q-expansion length above 2^18";
            return out;
        }
    }
    // the remaining caps of the settings line, decided before the read: a refusal inside ReadBlock past them is a defect (it throws)
    {
        unsigned __int128 pw = 1;
        bool fits = W >= 2 && k >= W + 1;
        for (uint32_t i = 0; i < W && fits; ++i) {
            pw *= ell;
            fits = pw < (static_cast<unsigned __int128>(1) << 84);
        }
        if (!fits) {
            out.refusal = "forms_cap: ell^W >= 2^84 or k - 1 < W";
            return out;
        }
    }
    const BlockReading b = ReadBlock(ell, k, W, extra);
    // theta on e M_k mod ell (T_ell = U on the ordinary part)
    const ZMod F(ell, 1);
    const uint32_t s = b.fw.M.s;
    Mat Th(s, Vec(s, 0));
    for (size_t i = 0; i < ops.size(); ++i) {
        const std::string nm = ops[i] == ell ? std::string("U") : "T" + std::to_string(ops[i]);
        const auto it = b.fw.M.mats.find(nm);
        ENGINE_CHECK_MSG(it != b.fw.M.mats.end(), ("HidaBlockAt: no operator " + nm + " in the forms data").c_str());
        const u128 c = theta[i] % ell;
        for (size_t r = 0; r < s; ++r)
            for (size_t t = 0; t < s; ++t) Th[r][t] = F.add(Th[r][t], F.mul(c, it->second[r][t] % ell));
    }
    const Vec cp = L::Charpoly(Th, F);
    for (u128 c : cp) out.theta_ord_charpoly.push_back(static_cast<uint64_t>(c));
    out.served = true;
    out.g = b.g;
    out.v_Bk = b.v_Bk;
    out.lambda_one = b.lambda_one;
    out.I0_q = static_cast<uint32_t>(b.I0_q);
    if (b.g >= 1) {
        bool ok = false;
        std::string line = K9Line(b, &ok);
        const std::string pre = "HIDA_K9 ";
        if (line.compare(0, pre.size(), pre) == 0) line = line.substr(pre.size());
        out.k9 = line;
    }
    out.secs = b.secs;
    return out;
}

Vec RungPrediction(uint64_t ell, uint32_t K, uint32_t m, uint32_t centre_k, uint64_t k_star, const std::vector<Vec>& F_M) {
    ENGINE_CHECK_MSG(F_M.size() >= 3, "RungPrediction: the Eisenstein factor on e M_k has rank >= 2");
    ENGINE_CHECK_MSG(m >= K, "RungPrediction: exact mod ell^K needs m >= K (v(Z) >= 1)");
    const ZMod RK(ell, K);
    // F_M = Y F_S(Y): F_M[0] == 0, F_S[i] = F_M[i + 1]
    for (u128 c : F_M[0]) ENGINE_CHECK_MSG(c % RK.M == 0, "RungPrediction: F(0) != 0 on e M_k (the Eisenstein series is not Y-torsion)");
    const u128 Z = RK.sub(RK.sub(RK.pow((ell + 1) % RK.M, k_star), 1 % RK.M), RK.sub(RK.pow((ell + 1) % RK.M, centre_k), 1 % RK.M));
    ENGINE_CHECK_MSG(Z % ell == 0, "RungPrediction: k* is not in the class of the centre");
    std::vector<u128> FS;   // F_S(Y) at X = x_{k*}, low -> high in Y
    for (size_t i = 1; i < F_M.size(); ++i) {
        u128 acc = 0;
        for (size_t j = F_M[i].size(); j-- > 0;) acc = RK.add(RK.mul(acc, Z), F_M[i][j] % RK.M);
        FS.push_back(acc);
    }
    // f(x) = F_S(x - e), e = 1 + 2^{k*-1}: Taylor shift by -e
    const u128 e = RK.add(1 % RK.M, RK.pow(2 % RK.M, k_star - 1));
    Vec f(FS.size(), 0);
    // Horner in (x - e): f = (...((FS[g] (x - e) + FS[g-1]) (x - e) + ...)
    for (size_t i = FS.size(); i-- > 0;) {
        Vec nf(f.size(), 0);
        for (size_t j = 0; j + 1 < f.size(); ++j) {   // multiply by (x - e)
            nf[j + 1] = RK.add(nf[j + 1], f[j]);
            nf[j] = RK.add(nf[j], RK.mul(RK.neg(e), f[j]));
        }
        nf[0] = RK.add(nf[0], FS[i]);
        f = nf;
    }
    return f;
}

}  // namespace eislambda
}  // namespace hecke_ext
