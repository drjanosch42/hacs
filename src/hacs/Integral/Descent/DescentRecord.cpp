// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// DESCENT milestone D6: the record `integral_descent_record/v1` -- see DescentRecord.h.

#include "DescentRecord.h"
#include "../../Util/CancelPoint.h"   // (NBCONFIRM S5) cooperative cancellation points

#include "DescentAssembly.h"
#include "DescentHecke.h"
#include "DescentLevelN.h"
#include "DescentPair.h"

#include "../../../ModArithmetic/EngineCheck.h"
#include "../../../ModArithmetic/Fnv.h"
#include "../../Hecke/HeckeSettings.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace integral_descent {
namespace record {

namespace {

double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

std::string Hex16(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return o.str();
}

std::string JsonEsc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

template <class T>
std::string Join(const std::vector<T>& v, const char* sep = ",") {
    std::ostringstream o;
    for (size_t i = 0; i < v.size(); ++i) o << (i ? sep : "") << v[i];
    return o.str();
}

std::string JsonU32List(const std::vector<uint32_t>& v) { return "[" + Join(v) + "]"; }

std::string JsonGroups(const std::vector<AbelianGroupType>& G) {
    std::ostringstream o;
    o << "[";
    for (size_t q = 0; q < G.size(); ++q) o << (q ? "," : "") << "\"" << JsonEsc(G[q].to_string()) << "\"";
    o << "]";
    return o.str();
}

// The ell-primary exponents of G, ascending.
std::vector<uint32_t> ExponentsAt(const AbelianGroupType& G, uint32_t ell) {
    std::vector<uint32_t> e;
    const auto it = G.torsion.find(ell);
    if (it != G.torsion.end()) e.assign(it->second.begin(), it->second.end());
    std::sort(e.begin(), e.end());
    return e;
}

bool IsPrimeSmall(uint32_t p) {
    if (p < 2) return false;
    for (uint32_t d = 2; d * d <= p; ++d)
        if (p % d == 0) return false;
    return true;
}

PrimeK KOfPiece(const LocalPiece& P, bool ran) {
    PrimeK k;
    k.ell = P.ell;
    k.route = P.route;
    k.K = P.K;
    k.W = P.K;
    for (const LocalDifferential& d : P.d) k.W = std::max(k.W, d.W);
    k.certified = ran && P.all_certified();
    return k;
}

std::string ComplexFieldsJson(uint32_t q_max) {
    std::vector<uint32_t> deg;
    for (uint32_t q = 0; q <= q_max; ++q) deg.push_back(q);
    return "\"q_max\":" + std::to_string(q_max) + ",\"truncated\":true,\"q_max_source\":\"derived\",\"degrees\":" + JsonU32List(deg);
}

}  // namespace

std::vector<uint32_t> DefaultRecordPrimes(uint32_t N) {
    std::vector<uint32_t> out;
    for (uint32_t p : {2u, 3u, 5u, 7u, 11u, 13u}) {
        if (N % p == 0) continue;
        out.push_back(p);
        if (out.size() == 3) break;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// (FMTMP) the faithful encoding of a coefficient beyond 2^63 -- DescentRecord.h
// ---------------------------------------------------------------------------------------------------------------------
std::string ExactHex(const std::vector<uint32_t>& digits, const std::vector<uint32_t>& primes, bool negative) {
    ENGINE_CHECK_MSG(digits.size() <= primes.size(), "ExactHex: more mixed-radix digits than residue primes");
    // c^+ by Horner from the top digit: acc = acc * primes[k] + digits[k] on little-endian base-2^32 limbs
    std::vector<uint32_t> acc;
    auto mul_add = [](std::vector<uint32_t>& a, uint32_t m, uint32_t add) {
        uint64_t carry = add;
        for (uint32_t& limb : a) {
            const uint64_t t = uint64_t(limb) * m + carry;
            limb = static_cast<uint32_t>(t);
            carry = t >> 32;
        }
        if (carry) a.push_back(static_cast<uint32_t>(carry));
    };
    for (size_t k = digits.size(); k-- > 0;) {
        ENGINE_CHECK_MSG(digits[k] < primes[k], "ExactHex: a mixed-radix digit is not below its prime");
        mul_add(acc, primes[k], digits[k]);
    }
    std::vector<uint32_t> mag = acc;
    if (negative) {
        // |c| = P - c^+ with P = prod_{k<D} primes[k] > c^+
        std::vector<uint32_t> P = {1};
        for (size_t k = 0; k < digits.size(); ++k) mul_add(P, primes[k], 0);
        mag = P;
        int64_t borrow = 0;
        for (size_t j = 0; j < mag.size(); ++j) {
            int64_t t = int64_t(mag[j]) - borrow - (j < acc.size() ? int64_t(acc[j]) : 0);
            borrow = t < 0 ? 1 : 0;
            if (t < 0) t += int64_t(1) << 32;
            mag[j] = static_cast<uint32_t>(t);
        }
        ENGINE_CHECK_MSG(borrow == 0 && acc.size() <= P.size(), "ExactHex: c^+ is not below the product of its primes");
    }
    while (!mag.empty() && mag.back() == 0) mag.pop_back();
    std::ostringstream o;
    if (negative && !mag.empty()) o << "-";
    o << "0x" << std::hex;
    if (mag.empty()) o << "0";
    else {
        o << mag.back();
        for (size_t j = mag.size() - 1; j-- > 0;) o << std::setw(8) << std::setfill('0') << mag[j];
    }
    return o.str();
}

bool ExactHexCanonical(const std::string& s) {
    size_t i = (!s.empty() && s[0] == '-') ? 1 : 0;
    if (s.size() < i + 3 || s[i] != '0' || s[i + 1] != 'x') return false;
    const size_t d0 = i + 2;
    for (size_t j = d0; j < s.size(); ++j)
        if (!((s[j] >= '0' && s[j] <= '9') || (s[j] >= 'a' && s[j] <= 'f'))) return false;
    if (s[d0] == '0') return s.size() == d0 + 1 && i == 0;   // "0x0" only, never "-0x0" nor a leading zero
    return true;
}

bool ExactHexModQ(const std::string& s, uint64_t Q, uint64_t* out) {
    if (!ExactHexCanonical(s)) return false;
    const bool neg = s[0] == '-';
    uint64_t acc = 0;
    for (size_t j = (neg ? 3 : 2); j < s.size(); ++j) {
        const uint64_t v = (s[j] <= '9') ? uint64_t(s[j] - '0') : uint64_t(s[j] - 'a' + 10);
        acc = hecke3::WordMulMod(acc, 16, Q);
        const uint64_t vq = Q == 0 ? v : v % Q;
        const uint64_t t = acc + vq;   // acc < Q, vq < Q: one conditional subtraction (wrap-around detected)
        acc = (Q == 0) ? t : ((t < acc || t >= Q) ? t - Q : t);
    }
    if (neg) acc = (Q == 0) ? (0 - acc) : (acc == 0 ? 0 : Q - acc);
    *out = acc;
    return true;
}

uint64_t RecordModuleRank(uint32_t N, uint32_t n) { return uint64_t(PsiIndex(N)) * (uint64_t(n) + 1); }

// ---------------------------------------------------------------------------------------------------------------------
// the hashed body
// ---------------------------------------------------------------------------------------------------------------------
std::string DescentRecordZ::body() const {
    std::ostringstream o;
    o << "schema=" << kSchema << "\n";
    o << "N=" << N << "\nn=" << n << "\nprimes=" << Join(ps) << "\n";
    o << "coefficient_ring=Z\nobject=" << object << "\n";
    o << "q_max=" << q_max << "\ntruncated=" << (truncated ? "true" : "false") << "\nq_max_source=" << q_max_source
      << "\ndegrees=" << Join(degrees) << "\nperiodic_from=" << periodic_from << "\n";
    for (size_t q = 0; q < H.size(); ++q) o << "H^" << q << "=" << H[q].to_string() << "\n";
    o << "bad_primes=";
    for (size_t i = 0; i < bad_primes.size(); ++i) o << (i ? "," : "") << bad_primes[i].first << ":" << bad_primes[i].second;
    o << "\ntorsion_primes=" << Join(torsion_primes) << "\n";
    if (!hecke_ran) o << "hecke=not_run\n";   // only when not run: a record with the part keeps its body, hence its hash
    for (const CharpolyEntry& c : charpoly) {
        o << "charpoly p=" << c.p << " r=" << c.r << " determined=" << (c.determined ? 1 : 0) << " digest=" << Hex16(c.digest);
        if (c.refused) o << " refused";
        o << "\n";
        // (FMTMP, [C] sec:recordschema item 7 (l)) the exact value of every coefficient beyond 2^63; none: no line (the body is unchanged)
        for (size_t i = 0; i < c.exact.size(); ++i)
            if (!c.exact[i].empty()) o << "charpoly p=" << c.p << " i=" << i << " exact=" << c.exact[i] << "\n";
    }
    for (const TorsionModule& t : torsion) {
        o << "torsion q=" << t.q << " ell=" << t.ell << " a=" << Join(t.exponents, ".") << " module=" << (t.module_certified ? 1 : 0);
        for (const auto& [p, canon] : t.invariants) o << " T_" << p << ":" << canon;
        if (t.refused) o << " refused";
        o << "\n";
    }
    if (pair.ran) {
        o << "pair q_max=" << pair.q_max << (pair.refused ? " refused" : "") << "\n";
        for (size_t q = 0; q < pair.H.size(); ++q)
            o << "pair q=" << q << " H=" << pair.H[q].to_string() << " Hd=" << pair.Hd[q].to_string() << " Hc=" << pair.Hc[q].to_string()
              << "\n";
    }
    o << "status=" << status << "\n";
    o << "settings_fingerprint=" << Hex16(settings_fingerprint) << "\n";
    o << "settings=" << settings_canonical;   // '\n'-terminated, key order fixed by hecke::MathSettings
    return o.str();
}

uint64_t DescentRecordZ::digest() const { return engine::FnvBytes(engine::kFnvOffset64, body()); }

std::string DescentRecordZ::to_json() const {
    std::ostringstream o;
    o << "{\"schema\":\"" << kSchema << "\",\"N\":" << N << ",\"n\":" << n << ",\"primes\":" << JsonU32List(ps)
      << ",\"coefficient_ring\":\"Z\",\"object\":\"" << object << "\",\"q_max\":" << q_max << ",\"truncated\":" << (truncated ? "true" : "false")
      << ",\"q_max_source\":\"" << q_max_source << "\",\"degrees\":" << JsonU32List(degrees) << ",\"periodic_from\":" << periodic_from
      << ",\"groups\":" << JsonGroups(H) << ",\"bad_primes\":[";
    for (size_t i = 0; i < bad_primes.size(); ++i)
        o << (i ? "," : "") << "{\"ell\":" << bad_primes[i].first << ",\"count\":" << bad_primes[i].second << "}";
    o << "],\"torsion_primes\":" << JsonU32List(torsion_primes) << ",\"K\":[";
    for (size_t i = 0; i < K.size(); ++i)
        o << (i ? "," : "") << "{\"ell\":" << K[i].ell << ",\"route\":\"" << K[i].route << "\",\"K\":" << K[i].K << ",\"W\":" << K[i].W
          << ",\"certified\":" << (K[i].certified ? "true" : "false") << "}";
    o << "],\"charpoly\":[";
    for (size_t i = 0; i < charpoly.size(); ++i) {
        const CharpolyEntry& c = charpoly[i];
        o << (i ? "," : "") << "{\"p\":" << c.p << ",\"r\":" << c.r << ",\"coefficients\":[";
        for (size_t j = 0; j < c.coefficients.size(); ++j) o << (j ? "," : "") << "\"" << JsonEsc(c.coefficients[j]) << "\"";
        o << "]";
        // (FMTMP, item 7 (l)) written only when a coefficient is beyond 2^63: an entry without one keeps its bytes
        bool any_big = false;
        for (size_t j = 0; j < c.exact.size(); ++j) {
            if (c.exact[j].empty()) continue;
            o << (any_big ? "," : ",\"big\":[") << "{\"i\":" << j << ",\"hex\":\"" << c.exact[j] << "\"}";
            any_big = true;
        }
        if (any_big) o << "]";
        o << ",\"determined\":" << (c.determined ? "true" : "false") << ",\"undetermined\":" << c.undetermined << ",\"digest\":\""
          << Hex16(c.digest) << "\",\"refused\":" << (c.refused ? "true" : "false") << ",\"refusal\":\"" << JsonEsc(c.refusal) << "\"}";
    }
    o << "],\"torsion\":[";
    for (size_t i = 0; i < torsion.size(); ++i) {
        const TorsionModule& t = torsion[i];
        o << (i ? "," : "") << "{\"q\":" << t.q << ",\"ell\":" << t.ell << ",\"route\":\"" << t.route << "\",\"K\":" << t.K
          << ",\"exponents\":" << JsonU32List(t.exponents) << ",\"module_certified\":" << (t.module_certified ? "true" : "false")
          << ",\"ops\":[";
        for (size_t j = 0; j < t.invariants.size(); ++j)
            o << (j ? "," : "") << "{\"p\":" << t.invariants[j].first << ",\"invariants\":\"" << JsonEsc(t.invariants[j].second) << "\"}";
        o << "],\"commute\":" << (t.commute ? "true" : "false") << ",\"refused\":" << (t.refused ? "true" : "false") << ",\"refusal\":\""
          << JsonEsc(t.refusal) << "\"}";
    }
    o << "]";
    if (!hecke_ran) o << ",\"hecke\":{\"ran\":false}";   // (ICP2) only when skipped: a full record keeps its bytes
    o << ",\"pair\":{\"ran\":" << (pair.ran ? "true" : "false");
    if (pair.ran) {
        o << ",\"bulk\":{" << ComplexFieldsJson(pair.q_max) << ",\"groups\":" << JsonGroups(pair.H) << "}"
          << ",\"boundary\":{" << ComplexFieldsJson(pair.q_max) << ",\"groups\":" << JsonGroups(pair.Hd) << "}"
          << ",\"compact\":{" << ComplexFieldsJson(pair.q_max) << ",\"derived_from\":[\"" << kCompactDerivedFrom[0] << "\",\""
          << kCompactDerivedFrom[1] << "\"],\"groups\":" << JsonGroups(pair.Hc) << "}"
          << ",\"primes\":" << JsonU32List(pair.primes) << ",\"route_two\":\"" << pair.route_two << "\"";
    }
    o << ",\"refused\":" << (pair.refused ? "true" : "false") << ",\"refusal\":\"" << JsonEsc(pair.refusal) << "\"}";
    o << ",\"status\":\"" << status << "\",\"refusal\":\"" << JsonEsc(refusal) << "\",\"settings\":" << settings_json
      << ",\"settings_fingerprint\":\"" << Hex16(settings_fingerprint) << "\",\"record_fnv1a64\":\"" << Hex16(digest()) << "\""
      << ",\"timings_s\":{\"groups\":" << std::fixed << std::setprecision(3) << t_groups << ",\"hecke\":" << t_hecke << ",\"pair\":" << t_pair
      << ",\"total\":" << t_total << "},\"rss_peak_mib\":" << (rss_peak >> 20) << "}";
    return o.str();
}

// ---------------------------------------------------------------------------------------------------------------------
// the builder
// ---------------------------------------------------------------------------------------------------------------------
DescentRecordZ BuildDescentRecord(uint32_t N, uint32_t n, const RecordOptions& opt) {
    if (!hecke::CoefficientRingIsZ())
        throw std::invalid_argument("IntegralDescentRecord: the descent record is an object over Z; set coefficient_ring = Z first (in force: " +
                                    hecke::math_settings().coefficient_ring.to_text() + ")");
    if (N == 0) throw std::invalid_argument("IntegralDescentRecord: N must be >= 1");
    std::vector<uint32_t> ps = opt.ps.empty() ? DefaultRecordPrimes(N) : opt.ps;
    // (D6 audit N3) P is a set: hashed and reported ascending, whatever the caller's order (the console verb, the campaign and the
    // defaults agree; the charpoly lines follow this order)
    std::sort(ps.begin(), ps.end());
    {
        std::vector<uint32_t> s = ps;
        std::sort(s.begin(), s.end());
        if (std::adjacent_find(s.begin(), s.end()) != s.end()) throw std::invalid_argument("IntegralDescentRecord: a repeated Hecke prime");
        for (uint32_t p : ps) {
            if (!IsPrimeSmall(p)) throw std::invalid_argument("IntegralDescentRecord: p = " + std::to_string(p) + " is not a prime");
            if (N % p == 0)
                throw std::invalid_argument("IntegralDescentRecord: p = " + std::to_string(p) + " divides N (U_p is deferred, [R-Q4])");
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    DescentRecordZ R;
    R.N = N;
    R.n = n;
    R.ps = ps;
    for (uint32_t q = 0; q <= R.q_max; ++q) R.degrees.push_back(q);
    // the settings in force are the record's (coefficient_ring = Z): hashed by their canonical text
    const hecke::MathSettings in_force = hecke::math_settings();
    R.settings_canonical = in_force.canonical_text();
    R.settings_fingerprint = in_force.fingerprint();
    R.settings_json = in_force.to_json();
    // the descent's tools run with the ring reset to F_ell (they build F_ell' and Z/ell^K objects of their own)
    hecke::MathSettings tools = in_force;
    tools.coefficient_ring = hecke::CoefficientRingSetting();
    tools.coefficient_ring_source = hecke::SettingSource::Default;
    const hecke::ScopedMathSettings scope(tools);
    RssSampler rs;
    auto note = [&R](const std::string& why) {
        if (R.status == "ok") {
            R.status = "refused";
            R.refusal = why;
        }
    };

    // (NBCONFIRM S5) cancellation points between the record's stages (Util/CancelPoint.h): every stage builds stack-local
    // results only, so a stop here leaves no cache entry half-written; a no-op outside a console statement
    oc_util::CancelPoint("descent.groups");
    // (1) the groups (thm:descentcert(1)): D1's assembly at every N (N = 1 included)
    {
        const auto t = std::chrono::steady_clock::now();
        LevelNOptions lo;
        lo.rss_cap_bytes = opt.rss_cap_bytes;
        const LevelNResult G = AssembleDescentN(N, n, lo);
        R.bad_primes = G.bad;
        if (G.ran_two) R.K.push_back(KOfPiece(G.two, true));
        if (G.ran_three) R.K.push_back(KOfPiece(G.three, true));
        if (G.three_shortcut) {
            PrimeK k;
            k.ell = 3;
            k.route = "shortcut";
            k.K = k.W = G.three_sparse.K;
            k.certified = G.three_sparse.certified;
            R.K.push_back(k);
        }
        for (const SparseSmithRun& s : G.sparse) R.K.push_back(PrimeK{s.ell, "sparse", s.K, s.K, s.certified});
        for (const BredonSmithRun& b : G.bredon) R.K.push_back(PrimeK{b.ell, "bredon", b.K, b.K, b.certified});
        std::sort(R.K.begin(), R.K.end(), [](const PrimeK& a, const PrimeK& b) { return a.ell < b.ell; });
        if (G.refused) note("groups: " + G.refusal);
        else
            for (uint32_t q = 0; q <= R.q_max; ++q) R.H.push_back(G.H.at(q));
        R.t_groups = since(t);
    }
    R.torsion_primes = hecke3::RecordTorsionPrimes(N, n);

    // (2), (3) char_Z and the torsion Hecke modules (thm:descentcert(2), (3)): D3's record
    if (opt.hecke) {
        oc_util::CancelPoint("descent.hecke");
        R.hecke_ran = true;
        const auto t = std::chrono::steady_clock::now();
        hecke3::HeckeRecordOptions ho;
        ho.q_max = R.q_max;
        ho.rss_cap_bytes = opt.rss_cap_bytes;
        ho.crt.rss_cap_bytes = opt.rss_cap_bytes;
        ho.crt.threads = opt.threads;
        const hecke3::HeckeRecord HR = hecke3::RunDescentHecke(N, n, ps, ho);
        for (const hecke3::CharpolyZ& C : HR.chi) {
            CharpolyEntry e;
            e.p = C.p;
            e.r = C.r;
            e.primes = C.primes;
            constexpr uint64_t kThree40 = 12157665459056928801ull;   // 3^40
            for (const hecke3::CrtCoefficient& x : C.c) {
                std::string hex;
                if (x.certified && !x.fits) {
                    hex = ExactHex(x.digits, C.primes, x.negative);
                    // CONSISTENCY (O(size), a defect if it fails): the text decodes, by an independent path, to D3's c mod 2^64, mod 3^40
                    uint64_t m64 = 0, m3 = 0;
                    ENGINE_CHECK_MSG(ExactHexModQ(hex, 0, &m64) && ExactHexModQ(hex, kThree40, &m3) && m64 == x.mod_2_64 && m3 == x.mod_3_40 &&
                                         (hex[0] == '-') == x.negative,
                                     ("IntegralDescentRecord: the exact value of c_" + std::to_string(x.i) + " of char_Z(T_" + std::to_string(C.p) +
                                      ") disagrees with D3's residues at (N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + ")")
                                         .c_str());
                }
                e.exact.push_back(hex);
                if (!x.certified) e.coefficients.push_back("?");
                else if (x.fits) e.coefficients.push_back(std::to_string(x.value));
                else e.coefficients.push_back(std::string("[") + (x.negative ? "-" : "+") + "," + std::to_string(x.digits.size()) + " digits]");
            }
            e.determined = C.determined;
            e.undetermined = C.undetermined();
            e.digest = C.digest();
            e.refused = C.refused;
            e.refusal = C.refusal;
            if (C.refused) note("charpoly T_" + std::to_string(C.p) + ": " + C.refusal);
            R.charpoly.push_back(std::move(e));
        }
        for (const hecke3::TorsionEntry& E : HR.tors) {
            TorsionModule m;
            m.q = E.q;
            m.ell = E.ell;
            m.route = E.route;
            m.K = E.read.K;
            m.refused = E.refused;
            m.refusal = E.refusal;
            if (E.refused) {
                note("torsion q=" + std::to_string(E.q) + " ell=" + std::to_string(E.ell) + ": " + E.refusal);
            } else {
                for (uint32_t s : E.read.sigma)
                    if (s > 0) m.exponents.push_back(s);
                std::sort(m.exponents.begin(), m.exponents.end());
                m.module_certified = E.read.module_certified;
                m.commute = E.commute;
                for (size_t i = 0; i < E.ps.size() && i < E.inv.size(); ++i)
                    m.invariants.push_back({E.ps[i], E.inv[i].ok ? E.inv[i].canon : E.inv[i].layers});
                // CONSISTENCY: the read's group is the assembly's ell-primary part (two independent Smith forms)
                if (!R.H.empty() && m.q < R.H.size())
                    ENGINE_CHECK_MSG(m.exponents == ExponentsAt(R.H[m.q], m.ell),
                                     ("IntegralDescentRecord: the torsion read at (q, ell) = (" + std::to_string(m.q) + ", " + std::to_string(m.ell) +
                                      ") disagrees with the groups' assembly at (N, n) = (" + std::to_string(N) + ", " + std::to_string(n) + ")")
                                         .c_str());
            }
            R.torsion.push_back(std::move(m));
        }
        R.t_hecke = since(t);
    }

    // the pair (sec:descentpair; D4, the tree route of sec:treepair by default)
    if (opt.pair) {
        oc_util::CancelPoint("descent.pair");
        const auto t = std::chrono::steady_clock::now();
        pair::PairOptions po;
        po.q_max = R.pair.q_max;
        po.rss_cap_bytes = opt.rss_cap_bytes;
        const pair::PairResult P = pair::AssembleDescentPair(N, n, po);
        R.pair.ran = true;
        R.pair.primes = P.primes;
        R.pair.route_two = P.route_two;
        if (P.refused) {
            R.pair.refused = true;
            R.pair.refusal = P.refusal;
            note("pair: " + P.refusal);
        } else {
            R.pair.H = P.H;
            R.pair.Hd = P.Hd;
            R.pair.Hc = P.Hc;
            ENGINE_CHECK_MSG(R.pair.H.size() == R.pair.q_max + 1 && R.pair.Hd.size() == R.pair.q_max + 1 && R.pair.Hc.size() == R.pair.q_max + 1,
                             "IntegralDescentRecord: the pair's groups are not reported in degrees 0 .. q_max");
            if (!R.H.empty())
                for (uint32_t q = 0; q <= R.q_max; ++q)
                    ENGINE_CHECK_MSG(R.pair.H[q] == R.H[q], ("IntegralDescentRecord: the pair's bulk H^" + std::to_string(q) +
                                                             " disagrees with the groups' assembly at (N, n) = (" + std::to_string(N) + ", " +
                                                             std::to_string(n) + ")")
                                                                .c_str());
        }
        R.t_pair = since(t);
    }
    R.rss_peak = rs.stop();
    R.t_total = since(t0);
    return R;
}

}  // namespace record
}  // namespace integral_descent
