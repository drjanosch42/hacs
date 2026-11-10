// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// (hecke-extensions slot 6, WP-8 / phase E6) HeckeExt/HeckeAtlas -- see the
// header for the contract.  [E] `sec:atlas`, `cl:sigma`, `cl:sep`,
// `sec:field` tier (ii); [X] Sec. 2.2 Stage 6.
// ===========================================================================

#include "HeckeAtlas.h"

#include "HeckeExtDriver.h"
#include "HeckeLocalAlgebra.h"

#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../Util/ParallelFor.h"
#include "../Util/Rss.h"
#include "../../ModArithmetic/ScalarMod.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace hecke_ext {

namespace {

using Key = std::vector<FieldElt>;
using Clock = std::chrono::steady_clock;

double Secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

bool IsPrimeSmall(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

uint32_t PowModSmall(uint64_t base, uint64_t e, uint32_t m) { return engine::scalar::powmod<uint32_t>(static_cast<uint32_t>(base % m), e, m); }

// Coordinate-wise CanonicalField::less, lexicographic over the vector.
bool KeyLess(const Key& a, const Key& b) {
    ENGINE_CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (CanonicalField::less(a[i], b[i])) return true;
        if (CanonicalField::less(b[i], a[i])) return false;
    }
    return false;
}

// The r Frobenius conjugates of a vector (conjugate 0 is v itself).
std::vector<Key> Conjugates(const CanonicalField& F, const Key& v) {
    std::vector<Key> out;
    out.reserve(F.degree());
    Key cur = v;
    for (uint32_t i = 0; i < F.degree(); ++i) {
        out.push_back(cur);
        for (FieldElt& x : cur) x = F.frobenius(x);
    }
    ENGINE_CHECK_MSG(cur == v, "HeckeAtlas: Frobenius^r is not the identity on a key");
    return out;
}

// A small cache of the canonical fields of one ell, one per r.  Held by the
// build (and by each caller that needs one), never global.
class FieldCache {
public:
    explicit FieldCache(uint32_t ell) : ell_(ell) {}
    const CanonicalField& get(uint32_t r) {
        auto it = fields_.find(r);
        if (it == fields_.end()) it = fields_.emplace(r, CanonicalField(ell_, r)).first;
        return it->second;
    }
    const std::map<uint32_t, CanonicalField>& all() const { return fields_; }

private:
    uint32_t ell_;
    std::map<uint32_t, CanonicalField> fields_;
};

// The chi_{a,b} family as T-systems: key -> least (a, b).
std::map<Key, std::pair<uint32_t, uint32_t>> ChiFamily(uint32_t ell,
                                                       const std::vector<uint64_t>& T) {
    std::map<Key, std::pair<uint32_t, uint32_t>> out;
    const uint32_t m = ell - 1u;
    for (uint32_t a = 0; a < m; ++a) {
        for (uint32_t b = a + 1u; b < m; ++b) {
            if (((a + b) & 1u) == 0u) continue;
            Key k;
            for (uint64_t q : T) {
                const uint32_t v = (PowModSmall(q, a, ell) + PowModSmall(q, b, ell)) % ell;
                k.push_back(FieldElt{v});
            }
            if (out.find(k) == out.end()) out.emplace(k, std::make_pair(a, b));
        }
    }
    return out;
}

// ------------------------------------------------------------ JSON writing --
void WriteU64List(std::ostringstream& o, const std::vector<uint64_t>& v) {
    o << '[';
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ']';
}
void WriteU32List(std::ostringstream& o, const std::vector<uint32_t>& v) {
    o << '[';
    for (size_t i = 0; i < v.size(); ++i) o << (i ? "," : "") << v[i];
    o << ']';
}
void WriteKey(std::ostringstream& o, const Key& k) {
    o << '[';
    for (size_t i = 0; i < k.size(); ++i) {
        if (i) o << ',';
        WriteU32List(o, k[i]);
    }
    o << ']';
}
void WriteStr(std::ostringstream& o, const std::string& s) {
    o << '"';
    for (char c : s) {
        ENGINE_CHECK_MSG(c != '"' && c != '\\' && static_cast<unsigned char>(c) >= 0x20,
                         "HeckeAtlas: a store string needs escaping, which v1 does not do");
        o << c;
    }
    o << '"';
}
void WriteSep(std::ostringstream& o, const AtlasSeparation& s) {
    o << "{\"primes\":";
    WriteU64List(o, s.primes);
    o << ",\"greedy_order\":";
    WriteU64List(o, s.greedy_order);
    o << ",\"separates\":" << (s.separates ? "true" : "false") << ",\"targets\":" << s.targets
      << ",\"pairs\":" << s.pairs << ",\"unseparated\":" << s.unseparated
      << ",\"undecided\":" << s.undecided << '}';
}

// ------------------------------------------------------------ JSON reading --
struct JVal {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    uint64_t u = 0;
    std::string s;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;
    // (slot-7 WP-5) An array whose elements are all numbers is kept as `nums`
    // (numarr = true, arr empty): the ell = 691 store holds ~10^8 numbers,
    // and one JVal per number cost ~30x the file size in memory.
    bool numarr = false;
    std::vector<uint64_t> nums;

    const JVal& at(const std::string& k) const {
        if (kind != Obj) throw std::runtime_error("hecke_atlas: expected an object for '" + k + "'");
        for (const auto& kv : obj)
            if (kv.first == k) return kv.second;
        throw std::runtime_error("hecke_atlas: missing key '" + k + "'");
    }
};

class JParser {
public:
    explicit JParser(const std::string& t) : t_(t) {}
    JVal parse() {
        JVal v = value();
        ws();
        if (i_ != t_.size()) fail("trailing text");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("hecke_atlas: parse error at byte " + std::to_string(i_) + ": "
                                 + why);
    }
    void ws() {
        while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\n' || t_[i_] == '\t'
                                  || t_[i_] == '\r'))
            ++i_;
    }
    bool lit(const char* w) {
        const std::string s(w);
        if (t_.compare(i_, s.size(), s) == 0) { i_ += s.size(); return true; }
        return false;
    }
    std::string str() {
        if (t_[i_] != '"') fail("expected a string");
        ++i_;
        std::string out;
        while (i_ < t_.size() && t_[i_] != '"') {
            if (t_[i_] == '\\') fail("escapes are not part of hecke_atlas/v1");
            out.push_back(t_[i_++]);
        }
        if (i_ >= t_.size()) fail("unterminated string");
        ++i_;
        return out;
    }
    JVal value() {
        ws();
        if (i_ >= t_.size()) fail("unexpected end");
        JVal v;
        const char c = t_[i_];
        if (c == '{') {
            v.kind = JVal::Obj;
            ++i_;
            ws();
            if (t_[i_] == '}') { ++i_; return v; }
            for (;;) {
                ws();
                std::string k = str();
                ws();
                if (t_[i_] != ':') fail("expected ':'");
                ++i_;
                v.obj.emplace_back(std::move(k), value());
                ws();
                if (t_[i_] == ',') { ++i_; continue; }
                if (t_[i_] == '}') { ++i_; return v; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.kind = JVal::Arr;
            ++i_;
            ws();
            if (i_ >= t_.size()) fail("unexpected end");
            if (t_[i_] == ']') { ++i_; return v; }
            if (t_[i_] >= '0' && t_[i_] <= '9') {
                // A numeric array, parsed flat.
                v.numarr = true;
                for (;;) {
                    ws();
                    if (i_ >= t_.size() || t_[i_] < '0' || t_[i_] > '9')
                        fail("a numeric array holds a non-number (not part of hecke_atlas/v1)");
                    uint64_t x = 0;
                    while (i_ < t_.size() && t_[i_] >= '0' && t_[i_] <= '9') {
                        x = x * 10u + static_cast<uint64_t>(t_[i_] - '0');
                        ++i_;
                    }
                    v.nums.push_back(x);
                    ws();
                    if (i_ >= t_.size()) fail("unexpected end");
                    if (t_[i_] == ',') { ++i_; continue; }
                    if (t_[i_] == ']') { ++i_; return v; }
                    fail("expected ',' or ']'");
                }
            }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (i_ >= t_.size()) fail("unexpected end");
                if (t_[i_] == ',') { ++i_; continue; }
                if (t_[i_] == ']') { ++i_; return v; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.kind = JVal::Str; v.s = str(); return v; }
        if (lit("true")) { v.kind = JVal::Bool; v.b = true; return v; }
        if (lit("false")) { v.kind = JVal::Bool; v.b = false; return v; }
        if (lit("null")) { v.kind = JVal::Null; return v; }
        if (c >= '0' && c <= '9') {
            v.kind = JVal::Num;
            while (i_ < t_.size() && t_[i_] >= '0' && t_[i_] <= '9') {
                v.u = v.u * 10u + static_cast<uint64_t>(t_[i_] - '0');
                ++i_;
            }
            return v;
        }
        fail("unexpected character");
    }
    const std::string& t_;
    size_t i_ = 0;
};

uint64_t JU(const JVal& v) {
    if (v.kind != JVal::Num) throw std::runtime_error("hecke_atlas: expected a number");
    return v.u;
}
uint32_t JU32(const JVal& v) { return static_cast<uint32_t>(JU(v)); }
bool JB(const JVal& v) {
    if (v.kind != JVal::Bool) throw std::runtime_error("hecke_atlas: expected a boolean");
    return v.b;
}
const std::string& JS(const JVal& v) {
    if (v.kind != JVal::Str) throw std::runtime_error("hecke_atlas: expected a string");
    return v.s;
}
const std::vector<JVal>& JA(const JVal& v) {
    if (v.kind != JVal::Arr) throw std::runtime_error("hecke_atlas: expected an array");
    if (v.numarr) throw std::runtime_error("hecke_atlas: expected an array of non-numbers");
    return v.arr;
}
std::vector<uint64_t> JU64List(const JVal& v) {
    if (v.kind != JVal::Arr || (!v.numarr && !v.arr.empty()))
        throw std::runtime_error("hecke_atlas: expected an array of numbers");
    return v.nums;   // empty for [] (numarr false, nums empty)
}
std::vector<uint32_t> JU32List(const JVal& v) {
    if (v.kind != JVal::Arr || (!v.numarr && !v.arr.empty()))
        throw std::runtime_error("hecke_atlas: expected an array of numbers");
    std::vector<uint32_t> out(v.nums.size());
    for (size_t i = 0; i < v.nums.size(); ++i) out[i] = static_cast<uint32_t>(v.nums[i]);
    return out;
}
Key JKey(const JVal& v) {
    Key out;
    for (const JVal& e : JA(v)) out.push_back(JU32List(e));
    return out;
}
AtlasSeparation JSep(const JVal& v) {
    AtlasSeparation s;
    s.primes = JU64List(v.at("primes"));
    s.greedy_order = JU64List(v.at("greedy_order"));
    s.separates = JB(v.at("separates"));
    s.targets = JU(v.at("targets"));
    s.pairs = JU(v.at("pairs"));
    s.unseparated = JU(v.at("unseparated"));
    s.undecided = JU(v.at("undecided"));
    return s;
}

std::vector<uint64_t> DefaultPool(uint32_t ell) {
    std::vector<uint64_t> T;
    for (uint64_t q = 2; q <= 31; ++q)
        if (IsPrimeSmall(q) && q != ell) T.push_back(q);
    return T;
}

} // namespace

// ===========================================================================
// Keys and twists.
// ===========================================================================

std::vector<FieldElt> AtlasOrbitKey(const CanonicalField& F, const std::vector<FieldElt>& v) {
    const std::vector<Key> conj = Conjugates(F, v);
    Key best = conj[0];
    for (const Key& c : conj)
        if (KeyLess(c, best)) best = c;
    return best;
}

std::vector<FieldElt> AtlasTwist(const CanonicalField& F, const std::vector<uint64_t>& primes,
                                 const std::vector<FieldElt>& key, uint32_t j) {
    ENGINE_CHECK(primes.size() == key.size());
    Key t(key.size());
    for (size_t i = 0; i < key.size(); ++i)
        t[i] = F.mul(F.from_scalar(PowModSmall(primes[i], j, F.ell())), key[i]);
    return AtlasOrbitKey(F, t);
}

int64_t AtlasFind(const AtlasStore& s, uint32_t r, const std::vector<FieldElt>& key) {
    for (const AtlasSystem& sys : s.systems)
        if (sys.r == r && sys.eigenvalues == key) return static_cast<int64_t>(sys.id);
    return -1;
}

// ===========================================================================
// Separation.
//
// (slot-7 WP-5) Rewritten to scale to ell = 691 (|Sigma u chi| ~ 10^6):
// the pairwise loop of slot 6 is replaced by counting pairs inside groups,
// with EXACTLY the slot-6 decision per pair --
//   restricted orbit sizes differ                  -> separated;
//   same size, same r                              -> separated iff the
//                                                     restricted orbits differ;
//   same size 1, different r                       -> separated iff the
//                                                     F_ell constants differ;
//   same size >= 2, different r                    -> `undecided`, counted
//                                                     unseparated.
// So, per restricted-orbit size s: the unseparated pairs are the pairs inside
// one (r, orbit) class, plus (s >= 2) every cross-r pair of that size, which
// is also the undecided count; for s = 1 the class is the constant vector,
// across all r.  A restricted orbit is identified by its least row (two
// Frobenius orbits are equal or disjoint), its size by its period.  A
// theta-twisted system's conjugates are the SCALED conjugates of its base's
// (theta^j multiplies by F_ell-scalars, which Frobenius fixes), so only the
// untwisted systems' conjugates are ever computed.
// HeckeAtlasSkeletonValidation compares this against the slot-6 pairwise
// loop, kept verbatim there.
// ===========================================================================

namespace {

// Coordinate-wise CanonicalField::less on flat keys of m coordinates of
// width r -- KeyLess on the fixed-width vectors, without allocation.
bool FlatKeyLess(const uint32_t* a, const uint32_t* b, size_t m, uint32_t r) {
    for (size_t i = 0; i < m; ++i) {
        const uint32_t* x = a + i * r;
        const uint32_t* y = b + i * r;
        size_t dx = r, dy = r;
        while (dx > 0 && x[dx - 1] == 0u) --dx;
        while (dy > 0 && y[dy - 1] == 0u) --dy;
        if (dx != dy) return dx < dy;
        for (size_t k = 0; k < dx; ++k)
            if (x[k] != y[k]) return x[k] < y[k];
    }
    return false;
}

// The r Frobenius conjugates of `key`, flat: row i = conjugate i, m * r words.
std::vector<uint32_t> FlatConjugates(const CanonicalField& F, const Key& key) {
    const uint32_t r = F.degree();
    const size_t m = key.size();
    std::vector<uint32_t> out;
    out.reserve(static_cast<size_t>(r) * m * r);
    for (const Key& c : Conjugates(F, key))
        for (const FieldElt& x : c) {
            ENGINE_CHECK(x.size() == r);
            out.insert(out.end(), x.begin(), x.end());
        }
    return out;
}

Key FlatToKey(const uint32_t* v, size_t m, uint32_t r) {
    Key k(m);
    for (size_t i = 0; i < m; ++i) k[i].assign(v + i * r, v + (i + 1) * r);
    return k;
}

// The least (FlatKeyLess) of the r conjugate rows scaled coordinate-wise by
// `scale` (empty = no scaling), flat in `best`.
void LeastScaledRow(const std::vector<uint32_t>& conj, size_t m, uint32_t r, uint32_t ell,
                    const std::vector<uint32_t>& scale, std::vector<uint32_t>& best,
                    std::vector<uint32_t>& cur) {
    const size_t w = m * r;
    best.resize(w);
    cur.resize(w);
    for (uint32_t i = 0; i < r; ++i) {
        const uint32_t* row = conj.data() + static_cast<size_t>(i) * w;
        if (scale.empty()) {
            std::copy(row, row + w, cur.begin());
        } else {
            for (size_t q = 0; q < m; ++q) {
                const uint64_t sq = scale[q];
                for (uint32_t k = 0; k < r; ++k)
                    cur[q * r + k] = static_cast<uint32_t>((row[q * r + k] * sq) % ell);
            }
        }
        if (i == 0 || FlatKeyLess(cur.data(), best.data(), m, r)) best.swap(cur);
    }
}

std::vector<uint32_t> TwistScale(const std::vector<uint64_t>& primes, uint32_t j, uint32_t ell) {
    std::vector<uint32_t> s(primes.size());
    for (size_t i = 0; i < primes.size(); ++i) s[i] = PowModSmall(primes[i], j, ell);
    return s;
}

// A deterministic parallel for over [0, n): every index is written by exactly
// one worker and nothing is reduced across workers, so the result does not
// depend on the thread count.  ENGINE_THREADS caps it (default: hardware,
// at most 16).
template <class Fn>
void ParallelFor(size_t n, Fn fn) {
    oc_util::ParallelForBlocks(n, engine::threads::configured(engine::threads::kCap16), fn);
}

// Everything AtlasSeparationBy needs, built once per store and reused by
// every trial of the greedy.
class SeparationContext {
public:
    explicit SeparationContext(const AtlasStore& s) : st_(s), fc_(s.ell) {
        m_ = s.primes.size();
        // Conjugates of every untwisted system (the collected ones, and any
        // twist-0 system of a constructed store).
        std::vector<int64_t> own(s.systems.size(), -1);
        for (size_t i = 0; i < s.systems.size(); ++i) {
            const AtlasSystem& sys = s.systems[i];
            if (sys.twist != 0u) continue;
            own[i] = static_cast<int64_t>(conj_.size());
            conj_.push_back(FlatConjugates(fc_.get(sys.r), sys.eigenvalues));
        }
        for (size_t i = 0; i < s.systems.size(); ++i) {
            const AtlasSystem& sys = s.systems[i];
            Target t;
            t.r = sys.r;
            if (sys.twist == 0u) {
                t.conj = static_cast<uint32_t>(own[i]);
            } else {
                ENGINE_CHECK_MSG(sys.base < s.systems.size() && own[sys.base] >= 0
                                     && s.systems[sys.base].r == sys.r,
                                 "AtlasSeparationBy: a twisted system whose base is not an "
                                 "untwisted system of the same r");
                t.conj = static_cast<uint32_t>(own[sys.base]);
                t.scale = TwistScale(s.primes, sys.twist, s.ell);
            }
            targets_.push_back(std::move(t));
        }
        // A twisted system's stored key must be what its (base, twist) says:
        // the least scaled conjugate row.
        const size_t n_sys = targets_.size();
        std::vector<uint8_t> bad(n_sys, 0);
        ParallelFor(n_sys, [&](size_t i) {
            if (targets_[i].scale.empty()) return;
            std::vector<uint32_t> best, cur;
            LeastScaledRow(conj_[targets_[i].conj], m_, targets_[i].r, s.ell, targets_[i].scale,
                           best, cur);
            if (FlatToKey(best.data(), m_, targets_[i].r) != s.systems[i].eigenvalues) bad[i] = 1;
        });
        for (size_t i = 0; i < n_sys; ++i)
            ENGINE_CHECK_MSG(bad[i] == 0,
                             "AtlasSeparationBy: a twisted system's key is not theta^j of its base");
        // chi not already in Sigma, as r = 1 targets.
        std::map<Key, uint32_t> r1;
        for (const AtlasSystem& sys : s.systems)
            if (sys.r == 1u) r1.emplace(sys.eigenvalues, sys.id);
        for (const auto& kv : ChiFamily(s.ell, s.primes)) {
            if (r1.count(kv.first)) continue;
            Target t;
            t.r = 1u;
            t.conj = static_cast<uint32_t>(conj_.size());
            std::vector<uint32_t> flat;
            for (const FieldElt& x : kv.first) flat.insert(flat.end(), x.begin(), x.end());
            conj_.push_back(std::move(flat));
            targets_.push_back(std::move(t));
        }
    }

    AtlasSeparation by(const std::vector<uint64_t>& S) const {
        std::vector<size_t> idx;
        for (uint64_t q : S) {
            const auto it = std::find(st_.primes.begin(), st_.primes.end(), q);
            ENGINE_CHECK_MSG(it != st_.primes.end(), "AtlasSeparationBy: a prime outside the pool");
            idx.push_back(static_cast<size_t>(it - st_.primes.begin()));
        }
        const size_t k = idx.size();
        const size_t N = targets_.size();
        // Per target: the restricted orbit's size and its least row (for
        // size 1: the F_ell constants), in one flat buffer.
        std::vector<uint32_t> size(N);
        std::vector<size_t> off(N + 1, 0);
        for (size_t i = 0; i < N; ++i) off[i + 1] = off[i] + k * targets_[i].r;
        std::vector<uint32_t> buf(off[N]);
        std::vector<uint8_t> off_field(N, 0);
        ParallelFor(N, [&](size_t i) {
            const Target& t = targets_[i];
            const uint32_t r = t.r;
            const std::vector<uint32_t>& conj = conj_[t.conj];
            const size_t w = k * r;
            std::vector<uint32_t> rows(static_cast<size_t>(r) * w);
            for (uint32_t c = 0; c < r; ++c)
                for (size_t a = 0; a < k; ++a) {
                    const uint32_t* src = conj.data() + (static_cast<size_t>(c) * m_ + idx[a]) * r;
                    uint32_t* dst = rows.data() + static_cast<size_t>(c) * w + a * r;
                    if (t.scale.empty()) {
                        std::copy(src, src + r, dst);
                    } else {
                        const uint64_t sq = t.scale[idx[a]];
                        for (uint32_t e = 0; e < r; ++e)
                            dst[e] = static_cast<uint32_t>((src[e] * sq) % st_.ell);
                    }
                }
            // The period: the least d >= 1 with row d == row 0 (d = r at worst).
            uint32_t d = 1;
            while (d < r && !std::equal(rows.begin(), rows.begin() + static_cast<long>(w),
                                        rows.begin() + static_cast<long>(d * w)))
                ++d;
            size[i] = d;
            uint32_t* out = buf.data() + off[i];
            if (d == 1u) {
                // Frobenius-fixed, hence F_ell-rational: the constants.
                for (size_t a = 0; a < k; ++a) {
                    for (uint32_t e = 1; e < r; ++e)
                        if (rows[a * r + e] != 0u) off_field[i] = 1;
                    out[a] = rows[a * r];
                }
                return;
            }
            size_t best = 0;
            for (uint32_t c = 1; c < d; ++c)
                if (std::lexicographical_compare(rows.begin() + static_cast<long>(c * w),
                                                 rows.begin() + static_cast<long>((c + 1) * w),
                                                 rows.begin() + static_cast<long>(best * w),
                                                 rows.begin() + static_cast<long>((best + 1) * w)))
                    best = c;
            std::copy(rows.begin() + static_cast<long>(best * w),
                      rows.begin() + static_cast<long>((best + 1) * w), out);
        });
        for (size_t i = 0; i < N; ++i)
            ENGINE_CHECK_MSG(off_field[i] == 0, "a Frobenius-fixed element off F_ell");
        // The class of target i: (size, r unless size 1, the row / constants).
        auto len = [&](size_t i) { return size[i] == 1u ? k : k * targets_[i].r; };
        auto rr = [&](size_t i) { return size[i] == 1u ? 0u : targets_[i].r; };
        std::vector<size_t> order(N);
        for (size_t i = 0; i < N; ++i) order[i] = i;
        auto less = [&](size_t a, size_t b) {
            if (size[a] != size[b]) return size[a] < size[b];
            if (rr(a) != rr(b)) return rr(a) < rr(b);
            return std::lexicographical_compare(buf.begin() + static_cast<long>(off[a]),
                                                buf.begin() + static_cast<long>(off[a] + len(a)),
                                                buf.begin() + static_cast<long>(off[b]),
                                                buf.begin() + static_cast<long>(off[b] + len(b)));
        };
        std::sort(order.begin(), order.end(), less);

        AtlasSeparation out;
        out.primes = S;
        std::sort(out.primes.begin(), out.primes.end());
        out.targets = N;
        auto c2 = [](uint64_t c) -> uint64_t { return c < 2 ? 0u : c * (c - 1u) / 2u; };
        out.pairs = c2(N);
        uint64_t same_class = 0;
        for (size_t a = 0; a < N;) {
            size_t b = a + 1;
            while (b < N && !less(order[a], order[b])) ++b;
            same_class += c2(b - a);
            a = b;
        }
        // Cross-r pairs of equal size >= 2: the conservative `undecided`.
        // (Within one size >= 2 the sort key is r first, so equal r are
        // contiguous.)
        uint64_t undecided = 0;
        for (size_t a = 0; a < N;) {
            const uint32_t sz = size[order[a]];
            size_t b = a;
            uint64_t total = 0, same_r = 0;
            while (b < N && size[order[b]] == sz) {
                size_t c = b;
                while (c < N && size[order[c]] == sz && targets_[order[c]].r == targets_[order[b]].r)
                    ++c;
                same_r += c2(c - b);
                total += c - b;
                b = c;
            }
            if (sz >= 2u) undecided += c2(total) - same_r;
            a = b;
        }
        out.undecided = undecided;
        out.unseparated = same_class + undecided;
        out.separates = (out.unseparated == 0u);
        return out;
    }

private:
    struct Target {
        uint32_t r = 0;
        uint32_t conj = 0;               // index into conj_
        std::vector<uint32_t> scale;     // theta^j scalars per pool prime, or empty
    };
    const AtlasStore& st_;
    FieldCache fc_;
    size_t m_ = 0;
    std::vector<std::vector<uint32_t>> conj_;
    std::vector<Target> targets_;
};

} // namespace

AtlasSeparation AtlasSeparationBy(const AtlasStore& s, const std::vector<uint64_t>& S) {
    return SeparationContext(s).by(S);
}

// ===========================================================================
// The build.
// ===========================================================================

namespace {

using oc_util::PeakRssMiB;

} // namespace

Atlas CollectAtlas(uint32_t ell, const AtlasOptions& opts) {
    ENGINE_CHECK_MSG(ell >= 5u && IsPrimeSmall(ell), "BuildAtlas: ell must be a prime >= 5");
    const Clock::time_point b0 = Clock::now();

    Atlas atlas;
    AtlasStore& st = atlas.store;
    st.ell = ell;
    st.n_max = ell - 1u;   // ell odd => ell - 1 even
    if (opts.n_limit < st.n_max) st.n_max = opts.n_limit & ~1u;
    {
        char buf[32];
        std::snprintf(buf, sizeof buf, "0x%016llx",
                      static_cast<unsigned long long>(hecke::math_settings().fingerprint()));
        st.settings_fingerprint = buf;
    }
    st.primes = opts.primes.empty() ? DefaultPool(ell) : opts.primes;
    ENGINE_CHECK_MSG(std::is_sorted(st.primes.begin(), st.primes.end()),
                     "BuildAtlas: the prime pool must be ascending");
    for (uint64_t q : st.primes)
        ENGINE_CHECK_MSG(q != ell && IsPrimeSmall(q), "BuildAtlas: the pool is anemic primes only");

    FieldCache fc(ell);

    // --- 1 + 2: the cells, the splits, the collected systems ---------------
    struct Collected { uint32_t r; Key key; std::vector<AtlasOccurrence> occ; bool in_H1; };
    std::vector<Collected> collected;
    std::map<std::pair<uint32_t, Key>, uint32_t> index;

    for (uint32_t n = 0; n <= st.n_max; n += 2u) {
        const Clock::time_point n0 = Clock::now();
        ExtDriverOptions dopts;
        dopts.good_primes = st.primes;
        dopts.run_commutant = opts.run_commutant;
        dopts.run_sturm = opts.run_sturm;
        dopts.consult_atlas = false;   // (slot-7 WP-3) Sigma is built FROM these runs
        HeckeExtDriver d(n, ell, dopts);
        d.run();
        const HeckeExtRecord& rec = d.record();

        AtlasCell cell;
        cell.n = n;
        cell.dims = d.H_dimensions();
        cell.ideals = static_cast<uint32_t>(rec.ideals.size());
        cell.u_ell_included = rec.u_ell.included;
        cell.u_ell_reason = rec.u_ell.reason;

        AtlasCellTiming tm;
        tm.n = n;
        tm.total_s = d.timings().total_s;

        for (uint32_t i = 0; i < cell.ideals; ++i) {
            const LocalAlgebra& la = d.local_algebra(i);
            const IdealRecord& id = rec.ideals[i];
            if (!la.ok || la.witness_poly.is_zero()) continue;
            ++cell.local_ideals;

            // The anemic columns must be exactly the pool, in order.
            std::vector<size_t> anemic;
            size_t ell_col = la.primes.size();
            for (size_t j = 0; j < la.primes.size(); ++j) {
                if (la.primes[j] == ell) ell_col = j;
                else anemic.push_back(j);
            }
            ENGINE_CHECK_MSG(anemic.size() == st.primes.size(),
                             "BuildAtlas: the block's operator set is not the pool");
            for (size_t j = 0; j < anemic.size(); ++j)
                ENGINE_CHECK(la.primes[anemic[j]] == st.primes[j]);
            ENGINE_CHECK_MSG((ell_col < la.primes.size()) == rec.u_ell.included,
                             "BuildAtlas: U_ell present iff the verdict admitted it");

            const uint32_t r = static_cast<uint32_t>(la.witness_poly.degree());
            const Clock::time_point c0 = Clock::now();
            const CanonicalField& F = fc.get(r);
            CanonicalSplit sp = SplitInCanonicalField(F, la.witness_poly, la.primes, la.character);
            const Clock::time_point c1 = Clock::now();
            tm.canonical_field_s += Secs(c0, c1);
            ++tm.splits;
            tm.r_max = std::max(tm.r_max, r);
            ENGINE_CHECK_MSG(sp.ok, ("BuildAtlas: the canonical split failed: " + sp.failure).c_str());

            // The anemic orbit and its key.
            std::vector<Key> orbit;
            for (const std::vector<FieldElt>& row : sp.orbit) {
                Key v;
                for (size_t j : anemic) v.push_back(row[j]);
                orbit.push_back(std::move(v));
            }
            {
                std::vector<Key> distinct = orbit;
                std::sort(distinct.begin(), distinct.end());
                distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
                // [WP-8 finding hook] The anemic eigenvalues generating a
                // proper subfield of F_{ell^r} would need a subfield embedding
                // to key canonically; it is refused loudly, not approximated.
                ENGINE_CHECK_MSG(distinct.size() == r,
                                 ("BuildAtlas: the anemic system generates a proper subfield "
                                  "of the block's residue field (a_ell adds degree) at n = "
                                  + std::to_string(n)).c_str());
            }
            size_t best = 0;
            for (size_t c = 1; c < orbit.size(); ++c)
                if (KeyLess(orbit[c], orbit[best])) best = c;
            const Key key = orbit[best];
            ENGINE_CHECK(key == AtlasOrbitKey(F, orbit[0]));

            AtlasOccurrence occ;
            occ.n = n;
            occ.ideal = id.id;
            occ.group = id.groups.empty() ? std::string("unknown") : id.groups[0].group;
            occ.dim = id.dim;
            if (ell_col < la.primes.size()) {
                occ.a_ell.known = true;
                occ.a_ell.value = sp.orbit[best][ell_col];
                occ.a_ell.reason = rec.u_ell.reason;
            } else {
                occ.a_ell.known = false;
                occ.a_ell.reason = rec.u_ell.reason.empty() ? std::string("unavailable")
                                                             : rec.u_ell.reason;
            }

            const auto ins = index.emplace(std::make_pair(r, key),
                                           static_cast<uint32_t>(collected.size()));
            const uint32_t sys_index = ins.first->second;
            if (ins.second) collected.push_back({r, key, {}, false});
            collected[sys_index].occ.push_back(occ);
            if (occ.group == "H1") collected[sys_index].in_H1 = true;

            AtlasSource src;
            src.n = n;
            src.ideal = id.id;
            src.group = occ.group;
            src.system = sys_index;
            src.witness_poly = la.witness_poly;
            src.primes = la.primes;
            src.character = la.character;
            src.split = std::move(sp);
            atlas.sources.push_back(std::move(src));
        }
        if (opts.progress != nullptr) {
            std::string dims;
            for (size_t q = 0; q < cell.dims.size(); ++q)
                dims += (q ? "," : "") + std::to_string(cell.dims[q]);
            *opts.progress << "ATLAS_CELL ell=" << ell << " n=" << n << " dims=" << dims
                           << " ideals=" << cell.ideals << " local=" << cell.local_ideals
                           << " r_max=" << tm.r_max << " u_ell=" << (cell.u_ell_included ? 1 : 0)
                           << " total_s=" << tm.total_s
                           << " canonical_field_s=" << tm.canonical_field_s
                           << " wall_s=" << Secs(n0, Clock::now())
                           << " collected=" << collected.size()
                           << " peak_rss_mb=" << PeakRssMiB() << std::endl;
        }
        st.cells.push_back(cell);
        atlas.timings.push_back(tm);
        // (slot-8 WP-5, [KI] Sec. 4c) the caller's between-cells hook -- the
        // engine's program-cache release when main.cpp's `--atlas` branch
        // injects it.  Everything this cell contributes to the store has been
        // copied out of the driver above; the hook sees none of it.
        if (opts.between_cells) opts.between_cells();
    }

    // --- the fields ---------------------------------------------------------
    for (const auto& kv : fc.all()) {
        AtlasField f;
        f.r = kv.first;
        f.C = kv.second.defining_poly().coeffs;
        st.fields.push_back(f);
    }

    for (size_t c = 0; c < collected.size(); ++c) {
        AtlasSystem s;
        s.id = static_cast<uint32_t>(c);
        s.r = collected[c].r;
        s.eigenvalues = collected[c].key;
        s.base = s.id;
        s.twist = 0;
        s.collected = true;
        s.in_H1 = collected[c].in_H1;
        s.occurrences = collected[c].occ;
        st.systems.push_back(std::move(s));
    }
    atlas.build_s = Secs(b0, Clock::now());
    return atlas;
}

void CloseAtlas(AtlasStore& st) {
    const uint32_t ell = st.ell;
    // Cut back to the collected prefix (a closed store re-closes to itself).
    size_t n_collected = 0;
    while (n_collected < st.systems.size() && st.systems[n_collected].collected) ++n_collected;
    for (size_t c = n_collected; c < st.systems.size(); ++c)
        ENGINE_CHECK_MSG(!st.systems[c].collected,
                         "CloseAtlas: the collected systems are not a prefix of the store");
    st.systems.resize(n_collected);
    for (size_t c = 0; c < n_collected; ++c)
        ENGINE_CHECK_MSG(st.systems[c].id == c && st.systems[c].base == c
                             && st.systems[c].twist == 0u,
                         "CloseAtlas: a collected system is not (id, base, twist) = (c, c, 0)");

    FieldCache fc(ell);
    const size_t m = st.primes.size();

    // --- 4: Sigma = the theta-twist closure --------------------------------
    const std::map<Key, std::pair<uint32_t, uint32_t>> chi = ChiFamily(ell, st.primes);
    auto classify = [&](AtlasSystem& s) {
        s.type = "cuspidal";
        s.has_chi = false;
        s.chi_a = s.chi_b = 0;
        if (s.r != 1u) return;
        const auto it = chi.find(s.eigenvalues);
        if (it == chi.end()) return;
        s.type = "eisenstein";
        s.has_chi = true;
        s.chi_a = it->second.first;
        s.chi_b = it->second.second;
    };
    std::map<std::pair<uint32_t, Key>, uint32_t> index;
    for (size_t c = 0; c < n_collected; ++c) {
        classify(st.systems[c]);
        const bool fresh = index.emplace(std::make_pair(st.systems[c].r, st.systems[c].eigenvalues),
                                         static_cast<uint32_t>(c)).second;
        ENGINE_CHECK_MSG(fresh, "CloseAtlas: two collected systems with one key");
    }
    // Twist j of a system = the least of its SCALED conjugates (theta^j
    // commutes with Frobenius); inserted in the slot-6 order -- collected
    // system by collected system, j = 1 .. ell - 2, first occurrence wins.
    std::vector<std::vector<uint32_t>> scales(ell - 1u);
    for (uint32_t j = 1; j + 1u < ell; ++j) scales[j] = TwistScale(st.primes, j, ell);
    for (size_t c = 0; c < n_collected; ++c) {
        const uint32_t r = st.systems[c].r;
        const std::vector<uint32_t> conj = FlatConjugates(fc.get(r), st.systems[c].eigenvalues);
        std::vector<std::vector<uint32_t>> least(ell - 1u);
        ParallelFor(ell - 2u, [&](size_t jj) {
            const uint32_t j = static_cast<uint32_t>(jj) + 1u;
            std::vector<uint32_t> cur;
            LeastScaledRow(conj, m, r, ell, scales[j], least[j], cur);
        });
        for (uint32_t j = 1; j + 1u < ell; ++j) {
            Key t = FlatToKey(least[j].data(), m, r);
            const auto ins = index.emplace(std::make_pair(r, t),
                                           static_cast<uint32_t>(st.systems.size()));
            if (!ins.second) continue;
            AtlasSystem s;
            s.id = static_cast<uint32_t>(st.systems.size());
            s.r = r;
            s.eigenvalues = std::move(t);
            s.base = static_cast<uint32_t>(c);
            s.twist = j;
            classify(s);
            st.systems.push_back(std::move(s));
        }
    }
    st.chi_count = static_cast<uint32_t>(chi.size());
    st.chi_in_sigma = 0;
    for (const auto& kv : chi)
        if (index.count(std::make_pair(1u, kv.first))) ++st.chi_in_sigma;

    // --- 5: the greedy S_sep ------------------------------------------------
    const SeparationContext ctx(st);
    {
        std::vector<uint64_t> S;
        AtlasSeparation cur = ctx.by(S);
        while (cur.unseparated > 0u) {
            uint64_t best_q = 0;
            uint64_t best_left = cur.unseparated;
            for (uint64_t q : st.primes) {
                if (std::find(S.begin(), S.end(), q) != S.end()) continue;
                std::vector<uint64_t> trial = S;
                trial.push_back(q);
                const AtlasSeparation t = ctx.by(trial);
                if (t.unseparated < best_left) { best_left = t.unseparated; best_q = q; }
            }
            if (best_q == 0) break;   // T itself does not separate: recorded
            S.push_back(best_q);
            cur = ctx.by(S);
        }
        cur.greedy_order = S;
        st.s_sep = cur;
    }
    {
        std::vector<uint64_t> S0;
        for (uint64_t q : {2u, 3u, 5u, 7u})
            if (q != ell && std::find(st.primes.begin(), st.primes.end(), q) != st.primes.end())
                S0.push_back(q);
        st.s0_anemic = ctx.by(S0);
    }
}

Atlas BuildAtlas(uint32_t ell, const AtlasOptions& opts) {
    const Clock::time_point b0 = Clock::now();
    Atlas atlas = CollectAtlas(ell, opts);
    CloseAtlas(atlas.store);
    atlas.build_s = Secs(b0, Clock::now());
    return atlas;
}

// ===========================================================================
// (slot-7 WP-3) The in-process atlas cache.
// ===========================================================================

std::shared_ptr<const Atlas> InProcessAtlas(uint32_t ell) {
    ENGINE_CHECK_MSG(IsInProcessAtlasEll(ell),
                     "InProcessAtlas: Sigma(ell) is built in-process only at ell in "
                     "{5, 7, 11, 13} (slot-7 owner ruling 3b (B))");
    static std::mutex mu;
    static std::map<std::pair<uint32_t, uint64_t>, std::shared_ptr<const Atlas>> cache;
    const std::pair<uint32_t, uint64_t> key(ell, hecke::math_settings().fingerprint());
    const std::lock_guard<std::mutex> guard(mu);
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    // Built under the lock: a concurrent caller for the same key waits rather
    // than building twice.  The build's driver runs never consult this cache.
    std::shared_ptr<const Atlas> a = std::make_shared<const Atlas>(BuildAtlas(ell));
    cache.emplace(key, a);
    return a;
}

std::vector<uint64_t> InProcessAtlasSSep(uint32_t ell) {
    return InProcessAtlas(ell)->store.s_sep.primes;
}

// ===========================================================================
// (slot-6 WP-9) Per-ideal characters of a completed driver run.
// ===========================================================================

std::vector<AtlasIdealCharacter> AtlasIdealCharacters(const HeckeExtDriver& d) {
    const HeckeExtRecord& rec = d.record();
    const uint32_t ell = rec.ell;
    FieldCache fc(ell);
    std::vector<AtlasIdealCharacter> out;
    for (uint32_t i = 0; i < rec.ideals.size(); ++i) {
        const IdealRecord& id = rec.ideals[i];
        AtlasIdealCharacter c;
        c.ideal = id.id;
        c.dim = id.dim;
        for (const IdealGroupMatch& g : id.groups)
            if (g.own) { c.group = g.group; break; }
        ENGINE_CHECK_MSG(c.group.size() == 2 && c.group[0] == 'H' && c.group[1] >= '0'
                             && c.group[1] <= '2',
                         "AtlasIdealCharacters: an unsplit ideal's own group is not H<q>");
        c.degree = static_cast<uint32_t>(c.group[1] - '0');

        const LocalAlgebra& la = d.local_algebra(i);
        if (!la.ok || la.witness_poly.is_zero()) {
            out.push_back(std::move(c));
            continue;
        }
        std::vector<size_t> anemic;
        for (size_t j = 0; j < la.primes.size(); ++j) {
            if (la.primes[j] == ell) continue;
            anemic.push_back(j);
            c.primes.push_back(la.primes[j]);
        }
        const uint32_t r = static_cast<uint32_t>(la.witness_poly.degree());
        const CanonicalField& F = fc.get(r);
        const CanonicalSplit sp = SplitInCanonicalField(F, la.witness_poly, la.primes,
                                                        la.character);
        ENGINE_CHECK_MSG(sp.ok, ("AtlasIdealCharacters: the canonical split failed: "
                                 + sp.failure).c_str());
        for (const std::vector<FieldElt>& row : sp.orbit) {
            Key v;
            for (size_t j : anemic) v.push_back(row[j]);
            c.orbit.push_back(std::move(v));
        }
        std::sort(c.orbit.begin(), c.orbit.end());
        std::vector<Key> distinct = c.orbit;
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        c.anemic_generates = (distinct.size() == r);
        c.r = r;
        c.local = true;
        out.push_back(std::move(c));
    }
    return out;
}

bool AtlasResidueMultisets(const std::vector<AtlasIdealCharacter>& chars,
                           std::vector<ResidueCharacterMultiset>& out, std::string& why) {
    out.assign(3, ResidueCharacterMultiset());
    std::vector<std::map<std::pair<uint32_t, std::vector<Key>>, uint32_t>> acc(3);
    for (const AtlasIdealCharacter& c : chars) {
        if (!c.local) {
            why = "ideal " + std::to_string(c.ideal) + " (" + c.group + ") is not certified local";
            return false;
        }
        if (!c.anemic_generates) {
            why = "ideal " + std::to_string(c.ideal) + " (" + c.group
                  + "): the anemic system does not generate its residue field";
            return false;
        }
        ENGINE_CHECK_MSG(c.r > 0 && c.dim % c.r == 0,
                         "AtlasResidueMultisets: a block dimension not divisible by r");
        acc[c.degree][std::make_pair(c.r, c.orbit)] += c.dim / c.r;
    }
    for (uint32_t q = 0; q < 3; ++q) {
        for (const auto& kv : acc[q]) {
            ResidueCharacterClass k;
            k.r = kv.first.first;
            k.orbit = kv.first.second;
            k.multiplicity = kv.second;
            out[q].push_back(std::move(k));
        }
    }
    return true;
}

// ===========================================================================
// The store.
// ===========================================================================

std::string AtlasStore::to_json() const {
    std::ostringstream o;
    o << "{\"schema\":";
    WriteStr(o, schema);
    o << ",\"ell\":" << ell << ",\"n_max\":" << n_max << ",\"settings_fingerprint\":";
    WriteStr(o, settings_fingerprint);
    o << ",\"primes\":";
    WriteU64List(o, primes);
    o << ",\n\"fields\":[";
    for (size_t i = 0; i < fields.size(); ++i) {
        o << (i ? "," : "") << "{\"r\":" << fields[i].r << ",\"C\":";
        WriteU32List(o, fields[i].C);
        o << '}';
    }
    o << "],\n\"cells\":[";
    for (size_t i = 0; i < cells.size(); ++i) {
        const AtlasCell& c = cells[i];
        o << (i ? ",\n" : "\n") << "{\"n\":" << c.n << ",\"dims\":";
        WriteU32List(o, c.dims);
        o << ",\"ideals\":" << c.ideals << ",\"local_ideals\":" << c.local_ideals
          << ",\"u_ell\":{\"included\":" << (c.u_ell_included ? "true" : "false")
          << ",\"reason\":";
        WriteStr(o, c.u_ell_reason);
        o << "}}";
    }
    o << "],\n\"systems\":[";
    for (size_t i = 0; i < systems.size(); ++i) {
        const AtlasSystem& s = systems[i];
        o << (i ? ",\n" : "\n") << "{\"id\":" << s.id << ",\"r\":" << s.r << ",\"eigenvalues\":";
        WriteKey(o, s.eigenvalues);
        o << ",\"type\":";
        WriteStr(o, s.type);
        o << ",\"chi\":";
        if (s.has_chi) o << '[' << s.chi_a << ',' << s.chi_b << ']';
        else o << "null";
        o << ",\"base\":" << s.base << ",\"twist\":" << s.twist
          << ",\"collected\":" << (s.collected ? "true" : "false")
          << ",\"in_H1\":" << (s.in_H1 ? "true" : "false") << ",\"occurrences\":[";
        for (size_t k = 0; k < s.occurrences.size(); ++k) {
            const AtlasOccurrence& oc = s.occurrences[k];
            o << (k ? "," : "") << "{\"n\":" << oc.n << ",\"ideal\":" << oc.ideal << ",\"group\":";
            WriteStr(o, oc.group);
            o << ",\"dim\":" << oc.dim << ",\"a_ell\":";
            if (oc.a_ell.known) {
                o << "{\"status\":\"known\",\"value\":";
                WriteU32List(o, oc.a_ell.value);
                o << ",\"reason\":";
                WriteStr(o, oc.a_ell.reason);
                o << '}';
            } else {
                o << "{\"status\":\"unknown\",\"reason\":";
                WriteStr(o, oc.a_ell.reason);
                o << '}';
            }
            o << '}';
        }
        o << "]}";
    }
    o << "],\n\"chi_count\":" << chi_count << ",\"chi_in_sigma\":" << chi_in_sigma
      << ",\n\"s_sep\":";
    WriteSep(o, s_sep);
    o << ",\n\"s0_anemic\":";
    WriteSep(o, s0_anemic);
    o << "}\n";
    return o.str();
}

AtlasStore ParseAtlasStore(const std::string& json) {
    const JVal root = JParser(json).parse();
    AtlasStore st;
    st.schema = JS(root.at("schema"));
    if (st.schema != kAtlasSchema)
        throw std::runtime_error("hecke_atlas: unknown schema '" + st.schema + "'");
    st.ell = JU32(root.at("ell"));
    st.n_max = JU32(root.at("n_max"));
    st.settings_fingerprint = JS(root.at("settings_fingerprint"));
    st.primes = JU64List(root.at("primes"));
    for (const JVal& f : JA(root.at("fields"))) {
        AtlasField af;
        af.r = JU32(f.at("r"));
        af.C = JU32List(f.at("C"));
        st.fields.push_back(af);
    }
    for (const JVal& c : JA(root.at("cells"))) {
        AtlasCell ac;
        ac.n = JU32(c.at("n"));
        ac.dims = JU32List(c.at("dims"));
        ac.ideals = JU32(c.at("ideals"));
        ac.local_ideals = JU32(c.at("local_ideals"));
        ac.u_ell_included = JB(c.at("u_ell").at("included"));
        ac.u_ell_reason = JS(c.at("u_ell").at("reason"));
        st.cells.push_back(ac);
    }
    for (const JVal& s : JA(root.at("systems"))) {
        AtlasSystem as;
        as.id = JU32(s.at("id"));
        as.r = JU32(s.at("r"));
        as.eigenvalues = JKey(s.at("eigenvalues"));
        as.type = JS(s.at("type"));
        const JVal& chi = s.at("chi");
        if (chi.kind != JVal::Null) {
            const std::vector<uint32_t> ab = JU32List(chi);
            if (ab.size() != 2u) throw std::runtime_error("hecke_atlas: chi is a pair");
            as.has_chi = true;
            as.chi_a = ab[0];
            as.chi_b = ab[1];
        }
        as.base = JU32(s.at("base"));
        as.twist = JU32(s.at("twist"));
        as.collected = JB(s.at("collected"));
        as.in_H1 = JB(s.at("in_H1"));
        for (const JVal& oc : JA(s.at("occurrences"))) {
            AtlasOccurrence ao;
            ao.n = JU32(oc.at("n"));
            ao.ideal = JU32(oc.at("ideal"));
            ao.group = JS(oc.at("group"));
            ao.dim = JU32(oc.at("dim"));
            const JVal& a = oc.at("a_ell");
            const std::string& status = JS(a.at("status"));
            if (status == "known") {
                ao.a_ell.known = true;
                ao.a_ell.value = JU32List(a.at("value"));
            } else if (status != "unknown") {
                throw std::runtime_error("hecke_atlas: a_ell status '" + status + "'");
            }
            ao.a_ell.reason = JS(a.at("reason"));
            as.occurrences.push_back(ao);
        }
        st.systems.push_back(std::move(as));
    }
    st.chi_count = JU32(root.at("chi_count"));
    st.chi_in_sigma = JU32(root.at("chi_in_sigma"));
    st.s_sep = JSep(root.at("s_sep"));
    st.s0_anemic = JSep(root.at("s0_anemic"));
    return st;
}

std::string WriteAtlasStoreIfRequested(const AtlasStore& s, const std::string& suffix) {
    const char* env = std::getenv("HECKE_ATLAS_OUT");
    if (env == nullptr || std::string(env).empty()) return std::string();
    namespace fs = std::filesystem;
    const fs::path dir(env);
#ifdef OC_SOURCE_DIR
    {
        // Slot-6 ruling 4 (i): not under docs/matrix-engine/results/ until
        // the campaign publishes.
        const fs::path results = fs::weakly_canonical(fs::path(OC_SOURCE_DIR) / "docs"
                                                      / "matrix-engine" / "results");
        const std::string d = fs::weakly_canonical(dir).string();
        const std::string rs = results.string();
        if (d.compare(0, rs.size(), rs) == 0)
            throw std::runtime_error("HECKE_ATLAS_OUT: the atlas store may not be written under "
                                     + rs + " (slot-6 owner ruling 4 (i); slot-7 confirmation 14: stores are COPIED into results/ by the publication commit, the guard stays)");
    }
#endif
    // Created only AFTER the refusal above, so a refused path leaves nothing.
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) throw std::runtime_error("HECKE_ATLAS_OUT: cannot create " + dir.string() + ": "
                                     + ec.message());
    const fs::path file = dir / ("hecke_atlas_ell" + std::to_string(s.ell) + suffix + ".json");
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("HECKE_ATLAS_OUT: cannot open " + file.string());
    out << s.to_json();
    if (!out) throw std::runtime_error("HECKE_ATLAS_OUT: write failed on " + file.string());
    return file.string();
}

} // namespace hecke_ext
