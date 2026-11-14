// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZRecord -- slot-25 HZ-REC: the record integral_hecke_structure/v1 (see the header and docs/implementation/slot25/HZREC_report.md).

#include "HeckeZRecord.h"

#include "HeckeZEisUp.h"
#include "HeckeZEisUpChar.h"
#include "HeckeZTor.h"
#include "HeckeZUTag.h"
#include "HeckeZUp.h"
#include "Descent/DescentModuleExport.h"
#include "Descent/DescentRecord.h"
#include "../Hecke/FpPoly.h"
#include "../Hecke/HeckeSettings.h"
#include "../NumberField/Sha256.h"
#include "../Util/CancelPoint.h"   // (NBCONFIRM S5) cooperative cancellation points
#include "../Util/Rss.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace heckez {
namespace record {

namespace {

using local::LatticeModule;
using local::LocalRead;
using local::Zlk;
namespace drec = integral_descent::record;

std::string Hex64(uint64_t v) {
    std::ostringstream o;
    o << "0x" << std::hex << std::setw(16) << std::setfill('0') << v;
    return o.str();
}

std::string JsonEsc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += c;
        } else if (c == '\n') {
            o += "\\n";
        } else if (static_cast<unsigned char>(c) < 0x20) {
            o += ' ';
        } else {
            o += c;
        }
    }
    return o;
}

template <class T>
std::string Join(const std::vector<T>& v, const char* sep = ",") {
    std::ostringstream o;
    for (size_t i = 0; i < v.size(); ++i) o << (i ? sep : "") << v[i];
    return o.str();
}

std::string Sha256Hex(const std::string& s) {
    nf::Sha256 h;
    h.update(s.data(), s.size());
    return h.hex();
}

bool IsPrimeU32(uint32_t p) {
    if (p < 2) return false;
    for (uint32_t d = 2; d * d <= p; ++d)
        if (p % d == 0) return false;
    return true;
}

// ---- limb products (FMTMP's limb output route: base-2^32 limbs, word multiply-add only) ------------------------------------------
using Limbs = std::vector<uint32_t>;   // little-endian
void LimbMul(Limbs& a, uint64_t m) {   // a *= m, m < 2^64 (two 32-bit passes)
    const uint32_t parts[2] = {static_cast<uint32_t>(m & 0xffffffffu), static_cast<uint32_t>(m >> 32)};
    Limbs out(a.size() + 3, 0);
    for (int h = 0; h < 2; ++h) {
        if (parts[h] == 0) continue;
        uint64_t carry = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const uint64_t t = uint64_t(a[i]) * parts[h] + out[i + h] + carry;
            out[i + h] = static_cast<uint32_t>(t);
            carry = t >> 32;
        }
        for (size_t i = a.size() + h; carry; ++i) {
            const uint64_t t = uint64_t(out[i]) + carry;
            out[i] = static_cast<uint32_t>(t);
            carry = t >> 32;
        }
    }
    while (out.size() > 1 && out.back() == 0) out.pop_back();
    a = out;
}
std::string LimbHex(const Limbs& a) {
    std::ostringstream o;
    o << "0x";
    size_t i = a.size();
    while (i > 1 && a[i - 1] == 0) --i;
    o << std::hex << a[i - 1];
    for (size_t j = i - 1; j-- > 0;) o << std::setw(8) << std::setfill('0') << a[j];
    return o.str();
}

// ---- vocabularies ------------------------------------------------------------------------------------------------------------------
struct StatusRow {
    const char* word;
    StatusClass cls;
};
const std::vector<StatusRow>& StatusTable() {
    static const std::vector<StatusRow> t = {
        {"certified", StatusClass::Final},
        {"upper_bound_certified", StatusClass::Final},
        {"non_reduced_certified", StatusClass::Final},
        {"bounds", StatusClass::Final},
        {"zero", StatusClass::Final},
        {"distinct_certified", StatusClass::Final},
        {"common_certified", StatusClass::Final},
        {"agrees_on_P", StatusClass::Final},
        {"partial(unfactored)", StatusClass::Final},
        {"open(sandwich)", StatusClass::Final},
        {"open(cap)", StatusClass::Final},
        {"open(precision)", StatusClass::Final},
        {"refused(by_theory)", StatusClass::Final},
        {"refused(word_modulus)", StatusClass::Final},
        {"refused(no_lift)", StatusClass::Final},
        {"refused(precision)", StatusClass::Final},
        {"not_run", StatusClass::Final},
        {"blocked(export)", StatusClass::Pending},
        {"blocked(U_p)", StatusClass::Pending},
        {"refused(export_word)", StatusClass::Pending},
        {"refused(k1_route)", StatusClass::Pending},
        // (HZ-RAT-PERF (a), H1 audit F2) the word-prime list of the rational step or of a U_p tag would exceed its count cap: pending
        // (only a larger cap, i.e. a later version, lifts it -- a faster CRT does not, the cap counts primes); the record is refused,
        // G1.C2_status carries the word ([C] v1.25.0-pre.7 sec:recordschema item 8 (c); the schema amendment A1 after the freeze,
        // HZREC_report.md section 11)
        {"refused(word_prime_cap)", StatusClass::Pending},
        // (slot-26 Z1, amendment A2 after the freeze, hash-neutral; [C] 1.26.0-pre.3 item 8 (c), (d')(ii)) a full record at N > 1 whose
        // U_p tag certifies T (x) Q non-reduced: the full (C1) is false for every theta (no search runs), final
        {"refused(non_reduced)", StatusClass::Final},
        // (schema audit F7) `pending(HZ-TOR)` is gone: a Tor read the export refuses is `blocked(export)` (or its named word, F8)
    };
    return t;
}

bool IsDecimal(const std::string& s) {
    if (s.empty()) return false;
    size_t i = s[0] == '-' ? 1 : 0;
    if (i == s.size()) return false;
    for (; i < s.size(); ++i)
        if (s[i] < '0' || s[i] > '9') return false;
    return true;
}
bool IsLowerHex(const std::string& s, size_t from) {
    if (from >= s.size()) return false;
    for (size_t i = from; i < s.size(); ++i)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return false;
    return true;
}

}  // namespace

StatusClass ClassOfStatus(const std::string& s) {
    for (const StatusRow& r : StatusTable())
        if (s == r.word) return r.cls;
    return StatusClass::Unknown;
}

const std::vector<std::string>& StatusWords() {
    static const std::vector<std::string> w = [] {
        std::vector<std::string> v;
        for (const StatusRow& r : StatusTable()) v.push_back(r.word);
        return v;
    }();
    return w;
}

namespace {
const std::map<std::string, std::vector<std::string>>& EnumTable() {
    static const std::map<std::string, std::vector<std::string>> m = {
        {"overall", {"final", "pending", "refused"}},
        {"algebra", {"full", "anemic"}},
        {"d_source", {"hES_N", "hES_N_an"}},
        {"reduced_Q", {"true", "non_reduced_certified", "undecided", "blocked(U_p)"}},
        {"C1", {"certified", "open(cap)", "blocked(U_p)", "refused(non_reduced)"}},   // (A2) refused(non_reduced)
        {"cofactor_test", {"one", "squarefree_by_size", "unfactored", "upper_bound"}},
        {"C2", {"complete", "partial(unfactored)", "blocked(U_p)", "refused(word_prime_cap)", "not_run"}},   // (A2) not_run
        {"prime_proof", {"small", "pratt", "probable"}},
        // (schema audit F16) v1_pending_LC3b is not a reason any more: v_disc = 1 is hashed beside it, the S rule is in the caps text
        {"S_reason", {"disc_square", "disc_factor", "torsion", "congruence"}},
        {"theory", {"H", "H_c"}},
        {"generation", {"certified", "open(sandwich)", "blocked(export)", "blocked(U_p)", "refused(by_theory)", "refused(word_modulus)",
                        "refused(no_lift)", "refused(export_word)", "refused(k1_route)"}},
        {"route", {"saturation", "eis_single", "rational", "cyclic", "commutant", "AK_unsaturated", "mazur"}},
        // (F7, F8) the export's words by name: blocked(export), refused(no_lift) (final), refused(export_word) (pending)
        {"support", {"free", "torsion", "both", "not_run", "blocked(export)", "refused(no_lift)", "refused(export_word)"}},
        {"degree_attr", {"not_run"}},   // (F12) v1 does not compute the degree attribution of prop:heckeZuct(4)
        {"local_kind", {"etale", "monogenic", "non_monogenic"}},
        {"bool", {"true", "false"}},
        // (F10) closed lists of the component and partition fields
        {"component_type", {"eis", "cusp"}},
        {"irreducible", {"true", "open(cap)"}},
        {"partition", {"eisenstein", "U_p_tags", "general", "none(C1_open)", "none(single_component)"}},
        {"pair_kind", {"meeting", "ramified_dvr", "non_maximal_order", "not_run"}},
        {"through", {"certified", "blocked(export)", "not_run"}},
        {"mi_status", {"certified", "not_run", "blocked(export)", "blocked(U_p)"}},
        {"ct_status", {"certified", "upper_bound_certified", "partial(unfactored)", "blocked(export)", "blocked(U_p)", "not_run"}},
        {"ct_ell", {"certified", "upper_bound_certified", "open(precision)", "refused(no_lift)", "refused(export_word)", "refused(k1_route)",
                    "blocked(export)"}},
        {"read_word", {"certified", "upper_bound_certified", "open(precision)", "refused(word_modulus)", "refused(no_lift)",
                       "refused(export_word)", "refused(k1_route)", "blocked(export)", "not_run"}},
        {"sign", {"+", "-", "both"}},
        {"fibre_product", {"certified", "not_certified"}},
        {"general_case", {"bounds", "from_certified_T"}},
        {"agrees", {"true", "false", "not_read"}},
        {"J", {"zero", "bounds", "blocked(export)", "refused(no_lift)", "refused(export_word)"}},
        {"relation_to_free", {"distinct_certified", "common_certified", "agrees_on_P", "blocked(export)", "refused(no_lift)",
                              "refused(export_word)", "not_run"}},
        {"coincidence_route", {"T_ell_level1", "nonsplit"}},
        {"generation_tors", {"not_run"}},   // (F12) v1 does not compute it
        // (F1) split_on_P: split over T' = Z[T_p : p in the read set]; `split` (for T) is kept for a later theorem and not written
        {"split", {"split_by_support", "split", "split_on_P", "non_split", "blocked(export)", "refused(no_lift)", "refused(export_word)",
                   "not_run"}},
        {"not_run", {"not_run"}},
        // (slot-25 H1) G8 under heckez.G8 = hida
        {"g8_scope", {"level1", "not_run(level_N)"}},
        {"g8_blocks", {"read", "not_run(no_decomposition)", "not_run(class_zero)", "not_run(forms_cap)"}},
        {"g8_structure", {"fibre_LxL", "regular_deg_g", "open"}},
        {"i0_generator", {"none_found"}},   // or T_<prime> (ValueInVocab)
        {"t_ell", {"in_P", "added", "omitted(cap:ell>23170)", "omitted(ell|N:U_ell_not_exported)"}},
        {"utag", {"upper_bound_certified", "blocked(U_p)", "not_run"}},   // (S26 Z2) not_run: the tag certifies no bound, final
        {"bezout", {"by_generator"}},
        {"lean_cap", {"in", "absent(cap)"}},
        {"cert_absent", {"absent(not_built)", "absent(not_applicable)", "absent(H1)", "absent(no_theta_read)"}},
    };
    return m;
}
}  // namespace

const std::vector<std::string>& EnumVocabNames() {
    static const std::vector<std::string> v = [] {
        std::vector<std::string> o;
        for (const auto& [k, w] : EnumTable()) o.push_back(k);
        return o;
    }();
    return v;
}

const std::vector<std::string>& EnumValues(const std::string& vocab) {
    const auto& m = EnumTable();
    static const std::vector<std::string> empty;
    const auto it = m.find(vocab);
    return it == m.end() ? empty : it->second;
}

bool ValueInVocab(const std::string& vocab, const std::string& value) {
    if (vocab.empty()) return true;
    if (vocab == "status") return ClassOfStatus(value) != StatusClass::Unknown;
    if (vocab == "int") return IsDecimal(value);
    if (vocab == "zbig") return drec::ExactHexCanonical(value);
    if (vocab == "hex64") return value.size() == 18 && value.compare(0, 2, "0x") == 0 && IsLowerHex(value, 2);
    if (vocab == "sha256") return value.size() == 64 && IsLowerHex(value, 0);
    if (vocab.compare(0, 5, "list:") == 0) {
        const std::string sub = vocab.substr(5);
        if (value.empty()) return true;
        std::stringstream ss(value);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!ValueInVocab(sub, tok)) return false;
        return true;
    }
    if (vocab == "op" || vocab == "op_or_theta" || vocab == "op_or_joint") {   // T_<prime> (F10: p0, the distinctness witness)
        if (vocab == "op_or_theta" && value == "theta") return true;
        if (vocab == "op_or_joint" && value == "joint") return true;
        if (value.size() < 3 || value.compare(0, 2, "T_") != 0 || !IsDecimal(value.substr(2)) || value[2] == '-' || value[2] == '0') return false;
        const unsigned long long p = std::stoull(value.substr(2));
        return p < (1ull << 32) && IsPrimeU32(static_cast<uint32_t>(p));
    }
    if (vocab == "i0_generator") return value == "none_found" || ValueInVocab("op", value);   // (H1) T_<prime> | none_found
    if (vocab == "chain_key") {   // (H1) (N_tame,ell,k mod (ell - 1)), three decimals
        if (value.size() < 7 || value.front() != '(' || value.back() != ')') return false;
        std::stringstream ss(value.substr(1, value.size() - 2));
        std::string tok;
        int n = 0;
        while (std::getline(ss, tok, ',')) {
            if (!IsDecimal(tok)) return false;
            ++n;
        }
        return n == 3;
    }
    const std::vector<std::string>& vals = EnumValues(vocab);
    if (vals.empty()) return false;   // an unknown vocabulary name is itself a defect
    if (std::find(vals.begin(), vals.end(), value) != vals.end()) return true;
    if (vocab == "t_ell") {   // omitted(export:<Name>), <Name> = ExportRefusalName only (Z0 referee F6)
        const std::string pre = "omitted(export:";
        if (value.compare(0, pre.size(), pre) != 0 || value.back() != ')') return false;
        const std::string name = value.substr(pre.size(), value.size() - pre.size() - 1);
        if (name.empty()) return false;
        for (char c : name)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
        return true;
    }
    return false;
}

std::string ExportRefusalNameOf(const std::string& text) {
    const std::string pre = "REFUSED [";
    if (text.compare(0, pre.size(), pre) != 0) return std::string();
    const size_t e = text.find(']', pre.size());
    if (e == std::string::npos) return std::string();
    const std::string name = text.substr(pre.size(), e - pre.size());
    namespace mx = integral_descent::modexport;
    for (int c = static_cast<int>(mx::ExportRefusal::None); c <= static_cast<int>(mx::ExportRefusal::DomainCap); ++c)
        if (name == mx::ExportRefusalName(static_cast<mx::ExportRefusal>(c))) return name;
    return std::string();
}

std::string LiftInnerCodeOf(const std::string& text) {
    if (ExportRefusalNameOf(text) != "Lift") return std::string();
    const std::string mark = "the strict lift is refused [";
    const size_t at = text.find(mark);
    if (at == std::string::npos) return std::string();
    const size_t b = at + mark.size(), e = text.find(']', b);
    return e == std::string::npos ? std::string() : text.substr(b, e - b);
}

std::string ExportWord(const std::string& text) {
    const std::string name = ExportRefusalNameOf(text);
    if (name == "Lift") {
        // (F8; re-audit R6) refused(no_lift) is final only when the lift is refused by a CERTIFICATE (D4's pair-lift codes
        // LiftCertificate, IsobClosure, StrictIdentity, InputTableCertificate): no lift exists on the route.  A cap of the lift
        // (RssCeiling, IntegerCap, DomainCap, ...) or a D3 refusal FromD3 sends to Lift without a pair code is a limit of the route,
        // pending: blocked(export).
        static const std::set<std::string> kCert = {"LiftCertificate", "IsobClosure", "StrictIdentity", "InputTableCertificate"};
        return kCert.count(LiftInnerCodeOf(text)) ? "refused(no_lift)" : "blocked(export)";
    }
    if (name == "ModulePrecision") return "refused(export_word)";   // EXPORT's ring limit: pending (EXPORT-Z step 2)
    return "blocked(export)";
}

// ---- (slot-27 EXW) the chi_Z check of a free-route module ---------------------------------------------------------------------------
bool FreeReadChiAgrees(const local::LatticeModule& M, const std::map<uint32_t, std::vector<std::string>>& chi_z, std::string* why) {
    if (!M.ok || M.T.size() != M.P.size()) {
        if (why) *why = "the module is not served";
        return false;
    }
    const Zlk R = Zlk::Make(M.ell, M.K);
    // a canonical signed hex ("0x..", "-0x..") or decimal integer, reduced mod ell^K
    auto residue = [&R](const std::string& t, uint64_t* out) -> bool {
        size_t i = 0;
        const bool neg = !t.empty() && t[0] == '-';
        if (neg) i = 1;
        const bool hex = t.compare(i, 2, "0x") == 0;
        if (hex) i += 2;
        if (i >= t.size()) return false;
        const unsigned __int128 q = R.q;
        unsigned __int128 acc = 0;
        for (; i < t.size(); ++i) {
            const char c = t[i];
            uint32_t d;
            if (c >= '0' && c <= '9') d = static_cast<uint32_t>(c - '0');
            else if (hex && c >= 'a' && c <= 'f') d = static_cast<uint32_t>(c - 'a' + 10);
            else return false;
            if (d >= (hex ? 16u : 10u)) return false;
            acc = (acc * (hex ? 16u : 10u) + d) % q;
        }
        const uint64_t v = static_cast<uint64_t>(acc);
        *out = neg ? R.neg(v) : v;
        return true;
    };
    for (size_t i = 0; i < M.P.size(); ++i) {
        const auto it = chi_z.find(M.P[i]);
        if (it == chi_z.end() || it->second.size() != size_t(M.D) + 1) continue;
        const local::Poly C = local::Charpoly(R, M.T[i], M.D);   // low -> high, monic (as chi_z)
        bool ok = C.size() == it->second.size();
        std::string at;
        for (size_t j = 0; ok && j < C.size(); ++j) {
            uint64_t r = 0;
            const std::string& t = it->second[j];
            if (t.empty() || t[0] == '[' || t == "?") continue;   // a coefficient beyond 2^63 in the descent's short text: not compared
            ok = residue(t, &r) && r == C[j];
            if (!ok) at = " at x^" + std::to_string(j) + ": " + std::to_string(C[j]) + " vs " + t;
        }
        if (!ok) {
            if (why) *why = "chi(T_" + std::to_string(M.P[i]) + ") mod ell^K differs from the descent's chi_Z (ell = " + std::to_string(M.ell) +
                            ", K = " + std::to_string(M.K) + ")" + at;
            return false;
        }
    }
    return true;
}

// ---- settings ----------------------------------------------------------------------------------------------------------------------
namespace {
// (slot-25 H1) the G8 settings line.  The `off` line is the frozen default (schema audit F2) and must stay byte for byte; `hida` is
// the non-default value of H1, with its rule in full (count-based caps only).
std::string G8Line(const std::string& g8) {
    if (g8 == "off") return "heckez.G8 = off (default off: no G8 field; another value is H1's Hida block per maximal ideal)\n";
    if (g8 == "hida")
        return "heckez.G8 = hida (H1, [U] sec:u-eislambda: level 1 only (N > 1: G8.scope = not_run(level_N)); per ell >= 5 of S whose "
               "maximal ideals are decomposed and per maximal ideal m of G2: ordinary = true at the Eisenstein ideal (by theorem), else "
               "iff the theta-residue of m divides chi(theta | e M_k) mod ell (T_ell = U on e M_k); at an ordinary Eisenstein ideal "
               "through the cuspidal component: chain_key = (1,ell,k mod (ell-1)), g = rank of the ordinary Eisenstein block of e S_k "
               "(Fitting split by the full residual system: T_p, p <= k/12 + 1 and p in {2,3,5,7}, and U), L_k = G3's certified "
               "C_T ell-length, v_Bk = v_ell(B_k/k) mod ell^W, structure = fibre_LxL iff g = 1, regular_deg_g iff g >= 2 and some "
               "T_q - e_q generates the block mod ell with v_ell det(T_q - e_q | block) = v_Bk and v(G(x_k) - G(x_{k+ell-1})) = 1, "
               "else open, I0_generator = the first such T_q (q ascending, q != ell) else none_found; forms side mod ell^W, "
               "W = max(2, upper_v + 2), served iff ell^W < 2^84, k - 1 >= W and the q-expansion length max(ell, k/12 + 1, 7, theta's operators) (dim S_k + 2) + 1 <= "
               "2^18, else blocks = not_run(forms_cap); (ell - 1) | k: blocks = not_run(class_zero))\n";
    throw std::invalid_argument("integral_hecke_structure/v1: heckez.G8 = " + g8 + " is not a value (off | hida)");
}
}  // namespace

std::string RecordSettings::caps_text() const {
    // (schema audit F11) one `rho=` key: HZ-RAT's word factoriser (`rho=nf::factor_u64(...)`); the multi-limb rho / ECM budget is
    // `multilimb_rho=none;ecm=none`.  The word primes (`word_primes=CrtPrimes(<2^30)`) are provenance ([C] item 8 (i)): no certified
    // value depends on which word primes carry the residues, so they leave the hashed text (G10.rat.word_primes).
    std::string rat_text = rat.canonical_text();
    const std::string wp = ";word_primes=";
    const size_t at = rat_text.find(wp);
    ENGINE_CHECK_MSG(at != std::string::npos && rat_text.find(';', at + 1) == std::string::npos,
                     "HeckeZRecord: HZ-RAT's canonical text no longer ends with its word_primes key");
    rat_text.erase(at);
    std::ostringstream o;
    o << "heckez.schema = " << kSchema << "\n"
      << "heckez.P_rule = descent primes (the first three of 2, 3, 5, 7, 11, 13 not dividing N), enlarged by the next primes not dividing N "
         "until (C1) is certified, |P| <= "
      << p_rule_cap << "; read set at ell = P u {T_ell}, ell !| N, ell <= " << local::kTEllCap << "\n"
      << "heckez.theta_search+factor_budget = " << rat_text << ";pool=fixed_point;multilimb_rho=none;ecm=none\n"
      << "heckez.S_rule = every certified prime of gcd_j D_j (v_disc = 1 included: no LC-3b drop rule in v1), the torsion primes of the "
         "descent, the certified primes of g_P (or of the U_p bounds)\n"
      << "heckez.K_read = start max(" << K_start_min << ", v_ell c(lambda) + 1), escalation steps <= " << K_steps
      << "; (C3) at K = 1 when the K0 read is not served\n"
      << "heckez.maximal_ideals = decomposed iff generation is certified by saturation or rational and theta mod ell generates A'_1 "
         "(Krylov rank = dim A'_1 = d), theta = the read's own generating theta, else the first of theta_search's order over P, else "
         "(theta_search = P u {T_ell}) the first of theta_search's order over the read set P u {T_ell} when t_ell = added"
      << (rat.theta_order != ThetaOrder::Lex
              ? "; else (theta_order = lex_then_greedy) the greedy continuation over P, then over the read set (score: Krylov rank, target "
                "t mod ell, c <= min(binom(d,2), ell - 1), at most theta_cap tests per list)"
              : "")
      << "; else not_run (eis_single: not_run)\n"
      << "heckez.irreducible = c_theta irreducible mod q for some prime q >= 3, the first " << irred_primes << " primes tried in order\n"
      << "heckez.second_route = v_ell(B_k/k) mod ell^(e+2), e = v_ell(g_P), level 1, ell >= 5, (ell-1) !| k, BernoulliOverK; served iff "
         "ell^(e+5) < 2^"
      << kSecondRouteBits << ", ell^(e+2) + ell <= 2^" << kSecondRouteRingBits << ", ell^(e+2+V) + ell <= 2^" << kSecondRouteTangentBits
      << " (V = v_ell(4^(k/2) - 1)) and v < e+2; else v_Bk = not_run, agrees = not_read\n"
      << G8Line(g8)
      << "heckez.bezout_record_cap = d <= " << bezout_cap_d << ", " << bezout_cap_kib << " KiB (v1 writes by_generator)\n";
    return o.str();
}

std::string RecordId(uint32_t N, uint32_t n, const std::string& algebra, const std::string& fp) {
    return "hz_N" + std::to_string(N) + "_n" + std::to_string(n) + "_" + algebra + "_s" + fp.substr(2);
}

// ---- the record ------------------------------------------------------------------------------------------------------------------
namespace {
bool Hashed(const std::string& g) {
    return g == "G0" || g == "G1" || g == "G2" || g == "G3" || g == "G4" || g == "G5" || g == "G6" || g == "G7" || g == "G8";
}
}  // namespace

std::string HeckeStructureRecord::body() const {
    std::ostringstream o;
    for (const Field& f : fields)
        if (Hashed(f.group)) o << f.path << "\t" << f.value << "\n";
    o << "settings.text\t" << settings_text;
    o << "settings.fingerprint\t" << settings_fingerprint << "\n";
    return o.str();
}

std::string HeckeStructureRecord::cert_text() const {
    std::ostringstream o;
    for (const Field& f : fields)
        if (f.group == "G9") o << f.path << "\t" << f.value << "\n";
    return o.str();
}

const Field* HeckeStructureRecord::find(const std::string& path) const {
    for (const Field& f : fields)
        if (f.path == path) return &f;
    return nullptr;
}

void HeckeStructureRecord::rehash() {
    if (refusal.empty()) {
        bool pend = false;
        for (const Field& f : fields)
            if (Hashed(f.group) && f.path != "G0.status" && ClassOfStatus(f.value) == StatusClass::Pending) pend = true;
        status = pend ? "pending" : "final";
    } else {
        status = "refused";
    }
    for (Field& f : fields)
        if (f.path == "G0.status") f.value = status;
    record_fnv1a64 = Hex64(engine::FnvBytes(engine::kFnvOffset64, body()));
    cert_sha256 = Sha256Hex(cert_text());
}

bool HeckeStructureRecord::golden_eligible() const {
    return status == "final" && settings_text.find("\n" + G8Line("off")) != std::string::npos;
}

std::string HeckeStructureRecord::to_json() const {
    std::ostringstream o;
    o << "{\"schema\":\"" << kSchema << "\",\"id\":\"" << id << "\",\"N\":" << N << ",\"n\":" << n << ",\"algebra\":\"" << algebra
      << "\",\"status\":\"" << status << "\",\"refusal\":\"" << JsonEsc(refusal) << "\",\"record_fnv1a64\":\"" << record_fnv1a64
      << "\",\"cert_sha256\":\"" << cert_sha256 << "\",\"settings_fingerprint\":\"" << settings_fingerprint << "\",\"settings\":\""
      << JsonEsc(settings_text) << "\",\"descent_ref\":{\"record_fnv1a64\":\"" << descent.record_fnv1a64 << "\",\"cell_sha256\":\""
      << descent.cell_sha256 << "\",\"primes\":[" << Join(descent.primes) << "]}";
    auto section = [&](const char* name, auto pred) {
        o << ",\"" << name << "\":[";
        bool first = true;
        for (const Field& f : fields) {
            if (!pred(f.group)) continue;
            o << (first ? "" : ",") << "[\"" << f.group << "\",\"" << JsonEsc(f.path) << "\",\"" << JsonEsc(f.value) << "\",\""
              << JsonEsc(f.vocab) << "\",\"" << JsonEsc(f.parent) << "\"]";
            first = false;
        }
        o << "]";
    };
    section("body", [](const std::string& g) { return Hashed(g); });
    section("cert", [](const std::string& g) { return g == "G9"; });
    section("provenance", [](const std::string& g) { return g == "G10" || g == "G0x"; });
    o << ",\"golden_eligible\":" << (golden_eligible() ? "true" : "false") << "}";
    return o.str();
}

// ---- the builder ------------------------------------------------------------------------------------------------------------------
namespace {

struct Builder {
    HeckeStructureRecord& R;
    void add(const std::string& g, const std::string& path, const std::string& value, const std::string& vocab = "",
             const std::string& parent = "") {
        // (the canonical texts are `path<TAB>value` lines: neither may contain a TAB or a newline)
        ENGINE_CHECK_MSG(path.find_first_of("\t\n") == std::string::npos && value.find_first_of("\t\n") == std::string::npos,
                         "HeckeZRecord: a TAB or newline in a field");
        R.fields.push_back(Field{g, path, value, vocab, parent});
    }
};

int32_t CertVal(const Factored& F, uint64_t ell) {
    for (const auto& pp : F.factors)
        if (pp.q == ell && pp.proof != "probable") return static_cast<int32_t>(pp.e);
    return F.complete() ? 0 : -1;
}

local::RationalInput RatFor(const RationalResult& R, uint64_t ell) {
    local::RationalInput ri;
    if (R.theta_index < 0 || !R.eis_rational) return ri;
    const auto& t = R.thetas[static_cast<size_t>(R.theta_index)];
    ri.present = true;
    ri.theta = t.coeffs;
    ri.lambda_hex = t.lambda_hex;
    ri.c_hex = R.cusp_part;
    ri.v_c_lambda = CertVal(t.eis_res, ell);
    // (S27 TS-FIX, TS_design.md Sec. 5.4 item 2) without a g_P bound (open(no primitive T_p)) the read gets no g_P: it uses its own
    // local value (v_gP_local) or none, by its own rule -- never the default Factored's "1"
    const bool gp_bound = GPIsBound(R);
    ri.v_gP = gp_bound ? CertVal(R.gP, ell) : -1;
    ri.gP_complete = gp_bound && R.gP.complete();
    return ri;
}

std::string DumpText(const LatticeModule& M) {
    std::ostringstream o;
    o << "N\t" << M.N << "\nn\t" << M.n << "\nell\t" << M.ell << "\nK\t" << M.K << "\nD\t" << M.D << "\nP\t" << Join(M.P) << "\n";
    for (size_t i = 0; i < M.T.size(); ++i) {
        o << "T_" << M.P[i] << "\t";
        for (size_t r = 0; r < M.D; ++r) {
            o << (r ? ";" : "");
            for (size_t c = 0; c < M.D; ++c) o << (c ? "," : "") << M.T[i][r * M.D + c];
        }
        o << "\n";
    }
    return o.str();
}

// a module's refusal word for the at-root read
std::string RootRefusal(const LatticeModule& M) {
    const std::string& s = M.status;
    if (s == "open(precision)") {
        // (Z0 merge report Sec. 7, F4) the export's K_max, not the step cap: EXPORT's ring limit, the pending export_word
        ENGINE_CHECK_MSG(M.source == "export", "HeckeZRecord: open(precision) from a non-export module");
        return "refused(export_word)";
    }
    if (s == "blocked(export)") return M.source == "export" ? ExportWord(M.detail) : s;   // (F8) mapped by the refusal's name
    if (s == "refused(word_modulus)" || s == "refused(export_word)") return s;
    if (!(false)) throw std::logic_error("HeckeZRecord: an unlisted module word '" + s + "' (" + M.detail + ")");
    return s;
}
// a module's refusal word for (C3) at K = 1
std::string C3Refusal(const LatticeModule& M) {
    const std::string& s = M.status;
    if (s == "refused(word_modulus)" && M.source == "h1_level1" && Zlk::Fits(M.ell, 1))
        return "refused(k1_route)";   // (Z0 O1, lead decision) the level-1 module route runs at K >= 2 only: pending
    return RootRefusal(M);
}
std::string RootWord(const LocalReadOut& o) {
    const std::string& r = o.L.root;
    if (r == "closes") return "certified";
    if (r == "open(precision)") {
        if (o.escalation.find("the export's K_max") != std::string::npos) return "refused(export_word)";
        return "open(precision)";   // the step cap of the settings: final
    }
    if (r == "upper_bound_certified" || r == "refused(word_modulus)" || r == "refused(export_word)" || r == "blocked(export)" ||
        r == "not_run" || r == "general_case")
        return r;
    if (r == "inconsistent") throw std::logic_error("HeckeZRecord: the at-root read is inconsistent at ell = " + std::to_string(o.L.ell));
    if (!(false)) throw std::logic_error("HeckeZRecord: an unlisted root word '" + r + "'");
    return r;
}

struct EllRead {
    uint64_t ell = 0;
    std::vector<std::string> reasons;
    uint32_t v_disc = 0;
    bool torsion = false;
    LocalReadOut out;                // the at-root read (K0 with escalation)
    LocalReadOut c3out;              // the (C3) read at K = 1 when the K0 read was not served
    bool c3_from_fallback = false;
    std::string c3;                  // the (C3) word (record vocabulary)
    std::string root;                // the at-root word (record vocabulary)
    uint32_t K0 = 0;
    double secs = 0;
    // (slot-27 EXW) the G10 provenance of a trigger of rule (alpha): "free_block_z; export: <the replaced read's status>" or
    // "refused(<Name>)" (the free block refused: the pending word kept); empty: no trigger at ell
    std::string exw;
};

std::vector<uint64_t> Residues(const std::vector<std::string>& hex, uint64_t ell) {
    const Zlk F = Zlk::Make(ell, 1);
    std::vector<uint64_t> r;
    for (const auto& h : hex) r.push_back(local::HexResidue(F, h));
    return r;
}

// the exact bit length of |c| for a canonical signed hex text (0 for 0x0)
uint64_t HexBits(const std::string& h) {
    const size_t s = h[0] == '-' ? 3 : 2;
    if (h.size() <= s || h.substr(s) == "0") return 0;
    const char c = h[s];
    const uint32_t d = static_cast<uint32_t>(c >= 'a' ? c - 'a' + 10 : c - '0');
    uint32_t b = 0;
    for (uint32_t x = d; x; x >>= 1) ++b;
    return 4 * (h.size() - s - 1) + b;
}

std::string PolyText(const std::vector<uint64_t>& c) { return "[" + Join(c) + "]"; }

// ---- (schema audit F6) does theta mod ell generate A'_1?  The Krylov rank of {1, theta, .., theta^{dim-1}} in End(L / ell) ----------
uint64_t MulMod(uint64_t a, uint64_t b, uint64_t m) { return static_cast<uint64_t>(static_cast<unsigned __int128>(a) * b % m); }
uint64_t PowMod(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m;
    for (a %= m; e; e >>= 1, a = MulMod(a, a, m))
        if (e & 1) r = MulMod(r, a, m);
    return r;
}
// (schema audit F6) the first theta of theta_search's order (the units T_p in the order of P, then the combinations over
// {0..cmax}^|P| in lexicographic order, at most theta_cap candidates: HZ-RAT's (C1) search, RationalStep) whose reduction generates
// A'_1 = the image of the read set on L / ell (dim A'_1 = dimA1): F_ell[theta] = A'_1 iff 1, theta, .., theta^{dimA1 - 1} are
// independent.  Then theta's minimal polynomial on L / ell (the Krylov relation of theta^{dimA1}) is f_theta mod ell.
struct GenTheta {
    bool ok = false;
    std::vector<uint32_t> coeffs;
    std::vector<uint64_t> fbar;      // monic, low -> high, degree dimA1
};
// (slot-26 HZ-EISUP, design Sec. 1.5) `first` (optional; full records at N > 1 only): a coefficient vector on P tried before the search (the
// read's own generating theta'); nullptr leaves the search exactly as before.
// (S27 TS, TS_design.md Sec. 3 "S2-c capped") `continuation`: instead of `first` + the lex search, the capped greedy continuation over P
// (KrylovContinuation, score = the Krylov rank, a fresh budget of theta_cap tests); called only under theta_order = lex_then_greedy and
// only after every lex search of the chain has failed (the last-resort placement).
GenTheta GeneratingTheta(const RationalSettings& rs, const LatticeModule& M, const std::vector<uint32_t>& P, uint64_t ell, uint32_t dimA1,
                         const std::vector<uint32_t>* first = nullptr, bool continuation = false) {
    GenTheta G;
    if (dimA1 == 0 || M.P.size() < P.size() || M.T.size() < P.size() || ell < 2 || P.empty()) return G;
    for (size_t i = 0; i < P.size(); ++i)
        if (M.P[i] != P[i]) return G;
    const size_t D = M.D, m = P.size();
    std::vector<std::vector<uint64_t>> T(m);
    for (size_t i = 0; i < m; ++i) {
        T[i].resize(D * D);
        for (size_t x = 0; x < D * D; ++x) T[i][x] = M.T[i][x] % ell;
    }
    uint32_t last_rank = 0;   // (S27 TS) the Krylov rank of the last test (= dimA1 when it generates)
    auto test = [&](const std::vector<uint32_t>& c) -> bool {
        std::vector<uint64_t> A(D * D, 0);
        for (size_t i = 0; i < m; ++i)
            for (size_t x = 0; x < D * D; ++x) A[x] = (A[x] + MulMod(c[i] % ell, T[i][x], ell)) % ell;
        std::vector<std::vector<uint64_t>> pw;   // 1, theta, .., theta^{dimA1}
        std::vector<uint64_t> X(D * D, 0);
        for (size_t x = 0; x < D; ++x) X[x * D + x] = 1 % ell;
        for (uint32_t t = 0; t <= dimA1; ++t) {
            pw.push_back(X);
            if (t == dimA1) break;
            std::vector<uint64_t> Y(D * D, 0);
            for (size_t r = 0; r < D; ++r)
                for (size_t q = 0; q < D; ++q) {
                    const uint64_t a = X[r * D + q];
                    if (!a) continue;
                    for (size_t cc = 0; cc < D; ++cc) Y[r * D + cc] = (Y[r * D + cc] + MulMod(a, A[q * D + cc], ell)) % ell;
                }
            X.swap(Y);
        }
        // forward elimination of the first dimA1 powers, tracking combinations
        std::vector<std::vector<uint64_t>> rows(pw.begin(), pw.begin() + dimA1), comb(dimA1, std::vector<uint64_t>(dimA1, 0));
        for (uint32_t i = 0; i < dimA1; ++i) comb[i][i] = 1 % ell;
        std::vector<size_t> pc;
        uint32_t rank = 0;
        for (size_t col = 0; col < D * D && rank < dimA1; ++col) {
            size_t piv = rank;
            while (piv < dimA1 && rows[piv][col] == 0) ++piv;
            if (piv == dimA1) continue;
            std::swap(rows[piv], rows[rank]);
            std::swap(comb[piv], comb[rank]);
            const uint64_t inv = PowMod(rows[rank][col], ell - 2, ell);
            for (auto& x : rows[rank]) x = MulMod(x, inv, ell);
            for (auto& x : comb[rank]) x = MulMod(x, inv, ell);
            for (size_t r = rank + 1; r < dimA1; ++r) {
                const uint64_t f = rows[r][col];
                if (!f) continue;
                for (size_t j2 = 0; j2 < D * D; ++j2) rows[r][j2] = (rows[r][j2] + ell - MulMod(f, rows[rank][j2], ell)) % ell;
                for (size_t j2 = 0; j2 < dimA1; ++j2) comb[r][j2] = (comb[r][j2] + ell - MulMod(f, comb[rank][j2], ell)) % ell;
            }
            pc.push_back(col);
            ++rank;
        }
        last_rank = rank;
        if (rank != dimA1) return false;
        std::vector<uint64_t> v = pw[dimA1], w(dimA1, 0);   // v = theta^d - sum w_i theta^i
        for (uint32_t r = 0; r < dimA1; ++r) {
            const uint64_t f = v[pc[r]];
            if (!f) continue;
            for (size_t j2 = 0; j2 < D * D; ++j2) v[j2] = (v[j2] + ell - MulMod(f, rows[r][j2], ell)) % ell;
            for (size_t j2 = 0; j2 < dimA1; ++j2) w[j2] = (w[j2] + MulMod(f, comb[r][j2], ell)) % ell;
        }
        for (uint64_t x : v)
            ENGINE_CHECK_MSG(x == 0, "HeckeZRecord: theta^d is not in the span of its lower powers (dim A'_1 is wrong)");
        G.fbar.assign(dimA1 + 1, 0);
        for (uint32_t i = 0; i < dimA1; ++i) G.fbar[i] = (ell - w[i]) % ell;
        G.fbar[dimA1] = 1 % ell;
        return true;
    };
    if (continuation) {
        const uint64_t B2 = rs.theta_coeff_max ? rs.theta_coeff_max : std::max<uint64_t>(1, uint64_t(dimA1) * (dimA1 - 1) / 2);
        const uint64_t tbar = kThetaGreedyTarget % ell ? kThetaGreedyTarget % ell : 1;
        const ContinuationResult C = KrylovContinuation(static_cast<uint32_t>(m), dimA1, B2, ell, tbar, rs.theta_cap,
                                                        [&](const std::vector<uint32_t>& c) -> uint32_t {
                                                            test(c);
                                                            return last_rank;
                                                        });
        if (C.found) {
            ENGINE_CHECK_MSG(test(C.theta), "HeckeZRecord: the continuation's theta does not generate A'_1");
            G.ok = true;
            G.coeffs = C.theta;
        }
        return G;
    }
    if (first && first->size() == m && test(*first)) {
        G.ok = true;
        G.coeffs = *first;
        return G;
    }
    uint32_t tried = 0;
    for (size_t i = 0; i < m && tried < rs.theta_cap; ++i, ++tried) {
        std::vector<uint32_t> c(m, 0);
        c[i] = 1;
        if (test(c)) {
            G.ok = true;
            G.coeffs = c;
            return G;
        }
    }
    const uint32_t cmax = rs.theta_coeff_max ? rs.theta_coeff_max : std::max<uint32_t>(1, dimA1 * (dimA1 - 1) / 2);
    std::vector<uint32_t> c(m, 0);
    while (tried < rs.theta_cap) {
        size_t i = m;
        bool done = false;
        while (i > 0) {
            --i;
            if (c[i] < cmax) { ++c[i]; break; }
            c[i] = 0;
            if (i == 0) { done = true; break; }
        }
        if (done) break;
        uint32_t nz = 0, sum = 0;
        for (uint32_t x : c) { nz += x != 0; sum += x; }
        if (nz == 0 || (nz == 1 && sum == 1)) continue;
        ++tried;
        if (test(c)) {
            G.ok = true;
            G.coeffs = c;
            return G;
        }
    }
    return G;
}

uint32_t SturmPrimeBound(uint32_t k) { return k / 12 + 1; }   // level 1: T_n, n <= k/12 span T (Sturm), conservative + 1

}  // namespace

// (HZ-TOR) the support word of a maximal ideal m = (pi) at ell for the decomposing theta (coefficients on thP), from the torsion read rd
// (nullptr: no read).  The T'-blocks of H^1 / ell^e whose theta-residue is pi: a common block -> both.  The theta-residue of a block is
// known when theta = T_p (its residue at T_p), or when every residue over the operators theta involves has degree 1 (then theta acts by
// sum c_p a_p); else not_run ([C] item 8 (d)).  involved_only = false is the rule as the anemic / level-1 rows have always applied it
// (every operator of thP, zero coefficients included -- equivalent there, every operator of thP being in the read); involved_only = true
// (the full rows at N > 1, HZ-EISUP code audit m2) skips the operators with coefficient 0, so not_run is written exactly where theta'
// involves an operator -- a U_p -- that the torsion read does not carry ([C] 1.27.0-pre.1 item 8 (d')(ii), M1), or a residue of degree > 1.
std::string SupportWordOfRead(const tor::TorRead* rd, uint64_t ell, const std::vector<uint32_t>& th_coeffs, const std::vector<uint32_t>& thP,
                              const std::vector<uint64_t>& m_pi, bool involved_only) {
    std::string sup = "free";
    if (rd && rd->status == "ok") {
        const tor::TorRead& t = *rd;
        bool known = true, both = false;
        for (const auto& b : t.blocks) {
            std::vector<uint64_t> res;   // the block's theta-residue (monic, low -> high), empty = unknown
            bool unit_hit = false;
            for (size_t jj = 0; jj < th_coeffs.size(); ++jj) {
                if (involved_only && th_coeffs[jj] == 0) continue;   // (code audit m2) an operator theta' does not involve
                const auto ip = std::find(t.P.begin(), t.P.end(), thP[jj]);
                if (ip == t.P.end()) { res.clear(); break; }
                const size_t idx = static_cast<size_t>(ip - t.P.begin());
                if (idx >= b.residue.size()) { res.clear(); break; }
                if (th_coeffs[jj] == 1 && std::count(th_coeffs.begin(), th_coeffs.end(), 0u) + 1 == static_cast<long>(th_coeffs.size())) {
                    res = b.residue[idx];
                    unit_hit = true;
                    break;
                }
            }
            if (!unit_hit) {
                res.clear();
                uint64_t sum = 0;
                bool deg1 = true;
                for (size_t jj = 0; jj < th_coeffs.size() && deg1; ++jj) {
                    if (involved_only && th_coeffs[jj] == 0) continue;   // (code audit m2) contributes 0 to the residue
                    const auto ip = std::find(t.P.begin(), t.P.end(), thP[jj]);
                    const size_t idx = ip == t.P.end() ? b.residue.size() : static_cast<size_t>(ip - t.P.begin());
                    if (idx >= b.residue.size() || b.residue[idx].size() != 2 || b.residue[idx][1] != 1) {
                        deg1 = false;
                        break;
                    }
                    const uint64_t a = (ell - b.residue[idx][0] % ell) % ell;   // the root of x - a
                    sum = (sum + MulMod(th_coeffs[jj] % ell, a, ell)) % ell;
                }
                if (deg1) res = {(ell - sum) % ell, 1};
            }
            if (res.empty()) {
                if (b.kind == "common") known = false;
                continue;
            }
            if (res == m_pi && b.kind == "common") both = true;
        }
        sup = both ? "both" : (known ? "free" : "not_run");
    } else if (rd) {
        sup = rd->status == "blocked(export)" ? ExportWord(rd->detail)
                                             : (rd->status == "open(precision)" ? "refused(export_word)" : "not_run");
    }
    return sup;
}

HeckeStructureRecord BuildHeckeStructureRecord(uint32_t N, uint32_t n, const std::string& algebra, const RecordSettings& s,
                                               const RecordSources& src) {
    if (!hecke::CoefficientRingIsZ())
        throw std::invalid_argument("integral_hecke_structure/v1: the record is an object over Z; set coefficient_ring = Z first "
                                    "([C, sec:recordschema] item 8 (a))");
    if (N == 0 || n % 2 != 0) throw std::invalid_argument("integral_hecke_structure/v1: N >= 1 and n even");
    if (algebra != "full" && algebra != "anemic") throw std::invalid_argument("integral_hecke_structure/v1: algebra in {full, anemic}");
    if (N == 1 && algebra != "full") throw std::invalid_argument("integral_hecke_structure/v1: at N = 1 only `full` (= the anemic algebra)");
    ENGINE_CHECK_MSG(static_cast<bool>(src.local_read) && static_cast<bool>(src.v_bernoulli), "HeckeZRecord: a source is missing");
    if (s.g8 != "off" && s.g8 != "hida")
        throw std::invalid_argument("integral_hecke_structure/v1: heckez.G8 = " + s.g8 + " is not a value (off: the default, no G8 field; "
                                    "hida: H1's Hida block per maximal ideal)");
    const bool g8_hida = s.g8 == "hida";
    if (g8_hida) ENGINE_CHECK_MSG(static_cast<bool>(src.hida_block), "HeckeZRecord: heckez.G8 = hida needs the hida_block source");
    const auto t0 = std::chrono::steady_clock::now();
    // (HZ-RAT-PERF (c)) the export's Hecke application (the T_ell read) runs on the record's threads; no value depends on it
    const integral_descent::modexport::ScopedExportThreads export_threads(s.threads);

    HeckeStructureRecord R;
    R.N = N;
    R.n = n;
    R.algebra = algebra;
    R.settings_text = hecke::math_settings().canonical_text() + s.caps_text();
    R.settings_fingerprint = Hex64(engine::FnvBytes(engine::kFnvOffset64, R.settings_text));
    R.id = RecordId(N, n, algebra, R.settings_fingerprint);
    Builder B{R};
    const uint32_t k = n + 2;
    const bool full_N = algebra == "full" && N > 1;

    // ---- the descent (parts = groups,hecke: the ed.1 configuration), descent_ref -----------------------------------------------
    drec::RecordOptions dopt;
    dopt.hecke = true;
    dopt.pair = false;
    dopt.threads = s.threads;
    const drec::DescentRecordZ D = drec::BuildDescentRecord(N, n, dopt);
    R.descent.record_fnv1a64 = Hex64(D.digest());
    R.descent.primes = D.ps;
    R.descent.cell_sha256 = "not_in_ed1";   // the caller (campaign / validation) joins ed.1
    for (const auto& c : D.charpoly) R.descent.charpoly_digests.push_back({c.p, c.digest});
    if (!D.ok()) {
        R.refusal = "descent refused: " + D.refusal;
        B.add("G0", "G0.schema", kSchema);
        B.add("G0", "G0.N", std::to_string(N), "int");
        B.add("G0", "G0.n", std::to_string(n), "int");
        B.add("G0", "G0.algebra", algebra, "algebra");
        B.add("G0", "G0.descent_ref.record_fnv1a64", R.descent.record_fnv1a64, "hex64");
        B.add("G0", "G0.status", "refused", "overall");
        R.rehash();
        return R;
    }
    R.descent.b1 = D.H.size() > 1 ? D.H[1].free_rank : 0;
    {
        std::set<uint32_t> te;
        for (const auto& tm : D.torsion)
            if ((tm.q == 1 || tm.q == 2) && !tm.exponents.empty()) {
                te.insert(tm.ell);
                R.descent.torsion_q_ell.push_back({tm.q, tm.ell});
            }
        R.descent.torsion_ells.assign(te.begin(), te.end());
        std::sort(R.descent.torsion_q_ell.begin(), R.descent.torsion_q_ell.end(),
                  [](auto a, auto b) { return std::make_pair(a.second, a.first) < std::make_pair(b.second, b.first); });
    }
    const uint32_t b1 = R.descent.b1;

    // ---- the rational step on P (P_rule) -------------------------------------------------------------------------------------------
    std::vector<uint32_t> P = drec::DefaultRecordPrimes(N);
    RationalSettings rs = s.rat;
    rs.threads = s.threads;
    rs.p_rule_cap = s.p_rule_cap;   // (S27 TS) the last-round rule of a step without U_p (unprinted; printed in heckez.P_rule)
    // (HZ-RAT-PERF (b2)) the descent's certified chi_Z(T_p | H^1/tors): a unit theta = T_p reads its residues from it (value-neutral;
    // RationalStep compares both routes at its probe primes)
    RationalInputs rin;
    for (const auto& c : D.charpoly) {
        if (!c.determined || c.refused || c.coefficients.size() != size_t(b1) + 1) continue;
        std::vector<std::string> v;
        for (size_t i = 0; i < c.coefficients.size(); ++i)
            v.push_back(i < c.exact.size() && !c.exact[i].empty() ? c.exact[i] : c.coefficients[i]);
        std::reverse(v.begin(), v.end());   // the descent's entry i is the coefficient of x^{r - i}; RationalInputs is low to high
        rin.chi_z[c.p] = std::move(v);
    }
    // (HZ-RAT-PERF (a), H1 audit F2) a named refusal, not a defect: the record is refused, G1.C2_status reads the word
    auto refuse_word_prime_cap = [&](const WordPrimeCapRefusal& e) {
        R.refusal = e.what();   // "refused(word_prime_cap): ..."
        B.add("G0", "G0.schema", kSchema);
        B.add("G0", "G0.N", std::to_string(N), "int");
        B.add("G0", "G0.n", std::to_string(n), "int");
        B.add("G0", "G0.algebra", algebra, "algebra");
        B.add("G0", "G0.descent_ref.record_fnv1a64", R.descent.record_fnv1a64, "hex64");
        B.add("G0", "G0.status", "refused", "overall");
        B.add("G1", "G1.C2_status", "refused(word_prime_cap)", "C2");
        B.add("G10", "G10.rat.settings", rs.canonical_text());
        R.rehash();
        return R;
    };
    RationalResult Q;
    try {
        for (;;) {
            oc_util::CancelPoint("hecke_structure.rational");   // (NBCONFIRM S5) per P_rule round
            Q = RationalStep(N, n, P, rs, &rin);
            if (Q.c1_status == "certified" || P.size() >= s.p_rule_cap) break;
            uint32_t p = P.back() + 1;
            while (!IsPrimeU32(p) || N % p == 0) ++p;
            P.push_back(p);
        }
    } catch (const WordPrimeCapRefusal& e) {
        return refuse_word_prime_cap(e);
    }
    ENGINE_CHECK_MSG(Q.b1 == b1, "HeckeZRecord: b1 of HZ-RAT differs from the descent's H^1 free rank");
    const bool c1_ok = Q.c1_status == "certified" && Q.theta_index >= 0;
    const ThetaEntry* TH = c1_ok ? &Q.thetas[static_cast<size_t>(Q.theta_index)] : nullptr;
    const bool eis_rational = Q.eis_rational && c1_ok;
    // the cuspidal block exists (schema audit F5): G3's congruence branch, G3.C_T.ell=<l> written per certified factor of g_P.
    // (slot-27 EXW) defined here so that rule (alpha)'s root trigger reads the same predicate as the G3 writer (CtListed below).
    const bool has_cusp = eis_rational && TH && !full_N && Q.d_target >= 2;

    // full at N > 1: HZ-UTAG per p | N
    std::vector<UTagResult> utags;
    if (full_N) {
        RationalSettings rsu = rs;   // (re-audit N4) the validation-only U_p cap; 0 = the same cap as the rational step
        if (s.utag_word_prime_cap) rsu.word_prime_cap = s.utag_word_prime_cap;
        try {
            for (uint32_t p = 2; p <= N; ++p)
                if (N % p == 0 && IsPrimeU32(p)) {
                    oc_util::CancelPoint("hecke_structure.utag");   // (NBCONFIRM S5) per p | N
                    utags.push_back(UTagStep(N, n, p, P, rsu));
                }
        } catch (const WordPrimeCapRefusal& e) {
            return refuse_word_prime_cap(e);
        }
    }
    // (slot-26 Z1, UP-Z (a); UPZ_design.md Sec. 1.3 R2, 1.4 (d')) full at N > 1: the full rational step on the operator list G1.P's T_p
    // then U_p (p | N ascending).  P stays the anemic record's P (P_rule's (C1) is the anemic (C1); U_p never enlarges P); theta, (C1)
    // (degree d) and (C2) are read on the list.  Everything below that reads QF sits under full_N.
    RationalResult QF;
    std::vector<uint32_t> Ulist;
    up::FullG1Words FW;   // ([C] 1.26.0-pre.3 item 8 (d')(ii)) the one decision of G1's words (HeckeZUp.h FullG1Rule)
    bool utag_nr = false, utag_split = false;
    const bool eisup_scope = full_N && eisup::InScope(N, n);   // (HZ-EISUP, design D3) e^an_k(N) = 1 written out (audit m1)
    // (slot-27 EISUPb, HZEISUPb_design.md v0.4 D1) e^an_k(N) != 1: HZ-EISUP (b)'s scope ([C] 1.28.0-pre.1 prop:heckeZeisupchar)
    const bool eisupb_scope = full_N && eisup::InScopeChar(N, n);
    if (full_N) {
        for (const auto& u : utags) {
            utag_nr = utag_nr || u.non_reduced_certified;
            utag_split = utag_split || (u.split_degree == u.b1 && u.distinct == static_cast<uint32_t>(Q.d));
        }
        FW = up::FullG1Rule(utag_nr, utag_split, nullptr);
    }
    std::vector<PrimeRun> qf_runs;   // (HZ-EISUP, audit n2 / Q7) the full step's retained runs, reused by EisUpStep (value-neutral)
    if (full_N && FW.run_search) {
        Ulist = up::PrimesDividing(N);
        try {
            oc_util::CancelPoint("hecke_structure.rational_full");
            QF = RationalStepFull(N, n, P, Ulist, rs, &rin, (eisup_scope || eisupb_scope) ? &qf_runs : nullptr);
        } catch (const WordPrimeCapRefusal& e) {
            return refuse_word_prime_cap(e);
        }
        ENGINE_CHECK_MSG(QF.b1 == b1, "HeckeZRecord: b1 of the full rational step differs from the descent's H^1 free rank");
        FW = up::FullG1Rule(utag_nr, utag_split, &QF, false, false, eisup_scope || eisupb_scope);
    }
    if (full_N && Ulist.empty()) {   // the non-reduced branch: the operator list without a search (G1.theta_ops)
        Ulist = up::PrimesDividing(N);
        QF.P = P;
        QF.P.insert(QF.P.end(), Ulist.begin(), Ulist.end());
        QF.nU = static_cast<uint32_t>(Ulist.size());
    }
    const bool c1f_ok = full_N && QF.c1_status == "certified" && QF.theta_index >= 0;
    const ThetaEntry* THF = c1f_ok ? &QF.thetas[static_cast<size_t>(QF.theta_index)] : nullptr;

    // (slot-26 HZ-EISUP; design Sec. 2.1, [C] 1.27.0-pre.1 prop:heckeZeisup) the Eisenstein U_p systems on a full record at N > 1 with (C1)
    // certified and e^an_k(N) = 1; the only refusal is (c)'s whole-record refused(word_prime_cap) (audit m8), every other failure an ENGINE_CHECK
    eisup::EisUpResult EU;
    const bool eisup_on = c1f_ok && eisup_scope;
    const bool eisupb_on = c1f_ok && eisupb_scope;   // (EISUPb) the (b) branch: taken on e^an_k(N) != 1 only, (a)'s path untouched
    eisup::EisUpCharResult EB;
    ENGINE_CHECK_MSG(!c1f_ok || FW.components == ((eisup_scope || eisupb_scope) ? "certified" : "blocked(U_p)"), "HeckeZRecord: the components word");
    if (eisup_on) {
        oc_util::CancelPoint("hecke_structure.eisup");
        eisup::EisUpInput ein;
        ein.N = N;
        ein.n = n;
        ein.ops = QF.P;
        ein.nU = QF.nU;
        ein.theta = THF->coeffs;
        ein.f = THF->f;
        ein.e_k = QF.e_k;
        ein.d = QF.d;
        ein.b1 = b1;
        ein.rs = rs;
        ein.runs = &qf_runs;
        try {
            EU = eisup::EisUpStep(ein);
        } catch (const WordPrimeCapRefusal& e) {
            return refuse_word_prime_cap(e);
        }
    }
    if (eisupb_on) {   // (slot-27 EISUPb; design Sec. 6, [C] 1.28.0-pre.1 prop:heckeZeisupchar) the only refusal is (c)'s
        oc_util::CancelPoint("hecke_structure.eisupb");
        eisup::EisUpCharInput ein;
        ein.N = N;
        ein.n = n;
        ein.ops = QF.P;
        ein.nU = QF.nU;
        ein.theta = THF->coeffs;
        ein.f = THF->f;
        ein.e_k = QF.e_k;
        ein.d = QF.d;
        ein.b1 = b1;
        ein.rs = rs;
        ein.runs = &qf_runs;
        try {
            EB = eisup::EisUpCharStep(ein);
        } catch (const WordPrimeCapRefusal& e) {
            return refuse_word_prime_cap(e);
        }
    }
    std::vector<PrimeRun>().swap(qf_runs);   // the retained runs are not needed beyond this point

    // ---- S_cand ----------------------------------------------------------------------------------------------------------------
    std::map<uint64_t, EllRead> S;
    auto addReason = [&](uint64_t ell, const std::string& why) {
        EllRead& e = S[ell];
        e.ell = ell;
        if (std::find(e.reasons.begin(), e.reasons.end(), why) == e.reasons.end()) e.reasons.push_back(why);
    };
    if (c1_ok && !full_N) {
        for (const auto& se : Q.S_cand) {
            addReason(se.ell, se.reason);
            S[se.ell].v_disc = se.v_disc;   // (F16) v_disc = 1 stays in S (the S_rule of the caps text): no reason word for it
        }
        if (eis_rational && GPIsBound(Q))   // (S27 TS-FIX) no factors without a bound; the predicate states it
            for (const auto& pp : Q.gP.factors)
                if (pp.proof != "probable" && pp.q != 0) addReason(pp.q, "congruence");
    }
    for (uint32_t ell : R.descent.torsion_ells) {
        addReason(ell, "torsion");
        S[ell].torsion = true;
    }
    if (c1f_ok)   // (Z1) the S_rule on the full step: every certified prime of gcd_j D_j (no g_P: no rational Eisenstein system assumed)
        for (const auto& se : QF.S_cand) {
            addReason(se.ell, se.reason);
            S[se.ell].v_disc = se.v_disc;
        }
    if (full_N)
        for (const auto& u : utags)
            for (const auto& bd : u.bounds) {
                uint64_t x = bd.bound;
                for (uint64_t q = 2; q * q <= x; ++q)
                    while (x % q == 0) {
                        addReason(q, "congruence");
                        x /= q;
                    }
                if (x > 1) addReason(x, "congruence");
            }
    static const std::vector<std::string> kReasonOrder = {"disc_square", "disc_factor", "torsion", "congruence"};
    for (auto& [ell, e] : S)
        std::sort(e.reasons.begin(), e.reasons.end(), [](const std::string& a, const std::string& b) {
            return std::find(kReasonOrder.begin(), kReasonOrder.end(), a) < std::find(kReasonOrder.begin(), kReasonOrder.end(), b);
        });

    // ---- the local reads ---------------------------------------------------------------------------------------------------------
    local::ReadOptions ro;
    ro.d_target = Q.d_target;
    ro.r_eis = static_cast<uint32_t>(Q.e_k);
    ro.eis_rational = Q.eis_rational;
    ro.k_weight = k;
    ro.full_algebra = N == 1;
    if (!full_N && c1_ok) {
        for (auto& [ell, e] : S) {
            oc_util::CancelPoint("hecke_structure.local_read");   // (NBCONFIRM S5) per ell of S_cand
            const auto te = std::chrono::steady_clock::now();
            local::ReadOptions o = ro;
            o.rat = RatFor(Q, ell);
            uint32_t K0 = s.K_start_min;
            if (o.rat.v_c_lambda >= 0) K0 = std::max<uint32_t>(K0, static_cast<uint32_t>(o.rat.v_c_lambda) + 1);
            e.K0 = K0;
            e.out = src.local_read(N, n, ell, K0, P, b1, o, s.K_steps);
            if (e.out.read) {
                e.c3 = e.out.L.c3;
                e.root = RootWord(e.out);
            } else {
                e.root = RootRefusal(e.out.M);
                e.c3out = src.local_read(N, n, ell, 1, P, b1, o, 0);
                e.c3_from_fallback = true;
                e.c3 = e.c3out.read ? e.c3out.L.c3 : C3Refusal(e.c3out.M);
            }
            ENGINE_CHECK_MSG(e.c3 != "A1_of_computed", "HeckeZRecord: a torsion-prime H^1 read reached the record (P2a-2)");
            e.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count();
        }
    }

    // (Z1; Z2 at ell in {2, 3}) full at N > 1: the local reads on the read set (G1.P's T_p, the U_p) u {T_ell}, U_p on the cokernel route
    // at ell >= 5 (prop:heilbronn), on the tree route at ell in {2, 3} (the free-model U_p of `prop:strictliftN` (9), transported); the
    // same escalation and fallback as the anemic reads above.
    if (c1f_ok) {
        const std::vector<uint32_t>& ops = QF.P;
        for (auto& [ell, e] : S) {
            oc_util::CancelPoint("hecke_structure.local_read_full");
            const auto te = std::chrono::steady_clock::now();
            local::ReadOptions o = ro;
            o.d_target = QF.d_target;
            o.eis_rational = false;
            o.full_algebra = true;
            o.allow_bad_primes = true;
            e.K0 = s.K_start_min;
            e.out = src.local_read(N, n, ell, e.K0, ops, b1, o, s.K_steps);
            if (e.out.read) {
                e.c3 = e.out.L.c3;
                e.root = RootWord(e.out);
            } else {
                e.root = RootRefusal(e.out.M);
                e.c3out = src.local_read(N, n, ell, 1, ops, b1, o, 0);
                e.c3_from_fallback = true;
                e.c3 = e.c3out.read ? e.c3out.L.c3 : C3Refusal(e.c3out.M);
            }
            ENGINE_CHECK_MSG(e.c3 != "A1_of_computed", "HeckeZRecord: a torsion-prime H^1 read reached the full record");
            e.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count();
        }
    }

    // ---- (slot-27 EXW, EXW_design.md v0.2 Sec. 3.2) rule (alpha), "the word is written": the at-root read is made again on the free block
    //      over Z of EXPORT-Z step 2 where its word would be WRITTEN as G3.C_T.ell=<l>.read = refused(export_word).  CtListed is the G3
    //      writer's own condition (a certified factor of g_P, the cuspidal block present); the (C3) word is not touched here (step (4)).
    //      With the seam empty (src.local_read_free unset) no trigger fires: the pre-EXW record byte for byte.
    std::map<uint32_t, std::vector<std::string>> chi_free;   // the descent's chi_Z(T_p) per p (the free reads' production check)
    for (const auto& [p, c] : rin.chi_z) chi_free[p] = c;
    auto ct_factor = [](const auto& pp) { return pp.proof != "probable" && pp.q != 0; };
    auto ct_listed = [&](uint64_t ell) {
        if (!has_cusp || !GPIsBound(Q)) return false;   // the writer's branch: the cuspidal block and (S27 TS-FIX) g_P a bound
        for (const auto& pp : Q.gP.factors)
            if (ct_factor(pp) && pp.q == ell) return true;
        return false;
    };
    auto module_status = [](const LocalReadOut& o) {
        return o.read ? std::string("ok") : o.M.status + " (" + o.M.detail + ")";
    };
    // a served or word-modulus-refused free read; false: the free block was refused by name (*exw set, nothing else moves)
    auto free_read = [&](uint64_t ell, uint32_t K, const std::vector<uint32_t>& ops, const local::ReadOptions& o, uint32_t steps,
                         LocalReadOut* out, std::string* exw) -> bool {
        local::ReadOptions of = o;
        of.free_threads = s.threads;
        *out = src.local_read_free(N, n, ell, K, ops, b1, of, steps);
        if (out->M.status == "blocked(free_block)") {
            std::string name = "Unnamed";
            if (out->M.detail.compare(0, 9, "REFUSED [") == 0 && out->M.detail.find(']') != std::string::npos)
                name = out->M.detail.substr(9, out->M.detail.find(']') - 9);
            *exw = "refused(" + name + ")";
            return false;
        }
        std::string why;
        if (out->read)
            ENGINE_CHECK_MSG(FreeReadChiAgrees(out->M, chi_free, &why),
                             std::string("HeckeZRecord (EXW): the free read at ell = " + std::to_string(ell) + " fails the chi_Z check: " + why)
                                 .c_str());
        return true;
    };
    if (src.local_read_free && !full_N && c1_ok) {
        for (auto& [ell, e] : S) {
            if (!(ct_listed(ell) && e.root == "refused(export_word)")) continue;
            oc_util::CancelPoint("hecke_structure.local_read_free");
            const auto te = std::chrono::steady_clock::now();
            local::ReadOptions o = ro;
            o.rat = RatFor(Q, ell);
            LocalReadOut fr;
            std::string exw;
            const std::string old_status = module_status(e.out) + (e.out.escalation.empty() ? "" : "; escalation: " + e.out.escalation);
            if (free_read(ell, e.K0, P, o, s.K_steps, &fr, &exw)) {
                // the (C3) path keeps the module it had: a served EXPORT read whose escalation stopped at the export's K_max (origin (b)
                // of the design's Sec. 1) moves to c3out, so that G2, G9.K1 and the tb_mat entry stay byte for byte (only the root moves)
                if (!e.c3_from_fallback) {
                    e.c3out = e.out;
                    e.c3_from_fallback = true;
                }
                e.out = fr;
                e.root = fr.read ? RootWord(fr) : RootRefusal(fr.M);
                e.exw = "free_block_z; export: " + old_status;
            } else {
                e.exw = exw;
            }
            e.secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count();
        }
    }

    // ---- the Tor reads (HZ-TOR, merged e0ddc957): ReadTor at every ell with tors H^1_(ell) != 0 (HZ-TOR audit N2: ReadTor, not
    //      SplitFromModuleZ -- the read set P u {T_ell} and K_J >= e)
    std::map<uint64_t, tor::TorRead> TR;
    for (const auto& [q, ell] : R.descent.torsion_q_ell)
        if (q == 1) {
            oc_util::CancelPoint("hecke_structure.tor_read");   // (NBCONFIRM S5) per torsion prime
            TR.emplace(ell, tor::ReadTor(N, n, ell, P));
        }

    // ---- derived: eis_single ([C, prop:heckeZeiscert](1)), rational route ------------------------------------------------------
    bool every_closes = eis_rational && GPIsBound(Q) && Q.gP.complete();   // (S27 TS-FIX) no eis_single without a g_P bound
    if (eis_rational)
        for (const auto& pp : Q.gP.factors) {
            if (pp.proof == "probable") continue;
            const auto it = S.find(pp.q);
            if (it == S.end() || it->second.root != "certified") every_closes = false;
        }
    bool sturm_ok = true;
    for (uint32_t p = 2; p <= SturmPrimeBound(k); ++p)
        if (IsPrimeU32(p) && std::find(P.begin(), P.end(), p) == P.end()) sturm_ok = false;
    const bool eis_single = N == 1 && k >= 4 && eis_rational && every_closes && sturm_ok;
    // the primitive theta_j with ell !| D_j (certified: the factorisation is complete, or ell certified with valuation 0)
    auto rational_theta = [&](uint64_t ell) -> int32_t {
        for (uint32_t j : Q.discriminants_used) {
            const Factored& F = Q.thetas[j].disc;
            if (!F.complete()) continue;
            if (CertVal(F, ell) == 0) return static_cast<int32_t>(j);
        }
        return -1;
    };
    // (slot-27 EXW, design M1) the generation decision of G2.ell=<l>.generation.status, ONE function for the writer (G2 below) and for the
    // (C3) trigger: falls_to_c3 iff the written word is the (C3) word e.c3 (no route certifies generation otherwise)
    struct GenDecision {
        std::string gs, route;
        bool falls_to_c3 = false;
    };
    auto decide_gen = [&](uint64_t ell, const EllRead& e) -> GenDecision {
        GenDecision g;
        if (full_N) {
            // (Z1; Z2) the anemic rule, every ell: no read without (C1); the read's certificate, then the rational route
            if (!c1f_ok) g.gs = "open(sandwich)";
            else if (e.c3 == "certified") { g.gs = "certified"; g.route = "saturation"; }
            else if (up::RationalThetaIndex(QF, ell) >= 0) { g.gs = "certified"; g.route = "rational"; }
            else { g.gs = e.c3; g.falls_to_c3 = true; }
        } else if (!c1_ok) {
            g.gs = "open(sandwich)";
        } else if (e.c3 == "certified") {
            g.gs = "certified";
            g.route = "saturation";
        } else if (eis_single) {
            g.gs = "certified";
            g.route = "eis_single";
        } else if (rational_theta(ell) >= 0) {
            g.gs = "certified";
            g.route = "rational";
        } else {
            g.gs = e.c3;
            g.falls_to_c3 = true;
        }
        return g;
    };
    // ---- (slot-27 EXW, design Sec. 3.2 step (4)) rule (alpha) for (C3): the (C3) read at K = 1 is made again on the free block where its
    //      word would be WRITTEN as G2.ell=<l>.generation.status = refused(export_word) / refused(k1_route)
    if (src.local_read_free && (full_N ? c1f_ok : c1_ok)) {
        for (auto& [ell, e] : S) {
            if (!(decide_gen(ell, e).falls_to_c3 && (e.c3 == "refused(export_word)" || e.c3 == "refused(k1_route)"))) continue;
            oc_util::CancelPoint("hecke_structure.local_read_free");
            const auto te = std::chrono::steady_clock::now();
            local::ReadOptions o = ro;
            if (full_N) {
                o.d_target = QF.d_target;
                o.eis_rational = false;
                o.full_algebra = true;
                o.allow_bad_primes = true;
            } else {
                o.rat = RatFor(Q, ell);
            }
            LocalReadOut fr;
            std::string exw;
            const std::string old_status = "c3_fallback_K1 " + (e.c3_from_fallback ? module_status(e.c3out) : std::string("not_run"));
            if (free_read(ell, 1, full_N ? QF.P : P, o, 0, &fr, &exw)) {
                e.c3out = fr;
                e.c3_from_fallback = true;
                e.c3 = fr.read ? fr.L.c3 : C3Refusal(fr.M);
                ENGINE_CHECK_MSG(e.c3 != "A1_of_computed", "HeckeZRecord (EXW): a torsion-prime read reached the record");
                const std::string w = "free_block_z; export: " + old_status;
                e.exw = e.exw.empty() ? w : e.exw + " | (C3) " + w;
            } else {
                e.exw = e.exw.empty() ? exw : e.exw + " | (C3) " + exw;
            }
            e.secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - te).count();
        }
    }

    // =================================================================================================================================
    // G0
    B.add("G0", "G0.schema", kSchema);
    B.add("G0", "G0.coefficient_ring", "Z");
    B.add("G0", "G0.object", "group");
    B.add("G0", "G0.conventions", "poly=low_to_high; hex=canonical_signed");
    B.add("G0", "G0.N", std::to_string(N), "int");
    B.add("G0", "G0.n", std::to_string(n), "int");
    B.add("G0", "G0.algebra", algebra, "algebra");
    {
        std::set<uint32_t> ops(P.begin(), P.end());
        for (const auto& [ell, e] : S)
            if (e.out.read && e.out.L.t_ell == "added") ops.insert(static_cast<uint32_t>(ell));
        std::vector<std::string> o;
        for (uint32_t p : ops) o.push_back("T_" + std::to_string(p));
        if (full_N)
            for (const auto& u : utags) o.push_back("U_" + std::to_string(u.p));
        B.add("G0", "G0.operators", Join(o));
    }
    B.add("G0", "G0.descent_ref.record_fnv1a64", R.descent.record_fnv1a64, "hex64");
    B.add("G0", "G0.status", "final", "overall");
    B.add("G0x", "G0.descent_ref.primes", Join(R.descent.primes));
    // G0.descent_ref.cell_sha256 is set by the caller's ed.1 join (descent.cell_sha256; JSON only)

    // =================================================================================================================================
    // G1
    B.add("G1", "G1.b1", std::to_string(b1), "int");
    B.add("G1", "G1.e_k", std::to_string(Q.e_k), "int");
    B.add("G1", "G1.d", std::to_string(full_N ? Q.d : Q.d_target), "int");
    B.add("G1", "G1.d_source", full_N || N == 1 ? "hES_N" : "hES_N_an", "d_source");
    std::string c1, redq;
    if (full_N) {
        c1 = FW.C1;          // (Z1) [C] 1.26.0-pre.3 item 8 (d')(ii): HeckeZUp.h FullG1Rule
        redq = FW.reduced_Q;
        ENGINE_CHECK_MSG(!(utag_nr && c1f_ok), "HeckeZRecord: the full (C1) is certified (T (x) Q reduced) but a U_p tag certifies non-reduced");
    } else {
        c1 = Q.c1_status;
        redq = algebra == "anemic" ? "true" : (c1 == "certified" ? "true" : "undecided");
    }
    B.add("G1", "G1.reduced_Q", redq, "reduced_Q");
    B.add("G1", "G1.C1.status", c1, "C1");
    if (c1 == "certified" && (!full_N || FW.disc_fields))
        B.add("G1", "G1.C1.deg_f", std::to_string(full_N ? QF.d_target : Q.d_target), "int", "G1.C1.status");
    B.add("G1", "G1.P", Join(P));
    // the cuspidal block (schema audit F5; re-audit R3): absent iff d = 1 (at level 1: d = e_k, no cusp forms; at N > 1 anemic d = 1
    // can hold with e_k > 1, and d = e_k with a cusp form); irreducibility certified by a prime q
    // modulo which c_theta is irreducible (count cap irred_primes, the prime in G9); p0 = T_p when theta is the unit T_p and the block
    // is irreducible (theta_search tries the units in ascending order, so p is the smallest prime whose eigenvalue generates).
    std::string cusp_irred;          // "true" | "open(cap)"
    uint64_t cusp_irred_q = 0;       // the certifying prime (0: degree 1, or none found)
    int32_t theta_unit_p = -1;       // theta = T_p
    if (TH) {
        for (size_t i = 0; i < TH->coeffs.size(); ++i)
            if (TH->coeffs[i] == 1 && std::count(TH->coeffs.begin(), TH->coeffs.end(), 0u) + 1 == static_cast<long>(TH->coeffs.size()))
                theta_unit_p = static_cast<int32_t>(P[i]);
    }
    uint32_t irred_tried = 0;        // (re-audit R8) provenance: primes tried, seconds
    double irred_secs = 0;
    // the irreducibility certificate of a cuspidal block c (hex, low to high) of degree deg ((HZ-EISUP) also c_theta = f_theta / e_theta)
    auto certify_irred = [&](const std::vector<std::string>& cpart, int64_t deg) {
        if (deg == 1) {
            cusp_irred = "true";
        } else {
            const auto ti = std::chrono::steady_clock::now();
            cusp_irred = "open(cap)";
            uint32_t& tried = irred_tried;
            for (uint32_t q = 3; tried < s.irred_primes; q += 2) {
                if (!IsPrimeU32(q)) continue;
                ++tried;
                const hecke::FpPoly64 cq(q, Residues(cpart, q));
                if (cq.coeffs.size() != cpart.size() || !cq.is_monic()) continue;   // (c_theta is monic: never)
                if (hecke::CertifyIrreducible(cq) == 0) {   // deterministic: irreducible mod q of full degree => irreducible over Q
                    cusp_irred = "true";
                    cusp_irred_q = q;
                    break;
                }
            }
            irred_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count();
        }
    };
    if (has_cusp) certify_irred(Q.cusp_part, Q.d_target - 1);
    const bool full_cusp = (eisup_on || eisupb_on) && QF.d > QF.e_k;   // (HZ-EISUP) the cuspidal block of the full record, absent iff d = e_k
    if (full_cusp) certify_irred(eisupb_on ? EB.c_theta : EU.c_theta, QF.d - QF.e_k);
    if (TH && !full_N) {
        B.add("G1", "G1.theta", Join(TH->coeffs), "", "G1.C1.status");
        if (eis_rational) {
            // the Eisenstein component: degree 1, so the smallest p0 = P[0] generates Q; lambda(T_p0) = 1 + p0^{k-1}
            Limbs lam0 = {1};
            for (uint32_t j = 0; j + 1 < k; ++j) LimbMul(lam0, P[0]);
            {
                size_t i = 0;
                uint64_t c = 1;
                while (c) {
                    if (i == lam0.size()) lam0.push_back(0);
                    const uint64_t t = uint64_t(lam0[i]) + c;
                    lam0[i] = static_cast<uint32_t>(t);
                    c = t >> 32;
                    ++i;
                }
            }
            B.add("G1", "G1.components[0].type", "eis", "component_type");
            B.add("G1", "G1.components[0].degree", "1", "int");
            // (re-audit R1) the multiplicity on H^1 is e_k: eis_rational means one anemic Eisenstein system, so all e_k Eisenstein
            // classes carry it (b1 = 2 dim S_k + e_k, and no cusp system equals it); e_k = 1 at level 1, e_k = #cusps (- 1 at k = 2)
            B.add("G1", "G1.components[0].mult", std::to_string(Q.e_k), "int");
            B.add("G1", "G1.components[0].p0", "T_" + std::to_string(P[0]), "op_or_theta");
            B.add("G1", "G1.components[0].minpoly", "[-" + LimbHex(lam0) + ",0x1]");
            B.add("G1", "G1.components[0].lambda_theta", TH->lambda_hex, "zbig");
            if (has_cusp) {
                B.add("G1", "G1.components[1].type", "cusp", "component_type");
                B.add("G1", "G1.components[1].degree", std::to_string(Q.d_target - 1), "int");
                B.add("G1", "G1.components[1].mult", N == 1 ? "2" : "not_run", N == 1 ? "int" : "not_run");
                B.add("G1", "G1.components[1].p0", cusp_irred == "true" && theta_unit_p > 0 ? "T_" + std::to_string(theta_unit_p) : "theta",
                      "op_or_theta");
                B.add("G1", "G1.components[1].minpoly", "[" + Join(Q.cusp_part) + "]");
                B.add("G1", "G1.components[1].irreducible", cusp_irred, "irreducible");
            }
        } else {
            B.add("G1", "G1.components", "not_run", "not_run");
        }
    } else if (full_N) {
        // (Z1; audit C2) theta's coefficient vector is indexed by G1.theta_ops (T_p of G1.P ascending, then U_p ascending), written on
        // full records only; the components need the Eisenstein U_p systems (not built by Z1): blocked(U_p) where (C1) holds
        if (FW.theta_status.empty()) B.add("G1", "G1.theta", Join(THF->coeffs), "", "G1.C1.status");
        else B.add("G1", "G1.theta", FW.theta_status, "status");
        B.add("G1", "G1.theta_ops", up::ThetaOpsText(QF.P, QF.nU));
        B.add("G1", "G1.components", FW.components, "status");
        if (eisup_on) {
            // (HZ-EISUP; design Sec. 3.1, [C] 1.27.0-pre.1 item 8 (d')(ii)) the e_k Eisenstein components (ordered by their U_p tuple), then the
            // cuspidal block iff d > e_k; (p0, minpoly) is the same for every Eisenstein component, so not a key of the component (audit n3)
            Limbs lam0 = {1};
            for (uint32_t j = 0; j + 1 < k; ++j) LimbMul(lam0, P[0]);
            {
                size_t i = 0;
                uint64_t c = 1;
                while (c) {
                    if (i == lam0.size()) lam0.push_back(0);
                    const uint64_t t = uint64_t(lam0[i]) + c;
                    lam0[i] = static_cast<uint32_t>(t);
                    c = t >> 32;
                    ++i;
                }
            }
            const std::vector<uint32_t> Up = EU.Uprimes();
            for (size_t j = 0; j < EU.lines.size(); ++j) {
                const std::string cp = "G1.components[" + std::to_string(j) + "]";
                B.add("G1", cp + ".type", "eis", "component_type");
                B.add("G1", cp + ".degree", "1", "int");
                B.add("G1", cp + ".mult", "1", "int");
                B.add("G1", cp + ".p0", "T_" + std::to_string(P[0]), "op_or_theta");
                B.add("G1", cp + ".minpoly", "[-" + LimbHex(lam0) + ",0x1]");
                B.add("G1", cp + ".lambda_theta", EU.lines[j].lambda_theta, "zbig");
                for (size_t i = 0; i < Up.size(); ++i) B.add("G1", cp + ".U_" + std::to_string(Up[i]), EU.lines[j].U[i], "zbig");
            }
            if (full_cusp) {
                const std::string cp = "G1.components[" + std::to_string(EU.lines.size()) + "]";
                int32_t unit_T = -1;   // theta = the unit T_p (p !| N): (d)'s p0 rule
                for (size_t i = 0; i + QF.nU < QF.P.size(); ++i)
                    if (THF->coeffs[i] == 1 && std::count(THF->coeffs.begin(), THF->coeffs.end(), 0u) + 1 == static_cast<long>(THF->coeffs.size()))
                        unit_T = static_cast<int32_t>(QF.P[i]);
                B.add("G1", cp + ".type", "cusp", "component_type");
                B.add("G1", cp + ".degree", std::to_string(QF.d - QF.e_k), "int");
                B.add("G1", cp + ".mult", "2", "int");
                B.add("G1", cp + ".p0", cusp_irred == "true" && unit_T > 0 ? "T_" + std::to_string(unit_T) : "theta", "op_or_theta");
                B.add("G1", cp + ".minpoly", "[" + Join(EU.c_theta) + "]");
                B.add("G1", cp + ".irreducible", cusp_irred, "irreducible");
            }
        }
        if (eisupb_on) {
            // (slot-27 EISUPb; design Sec. 5.1, [C] 1.28.0-pre.1 item 8 (d')(ii) "... from HZ-EISUP (b) on") one row per Eisenstein component:
            // degree 1 first (lambda_theta, U_<p>), then degree >= 2 (theta_poly, U_<p>.charpoly); psi_cond, psi_order on every row; then the
            // cuspidal block iff d > e_k
            const std::vector<uint32_t> Up = EB.Uprimes();
            for (size_t j = 0; j < EB.comps.size(); ++j) {
                const eisup::EisCharComp& c = EB.comps[j];
                const std::string cp = "G1.components[" + std::to_string(j) + "]";
                B.add("G1", cp + ".type", "eis", "component_type");
                B.add("G1", cp + ".degree", std::to_string(c.degree), "int");
                B.add("G1", cp + ".mult", "1", "int");
                B.add("G1", cp + ".p0", "T_" + std::to_string(P[0]), "op_or_theta");
                B.add("G1", cp + ".minpoly", "[" + Join(c.minpoly) + "]");
                if (c.degree == 1) {
                    B.add("G1", cp + ".lambda_theta", c.lambda_theta, "zbig");
                    for (size_t i = 0; i < Up.size(); ++i) B.add("G1", cp + ".U_" + std::to_string(Up[i]), c.U[i], "zbig");
                } else {
                    B.add("G1", cp + ".theta_poly", "[" + Join(c.theta_poly) + "]");
                    for (size_t i = 0; i < Up.size(); ++i) B.add("G1", cp + ".U_" + std::to_string(Up[i]) + ".charpoly", "[" + Join(c.Ucharpoly[i]) + "]");
                }
                B.add("G1", cp + ".psi_cond", std::to_string(c.psi_cond), "int");
                B.add("G1", cp + ".psi_order", std::to_string(c.psi_order), "int");
            }
            if (full_cusp) {
                const std::string cp = "G1.components[" + std::to_string(EB.comps.size()) + "]";
                int32_t unit_T = -1;   // theta = the unit T_p (p !| N): (d)'s p0 rule
                for (size_t i = 0; i + QF.nU < QF.P.size(); ++i)
                    if (THF->coeffs[i] == 1 && std::count(THF->coeffs.begin(), THF->coeffs.end(), 0u) + 1 == static_cast<long>(THF->coeffs.size()))
                        unit_T = static_cast<int32_t>(QF.P[i]);
                B.add("G1", cp + ".type", "cusp", "component_type");
                B.add("G1", cp + ".degree", std::to_string(QF.d - QF.e_k), "int");
                B.add("G1", cp + ".mult", "2", "int");
                B.add("G1", cp + ".p0", cusp_irred == "true" && unit_T > 0 ? "T_" + std::to_string(unit_T) : "theta", "op_or_theta");
                B.add("G1", cp + ".minpoly", "[" + Join(EB.c_theta) + "]");
                B.add("G1", cp + ".irreducible", cusp_irred, "irreducible");
            }
        }
    } else {
        B.add("G1", "G1.theta", full_N ? "blocked(U_p)" : "open(cap)", "status");
        B.add("G1", "G1.components", full_N ? "blocked(U_p)" : "not_run", "status");
    }
    for (const auto& [ell, e] : S) {
        const std::string pre = "G1.S_cand.ell=" + std::to_string(ell);
        B.add("G1", pre + ".reason", Join(e.reasons), "list:S_reason");
        B.add("G1", pre + ".v_disc", std::to_string(e.v_disc), "int");
    }
    if (TH && !full_N) {
        B.add("G1", "G1.disc.digest", Hex64(engine::FnvBytes(engine::kFnvOffset64, TH->disc.value_hex)), "hex64");
        B.add("G1", "G1.disc.bits", std::to_string(HexBits(TH->disc.value_hex)), "int");   // the exact bit length of |D|
        const Factored& G = Q.disc_gcd;
        for (const auto& pp : G.factors) {
            B.add("G1", "G1.disc.gcd.factor.q=" + pp.q_hex + ".e", std::to_string(pp.e), "int");
            B.add("G1", "G1.disc.gcd.factor.q=" + pp.q_hex + ".prime_proof", pp.proof, "prime_proof");
        }
        B.add("G1", "G1.disc.gcd.cofactor", G.cofactor_hex, "zbig");
        B.add("G1", "G1.disc.gcd.cofactor_test", G.cofactor_test, "cofactor_test");
        B.add("G1", "G1.C2_status", Q.c2_status, "C2");
        std::vector<std::string> du;
        for (uint32_t j : Q.discriminants_used) du.push_back("(" + Join(Q.thetas[j].coeffs, " ") + ")");
        B.add("G1", "G1.discriminants_used", Join(du));
    } else if (full_N && FW.disc_fields) {
        // (Z1) (C2) of the full step, the same fields as above from QF
        B.add("G1", "G1.disc.digest", Hex64(engine::FnvBytes(engine::kFnvOffset64, THF->disc.value_hex)), "hex64");
        B.add("G1", "G1.disc.bits", std::to_string(HexBits(THF->disc.value_hex)), "int");
        const Factored& G = QF.disc_gcd;
        for (const auto& pp : G.factors) {
            B.add("G1", "G1.disc.gcd.factor.q=" + pp.q_hex + ".e", std::to_string(pp.e), "int");
            B.add("G1", "G1.disc.gcd.factor.q=" + pp.q_hex + ".prime_proof", pp.proof, "prime_proof");
        }
        B.add("G1", "G1.disc.gcd.cofactor", G.cofactor_hex, "zbig");
        B.add("G1", "G1.disc.gcd.cofactor_test", G.cofactor_test, "cofactor_test");
        B.add("G1", "G1.C2_status", FW.C2.empty() ? QF.c2_status : FW.C2, "C2");
        std::vector<std::string> du;
        for (uint32_t j : QF.discriminants_used) du.push_back("(" + Join(QF.thetas[j].coeffs, " ") + ")");
        B.add("G1", "G1.discriminants_used", Join(du));
    } else if (full_N) {
        // (Z1) full at N > 1 without a certified (C1): the rule's word (partial(unfactored) beside open(cap); not_run in the non-reduced
        // branch; blocked(U_p) where no search ran)
        B.add("G1", "G1.C2_status", FW.C2, "C2");
    } else {
        B.add("G1", "G1.C2_status", full_N ? "blocked(U_p)" : "partial(unfactored)", "C2");
    }

    // =================================================================================================================================
    // G2 (per ell in S, per maximal ideal)
    std::map<uint64_t, std::string> gen_status;
    // (slot-25 H1) what G8 (heckez.G8 = hida) needs from G2: per ell, the decomposition's theta and its maximal ideals
    struct G8Mi {
        std::vector<uint64_t> pi;
        uint32_t mult = 0;
        bool eis = false, cusp = false;
    };
    struct G8Ell {
        bool decomposed = false;
        std::vector<uint32_t> ops, theta;
        std::vector<G8Mi> mis;
    };
    std::map<uint64_t, G8Ell> g8info;
    std::map<uint64_t, std::string> g8_k9, g8_prov;   // (H1) per ell: K9's integers (G9) and the forms read's provenance (G10)
    // the local type of T (x) Z_ell at m = (pi) of multiplicity mult in f-bar (monogenic: Z_ell[x]/(pi^mult)); (HZ-EISUP) shared with the full rows
    auto add_local_type = [&](const std::string& pre, const std::string& mp, uint32_t mult, uint32_t fdeg) {
        B.add("G2", mp + ".local_type.kind", mult == 1 ? "etale" : "monogenic", "local_kind", pre + ".generation.status");
        B.add("G2", mp + ".local_type.rank", std::to_string(mult * fdeg), "int", pre + ".generation.status");
        B.add("G2", mp + ".local_type.embdim", mult > 1 ? "1" : "0", "int", pre + ".generation.status");
        B.add("G2", mp + ".local_type.gorenstein", "true", "bool", pre + ".generation.status");
        B.add("G2", mp + ".local_type.length_mod_l", std::to_string(mult), "int", pre + ".generation.status");
        std::vector<uint32_t> hil(mult, 1);
        B.add("G2", mp + ".local_type.hilbert", Join(hil), "", pre + ".generation.status");
        B.add("G2", mp + ".local_type.loewy", std::to_string(mult), "int", pre + ".generation.status");
    };
    // (HZ-TOR) the support word of a maximal ideal m = (pi) at ell for the decomposing theta (SupportWordOfRead); (HZ-EISUP) shared by
    // the anemic / level-1 rows (involved_only = false, byte-identical) and the full rows at N > 1 (involved_only = true, code audit m2)
    auto support_word = [&](uint64_t ell, const std::vector<uint32_t>& th_coeffs, const std::vector<uint32_t>& thP,
                            const std::vector<uint64_t>& m_pi, bool involved_only) -> std::string {
        const auto it = TR.find(ell);
        return SupportWordOfRead(it == TR.end() ? nullptr : &it->second, ell, th_coeffs, thP, m_pi, involved_only);
    };
    for (const auto& [ell, e] : S) {
        const std::string pre = "G2.ell=" + std::to_string(ell);
        B.add("G2", pre + ".theory", "H", "theory");
        // (slot-27 EXW) the decision is decide_gen's (shared with the (C3) trigger); gen_theta follows its route
        const GenDecision gdec = decide_gen(ell, e);
        const std::string gs = gdec.gs, route = gdec.route;
        int32_t gen_theta = -1;   // the theta generating T (x) Z_ell, an index into Q.thetas
        if (full_N || !c1_ok) {
        } else if (route == "saturation") {
            const std::string& mono = e.c3_from_fallback ? e.c3out.L.mono : e.out.L.mono;
            if (mono == "theta") {
                gen_theta = Q.theta_index;
            } else if (mono.compare(0, 2, "T_") == 0) {
                const uint32_t p = static_cast<uint32_t>(std::stoul(mono.substr(2)));
                for (size_t j = 0; j < Q.thetas.size(); ++j) {
                    const auto& c = Q.thetas[j].coeffs;
                    const auto ip = std::find(P.begin(), P.end(), p);
                    if (ip == P.end() || !Q.thetas[j].primitive) continue;
                    bool unit = true;
                    for (size_t i = 0; i < c.size(); ++i) unit = unit && c[i] == (P[i] == p ? 1u : 0u);
                    if (unit) gen_theta = static_cast<int32_t>(j);
                }
            }
        } else if (route == "eis_single" || route == "rational") {
            gen_theta = rational_theta(ell);
        }
        gen_status[ell] = gs;
        B.add("G2", pre + ".generation.status", gs, "generation");
        if (!route.empty()) B.add("G2", pre + ".generation.route", route, "route", pre + ".generation.status");
        if (full_N) {
            // (Z1) the read's fields as on the anemic record; the maximal ideals need the Eisenstein U_p systems (not built by Z1):
            // blocked(U_p) where generation is certified, else not_run (the decomposition rule needs certified generation)
            g8info[ell] = G8Ell{};
            const LocalReadOut& rd = e.c3_from_fallback ? e.c3out : e.out;
            if (c1f_ok) {
                if (rd.read) {
                    B.add("G2", pre + ".generation.dim_A1", std::to_string(rd.L.dim_A1), "int");
                    B.add("G2", pre + ".t_ell", rd.L.t_ell, "t_ell");
                } else {
                    B.add("G2", pre + ".t_ell", rd.M.t_ell.empty() ? "in_P" : rd.M.t_ell, "t_ell");
                }
            }
            if (!eisup_on && !eisupb_on) {   // no (C1): blocked(U_p) where generation is certified (pending), else not_run
                B.add("G2", pre + ".maximal_ideals", gs == "certified" ? "blocked(U_p)" : "not_run", "mi_status", pre + ".generation.status");
                if (gs != "certified" && c1f_ok && rd.read)
                    B.add("G2", pre + ".A1_of_computed.dim_A1", std::to_string(rd.L.dim_A1), "int");
                continue;
            }
            // (HZ-EISUP; design Sec. 1.5, [C] 1.27.0-pre.1 item 8 (d')(ii)) (d)'s rule over the operator list: decomposed iff generation is
            // certified by saturation or rational and some theta' generates A'_1 on the ell-read (Krylov rank = dim A'_1 = d): the read's own
            // generating theta' first, then theta_search's order over the list, then over the read set (T_ell appended) where t_ell = added
            const std::vector<uint32_t>& ops = QF.P;
            GenTheta gt;
            std::vector<uint32_t> thP = ops;
            if (gs == "certified" && (route == "saturation" || route == "rational") && rd.read &&
                rd.L.dim_A1 == static_cast<uint32_t>(QF.d_target) && Zlk::Fits(ell, 1)) {
                std::vector<uint32_t> own;   // the read's mono "T_<x>": the unit of the operator x of its read set (a U_p is named T_<p> there)
                std::vector<uint32_t> ownP;
                const std::string& mono = rd.L.mono;
                if (mono.compare(0, 2, "T_") == 0 && mono.find_first_not_of("0123456789", 2) == std::string::npos && mono.size() > 2) {
                    const uint32_t x = static_cast<uint32_t>(std::stoul(mono.substr(2)));
                    const auto ip = std::find(rd.M.P.begin(), rd.M.P.end(), x);
                    if (ip != rd.M.P.end()) {
                        const size_t i = static_cast<size_t>(ip - rd.M.P.begin());
                        ownP = i < ops.size() ? ops : rd.M.P;
                        own.assign(ownP.size(), 0);
                        own[i] = 1;
                    }
                }
                if (!own.empty()) {
                    gt = GeneratingTheta(s.rat, rd.M, ownP, ell, static_cast<uint32_t>(rd.L.dim_A1), &own);
                    if (gt.ok && gt.coeffs == own) thP = ownP;
                    else gt = GenTheta{};
                }
                if (!gt.ok) gt = GeneratingTheta(s.rat, rd.M, ops, ell, static_cast<uint32_t>(rd.L.dim_A1));
                if (!gt.ok && rd.L.t_ell == "added" && rd.M.P.size() == ops.size() + 1 && rd.M.P.back() == ell) {
                    gt = GeneratingTheta(s.rat, rd.M, rd.M.P, ell, static_cast<uint32_t>(rd.L.dim_A1));
                    if (gt.ok) thP = rd.M.P;
                }
                // (S27 TS) lex_then_greedy: the capped continuation as the last resort, over the list, then over the read set
                if (!gt.ok && s.rat.theta_order == ThetaOrder::LexThenGreedy) {
                    gt = GeneratingTheta(s.rat, rd.M, ops, ell, static_cast<uint32_t>(rd.L.dim_A1), nullptr, true);
                    if (!gt.ok && rd.L.t_ell == "added" && rd.M.P.size() == ops.size() + 1 && rd.M.P.back() == ell) {
                        gt = GeneratingTheta(s.rat, rd.M, rd.M.P, ell, static_cast<uint32_t>(rd.L.dim_A1), nullptr, true);
                        if (gt.ok) thP = rd.M.P;
                    }
                }
            }
            if (!gt.ok) {
                B.add("G2", pre + ".maximal_ideals", "not_run", "mi_status", pre + ".generation.status");
                if (gs != "certified" && rd.read) B.add("G2", pre + ".A1_of_computed.dim_A1", std::to_string(rd.L.dim_A1), "int");
                continue;
            }
            // theta' = a primitive theta of HZ-RAT's list: its f is exact (C-E7 second part; the Eisenstein--cusp depth)
            const ThetaEntry* exact = nullptr;
            if (thP.size() == ops.size())
                for (uint32_t j : QF.discriminants_used)
                    if (QF.thetas[j].coeffs == gt.coeffs) exact = &QF.thetas[j];
            if (exact) {
                std::vector<uint64_t> fr = Residues(exact->f, ell);
                ENGINE_CHECK_MSG(fr == gt.fbar, "HeckeZRecord C-E7: the Krylov relation of theta' at ell is not f_theta' mod ell");
            }
            if (eisupb_on) {
                // (slot-27 EISUPb; design Sec. 2.9, [C] 1.28.0-pre.1 item 8 (d')(ii)) f_j = charpoly(theta' | E_j) exactly (the orbit product of
                // the certified systems; T_ell by lambda_psi(T_ell), prop:heckeZeisupchar (7)); C-B7; component j passes through m iff pi | f-bar_j
                const size_t ne = EB.comps.size();
                std::vector<std::vector<std::string>> fZ(ne);
                std::vector<hecke::FpPoly64> fM;
                for (size_t j = 0; j < ne; ++j) {
                    fZ[j] = eisup::LambdaPolyOf(EB, j, thP, gt.coeffs);
                    if (s.eisup_tamper_ell == ell && j == 0) fZ[0][0] = eisup::HexPlus(fZ[0][0], 1);   // (validation) the C-B7 tamper
                    fM.push_back(hecke::FpPoly64(ell, eisup::PolyModEll(fZ[j], ell)));
                }
                hecke::FpPoly64 cb(ell, gt.fbar);
                for (size_t j = 0; j < ne; ++j) {   // C-B7: prod_j f-bar_j divides f-bar with remainder 0, at multiplicity
                    hecke::FpPoly64 qq, rr;
                    hecke::FpPoly64::divrem(cb, fM[j], qq, rr);
                    ENGINE_CHECK_MSG(rr.is_zero(), "HeckeZRecord C-B7: prod_j charpoly(theta' | E_j) does not divide f-bar_{theta'} mod ell (remainder != 0)");
                    cb = qq;
                }
                const auto fac = hecke::FactorPoly(hecke::FpPoly64(ell, gt.fbar));
                struct MI {
                    std::vector<uint64_t> pi;
                    uint32_t mult;
                };
                std::vector<MI> mis;
                for (const auto& fa : fac.factors) mis.push_back(MI{fa.pi.coeffs, fa.multiplicity});
                std::sort(mis.begin(), mis.end(), [](const MI& a, const MI& b) {
                    if (a.pi.size() != b.pi.size()) return a.pi.size() < b.pi.size();
                    return a.pi < b.pi;
                });
                B.add("G2", pre + ".maximal_ideals", "certified", "mi_status", pre + ".generation.status");
                {
                    std::string on;
                    if (thP.size() != ops.size()) {
                        std::vector<std::string> names;
                        for (uint32_t p : thP) names.push_back((N % p == 0 ? "U_" : "T_") + std::to_string(p));
                        on = " on " + Join(names, ",");
                    }
                    B.add("G2", pre + ".decomposed_by", "theta=(" + Join(gt.coeffs, " ") + ")" + on);
                }
                const bool cusp_comp = QF.d > QF.e_k;
                const size_t ncomp = ne + (cusp_comp ? 1 : 0);
                std::vector<std::set<std::vector<uint64_t>>> irr(ncomp);   // the distinct irreducible factors mod ell of each f-bar_j, c-bar
                for (size_t j = 0; j < ne; ++j)
                    for (const auto& fa : hecke::FactorPoly(fM[j]).factors) irr[j].insert(fa.pi.coeffs);
                if (cusp_comp && cb.coeffs.size() > 1)
                    for (const auto& fa : hecke::FactorPoly(cb).factors) irr[ne].insert(fa.pi.coeffs);
                std::vector<std::string> cZ;   // c_theta' over Z where f_theta' is exact
                if (cusp_comp && exact) {
                    cZ = exact->f;
                    for (size_t j = 0; j < ne; ++j) cZ = eisup::PolyDivExact(cZ, fZ[j]);
                }
                auto degree_of = [&](size_t j) -> int64_t { return j < ne ? static_cast<int64_t>(EB.comps[j].degree) : QF.d - QF.e_k; };
                for (size_t i = 0; i < mis.size(); ++i) {
                    const MI& m = mis[i];
                    const uint32_t fdeg = static_cast<uint32_t>(m.pi.size() - 1);
                    const std::string mp = pre + ".m[" + std::to_string(i) + "]";
                    B.add("G2", mp + ".residue_degree", std::to_string(fdeg), "int");
                    B.add("G2", mp + ".residue_poly", PolyText(m.pi));
                    B.add("G2", mp + ".support", support_word(ell, gt.coeffs, thP, m.pi, true), "support");
                    if (ell == 2 || ell == 3) B.add("G2", mp + ".degree_attr", "not_run", "degree_attr");
                    add_local_type(pre, mp, m.mult, fdeg);
                    std::vector<bool> thr(ncomp, false);
                    for (size_t j = 0; j < ncomp; ++j) thr[j] = irr[j].count(m.pi) > 0;
                    for (size_t j = 0; j < ncomp; ++j)
                        B.add("G2", mp + ".components_through." + std::to_string(j), thr[j] ? "certified" : "not_run", "through",
                              pre + ".generation.status");
                    for (size_t j = 0; j < ncomp; ++j)
                        for (size_t j2 = j + 1; j2 < ncomp; ++j2) {
                            if (!thr[j] || !thr[j2]) continue;
                            // depth = v_ell Res(f_j, f_j') iff exactly one residue factor is common to the two (then the local length at m)
                            size_t common = 0;
                            for (const auto& pi : irr[j]) common += irr[j2].count(pi);
                            int64_t depth = -1;
                            if (common == 1) {
                                if (j2 < ne) depth = eisup::ResultantValuation(fZ[j], fZ[j2], ell);
                                else if (!cZ.empty()) depth = eisup::ResultantValuation(fZ[j], cZ, ell);
                            }
                            const std::string pk = mp + ".pair_kinds[" + std::to_string(j) + "," + std::to_string(j2) + "]";
                            B.add("G2", pk + ".depth", depth >= 0 ? std::to_string(depth) : "not_run", depth >= 0 ? "int" : "not_run",
                                  pre + ".generation.status");
                            B.add("G2", pk + ".status", depth >= 0 ? "certified" : "not_run", "status", pre + ".generation.status");
                            const bool deg1 = degree_of(j) == 1 && degree_of(j2) == 1;
                            B.add("G2", pk + ".kind", deg1 ? "meeting" : "not_run", "pair_kind", pre + ".generation.status");
                        }
                }
                continue;
            }
            // lambda_j(theta') exactly ((F2) for T_p, T_ell included; the line's U_p), the tamper of the validation (audit M4) before it
            eisup::EisUpResult EUl = EU;
            if (s.eisup_tamper_ell == ell) {
                const std::vector<uint32_t> Up = EUl.Uprimes();
                size_t ip = 0;
                while (ip < Up.size() && Up[ip] != ell) ++ip;
                if (ip == Up.size()) ip = 0;
                EUl.lines[0].U[ip] = eisup::HexPlus(EUl.lines[0].U[ip], 1);
            }
            const size_t ne = EUl.lines.size();
            std::vector<std::string> lamZ(ne);
            std::vector<uint64_t> lamM(ne);
            for (size_t j = 0; j < ne; ++j) {
                lamZ[j] = eisup::LambdaOf(EUl, j, thP, gt.coeffs);
                lamM[j] = eisup::HexModEll(lamZ[j], ell);
            }
            const eisup::Attribution att = eisup::AttributeAtEll(gt.fbar, lamM, ell);   // C-E7: remainder 0, each root at its multiplicity
            const auto fac = hecke::FactorPoly(hecke::FpPoly64(ell, gt.fbar));
            struct MI {
                std::vector<uint64_t> pi;
                uint32_t mult;
            };
            std::vector<MI> mis;
            for (const auto& fa : fac.factors) mis.push_back(MI{fa.pi.coeffs, fa.multiplicity});
            std::sort(mis.begin(), mis.end(), [](const MI& a, const MI& b) {
                if (a.pi.size() != b.pi.size()) return a.pi.size() < b.pi.size();
                return a.pi < b.pi;
            });
            B.add("G2", pre + ".maximal_ideals", "certified", "mi_status", pre + ".generation.status");
            {
                std::string on;
                if (thP.size() != ops.size()) {
                    std::vector<std::string> names;
                    for (uint32_t p : thP) names.push_back((N % p == 0 ? "U_" : "T_") + std::to_string(p));
                    on = " on " + Join(names, ",");
                }
                B.add("G2", pre + ".decomposed_by", "theta=(" + Join(gt.coeffs, " ") + ")" + on);
            }
            const bool cusp_comp = QF.d > QF.e_k;
            const size_t ncomp = ne + (cusp_comp ? 1 : 0);
            for (size_t i = 0; i < mis.size(); ++i) {
                const MI& m = mis[i];
                const uint32_t fdeg = static_cast<uint32_t>(m.pi.size() - 1);
                const std::string mp = pre + ".m[" + std::to_string(i) + "]";
                B.add("G2", mp + ".residue_degree", std::to_string(fdeg), "int");
                B.add("G2", mp + ".residue_poly", PolyText(m.pi));
                B.add("G2", mp + ".support", support_word(ell, gt.coeffs, thP, m.pi, true), "support");
                if (ell == 2 || ell == 3) B.add("G2", mp + ".degree_attr", "not_run", "degree_attr");
                add_local_type(pre, mp, m.mult, fdeg);
                // component j passes through m iff pi | f_j mod ell ([C, prop:heckeZthrough]); f_j = x - lambda_j(theta') (Eisenstein), c-bar (cusp)
                std::vector<bool> thr(ncomp, false);
                for (size_t j = 0; j < ne; ++j) thr[j] = fdeg == 1 && m.pi[0] == (ell - lamM[j]) % ell && m.pi[1] == 1;
                if (cusp_comp) {
                    const hecke::FpPoly64 piP(ell, m.pi), cP(ell, att.cbar);
                    thr[ne] = att.cbar.size() > 1 && hecke::FpPoly64::mod(cP, piP).is_zero();
                }
                for (size_t j = 0; j < ncomp; ++j)
                    B.add("G2", mp + ".components_through." + std::to_string(j), thr[j] ? "certified" : "not_run", "through",
                          pre + ".generation.status");
                for (size_t j = 0; j < ncomp; ++j)
                    for (size_t j2 = j + 1; j2 < ncomp; ++j2) {
                        if (!thr[j] || !thr[j2]) continue;
                        // depth = v_ell Res(f_j, f_j') at theta': exact for two Eisenstein components, or where f_theta' is exact
                        int64_t depth = -1;
                        if (j2 < ne) depth = eisup::ValuationOfDifference(lamZ[j], lamZ[j2], ell);
                        else if (exact) depth = eisup::CuspResultantValuation(exact->f, lamZ, j, ell);
                        const std::string pk = mp + ".pair_kinds[" + std::to_string(j) + "," + std::to_string(j2) + "]";
                        B.add("G2", pk + ".depth", depth >= 0 ? std::to_string(depth) : "not_run", depth >= 0 ? "int" : "not_run",
                              pre + ".generation.status");
                        B.add("G2", pk + ".status", depth >= 0 ? "certified" : "not_run", "status", pre + ".generation.status");
                        const bool deg1 = j2 < ne || QF.d - QF.e_k == 1;
                        B.add("G2", pk + ".kind", deg1 ? "meeting" : "not_run", "pair_kind", pre + ".generation.status");
                    }
            }
            continue;
        }
        if (!full_N && c1_ok) {
            const LocalReadOut& rd = e.c3_from_fallback ? e.c3out : e.out;
            if (rd.read) {
                B.add("G2", pre + ".generation.dim_A1", std::to_string(rd.L.dim_A1), "int");
                B.add("G2", pre + ".t_ell", rd.L.t_ell, "t_ell");
            } else {
                B.add("G2", pre + ".t_ell", rd.M.t_ell.empty() ? "in_P" : rd.M.t_ell, "t_ell");
            }
        }
        // (schema audit F6) under saturation with no generating theta from the read itself, search theta_search's order
        GenTheta searched;
        std::vector<uint32_t> thP = P;   // the operators theta is a combination of (P, or the read set P u {T_ell}: re-audit R9)
        if (gs == "certified" && route == "saturation" && gen_theta < 0 && eis_rational && Zlk::Fits(ell, 1)) {
            const LocalReadOut& rd = e.c3_from_fallback ? e.c3out : e.out;
            if (rd.read && rd.L.dim_A1 == static_cast<uint32_t>(Q.d_target)) {
                searched = GeneratingTheta(s.rat, rd.M, P, ell, static_cast<uint32_t>(rd.L.dim_A1));
                // (re-audit R9, lead decision 2026-11-06) where generation is certified only with T_ell in the read set, no theta over
                // P generates A'_1: search theta_search's order over the read set (T_ell appended, ell <= 23170: the HZ-LOC amendment)
                if (!searched.ok && rd.L.t_ell == "added" && rd.M.P.size() == P.size() + 1 && rd.M.P.back() == ell) {
                    searched = GeneratingTheta(s.rat, rd.M, rd.M.P, ell, static_cast<uint32_t>(rd.L.dim_A1));
                    if (searched.ok) thP = rd.M.P;
                }
                // (S27 TS) lex_then_greedy: the capped continuation as the last resort, over P, then over the read set
                if (!searched.ok && s.rat.theta_order == ThetaOrder::LexThenGreedy) {
                    searched = GeneratingTheta(s.rat, rd.M, P, ell, static_cast<uint32_t>(rd.L.dim_A1), nullptr, true);
                    if (!searched.ok && rd.L.t_ell == "added" && rd.M.P.size() == P.size() + 1 && rd.M.P.back() == ell) {
                        searched = GeneratingTheta(s.rat, rd.M, rd.M.P, ell, static_cast<uint32_t>(rd.L.dim_A1), nullptr, true);
                        if (searched.ok) thP = rd.M.P;
                    }
                }
            }
            if (searched.ok && thP.size() == P.size())   // a theta of HZ-RAT's own list: its exact f and c_theta(lambda) are used
                for (size_t j = 0; j < Q.thetas.size(); ++j)
                    if (Q.thetas[j].primitive && Q.thetas[j].coeffs == searched.coeffs) gen_theta = static_cast<int32_t>(j);
        }
        // maximal ideals: only where T (x) Z_ell = Z_ell[theta] is certified (saturation + theta generating, or the rational route)
        const bool decomposable = gs == "certified" && (gen_theta >= 0 || searched.ok) && (route == "saturation" || route == "rational") &&
                                  Zlk::Fits(ell, 1) && eis_rational;
        g8info[ell] = G8Ell{};
        if (!decomposable) {
            B.add("G2", pre + ".maximal_ideals", "not_run", "mi_status", pre + ".generation.status");
            if (gs != "certified" && !full_N && c1_ok) {
                const LocalReadOut& rd = e.c3_from_fallback ? e.c3out : e.out;
                if (rd.read) B.add("G2", pre + ".A1_of_computed.dim_A1", std::to_string(rd.L.dim_A1), "int");
            }
            continue;
        }
        std::vector<uint32_t> th_coeffs;
        hecke::FpPoly64 fbar(ell);
        uint64_t lam = 0;
        int32_t v_res = -1;   // v_ell c_theta(lambda): known for HZ-RAT's thetas only
        if (gen_theta >= 0) {
            const ThetaEntry& T = Q.thetas[static_cast<size_t>(gen_theta)];
            th_coeffs = T.coeffs;
            fbar = hecke::FpPoly64(ell, Residues(T.f, ell));
            lam = local::HexResidue(Zlk::Make(ell, 1), T.lambda_hex);
            v_res = CertVal(T.eis_res, ell);
        } else {
            th_coeffs = searched.coeffs;
            fbar = hecke::FpPoly64(ell, searched.fbar);
            for (size_t i = 0; i < thP.size(); ++i)   // lambda_theta = sum c_p (1 + p^{k-1}) mod ell (T_ell: 1 + ell^{k-1} = 1)
                lam = (lam + MulMod(th_coeffs[i] % ell, (1 + PowMod(thP[i], k - 1, ell)) % ell, ell)) % ell;
        }
        const auto fac = hecke::FactorPoly(fbar);
        // c = f / (x - lambda) mod ell (synthetic division: q_{d-1} = f_d, q_{i-1} = f_i + lambda q_i)
        std::vector<uint64_t> cbar;
        {
            const std::vector<uint64_t>& fc = fbar.coeffs;
            const size_t dg = fc.size() - 1;
            cbar.assign(dg, 0);
            unsigned __int128 qi = fc[dg];
            cbar[dg - 1] = static_cast<uint64_t>(qi);
            for (size_t i = dg - 1; i >= 1; --i) {
                qi = (static_cast<unsigned __int128>(fc[i]) + qi * lam) % ell;
                cbar[i - 1] = static_cast<uint64_t>(qi);
            }
        }
        struct MI {
            std::vector<uint64_t> pi;
            uint32_t mult;
        };
        std::vector<MI> mis;
        for (const auto& fa : fac.factors) mis.push_back(MI{fa.pi.coeffs, fa.multiplicity});
        std::sort(mis.begin(), mis.end(), [](const MI& a, const MI& b) {
            if (a.pi.size() != b.pi.size()) return a.pi.size() < b.pi.size();
            return a.pi < b.pi;
        });
        B.add("G2", pre + ".maximal_ideals", "certified", "mi_status", pre + ".generation.status");
        g8info[ell].decomposed = true;
        g8info[ell].ops = thP;
        g8info[ell].theta = th_coeffs;
        // theta's coefficients on P, or (R9) on the read set P u {T_ell}: then the operators are named
        B.add("G2", pre + ".decomposed_by",
              "theta=(" + Join(th_coeffs, " ") + ")" + (thP.size() == P.size() ? std::string() : " on " + [&] {
                  std::vector<std::string> ops;
                  for (uint32_t p : thP) ops.push_back("T_" + std::to_string(p));
                  return Join(ops, ",");
              }()));
        for (size_t i = 0; i < mis.size(); ++i) {
            const MI& m = mis[i];
            const uint32_t fdeg = static_cast<uint32_t>(m.pi.size() - 1);
            const std::string mp = pre + ".m[" + std::to_string(i) + "]";
            B.add("G2", mp + ".residue_degree", std::to_string(fdeg), "int");
            B.add("G2", mp + ".residue_poly", PolyText(m.pi));
            B.add("G2", mp + ".support", support_word(ell, th_coeffs, thP, m.pi, false), "support");
            if (ell == 2 || ell == 3) B.add("G2", mp + ".degree_attr", "not_run", "degree_attr");   // B4(2): no route computes it
            add_local_type(pre, mp, m.mult, fdeg);
            // components through m (01 P 2.6 at a generating theta: component i passes iff pi | f_i mod ell)
            const bool eis_through = fdeg == 1 && m.pi[0] == (ell - lam) % ell && m.pi[1] == 1;
            const hecke::FpPoly64 piP(ell, m.pi), cP(ell, cbar);
            const bool cusp_through = cbar.size() > 1 && hecke::FpPoly64::mod(cP, piP).is_zero();
            B.add("G2", mp + ".components_through.eis", eis_through ? "certified" : "not_run", "through", pre + ".generation.status");
            g8info[ell].mis.push_back(G8Mi{m.pi, m.mult, eis_through, cusp_through});
            B.add("G2", mp + ".components_through.cusp", cusp_through ? "certified" : "not_run", "through", pre + ".generation.status");
            if (eis_through && cusp_through) {
                B.add("G2", mp + ".pair_kinds[eis,cusp].depth", v_res >= 0 ? std::to_string(v_res) : "not_run",
                      v_res >= 0 ? "int" : "not_run", pre + ".generation.status");
                B.add("G2", mp + ".pair_kinds[eis,cusp].status", v_res >= 0 ? "certified" : "not_run", "status", pre + ".generation.status");
                B.add("G2", mp + ".pair_kinds[eis,cusp].kind", Q.d_target - 1 == 1 ? "meeting" : "not_run", "pair_kind",
                      pre + ".generation.status");
            }
        }
    }

    // =================================================================================================================================
    // G3 congruence modules (the Eisenstein partition)
    if (full_N) {
        B.add("G3", "G3.partition", "U_p_tags", "partition");
        bool all_ub = !utags.empty();
        for (const auto& u : utags) all_ub = all_ub && u.upper_bound_certified;
        // (Z2 fix round, audit S1; [C] 1.26.0-pre.10 W-1) the rows follow the primes p | N ascending, which is the order of the tag loop above,
        // so where every tag ran (every record of this version) the rows and their order are exactly those of the loop over utags
        const std::vector<uint32_t> g3_primes = up::PrimesDividing(N);
        const auto tag_of = [&](uint32_t p) -> const UTagResult* {
            for (const auto& u : utags)
                if (u.p == p) return &u;
            return nullptr;
        };
        bool all_ran = true;
        for (uint32_t p : g3_primes) all_ran = all_ran && tag_of(p) != nullptr;
        for (uint32_t p : g3_primes) {
            const UTagResult* const tag = tag_of(p);
            if (!tag) {
                B.add("G3", "G3.C_T[U_" + std::to_string(p) + "].status", up::FullG3Rule(false, false, false, all_ran).tag_status, "utag");
                continue;
            }
            const UTagResult& u = *tag;
            const std::string st = up::FullG3Rule(u.upper_bound_certified, all_ub, true, all_ran).tag_status;   // (S26 Z2) the tag's status word
            for (const auto& bd : u.bounds) {
                const std::string bp = "G3.C_T[U_" + std::to_string(u.p) + ":" + bd.a + "," + bd.b + "]";
                B.add("G3", bp + ".upper", std::to_string(bd.bound), "int");
                B.add("G3", bp + ".status", st, "utag");
            }
            B.add("G3", "G3.C_T[U_" + std::to_string(u.p) + "].status", st, "utag");
        }
        const up::FullG3Words G3W = up::FullG3Rule(all_ub, all_ub, all_ran, all_ran);   // (S26 Z2) [C] item 8 (d')(ii): the one place
        B.add("G3", "G3.C_T.status", G3W.ct_status, "ct_status");
        B.add("G3", "G3.C_T.lower", G3W.ct_lower, "status");
        B.add("G3", "G3.fibre_product", "not_certified", "fibre_product");
    } else if (!eis_rational) {
        B.add("G3", "G3.partition", c1_ok ? "general" : "none(C1_open)", "partition");
        B.add("G3", "G3.general_case", "bounds", "general_case");
        B.add("G3", "G3.fibre_product", "not_certified", "fibre_product");
    } else if (!has_cusp) {
        // (schema audit F5; re-audit R3) d = 1: one component (the Eisenstein one), no partition and no congruence module
        B.add("G3", "G3.partition", "none(single_component)", "partition");
    } else {
        B.add("G3", "G3.partition", "eisenstein", "partition");   // S = the Eisenstein component, S' = the cuspidal block
        B.add("G3", "G3.S", "0", "list:int");
        B.add("G3", "G3.S_prime", "1", "list:int");
        if (!GPIsBound(Q)) {
            // (S27 TS-FIX, TS_design.md Sec. 5.4 item 4; [C] item 8 (j)) g_P open(no primitive T_p) -- the step was certified by a combination,
            // so it has no primitive unit -- is not a bound: G3 has no global set of primes to read ell-wise and the congruence module is not
            // run.  No upper / upper_v / read / cofactor / cofactor_test / exact / lower_v / eta_L / second-route / Mazur word (before the fix
            // the default Factored was read as g_P = 1, and G3 would have claimed C_T certified with exact 0x1).
            B.add("G3", "G3.C_T.status", "not_run", "ct_status");
            B.add("G3", "G3.fibre_product", "not_certified", "fibre_product");
        } else {
            bool all_cert = Q.gP.complete();
            Limbs exact = {1};
            std::vector<std::string> upper_txt;
            for (const auto& pp : Q.gP.factors) {
                if (!ct_factor(pp)) continue;   // (EXW) the predicate of the root trigger (ct_listed)
                const uint64_t ell = pp.q;
                const auto it = S.find(ell);
                ENGINE_CHECK(it != S.end());
                const EllRead& e = it->second;
                int32_t v_up = static_cast<int32_t>(pp.e);
                if (e.out.read && e.out.L.v_gP >= 0) v_up = std::min(v_up, e.out.L.v_gP);   // g_{P u {ell}}, ell-wise (item 8 (j))
                const std::string pre = "G3.C_T.ell=" + std::to_string(ell);
                B.add("G3", pre + ".upper_v", std::to_string(v_up), "int");
                B.add("G3", pre + ".read", e.root, "read_word");
                std::string st;
                if (e.root == "certified") st = "certified";
                else if (e.root == "upper_bound_certified" || e.root == "refused(word_modulus)") st = "upper_bound_certified";
                else st = e.root;   // open(precision) final; refused(export_word) / refused(k1_route) / blocked(export) pending
                B.add("G3", pre + ".status", st, "ct_ell");
                if (e.out.read && e.out.L.v_eta >= 0) {
                    B.add("G3", pre + ".lower_v", std::to_string(e.out.L.v_eta), "int");
                    B.add("G3", "G3.eta_L.ell=" + std::to_string(ell) + ".exponents", Join(e.out.L.eta));
                    B.add("G3", "G3.eta_L.ell=" + std::to_string(ell) + ".r", std::to_string(e.out.L.eta.size()), "int");
                    B.add("G3", "G3.eta_L.ell=" + std::to_string(ell) + ".sign", N == 1 ? "+" : "both", "sign");
                }
                if (st == "certified") {
                    for (int32_t j = 0; j < v_up; ++j) LimbMul(exact, ell);
                } else {
                    all_cert = false;
                }
                upper_txt.push_back(std::to_string(ell) + "^" + std::to_string(v_up));
            }
            std::string ups = upper_txt.empty() ? "1" : Join(upper_txt, "*");
            if (!Q.gP.complete()) ups += "*[c:" + Q.gP.cofactor_hex + "]";
            B.add("G3", "G3.C_T.upper", ups);
            B.add("G3", "G3.C_T.cofactor", Q.gP.cofactor_hex, "zbig");
            B.add("G3", "G3.C_T.cofactor_test", Q.gP.cofactor_test, "cofactor_test");
            const std::string cts = !Q.gP.complete() ? "partial(unfactored)" : (all_cert ? "certified" : "upper_bound_certified");
            B.add("G3", "G3.C_T.status", cts, "ct_status");
            if (cts == "certified") B.add("G3", "G3.C_T.exact", LimbHex(exact), "zbig", "G3.C_T.status");
            B.add("G3", "G3.fibre_product", cts == "certified" ? "certified" : "not_certified", "fibre_product");
            // second route (never lifts a status)
            if (N == 1) {
                for (const auto& pp : Q.gP.factors) {
                    if (pp.proof == "probable" || pp.q < 5 || k % (pp.q - 1) == 0) continue;
                    const uint64_t ell = pp.q;
                    const EllRead& e = S.at(ell);
                    const int32_t vB = src.v_bernoulli(ell, k, pp.e + 2);
                    const std::string pre = "G3.second_route.ell=" + std::to_string(ell);
                    B.add("G3", pre + ".v_Bk", vB >= 0 ? std::to_string(vB) : "not_run", vB >= 0 ? "int" : "not_run");
                    std::string ag = "not_read";
                    if (vB >= 0 && e.out.read && e.out.L.v_eta >= 0 && e.root == "certified") ag = vB == e.out.L.v_eta ? "true" : "false";
                    B.add("G3", pre + ".agrees", ag, "agrees");
                }
            } else if (n == 0 && IsPrimeU32(N)) {
                uint64_t num = N - 1, den = 12;
                const uint64_t g = std::gcd(num, den);
                num /= g;
                for (const auto& pp : Q.gP.factors) {
                    if (pp.proof == "probable" || pp.q < 5) continue;
                    uint32_t v = 0;
                    uint64_t x = num;
                    while (x % pp.q == 0) {
                        x /= pp.q;
                        ++v;
                    }
                    B.add("G3", "G3.second_route.ell=" + std::to_string(pp.q) + ".mazur_v_n", std::to_string(v), "int");
                }
            }
        }
    }

    // =================================================================================================================================
    // G4 - G6 from HZ-TOR's ReadTor (q = 1: the extension 0 -> tors -> H^1 / ell^K -> L / ell^K -> 0); q = 2 has no route (not_run).
    // (schema audit F3) one row group per TORSION POINT: a joint T'-block of H^1 / ell^e with t_m > 0, keyed pt[i] in the read's block
    // order (sorted by the residual system), its system in G5.  G4 at the common points (support = both), G5 and G6 at every torsion
    // point.  (F13) G4, then G5, then G6.  (F8) a refused read maps by the export's refusal name.
    auto tor_word = [](const tor::TorRead& t) -> std::string {
        if (t.status == "blocked(export)") return ExportWord(t.detail);
        if (t.status == "open(precision)") return "refused(export_word)";   // K_max < e: EXPORT's ring limit ([C] item 8 (c))
        return t.status;
    };
    auto system_text = [](const tor::TorRead& t, const tor::BlockRead& b) {
        std::vector<std::string> parts;
        for (size_t jj = 0; jj < t.P.size() && jj < b.residue.size(); ++jj) parts.push_back("T_" + std::to_string(t.P[jj]) + ":" + PolyText(b.residue[jj]));
        return Join(parts, ";");
    };
    auto torsion_points = [](const tor::TorRead& t) {
        std::vector<const tor::BlockRead*> v;
        for (const auto& b : t.blocks)
            if (b.t_m > 0) v.push_back(&b);
        return v;
    };
    // G4: the glue J per common point
    for (const auto& [ell, t] : TR) {
        const std::string pre = "G4.ell=" + std::to_string(ell);
        if (t.status != "ok") {
            B.add("G4", pre + ".J.status", t.status == "no_torsion" ? "zero" : tor_word(t), "J");
            continue;
        }
        const auto pts = torsion_points(t);
        for (size_t i = 0; i < pts.size(); ++i) {
            const tor::BlockRead& b = *pts[i];
            if (b.kind != "common") continue;
            const std::string pp = pre + ".pt[" + std::to_string(i) + "].J";
            if (b.J == "zero") {
                B.add("G4", pp + ".status", "zero", "J");
                continue;
            }
            const size_t c = b.J.find("^");
            ENGINE_CHECK_MSG(b.J.compare(0, 7, "bounds(") == 0 && c != std::string::npos, "HeckeZRecord: an unlisted HZ-TOR J word");
            // (F12) lower = 0, a valid lower bound for log_ell |J cap T'| (v1 has no other); upper = the Hom bound, never exact (CB-3)
            B.add("G4", pp + ".status", "bounds", "J");
            B.add("G4", pp + ".lower", "0", "int");
            B.add("G4", pp + ".upper", std::to_string(std::stoul(b.J.substr(c + 1))), "int");
        }
    }
    // G5: per torsion point, its system and its relation to the free part
    for (const auto& [q, ell] : R.descent.torsion_q_ell) {
        const std::string pre5 = "G5.q=" + std::to_string(q) + ".ell=" + std::to_string(ell);
        const auto it = q == 1 ? TR.find(ell) : TR.end();
        if (it == TR.end() || it->second.status != "ok") {   // q = 2: no package reads these (F12); a refused q = 1 read: its word
            const std::string w = it == TR.end() || it->second.status == "no_torsion" ? "not_run" : tor_word(it->second);
            B.add("G5", pre5 + ".relation_to_free", w, "relation_to_free");
            B.add("G5", pre5 + ".generation_tors", "not_run", "generation_tors");
            continue;
        }
        const tor::TorRead& t = it->second;
        const auto pts = torsion_points(t);
        for (size_t i = 0; i < pts.size(); ++i) {
            const tor::BlockRead& b = *pts[i];
            const std::string pp = pre5 + ".pt[" + std::to_string(i) + "]";
            B.add("G5", pp + ".system", system_text(t, b));
            if (b.kind == "common") {
                if (b.coincidence == "certified(B8)") {
                    B.add("G5", pp + ".relation_to_free", "common_certified", "relation_to_free");
                    B.add("G5", pp + ".route", "T_ell_level1", "coincidence_route");
                } else if (b.coincidence == "certified(B14)") {
                    B.add("G5", pp + ".relation_to_free", "common_certified", "relation_to_free");
                    B.add("G5", pp + ".route", "nonsplit", "coincidence_route");
                } else {
                    B.add("G5", pp + ".relation_to_free", "agrees_on_P", "relation_to_free");
                }
            } else {
                // a torsion-only point: distinct from every free point (B7); the witness is the first T_p whose residue differs
                // from that of every point with r_m > 0, else the joint system ("joint")
                std::string wit = "joint";
                for (size_t jj = 0; jj < t.P.size() && wit == "joint"; ++jj) {
                    bool differs = true;
                    for (const auto& f : t.blocks)
                        if (f.r_m > 0 && jj < f.residue.size() && jj < b.residue.size() && f.residue[jj] == b.residue[jj]) differs = false;
                    if (differs) wit = "T_" + std::to_string(t.P[jj]);
                }
                B.add("G5", pp + ".relation_to_free", "distinct_certified", "relation_to_free");
                B.add("G5", pp + ".witness", wit, "op_or_joint");
            }
            B.add("G5", pp + ".generation_tors", "not_run", "generation_tors");
        }
    }
    // G6: per torsion point the splitting (F1: split(T') is split_on_P, a statement about T'), the Hom bound; the Tor term by reference
    for (const auto& [q, ell] : R.descent.torsion_q_ell) {
        const std::string pre6 = "G6.q=" + std::to_string(q) + ".ell=" + std::to_string(ell);
        const auto it = q == 1 ? TR.find(ell) : TR.end();
        if (it == TR.end() || it->second.status != "ok") {
            const std::string w = it == TR.end() || it->second.status == "no_torsion" ? "not_run" : tor_word(it->second);
            B.add("G6", pre6 + ".split", w, "split");
            B.add("G6", pre6 + ".hom_order", w, "status");
        } else {
            const tor::TorRead& t = it->second;
            const auto pts = torsion_points(t);
            for (size_t i = 0; i < pts.size(); ++i) {
                const tor::BlockRead& b = *pts[i];
                const std::string pp = pre6 + ".pt[" + std::to_string(i) + "]";
                std::string sp;
                if (b.kind != "common") sp = "split_by_support";
                else if (b.split == "split(T')") sp = "split_on_P";
                else if (b.split == "non_split") sp = "non_split";
                else throw std::logic_error("HeckeZRecord: an unlisted HZ-TOR block split word '" + b.split + "'");
                B.add("G6", pp + ".split", sp, "split");
                B.add("G6", pp + ".hom_order", std::to_string(b.hom_length), "int");
            }
        }
        B.add("G6", pre6 + ".tor_ref", "descent:(q=" + std::to_string(q) + ",ell=" + std::to_string(ell) + ")");
    }
    // G7: the descent cell of descent_ref ran parts = groups,hecke (no pair)
    B.add("G7", "G7.pair", "not_run", "not_run");
    B.add("G7", "G7.theory", "H_c", "theory");
    B.add("G7", "G7.hecke", "not_run", "not_run");
    // G8: (schema audit F2, lead decision 2026-11-06) not in default v1 records; H1 writes it under the non-default knob G8 (above).
    // (slot-25 H1) heckez.G8 = hida: the Hida block per maximal ideal, by the rule of its settings line (G8Line).
    if (g8_hida) {
        if (N != 1) {
            B.add("G8", "G8.scope", "not_run(level_N)", "g8_scope");
        } else {
            B.add("G8", "G8.scope", "level1", "g8_scope");
            for (const auto& [ell, gi] : g8info) {
                if (ell < 5) continue;
                const std::string pre = "G8.ell=" + std::to_string(ell);
                if (!gi.decomposed) {
                    B.add("G8", pre + ".blocks", "not_run(no_decomposition)", "g8_blocks");
                    continue;
                }
                if (k % (ell - 1) == 0) {   // the class a = 0: no Eisenstein congruence ([U, rmk:u-classzero]), no Bernoulli route
                    B.add("G8", pre + ".blocks", "not_run(class_zero)", "g8_blocks");
                    continue;
                }
                // G3's ell-part of C_T (upper exponent; exact when certified): the precision of the forms read
                int32_t upper_v = 0;
                bool ct_cert = false;
                if (const Field* f = R.find("G3.C_T.ell=" + std::to_string(ell) + ".upper_v")) upper_v = std::stoi(f->value);
                if (const Field* f = R.find("G3.C_T.ell=" + std::to_string(ell) + ".status")) ct_cert = f->value == "certified";
                const uint32_t W = static_cast<uint32_t>(std::max(2, upper_v + 2));
                bool fits = k >= W + 1;
                {
                    unsigned __int128 pw = 1;
                    for (uint32_t i = 0; i < W && fits; ++i) {
                        pw *= ell;
                        fits = pw < (static_cast<unsigned __int128>(1) << 84);
                    }
                }
                RecordSources::HidaBlockOut hb;
                if (fits) hb = src.hida_block(ell, k, W, gi.ops, gi.theta);
                {
                    std::ostringstream pv;
                    pv << "W=" << W << ";served=" << (fits && hb.served ? "yes" : "no") << ";secs=" << hb.secs;
                    if (!fits) pv << ";refusal=ell^W >= 2^84 or k - 1 < W";
                    else if (!hb.served) pv << ";refusal=" << hb.refusal;
                    g8_prov[ell] = pv.str();
                }
                if (!fits || !hb.served) {
                    B.add("G8", pre + ".blocks", "not_run(forms_cap)", "g8_blocks");
                    continue;
                }
                B.add("G8", pre + ".blocks", "read", "g8_blocks");
                const hecke::FpPoly64 chi(ell, hb.theta_ord_charpoly);
                for (size_t i = 0; i < gi.mis.size(); ++i) {
                    const G8Mi& m = gi.mis[i];
                    const std::string mp = pre + ".m[" + std::to_string(i) + "]";
                    // the multiplicity of pi in chi(theta | e M_k) mod ell: all of m.mult (ordinary) or 0 (not); a local ring is either
                    const hecke::FpPoly64 piP(ell, m.pi);
                    uint32_t mu = 0;
                    hecke::FpPoly64 rest = chi;
                    while (!rest.is_zero() && rest.degree() > 0 && hecke::FpPoly64::mod(rest, piP).is_zero()) {
                        rest = hecke::FpPoly64::divide_exact(rest, piP);
                        ++mu;
                    }
                    ENGINE_CHECK_MSG(mu == 0 || mu == m.mult, ("HeckeZRecord G8: the theta-residue " + PolyText(m.pi) + " at ell = " +
                                                               std::to_string(ell) + " has multiplicity " + std::to_string(mu) + " in e M_k but " +
                                                               std::to_string(m.mult) + " in T (a local ring is ordinary or not)").c_str());
                    const bool ordinary = mu == m.mult;
                    // at level 1 the Eisenstein ideal is ordinary by theorem (a_ell(E_k) = 1 + ell^{k-1}); the forms side must agree
                    ENGINE_CHECK_MSG(!m.eis || ordinary, "HeckeZRecord G8: the Eisenstein ideal is not ordinary on the forms side (a defect)");
                    B.add("G8", mp + ".ordinary", ordinary ? "true" : "false", "bool");
                    if (!(m.eis && m.cusp && ordinary)) continue;
                    ENGINE_CHECK_MSG(hb.g >= 1, "HeckeZRecord G8: an Eisenstein ideal through the cuspidal component with g = 0 on the forms side");
                    B.add("G8", mp + ".chain_key", "(1," + std::to_string(ell) + "," + std::to_string(k % (ell - 1)) + ")", "chain_key");
                    if (!hb.k9.empty()) g8_k9[ell] = hb.k9;
                    B.add("G8", mp + ".g", std::to_string(hb.g), "int");
                    B.add("G8", mp + ".L_k", ct_cert ? std::to_string(upper_v) : "not_run", ct_cert ? "int" : "not_run");
                    B.add("G8", mp + ".v_Bk", hb.v_Bk >= 0 ? std::to_string(hb.v_Bk) : "not_run", hb.v_Bk >= 0 ? "int" : "not_run");
                    std::string st = "open";
                    if (hb.g == 1) st = "fibre_LxL";   // [U, prop:u-H11]
                    else if (hb.v_Bk >= 0 && hb.I0_q != 0 && hb.lambda_one) st = "regular_deg_g";   // [U, prop:u-H12](1), (2), Steps 1-2
                    B.add("G8", mp + ".structure", st, "g8_structure");
                    B.add("G8", mp + ".I0_generator", hb.I0_q ? "T_" + std::to_string(hb.I0_q) : "none_found", "i0_generator");
                }
            }
        }
    }

    // =================================================================================================================================
    // G9 the certificate layer (07-lean-plan.md Sec. 11.1)
    B.add("G9", "G9.caps.k1_cap", "f<=" + std::to_string(s.k1_cap_f) + ",b1<=" + std::to_string(s.k1_cap_b1));
    B.add("G9", "G9.caps.k7_cap", "rank<=" + std::to_string(s.k7_cap_rank) + ",tors<=" + std::to_string(s.k7_cap_tors));
    B.add("G9", "G9.ranks", "b1=" + std::to_string(b1) + ";e_k=" + std::to_string(Q.e_k) + ";d=" + std::to_string(Q.d) +
                                ";d_an=" + std::to_string(Q.d_an) + ";hES=" + (N == 1 || full_N ? "hES_N" : "hES_N_an"));
    if (!full_N && c1_ok) {
        for (uint32_t j : Q.discriminants_used) {
            const ThetaEntry& T = Q.thetas[j];
            const std::string pre = "G9.K2.theta=(" + Join(T.coeffs, " ") + ")";
            B.add("G9", pre + ".f", "[" + Join(T.f) + "]");
            std::string chi = "digest_only";
            for (size_t i = 0; i < T.coeffs.size(); ++i)
                if (T.coeffs[i] == 1 && std::count(T.coeffs.begin(), T.coeffs.end(), 0u) + 1 == static_cast<long>(T.coeffs.size()) &&
                    std::find(R.descent.primes.begin(), R.descent.primes.end(), P[i]) != R.descent.primes.end())
                    chi = "ref:descent_ref.charpoly(p=" + std::to_string(P[i]) + ")";
            B.add("G9", pre + ".chi", chi);
            const bool small = T.f.size() - 1 <= s.bezout_cap_d;
            B.add("G9", pre + ".bezout", "by_generator", "bezout");
            B.add("G9", pre + ".bezout_within_record_cap", small ? "true" : "false", "bool");
            for (const auto& pp : T.disc.factors)
                B.add("G9", pre + ".D_factor.q=" + pp.q_hex, std::to_string(pp.e) + ":" + pp.proof + ":pratt_ref=by_generator");
            B.add("G9", pre + ".cofactor", T.disc.cofactor_hex, "zbig");
        }
        std::vector<std::string> sl;
        for (const auto& [ell, e] : S) sl.push_back(std::to_string(ell));
        B.add("G9", "G9.S", Join(sl));
        std::vector<std::string> tb_mat;
        for (const auto& [ell, e] : S) {
            const LocalReadOut& rd = e.c3_from_fallback ? e.c3out : e.out;
            // (slot-27 EXW) K3 below follows the root and no longer sits behind rd.read (byte-neutral before EXW: rd = e.out there)
            const std::string dsha = rd.read ? Sha256Hex(DumpText(rd.M)) : std::string();
            if (rd.read) tb_mat.push_back(std::to_string(ell) + ":" + dsha);
            const std::string pre = "G9.K1.ell=" + std::to_string(ell);
            if (rd.read && gen_status[ell] == "certified" && e.c3 == "certified") {
                B.add("G9", pre + ".K", std::to_string(rd.M.K), "int");
                B.add("G9", pre + ".f_m", std::to_string(rd.M.D), "int");
                B.add("G9", pre + ".r", std::to_string(Q.d_target), "int");
                B.add("G9", pre + ".e_hat_m", std::to_string(Q.e_k), "int");
                B.add("G9", pre + ".generators", Join(rd.M.P));
                B.add("G9", pre + ".scope", "global");
                B.add("G9", pre + ".satcert_sha256", dsha, "sha256");
            }
            // K3: ell | g_P only (S27 TS-FIX: a bound).  (slot-27 EXW, design Sec. 3.2) the guard follows the root (e.out.read), not the (C3)
            // path: before EXW c3_from_fallback == !e.out.read on every record (byte-neutral); with a root trigger a served root stands beside
            // a K = 1 (C3).
            if (e.out.read && eis_rational && GPIsBound(Q) && CertVal(Q.gP, ell) > 0) {
                const std::string p3 = "G9.K3.ell=" + std::to_string(ell);
                const local::RootRead* th = nullptr;
                for (const auto& rr : e.out.L.roots)
                    if (rr.op == "theta" && rr.status == "ok") th = &rr;
                if (!th) {
                    B.add("G9", p3, "absent(no_theta_read)", "cert_absent");
                    continue;
                }
                // the witness: the first entry (row-major) of c_theta(theta) mod ell^K of minimal valuation
                const Zlk Rk = Zlk::Make(ell, e.out.M.K);
                const size_t Dm = e.out.M.D;
                local::Mat th_m(Dm * Dm, 0);
                for (size_t i = 0; i < TH->coeffs.size(); ++i)
                    th_m = local::MatAdd(Rk, th_m, local::MatScale(Rk, e.out.M.T[i], TH->coeffs[i] % Rk.q));
                std::vector<uint64_t> cpol;
                for (const auto& h : Q.cusp_part) cpol.push_back(local::HexResidue(Rk, h));
                const local::Mat C = local::MatPolyEval(Rk, cpol, th_m, Dm);
                uint32_t vmin = e.out.M.K;
                size_t wi = 0, wj = 0;
                for (size_t r = 0; r < Dm; ++r)
                    for (size_t c = 0; c < Dm; ++c) {
                        const uint32_t v = Rk.val(C[r * Dm + c]);
                        if (v < vmin) {
                            vmin = v;
                            wi = r;
                            wj = c;
                        }
                    }
                B.add("G9", p3 + ".K", std::to_string(e.out.M.K), "int");
                B.add("G9", p3 + ".v_c_lambda", std::to_string(th->v_c_lambda), "int");
                B.add("G9", p3 + ".a", Join(th->a));
                B.add("G9", p3 + ".r", std::to_string(th->a.size()), "int");
                B.add("G9", p3 + ".witness", "(" + std::to_string(wi) + "," + std::to_string(wj) + "):v=" + std::to_string(vmin));
                B.add("G9", p3 + ".lambda", "formula:1+p^(k-1)");
                B.add("G9", p3 + ".c_theta", "ref:G1.components[1].minpoly");
                // (EXW) the dump of the root's own module (= rd's before EXW, where e.out is rd whenever K3 is written)
                B.add("G9", p3 + ".dump_sha256", e.c3_from_fallback ? Sha256Hex(DumpText(e.out.M)) : dsha, "sha256");
            }
        }
        if (cusp_irred_q) B.add("G9", "G9.irreducible.cusp.q", std::to_string(cusp_irred_q), "int");   // (F5) c_theta irreducible mod q
        B.add("G9", "G9.K4", "absent(not_built)", "cert_absent");
        B.add("G9", "G9.trust.tb_mat", Join(tb_mat, ";"));
    }
    if (TR.empty()) B.add("G9", "G9.K5_K8", "absent(not_applicable)", "cert_absent");
    for (const auto& [ell, t] : TR) {
        const std::string pre = "G9.K6_K7.ell=" + std::to_string(ell);
        uint32_t sep = 0, chk = 0;
        for (const auto& x : t.sep) {
            sep += x.separated;
            chk += x.bezout_checked;
        }
        std::vector<std::string> lean;
        for (const auto& bl : t.blocks) lean.push_back(bl.kind + ":" + bl.split + ":" + bl.lean);
        B.add("G9", pre + ".K6", "separated=" + std::to_string(sep) + "/" + std::to_string(t.sep.size()) + ";bezout_checked=" + std::to_string(chk));
        B.add("G9", pre + ".K7", Join(lean, ";"));
        B.add("G9", pre + ".tor_digest", Hex64(t.digest()), "hex64");
    }
    if (g8_hida && !g8_k9.empty()) {   // (H1) heckez.G8 = hida: K9's integers per ordinary Eisenstein ideal (07 v0.3 Sec. 5.9)
        for (const auto& [ell, t] : g8_k9) B.add("G9", "G9.K9.ell=" + std::to_string(ell), t);
    } else {   // (H1 audit F3) under `hida` H1 ran and K9 does not apply (no ordinary Eisenstein ideal through a cusp form)
        B.add("G9", "G9.K9", g8_hida ? "absent(not_applicable)" : "absent(H1)", "cert_absent");
    }
    if (eisup_on) {   // (HZ-EISUP, design Sec. 3.4) the certificate of the Eisenstein U_p systems (cert_sha256 only; none of it depends on threads)
        B.add("G9", "G9.eisup.e_theta", "[" + Join(EU.e_theta) + "]");
        B.add("G9", "G9.eisup.primes", std::to_string(EU.primes_used), "int");
        B.add("G9", "G9.eisup.skipped", Join(EU.skipped));
        B.add("G9", "G9.eisup.id_prime", std::to_string(EU.id_prime), "int");
        B.add("G9", "G9.eisup.digest", Hex64(EU.digest()), "hex64");
        if (cusp_irred_q) B.add("G9", "G9.irreducible.cusp.q", std::to_string(cusp_irred_q), "int");   // (d)'s certificate of c_theta
    }
    if (eisupb_on) {   // (EISUPb, design Sec. 5.3) the certificate of the Eisenstein U_p systems with characters (none of it depends on threads)
        B.add("G9", "G9.eisupb.s_N", std::to_string(EB.s_N), "int");
        B.add("G9", "G9.eisupb.o_N", std::to_string(EB.o_N), "int");
        B.add("G9", "G9.eisupb.P_eis", "[" + Join(EB.P_eis) + "]");
        B.add("G9", "G9.eisupb.e_theta", "[" + Join(EB.e_theta) + "]");
        B.add("G9", "G9.eisupb.primes", std::to_string(EB.primes_used), "int");
        B.add("G9", "G9.eisupb.skipped", Join(EB.skipped));
        B.add("G9", "G9.eisupb.id_prime", std::to_string(EB.id_prime), "int");
        B.add("G9", "G9.eisupb.id_prime_extra", std::to_string(EB.id_prime_extra), "int");
        B.add("G9", "G9.eisupb.S", Join(EB.S));
        B.add("G9", "G9.eisupb.digest", Hex64(EB.digest()), "hex64");
        if (cusp_irred_q) B.add("G9", "G9.irreducible.cusp.q", std::to_string(cusp_irred_q), "int");
    }
    B.add("G9", "G9.K10", N > 1 && IsPrimeU32(N) && n == 0 ? "absent(not_built)" : "absent(not_applicable)", "cert_absent");
    B.add("G9", "G9.trust.tb_chi", R.descent.record_fnv1a64, "hex64");

    // =================================================================================================================================
    // G10 provenance
    for (const auto& [ell, t] : g8_prov) B.add("G10", "G10.hida.ell=" + std::to_string(ell), t);   // (H1) heckez.G8 = hida only
    if (full_N) {   // (Z1; audit C6) the contract text the full record was computed under, and the full step's provenance
        B.add("G10", "G10.contract", up::kFullContract);   // (d')(vii)
        B.add("G10", "G10.full.rat.primes_used", std::to_string(THF ? THF->primes_used : 0));
        if (eisup_on) {   // (HZ-EISUP) provenance; `reused` depends on the thread count (the step's look-ahead), so it is G10, not G9
            std::ostringstream t;
            t << EU.secs;
            B.add("G10", "G10.eisup.secs", t.str());
            B.add("G10", "G10.eisup.reused", std::to_string(EU.reused));
            B.add("G10", "G10.eisup.computed", std::to_string(EU.computed));
        }
        if (eisupb_on) {   // (EISUPb) provenance
            std::ostringstream t;
            t << EB.secs;
            B.add("G10", "G10.eisupb.secs", t.str());
            B.add("G10", "G10.eisupb.reused", std::to_string(EB.reused));
            B.add("G10", "G10.eisupb.computed", std::to_string(EB.computed));
        }
        for (const auto& [ell, e] : S) {
            if (!e.exw.empty()) B.add("G10", "G10.read_full.ell=" + std::to_string(ell) + ".exw", e.exw);   // (slot-27 EXW) not hashed
            if (ell < 5 || !c1f_ok) continue;
            const std::string pre = "G10.read_full.ell=" + std::to_string(ell);
            B.add("G10", pre + ".source", e.out.read ? e.out.L.source : e.out.M.source);
            B.add("G10", pre + ".K", std::to_string(e.out.read ? e.out.L.K : e.out.M.K));
            B.add("G10", pre + ".module_status", e.out.read ? "ok" : e.out.M.status + " (" + e.out.M.detail + ")");
            if (!e.out.escalation.empty()) B.add("G10", pre + ".escalation", e.out.escalation);
            if (e.c3_from_fallback)
                B.add("G10", pre + ".c3_fallback_K1", e.c3out.read ? "read" : e.c3out.M.status + " (" + e.c3out.M.detail + ")");
            std::ostringstream t;
            t << e.secs;
            B.add("G10", pre + ".secs", t.str());
        }
    }
    for (const auto& [ell, e] : S) {
        const std::string pre = "G10.read.ell=" + std::to_string(ell);
        if (full_N || !c1_ok) continue;
        B.add("G10", pre + ".K0", std::to_string(e.K0));
        B.add("G10", pre + ".source", e.out.read ? e.out.L.source : e.out.M.source);
        B.add("G10", pre + ".K", std::to_string(e.out.read ? e.out.L.K : e.out.M.K));
        B.add("G10", pre + ".word", e.out.read ? e.out.L.word : e.out.M.word);
        B.add("G10", pre + ".regime", e.out.read ? e.out.L.regime : e.out.M.regime);
        B.add("G10", pre + ".module_status", e.out.read ? "ok" : e.out.M.status + " (" + e.out.M.detail + ")");
        if (!e.out.escalation.empty()) B.add("G10", pre + ".escalation", e.out.escalation);
        if (e.c3_from_fallback)
            B.add("G10", pre + ".c3_fallback_K1", e.c3out.read ? "read" : e.c3out.M.status + " (" + e.c3out.M.detail + ")");
        if (!e.exw.empty()) B.add("G10", pre + ".exw", e.exw);   // (slot-27 EXW) the trigger of rule (alpha); not hashed
        std::ostringstream t;
        t << e.secs;
        B.add("G10", pre + ".secs", t.str());
    }
    for (const auto& [ell, t] : TR) {
        std::ostringstream o;
        o << "ReadTor status=" << t.status << " e=" << t.e << " K=" << t.K << " K_J=" << t.K_J << " t_ell=" << t.t_ell << " secs=" << t.secs;
        B.add("G10", "G10.tor.ell=" + std::to_string(ell), o.str());
    }
    B.add("G10", "G10.rat.primes_used", std::to_string(TH ? TH->primes_used : 0));
    B.add("G10", "G10.rat.settings", Q.settings);
    B.add("G10", "G10.rat.word_primes", "CrtPrimes(<2^30,sized_from_bound,cap=" +
                                            (rs.word_prime_cap == (1u << 21) ? std::string("2^21") : std::to_string(rs.word_prime_cap)) +
                                            ")");   // (F11) provenance, out of the hashed settings text
    for (const auto& [ell, e] : S)
        if (e.v_disc == 1) B.add("G10", "G10.S_cand.ell=" + std::to_string(ell) + ".note", "v_disc=1 kept in S (S_rule; LC-3b not applied in v1)");
    if (irred_tried) {
        std::ostringstream t;
        t << "tried=" << irred_tried << " q=" << cusp_irred_q << " secs=" << irred_secs;
        B.add("G10", "G10.irreducible.cusp", t.str());
    }
    B.add("G10", "G10.threads", std::to_string(s.threads));
    R.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    R.rss_peak = static_cast<uint64_t>(oc_util::PeakRssMiB());
    R.rehash();
    return R;
}

// ---- the v1 schema of the hashed groups (re-audit R5 (b)) ------------------------------------------------------------------------
// The vocabulary and the parent of every hashed field follow from its path: a cell file's own vocabulary / parent columns are checked
// against this table, never trusted.  Paths are normalised (ell=<n>, q=<n|hex>, pt[<i>], m[<i>], [U_<p>(:<a,b>)] -> #, *); a parent
// "$" stands for "G2.ell=<the path's ell>.generation.status".  A field may have two shapes (a value or not_run, a value or a status).
namespace {
struct Shape {
    const char* vocab;
    const char* parent;
};
std::string NormalisePath(const std::string& p) {
    std::string o;
    size_t i = 0;
    auto starts = [&](const char* t) { return p.compare(i, std::char_traits<char>::length(t), t) == 0; };
    while (i < p.size()) {
        if (starts("ell=") || starts("q=")) {
            const size_t L = starts("ell=") ? 4 : 2;
            o += p.substr(i, L);
            i += L;
            const size_t b = i;
            while (i < p.size() && (std::isxdigit(static_cast<unsigned char>(p[i])) || p[i] == 'x')) ++i;
            o += i > b ? "#" : "";
        } else if (starts("pt[") || starts("m[") || starts("[U_")) {
            const size_t L = starts("pt[") ? 3 : (starts("m[") ? 2 : 3);
            const bool u = starts("[U_");
            o += p.substr(i, L);
            i += L;
            const size_t b = i;
            while (i < p.size() && std::isdigit(static_cast<unsigned char>(p[i]))) ++i;
            o += i > b ? "#" : "";
            if (u && i < p.size() && p[i] == ':') {
                o += ":*";
                while (i < p.size() && p[i] != ']') ++i;
            }
        } else {
            o += p[i++];
        }
    }
    return o;
}
const std::map<std::string, std::vector<Shape>>& SchemaTable() {
    static const std::map<std::string, std::vector<Shape>> t = {
        {"G0.schema", {{"", ""}}}, {"G0.coefficient_ring", {{"", ""}}}, {"G0.object", {{"", ""}}}, {"G0.conventions", {{"", ""}}},
        {"G0.N", {{"int", ""}}}, {"G0.n", {{"int", ""}}}, {"G0.algebra", {{"algebra", ""}}}, {"G0.operators", {{"", ""}}},
        {"G0.descent_ref.record_fnv1a64", {{"hex64", ""}}}, {"G0.status", {{"overall", ""}}},
        {"G1.b1", {{"int", ""}}}, {"G1.e_k", {{"int", ""}}}, {"G1.d", {{"int", ""}}}, {"G1.d_source", {{"d_source", ""}}},
        {"G1.reduced_Q", {{"reduced_Q", ""}}}, {"G1.C1.status", {{"C1", ""}}}, {"G1.C1.deg_f", {{"int", "G1.C1.status"}}},
        {"G1.P", {{"", ""}}}, {"G1.theta", {{"", "G1.C1.status"}, {"status", ""}}},
        {"G1.theta_ops", {{"", ""}}},   // (slot-26 Z1, audit C2) full level-N records only: theta's index list
        {"G1.components[0].type", {{"component_type", ""}}}, {"G1.components[0].degree", {{"int", ""}}},
        {"G1.components[0].mult", {{"int", ""}}}, {"G1.components[0].p0", {{"op_or_theta", ""}}},
        {"G1.components[0].minpoly", {{"", ""}}}, {"G1.components[0].lambda_theta", {{"zbig", ""}}},
        {"G1.components[1].type", {{"component_type", ""}}}, {"G1.components[1].degree", {{"int", ""}}},
        {"G1.components[1].mult", {{"int", ""}, {"not_run", ""}}}, {"G1.components[1].p0", {{"op_or_theta", ""}}},
        {"G1.components[1].minpoly", {{"", ""}}}, {"G1.components[1].irreducible", {{"irreducible", ""}}},
        {"G1.components", {{"not_run", ""}, {"status", ""}}},
        {"G1.S_cand.ell=#.reason", {{"list:S_reason", ""}}}, {"G1.S_cand.ell=#.v_disc", {{"int", ""}}},
        {"G1.disc.digest", {{"hex64", ""}}}, {"G1.disc.bits", {{"int", ""}}}, {"G1.disc.gcd.factor.q=#.e", {{"int", ""}}},
        {"G1.disc.gcd.factor.q=#.prime_proof", {{"prime_proof", ""}}}, {"G1.disc.gcd.cofactor", {{"zbig", ""}}},
        {"G1.disc.gcd.cofactor_test", {{"cofactor_test", ""}}}, {"G1.C2_status", {{"C2", ""}}}, {"G1.discriminants_used", {{"", ""}}},
        {"G2.ell=#.theory", {{"theory", ""}}}, {"G2.ell=#.generation.status", {{"generation", ""}}},
        {"G2.ell=#.generation.route", {{"route", "$"}}}, {"G2.ell=#.generation.dim_A1", {{"int", ""}}}, {"G2.ell=#.t_ell", {{"t_ell", ""}}},
        {"G2.ell=#.maximal_ideals", {{"mi_status", "$"}}}, {"G2.ell=#.A1_of_computed.dim_A1", {{"int", ""}}},
        {"G2.ell=#.decomposed_by", {{"", ""}}}, {"G2.ell=#.m[#].residue_degree", {{"int", ""}}}, {"G2.ell=#.m[#].residue_poly", {{"", ""}}},
        {"G2.ell=#.m[#].support", {{"support", ""}}}, {"G2.ell=#.m[#].degree_attr", {{"degree_attr", ""}}},
        {"G2.ell=#.m[#].local_type.kind", {{"local_kind", "$"}}}, {"G2.ell=#.m[#].local_type.rank", {{"int", "$"}}},
        {"G2.ell=#.m[#].local_type.embdim", {{"int", "$"}}}, {"G2.ell=#.m[#].local_type.gorenstein", {{"bool", "$"}}},
        {"G2.ell=#.m[#].local_type.length_mod_l", {{"int", "$"}}}, {"G2.ell=#.m[#].local_type.hilbert", {{"", "$"}}},
        {"G2.ell=#.m[#].local_type.loewy", {{"int", "$"}}}, {"G2.ell=#.m[#].components_through.eis", {{"through", "$"}}},
        {"G2.ell=#.m[#].components_through.cusp", {{"through", "$"}}},
        {"G2.ell=#.m[#].pair_kinds[eis,cusp].depth", {{"int", "$"}, {"not_run", "$"}}},
        {"G2.ell=#.m[#].pair_kinds[eis,cusp].status", {{"status", "$"}}}, {"G2.ell=#.m[#].pair_kinds[eis,cusp].kind", {{"pair_kind", "$"}}},
        {"G3.partition", {{"partition", ""}}}, {"G3.C_T[U_#:*].upper", {{"int", ""}}}, {"G3.C_T[U_#:*].status", {{"utag", ""}}},
        {"G3.C_T[U_#].status", {{"utag", ""}}}, {"G3.C_T.status", {{"ct_status", ""}}}, {"G3.C_T.lower", {{"status", ""}}},
        {"G3.fibre_product", {{"fibre_product", ""}}}, {"G3.general_case", {{"general_case", ""}}}, {"G3.S", {{"list:int", ""}}},
        {"G3.S_prime", {{"list:int", ""}}}, {"G3.C_T.ell=#.upper_v", {{"int", ""}}}, {"G3.C_T.ell=#.read", {{"read_word", ""}}},
        {"G3.C_T.ell=#.status", {{"ct_ell", ""}}}, {"G3.C_T.ell=#.lower_v", {{"int", ""}}}, {"G3.eta_L.ell=#.exponents", {{"", ""}}},
        {"G3.eta_L.ell=#.r", {{"int", ""}}}, {"G3.eta_L.ell=#.sign", {{"sign", ""}}}, {"G3.C_T.upper", {{"", ""}}},
        {"G3.C_T.cofactor", {{"zbig", ""}}}, {"G3.C_T.cofactor_test", {{"cofactor_test", ""}}}, {"G3.C_T.exact", {{"zbig", "G3.C_T.status"}}},
        {"G3.second_route.ell=#.v_Bk", {{"int", ""}, {"not_run", ""}}}, {"G3.second_route.ell=#.agrees", {{"agrees", ""}}},
        {"G3.second_route.ell=#.mazur_v_n", {{"int", ""}}},
        {"G4.ell=#.J.status", {{"J", ""}}}, {"G4.ell=#.pt[#].J.status", {{"J", ""}}}, {"G4.ell=#.pt[#].J.lower", {{"int", ""}}},
        {"G4.ell=#.pt[#].J.upper", {{"int", ""}}},
        {"G5.q=#.ell=#.relation_to_free", {{"relation_to_free", ""}}}, {"G5.q=#.ell=#.generation_tors", {{"generation_tors", ""}}},
        {"G5.q=#.ell=#.pt[#].system", {{"", ""}}}, {"G5.q=#.ell=#.pt[#].relation_to_free", {{"relation_to_free", ""}}},
        {"G5.q=#.ell=#.pt[#].route", {{"coincidence_route", ""}}}, {"G5.q=#.ell=#.pt[#].witness", {{"op_or_joint", ""}}},
        {"G5.q=#.ell=#.pt[#].generation_tors", {{"generation_tors", ""}}},
        {"G6.q=#.ell=#.split", {{"split", ""}}}, {"G6.q=#.ell=#.hom_order", {{"status", ""}}}, {"G6.q=#.ell=#.pt[#].split", {{"split", ""}}},
        {"G6.q=#.ell=#.pt[#].hom_order", {{"int", ""}}}, {"G6.q=#.ell=#.tor_ref", {{"", ""}}},
        {"G7.pair", {{"not_run", ""}}}, {"G7.theory", {{"theory", ""}}}, {"G7.hecke", {{"not_run", ""}}},
        // (slot-25 H1) G8, only under heckez.G8 = hida (ScanRecord rejects every G8 field under the default `off`)
        {"G8.scope", {{"g8_scope", ""}}}, {"G8.ell=#.blocks", {{"g8_blocks", ""}}}, {"G8.ell=#.m[#].ordinary", {{"bool", ""}}},
        {"G8.ell=#.m[#].chain_key", {{"chain_key", ""}}}, {"G8.ell=#.m[#].g", {{"int", ""}}},
        {"G8.ell=#.m[#].L_k", {{"int", ""}, {"not_run", ""}}}, {"G8.ell=#.m[#].v_Bk", {{"int", ""}, {"not_run", ""}}},
        {"G8.ell=#.m[#].structure", {{"g8_structure", ""}}}, {"G8.ell=#.m[#].I0_generator", {{"i0_generator", ""}}},
    };
    return t;
}
// ---- (slot-26 HZ-EISUP, amendment A4; design Sec. 3.6, audit M2) the full layout at N > 1 -----------------------------------------------
// NormaliseFullPath = NormalisePath followed by four rewrites, each of a digit run in a fixed context: components[<d>] -> components[#];
// in a path starting G1.components[#]., a final segment U_<d> -> U_#; a final components_through.<d> -> components_through.#;
// pair_kinds[<d>,<d>] -> pair_kinds[#,#].  The literal components_through.eis / .cusp and pair_kinds[eis,cusp] have no such digit run.
// NormalisePath is unchanged (byte-identical), so every path that normalised before normalises identically; the second table is reached
// only on a full record at N > 1 carrying G1.components = certified (ScanRecord).
bool AllDigits(const std::string& s, size_t a, size_t b) {
    if (a >= b) return false;
    for (size_t i = a; i < b; ++i)
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    return true;
}
std::string NormaliseFullPath(const std::string& p0) {
    std::string p = NormalisePath(p0);
    {   // components[<digits>] -> components[#]
        const std::string t = "components[";
        std::string o;
        size_t i = 0;
        while (i < p.size()) {
            if (p.compare(i, t.size(), t) == 0) {
                const size_t a = i + t.size();
                size_t b = a;
                while (b < p.size() && std::isdigit(static_cast<unsigned char>(p[b]))) ++b;
                if (b > a && b < p.size() && p[b] == ']') {
                    o += t + "#";
                    i = b;
                    continue;
                }
            }
            o += p[i++];
        }
        p = o;
    }
    {   // G1.components[#].U_<digits> (final segment) -> U_#
        const std::string pre = "G1.components[#].U_";
        if (p.compare(0, pre.size(), pre) == 0 && AllDigits(p, pre.size(), p.size())) p = pre + "#";
    }
    {   // (slot-27 EISUPb, amendment A5) G1.components[#].U_<digits>.charpoly -> G1.components[#].U_#.charpoly
        const std::string pre = "G1.components[#].U_", suf = ".charpoly";
        if (p.compare(0, pre.size(), pre) == 0 && p.size() > pre.size() + suf.size() &&
            p.compare(p.size() - suf.size(), suf.size(), suf) == 0 && AllDigits(p, pre.size(), p.size() - suf.size()))
            p = pre + "#" + suf;
    }
    {   // a final components_through.<digits> -> components_through.#
        const std::string t = ".components_through.";
        const size_t a = p.rfind(t);
        if (a != std::string::npos && AllDigits(p, a + t.size(), p.size())) p = p.substr(0, a + t.size()) + "#";
    }
    {   // pair_kinds[<digits>,<digits>] -> pair_kinds[#,#]
        const std::string t = "pair_kinds[";
        const size_t a = p.find(t);
        if (a != std::string::npos) {
            const size_t c = p.find(',', a);
            const size_t e = p.find(']', a);
            if (c != std::string::npos && e != std::string::npos && c < e && AllDigits(p, a + t.size(), c) && AllDigits(p, c + 1, e))
                p = p.substr(0, a + t.size()) + "#,#" + p.substr(e);
        }
    }
    return p;
}
const std::map<std::string, std::vector<Shape>>& FullSchemaTable() {
    static const std::map<std::string, std::vector<Shape>> t = {
        {"G1.components[#].type", {{"component_type", ""}}}, {"G1.components[#].degree", {{"int", ""}}},
        {"G1.components[#].mult", {{"int", ""}}}, {"G1.components[#].p0", {{"op_or_theta", ""}}},
        {"G1.components[#].minpoly", {{"", ""}}}, {"G1.components[#].lambda_theta", {{"zbig", ""}}},
        {"G1.components[#].irreducible", {{"irreducible", ""}}}, {"G1.components[#].U_#", {{"zbig", ""}}},
        // (slot-27 EISUPb, amendment A5) reached only on full records at N > 1 with G1.components = certified and e^an_k(N) != 1
        {"G1.components[#].psi_cond", {{"int", ""}}}, {"G1.components[#].psi_order", {{"int", ""}}},
        {"G1.components[#].theta_poly", {{"", ""}}}, {"G1.components[#].U_#.charpoly", {{"", ""}}},
        {"G2.ell=#.m[#].components_through.#", {{"through", "$"}}},
        {"G2.ell=#.m[#].pair_kinds[#,#].depth", {{"int", "$"}, {"not_run", "$"}}},
        {"G2.ell=#.m[#].pair_kinds[#,#].status", {{"status", "$"}}},
        {"G2.ell=#.m[#].pair_kinds[#,#].kind", {{"pair_kind", "$"}}},
    };
    return t;
}
// (A4 item 4) the full layout of a full record at N > 1 with G1.components = certified; one problem text per violation
std::vector<std::string> FullLayoutProblems(const HeckeStructureRecord& R, const std::map<std::string, std::string>& val) {
    std::vector<std::string> out;
    std::map<size_t, std::map<std::string, std::string>> comp;   // index -> key -> value
    std::set<size_t> thr_idx;
    std::vector<std::pair<size_t, size_t>> pairs;
    for (const Field& f : R.fields) {
        const std::string& p = f.path;
        const std::string c0 = "G1.components[";
        if (p.compare(0, c0.size(), c0) == 0) {
            const size_t b = p.find(']', c0.size());
            if (b == std::string::npos || !AllDigits(p, c0.size(), b) || b + 1 >= p.size() || p[b + 1] != '.') {
                out.push_back("a malformed component path: " + p);
                continue;
            }
            comp[std::stoul(p.substr(c0.size(), b - c0.size()))][p.substr(b + 2)] = f.value;
            continue;
        }
        const std::string t = ".components_through.";
        const size_t a = p.rfind(t);
        if (f.group == "G2" && a != std::string::npos && AllDigits(p, a + t.size(), p.size())) thr_idx.insert(std::stoul(p.substr(a + t.size())));
        const std::string pk = "pair_kinds[";
        const size_t k = p.find(pk);
        if (f.group == "G2" && k != std::string::npos) {
            const size_t c = p.find(',', k), e = p.find(']', k);
            if (c != std::string::npos && e != std::string::npos && AllDigits(p, k + pk.size(), c) && AllDigits(p, c + 1, e))
                pairs.push_back({std::stoul(p.substr(k + pk.size(), c - k - pk.size())), std::stoul(p.substr(c + 1, e - c - 1))});
        }
    }
    auto num = [&](const char* key) -> int64_t {
        const auto it = val.find(key);
        return it != val.end() && IsDecimal(it->second) ? std::stoll(it->second) : -1;
    };
    const int64_t ek = num("G1.e_k"), d = num("G1.d");
    std::vector<uint32_t> ps;
    for (uint32_t x = R.N, q = 2; x > 1; ++q) {
        if (static_cast<uint64_t>(q) * q > x) q = x;
        if (x % q == 0) {
            ps.push_back(q);
            while (x % q == 0) x /= q;
        }
    }
    size_t expect = 0;
    for (const auto& [j, kv] : comp) {
        if (j != expect) out.push_back("component indices not contiguous from 0 at " + std::to_string(j));
        expect = j + 1;
    }
    int64_t n_eis = 0, n_cusp = 0, sum_deg = 0;
    // (slot-27 EISUPb, amendment A5) e^an_k(N) != 1: the degrees of the eis rows sum to e_k; degree-1 rows carry U_<p>, degree >= 2 rows
    // U_<p>.charpoly, for exactly the p | N; psi_cond | s_N, psi_order | o_N, phi(psi_order) = degree.  e^an_k(N) = 1: unchanged.
    const bool bscope = R.N > 1 && eisup::InScopeChar(R.N, R.n);
    uint32_t s_N = 1, o_N = 1;
    if (bscope)
        for (uint32_t p : ps)
            if (p % 2 == 1 && static_cast<uint64_t>(R.N) % (static_cast<uint64_t>(p) * p) == 0) {   // (audit m3) 64-bit
                s_N *= p;
                o_N = o_N / std::gcd(o_N, p - 1) * (p - 1);
            }
    for (const auto& [j, kv] : comp) {
        const auto it = kv.find("type");
        const std::string ty = it == kv.end() ? std::string() : it->second;
        if (ty == "eis") {
            ++n_eis;
            std::set<std::string> want;
            for (uint32_t p : ps) want.insert(std::to_string(p));
            if (!bscope) {
                if (static_cast<int64_t>(j) >= ek) out.push_back("an Eisenstein component beyond e_k: [" + std::to_string(j) + "]");
                std::set<std::string> us;
                for (const auto& [key, v] : kv)
                    if (key.compare(0, 2, "U_") == 0) us.insert(key.substr(2));
                if (us != want) out.push_back("Eisenstein component [" + std::to_string(j) + "] does not carry U_<p> for exactly the p | N");
            } else {
                const std::string at = "Eisenstein component [" + std::to_string(j) + "]";
                auto num_of = [&](const char* key) -> int64_t {
                    const auto f = kv.find(key);
                    return f != kv.end() && IsDecimal(f->second) ? std::stoll(f->second) : -1;
                };
                const int64_t deg = num_of("degree"), pc = num_of("psi_cond"), po = num_of("psi_order");
                if (!kv.count("psi_cond") && !kv.count("psi_order"))   // an (a) row (A4) where e^an_k(N) != 1: (a)'s scope message stands
                    out.push_back("the layout outside HZ-EISUP's scope (e^an_k(N) != 1): " + at + " is an HZ-EISUP row without psi_cond / psi_order");
                if (deg < 1) {
                    out.push_back(at + " without a degree");
                    continue;
                }
                sum_deg += deg;
                std::set<std::string> us, uc;
                for (const auto& [key, v] : kv) {
                    if (key.compare(0, 2, "U_") != 0) continue;
                    const size_t dot = key.find('.');
                    if (dot == std::string::npos) us.insert(key.substr(2));
                    else if (key.substr(dot) == ".charpoly") uc.insert(key.substr(2, dot - 2));
                    else out.push_back(at + " carries an unknown U_ key " + key);
                }
                if (deg == 1) {
                    if (us != want || !uc.empty() || !kv.count("lambda_theta") || kv.count("theta_poly"))
                        out.push_back(at + " of degree 1 does not carry lambda_theta and U_<p> for exactly the p | N");
                } else {
                    if (uc != want || !us.empty() || kv.count("lambda_theta") || !kv.count("theta_poly"))
                        out.push_back(at + " of degree >= 2 does not carry theta_poly and U_<p>.charpoly for exactly the p | N");
                }
                int64_t phi = 0;
                for (int64_t u = 1; po >= 1 && u <= po; ++u) phi += std::gcd(u, po) == 1;
                if (pc < 1 || po < 1 || s_N % pc != 0 || o_N % po != 0 || phi != deg)
                    out.push_back(at + ": psi_cond | s_N, psi_order | o_N, phi(psi_order) = degree violated");
            }
        } else if (ty == "cusp") {
            ++n_cusp;
            if (!bscope && static_cast<int64_t>(j) != ek) out.push_back("the cuspidal block is not at index e_k: [" + std::to_string(j) + "]");
            if (bscope && static_cast<int64_t>(j) != n_eis) out.push_back("the cuspidal block is not after the Eisenstein rows: [" + std::to_string(j) + "]");
            for (const auto& [key, v] : kv)
                if (key.compare(0, 2, "U_") == 0) out.push_back("a U_<p> key on the cuspidal block: [" + std::to_string(j) + "]." + key);
        } else {
            out.push_back("a component without a type: [" + std::to_string(j) + "]");
        }
    }
    if (!bscope && n_eis != ek) out.push_back("Eisenstein components " + std::to_string(n_eis) + " != e_k = " + std::to_string(ek));
    if (bscope && sum_deg != ek) out.push_back("the degrees of the Eisenstein rows sum to " + std::to_string(sum_deg) + " != e_k = " + std::to_string(ek));
    if (n_cusp != (d > ek ? 1 : 0)) out.push_back("cuspidal blocks " + std::to_string(n_cusp) + ", expected " + (d > ek ? "1" : "0"));
    for (size_t j : thr_idx)
        if (!comp.count(j)) out.push_back("components_through." + std::to_string(j) + " names no component");
    for (const auto& [a, b] : pairs)
        if (!(a < b) || !comp.count(a) || !comp.count(b))
            out.push_back("pair_kinds[" + std::to_string(a) + "," + std::to_string(b) + "] is not a pair j < j' of components");
    // (code audit m4) the scope of (d')(ii)'s paragraph: e^an_k(N) = 1 and (C1) certified
    if (!eisup::InScope(R.N, R.n) && !bscope) out.push_back("the layout outside HZ-EISUP's scope (e^an_k(N) != 1)");
    {
        const auto it = val.find("G1.C1.status");
        if (it == val.end() || it->second != "certified") out.push_back("the layout without G1.C1.status = certified");
    }
    // (code audit m4) per maximal ideal m[i] of an ell with maximal_ideals = certified: components_through.<j> for every component j,
    // no key components_through.eis / .cusp or pair_kinds[eis,cusp] (those of level-1 and anemic records), and pair_kinds[<j>,<j'>] only
    // for j < j' both passing (components_through = certified); no m[i] row under any other ell
    struct MRow {
        std::map<size_t, std::string> thr;
        std::set<std::pair<size_t, size_t>> pk;
        bool legacy = false;
    };
    std::map<std::string, std::map<size_t, MRow>> mrows;   // ell -> i -> row
    for (const Field& f : R.fields) {
        const std::string& p = f.path;
        if (f.group != "G2" || p.compare(0, 7, "G2.ell=") != 0) continue;
        const size_t e = p.find('.', 7);
        if (e == std::string::npos || p.compare(e, 3, ".m[") != 0) continue;
        const size_t rb = p.find(']', e + 3);
        if (rb == std::string::npos || !AllDigits(p, e + 3, rb) || rb + 1 >= p.size() || p[rb + 1] != '.') continue;
        MRow& M = mrows[p.substr(7, e - 7)][std::stoul(p.substr(e + 3, rb - e - 3))];
        const std::string rest = p.substr(rb + 2);
        const std::string ct = "components_through.", pk = "pair_kinds[";
        if (rest.compare(0, ct.size(), ct) == 0) {
            if (AllDigits(rest, ct.size(), rest.size())) M.thr[std::stoul(rest.substr(ct.size()))] = f.value;
            else M.legacy = true;
        } else if (rest.compare(0, pk.size(), pk) == 0) {
            const size_t c = rest.find(','), b = rest.find(']');
            if (c != std::string::npos && b != std::string::npos && c < b && AllDigits(rest, pk.size(), c) && AllDigits(rest, c + 1, b))
                M.pk.insert({std::stoul(rest.substr(pk.size(), c - pk.size())), std::stoul(rest.substr(c + 1, b - c - 1))});
            else
                M.legacy = true;
        }
    }
    for (const auto& [ell, rows] : mrows) {
        const auto mi = val.find("G2.ell=" + ell + ".maximal_ideals");
        const bool cert = mi != val.end() && mi->second == "certified";
        for (const auto& [i, M] : rows) {
            const std::string at = "G2.ell=" + ell + ".m[" + std::to_string(i) + "]";
            if (!cert) {
                out.push_back(at + " under maximal_ideals != certified");
                continue;
            }
            if (M.legacy) out.push_back(at + " carries a level-1 / anemic attribution key (eis / cusp)");
            for (const auto& [j, kv] : comp)
                if (!M.thr.count(j)) out.push_back(at + " lacks components_through." + std::to_string(j));
            for (const auto& [a, b] : M.pk) {
                const auto ta = M.thr.find(a), tb = M.thr.find(b);
                if (ta == M.thr.end() || tb == M.thr.end() || ta->second != "certified" || tb->second != "certified")
                    out.push_back(at + ".pair_kinds[" + std::to_string(a) + "," + std::to_string(b) + "] for a pair not both passing");
            }
        }
    }
    return out;
}
std::string EllOfPath(const std::string& p) {
    const size_t a = p.find("ell=");
    if (a == std::string::npos) return std::string();
    size_t b = a + 4;
    while (b < p.size() && std::isdigit(static_cast<unsigned char>(p[b]))) ++b;
    return p.substr(a + 4, b - a - 4);
}
}  // namespace

// ---- scans ------------------------------------------------------------------------------------------------------------------------
ScanResult ScanRecord(const HeckeStructureRecord& R) {
    ScanResult S;
    std::map<std::string, std::string> val;
    for (const Field& f : R.fields) val[f.path] = f.value;
    bool pend = false;
    const bool g8_off = R.settings_text.find("\nheckez.G8 = off ") != std::string::npos ||
                        R.settings_text.compare(0, 14, "heckez.G8 = off") == 0;
    const bool g8_hida = !g8_off && R.settings_text.find("\nheckez.G8 = hida ") != std::string::npos;
    // (HZ-EISUP, A4 item 3) the second table is reached only on a full record at N > 1 carrying G1.components = certified
    const auto comp_it = val.find("G1.components");
    const bool full_layout = R.algebra == "full" && R.N > 1 && comp_it != val.end() && comp_it->second == "certified";
    if (full_layout)
        for (const std::string& pr : FullLayoutProblems(R, val)) {
            ++S.values_outside_lists;
            S.problems.push_back("the full layout (A4): " + pr);
        }
    for (const Field& f : R.fields) {
        // (re-audit R5 (b)) the schema of the path decides the vocabulary and the parent; the file's columns must agree with it
        if (Hashed(f.group) || f.group == "G0x") {
            const std::string pre = f.path.substr(0, f.path.find('.'));
            const bool group_ok = f.group == "G0x" ? f.path == "G0.descent_ref.primes" : pre == f.group;
            if (!group_ok) {
                ++S.values_outside_lists;
                S.problems.push_back("a field outside its group: " + f.group + " " + f.path);
            }
            if (f.group == "G8" && !g8_hida) {   // (re-audit R5 (c)) v1 writes no G8 field under heckez.G8 = off (the default); (H1)
                                                 // only the non-default `hida` writes G8, checked by the schema table below
                ++S.values_outside_lists;
                S.problems.push_back(std::string("a G8 field in a record whose settings say ") + (g8_off ? "heckez.G8 = off" : "another G8 value") +
                                     ": " + f.path);
            } else if (Hashed(f.group)) {
                auto it = SchemaTable().find(NormalisePath(f.path));
                bool known = it != SchemaTable().end();
                if (!known && full_layout) {   // (HZ-EISUP, A4) only after the v1 table misses
                    it = FullSchemaTable().find(NormaliseFullPath(f.path));
                    known = it != FullSchemaTable().end();
                }
                if (!known) {
                    ++S.values_outside_lists;
                    S.problems.push_back("a field the v1 schema does not have: " + f.path + "=" + f.value);
                } else {
                    bool match = false;
                    for (const Shape& sh : it->second) {
                        const std::string par = std::string(sh.parent) == "$" ? "G2.ell=" + EllOfPath(f.path) + ".generation.status" : sh.parent;
                        match = match || (f.vocab == sh.vocab && f.parent == par);
                    }
                    if (!match) {
                        ++S.values_outside_lists;
                        S.problems.push_back("vocabulary / parent not the schema's: " + f.path + " (vocab '" + f.vocab + "', parent '" + f.parent + "')");
                    }
                }
            }
        }
        // (re-audit R5 (c)) `split` (a statement about T, kept in the list for a later theorem) is never written by v1
        if (f.group == "G6" && f.value == "split") {
            ++S.values_outside_lists;
            S.problems.push_back("v1 never writes `split` (split(T') is split_on_P): " + f.path);
        }
        if (!ValueInVocab(f.vocab, f.value)) {
            ++S.values_outside_lists;
            S.problems.push_back("outside list: " + f.path + "=" + f.value + " (vocab " + f.vocab + ")");
        }
        if (f.group != "G9" && f.group != "G10" && f.group != "G0x" && f.vocab.empty() && ClassOfStatus(f.value) == StatusClass::Unknown) {
            // a free field must not smuggle a status-like word: '(' and ')' only in known patterns
            const bool looks = f.value.find("blocked(") != std::string::npos || f.value.find("refused(") != std::string::npos ||
                               f.value.find("pending(") != std::string::npos || f.value.find("open(") != std::string::npos ||
                               f.value.find("omitted(") != std::string::npos || f.value.find("absent(") != std::string::npos;
            if (looks) {
                ++S.values_outside_lists;
                S.problems.push_back("status word in a free field: " + f.path + "=" + f.value);
            }
            // (schema audit F10) a hashed field without a vocabulary must be one of the listed free texts (lists, polynomials,
            // systems, references); any other enumeration-like field must carry its closed list
            static const std::set<std::string> kFree = {"schema", "coefficient_ring", "object", "conventions", "operators", "P", "theta",
                                                        "minpoly", "discriminants_used", "decomposed_by", "residue_poly", "hilbert",
                                                        "upper", "exponents", "system", "tor_ref",
                                                        "theta_ops",   // (S26 Z1, audit C2)
                                                        "theta_poly", "charpoly"};   // (S27 EISUPb, amendment A5: polynomials)
            const std::string last = f.path.substr(f.path.rfind('.') + 1);
            if (Hashed(f.group) && !kFree.count(last)) {
                ++S.values_outside_lists;
                S.problems.push_back("a hashed field without a closed list: " + f.path + "=" + f.value);
            }
        }
        if (Hashed(f.group) && f.path != "G0.status" && ClassOfStatus(f.value) == StatusClass::Pending) {
            pend = true;
            ++S.pending_fields;
        }
        if (!f.parent.empty()) {
            const auto it = val.find(f.parent);
            const std::string ps = it == val.end() ? std::string("<missing>") : it->second;
            // a conditional (exact / certified / local-type) field may sit only under a certified parent
            const bool parent_cert = ps == "certified";
            const bool child_neutral = f.value == "not_run";
            if (!parent_cert && !child_neutral) {
                ++S.exact_under_bound;
                S.problems.push_back("conditional field under a non-certified parent: " + f.path + "=" + f.value + " under " + f.parent +
                                     "=" + ps);
            }
        }
    }
    if (R.status != "refused") S.overall_rule_ok = (R.status == "pending") == pend;
    return S;
}

std::string StatusSignature(const HeckeStructureRecord& R) {
    std::ostringstream o;
    auto v = [&](const std::string& p) {
        const Field* f = R.find(p);
        return f ? f->value : std::string("-");
    };
    o << "(" << R.N << "," << R.n << "," << R.algebra << ") C1=" << v("G1.C1.status") << " redQ=" << v("G1.reduced_Q")
      << " C2=" << v("G1.C2_status") << " gen=[";
    bool first = true;
    for (const Field& f : R.fields) {
        const std::string suf = ".generation.status";
        if (f.group == "G2" && f.path.size() > suf.size() && f.path.compare(f.path.size() - suf.size(), suf.size(), suf) == 0) {
            const std::string ell = f.path.substr(7, f.path.size() - 7 - suf.size());
            const Field* r = R.find("G2.ell=" + ell + ".generation.route");
            o << (first ? "" : ";") << ell << ":" << f.value << (r ? "(" + r->value + ")" : "");
            first = false;
        }
    }
    o << "] C_T=" << v("G3.C_T.status");
    if (const Field* e = R.find("G3.C_T.exact")) o << ":" << e->value;
    o << " fp=" << v("G3.fibre_product");
    // G6 (q = 1) per torsion point: ell:pt[i]=split word
    o << " G6=[";
    first = true;
    for (const Field& f : R.fields) {
        const std::string suf = ".split";
        if (f.group != "G6" || f.path.compare(0, 9, "G6.q=1.el") != 0 || f.path.size() <= suf.size() ||
            f.path.compare(f.path.size() - suf.size(), suf.size(), suf) != 0)
            continue;
        o << (first ? "" : ";") << f.path.substr(11, f.path.size() - 11 - suf.size()) << "=" << f.value;
        first = false;
    }
    o << "] status=" << R.status;
    return o.str();
}

// ---- the ed.1 join ---------------------------------------------------------------------------------------------------------------
Ed1Row LookupEd1(uint32_t N, uint32_t n, const std::string& path) {
    Ed1Row r;
    std::ifstream in(path);
    if (!in) return r;
    std::string line;
    std::vector<std::string> head;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        if (head.empty()) {
            head = f;
            continue;
        }
        auto col = [&](const std::string& name) -> std::string {
            for (size_t i = 0; i < head.size() && i < f.size(); ++i)
                if (head[i] == name) return f[i];
            return std::string();
        };
        if (col("N") != std::to_string(N) || col("n") != std::to_string(n) || col("parts") != "groups,hecke") continue;
        r.found = true;
        r.record_fnv1a64 = col("record_fnv1a64");
        r.cell_sha256 = col("cell_sha256");
        std::stringstream cs(col("charz_digests"));
        while (std::getline(cs, t, ';')) {
            const size_t c = t.find(':');
            if (c != std::string::npos) r.charz.push_back({static_cast<uint32_t>(std::stoul(t.substr(0, c))), t.substr(c + 1)});
        }
        break;
    }
    return r;
}

uint32_t JoinEd1(HeckeStructureRecord& R, const Ed1Row& row) {
    if (!row.found) {
        R.descent.cell_sha256 = "not_in_ed1";
        return 0;
    }
    uint32_t bad = row.record_fnv1a64 == R.descent.record_fnv1a64 ? 0 : 1;
    for (const auto& [p, d] : row.charz)
        for (const auto& [q, dig] : R.descent.charpoly_digests)
            if (p == q && d != Hex64(dig)) ++bad;
    R.descent.cell_sha256 = bad ? "ed1_mismatch" : row.cell_sha256;
    return bad;
}

// ---- the campaign's cell file and its verification (schema audit F4) ----------------------------------------------------------------
std::string CellFileText(const HeckeStructureRecord& R, const std::string& campaign) {
    std::ostringstream o;
    o << "{\"schema\":\"" << kCellSchema << "\",\"id\":\"" << R.id << "\",\"campaign\":\"" << JsonEsc(campaign) << "\",\"record\":" << R.to_json()
      << ",\"seconds\":" << std::fixed << std::setprecision(3) << R.secs << ",\"rss_mib\":" << R.rss_peak << ",\"complete\":true}\n";
    return o.str();
}

namespace {
// A minimal JSON reader for the cell file (objects, arrays, strings with \" \\ \n escapes, numbers and literals as text).
struct JVal {
    char kind = 0;   // 'o' object, 'a' array, 's' string, 'l' literal (number / true / false / null)
    std::string text;
    std::vector<std::pair<std::string, JVal>> obj;
    std::vector<JVal> arr;
    const JVal* get(const std::string& k) const {
        for (const auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};
struct JReader {
    const std::string& s;
    size_t i = 0;
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
    }
    bool str(std::string& out) {
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                if (++i >= s.size()) return false;
                if (s[i] == 'n') out += '\n';
                else if (s[i] == '"' || s[i] == '\\') out += s[i];
                else return false;
            } else {
                out += s[i];
            }
            ++i;
        }
        if (i >= s.size()) return false;
        ++i;
        return true;
    }
    bool val(JVal& v, int depth = 0) {
        if (depth > 16) return false;
        ws();
        if (i >= s.size()) return false;
        if (s[i] == '{') {
            v.kind = 'o';
            ++i;
            ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            for (;;) {
                ws();
                std::string k;
                if (!str(k)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return false;
                ++i;
                JVal x;
                if (!val(x, depth + 1)) return false;
                v.obj.emplace_back(k, std::move(x));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return false;
            }
        }
        if (s[i] == '[') {
            v.kind = 'a';
            ++i;
            ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            for (;;) {
                JVal x;
                if (!val(x, depth + 1)) return false;
                v.arr.push_back(std::move(x));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return false;
            }
        }
        if (s[i] == '"') {
            v.kind = 's';
            return str(v.text);
        }
        v.kind = 'l';
        const size_t b = i;
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && s[i] != ' ' && s[i] != '\n') ++i;
        v.text = s.substr(b, i - b);
        return !v.text.empty();
    }
};
}  // namespace

bool VerifyCellText(const std::string& text, const std::string& expect_id, HeckeStructureRecord* out, std::string* why) {
    auto fail = [&](const std::string& w) {
        if (why) *why = w;
        return false;
    };
    JVal root;
    JReader rd{text};
    if (!rd.val(root) || root.kind != 'o') return fail("not a JSON object");
    auto sget = [](const JVal* o, const std::string& k) -> std::string {
        const JVal* v = o ? o->get(k) : nullptr;
        return v && (v->kind == 's' || v->kind == 'l') ? v->text : std::string("\x01");
    };
    if (sget(&root, "schema") != kCellSchema) return fail("cell schema");
    if (sget(&root, "complete") != "true") return fail("no completion trailer");
    if (sget(&root, "id") != expect_id) return fail("id differs from the planned id");
    const JVal* rec = root.get("record");
    if (!rec || rec->kind != 'o') return fail("no record");
    HeckeStructureRecord R;
    if (sget(rec, "schema") != kSchema) return fail("record schema");
    R.id = sget(rec, "id");
    R.algebra = sget(rec, "algebra");
    try {
        R.N = static_cast<uint32_t>(std::stoul(sget(rec, "N")));
        R.n = static_cast<uint32_t>(std::stoul(sget(rec, "n")));
        R.secs = std::stod(sget(&root, "seconds"));
        R.rss_peak = std::stoull(sget(&root, "rss_mib"));
    } catch (const std::exception&) {
        return fail("a numeric key");
    }
    R.refusal = sget(rec, "refusal");
    R.settings_text = sget(rec, "settings");
    R.settings_fingerprint = sget(rec, "settings_fingerprint");
    const JVal* dref = rec->get("descent_ref");
    R.descent.record_fnv1a64 = sget(dref, "record_fnv1a64");
    R.descent.cell_sha256 = sget(dref, "cell_sha256");
    for (const char* sec : {"body", "cert", "provenance"}) {
        const JVal* a = rec->get(sec);
        if (!a || a->kind != 'a') return fail(std::string("no ") + sec);
        for (const JVal& e : a->arr) {
            if (e.kind != 'a' || e.arr.size() != 5) return fail(std::string("a malformed ") + sec + " entry");
            for (const JVal& x : e.arr)
                if (x.kind != 's') return fail(std::string("a malformed ") + sec + " entry");
            R.fields.push_back(Field{e.arr[0].text, e.arr[1].text, e.arr[2].text, e.arr[3].text, e.arr[4].text});
        }
    }
    const std::string claimed_fnv = sget(rec, "record_fnv1a64"), claimed_cert = sget(rec, "cert_sha256"), claimed_st = sget(rec, "status");
    if (R.settings_fingerprint != Hex64(engine::FnvBytes(engine::kFnvOffset64, R.settings_text))) return fail("settings fingerprint");
    if (R.id != expect_id || R.id != RecordId(R.N, R.n, R.algebra, R.settings_fingerprint)) return fail("record id");
    R.rehash();   // the status from the fields, both hashes
    if (R.record_fnv1a64 != claimed_fnv) return fail("record_fnv1a64 does not recompute");
    if (R.cert_sha256 != claimed_cert) return fail("cert_sha256 does not recompute");
    if (R.status != claimed_st) return fail("status does not follow from the fields");
    // (re-audit R5 (a)) the header's descent_ref.record_fnv1a64 is the hashed body field, not a second value
    {
        const Field* d = R.find("G0.descent_ref.record_fnv1a64");
        if (!d || d->value != R.descent.record_fnv1a64) return fail("descent_ref.record_fnv1a64 differs from the hashed G0 field");
    }
    const ScanResult sc = ScanRecord(R);
    if (sc.values_outside_lists || sc.exact_under_bound || !sc.overall_rule_ok)
        return fail("scan: outside=" + std::to_string(sc.values_outside_lists) + " under=" + std::to_string(sc.exact_under_bound));
    if (out) *out = std::move(R);
    return true;
}

// ---- HZ-EMIT -------------------------------------------------------------------------------------------------------------------------
std::vector<std::string> WriteCertFiles(const HeckeStructureRecord& R, const RecordSources& src, const RecordSettings& s,
                                        const std::string& dir, uint64_t only_ell) {
    std::vector<std::string> names;
    if (R.status == "refused") return names;
    auto val = [&](const std::string& p) {
        const Field* f = R.find(p);
        return f ? f->value : std::string();
    };
    const std::string stem = "hzcert_N" + std::to_string(R.N) + "_n" + std::to_string(R.n);
    auto write = [&](const std::string& name, const std::string& text) {
        std::ofstream out(dir + "/" + name, std::ios::binary | std::ios::trunc);
        if (!(static_cast<bool>(out))) throw std::logic_error("WriteCertFiles: cannot write " + dir + "/" + name);
        out << text;
        names.push_back(name);
    };
    // K2 (the cell)
    {
        std::ostringstream o;
        o << "schema\thzcert/v1\nchecker\tK2\nrecord\t" << kSchema << "\nrecord_fnv1a64\t" << R.record_fnv1a64 << "\ncert_sha256\t"
          << R.cert_sha256 << "\nN\t" << R.N << "\nn\t" << R.n << "\nalgebra\t" << R.algebra << "\n";
        for (const Field& f : R.fields)
            if (f.path.compare(0, 3, "G1.") == 0 || f.path.compare(0, 6, "G9.K2.") == 0 || f.path == "G9.S" || f.path == "G9.ranks" ||
                f.path == "G9.trust.tb_chi" || f.path.compare(0, 8, "G9.caps.") == 0)
                o << f.path << "\t" << f.value << "\n";
        write(stem + "_K2_l0.txt", o.str());
    }
    if (R.N > 1 && R.algebra == "full") {
        std::sort(names.begin(), names.end());
        return names;
    }
    // K1 / K3 per ell (the dumps re-read through the same source: deterministic, the sha256 must equal G9's)
    std::vector<uint32_t> P;
    {
        std::stringstream ss(val("G1.P"));
        std::string t;
        while (std::getline(ss, t, ',')) P.push_back(static_cast<uint32_t>(std::stoul(t)));
    }
    for (const Field& f : R.fields) {
        const bool k1 = f.path.compare(0, 9, "G9.K1.ell") == 0 && f.path.size() > 15 && f.path.compare(f.path.size() - 15, 15, ".satcert_sha256") == 0;
        const bool k3 = f.path.compare(0, 9, "G9.K3.ell") == 0 && f.path.size() > 12 && f.path.compare(f.path.size() - 12, 12, ".dump_sha256") == 0;
        if (!k1 && !k3) continue;
        const std::string ells = f.path.substr(10, f.path.find('.', 10) - 10);
        const uint64_t ell = std::stoull(ells);
        if (only_ell && ell != only_ell) continue;
        const std::string pre = std::string(k1 ? "G9.K1" : "G9.K3") + ".ell=" + ells;
        const uint32_t K = static_cast<uint32_t>(std::stoul(val(pre + ".K")));
        local::ReadOptions o;   // the matrices do not depend on the options; only the module is used
        o.d_target = std::stoll(val("G1.d"));
        o.k_weight = R.n + 2;
        o.full_algebra = R.N == 1;
        // (slot-27 EXW, design M2) the module whose dump sha256 equals G9's: the EXPORT / level-1 route first, then the free route; the
        // sha256 is the proof of which module was written (G10's unhashed source is not read).  Neither -> a defect.
        const std::string want = f.value;
        LocalReadOut rd = src.local_read(R.N, R.n, ell, K, P, R.descent.b1, o, 0);
        bool found = rd.M.ok && Sha256Hex(DumpText(rd.M)) == want;
        if (!found && src.local_read_free) {
            o.allow_bad_primes = false;
            rd = src.local_read_free(R.N, R.n, ell, K, P, R.descent.b1, o, 0);
            found = rd.M.ok && Sha256Hex(DumpText(rd.M)) == want;
        }
        if (!found)
            throw std::logic_error("WriteCertFiles: the module at ell = " + ells + " re-reads to G9's sha256 on neither route");
        const std::string dump = DumpText(rd.M);
        std::ostringstream t;
        t << "schema\thzcert/v1\nchecker\t" << (k1 ? "K1" : "K3") << "\nrecord_fnv1a64\t" << R.record_fnv1a64 << "\nk\t" << R.n + 2 << "\n";
        for (const Field& g : R.fields)
            if (g.path.compare(0, pre.size() + 1, pre + ".") == 0) t << g.path.substr(pre.size() + 1) << "\t" << g.value << "\n";
        if (k3) {
            t << "theta\t" << val("G1.theta") << "\nlambda_theta\t" << val("G1.components[0].lambda_theta") << "\nc_theta_poly\t"
              << val("G1.components[1].minpoly") << "\nupper_v\t" << val("G3.C_T.ell=" + ells + ".upper_v") << "\n";
        }
        t << "dump_sha256_recomputed\t" << Sha256Hex(dump) << "\n" << dump;
        write(stem + "_" + (k1 ? "K1" : "K3") + "_l" + ells + ".txt", t.str());
    }
    (void)s;
    std::sort(names.begin(), names.end());
    return names;
}

}  // namespace record
}  // namespace heckez
