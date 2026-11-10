// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// HeckeZRecordValidation / HeckeZCertDumpValidation -- slot-25 HZ-REC / HZ-EMIT (see the header).

#include "HeckeZRecordValidation.h"

#include "HeckeZRecord.h"
#include "HeckeZUp.h"
#include "../HeckeExt/HeckeZRecordSource.h"
#include "../Hecke/HeckeSettings.h"
#include "../../ModArithmetic/Fnv.h"

#include <climits>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace heckez::record;

std::string SourcePath(const std::string& rel) {
#ifdef OC_SOURCE_DIR
    return std::string(OC_SOURCE_DIR) + "/" + rel;
#else
    return rel;
#endif
}

struct Tally {
    uint32_t checks = 0, failing = 0;
    void check(bool ok, const std::string& what) {
        ++checks;
        if (!ok) {
            ++failing;
            std::cout << "  [MISMATCH] " << what << "\n";
        }
    }
};

struct Cell {
    uint32_t N, n;
    const char* alg;
    const char* name;
};
const std::vector<Cell>& ToyCells() {
    static const std::vector<Cell> c = {
        {1, 10, "full", "T1"}, {1, 22, "full", "T2"}, {11, 0, "anemic", "T3"}, {2, 6, "anemic", "T4"},
        {2, 6, "full", "T4"},  {8, 10, "anemic", "T5"}, {8, 10, "full", "T5"},
    };
    return c;
}

// The status table of 04-record-census.md Sec. 2 AS BUILT (HZ-REC, 2026-11-06): EXPORT, HZ-RAT, HZ-UTAG, HZ-LOC merged; HZ-TOR not.
// Every difference from the 2026-11-04 table (written before EXPORT) is explained in HZREC_report.md Sec. 3.
// (schema audit F14) each entry pins the status signature AND the record's record_fnv1a64 (the hashed body, every field).
struct Expected {
    const char* signature;
    const char* record_fnv1a64;
};
const std::map<std::string, Expected>& ExpectedRecords() {
    static const std::map<std::string, Expected> m = {
#include "HeckeZRecordExpected.inc"
    };
    return m;
}

std::string Key(const Cell& c) { return std::to_string(c.N) + "," + std::to_string(c.n) + "," + c.alg; }

bool IsDirectory(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
std::string RealPath(const std::string& p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) == nullptr) return std::string();
    return std::string(buf);
}
bool IsInside(const std::string& inner, const std::string& outer) {
    if (outer.empty() || inner.size() < outer.size()) return false;
    if (inner.compare(0, outer.size(), outer) != 0) return false;
    return inner.size() == outer.size() || inner[outer.size()] == '/';
}
std::string ReadFile(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

}  // namespace

uint32_t HeckeZRecordValidation() {
    std::cout << "HeckeZRecordValidation: integral_hecke_structure/v1 on the toy cells ([C, sec:recordschema] item 8; 04 Sec. 2)\n";
    Tally T;
    const hecke::ScopedCoefficientRing ring(0);   // coefficient_ring = Z
    const RecordSources src = DefaultRecordSources();
    const RecordSettings S1;
    const std::string rows = SourcePath("docs/integral_census/results/ed1/rows.tsv");

    uint32_t table_ok = 0, hash_ok = 0, outside = 0, under = 0, ed1_bad = 0, golden = 0;
    std::map<std::string, HeckeStructureRecord> recs;
    uint64_t digest = engine::kFnvOffset64;
    for (const Cell& c : ToyCells()) {
        HeckeStructureRecord R = BuildHeckeStructureRecord(c.N, c.n, c.alg, S1, src);
        const uint32_t bad = JoinEd1(R, LookupEd1(c.N, c.n, rows));
        ed1_bad += bad;
        T.check(bad == 0, std::string(c.name) + " ed.1 join: descent record hash and char_Z digests agree");
        T.check(R.descent.cell_sha256.size() == 64, std::string(c.name) + " is an ed.1 cell (cell_sha256 joined)");
        const ScanResult sc = ScanRecord(R);
        outside += sc.values_outside_lists;
        under += sc.exact_under_bound;
        for (const auto& p : sc.problems) std::cout << "  [SCAN] " << c.name << " " << p << "\n";
        T.check(sc.overall_rule_ok, std::string(c.name) + " overall status rule (pending iff a pending field)");
        const std::string sig = StatusSignature(R);
        std::cout << "  RECORD " << c.name << " " << sig << "\n";
        std::cout << "  HASH " << R.id << " record_fnv1a64=" << R.record_fnv1a64 << " cert_sha256=" << R.cert_sha256.substr(0, 16)
                  << " status=" << R.status << " fields=" << R.fields.size() << " golden_eligible=" << (R.golden_eligible() ? "yes" : "no")
                  << " descent=" << R.descent.record_fnv1a64 << " secs=" << R.secs << "\n";
        const auto it = ExpectedRecords().find(Key(c));
        const bool same = it != ExpectedRecords().end() && it->second.signature == sig;
        if (!same) std::cout << "  [TABLE] " << c.name << " expected " << (it == ExpectedRecords().end() ? "<none>" : it->second.signature) << "\n";
        table_ok += same ? 1 : 0;
        T.check(same, std::string(c.name) + " status signature = the frozen table");
        const bool hsame = it != ExpectedRecords().end() && R.record_fnv1a64 == it->second.record_fnv1a64;
        if (!hsame) std::cout << "  [PIN] " << c.name << " " << c.alg << " record_fnv1a64=" << R.record_fnv1a64 << " expected "
                              << (it == ExpectedRecords().end() ? "<none>" : it->second.record_fnv1a64) << "\n";
        hash_ok += hsame ? 1 : 0;
        T.check(hsame, std::string(c.name) + " " + c.alg + " record_fnv1a64 = the pinned value (F14)");
        golden += R.golden_eligible() ? 1 : 0;
        digest = engine::FnvBytes(digest, R.record_fnv1a64 + R.cert_sha256);
        recs.emplace(Key(c), std::move(R));
    }
    const HeckeStructureRecord& T1 = recs.at("1,10,full");

    // ---- thread counts --------------------------------------------------------------------------------------------------------------
    bool threads_equal = true;
    for (uint32_t th : {4u, 16u}) {
        RecordSettings St = S1;
        St.threads = th;
        for (const Cell& c : ToyCells()) {
            if (th == 4 && c.N != 1) continue;   // 4 threads on the level-1 cells, 16 on all
            const HeckeStructureRecord R = BuildHeckeStructureRecord(c.N, c.n, c.alg, St, src);
            const HeckeStructureRecord& R1 = recs.at(Key(c));
            const bool eq = R.record_fnv1a64 == R1.record_fnv1a64 && R.cert_sha256 == R1.cert_sha256;
            threads_equal = threads_equal && eq;
            T.check(eq, std::string(c.name) + " " + c.alg + " record_fnv1a64 and cert_sha256 at " + std::to_string(th) + " threads");
        }
    }

    // ---- tamper arms ----------------------------------------------------------------------------------------------------------------
    uint32_t tampered = 0;
    {   // (1) a status flip moves the record hash (and the scan sees the now-orphaned local type)
        HeckeStructureRecord X = T1;
        bool flipped = false;
        for (Field& f : X.fields)
            if (f.path == "G2.ell=691.generation.status") {
                f.value = "open(sandwich)";
                flipped = true;
            }
        X.rehash();
        const bool ok = flipped && X.record_fnv1a64 != T1.record_fnv1a64 && ScanRecord(X).exact_under_bound > 0;
        T.check(ok, "tamper (1): a status flip moves record_fnv1a64 and orphans the local type");
        tampered += ok;
    }
    {   // (2) a G9 change moves cert_sha256 and not the record hash
        HeckeStructureRecord X = T1;
        bool changed = false;
        for (Field& f : X.fields)
            if (f.group == "G9" && !changed) {
                f.value += "x";
                changed = true;
            }
        X.rehash();
        const bool ok = changed && X.cert_sha256 != T1.cert_sha256 && X.record_fnv1a64 == T1.record_fnv1a64;
        T.check(ok, "tamper (2): a G9 change moves cert_sha256, not record_fnv1a64");
        tampered += ok;
    }
    {   // (3) a cap change moves the settings fingerprint (and the id, and the hash input)
        RecordSettings Sx = S1;
        Sx.K_steps = 4;
        const HeckeStructureRecord X = BuildHeckeStructureRecord(1, 10, "full", Sx, src);
        const bool ok = X.settings_fingerprint != T1.settings_fingerprint && X.id != T1.id && X.record_fnv1a64 != T1.record_fnv1a64;
        T.check(ok, "tamper (3): a cap change (K_steps 3 -> 4) moves the fingerprint, the id and record_fnv1a64");
        tampered += ok;
    }
    {   // (4) a Lean cap change moves cert_sha256 and not the record hash (P2b-6)
        RecordSettings Sx = S1;
        Sx.k1_cap_f = 9;
        const HeckeStructureRecord X = BuildHeckeStructureRecord(1, 10, "full", Sx, src);
        const bool ok = X.cert_sha256 != T1.cert_sha256 && X.record_fnv1a64 == T1.record_fnv1a64 && X.id == T1.id;
        T.check(ok, "tamper (4): a Lean cap change (k1_cap f <= 9) moves cert_sha256, not record_fnv1a64 nor the id");
        tampered += ok;
    }
    {   // (5) (schema audit F14) a provenance change (G10, the ed.1 join handle) moves neither hash
        HeckeStructureRecord X = T1;
        bool changed = false;
        for (Field& f : X.fields)
            if (f.path == "G10.threads") {
                f.value = "999";
                changed = true;
            }
        X.descent.cell_sha256 = std::string(64, 'f');
        X.rehash();
        const bool ok = changed && X.cert_sha256 == T1.cert_sha256 && X.record_fnv1a64 == T1.record_fnv1a64;
        T.check(ok, "tamper (5): a provenance change (G10.threads, descent_ref.cell_sha256) moves neither hash");
        tampered += ok;
    }
    {   // (6) (schema audit F4) the campaign's resume check: T1's cell file verifies; the auditor's tamper (a G6 split word changed,
        //     hashes left alone) and a forged out-of-list value with a recomputed hash are both rejected
        const std::string text = CellFileText(T1, "hzrec_validation");
        HeckeStructureRecord back;
        std::string why;
        const bool clean = VerifyCellText(text, T1.id, &back, &why) && back.record_fnv1a64 == T1.record_fnv1a64 &&
                           back.cert_sha256 == T1.cert_sha256 && back.status == T1.status;
        if (!clean) std::cout << "  [RESUME] clean file rejected: " << why << "\n";
        std::string t1 = text;
        const std::string from = "\"G6.q=1.ell=2.pt[0].split\",\"split_on_P\"";
        const size_t at = t1.find(from);
        bool rej1 = false;
        if (at != std::string::npos) {
            t1.replace(at, from.size(), "\"G6.q=1.ell=2.pt[0].split\",\"split\"");
            rej1 = !VerifyCellText(t1, T1.id, nullptr, &why);
            std::cout << "  RESUME tamper=G6_split rejected=" << (rej1 ? "yes" : "no") << " (" << why << ")\n";
        }
        HeckeStructureRecord Y = T1;   // an out-of-list value whose hashes are recomputed: only the scan sees it
        for (Field& f : Y.fields)
            if (f.path == "G6.q=1.ell=2.pt[0].split") f.value = "splits_probably";
        Y.rehash();
        const bool rej2 = !VerifyCellText(CellFileText(Y, "hzrec_validation"), T1.id, nullptr, &why);
        std::cout << "  RESUME tamper=outside_list_rehashed rejected=" << (rej2 ? "yes" : "no") << " (" << why << ")\n";
        const bool ok = clean && rej1 && rej2;
        T.check(ok, "tamper (6): a resumed cell file is verified (hashes, status, scan); the G6 tamper and an out-of-list value are rejected");
        tampered += ok;
    }
    {   // (7) (re-audit R5) tampers WITH the hashes recomputed that only the schema rules can see: split_on_P -> split (the re-audit's
        //     split_rehash), a vocabulary swapped to `status`, a parent blanked, a G8 field added, the header's descent_ref changed
        auto rejected = [&](HeckeStructureRecord X, const char* name, bool rehash, const std::function<bool(HeckeStructureRecord&)>& edit) {
            if (!edit(X)) {
                std::cout << "  RESUME tamper=" << name << " NOT APPLIED\n";
                return false;
            }
            if (rehash) X.rehash();
            std::string why;
            const bool rej = !VerifyCellText(CellFileText(X, "hzrec_validation"), T1.id, nullptr, &why);
            std::cout << "  RESUME tamper=" << name << " rejected=" << (rej ? "yes" : "no") << " (" << why << ")\n";
            return rej;
        };
        auto setf = [](HeckeStructureRecord& X, const std::string& path, const std::function<void(Field&)>& g) {
            bool hit = false;
            for (Field& f : X.fields)
                if (f.path == path) {
                    g(f);
                    hit = true;
                }
            return hit;
        };
        const bool a = rejected(T1, "split_rehash", true, [&](HeckeStructureRecord& X) {
            return setf(X, "G6.q=1.ell=2.pt[0].split", [](Field& f) { f.value = "split"; });
        });
        const bool b = rejected(T1, "vocab_swap_rehash", true, [&](HeckeStructureRecord& X) {
            return setf(X, "G6.q=1.ell=2.pt[0].split", [](Field& f) {
                f.vocab = "status";
                f.value = "certified";
            });
        });
        const bool c = rejected(T1, "under_parent_swap_rehash", true, [&](HeckeStructureRecord& X) {
            bool hit = setf(X, "G2.ell=691.generation.status", [](Field& f) { f.value = "open(sandwich)"; });
            for (Field& f : X.fields)
                if (f.parent == "G2.ell=691.generation.status") f.parent.clear();
            return hit;
        });
        const bool d = rejected(T1, "add_g8_free_rehash", true, [&](HeckeStructureRecord& X) {
            X.fields.push_back(Field{"G8", "G8.ell=691.m[0].L_k", "3", "int", ""});
            return true;
        });
        const bool e = rejected(T1, "dref_header", false, [&](HeckeStructureRecord& X) {
            X.descent.record_fnv1a64 = "0x0000000000000001";
            return true;
        });
        const bool ok = a && b && c && d && e;
        T.check(ok, "tamper (7): with the hashes recomputed, split_on_P -> split, a vocabulary swap, a blanked parent, a G8 field and a "
                    "changed header descent_ref are rejected (re-audit R5)");
        tampered += ok;
    }

    // ---- the word mapping of the Z0 merge report Sec. 7 (F4, O1) through forged sources at T1's ell = 691 -------------------------
    uint32_t mapped = 0;
    {
        auto forged = [&](int mode) {
            RecordSources f = src;
            f.local_read = [&src, mode](uint32_t N, uint32_t n, uint64_t ell, uint32_t K, const std::vector<uint32_t>& P, uint32_t b1,
                                        const heckez::local::ReadOptions& o, uint32_t steps) {
                if (ell != 691) return src.local_read(N, n, ell, K, P, b1, o, steps);
                if (mode == 1 && K >= 2) {   // the export's own K_max (HeckeZLocal.cpp LatticeFromExport at ell < 46341)
                    LocalReadOut x;
                    x.M.N = N; x.M.n = n; x.M.ell = ell; x.M.K = K; x.M.P = P;
                    x.M.source = "export";
                    x.M.status = "open(precision)";
                    x.M.detail = "K = " + std::to_string(K) + " > the export's K_max = 1";
                    return x;
                }
                if (mode == 2) {   // the level-1 module route at ell^2 >= 2^63 (LatticeFromH1Level1: Krun = max(K, 2))
                    LocalReadOut x;
                    x.M.N = N; x.M.n = n; x.M.ell = ell; x.M.K = K; x.M.P = P;
                    x.M.source = "h1_level1";
                    x.M.status = "refused(word_modulus)";
                    x.M.detail = "ell^2 >= 2^63 (the module route runs at K >= 2)";
                    return x;
                }
                if (mode >= 4) {   // (re-audit R6) a refused export module: [Lift] with a cap, [Lift] with a certificate, [ModulePrecision]
                    LocalReadOut x;
                    x.M.N = N; x.M.n = n; x.M.ell = ell; x.M.K = K; x.M.P = P;
                    x.M.source = "export";
                    x.M.status = "blocked(export)";
                    x.M.detail = mode == 4 ? "REFUSED [Lift] the cochain-level lift of T_2 at ell = 691: REFUSED [Lift] T~_2 on Iso: the strict "
                                             "lift is refused [RssCeiling]: predicted above kPairLiftRssCap"
                               : mode == 5 ? "REFUSED [Lift] the cochain-level lift of T_2 at ell = 691: REFUSED [Lift] T~_2 on Iso: the strict "
                                             "lift is refused [LiftCertificate]: (T2) fails"
                                           : "REFUSED [ModulePrecision] K = 3 above the read's precision";
                    return x;
                }
                LocalReadOut x = src.local_read(N, n, ell, K, P, b1, o, steps);
                if (mode == 3) {   // the step cap of the settings
                    x.L.root = "open(precision)";
                    x.escalation = "open(precision) at the step cap K=" + std::to_string(K);
                }
                return x;
            };
            return f;
        };
        auto v = [](const HeckeStructureRecord& R, const std::string& p) {
            const Field* f = R.find(p);
            return f ? f->value : std::string("-");
        };
        struct Exp {
            int mode;
            const char* gen;
            const char* read;
            const char* st;
            const char* overall;
        };
        const std::vector<Exp> E = {{1, "certified", "refused(export_word)", "refused(export_word)", "pending"},
                                    {2, "refused(k1_route)", "refused(word_modulus)", "upper_bound_certified", "pending"},
                                    {3, "certified", "open(precision)", "open(precision)", "final"},
                                    // (re-audit R6) the export's refusals by name: a cap of the lift is pending, a certificate final
                                    {4, "blocked(export)", "blocked(export)", "blocked(export)", "pending"},
                                    {5, "refused(no_lift)", "refused(no_lift)", "refused(no_lift)", "final"},
                                    {6, "refused(export_word)", "refused(export_word)", "refused(export_word)", "pending"}};
        for (const Exp& e : E) {
            const HeckeStructureRecord X = BuildHeckeStructureRecord(1, 10, "full", S1, forged(e.mode));
            const ScanResult sc = ScanRecord(X);
            const bool ok = v(X, "G2.ell=691.generation.status") == e.gen && v(X, "G3.C_T.ell=691.read") == e.read &&
                            v(X, "G3.C_T.ell=691.status") == e.st && v(X, "G3.C_T.status") == "upper_bound_certified" &&
                            X.find("G3.C_T.exact") == nullptr && X.status == e.overall && sc.values_outside_lists == 0 &&
                            sc.exact_under_bound == 0 && sc.overall_rule_ok;
            std::cout << "  MAP mode=" << e.mode << " gen=" << v(X, "G2.ell=691.generation.status") << " read=" << v(X, "G3.C_T.ell=691.read")
                      << " C_T.ell=" << v(X, "G3.C_T.ell=691.status") << " C_T=" << v(X, "G3.C_T.status") << " status=" << X.status << "\n";
            T.check(ok, "word mapping mode " + std::to_string(e.mode) + " (1: export K_max -> refused(export_word), 2: level-1 K = 1 -> "
                        "refused(k1_route), 3: step cap -> open(precision), 4: [Lift] cap -> blocked(export), 5: [Lift] certificate -> "
                        "refused(no_lift), 6: [ModulePrecision] -> refused(export_word))");
            mapped += ok;
        }
        // (re-audit R6) ExportWord on the texts themselves, the cokernel route's bare [Lift] (FromD3's default) included
        const bool words = ExportWord("REFUSED [Lift] x: the strict lift is refused [IsobClosure]: y") == "refused(no_lift)" &&
                           ExportWord("REFUSED [Lift] x: the strict lift is refused [StrictIdentity]: y") == "refused(no_lift)" &&
                           ExportWord("REFUSED [Lift] x: the strict lift is refused [InputTableCertificate]: y") == "refused(no_lift)" &&
                           ExportWord("REFUSED [Lift] x: the strict lift is refused [IntegerCap]: y") == "blocked(export)" &&
                           ExportWord("REFUSED [Lift] the cokernel operator: REFUSED [NotCocycle] z") == "blocked(export)" &&
                           ExportWord("REFUSED [RssCeiling] w") == "blocked(export)" && ExportWord("no refusal") == "blocked(export)";
        T.check(words, "R6: ExportWord gives refused(no_lift) for a certificate code only");
    }

    // ---- refusals (item 8 (a)) and the vocabulary ------------------------------------------------------------------------------------
    {
        bool threw = false;
        {
            const hecke::ScopedCoefficientRing fl(1);   // F_ell
            try {
                (void)BuildHeckeStructureRecord(1, 10, "full", S1, src);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
        }
        T.check(threw, "refusal: the record under coefficient_ring = F_ell throws invalid_argument (item 8 (a))");
        threw = false;
        try {
            (void)BuildHeckeStructureRecord(1, 10, "anemic", S1, src);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        T.check(threw, "refusal: `anemic` at N = 1 throws invalid_argument");
    }
    T.check(ClassOfStatus("refused(export_word)") == StatusClass::Pending && ClassOfStatus("refused(k1_route)") == StatusClass::Pending &&
                ClassOfStatus("open(precision)") == StatusClass::Final && ClassOfStatus("refused(word_modulus)") == StatusClass::Final &&
                ClassOfStatus("pending(HZ-TOR)") == StatusClass::Unknown && ClassOfStatus("exact") == StatusClass::Unknown,
            "vocabulary: the pending / final classes of item 8 (c)");
    // (slot-25 HZ-RAT-PERF (a), H1 audit F2) beyond the word-prime cap the record is refused BY NAME, not by an ENGINE_CHECK:
    // G1.C2_status = refused(word_prime_cap) (pending class), the scan clean; the cap is outside the hashed caps text
    {
        RecordSettings Sc;
        Sc.rat.word_prime_cap = 8;
        const HeckeStructureRecord X = BuildHeckeStructureRecord(1, 22, "full", Sc, src);
        const Field* c2 = X.find("G1.C2_status");
        const ScanResult sc = ScanRecord(X);
        for (const auto& p : sc.problems) std::cout << "  [SCAN] word_prime_cap " << p << "\n";
        std::cout << "  WORD_PRIME_CAP (1,22) cap=8 status=" << X.status << " C2=" << (c2 ? c2->value : std::string("-")) << " refusal=" << X.refusal
                  << "\n";
        T.check(X.status == "refused" && c2 && c2->value == "refused(word_prime_cap)" && sc.values_outside_lists == 0 &&
                    ClassOfStatus("refused(word_prime_cap)") == StatusClass::Pending && ValueInVocab("C2", "refused(word_prime_cap)"),
                "refusal: beyond the word-prime cap the record is refused(word_prime_cap), a pending word of the closed list");
        T.check(Sc.caps_text() == S1.caps_text(), "refusal: the word-prime cap is provenance (the hashed caps text is unchanged)");
    }
    // (HZ-RAT-PERF re-audit N4) the U_p tag's refusal site: (2, 6) full, the record's rational step at its default cap (10 word primes),
    // the U_2 tag capped at 2 by the validation-only RecordSettings::utag_word_prime_cap (its chi_Q(U_2) needs 3, its nested anemic
    // RationalStep on anemic_P = P about 10: either refuses) -> the record-level catch after UTagStep; the cap given explicitly as the
    // default 2^21 is the default record, hash for hash
    {
        RecordSettings Su2 = S1, Su3 = S1;
        Su2.utag_word_prime_cap = 2;
        Su3.utag_word_prime_cap = 1u << 21;
        const HeckeStructureRecord X = BuildHeckeStructureRecord(2, 6, "full", Su2, src);
        const HeckeStructureRecord Y = BuildHeckeStructureRecord(2, 6, "full", Su3, src);
        const HeckeStructureRecord Z = BuildHeckeStructureRecord(2, 6, "full", S1, src);
        const Field* c2 = X.find("G1.C2_status");
        const ScanResult sc = ScanRecord(X);
        std::cout << "  WORD_PRIME_CAP_UTAG (2,6) full utag_cap=2 status=" << X.status << " C2=" << (c2 ? c2->value : std::string("-"))
                  << " refusal=" << X.refusal << " utag_cap_2^21_equal=" << (Y.record_fnv1a64 == Z.record_fnv1a64 ? "yes" : "no") << "\n";
        T.check(X.status == "refused" && c2 && c2->value == "refused(word_prime_cap)" && X.refusal.compare(0, 23, "refused(word_prime_cap)") == 0 &&
                    sc.values_outside_lists == 0 && Y.record_fnv1a64 == Z.record_fnv1a64 && Y.status != "refused",
                "refusal (re-audit N4): the U_p tag's WordPrimeCapRefusal is caught by the record (refused(word_prime_cap)); at 2^21 the default record");
    }
    T.check(ValueInVocab("t_ell", "omitted(export:NoLift)") && !ValueInVocab("t_ell", "omitted(export:REFUSED [x] why)") &&
                ValueInVocab("t_ell", "omitted(cap:ell>23170)"),
            "vocabulary: t_ell's closed list, omitted(export:<Name>) by name only (Z0 referee F6)");
    // (schema audit F2) G8 is not in default records, and a non-default G8 knob is refused (H1's block is not built)
    {
        bool g8 = false;
        for (const auto& [k, R] : recs)
            for (const Field& f : R.fields) g8 = g8 || f.group == "G8";
        bool threw = false;
        RecordSettings Sg = S1;
        Sg.g8 = "H1";
        try {
            (void)BuildHeckeStructureRecord(1, 10, "full", Sg, src);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        T.check(!g8 && threw, "F2: no G8 field in a default record; G8 != off throws invalid_argument");
    }
    // (F1) no G6 point carries `split` (generation by P on H^1 / ell^K is CB-3); T1's points at 2 and 3 are split_on_P
    {
        bool plain = false;
        for (const auto& [k, R] : recs)
            for (const Field& f : R.fields) plain = plain || (f.group == "G6" && f.value == "split");
        const Field* a = T1.find("G6.q=1.ell=2.pt[0].split");
        const Field* b = T1.find("G6.q=1.ell=3.pt[0].split");
        T.check(!plain && a && a->value == "split_on_P" && b && b->value == "split_on_P", "F1: split(T') is written split_on_P; no `split`");
    }
    // (F5) the cuspidal block: absent at d = e_k (here (1,4)); T2's quadratic block irreducible with p0 = T_2
    {
        const HeckeStructureRecord R4 = BuildHeckeStructureRecord(1, 4, "full", S1, src);
        const Field* part = R4.find("G3.partition");
        const ScanResult s4 = ScanRecord(R4);
        T.check(R4.find("G1.components[1].type") == nullptr && part && part->value == "none(single_component)" &&
                    R4.find("G3.C_T.status") == nullptr && s4.values_outside_lists == 0 && s4.exact_under_bound == 0,
                "F5 (a): (1,4) has no cuspidal component and no partition");
        const HeckeStructureRecord& T2 = recs.at("1,22,full");
        const Field* ir = T2.find("G1.components[1].irreducible");
        const Field* p0 = T2.find("G1.components[1].p0");
        T.check(ir && ir->value == "true" && p0 && p0->value == "T_2" && T2.find("G9.irreducible.cusp.q") != nullptr,
                "F5 (b): (1,22)'s cuspidal block is certified irreducible (a prime in G9), p0 = T_2");
        // (F6) (1,22) at 5: saturation, etale, decomposed by a searched theta (three points)
        const Field* mi = T2.find("G2.ell=5.maximal_ideals");
        T.check(mi && mi->value == "certified" && T2.find("G2.ell=5.m[2].residue_degree") != nullptr && T2.find("G2.ell=5.m[3].residue_degree") == nullptr,
                "F6: (1,22) at ell = 5 is decomposed (three maximal ideals)");
    }
    // the golden rule (item 8 (k)): eligible iff final; every final record has no pending field (the scan's overall rule) -- and T1, T2
    // (the plan's first-set members among the toy cells) are final once HZ-TOR fills G4-G6
    T.check(recs.at("1,10,full").golden_eligible() && recs.at("1,22,full").golden_eligible(),
            "golden rule: T1 and T2 are final (no pending field) with HZ-TOR wired");

    std::cout << "  HZREC_DIGEST 0x" << std::hex << digest << std::dec << " records=" << recs.size() << " golden_eligible=" << golden << "\n";
    // (re-audit R9) (1,26) at 7: generation is certified by saturation with T_7 in the read set only; theta_search over P u {T_7}
    // decomposes it (the first theta of the order is T_5 + T_7, the re-audit's independent check)
    {
        const HeckeStructureRecord R26 = BuildHeckeStructureRecord(1, 26, "full", S1, src);
        const Field* mi = R26.find("G2.ell=7.maximal_ideals");
        const Field* db = R26.find("G2.ell=7.decomposed_by");
        const ScanResult s26 = ScanRecord(R26);
        std::cout << "  R9 (1,26) ell=7 maximal_ideals=" << (mi ? mi->value : "-") << " decomposed_by=" << (db ? db->value : "-") << "\n";
        T.check(mi && mi->value == "certified" && db && db->value == "theta=(0 0 1 1) on T_2,T_3,T_5,T_7" && s26.values_outside_lists == 0 &&
                    s26.exact_under_bound == 0 && R26.status == "final",
                "R9: (1,26) at ell = 7 is decomposed by theta = T_5 + T_7 over the read set");
    }
    // (re-audit R1) the Eisenstein multiplicity on H^1 is e_k: 2 at (2,6) anemic (the lattice's (x - 2188)^2 (x - 12)^2), 1 at level 1
    {
        const HeckeStructureRecord& A = recs.at("2,6,anemic");
        const Field* m = A.find("G1.components[0].mult");
        const Field* ek = A.find("G1.e_k");
        const Field* m1 = T1.find("G1.components[0].mult");
        T.check(m && ek && m->value == ek->value && m->value == "2" && m1 && m1->value == "1",
                "R1: G1.components[0].mult = e_k (2 at (2,6) anemic, 1 at (1,10))");
    }
    const bool ok = T.failing == 0 && table_ok == ToyCells().size() && hash_ok == ToyCells().size() && outside == 0 && under == 0 &&
                    threads_equal && tampered == 7 && ed1_bad == 0 && mapped == 6;
    std::cout << "HeckeZRecordValidation: " << T.checks << " checks, " << T.failing << " failing\n";
    std::cout << "HECKEZ_RECORD_WORDS export_word=" << (mapped >= 1 ? "mapped" : "MISSING") << " modes=" << mapped << "/6\n";
    std::cout << "HECKEZ_RECORD cells=5 status_table=" << (table_ok == ToyCells().size() ? "exact" : "MISMATCH")
              << " hashes_pinned=" << hash_ok << "/" << ToyCells().size()
              << " exact_under_bound=" << under << " values_outside_lists=" << outside << " threads_equal=" << (threads_equal ? "yes" : "no")
              << " tampered_detected=" << tampered << " ed1_digest_mismatch=" << ed1_bad << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

// =====================================================================================================================================
// HZ-EMIT
// =====================================================================================================================================
uint32_t HeckeZCertDumpValidation() {
    const char* env = std::getenv("HECKE_DUMP_HZ_CERT");
    if (env == nullptr || *env == '\0') {
        std::cout << "[SKIP] HeckeZCertDumpValidation: HECKE_DUMP_HZ_CERT not set (scratch emitter of the HeckeZ certificate files, "
                     "HZ-EMIT)\n";
        return 0;
    }
    const std::string dir(env);
    if (!IsDirectory(dir)) {
        std::cout << "HeckeZCertDumpValidation: REFUSED: HECKE_DUMP_HZ_CERT=" << dir << " is not an existing directory\n";
        return 1;
    }
    const std::string real_dir = RealPath(dir);
#ifdef OC_SOURCE_DIR
    const std::string real_src = RealPath(OC_SOURCE_DIR);
#else
    const std::string real_src;
#endif
    if (real_dir.empty() || IsInside(real_dir, real_src)) {
        std::cout << "HeckeZCertDumpValidation: REFUSED: HECKE_DUMP_HZ_CERT=" << dir << " lies inside the source tree (" << real_src
                  << "); dump into scratch only\n";
        return 1;
    }
    struct DumpCell {
        uint32_t N, n;
        uint64_t ell;   // 0: every ell of the record
        bool full = false;   // (S26 Z1) N:n:ell[:K]:full -- the full record at N > 1, plus its U_p files <record id>.Up<p>.mat
    };
    std::vector<DumpCell> cells = {{1, 10, 691}, {1, 22, 103}, {1, 22, 2294797}};
    std::string arm = "default";
    if (const char* cenv = std::getenv("HECKE_DUMP_HZ_CERT_CELLS")) {
        if (!*cenv) {
            std::cout << "HeckeZCertDumpValidation: REFUSED: HECKE_DUMP_HZ_CERT_CELLS: empty value (unset it for the default)\n";
            return 1;
        }
        cells.clear();
        arm = "cells";
        std::stringstream ss(cenv);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            uint32_t N = 0, n = 0, K = 0;
            unsigned long long ell = 0;
            bool full = false;   // (S26 Z1) a trailing ":full" token
            if (tok.size() > 5 && tok.compare(tok.size() - 5, 5, ":full") == 0) {
                full = true;
                tok.erase(tok.size() - 5);
            }
            if (std::sscanf(tok.c_str(), "%u:%u:%llu:%u", &N, &n, &ell, &K) < 3 || N == 0 || n % 2 || n > 60 || (full && N == 1)) {
                std::cout << "HeckeZCertDumpValidation: REFUSED: HECKE_DUMP_HZ_CERT_CELLS: bad entry '" << tok
                          << "' (N:n:ell[:K][:full], N >= 1, n even <= 60, :full only at N > 1; n > 60 is emitted by the campaign, hz_cert = on)\n";
                return 1;
            }
            cells.push_back({N, n, ell, full});   // K: the record's own (the read K is a certificate-layer value)
        }
    }
    std::cout << "EFFECTIVE HeckeZCertDumpValidation arm=" << arm << " cells=" << cells.size() << "\n";
    const hecke::ScopedCoefficientRing ring(0);
    const RecordSources src = DefaultRecordSources();
    const RecordSettings S;
    uint32_t files = 0, unstable = 0, sha_mismatch = 0, up_files = 0, up_refused = 0;
    std::set<std::string> written;
    std::map<std::string, HeckeStructureRecord> cache;
    for (const DumpCell& c : cells) {
        const std::string alg = c.N == 1 || c.full ? "full" : "anemic";
        const std::string key = std::to_string(c.N) + ":" + std::to_string(c.n) + ":" + alg;
        if (!cache.count(key))
            cache.emplace(key, BuildHeckeStructureRecord(c.N, c.n, alg, S, src));
        const HeckeStructureRecord& R = cache.at(key);
        if (c.full) {
            // (S26 Z1, UPZ_design.md Sec. 3) the U_p files of the full record, written twice and compared, each with its own sha256
            std::vector<std::string> why;
            const std::vector<std::string> ua = heckez::up::WriteUpMatrixFiles(c.N, c.n, R.id, R.record_fnv1a64, dir, &why);
            std::map<std::string, std::string> ufirst;
            for (const auto& f : ua) ufirst[f] = ReadFile(dir + "/" + f);
            const std::vector<std::string> ub = heckez::up::WriteUpMatrixFiles(c.N, c.n, R.id, R.record_fnv1a64, dir);
            for (const auto& f : ub) {
                const std::string t = ReadFile(dir + "/" + f);
                if (!ufirst.count(f) || ufirst[f] != t) ++unstable;
                if (!heckez::up::UpMatrixShaOk(t)) ++sha_mismatch;
                std::cout << "  FILE " << f << " bytes=" << t.size() << "\n";
            }
            if (ua != ub) ++unstable;
            for (const auto& w : why) std::cout << "  UP_REFUSED " << R.id << " " << w << "\n";
            up_files += static_cast<uint32_t>(ub.size());
            up_refused += static_cast<uint32_t>(why.size());
            written.insert(ub.begin(), ub.end());
        }
        const std::vector<std::string> a = WriteCertFiles(R, src, S, dir, c.ell);
        std::map<std::string, std::string> first;
        for (const auto& f : a) first[f] = ReadFile(dir + "/" + f);
        const std::vector<std::string> b = WriteCertFiles(R, src, S, dir, c.ell);
        for (const auto& f : b) {
            const std::string t = ReadFile(dir + "/" + f);
            if (!first.count(f) || first[f] != t) ++unstable;
            // the recomputed dump sha256 must equal the record's G9 value
            const size_t p = t.find("dump_sha256_recomputed\t");
            const size_t q1 = t.find("satcert_sha256\t"), q3 = t.find("\ndump_sha256\t");
            if (p != std::string::npos) {
                const std::string rec = t.substr(p + 23, 64);
                const size_t q = q1 != std::string::npos ? q1 + 15 : (q3 != std::string::npos ? q3 + 13 : std::string::npos);
                if (q == std::string::npos || t.substr(q, 64) != rec) ++sha_mismatch;
            }
            std::cout << "  FILE " << f << " bytes=" << t.size() << "\n";
        }
        written.insert(b.begin(), b.end());
        files = static_cast<uint32_t>(written.size());
    }
    // (S26 Z1 audit S5) a :full cell whose U_p file is refused fails the dump (the line is read by eye)
    const bool ok = unstable == 0 && sha_mismatch == 0 && files > 0 && up_refused == 0;
    std::cout << "HECKEZ_CERT_DUMP cells=" << cells.size() << " files=" << files << " byte_stable=" << (unstable == 0 ? "yes" : "no")
              << " dump_sha_mismatch=" << sha_mismatch << " up_files=" << up_files << " up_refused=" << up_refused
              << " verdict=" << (ok ? "OK" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
