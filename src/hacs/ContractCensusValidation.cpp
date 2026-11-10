// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions slot 3, WP-10 -- ContractCensusValidation.
// Extended in hecke-extensions slot 4, WP-6 (label_census.allow becomes
// content-keyed): four scratch arms added, see the header for what they gate.
//
// It runs docs/matrix-engine/tools/label_census.sh over the checkout, the
// same std::system pattern MatrixRetirementValidation /
// HeckeExtArchitectureValidation use for their scripts, then -- unless
// python3 is unavailable -- exercises the production
// docs/matrix-engine/tools/label_census.py directly against scratch roots +
// scratch allow-lists built under a throwaway temp directory (never touching
// the real label_census.allow or docs/latex/label_census.tsv), following the
// CampaignValidation.cpp / ConsoleValidation.cpp precedent for a
// getpid()-scoped std::filesystem scratch directory and a popen()-captured
// child process respectively.

#include "ContractCensusValidation.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

const char* kScriptRel = "docs/matrix-engine/tools/label_census.sh";
const char* kPyScriptRel = "docs/matrix-engine/tools/label_census.py";

bool PathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool LooksLikeRepoRoot(const std::string& dir) {
    return PathExists(dir + "/" + kScriptRel) && PathExists(dir + "/docs/latex/gl2_contract.tex");
}

// The two independent routes of ConsumerMigrationValidation / HeckeExtArchitectureValidation:
// OC_SOURCE_DIR as baked in by CMake (exact, but stale if the tree moved) and
// walking up from /proc/self/exe (survives a move, but not an out-of-source
// build), then the working directory (the in-source build runs the binary
// from the source root).
std::string FindRepoRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string cmake_root(OC_SOURCE_DIR);
        if (!cmake_root.empty() && LooksLikeRepoRoot(cmake_root)) return cmake_root;
    }
#endif
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            const size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
            if (LooksLikeRepoRoot(dir)) return dir;
        }
    }
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (LooksLikeRepoRoot(dir)) return dir;
            const size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
        }
    }
    return std::string();
}

bool HasPython3() {
    return std::system("command -v python3 > /dev/null 2>&1") == 0;
}

// ---------------------------------------------------------------------------
// Scratch-tree arms (WP-6): exercise the production label_census.py against
// throwaway roots + throwaway allow-lists, never the real
// docs/matrix-engine/tools/label_census.allow or docs/latex/label_census.tsv.
// ---------------------------------------------------------------------------

std::filesystem::path MakeScratchDir(const std::string& tag) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec);
    const fs::path dir = (ec ? fs::path(".") : base) /
                         ("oc_label_census_" + tag + "_" + std::to_string(::getpid()));
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir;
}

void WriteFile(const std::filesystem::path& p, const std::string& content) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << content;
}

// Runs `cmd`, capturing stdout+stderr merged (the python script's diagnostic
// lines are what these arms assert on). Returns the exit status, or -1.
int RunCaptured(const std::string& cmd, std::string& out) {
    out.clear();
    FILE* pipe = ::popen((cmd + " 2>&1").c_str(), "r");
    if (pipe == nullptr) return -1;
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) out += buf;
    const int status = ::pclose(pipe);
    if (status == -1 || !WIFEXITED(status)) return -1;
    return WEXITSTATUS(status);
}

// Invokes the PRODUCTION label_census.py exactly as label_census.sh itself
// would, but with --allow / --tsv pointed at the scratch tree.
int RunScratchCensus(const std::string& py_script, const std::string& scratch_root,
                      const std::string& allow_path, std::string& out) {
    const std::string cmd = "python3 \"" + py_script + "\" \"" + scratch_root +
                             "\" --tsv \"" + scratch_root + "/label_census.tsv\" --allow \"" +
                             allow_path + "\"";
    return RunCaptured(cmd, out);
}

// The arms below need to write a literal bracket-form citation (opening
// bracket, "C", comma, a token, closing bracket) into throwaway scratch
// files -- that IS the payload label_census.py is being tested against. But
// this .cpp file is ITSELF a candidate file the real, production
// label_census.py scans (every .cpp under src/hacs/ is),
// so spelling that whole bracket span out on one physical source line here
// would make THIS file a citation site the real census has to resolve --
// exactly the kind of self-triggering WP-10's report
// (docs/implementation/slot3/WP-10_report.md Sec. 3.1) hit once already.
// kCiteOpen and kCiteClose are therefore defined on two separate physical
// lines, each alone incomplete against label_census.py's CITATION_RE (which
// matches only within a single source line), and Cite() joins them at
// RUNTIME -- so the scratch files this validation writes still carry the
// exact byte-for-byte citation text the arms need, while this source file
// itself never spells out a complete bracket span.
const char kCiteOpen[] = "[C, ";
const char kCiteClose[] = "]";
std::string Cite(const std::string& token) { return std::string(kCiteOpen) + token + kCiteClose; }

// Arm 1: an allow-listed site must stay allow-listed after a line is
// inserted above it -- the whole point of a content-keyed allow-list. A
// line-number-keyed entry would silently stop matching once the citation's
// line number moved; the census must stay green regardless.
uint32_t CheckInsertionKeepsAllowed(const std::string& py_script) {
    namespace fs = std::filesystem;
    const fs::path dir = MakeScratchDir("insert");
    WriteFile(dir / "docs/latex/gl2_contract.tex", "% scratch contract -- no \\label{}s defined\n");
    WriteFile(dir / "note.md",
              "Prose describing the citation syntax uses the generic placeholder `" +
              Cite("label") + "` here.\n");
    WriteFile(dir / "allow.txt",
              "# scratch reason: the bare word \"label\" is the placeholder, not a real label\n"
              "note.md|" + Cite("label") + "\n");

    std::string out_before;
    const int rc_before = RunScratchCensus(py_script, dir.string(), (dir / "allow.txt").string(), out_before);
    const bool before_ok = rc_before == 0 && out_before.find("unresolved=0") != std::string::npos;

    // Insert a line ABOVE the citation: it moves from source line 1 to 2.
    WriteFile(dir / "note.md",
              "This line was inserted above the citation to shift its line number.\n"
              "Prose describing the citation syntax uses the generic placeholder `" +
              Cite("label") + "` here.\n");
    std::string out_after;
    const int rc_after = RunScratchCensus(py_script, dir.string(), (dir / "allow.txt").string(), out_after);
    const bool after_ok = rc_after == 0 && out_after.find("unresolved=0") != std::string::npos;

    std::error_code ec;
    fs::remove_all(dir, ec);

    if (before_ok && after_ok) {
        std::cout << "  [content-keyed arm 1/4] insertion above an allow-listed site: PASSED "
                     "(unresolved=0 both before and after the citation's line number moved)\n";
        return 0u;
    }
    std::cout << "  [content-keyed arm 1/4] insertion above an allow-listed site: FAILED "
                 "(rc_before=" << rc_before << " rc_after=" << rc_after << ")\n"
              << out_before << out_after;
    return 1u;
}

// Arm 2: a genuinely undefined citation, not covered by any allow-list
// entry, must still fail the census (the negative control WP-10's report
// verified by hand; this makes it part of the validation itself).
uint32_t CheckPlantedBadCitationFails(const std::string& py_script) {
    namespace fs = std::filesystem;
    const fs::path dir = MakeScratchDir("bad");
    WriteFile(dir / "docs/latex/gl2_contract.tex", "% scratch contract -- no \\label{}s defined\n");
    WriteFile(dir / "bad.md",
              "This cites an undefined label `" + Cite("no_such_label_xyz") + "` on purpose.\n");
    WriteFile(dir / "allow.txt", "# empty scratch allow-list -- nothing is allow-listed here\n");

    std::string out;
    const int rc = RunScratchCensus(py_script, dir.string(), (dir / "allow.txt").string(), out);

    std::error_code ec;
    fs::remove_all(dir, ec);

    if (rc == 1 && out.find("no_such_label_xyz") != std::string::npos &&
        out.find("unresolved=1") != std::string::npos) {
        std::cout << "  [content-keyed arm 2/4] planted bad citation: PASSED (rc=1, flagged)\n";
        return 0u;
    }
    std::cout << "  [content-keyed arm 2/4] planted bad citation: FAILED (rc=" << rc << ")\n" << out;
    return 1u;
}

// Arm 3: an allow-list entry that matches nothing must be reported STALE and
// fail the census -- otherwise a rewritten or removed passage could leave a
// dead entry masking a real regression forever.
uint32_t CheckStaleEntryFails(const std::string& py_script) {
    namespace fs = std::filesystem;
    const fs::path dir = MakeScratchDir("stale");
    WriteFile(dir / "docs/latex/gl2_contract.tex", "% scratch contract -- no \\label{}s defined\n");
    WriteFile(dir / "note.md", "No citation of any kind on this line.\n");
    WriteFile(dir / "allow.txt",
              "# an entry whose fragment appears nowhere in this scratch tree\n"
              "note.md|" + Cite("this_fragment_is_absent") + "\n");

    std::string out;
    const int rc = RunScratchCensus(py_script, dir.string(), (dir / "allow.txt").string(), out);

    std::error_code ec;
    fs::remove_all(dir, ec);

    if (rc != 0 && out.find("STALE") != std::string::npos && out.find("stale=1") != std::string::npos) {
        std::cout << "  [content-keyed arm 3/4] stale allow-list entry: PASSED (rc=" << rc
                  << ", flagged)\n";
        return 0u;
    }
    std::cout << "  [content-keyed arm 3/4] stale allow-list entry: FAILED (rc=" << rc << ")\n" << out;
    return 1u;
}

// Arm 4: the retired `relpath:line` form must be rejected with a named
// error, never silently accepted (which would defeat the whole migration)
// and never silently ignored (which would leave the census running with an
// incomplete allow-list, i.e. wrongly strict, without saying so).
uint32_t CheckOldFormRejected(const std::string& py_script) {
    namespace fs = std::filesystem;
    const fs::path dir = MakeScratchDir("oldform");
    WriteFile(dir / "docs/latex/gl2_contract.tex", "% scratch contract -- no \\label{}s defined\n");
    WriteFile(dir / "note.md",
              "Prose describing the citation syntax uses the generic placeholder `" +
              Cite("label") + "` here.\n");
    WriteFile(dir / "allow.txt",
              "# the retired line-number form -- must be rejected, not silently reinterpreted\n"
              "note.md:1\n");

    std::string out;
    const int rc = RunScratchCensus(py_script, dir.string(), (dir / "allow.txt").string(), out);

    std::error_code ec;
    fs::remove_all(dir, ec);

    const bool named = out.find("retired") != std::string::npos &&
                        out.find("ALLOW_LIST_ERROR") != std::string::npos;
    if (rc != 0 && named) {
        std::cout << "  [content-keyed arm 4/4] retired relpath:line form: PASSED (rc=" << rc
                  << ", named error)\n";
        return 0u;
    }
    std::cout << "  [content-keyed arm 4/4] retired relpath:line form: FAILED (rc=" << rc << ")\n" << out;
    return 1u;
}

} // namespace

uint32_t ContractCensusValidation(void) {
    std::cout << "ContractCensusValidation: every [C, <label>] / \\HA{<label>} citation "
                 "resolves to a \\label{} in docs/latex/gl2_contract.tex ("
              << kScriptRel << ").\n";

    const std::string root = FindRepoRoot();
    if (root.empty()) {
        std::cout << "  SKIPPED: " << kScriptRel
                  << " not found from the working directory, from OC_SOURCE_DIR or from "
                     "/proc/self/exe (binary run outside the checkout?). Nothing to check.\n";
        return 0u;
    }

    uint32_t failures = 0u;

    std::cout.flush();  // the script writes to the same stream
    const std::string cmd = "bash \"" + root + "/" + kScriptRel + "\" \"" + root + "\"";
    const int status = std::system(cmd.c_str());
    if (status == -1) {
        std::cout << "  SKIPPED: could not spawn a shell to run the check.\n";
        return 0u;
    }
    const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (rc == 0) {
        std::cout << "  main census: PASSED.\n";
    } else if (rc == 2) {
        // Usage / environment error the script itself detects (no python3,
        // or the passed root does not look like the repository) -- a missing
        // *capability*, not an unresolved citation, so it is a SKIP like a
        // missing script, not a FAIL. Also skips the scratch arms below,
        // since they need the same python3.
        std::cout << "  SKIPPED: " << kScriptRel
                  << " reported a usage/environment error (exit 2) -- see the output above "
                     "(python3 unavailable?). Nothing was asserted.\n";
        return 0u;
    } else {
        std::cout << "  main census: FAILED (see the FAIL/STALE lines above).\n";
        ++failures;
    }

    // Content-keyed allow-list arms (hecke-extensions slot 4, WP-6): exercise
    // the production label_census.py directly against scratch roots +
    // scratch allow-lists. Skipped, not failed, without python3 (same
    // capability the main census above already needed to have gotten past
    // rc == 2).
    if (HasPython3()) {
        const std::string py_script = root + "/" + kPyScriptRel;
        failures += CheckInsertionKeepsAllowed(py_script);
        failures += CheckPlantedBadCitationFails(py_script);
        failures += CheckStaleEntryFails(py_script);
        failures += CheckOldFormRejected(py_script);
    } else {
        std::cout << "  SKIPPED: content-keyed allow-list scratch arms (python3 unavailable).\n";
    }

    if (failures == 0u) {
        std::cout << "ContractCensusValidation: PASSED.\n";
    } else {
        std::cout << "ContractCensusValidation: FAILED (" << failures << " check(s) failed).\n";
    }
    return failures;
}
