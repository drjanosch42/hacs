// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// ConsumerMigrationValidation.cpp -- matrix-engine Phase 3, Stage A.
// See ConsumerMigrationValidation.h for what this gates and why the target is
// not zero.
// ===========================================================================

#include "ConsumerMigrationValidation.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>   // WIFEXITED / WEXITSTATUS on std::system's wait status
#include <unistd.h>
#include <vector>

namespace {

bool PathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

// A tree is "the repository root" iff it carries the tools directory this
// validation drives. Checking for the directory rather than for, say,
// CMakeLists.txt keeps the probe honest: a root without the tools is useless
// here even if it is a valid checkout.
bool LooksLikeRepoRoot(const std::string& dir) {
    return PathExists(dir + "/docs/matrix-engine/tools") &&
           PathExists(dir + "/src/hacs");
}

// Locating the source tree from inside the binary. Two independent routes,
// tried in order, because each fails in a case the other survives:
//
//  (1) OC_SOURCE_DIR, baked in by CMake (see CMakeLists.txt, Phase 3 options).
//      Exact and free -- but stale if the tree was moved, renamed or copied
//      after the build, and absent if some other build system compiled this TU.
//  (2) Walking up from the executable's own path (/proc/self/exe, falling back
//      to argv[0]-independent cwd). Survives a moved tree -- but NOT an
//      out-of-source build, where the executable lives in a build directory
//      that is not under the source root.
//
// Neither is robust alone; together they cover in-source builds (the tree's
// former documented configuration), out-of-source builds, and a
// relocated in-source tree. Returns "" if the root cannot be established.
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

// std::system's return value is a wait(2) status, not the child's exit code.
// A tool that reports "17 sites outside the allow-list" as exit code 17 would
// be read as success by a naive `== 0` test on some of the ways this can go
// wrong, so decode it explicitly: -1 (couldn't run), a signal, or an exit code.
struct RunResult {
    bool ran = false;      // the shell was started and the child terminated normally
    int exit_code = -1;
};

RunResult RunTool(const std::string& root, const std::string& script,
                  const std::string& args) {
    // cd into the root first: the scripts derive their own defaults from their
    // location, but running them from the root makes the printed paths and any
    // future relative default match what a developer sees on the command line.
    const std::string cmd = "cd '" + root + "' && bash '" + root + "/docs/matrix-engine/tools/" +
                            script + "'" + (args.empty() ? "" : " " + args);
    std::cout << "  $ " << cmd << std::endl;
    std::cout.flush();
    const int status = std::system(cmd.c_str());
    RunResult r;
    if (status == -1) return r;
    if (WIFEXITED(status)) {
        r.ran = true;
        r.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        std::cout << "  [FAIL] " << script << " died on signal " << WTERMSIG(status) << std::endl;
    }
    return r;
}

} // namespace

uint32_t ConsumerMigrationValidation() {
    std::cout << "=== ConsumerMigrationValidation (matrix-engine Phase 3, Stage A) ===" << std::endl;

    // A shell is a hard requirement for this validation -- without it the gate
    // cannot run at all, and silently passing would make it worthless.
    if (std::system(nullptr) == 0) {
        std::cout << "[SKIP] ConsumerMigrationValidation: no command processor available;"
                     " the migration tool scripts cannot be run." << std::endl;
        return 0;
    }

    const std::string root = FindRepoRoot();
    if (root.empty()) {
        std::cout << "[SKIP] ConsumerMigrationValidation: could not locate the source tree"
                     " (no OC_SOURCE_DIR, and no docs/matrix-engine/tools above the executable"
                     " or the working directory). The static migration gate did NOT run."
                  << std::endl;
        return 0;
    }
    std::cout << "  source tree: " << root << std::endl;

    uint32_t failures = 0;
    uint32_t ran = 0;
    uint32_t skipped = 0;

    struct Tool {
        const char* script;
        const char* args;
        bool required;     // false => a missing script is a SKIP, not a failure
        const char* what;
    };
    const std::vector<Tool> tools = {
        // Owned by this Phase 3 work; must exist.
        {"raw_mod_calls.sh", "--census", true,
         "raw ModArithmetic mod_* call sites outside raw_mod_calls.allow"},
        // Owned by the concurrent `Matrix` retirement work. Probe, do not require:
        // this validation must be green on a tree where that script has not
        // landed yet (see the header).
        {"check_matrix_includes.sh", "", false,
         "Matrix.h includes in pipeline TUs outside the retirement allow-list"},
    };

    for (const Tool& t : tools) {
        const std::string path = root + "/docs/matrix-engine/tools/" + std::string(t.script);
        if (!PathExists(path)) {
            if (t.required) {
                std::cout << "[FAIL] missing required tool script: " << path << std::endl;
                ++failures;
            } else {
                std::cout << "[SKIP] " << t.script << " is not present in this tree"
                             " (not yet landed); " << t.what << " is NOT gated by this run."
                          << std::endl;
                ++skipped;
            }
            continue;
        }
        if (::access(path.c_str(), X_OK) != 0) {
            std::cout << "[FAIL] tool script is not executable: " << path
                      << " (chmod +x it, and commit the executable bit)" << std::endl;
            ++failures;
            continue;
        }
        const RunResult r = RunTool(root, t.script, t.args);
        if (!r.ran) {
            std::cout << "[FAIL] could not run " << t.script << std::endl;
            ++failures;
            continue;
        }
        ++ran;
        if (r.exit_code == 0) {
            std::cout << "[PASS] " << t.script << ": no " << t.what << std::endl;
        } else if (r.exit_code == 126) {
            std::cout << "[FAIL] " << t.script << ": usage/environment error (exit 126)" << std::endl;
            ++failures;
        } else {
            std::cout << "[FAIL] " << t.script << ": exit " << r.exit_code
                      << " -- that many " << t.what << ".\n"
                         "       Either migrate them off the raw primitives, or -- if the file is"
                         " genuinely out of\n"
                         "       scope -- add it to the tool's allow-list WITH A REASON."
                      << std::endl;
            ++failures;
        }
    }

    std::cout << "ConsumerMigrationValidation: " << ran << " tool(s) run, "
              << skipped << " skipped, " << failures << " failure(s)." << std::endl;
    if (failures == 0) {
        std::cout << "ConsumerMigrationValidation: PASSED" << std::endl;
    } else {
        std::cout << "ConsumerMigrationValidation: FAILED" << std::endl;
    }
    return failures;
}
