// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

/*
 * MatrixRetirementValidation.cpp -- matrix-engine Phase 3.
 *
 * Runs docs/matrix-engine/tools/check_matrix_includes.sh over the checkout and
 * turns its exit code into a validation result. See the header for the
 * contract and for why a missing checkout is a SKIP rather than a failure.
 *
 * (slot-14 WP-2, [C3] finding 5) The original `find_repo_root()` looked only
 * at the working directory and a few relative parents, so a run from a
 * cluster case directory on /scratch (cwd == the case dir, nowhere near the
 * checkout) never found the tree and silently printed nothing but "SKIPPED"
 * -- while the validation still returned 0, so the harness printed
 * "[PASS] MatrixRetirementValidation" for a check that never ran. Fall back
 * to OC_SOURCE_DIR exactly as ContractCensusValidation / HeckeExtArchitectureValidation /
 * ConsumerMigrationValidation do (CMake-baked, then /proc/self/exe walk-up,
 * then a cwd walk-up), and print the vacuous-arm text with the bracketed
 * "[SKIP]" the kit's expect-strings key on, not the bare word "SKIPPED".
 */

#include "MatrixRetirementValidation.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

// The script's own location relative to a repository root.
const char* kScriptRel = "docs/matrix-engine/tools/check_matrix_includes.sh";

bool PathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool LooksLikeRepoRoot(const std::string& dir) {
    return PathExists(dir + "/" + kScriptRel);
}

// Three independent routes, tried in order -- the same recipe as
// ConsumerMigrationValidation / ContractCensusValidation / HeckeExtArchitectureValidation:
//  (1) OC_SOURCE_DIR, baked in by CMake (exact, but stale if the tree moved);
//  (2) walking up from /proc/self/exe (survives a moved tree, not an
//      out-of-source build);
//  (3) walking up from the working directory (the in-source build runs the
//      binary from the source root; also covers a few relative parents for
//      an out-of-source build directory nested under the tree).
std::string find_repo_root() {
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

} // namespace

uint32_t MatrixRetirementValidation(void) {
    std::cout << "MatrixRetirementValidation: no direct Matrix.h / BlockFlatMatrix.h "
                 "include outside the allow-lists of "
              << kScriptRel << "\n";

    const std::string root = find_repo_root();
    if (root.empty()) {
        std::cout << "  [SKIP] " << kScriptRel
                  << " not found from the working directory, from OC_SOURCE_DIR or from "
                     "/proc/self/exe (binary run outside the checkout?). Nothing to check.\n";
        return 0;
    }

    // Quoting: both operands are literals / a fixed relative candidate, so no
    // user-controlled text reaches the shell.
    std::cout.flush(); // the script writes to the same stream
    const std::string cmd = "bash \"" + root + "/" + kScriptRel + "\" \"" + root + "\"";
    const int rc = std::system(cmd.c_str());
    if (rc == -1) {
        std::cout << "  [SKIP] could not spawn a shell to run the check.\n";
        return 0;
    }
    // std::system's encoding: 0 == the script exited 0.
    if (rc == 0) {
        std::cout << "MatrixRetirementValidation: PASSED.\n";
        return 0;
    }
    std::cout << "MatrixRetirementValidation: FAILED (see the FAIL lines above).\n";
    return 1;
}
