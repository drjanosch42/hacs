// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// hecke-extensions phase E1s (WP-3) -- HeckeExtArchitectureValidation.
// See the header for what it gates.  It runs
// docs/matrix-engine/tools/check_heckeext_includes.sh over the checkout, the
// same std::system pattern MatrixRetirementValidation uses for
// check_matrix_includes.sh.

#include "HeckeExtArchitectureValidation.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

const char* kScriptRel = "docs/matrix-engine/tools/check_heckeext_includes.sh";

bool PathExists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

bool LooksLikeRepoRoot(const std::string& dir) {
    return PathExists(dir + "/" + kScriptRel) && PathExists(dir + "/src/hacs");
}

// The two independent routes of ConsumerMigrationValidation: OC_SOURCE_DIR as
// baked in by CMake (exact, but stale if the tree moved) and walking up from
// /proc/self/exe (survives a move, but not an out-of-source build), then the
// working directory (the in-source build runs the binary from the source root).
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

} // namespace

uint32_t HeckeExtArchitectureValidation(void) {
    std::cout << "HeckeExtArchitectureValidation: the HeckeExt/ one-way include guard "
                 "(" << kScriptRel << ").\n";

    const std::string root = FindRepoRoot();
    if (root.empty()) {
        std::cout << "  SKIPPED: " << kScriptRel
                  << " not found from the working directory, from OC_SOURCE_DIR or from "
                     "/proc/self/exe (binary run outside the checkout?). Nothing to check.\n";
        return 0u;
    }

    std::cout.flush();  // the script writes to the same stream
    const std::string cmd = "bash \"" + root + "/" + kScriptRel + "\" \"" + root + "\"";
    const int rc = std::system(cmd.c_str());
    if (rc == -1) {
        std::cout << "  SKIPPED: could not spawn a shell to run the check.\n";
        return 0u;
    }
    if (rc == 0) {
        std::cout << "HeckeExtArchitectureValidation: PASSED.\n";
        return 0u;
    }
    std::cout << "HeckeExtArchitectureValidation: FAILED (see the FAIL lines above).\n";
    return 1u;
}
