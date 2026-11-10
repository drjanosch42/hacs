# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
#
# (S26 INTEG, R1 audit S1) Writes generated/OcBuildInfo.cpp from cmake/OcBuildInfo.cpp.in.  Run with `cmake -P` at
# configure time AND on every build (target oc_buildinfo), so `hacs --version` names the commit that was BUILT, not the one
# that was configured.  configure_file rewrites the output only when its content changes, so an unchanged HEAD
# recompiles nothing.  Inputs (-D): OC_SRC, OC_IN, OC_OUT, OC_GIT (may be empty), OC_HACS_VERSION, OC_CONTRACT_VERSIONS.
#   - The sha is taken only if the source directory IS the top level of its git checkout (a tarball unpacked inside
#     another checkout prints `unknown`, not that checkout's HEAD).
#   - `--version` appends `-dirty` when a TRACKED file differs from HEAD (untracked files are ignored).  The full sha of
#     GitCommit() (the `commit` key of records; metadata, redacted from every digest) carries no marker.
set(OC_GIT_COMMIT_VALUE "unknown")
set(OC_DIRTY "")
if(OC_GIT AND EXISTS "${OC_GIT}")
    execute_process(COMMAND "${OC_GIT}" rev-parse --show-toplevel WORKING_DIRECTORY "${OC_SRC}"
        OUTPUT_VARIABLE top ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE rc)
    if(rc EQUAL 0)
        get_filename_component(top_real "${top}" REALPATH)
        get_filename_component(src_real "${OC_SRC}" REALPATH)
        if(top_real STREQUAL src_real)
            execute_process(COMMAND "${OC_GIT}" rev-parse HEAD WORKING_DIRECTORY "${OC_SRC}"
                OUTPUT_VARIABLE head ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE rc2)
            if(rc2 EQUAL 0 AND NOT head STREQUAL "")
                set(OC_GIT_COMMIT_VALUE "${head}")
                execute_process(COMMAND "${OC_GIT}" status --porcelain --untracked-files=no WORKING_DIRECTORY "${OC_SRC}"
                    OUTPUT_VARIABLE st ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE rc3)
                if(rc3 EQUAL 0 AND NOT st STREQUAL "")
                    set(OC_DIRTY "-dirty")
                endif()
            endif()
        endif()
    endif()
endif()
string(SUBSTRING "${OC_GIT_COMMIT_VALUE}" 0 12 OC_GIT_COMMIT_SHORT)
set(OC_GIT_COMMIT_SHORT "${OC_GIT_COMMIT_SHORT}${OC_DIRTY}")
configure_file("${OC_IN}" "${OC_OUT}" @ONLY)
