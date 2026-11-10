# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# OcNotebook.cmake -- the read-only notebook hooks (hecke-extensions slot 19,
# NB-0 plumbing; the content arrives with NB-2 / NB-3). Included by
# CMakeLists.txt ONLY when OC_BUILD_NOTEBOOK=ON: with the switch OFF this file
# is never read, so neither Python nor KaTeX is required or probed.
#
# Design (docs/implementation/slot18/WP-19d_notebook_design_note.md Sec. 0, 2,
# 10): the C++ side gains only `--serve` (NDJSON on stdio); the gateway is a
# stdlib-only Python 3 script living outside the C++ build; the page renders
# math with vendored KaTeX (MIT, its licence file ships alongside).
#
#   * `--serve` compile flag: OC_BUILD_NOTEBOOK=1 on cohomology_app is set by
#     CMakeLists.txt in every configuration (0 when OFF); NB-2's main.cpp
#     branch and ConsoleServeValidation key on it through BuildSwitches.h.
#   * gateway install (NB-3): notebook/gateway/oc_gateway.py + README.md (the
#     source-tree self-test test_gateway.py is not installed) -> <prefix>/share/oc-notebook/gateway; notebook/page/ ->
#     <prefix>/share/oc-notebook/page; the read-only JSON twin
#     docs/latex/cli_manual/read_only_verbs.json -> <prefix>/share/oc-notebook/.
#     The installed gateway finds all of these (and <prefix>/bin/cohomology_app)
#     relative to itself, so `python3 <prefix>/share/oc-notebook/gateway/oc_gateway.py`
#     needs no flags.
#   * vendored KaTeX (NB-3): notebook/vendor/katex/ (katex 0.19.0, LICENSE,
#     PROVENANCE.txt, SHA256SUMS) -> <prefix>/share/oc-notebook/katex.
# Python 3 is needed at install / run time of the gateway, never to compile
# the binary: a missing interpreter is reported, not fatal.

set(OC_NOTEBOOK_GATEWAY_DIR "${CMAKE_SOURCE_DIR}/notebook/gateway")
set(OC_NOTEBOOK_KATEX_DIR   "${CMAKE_SOURCE_DIR}/notebook/vendor/katex")
set(OC_NOTEBOOK_PAGE_DIR    "${CMAKE_SOURCE_DIR}/notebook/page")
set(OC_NOTEBOOK_VERBS_JSON  "${CMAKE_SOURCE_DIR}/docs/latex/cli_manual/read_only_verbs.json")
set(OC_NOTEBOOK_INSTALL_DIR "share/oc-notebook" CACHE STRING
    "Install subdirectory (under CMAKE_INSTALL_PREFIX) of the notebook gateway and assets")

find_program(OC_PYTHON3 python3)
if(OC_PYTHON3)
    message(STATUS "OC_BUILD_NOTEBOOK=ON: gateway interpreter ${OC_PYTHON3} (stdlib only; run time, not build time)")
else()
    message(STATUS "OC_BUILD_NOTEBOOK=ON: no python3 found -- the binary builds; the gateway needs Python 3 at run time")
endif()

if(EXISTS "${OC_NOTEBOOK_GATEWAY_DIR}")
    install(DIRECTORY "${OC_NOTEBOOK_GATEWAY_DIR}/"
            DESTINATION "${OC_NOTEBOOK_INSTALL_DIR}/gateway"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE
            PATTERN "test_gateway.py" EXCLUDE)
else()
    message(STATUS "OC_BUILD_NOTEBOOK=ON: notebook/gateway/ not present yet (NB-3) -- no gateway installed")
endif()

if(EXISTS "${OC_NOTEBOOK_PAGE_DIR}")
    install(DIRECTORY "${OC_NOTEBOOK_PAGE_DIR}/"
            DESTINATION "${OC_NOTEBOOK_INSTALL_DIR}/page")
else()
    message(STATUS "OC_BUILD_NOTEBOOK=ON: notebook/page/ not present -- no page installed")
endif()

# (slot-24 NB-FILE) the example notebooks the gateway offers read-only (--examples-dir, default <install>/examples).
if(EXISTS "${CMAKE_SOURCE_DIR}/notebook/examples/notebook")
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/notebook/examples/notebook/"
            DESTINATION "${OC_NOTEBOOK_INSTALL_DIR}/examples"
            FILES_MATCHING PATTERN "*.oc")
endif()

if(EXISTS "${OC_NOTEBOOK_VERBS_JSON}")
    install(FILES "${OC_NOTEBOOK_VERBS_JSON}" DESTINATION "${OC_NOTEBOOK_INSTALL_DIR}")
endif()

if(EXISTS "${OC_NOTEBOOK_KATEX_DIR}")
    install(DIRECTORY "${OC_NOTEBOOK_KATEX_DIR}/"
            DESTINATION "${OC_NOTEBOOK_INSTALL_DIR}/katex")
else()
    message(STATUS "OC_BUILD_NOTEBOOK=ON: notebook/vendor/katex/ not present yet (NB-3) -- no KaTeX installed")
endif()

install(TARGETS cohomology_app RUNTIME DESTINATION bin)
# (R1) the alias of the pre-rename name, beside the installed binary
install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink hacs \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/cohomology_app\")")
