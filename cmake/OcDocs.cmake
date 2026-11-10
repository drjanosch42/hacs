# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# OcDocs.cmake -- the LaTeX documentation targets (hecke-extensions slot 19,
# NB-0). Included by CMakeLists.txt ONLY when OC_BUILD_DOCS=ON: with the switch
# OFF this file is never read, so no TeX program is required or probed.
#
#   make docs                   all documents below
#   make docs_<name>            one document
#
# The PDFs land in OC_DOCS_OUTPUT_DIR (default <build>/docs-pdf; git-ignored
# for the in-source build), never next to the sources: every target passes
# that directory to the existing docs/latex/build_*.sh script, which runs the
# pdflatex / bibtex / makeindex passes and FAILS on an undefined reference or
# citation. The targets are not part of `all` -- building the binary never
# runs TeX; `make docs` is the opt-in.

find_program(OC_PDFLATEX pdflatex)
find_program(OC_BIBTEX bibtex)
find_program(OC_MAKEINDEX makeindex)
find_program(OC_BASH bash)
if(NOT OC_PDFLATEX OR NOT OC_BIBTEX OR NOT OC_MAKEINDEX OR NOT OC_BASH)
    message(FATAL_ERROR
        "OC_BUILD_DOCS=ON needs pdflatex, bibtex, makeindex and bash "
        "(found: pdflatex='${OC_PDFLATEX}' bibtex='${OC_BIBTEX}' "
        "makeindex='${OC_MAKEINDEX}' bash='${OC_BASH}'). "
        "Install a TeX distribution or configure with -DOC_BUILD_DOCS=OFF.")
endif()

set(OC_DOCS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs-pdf" CACHE PATH
    "Output directory of the OC_BUILD_DOCS targets (never inside docs/)")

set(OC_LATEX_DIR "${CMAKE_SOURCE_DIR}/docs/latex")

# name -> the build script of docs/latex/ that knows the document's passes.
set(OC_DOCS_SCRIPTED
    "cli_manual:build_cli_manual.sh"
    "gl2_contract:build_gl2_contract.sh"
    "hida_hecke_algebra:build_hida_hecke_algebra.sh"
    "hecke_examples:build_hecke_examples.sh"
    "loewy_and_galois:build_loewy_and_galois.sh"
    "l37_loewy4:build_l37_loewy4.sh"
)

add_custom_target(docs)
foreach(entry IN LISTS OC_DOCS_SCRIPTED)
    string(REPLACE ":" ";" pair "${entry}")
    list(GET pair 0 name)
    list(GET pair 1 script)
    # (R1 / X1) a document whose build script is not in the tree (a trimmed release tree) defines no target.
    if(NOT EXISTS "${OC_LATEX_DIR}/${script}")
        message(STATUS "OC_BUILD_DOCS=ON: ${script} not present -- no docs_${name} target")
        continue()
    endif()
    add_custom_target(docs_${name}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OC_DOCS_OUTPUT_DIR}"
        COMMAND "${OC_BASH}" "${OC_LATEX_DIR}/${script}" "${OC_DOCS_OUTPUT_DIR}"
        WORKING_DIRECTORY "${OC_LATEX_DIR}"
        COMMENT "pdflatex: docs/latex/${script} -> ${OC_DOCS_OUTPUT_DIR}"
        VERBATIM)
    add_dependencies(docs docs_${name})
endforeach()

if(EXISTS "${OC_LATEX_DIR}/hecke_extensions.tex")
    # hecke_extensions.tex [E] has no build script: its bibliography is an inline
    # thebibliography, so three pdflatex passes settle every reference.
    set(_oc_pdflatex "${OC_PDFLATEX}" -interaction=nonstopmode -halt-on-error
        "-output-directory=${OC_DOCS_OUTPUT_DIR}")
    add_custom_target(docs_hecke_extensions
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OC_DOCS_OUTPUT_DIR}"
        COMMAND ${_oc_pdflatex} hecke_extensions.tex
        COMMAND ${_oc_pdflatex} hecke_extensions.tex
        COMMAND ${_oc_pdflatex} hecke_extensions.tex
        WORKING_DIRECTORY "${OC_LATEX_DIR}"
        COMMENT "pdflatex: docs/latex/hecke_extensions.tex -> ${OC_DOCS_OUTPUT_DIR}"
        VERBATIM)
    add_dependencies(docs docs_hecke_extensions)
endif()

message(STATUS "OC_BUILD_DOCS=ON: `make docs` -> ${OC_DOCS_OUTPUT_DIR} (pdflatex: ${OC_PDFLATEX})")
