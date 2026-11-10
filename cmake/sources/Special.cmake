# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Non-cohomology_app sources, compile flags and notebook-conditional sources of slots 9..25 (S24 CL-10, CL-10b, S25 FOLD; slot order kept).
if(OC_BUILD_NOTEBOOK)
    target_sources(cohomology_app PRIVATE
        src/hacs/Console/Bindings/ServeJson.cpp
        src/hacs/Console/Bindings/PrintJson.cpp
        src/hacs/Console/Bindings/Serve.cpp
    )
    set_source_files_properties(src/hacs/Console/Bindings/Serve.cpp
        PROPERTIES COMPILE_DEFINITIONS "OC_BUILD_CONFIG=\"$<CONFIG>\"")
endif()
if(OC_BUILD_NOTEBOOK)
    target_sources(cohomology_app PRIVATE
        src/hacs/Console/Bindings/ServeAtlas.cpp
    )
    set_source_files_properties(src/hacs/Console/Bindings/ServeAtlas.cpp
        PROPERTIES COMPILE_DEFINITIONS "OC_BUILD_CONFIG=\"$<CONFIG>\"")
endif()
target_sources(ModArithmetic PRIVATE src/ModArithmetic/Binary/Ring2k.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/ModArithmetic/Binary/Ring2k.cpp PROPERTIES
        COMPILE_OPTIONS "-ftree-vectorize")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(SOURCE src/ModArithmetic/Binary/Ring2k.cpp APPEND PROPERTY COMPILE_OPTIONS "-fvect-cost-model=dynamic")
endif()
target_sources(ModArithmetic PRIVATE src/ModArithmetic/Binary/TritMatrix.cpp)
target_sources(NumberField PRIVATE
    src/hacs/NumberField/CyclicCubic.cpp
)
target_sources(ModArithmetic PRIVATE src/ModArithmetic/Binary/Ring2kR5.cpp)
target_sources(ModArithmetic PRIVATE src/ModArithmetic/Binary/Ring3k.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/ModArithmetic/Binary/Ring3k.cpp PROPERTIES
        COMPILE_OPTIONS "-ftree-vectorize")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_property(SOURCE src/ModArithmetic/Binary/Ring3k.cpp APPEND PROPERTY COMPILE_OPTIONS "-fvect-cost-model=dynamic")
endif()

# ---- Slot 24 (folded at the slot-24 close, S24 CL-10b; the former cmake/slot24/<pkg>.cmake, slot order kept) ----
# PAR-2K (docs/implementation/slot24/PAR2K_report.md): src/ModArithmetic/Binary/SmithPar.cpp (the private Smith pool and the
# HECKE_DESCENT_SMITH_MT lever) joins the ModArithmetic library; it needs Threads (found before this file is included).
target_sources(ModArithmetic PRIVATE src/ModArithmetic/Binary/SmithPar.cpp)
target_link_libraries(ModArithmetic PUBLIC Threads::Threads)

# PAR-GPU (docs/implementation/slot24/PARGPU_report.md): the opt-in device-resident 2-adic Smith LU.
# ModArithmetic: the device-operations table and its counters (no CUDA). EngineCuda: the device side (ring2k_device.cu), or in a
# CPU-only build (OC_ENABLE_CUDA=OFF) its throwing stub CudaRing2kStub.cpp. Device code stays in EngineCuda.
target_sources(ModArithmetic PRIVATE src/ModArithmetic/Binary/Ring2kDevice.cpp)
if(OC_ENABLE_CUDA)
    target_sources(EngineCuda PRIVATE src/Cuda/ring2k_device.cu)
else()
    target_sources(EngineCuda PRIVATE src/Cuda/CudaRing2kStub.cpp)
endif()

# NB-TOURFIX: IntPolyFactor.cpp (Yun + Zassenhaus over Z, exact and verified) joins the STL-only NumberField target.
target_sources(NumberField PRIVATE
    src/hacs/NumberField/IntPolyFactor.cpp
)

# ---- Slot 25 (folded at the slot-25 close, S25 FOLD; the former cmake/slot25/<pkg>.cmake, slot order kept) ----
# HeckeZFreeBlock (EXZ2-P2, docs/implementation/slot25/EXZ2P2_report.md): the RNS kernels (garner_step) are word-only engine code: they
# go into the ModArithmetic library (KernelBitIdentityValidation, which lives there, compares their three bodies); the lanes, the bound,
# the CRT and the validation into cohomology_app (Integral.cmake).
target_sources(ModArithmetic PRIVATE
    src/ModArithmetic/RnsKernels.cpp
)

# NbConfirm (NBCONFIRM S3/S5; docs/console/NBCONFIRM_design.md, docs/console/NBCONFIRM_S3_report.md).
# The console cost models, embedded at CONFIGURE time (S2 brief item 1): cost_models.tsv is read into one raw string literal of a
# generated ConsoleCore translation unit under the build dir.  The TSV stays the single source; nothing is read at run time (a
# Termux binary needs no source tree), and an edit of the TSV re-runs the configure step.
set(OC_COST_MODELS_TSV ${CMAKE_SOURCE_DIR}/src/hacs/Console/cost/cost_models.tsv)
file(READ ${OC_COST_MODELS_TSV} OC_COST_MODELS_TEXT)
string(FIND "${OC_COST_MODELS_TEXT}" ")OCCOSTTSV\"" OC_COST_MODELS_DELIM)
if(NOT OC_COST_MODELS_DELIM EQUAL -1)
    message(FATAL_ERROR "cost_models.tsv contains the raw-string delimiter )OCCOSTTSV\"")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${OC_COST_MODELS_TSV})
configure_file(${CMAKE_SOURCE_DIR}/cmake/CostModelsEmbedded.cpp.in
               ${CMAKE_BINARY_DIR}/generated/nbconfirm/CostModelsEmbedded.cpp @ONLY)

target_sources(ConsoleCore PRIVATE
    src/hacs/Console/Core/CostModel.cpp
    src/hacs/Console/Core/Confirm.cpp
    # (NBCONFIRM S5) the cooperative cancel: the request flag, the points' poll, the test interlock
    src/hacs/Console/Core/Cancel.cpp
    ${CMAKE_BINARY_DIR}/generated/nbconfirm/CostModelsEmbedded.cpp
)
