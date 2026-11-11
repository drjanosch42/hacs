# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# Set the minimum required version of CMake.
cmake_minimum_required(VERSION 3.18)

# Out-of-source builds only (repository layout, owner ruling Q3 of 2026-11-09,
# docs/hacs/LAYOUT_plan.md): the sources live under src/, every build goes to a
# separate directory (`cmake -B build`, or a preset). An in-source configure is
# refused before project(), so no toolchain is probed.
get_filename_component(_oc_src_real "${CMAKE_SOURCE_DIR}" REALPATH)
get_filename_component(_oc_bin_real "${CMAKE_BINARY_DIR}" REALPATH)
if(_oc_src_real STREQUAL _oc_bin_real)
    message(FATAL_ERROR
        "In-source builds are not supported. Configure into a separate directory:\n"
        "    cmake -B build && cmake --build build -j\n"
        "(or `cmake --preset <name>`). Remove the CMakeCache.txt and CMakeFiles/ "
        "this attempt left in the source directory first.")
endif()
unset(_oc_src_real)
unset(_oc_bin_real)

# =====================================================================
# Build switches (hecke-extensions slot 19, NB-0; owner decisions 2026-10-03,
# docs/implementation/slot18/WP-19d_notebook_design_note.md Sec. 0 and 10)
# =====================================================================
# Three independent options, all decided BEFORE project() so that an OFF
# switch never probes its toolchain:
#   OC_ENABLE_CUDA     ON  (default): the CUDA layer EngineCuda is compiled with
#                      nvcc against the CUDA toolkit, the entry point main.cpp is a host-compiler TU.
#                      OFF: a CPU-only build -- the CUDA language is never
#                      enabled, CUDAToolkit is never searched, no nvcc / libcudart
#                      is needed. EngineCuda is then built from its CPU stub
#                      (src/Cuda/CudaStub.cpp: available() == false),
#                      the entry point (main.cpp, host-compiled in both builds), and every device validation prints its
#                      [SKIP] line. ENGINE_DEVICE=cpu -- the golden path and the
#                      default everywhere -- is the only device path.
#   OC_BUILD_DOCS      OFF (default): no TeX toolchain is required or probed.
#                      ON: pdflatex targets for the manual and the contracts
#                      (`make docs`; cmake/OcDocs.cmake).
#   OC_BUILD_NOTEBOOK  OFF (default): nothing of the notebook is compiled,
#                      required or probed. ON: the `--serve` compile flag
#                      (OC_BUILD_NOTEBOOK=1 in cohomology_app), the gateway
#                      install and the vendored KaTeX (cmake/OcNotebook.cmake).
#                      Placeholder plumbing until NB-2 / NB-3 land.
#   OC_ENABLE_LEGACY   OFF (default): the retired routes (the `legacy/`
#                      directories, cmake/legacy.cmake; S24 CL-L0, owner ruling 1
#                      of 2026-10-30, docs/implementation/slot24/CLEANUP_plan_1030.md
#                      Sec. 2) are not compiled; a request for one is REFUSED
#                      (ENGINE_BACKEND=legacy, a retired lever's X=0: stderr
#                      `REFUSED ...: compiled out (OC_ENABLE_LEGACY=OFF)`, exit 1),
#                      a legacy-gated validation prints `[SKIP] <name>:
#                      OC_ENABLE_LEGACY=OFF`, a gated arm `SKIP-ARM ...`.
#                      ON: the retired routes compile and run as before their
#                      retirement. The macro is defined 0/1 on EVERY target that
#                      sees the engine headers (it will guard class members: ODR).
# The reference machine's configuration is the `reference` preset of
# CMakePresets.json (CUDA ON, sm_120, docs ON, notebook ON). A switch never
# changes a value: golden, the digests and the registered-validation count
# (names are always registered; excluded bodies print `[SKIP] <name>:
# OC_BUILD_<X>=OFF` and return 0) are the same in every combination.
option(OC_ENABLE_CUDA "Build the CUDA device layer (OFF: CPU-only build, no CUDA toolkit needed)" ON)
option(OC_BUILD_DOCS "Add the pdflatex targets for the manual and the contracts (needs a TeX toolchain)" OFF)
option(OC_BUILD_NOTEBOOK "Build the notebook hooks: --serve, gateway install, vendored KaTeX (NB-2/NB-3)" OFF)
option(OC_ENABLE_LEGACY "Compile the retired legacy/ routes (OFF: they are compiled out and a request for one is refused)" OFF)

# (R1 / X1) The release version lives in the tracked VERSION file (one line, X.Y.Z); a change re-configures.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" OC_HACS_VERSION LIMIT_COUNT 1)
# VERSION must stay purely numeric (project() rejects e.g. 0.1.0-rc.1; an rc label goes in the tag / directory name only).
if(NOT OC_HACS_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION must be X.Y.Z (digits only), got '${OC_HACS_VERSION}'")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")

# Define the project, enabling C++ (CXX) and -- unless CPU-only -- CUDA.
if(OC_ENABLE_CUDA)
    project(HACS VERSION ${OC_HACS_VERSION} LANGUAGES CXX CUDA)
else()
    project(HACS VERSION ${OC_HACS_VERSION} LANGUAGES CXX)
endif()
message(STATUS "OC_ENABLE_CUDA=${OC_ENABLE_CUDA} OC_BUILD_DOCS=${OC_BUILD_DOCS} OC_BUILD_NOTEBOOK=${OC_BUILD_NOTEBOOK} OC_ENABLE_LEGACY=${OC_ENABLE_LEGACY} host=${CMAKE_SYSTEM_PROCESSOR}")
include(cmake/termux/MallocTrim.cmake)  # no-op malloc_trim where the libc lacks it (bionic)

# =====================================================================
# Toolchain / build-configuration policy  (matrix-engine Phase 0)
# =====================================================================
#
# Compiler of record on the reference machine:
#   * C++  : GCC 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3), selected as /usr/bin/c++
#   * CUDA : nvcc release 13.0, V13.0.48
#   * GPU  : NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 (sm_120)
# (docs/matrix-engine/01-matrix-substrate-audit.md was measured with GCC 15;
#  the build here pins whatever CMake selects, which is GCC 11.4 on this box.)
#
# A build directory's CMakeCache.txt pins the CMAKE_BUILD_TYPE / CMAKE_CUDA_*
# values it was FIRST configured with; the defaults below only take effect on a
# *fresh* configure (a new build directory, or one with CMakeCache.txt and
# CMakeFiles/ removed). Build types and presets: BUILDING.md.

# --- Available configurations -----------------------------------------
# Debug / Release / RelWithDebInfo / MinSizeRel are CMake's built-ins.
# Two custom configurations are added:
#   Bench : -O3 -march=native -DNDEBUG           (timing runs; asserts OFF)
#   Check : -O1 -g -fsanitize=address,undefined  (asserts ON, no -DNDEBUG)
# A custom configuration needs BOTH CMAKE_CXX_FLAGS_<CONFIG> and
# CMAKE_CUDA_FLAGS_<CONFIG>: an nvcc TU (cuda_kernels.cu) inherits neither the
# CXX flags nor -march=native (which nvcc only accepts as
# -Xcompiler=-march=native).
set(OC_CONFIG_TYPES Debug Release RelWithDebInfo MinSizeRel Bench Check CheckThread)

get_property(OC_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(OC_IS_MULTI_CONFIG)
    # Multi-config generators (Visual Studio, Ninja Multi-Config) ignore
    # CMAKE_BUILD_TYPE; they take the list of configurations instead.
    set(CMAKE_CONFIGURATION_TYPES "${OC_CONFIG_TYPES}" CACHE STRING
        "Available build configurations" FORCE)
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
            "Build configuration" FORCE)
        message(STATUS "CMAKE_BUILD_TYPE was not set; defaulting to RelWithDebInfo.")
    endif()
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${OC_CONFIG_TYPES})
    message(STATUS "Build configuration: ${CMAKE_BUILD_TYPE}")
endif()

# --- Custom configuration: Bench --------------------------------------
set(CMAKE_CXX_FLAGS_BENCH "-O3 -march=native -DNDEBUG" CACHE STRING
    "C++ flags for the Bench configuration" FORCE)
set(CMAKE_CUDA_FLAGS_BENCH "-O3 -Xcompiler=-march=native -DNDEBUG" CACHE STRING
    "CUDA flags for the Bench configuration" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_BENCH "" CACHE STRING
    "Executable linker flags for the Bench configuration" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_BENCH "" CACHE STRING
    "Shared linker flags for the Bench configuration" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_BENCH "" CACHE STRING
    "Static linker flags for the Bench configuration" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_BENCH "" CACHE STRING
    "Module linker flags for the Bench configuration" FORCE)

# --- Custom configuration: Check --------------------------------------
# Deliberately does NOT define NDEBUG: ENGINE_DEBUG_CHECK (and any remaining
# assert) must be live here. See src/ModArithmetic/EngineCheck.h.
set(CMAKE_CXX_FLAGS_CHECK "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined" CACHE STRING
    "C++ flags for the Check (sanitizer) configuration" FORCE)
# NOTE (nvcc gotcha): nvcc splits a -Xcompiler argument on COMMAS, so
# `-Xcompiler=-fsanitize=address,undefined` reaches gcc as two arguments
# (`-fsanitize=address` and a bare `undefined`, which gcc then treats as an
# input file: "cannot specify -o with -c ... with multiple files"). The two
# sanitizers must therefore be passed as two separate, comma-free options.
set(CMAKE_CUDA_FLAGS_CHECK "-O1 -g -Xcompiler=-fno-omit-frame-pointer -Xcompiler=-fsanitize=address -Xcompiler=-fsanitize=undefined" CACHE STRING
    "CUDA flags for the Check (sanitizer) configuration" FORCE)
# The LINK step, unlike the CUDA compile step, is driven by the C++ driver
# (CMake picks CXX as cohomology_app's linker language -- there is no separable
# device code), so the sanitizer runtimes are requested with plain gcc syntax
# here, NOT through -Xcompiler.
set(CMAKE_EXE_LINKER_FLAGS_CHECK "-fsanitize=address,undefined" CACHE STRING
    "Executable linker flags for the Check configuration" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_CHECK "-fsanitize=address,undefined" CACHE STRING
    "Shared linker flags for the Check configuration" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_CHECK "" CACHE STRING
    "Static linker flags for the Check configuration" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_CHECK "-fsanitize=address,undefined" CACHE STRING
    "Module linker flags for the Check configuration" FORCE)

# --- Custom configuration: CheckThread ---------------------------------
# ThreadSanitizer twin of Check (plan Phase 4 mitigation: "ASan/TSan Check
# builds"). TSan cannot be combined with ASan, hence a separate configuration.
# NDEBUG deliberately undefined, as in Check. Use for the engine's thread pool
# and the determinism validations under ENGINE_THREADS > 1.
set(CMAKE_CXX_FLAGS_CHECKTHREAD "-O1 -g -fno-omit-frame-pointer -fsanitize=thread" CACHE STRING
    "C++ flags for the CheckThread (ThreadSanitizer) configuration" FORCE)
set(CMAKE_CUDA_FLAGS_CHECKTHREAD "-O1 -g -Xcompiler=-fno-omit-frame-pointer -Xcompiler=-fsanitize=thread" CACHE STRING
    "CUDA flags for the CheckThread configuration" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_CHECKTHREAD "-fsanitize=thread" CACHE STRING
    "Executable linker flags for the CheckThread configuration" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_CHECKTHREAD "-fsanitize=thread" CACHE STRING
    "Shared linker flags for the CheckThread configuration" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_CHECKTHREAD "" CACHE STRING
    "Static linker flags for the CheckThread configuration" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_CHECKTHREAD "-fsanitize=thread" CACHE STRING
    "Module linker flags for the CheckThread configuration" FORCE)

mark_as_advanced(
    CMAKE_CXX_FLAGS_BENCH CMAKE_CUDA_FLAGS_BENCH
    CMAKE_EXE_LINKER_FLAGS_BENCH CMAKE_SHARED_LINKER_FLAGS_BENCH
    CMAKE_STATIC_LINKER_FLAGS_BENCH CMAKE_MODULE_LINKER_FLAGS_BENCH
    CMAKE_CXX_FLAGS_CHECK CMAKE_CUDA_FLAGS_CHECK
    CMAKE_EXE_LINKER_FLAGS_CHECK CMAKE_SHARED_LINKER_FLAGS_CHECK
    CMAKE_STATIC_LINKER_FLAGS_CHECK CMAKE_MODULE_LINKER_FLAGS_CHECK
    CMAKE_CXX_FLAGS_CHECKTHREAD CMAKE_CUDA_FLAGS_CHECKTHREAD
    CMAKE_EXE_LINKER_FLAGS_CHECKTHREAD CMAKE_SHARED_LINKER_FLAGS_CHECKTHREAD
    CMAKE_STATIC_LINKER_FLAGS_CHECKTHREAD CMAKE_MODULE_LINKER_FLAGS_CHECKTHREAD
)

# --- CUDA architecture -------------------------------------------------
# Was 75 (Turing) in the stale in-source cache; the reference GPU is an
# RTX 5070 Ti (cc 12.0) and nvcc 13.0 accepts sm_120 / compute_120.
# Overridable with -DOC_CUDA_ARCHITECTURES=... on a fresh configure.
# Jetson Orin (aarch64): -DOC_CUDA_ARCHITECTURES=87 (preset jetson-orin).
# Unused (and not forced into the cache) in a CPU-only build.
set(OC_CUDA_ARCHITECTURES "120" CACHE STRING
    "CUDA architectures to compile for (CMAKE_CUDA_ARCHITECTURES value)")
if(OC_ENABLE_CUDA)
    # FORCE so that the cache entry CMake itself seeds (75, from the compiler probe /
    # an older in-source cache) does not silently disagree with what is compiled.
    set(CMAKE_CUDA_ARCHITECTURES ${OC_CUDA_ARCHITECTURES} CACHE STRING
        "CUDA architectures (driven by OC_CUDA_ARCHITECTURES)" FORCE)
endif()

# --- Warnings ----------------------------------------------------------
# Applied per target below. Guarded by COMPILE_LANGUAGE:CXX so nvcc never
# sees the raw GCC warning switches.
set(OC_CXX_WARNINGS -Wall -Wextra)

# --- Global Settings ---
# Set the C++ standard for the entire project for portability.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Small / stripped binary (slot-24 BUILD-SLIM, F1 / F3) --------------------
# OC_SMALL_BINARY: compile C++ with -ffunction-sections -fdata-sections and link with -Wl,--gc-sections (plus
# --icf=safe where the linker has it: gold, lld). AUTO (default) = ON for the Release and MinSizeRel configurations
# only, OFF for RelWithDebInfo / Bench / Check / Debug, so that the standing-gate timings of the default binary stay
# comparable. No value can change (section layout and identical-code folding do not alter code); folded are only
# functions whose address is never taken (--icf=safe). CUDA TUs (EngineCuda) are not given the section flags.
# OC_STRIP_BINARY: link with -s (no symbols, no DWARF); the `termux` / `aarch64-cpu` presets turn it ON. The same
# result is `cmake --build <dir> && cmake --install <dir> --strip` / `make install/strip`.
set(OC_SMALL_BINARY "AUTO" CACHE STRING "Section gc + ICF link of the C++ code: AUTO (Release/MinSizeRel), ON, OFF")
set_property(CACHE OC_SMALL_BINARY PROPERTY STRINGS AUTO ON OFF)
option(OC_STRIP_BINARY "Strip symbols and debug info from cohomology_app at link time (-s)" OFF)
set(OC_SMALL_BINARY_EFFECTIVE OFF)
if(OC_SMALL_BINARY STREQUAL "ON" OR (OC_SMALL_BINARY STREQUAL "AUTO" AND CMAKE_BUILD_TYPE MATCHES "^(Release|MinSizeRel)$"))
    set(OC_SMALL_BINARY_EFFECTIVE ON)
endif()
if(OC_SMALL_BINARY_EFFECTIVE)
    include(CheckCXXSourceCompiles)
    set(OC_ICF_SRC "int main() { return 0; }")
    # 1) the default linker (lld on clang / Termux, or a bfd that has no --icf: the check then fails);
    set(CMAKE_REQUIRED_LINK_OPTIONS "-Wl,--gc-sections;-Wl,--icf=safe")
    check_cxx_source_compiles("${OC_ICF_SRC}" OC_LINKER_HAS_ICF)
    set(OC_ICF_LINKER_FLAGS "")
    if(OC_LINKER_HAS_ICF)
        set(OC_ICF_LINKER_FLAGS "-Wl,--icf=safe")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # 2) GNU ld.bfd has no ICF: use gold for this link only, when installed and when it accepts the flag.
        set(CMAKE_REQUIRED_LINK_OPTIONS "-fuse-ld=gold;-Wl,--gc-sections;-Wl,--icf=safe")
        check_cxx_source_compiles("${OC_ICF_SRC}" OC_GOLD_HAS_ICF)
        if(OC_GOLD_HAS_ICF)
            set(OC_ICF_LINKER_FLAGS "-fuse-ld=gold;-Wl,--icf=safe")
        endif()
    endif()
    unset(CMAKE_REQUIRED_LINK_OPTIONS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections> $<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>)
    add_link_options(-Wl,--gc-sections ${OC_ICF_LINKER_FLAGS})
    message(STATUS "OC_SMALL_BINARY: section gc on, ICF flags: '${OC_ICF_LINKER_FLAGS}'")
endif()
if(OC_STRIP_BINARY)
    add_link_options(-s)
    message(STATUS "OC_STRIP_BINARY=ON: cohomology_app is linked stripped (-s)")
endif()
# Strict IEEE rounding on every target and architecture (CL-12, audit D section 4): GCC defaults to -ffp-contract=fast and
# contracts a*b+c to FMA on aarch64 (x86-64 only with -mfma / -march=native), which LTO or inlining could spread. It is a
# safety measure: the value paths are integer, and the one certified FP filter (Hecke/Predicates.cpp) already forces its
# products through volatile stores. Host C++ only; CUDA device-code FP semantics are untouched. No x86-64 effect as built.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

# Find the CUDA toolkit. This is necessary for compiling .cu files. Never
# searched in a CPU-only build (OC_ENABLE_CUDA=OFF).
if(OC_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# --- ModArithmetic Library ---
# This section defines the helper library for modular arithmetic and matrices.

add_library(ModArithmetic
    # Source files for the library
    src/ModArithmetic/ModArithmetic.cpp
    src/ModArithmetic/ModArithmeticValidation.cpp
    src/ModArithmetic/Matrix.cpp
    src/ModArithmetic/MatrixValidation.cpp
    src/ModArithmetic/FlatMatrix.cpp
    src/ModArithmetic/FlatMatrixValidation.cpp
    # (S24 CL-L2) BlockFlatMatrix.cpp + BlockFlatMatrixValidation.cpp moved to src/ModArithmetic/legacy/
    # (cmake/legacy.cmake, OC_ENABLE_LEGACY=ON only).
    # Phase 0 -- substrate differential harness (01 Sec. 7 gap #9): compares
    # Matrix / FlatMatrix / BlockFlatMatrix against each other.
    src/ModArithmetic/SubstrateDifferentialValidation.cpp
    # Phase 1 -- rho_n (symmetric-power) validation (plan items 7.0 / 7.5):
    # bit-comparison of the new closed-form/recurrence rho_n against the
    # pre-Phase-1 reference, plus large-n structural identities.
    src/ModArithmetic/SymmetricPowerValidation.cpp
    # Phase 2 -- arithmetic primitive layer (plan D2/D3): modulus context,
    # dense buffers/views, the vectorisable kernels and the naive oracle.
    src/ModArithmetic/DenseBuf.cpp
    src/ModArithmetic/Kernels.cpp
    src/ModArithmetic/RefLinAlg.cpp
    src/ModArithmetic/ModContextValidation.cpp
    src/ModArithmetic/KernelBitIdentityValidation.cpp
    # Phase 2 -- in-buffer blocked elimination (plan D5/D5b) + its entrywise
    # validation against RefLinAlg and the legacy FlatMatrix/BlockFlatMatrix.
    src/ModArithmetic/DenseElim.cpp
    src/ModArithmetic/DenseElimValidation.cpp
    # (slot-22 BIN-1) native F_2 arithmetic on bit-packed rows (M4RI / M4RM), the opt-in
    # ENGINE_BINARY_F2 route of DenseElim.cpp; ENGINE_API.md Sec. 8.1.
    src/ModArithmetic/Binary/BitMatrix.cpp
    # Phase 3 -- the block-sparse data model (plan D3): BlockPool / BlockMat /
    # BlockBuilder (header-only), the pattern-driven block kernels and the
    # block structured elimination (plan D5) + their validations.
    src/ModArithmetic/BlockKernels.cpp
    src/ModArithmetic/BlockElim.cpp
    src/ModArithmetic/BlockMatValidation.cpp
    src/ModArithmetic/BlockEliminationValidation.cpp
    # matrix-engine Phase 2 -- arithmetic micro-benchmarks + performance smoke.
    # Lives in ModArithmetic (not Bench/) because it measures the substrate and
    # must be callable from a validation registered inside this library; see
    # src/ModArithmetic/PerformanceSmokeValidation.h.
    src/ModArithmetic/PerformanceSmokeValidation.cpp
    src/ModArithmetic/helper.cpp
    src/ModArithmetic/pch.cpp
)

# Specify the include directory. 'PUBLIC' makes it available to this library
# and any targets that link against it (like our main app).
target_include_directories(ModArithmetic
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/ModArithmetic
)

# Tell CMake to use pch.h as the precompiled header for the ModArithmetic library.
# This command automatically handles the platform-specific compiler flags.
target_precompile_headers(ModArithmetic
    PUBLIC src/ModArithmetic/pch.h
)

target_compile_options(ModArithmetic
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OC_CXX_WARNINGS}>
)

# --- MatrixEngine Library (matrix-engine Phase 4) ---
# The tape engine of plan D4: Program / Recorder, passes 1-6, the wavefront
# Executor over a persistent thread pool, the arena, the program cache and the
# panel-driven elimination driver (src/Engine/).  A separate static
# library on top of ModArithmetic (it consumes the dense and block layers and
# never modifies them); pure C++, links Threads.  ENGINE_API.md Sec. 10.
find_package(Threads REQUIRED)

add_library(MatrixEngine
    src/Engine/Passes.cpp
    src/Engine/ThreadPool.cpp
    src/Engine/Executor.cpp
    src/Engine/DeviceBackend.cpp
    src/Engine/ProgramCache.cpp
    src/Engine/TapeElim.cpp
    src/Engine/EngineRuntime.cpp
    src/Engine/EngineTrace.cpp
    src/Engine/EngineTapeValidation.cpp
    src/Engine/EngineTapeEliminationValidation.cpp
    src/Engine/EngineDeterminismCoreValidation.cpp
    src/Engine/EngineArenaValidation.cpp
    src/Engine/EngineCoreScalingValidation.cpp
    src/Engine/EngineCrossDeviceValidation.cpp
    src/Engine/EngineCudaMemoryValidation.cpp
    src/Engine/EngineDeviceTrapValidation.cpp
)

target_include_directories(MatrixEngine
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/Engine
)

# Phase 5 Stage B: the executor's device arm (DeviceBackend.cpp) is a plain
# C++ TU over the CUDA-header-free public headers of EngineCuda, so
# MatrixEngine links EngineCuda (which links CUDA::cudart).  On a machine
# without a usable device engine::cuda::available() is false and the backend
# is never constructed.
target_link_libraries(MatrixEngine PUBLIC ModArithmetic EngineCuda Threads::Threads)

# ENGINE_VERIFY_DEVICE (plan Sec. 4.3): every GPU node is re-executed on the
# CPU by the executor and the two outputs compared entrywise; a difference
# throws std::logic_error.  ~2x the device work; the validations can also
# switch it on per executor (Executor::set_verify_device).
option(ENGINE_VERIFY_DEVICE "Re-execute every GPU tape node on the CPU and compare (throws on mismatch)" OFF)
if(ENGINE_VERIFY_DEVICE)
    target_compile_definitions(MatrixEngine PUBLIC ENGINE_VERIFY_DEVICE=1)
    message(STATUS "ENGINE_VERIFY_DEVICE=ON: GPU tape nodes are re-executed on the CPU and compared.")
endif()

target_compile_options(MatrixEngine
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OC_CXX_WARNINGS}>
)

# --- HeckeCore Library (Stage S1) ---
# Design choice (documented per plan stage S1): the Hecke sources live in a
# dedicated static library target `HeckeCore` (linked into cohomology_app and
# available to future Hecke stages/tools) rather than being compiled directly
# into cohomology_app. It is pure C++ (no CUDA) and deliberately does not
# depend on ModArithmetic: Stage S1 is the exact bounded-integer layer and the
# certified FP sign filter ([C, sec:reduction], [C, sec:fpfilter] in
# docs/latex/hecke_action.tex).

add_library(HeckeCore
    src/hacs/Hecke/ExactInt.cpp
    src/hacs/Hecke/Predicates.cpp
    src/hacs/Hecke/QuadPoint.cpp
)

target_include_directories(HeckeCore
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/hacs
)

# EngineCheck.h (ENGINE_CHECK / ENGINE_DEBUG_CHECK, matrix-engine Phase 0) is a
# header-only macro/inline-function header living in the ModArithmetic source
# tree. Adding its directory to the include path does NOT make HeckeCore depend
# on the ModArithmetic *library* (nothing is linked); the S1 design statement
# above -- pure C++, no ModArithmetic link dependency -- still holds.
target_include_directories(HeckeCore
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/ModArithmetic
)

target_compile_options(HeckeCore
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OC_CXX_WARNINGS}>
)

# --- EngineCuda Library (matrix-engine Phase 5) ---
# The self-contained CUDA device layer: runtime context (device selection,
# streams, the <= 128 MB pinned staging pool, the device arena, residency
# tracking), the device twins of engine::kernels (bit-identical by
# construction, asserted by CudaBitIdentityValidation), the CUDA-graph
# capture helper, and the two registered validations. It depends on
# ModArithmetic (ModCtx / DenseBuf / Kernels for the CPU twins) and on the
# CUDA runtime; ModArithmetic does NOT depend on it (the `--bench micro
# --backend gpu` arm reaches it through a hook installed by cohomology_app,
# see src/ModArithmetic/PerformanceSmokeValidation.h).
# Only cuda_kernels.cu is an nvcc TU; every public header is CUDA-header-free
# so the Phase-4 tape can include them from plain C++ (MatrixEngine's
# DeviceBackend.cpp does, Phase 5 Stage B: MatrixEngine links EngineCuda). On a machine without a
# usable GPU at RUNTIME the layer reports engine::cuda::available() == false
# and every validation here skips (returns 0).

#
# CPU-only build (OC_ENABLE_CUDA=OFF, slot-19 NB-0): the same target name and
# public headers, but CudaCtx.cpp + cuda_kernels.cu are replaced by the plain
# C++ CudaStub.cpp (available() == false, reason "built without CUDA
# (OC_ENABLE_CUDA=OFF)"; every device entry point throws CudaError) and nothing
# links CUDA::cudart. The two validations and the bench hooks compile unchanged
# and take their existing no-device branch.

if(OC_ENABLE_CUDA)
    add_library(EngineCuda STATIC
        src/Cuda/CudaCtx.cpp
        src/Cuda/cuda_kernels.cu
        src/Cuda/CudaValidation.cpp
        src/Cuda/CudaBench.cpp
    )
else()
    add_library(EngineCuda STATIC
        src/Cuda/CudaStub.cpp
        src/Cuda/CudaValidation.cpp
        src/Cuda/CudaBench.cpp
    )
endif()

target_include_directories(EngineCuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/Cuda
)

target_link_libraries(EngineCuda PUBLIC ModArithmetic)
if(OC_ENABLE_CUDA)
    target_link_libraries(EngineCuda PUBLIC CUDA::cudart)
endif()

target_compile_options(EngineCuda
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OC_CXX_WARNINGS}>
)
if(OC_ENABLE_CUDA)
    target_compile_options(EngineCuda
        # nvcc: the same warnings for the host pass of the .cu TU
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall -Xcompiler=-Wextra>
    )
endif()

# --- ConsoleCore Library (console phase C0) ---
# The language core of the CAS-style console specified by
# docs/console/00-console-interface-plan.md v2 (Sec. 2 C8 and Sec. 9), built by
# work package WP-5 of docs/implementation/hecke_extensions_slot1_plan.md.
#
# Design choice, exactly the HeckeCore one above: ConsoleCore is a static
# library that depends on NOTHING else in the tree -- lexer, parser, evaluator,
# the Int/Bool/String/List/Record value model, the Object handle, the settings
# layer, the type/verb registry, the canonical printers, the session and the
# dumb line reader, all STL-only.  That one-way dependency is what lets the
# language be validated without constructing a single matrix, and it is checked
# by docs/matrix-engine/tools/check_console_includes.sh (run by
# ConsoleC0Validation).  Everything that needs a matrix, a polynomial or a
# pipeline object is a BINDING and arrives in phase C1, compiled directly into
# cohomology_app the way BenchDriver.cpp is.
#
# C0 adds NO kernel.cu branch: the `--console` / `--script` / `-c` CLI surface
# is phase C1, frozen by ConsoleCliPreservationValidation.

add_library(ConsoleCore STATIC
    src/hacs/Console/Core/ConsoleError.cpp
    src/hacs/Console/Core/Value.cpp
    src/hacs/Console/Core/Object.cpp
    src/hacs/Console/Core/Settings.cpp
    src/hacs/Console/Core/Lexer.cpp
    src/hacs/Console/Core/Parser.cpp
    src/hacs/Console/Core/Registry.cpp
    src/hacs/Console/Core/Printer.cpp
    src/hacs/Console/Core/Evaluator.cpp
    src/hacs/Console/Core/CoreVerbs.cpp
    src/hacs/Console/Core/Session.cpp
    src/hacs/Console/Core/LineInput.cpp
    src/hacs/Console/Core/ScriptRunner.cpp
    # Console phase C1a: the command-line surface.  It is in Core/ (plan v3
    # Sec. 9) and stays STL-only -- it touches argv, files and streams and
    # nothing of the pipeline, so check_console_includes.sh rule 1 still holds.
    src/hacs/Console/Core/ConsoleCli.cpp
)

target_include_directories(ConsoleCore
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/hacs
)

target_compile_options(ConsoleCore
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OC_CXX_WARNINGS}>
)

# --- NumberField Library (hecke-extensions slot 19, WP-NF) ---
# The seventh static target ([E] sec:lvnf, [C] sec:codemap; plan
# docs/implementation/slot19/00-slot19-plan.md "WP-NF"): the in-house
# number-field core -- own BigInt (owner ruling O3: no GMP, no CAS), exact
# rationals and Bernoulli numbers, Z[zeta_{ell^k}] exactly and mod ell^M with a
# multimodular NTT, cyclotomic units, v_pi, the Kummer local ell-th-power /
# unramifiedness tests (layer 1 and the equivariant Z/ell^2 layer), Q(alpha),
# Newton polygons and Hensel lifting. Design choice exactly the HeckeCore /
# ConsoleCore one: STL ONLY, depends on nothing else in the tree, no CUDA, no
# __int128 without a portable fallback (aarch64 / HPC). Checked by
# docs/matrix-engine/tools/check_numberfield_includes.sh, run by
# NumberFieldValidation (compiled into cohomology_app via cmake/slot19/WPNF.cmake).
add_library(NumberField STATIC
    src/hacs/NumberField/BigInt.cpp
    src/hacs/NumberField/BigRational.cpp
    src/hacs/NumberField/Ntt.cpp
    src/hacs/NumberField/Cyclotomic.cpp
    src/hacs/NumberField/Kummer.cpp
    src/hacs/NumberField/PolyAlg.cpp
    # slot-19 WP-NF2: the Coates--Wiles layer route (62-bit NTT, SHA-256, CwLayer)
    src/hacs/NumberField/Ntt64.cpp
    src/hacs/NumberField/Sha256.cpp
    src/hacs/NumberField/CwLayer.cpp
    # slot-21 T1: exact LLL (Cohen Alg. 2.6.3 / 2.6.7) and POLRED (Alg. 4.4.11 / 4.4.12)
    src/hacs/NumberField/Lll.cpp
    src/hacs/NumberField/Polred.cpp
    # slot-21 T2: multiprecision reals (MpReal) and the degree-ell Ribet field polynomial
    src/hacs/NumberField/MpReal.cpp
    src/hacs/NumberField/RibetField.cpp
    # slot-22 NF-PRANK: the pure field Q(N^{1/p}) (F_q factorisation, Dedekind, Round 2), Chevalley, the relation class group
    src/hacs/NumberField/PureField.cpp
    src/hacs/NumberField/Ambiguous.cpp
    src/hacs/NumberField/ClassGroupRel.cpp
)

target_include_directories(NumberField
    # consumers include "NumberField/<header>.h" (HeckeCore convention)
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/hacs
)

target_compile_options(NumberField
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OC_CXX_WARNINGS}>
)
# WP-NF2: CwLayer parallelises with std::thread (STL only).
target_link_libraries(NumberField PUBLIC Threads::Threads)

# --- Main Executable ---
# This section defines your main application.

# The main entry point is main.cpp, a plain host-compiler TU in BOTH builds (slot-24 BUILD-SLIM, F8: it was
# kernel.cu, an nvcc TU with no device code, plus a kernel_cpu.cpp shim for the CPU-only build). The CUDA language is
# needed only for the EngineCuda device code.
set(OC_APP_ENTRY_SOURCE src/hacs/main.cpp)

add_executable(cohomology_app
    # The main entry point: main.cpp (formerly kernel.cu).
    ${OC_APP_ENTRY_SOURCE}

    # Other source files for the executable
    src/hacs/EquivariantCochainComplexMorphismOnK.cpp
    src/hacs/EquivariantCohomologyOnK.cpp
    # (hecke-extensions slot 3, WP-5 / phase E5-2) The equivariant mapping
    # cone Cone(r) of gl2_contract.tex sec:cone (def:pair, eq:conediff,
    # lem:conerelative, eq:coneles, open:ell2, obl:coneexact(iv)-(v)):
    # generic (no Hecke include), on FlatMatrix / BlockMat / engine::blk /
    # engine::elim, with ConeCertificate.  Validation: HeckeConeValidation.
    src/hacs/EquivariantMappingCone.cpp
    # Stage S13 -- extended coefficient-regime validation (ell < 2^64).
    src/hacs/Hecke/HeckeExtendedRegimeValidation.cpp
    src/hacs/Hecke/ThetaEqSolveValidation.cpp
    src/hacs/LocalSystemCohomologyValidation.cpp
    src/hacs/LocalSystemPresets.cpp
    src/hacs/LocalSystemSimplicialComplex.cpp
    src/hacs/LocalSystemSL2Z.cpp
    src/hacs/LocalSystemUtils.cpp
    src/hacs/SimplicialCohomology.cpp
    src/hacs/ValidationRegistry.cpp

    # matrix-engine Phase 1 -- item 7.4 (vacuous-relation short-circuit).
    src/hacs/VacuousRelationValidation.cpp
    src/hacs/MatrixRetirementValidation.cpp
    # hecke-extensions slot 3, WP-10 -- the [C, <label>] / \HA{<label>} citation
    # census (docs/matrix-engine/tools/label_census.sh, [C] sec:machinery item 2).
    src/hacs/ContractCensusValidation.cpp
    # matrix-engine Phase 2 -- DenseElim on the REAL pipeline matrices (A_Cq of
    # the SL(2,Z) model) and the whole compute_cohomology() legacy-vs-engine.
    src/hacs/DenseElimPipelineValidation.cpp
    # matrix-engine Phase 3 -- static consumer-migration gate: drives the
    # build-time grep scripts in docs/matrix-engine/tools/ (raw mod_* call
    # sites, Matrix.h includes) against their explicit allow-lists.
    src/hacs/ConsumerMigrationValidation.cpp
    # matrix-engine Phase 3 -- the block model on the REAL A_Cq / W^1 / H^1 at
    # n_sym = 128, entrywise vs the legacy path, timed (no consumer migrated).
    src/hacs/BlockModelPipelineValidation.cpp
    # (S24 CL-L2) CohomologyBlockModelValidation.cpp and Hecke/HeckeThetaBlockValidation.cpp (the block model vs the
    # legacy arms) moved to src/hacs/legacy/ and Hecke/legacy/ (cmake/legacy.cmake, ON only).

    # Phase 0 -- the `--bench` CLI. A PLAIN C++ TU on purpose: kernel.cu is an
    # nvcc TU that inherits neither CMAKE_CXX_FLAGS_<CONFIG> nor
    # -march=native, so keeping the benchmark driver out of it is what makes
    # the measured numbers reflect the C++ flags under study
    # (00-implementation-plan.md, Phase 0, mechanic (ii)). kernel.cu only
    # dispatches --bench into bench::RunBench().
    src/hacs/Bench/BenchDriver.cpp

    # Hecke validations (registered in ValidationRegistry.cpp; the Hecke core
    # sources themselves live in the HeckeCore static library above).
    src/hacs/Hecke/HeckeExactIntSmokeValidation.cpp
    src/hacs/Hecke/HeckeFpFilterValidation.cpp
    src/hacs/Hecke/HeckeTessellationSmokeValidation.cpp
    # Stage S2V -- independent validation of Stage S2 (see the file header
    # for the independence statement).
    src/hacs/Hecke/HeckeTessellationValidation.cpp

    # Stage S2 design choice (documented per plan): SL2ZTessellationTables is
    # compiled into cohomology_app, NOT into HeckeCore, because it consumes
    # the SimplicialComplex model and the relation/generator seed table of
    # LocalSystemSL2Z.cpp (invariant I3 single source of truth), both of which
    # depend on ModArithmetic -- HeckeCore deliberately does not. It still
    # links against HeckeCore for the exact-integer layer (Mat22, QuadPoint).
    src/hacs/Hecke/SL2ZTessellationTables.cpp

    # Stage S3 design choice (documented per plan, same reasoning as S2's
    # SL2ZTessellationTables above): HeckeCosetTables depends on FlatMatrix /
    # ComputeSymmetricPowerMatrixFM (ModArithmetic) for the rho_n(.) blocks
    # ([C, prop:alpha]/[C, prop:omega]/[C, prop:blockmatrices]), so it is
    # compiled into cohomology_app, NOT into HeckeCore. It links against
    # HeckeCore for hecke::Mat22 (Stage S1's exact-integer layer).
    src/hacs/Hecke/HeckeCosetTables.cpp
    src/hacs/Hecke/HeckeCosetTablesSelfCheckValidation.cpp

    # Stage S3V -- INDEPENDENT validation of Stage S3 against the worked
    # p=2 contract example (oracle O16, [C, sec:example]). Written by the
    # S3V validation agent directly from docs/latex/hecke_action.tex,
    # without reading HeckeCosetTables.cpp's or
    # HeckeCosetTablesSelfCheckValidation.cpp's expected values.
    src/hacs/Hecke/HeckeCosetTablesValidation.cpp

    # Stage S4 design choice (documented per plan, same reasoning as S2/S3
    # above): HeckeTauTable consumes the SimplicialComplex model, the S2
    # tessellation tables and FlatMatrix / ComputeSymmetricPowerMatrixFM
    # (ModArithmetic) for the rho_n(g gamma_k) Xi-term blocks
    # ([C, eq:xiexplicit]), so it is compiled into cohomology_app, NOT into
    # the pure-integer HeckeCore static library. It links against HeckeCore
    # for the exact-integer layer (Mat22, QuadPoint, certified predicates).
    src/hacs/Hecke/HeckeTauTable.cpp
    src/hacs/Hecke/HeckeTauTableSelfCheckValidation.cpp

    # (hecke-extensions slot 3, WP-1 / phase E1b-0) The U_ell exact-integer
    # reach: the two tiers of [C, thm:backend] on [C, alg:tau] step 1's
    # fundamental-domain reduction (QuadPointT<int64_t> / QuadPointT<__int128>,
    # hecke::escalate_on_overflow), the forced-tier identity arms and the
    # ell = 5003 / 10007 / 43867 reach proof.  It drives the whole S3/S4/S7/S9
    # pipeline (coset tables -> tau -> HeckeOperatorFamily), so it belongs to
    # cohomology_app and not to the pure-integer HeckeCore.
    src/hacs/Hecke/HeckeUEllReachValidation.cpp

    # Stage S4V -- INDEPENDENT validation of Stage S4 ([C, obl:tau],
    # [C, obl:backend]). Written by the S4V validation agent directly from
    # docs/latex/hecke_action.tex and the implementation plan, re-deriving
    # the boundary operator and the (T1) twisted-equivariance translation
    # from scratch (never calling HeckeTauTable::certify() or its private
    # helpers). See Hecke/HeckeTauTableValidation.h for the independence
    # statement.
    src/hacs/Hecke/HeckeTauTableValidation.cpp
    src/hacs/Hecke/HeckeIotaTauSymmetryValidation.cpp
    # gl2_contract.tex chapter 9, obl:iota(ii),(iii),(v): iota^* as a cochain
    # map, its +-split of H^1_eq, and the OQ3 oracle (Eisenstein sign).
    src/hacs/Hecke/HeckeIotaSignValidation.cpp

    # hecke-extensions slot 1, WP-4a (phase E4a) -- the certified Gamma-level
    # boundary datum tau_bd of gl2_contract.tex chapter sec:boundarydatum
    # (def:taubdry, lem:taubdryexists, obl:taubdry(i)-(iv)) and the
    # B_N / Gamma_inf table (prop:bdrycosets(1)-(3)). Same placement rule as
    # S3/S4 above: it consumes the SimplicialComplex model, the S2 tables
    # and the templated HeckeCosetTablesT<Elt> (FlatMatrix rho_n blocks), so
    # it is compiled into cohomology_app, NOT into the pure-integer HeckeCore
    # static library, and links HeckeCore for Mat22 / QuadPoint. Certified in
    # the constructor (invariant I8). Validation: HeckeBoundaryTauValidation.
    src/hacs/Hecke/HeckeBoundaryDatum.cpp
    src/hacs/Hecke/HeckeBoundaryTauValidation.cpp

    # hecke-extensions slot 1, WP-4b (phase E4b) -- the boundary Hecke operator
    # Theta^bd_g / T^bd_p of gl2_contract.tex sec:boundarydatum on the circle
    # model (def:bdryop, eq:bdrychain, prop:bdryop, prop:bdrychar; routes A/B
    # of obl:taubdry(v); the model recording of owner confirmation 11). Same
    # placement rule as HeckeOperatorOnK (consumes the cohomology layer and
    # ModArithmetic; links HeckeCore for Mat22). Validation:
    # HeckeBoundaryOperatorValidation (obl:taubdry(v)-(vii), sec:oraclesnew (A)).
    src/hacs/Hecke/HeckeBoundaryOperator.cpp
    src/hacs/Hecke/HeckeBoundaryOperatorValidation.cpp

    # hecke-extensions slot 2, WP-7 (phase E4b) -- the two RECORDED diagnostics
    # of gl2_contract.tex rmk:conestrict / [X] C-A3(iii)(5): cochain_commutes
    # (r Theta_g = Theta_g^bd r on C^q_eq(K), through HeckeOperatorOnK::
    # apply_Theta and the restriction morphism) and eta_zero (the cusp-support
    # walk of the bulk tau against tau_bd). Lives in Hecke/ and not HeckeExt/
    # because it consumes HeckeOperatorOnK (rule 2 of check_heckeext_includes.sh)
    # and because WP-8's eta-table (Hecke/HeckePairHomotopy) must be able to
    # compare its own eta == 0 verdict with the walk without crossing the
    # guard. Same placement rule as HeckeBoundaryOperator above.
    src/hacs/Hecke/HeckeBoundaryDiagnostics.cpp

    # hecke-extensions slot 2, WP-8 (phase E5-1) -- the certified pair
    # homotopy eta of gl2_contract.tex sec:cone (def:pairhomotopy,
    # lem:etaexists, obl:eta(i)-(iv)), the first third of the Hecke datum of
    # the pair (tau, tau_bd, eta) the cone action consumes. HeckeIntegerSolve
    # is the ONE integer solver of every chain fill, moved verbatim out of
    # HeckeTauTable.cpp (value-preserving; gated by the golden suite and
    # tau_table_p2.txt byte-identical) so that the eta fills use the bulk
    # tau's solver. Same placement rule as HeckeBoundaryDatum above (consumes
    # the templated tau / tau_bd tables; links HeckeCore for Mat22).
    # Validation: HeckePairHomotopyValidation.
    src/hacs/Hecke/HeckeIntegerSolve.cpp
    src/hacs/Hecke/HeckePairHomotopy.cpp
    src/hacs/Hecke/HeckePairHomotopyValidation.cpp

    # hecke-extensions slot 3, WP-5 (phase E5-2) -- the PIPELINE-SIDE pair
    # (K, K_bd) with the restriction r = j^* of gl2_contract.tex def:pair,
    # lifted verbatim out of HeckeExt/HeckeBoundary (which now consumes it
    # one-way) so that the cone and slot-3 WP-6's cone action can build on
    # it without crossing rule 1 of check_heckeext_includes.sh (slot-3 plan
    # Sec. 2.3 item 4, audit F3).  Validation: HeckeConeValidation, with
    # HeckeBoundaryParabolicValidation's unchanged summary lines as the
    # value-preservation proof of the re-route.
    src/hacs/Hecke/SL2ZBoundaryPair.cpp
    src/hacs/Hecke/HeckeConeValidation.cpp

    # hecke-extensions slot 3, WP-6 (phase E5-3): the cone HECKE ACTION -- the
    # derived cochain homotopy h = eta~ of [C, eq:derivedh], the cone cochain
    # map Phi_g of [C, eq:conemap], A_c^{(q)} by [C, alg:assembly] on Cone(r),
    # ConeActionCertificate ([C, obl:eta](v), [C, obl:coneexact](i)-(iii)) and
    # ConeRelationsCertificate ([C, obl:coneexact](vi)).  In Hecke/, not
    # HeckeExt/: it consumes HeckeOperatorOnK / EquivariantMappingCone (the
    # block model, and through the pipeline the tape), which rule 2 of
    # check_heckeext_includes.sh forbids in the extension layer, and it is part
    # of the pipeline E2 will consume, which rule 1 forbids depending on
    # HeckeExt/ (slot-3 plan Sec. 2.3 item 4).  Validation:
    # HeckeConeActionValidation.
    src/hacs/Hecke/HeckeConeAction.cpp
    src/hacs/Hecke/HeckeConeActionValidation.cpp

    # (hecke-extensions slot 4, WP-3 / phase E3) The PRODUCTION involution
    # iota^* of [C, sec:involution]: [C, def:iota](iv) with the v1.2.0 negation
    # iota^* := -iota^*_geo, realised -- verbatim as [C, sec:codemap] specifies
    # it -- as an EquivariantCochainComplexMorphismOnK whose FiberBlockTable
    # holds one PerSimplexBlock per simplex (source |iota_K sigma|, block
    # -eps_sigma rho_n(iota)); [C, obl:iota](i)-(iii) aborting in compute(); and
    # the +- split of [C, prop:iotasplit] on the stored C/Z/B/H objects.  Here
    # in Hecke/ and not in HeckeExt/ for the HeckeConeAction reason above: it is
    # part of the pipeline (it consumes EquivariantCohomologyOnK and the
    # morphism class, and E2 will consume it), which
    # check_heckeext_includes.sh rule 1 forbids depending on HeckeExt/.
    # Validation: HeckeInvolutionValidation (in HeckeExt/, where the sign split
    # lives).
    src/hacs/Hecke/HeckeInvolution.cpp

    # Stage S5 design choice (documented per plan, same reasoning as
    # S2/S3/S4 above): InducedLocalSystem builds a plain
    # LocalSystemSimplicialComplex ([C, def:inducedls] / [C, eq:MH]) from
    # HeckeCosetTables' rho_n(.) blocks (ModArithmetic), so it is compiled
    # into cohomology_app, NOT into the pure-integer HeckeCore static
    # library. It links against HeckeCore for hecke::Mat22.
    src/hacs/Hecke/InducedLocalSystem.cpp
    src/hacs/Hecke/HeckeInducedLocalSystemSmokeValidation.cpp

    # Stage S5V -- INDEPENDENT Shapiro validation of Stage S5 (oracle O11,
    # [C, obl:shapiro]; written by the S5V validation agent without reading
    # HeckeInducedLocalSystemSmokeValidation.cpp's expected values -- see
    # Hecke/HeckeShapiroValidation.h/.cpp for the independence statement).
    src/hacs/Hecke/HeckeShapiroValidation.cpp

    # Stage S6 -- self-check for the per-simplex sparse-block-row generalization
    # of EquivariantCochainComplexMorphismOnK (the code-level v5 fix,
    # [C, rmk:nogo]): legacy ctor vs. explicit one-entry FiberBlockTable produce
    # elementwise-identical induced H*-maps.
    src/hacs/Hecke/HeckeMorphismGeneralizationValidation.cpp

    # Stage S6V -- INDEPENDENT validation agent: bit-identity regression of
    # get_induced_H_eq_maps() against reference values captured on the pre-S6
    # parent commit (41df5e278460f4c1a9f7ee3270878052a0802983), plus a
    # 100-fixed-seed legacy-ctor-vs-FiberBlockTable equivalence check -- see
    # Hecke/HeckeMorphismRegressionValidation.h for the full statement.
    src/hacs/Hecke/HeckeMorphismRegressionValidation.cpp

    # Stage S7 design choice (documented per plan, same reasoning as
    # S2/S3/S4/S5 above): HeckeOperatorOnK consumes ModArithmetic
    # (FlatMatrix / Matrix / ComputeSymmetricPowerMatrixFM for the fused
    # blocks and the level-(a)/(b) linear algebra of [C, alg:assembly]) and
    # the EquivariantCohomologyOnK / SimplicialComplex layer, so it is
    # compiled into cohomology_app, NOT into the pure-integer HeckeCore
    # static library. It links against HeckeCore for hecke::Mat22 (route 2's
    # exact integer products delta_i * gamma_k, invariant I6).
    src/hacs/Hecke/HeckeOperatorOnK.cpp
    src/hacs/Hecke/HeckeOperatorSmokeValidation.cpp

    # Stage S7V -- INDEPENDENT primary oracle battery for HeckeOperatorOnK
    # (oracles O1-O8/O14, gated per plan amendment A-F4). Written by the S7V
    # validation agent directly from docs/latex/hecke_action.tex and the
    # implementation plan; includes its own LOCAL, division-free char-poly
    # helper (does not depend on Stage S10, which has not landed).
    src/hacs/Hecke/HeckeOracleValidation.cpp

    # Stage S8 design choice (documented per plan, same reasoning as
    # S2-S7 above): the two independent cross-check routes of
    # [C, obl:agreement] consume ModArithmetic (FlatMatrix / Matrix /
    # ComputeSymmetricPowerMatrixFM), the induced-local-system layer (S5),
    # the generalized morphism class (S6) and the S7 operator, so they are
    # compiled into cohomology_app, NOT into the pure-integer HeckeCore
    # static library. They link against HeckeCore for hecke::Mat22 (the
    # exact beta cocycles gamma_k t_j^{-1} and the exact bar-route
    # kappa_j(gamma) = delta_{gamma*j}^{-1} gamma delta_j products).
    # BarCocycleH1 carries its own LOCAL division-free charpoly/minpoly
    # helpers (Stage S10 has not landed).
    src/hacs/Hecke/HeckeCrossChecks.cpp
    src/hacs/Hecke/BarCocycleH1.cpp
    src/hacs/Hecke/HeckeCrossCheckValidation.cpp
    # matrix-engine Phase 3 Stage B -- the conditional absorption of
    # BarCocycleH1's hand-rolled RREF-with-RHS solver into
    # engine::elim::solve_multi_rhs, pinned against a frozen copy of the
    # pre-Stage-B body.
    # (S24 CL-L2) Hecke/BarCocycleSolverValidation.cpp moved to Hecke/legacy/ (cmake/legacy.cmake, ON only).

    # Stage S8V -- INDEPENDENT agreement & independence validation (oracle
    # O12 in full: route bit-identity, custom Gamma_0(p) coset
    # representatives, TauVertexPolicy FixedV6 vs GeometricNearest, the
    # documented weaker-reachable-form check for the double-coset
    # representative g, and HECKE_PURE_INTEGER ON-vs-OFF verified externally
    # by diffing this validation's stdout). Depends only on the public
    # HeckeOperatorOnK / HeckeCosetTables / HeckeTauTable API, so it is
    # compiled into cohomology_app, NOT into the pure-integer HeckeCore
    # static library (same reasoning as S7/S8 above). Links against HeckeCore
    # for hecke::Mat22 (the exact h_i * s_i / gamma * s_i products).
    src/hacs/Hecke/HeckeIndependenceValidation.cpp

    # Stage S9 design choice (documented per plan, same reasoning as
    # S2-S8 above): HeckeOperatorFamily assembles repeated Stage S7
    # HeckeOperatorOnK pipeline runs (HeckeCosetTables -> HeckeTauTable ->
    # HeckeOperatorOnK, for varying (p_hecke, r)) and does per-degree Matrix
    # linear algebra (ModArithmetic) for the [C, thm:relations] recursion
    # and the [C, prop:diamond] central scalar, so it is compiled into
    # cohomology_app, NOT into the pure-integer HeckeCore static library.
    # It is "no new math" (invariant I7): T_p(p) with p == ell IS U_ell,
    # the same pipeline, never special-cased.
    src/hacs/Hecke/HeckeOperatorFamily.cpp
    src/hacs/Hecke/HeckeOperatorFamilySmokeValidation.cpp

    # Stage S9V -- INDEPENDENT Hecke-relations validation (oracles O9, O10;
    # written by the independent S9V validation agent from
    # docs/latex/hecke_action.tex and the implementation plan's amendment
    # A-F1, without reading HeckeOperatorFamilySmokeValidation.cpp's
    # expected values). Depends only on the public HeckeOperatorFamily /
    # EquivariantCohomologyOnK / LocalSystemSL2Z API, so it is compiled into
    # cohomology_app, NOT into the pure-integer HeckeCore static library
    # (same reasoning as S7-S9 above).
    src/hacs/Hecke/HeckeRelationsValidation.cpp

    # Stage S10 design choice (documented per plan, same reasoning as
    # S2-S9 above): the F_ell[X] toolkit (FpPoly: exact polynomial
    # arithmetic, division-free Berkowitz charpoly, Krylov minpoly,
    # certified factorization) and the level-(c) single-operator
    # block-diagonalization (HeckeModuleDecomposition, plan Sec. 3) consume
    # ModArithmetic's Matrix (kernel / rank / inverse / RREF) and the
    # ModArithmetic primitives, so they are compiled into cohomology_app,
    # NOT into the pure-integer HeckeCore static library. Stage S10 is
    # mathematically INDEPENDENT of the Hecke pipeline S1-S9: it operates on
    # given F_ell matrices and needs only ModArithmetic + the S0 harness.
    src/hacs/Hecke/FpPoly.cpp
    # (hecke-extensions E0a) The exact F_ell subspace toolkit: SolveColumnsExact /
    # HStack / KernelAsMatrix / PrimaryRCFBasis moved verbatim out of
    # HeckeModuleDecomposition.cpp so E0b and E1a can share them without
    # `Hecke/` including `HeckeExt/` (slot-1 plan Sec. 2.3 item 5). Same
    # placement reasoning as HeckeModuleDecomposition.cpp below: it consumes
    # ModArithmetic's Matrix, so it is NOT in the pure-integer HeckeCore.
    src/hacs/Hecke/SubspaceOps.cpp
    # (hecke-extensions E0b / slot 2 WP-1) The MATHEMATICAL SETTINGS layer
    # ([X] Sec. 1.2 `MathSettings`, Sec. 1.5, [OD] R1): engine-owned defaults,
    # validation, canonical text, fingerprint and the thread-local scoped
    # override propagated onto engine::tape::run_concurrently() tasks. It
    # consumes NOTHING of the pipeline -- only <string>, ModArithmetic/Fnv.h
    # and src/Engine/EngineRuntime.h -- but it is placed here and
    # not in HeckeCore because its first consumer, `locality`, is
    # HeckeModuleDecomposition, and because EngineRuntime belongs to the
    # MatrixEngine target that HeckeCore must not depend on.
    src/hacs/Hecke/HeckeSettings.cpp
    src/hacs/Hecke/HeckeModuleDecomposition.cpp
    src/hacs/Hecke/HeckeDecompositionSmokeValidation.cpp

    # (hecke-extensions E0a) The read-only locality certificate's validation.
    # Same layer as HeckeDecompositionSmokeValidation, plus the golden harness
    # (its grid arm drives GoldenLocalitySweep) and, for the interlocked
    # (160, 691) probe, the S7/S9 pipeline through HeckeOperatorFamily.
    src/hacs/Hecke/HeckeLocalityValidation.cpp

    # (hecke-extensions E1s / WP-3) The extension layer's first unit: the
    # computed Sturm bound B'(k, ell) ([E] def:bprime / cl:bprime), its
    # certificate and its resumable state ([X] Sec. 1.3.1 R3, instance 1).
    # HeckeExt/ is compiled straight INTO cohomology_app and not into a static
    # library ([X] Sec. 1.1: the layer consumes MatrixT, FpPolyT,
    # HeckeModuleDecomposition, HeckeOperatorFamily and EquivariantCohomologyOnK,
    # all of them executable TUs; a HeckeExt library would first need the
    # Pipeline-library refactor the console plan records as an unscheduled
    # follow-up).  The one-way dependency is enforced instead by the grep guard
    # docs/matrix-engine/tools/check_heckeext_includes.sh, run by
    # HeckeExtArchitectureValidation below.  HeckeSturmBound itself consumes
    # NOTHING of the pipeline -- only Hecke/SubspaceOps, MatrixT and
    # ModArithmetic -- which is what lets it be built and validated in isolation.
    src/hacs/HeckeExt/HeckeSturmBound.cpp
    src/hacs/HeckeExt/HeckeSturmBoundValidation.cpp
    src/hacs/HeckeExt/HeckeExtArchitectureValidation.cpp

    # (hecke-extensions slot 3, WP-1 / phase E1b-0) The FROZEN U_ell RSS model
    # behind the `u_ell` mathematical setting ([X] Sec. 1.5, Sec. 2.2 Stage 1,
    # [XA B3]).  HeckeExt/HeckeUEllBudget.h is a header-only pure function --
    # no TU of its own -- consumed by HeckeExtDriver's Stage 1 in slot-3 WP-2;
    # only its validation is compiled.  Like every HeckeExt/ TU it obeys
    # check_heckeext_includes.sh rules 2 and 4, which is why the setting's
    # run_concurrently propagation arm sits in Hecke/HeckeUEllReachValidation.
    src/hacs/HeckeExt/HeckeUEllBudgetValidation.cpp
    # (hecke-extensions slot 1, WP-2) The E0a remainder's identity
    # certificate ([X Sec.1.3]'s subset needing no unscheduled machinery):
    # HeckeExt/HeckeIdentityCertificate.{h,cpp} is a pure MatrixT<uint32_t> +
    # engine::ModCtx unit (the plan's "MatrixT<uint32_t> for everything in
    # HeckeExt/" decision -- see check_matrix_includes.sh's allow-list entry
    # for it); its validation drives the same S7/S9 HeckeOperatorFamily
    # pipeline as HeckeOperatorFamilySmokeValidation above, so it is
    # compiled into cohomology_app, NOT into the pure-integer HeckeCore.
    src/hacs/HeckeExt/HeckeIdentityCertificate.cpp
    src/hacs/HeckeExt/HeckeIdentityValidation.cpp
    # (hecke-extensions slot 2, WP-7 / phase E4b) HeckeExt/HeckeBoundary: the
    # ext-layer CONSUMER of Hecke/HeckeBoundaryDatum + HeckeBoundaryOperator
    # ([X] Sec. 1.1) -- H^1_! = ker r^1 with its Hecke action, the Eisenstein
    # line, Harder's criterion from a pinned Bernoulli table, the Harder split,
    # the certificate, the golden collectors (defined, never emitted here) and
    # the per-(p, nu) diagnostics. MatrixT<uint32_t> for its stored objects
    # (check_matrix_includes.sh carries it with that reason); drives the bulk
    # and boundary pipelines (HeckeCosetTables -> HeckeTauTable ->
    # HeckeOperatorOnK, HeckeBoundaryTauTable -> HeckeBoundaryOperator) and the
    # restriction morphism, so it is compiled into cohomology_app like every
    # other HeckeExt/ TU. Validation: HeckeBoundaryParabolicValidation.
    src/hacs/HeckeExt/HeckeBoundary.cpp
    src/hacs/HeckeExt/HeckeBoundaryParabolicValidation.cpp

    # (hecke-extensions slot 2, WP-4 / E1a) The local algebra A_S of a
    # certified-local block, its invariants and generation certificate (a)
    # ([E] sec:localalg / alg:stage3 / cert:cyclic, [X] Sec. 2.2 Stage 3 and
    # Stage 4(a)).  MatrixT<uint32_t> like the rest of HeckeExt/ (see
    # check_matrix_includes.sh's allow-list entry for it); it consumes
    # Hecke/SubspaceOps, Hecke/FpPoly and Hecke/HeckeModuleDecomposition --
    # all of them executable TUs -- so it is compiled into cohomology_app and
    # NOT into the pure-integer HeckeCore, exactly like its neighbours above.
    src/hacs/HeckeExt/HeckeLocalAlgebra.cpp

    # (hecke-extensions slot 2, WP-5 / E1a) Generation certificate (b): the
    # socle bound, the structured commutant on the rational canonical form of
    # a cyclic-most element of A_S, the naive oracle, the resumable
    # CommutantState ([E] cert:comm, [X] Sec. 2.2 Stage 4(b), Sec. 1.3 and
    # Sec. 1.3.1 R3.3 item 3) and the route-combining `final` verdict of
    # R3.2.  Same placement reasoning as HeckeLocalAlgebra.cpp above: it
    # consumes that unit plus Hecke/SubspaceOps and Hecke/FpPoly, all
    # executable TUs, so it is compiled into cohomology_app and NOT into the
    # pure-integer HeckeCore.
    src/hacs/HeckeExt/HeckeGenerationCertificates.cpp

    src/hacs/HeckeExt/HeckeExtAlgebraValidation.cpp

    # (hecke-extensions slot 3, WP-2 / E1b) Generation certificate (c): the
    # Sturm sweep on the pipeline's own blocks, and Stage 1's `u_ell` budget
    # check ([E] cert:sturm on the range of cl:sturmmodl, [X] Sec. 2.2 Stage 1
    # and Stage 4(c), Sec. 1.3.1 R3).  Unlike HeckeSturmBound.cpp -- which the
    # HeckeExt/ guard's rule 3 keeps pipeline-free -- this TU consumes
    # Hecke/HeckeOperatorFamily and Hecke/SubspaceOps, so it is compiled into
    # cohomology_app and NOT into the pure-integer HeckeCore, exactly like its
    # neighbours above.
    src/hacs/HeckeExt/HeckeSturmSweep.cpp

    # (hecke-extensions slot 4, WP-3 / phase E3) The SIGN SPLIT of the Hecke
    # module: [C, obl:iota](v) ([iota^*, A^{(q)}] = 0, asserted at ell >= 5 and
    # recorded at ell = 3), A^{(q)}_+- by an exact solve, the per-sign primary
    # decomposition beside the unsplit one, cyclicity per sign through slot-2
    # WP-4's generation certificate (a), and the Eisenstein line
    # soc(X^+_{4,m}) = m_S X^+_{4,m} of [E] D6 ("undefined" off dim_F = 1).
    # MatrixT<uint32_t> and no pipeline object at all: it takes the matrices
    # Hecke/HeckeInvolution produces.  Validation: HeckeInvolutionValidation.
    src/hacs/HeckeExt/HeckeSignSplit.cpp
    src/hacs/HeckeExt/HeckeInvolutionValidation.cpp
    src/hacs/HeckeExt/HeckeExtSturmSweepValidation.cpp

    # (hecke-extensions slot 4, WP-5 / console C1b) The extension layer's OWN
    # console registration TU ([OD] slot-3 close-out 1): `sturm_bound(k, ell)`
    # and `u_ell_decision(n, ell)`.  It is the ONE HeckeExt/ file that includes
    # Console/ headers, admitted by a one-element rule-2 exception in
    # check_heckeext_includes.sh and one ALLOW_CONSOLE line in
    # check_console_includes.sh, both with the reason "[OD] slot-3 close-out 1".
    # Its header forward-declares console::Registry and includes no Console/
    # header, so only the .cpp carries an exception.
    src/hacs/HeckeExt/RegisterHeckeExtensionVerbs.cpp

    # (hecke-extensions slot 4, WP-1 / E2a) THE EXTENSION RECORD.
    #   * HeckeExt/HeckeExtFinal.h -- header-only, dependency-free: the ONE
    #     tri-state `Final` of [X] Sec. 1.3.1 R3.2 and its fixed reason list;
    #     no TU of its own.  `SturmFinal` is an alias of it since this package.
    #   * HeckeModuleStructure -- Stage 5 ([E] `alg:stage5`): the Loewy and
    #     socle series, the Jordan types in units of f, and the structure-map
    #     matrices, with a re-deriving verify().  Consumes HeckeLocalAlgebra
    #     only, so it is a HeckeExt/ TU like its neighbours above.
    #   * HeckeExtRecord.h (the data model, header-only) + HeckeExtReport --
    #     the `hecke_ext/v1` writer of [E] Sec. 7 / [X] Sec. 1.4.  It is a
    #     SEPARATE writer from Hecke/HeckeReport (`hecke_report/v2`), which
    #     slot 4 does not touch: the golden harness hashes the latter, so
    #     extending it would move a golden value ([X] Sec. 1.6).
    #   * HeckeExtRecordValidation -- the schema, `is_final`, the sidecar, the
    #     settings block and the campaign job-id arm.
    src/hacs/HeckeExt/HeckeModuleStructure.cpp
    src/hacs/HeckeExt/HeckeExtReport.cpp
    src/hacs/HeckeExt/HeckeExtRecordValidation.cpp

    # (hecke-extensions slot 4, WP-2 / E2b) THE STAGE 1-5 ORCHESTRATION.
    #   * HeckeExtDriver -- one cell (n_sym, ell): one triangulation, one local
    #     system, ONE computed cohomology and ONE HeckeOperatorFamily, walked
    #     through [X] Sec. 2.2 Stages 1-5 into WP-1's HeckeExtRecord.  Stage 3
    #     CONSUMES the decomposition's own certified PrimaryBlock locality data
    #     (the slot-4 plan Sec. 2.3 item 2); a Sturm prime outside S_0 is
    #     STREAMED through HeckeSturmSweep's StreamedOperatorSupplier, so the
    #     family's never-evicting pipeline cache holds only S_0; and the `u_ell`
    #     resume of [X] Sec. 1.3.1 R3.3 item 4 lives here, because "re-run
    #     Stage 1 with U_ell attached to the same family" is a statement about
    #     this driver.  Like HeckeSturmSweep.cpp it includes Hecke/ pipeline
    #     headers and no Engine/, Cuda/ or Console/ header (guard rule 2).
    #   * HeckeExtRamanujanValidation -- [E] `cl:ramanujan` (Claim 8.1) at
    #     (10, 691) with U_691 inside the budget, the streamed-operator arm and
    #     the u_ell resume arm.  `--bench ext` is in Bench/BenchDriver.cpp.
    src/hacs/HeckeExt/HeckeExtDriver.cpp
    src/hacs/HeckeExt/HeckeExtRamanujanValidation.cpp

    # (hecke-extensions slot 5, WP-1 / E2 cont.) THE RECORD LAYER'S VIEW OF THE
    # E3 / E4b / E5 OBJECTS, for `hecke_ext/v2`.  `hecke_ext` is a record about
    # THREE complexes and about the involution's split, and each of those is
    # built by a unit that knows nothing about the record; this TU is the one
    # place the two vocabularies meet.  Pure translation: every argument is an
    # already-computed object, nothing here builds a cone, a Hecke operator or
    # an involution, and every `ComplexRecord` takes its degree count from the
    # object it is about and from nothing else ([C, sec:recordschema] item 5).
    # It includes pipeline headers in its HEADER exactly as
    # HeckeExt/HeckeBoundary.h does -- guard rule 1 is about the reverse
    # direction -- and no Console/, Cuda/, Engine/ or BlockFlatMatrix.h.
    # Validation: HeckeExtRecordValidation (the per-complex, les_maps,
    # per-sign and sum arms).
    src/hacs/HeckeExt/HeckeExtComplexRecords.cpp

    # (hecke-extensions slot 4, WP-7 / phase E6) The two E6 PRE-RECORD units:
    # Bernoulli numbers mod ell with the Kummer / Harder divisibility predicate
    # ([X] Sec. 2.1, Sec. 2.2 Stage 6), and the Glover / Ash-Stevens composition
    # factors of Sym^n(F_ell^2) ([E] Sec. identities item 6).  Both consume
    # NOTHING of the pipeline -- HeckePredictions.cpp only engine::ModCtx and
    # EngineCheck.h, GloverFactors.cpp only EngineCheck.h and the STL -- so they
    # could in principle live beside HeckeSturmBound in a pipeline-free corner;
    # they are compiled into cohomology_app with the rest of HeckeExt/ because
    # the layer has no static library of its own ([X] Sec. 1.1) and because
    # their validation drives ComputeSymmetricPowerMatrixFM.  They are NOT
    # named in check_heckeext_includes.sh rule 3, which is HeckeSturmBound's
    # own allow-list; rules 1, 2 and 4 bind them like every other file here.
    src/hacs/HeckeExt/HeckePredictions.cpp
    src/hacs/HeckeExt/GloverFactors.cpp
    src/hacs/HeckeExt/HeckePredictionsValidation.cpp

    # (hecke-extensions slot 5, WP-7 / phase E6) THE PROJECT-CANONICAL SPLIT
    # FIELD ([X] Sec. 2.2 Stage 6, [E] def:canon / cl:canon / alg:split; the
    # package planned as slot-4 WP-10, which did not run).  C_{ell,r} by
    # enumeration in CanonicalLess order + hecke::CertifyIrreducible, the
    # arithmetic of F = F_ell[Y]/(C) and of univariate polynomials over it, and
    # the root finding of alg:split step 1 with its FIXED candidate sequence.
    # Like the two units above it consumes NOTHING of the pipeline -- only
    # Hecke/FpPoly, Hecke/HeckeSettings, ModCtx.h and EngineCheck.h -- and is
    # bound by rules 1, 2 and 4 of check_heckeext_includes.sh (rule 3 is
    # HeckeSturmBound's own allow-list and does not name it).  There is NO
    # REPORT CONSUMER in this slot: the record integration is E6's, in slot 6.
    src/hacs/HeckeExt/HeckeCanonicalField.cpp
    src/hacs/HeckeExt/HeckeCanonicalFieldValidation.cpp

    # (hecke-extensions slot 5, WP-8) IDENTITY CLASS 5 AND M1.
    #   * HeckeDicksonMultiplication.{h,cpp} -- `L.`, multiplication by the
    #     Dickson invariant L = x^ell y - x y^ell, Sym^{n-ell-1} -> Sym^n
    #     ([X] Sec. 1.1's inventory gap [I G12] / [XA m10]), with the
    #     det-twisted intertwining rho_n(A) L = det(A) L rho_m(A) re-derived
    #     entrywise on seeded A.  It is typed `FlatMatrix`, NOT `ExtMatrix`:
    #     it is a FIBRE-level object of size (n+1) x (n-ell), the same kind of
    #     object as rho_n, and its consumer is the COCHAIN layer
    #     (PerSimplexBlock::F).  It therefore includes no Matrix.h and needs
    #     no entry in check_matrix_includes.sh's allow-list.  Its
    #     cohomology-level statement lives in HeckeIdentityCertificate's new
    #     item 5, whose file IS already allow-listed.
    #   * HeckeM1Table.{h,cpp} -- ONE definition of "an M1 row" and of "a
    #     well-formed M1 row" ([X] Sec. 6.2 E2's M1, Sec. 1.3.1 R3.2's fixed
    #     reason list), shared by the three case validations below so that the
    #     deliverable is one object and not three.  It computes nothing.
    #   * the three case validations of [E] Sec. 8 on X_1 (+) X_4.
    # All five files obey the four rules of check_heckeext_includes.sh.
    src/hacs/HeckeExt/HeckeDicksonMultiplication.cpp
    src/hacs/HeckeExt/HeckeM1Table.cpp
    src/hacs/HeckeExt/HeckeExtKummerValidation.cpp
    src/hacs/HeckeExt/HeckeExtSmallEllValidation.cpp
    src/hacs/HeckeExt/HeckeExtSeparationValidation.cpp

    # (hecke-extensions slot 6, WP-1 / phase E7a) THE CONTRAGREDIENT SIDE:
    # rho_n^vee(g) = rho_n(adj g)^T, the local system D_n, Phi_n of
    # [C, lem:symdual] as a FiberBlockTable morphism, the evaluation pairing
    # of [C, prop:pairing], H^*(D_n) and A(D_n)(T_p) through the shipped
    # pipeline (HeckeOperatorOnK with FibreRepresentation::ContragredientN).
    # In Hecke/, not HeckeExt/: it is pipeline-level (it consumes
    # EquivariantCohomologyOnK / HeckeOperatorOnK and is consumed by E7b), so
    # it is compiled into cohomology_app like every other Hecke/ pipeline TU.
    # No direct Matrix.h include (check_matrix_includes.sh unchanged).
    src/hacs/Hecke/HeckeContragredient.cpp
    # (hecke-extensions slot 6, WP-3 / phase E7b) THE COHOMOLOGICAL PAIRINGS:
    # the relative fundamental 2-chain of the model, the Alexander-Whitney
    # cup-product evaluation form, the cone-level pairing
    # H^t_c(M_n) x H^{2-t}(D_n) and the circle pairing, with their descent
    # certificate ([C, cl:pd](1)-(2), OPEN: HeckeDualityValidation tests it).
    # Hecke/, pipeline-level (it reads EquivariantMappingCone read-only).
    src/hacs/Hecke/HeckePairing.cpp
    src/hacs/Hecke/HeckeDualityValidation.cpp

    # (hecke-extensions slot 6, WP-2 / phase E2, OQ11) THE ON-REQUEST
    # PRESENTATION of [X] Sec. 2.2 Stage 5: generators (the least-index
    # Nakayama lift) and minimal relations of a certified-local block over
    # T_m = A_S|_V, with a re-deriving verify().  HeckeExt/, MatrixT through
    # HeckeLocalAlgebra.h (no direct Matrix.h include), zero raw mod_* sites;
    # reached through ExtDriverOptions::with_presentation (default off) and
    # NOT through a settings key -- the key is slot 7's re-baseline.
    src/hacs/HeckeExt/HeckePresentation.cpp
    src/hacs/HeckeExt/HeckePresentationValidation.cpp
    # (hecke-extensions slot 6, WP-8 / phase E6) THE PER-ell RESIDUE ATLAS
    # Sigma(ell), S_sep(ell) and the canonical split field's first consumer
    # ([E] sec:atlas, cl:sigma, cl:sep, sec:field tier (ii); [X] Sec. 2.2
    # Stage 6).  Runs HeckeExtDriver per cell; its `hecke_atlas/v1` store is
    # written ONLY under HECKE_ATLAS_OUT=<dir> and never enters the record or
    # golden.  Obeys the four rules of check_heckeext_includes.sh.
    src/hacs/HeckeExt/HeckeAtlas.cpp
    src/hacs/HeckeExt/HeckeAtlasValidation.cpp
    # (hecke-extensions slot 6, WP-9 / E6) identity 6 (IdentityCertificate
    # item 6) with its det-twist negative control, S_sep / Stage 2(3) as an
    # opt-in (ExtDriverOptions::s_sep, default empty), and the recorded
    # proof_range comparison.
    src/hacs/HeckeExt/HeckeExtEndToEndValidation.cpp

    # Stage S10V -- INDEPENDENT decomposition validation (oracles O6, O8;
    # see Hecke/HeckeBlockFormValidation.h for the independence statement).
    # Depends on the same layer as HeckeDecompositionSmokeValidation above
    # (ModArithmetic's Matrix + the S10 FpPoly/HeckeModuleDecomposition
    # public API) plus the full S3/S4/S7 Hecke pipeline (HeckeCosetTables /
    # HeckeTauTable / HeckeOperatorOnK) to build A^{(1)}(T_2), so it is
    # compiled into cohomology_app, NOT into the pure-integer HeckeCore
    # static library (same reasoning as S7-S9 above).
    src/hacs/Hecke/HeckeBlockFormValidation.cpp

    # Stage S11 design choice (documented per plan, same reasoning as S10
    # above): the multi-operator branch of HeckeModuleDecomposition (the
    # simultaneous MaxSpec(T) refinement, plan Sec. 3) lives in the SAME
    # HeckeModuleDecomposition.cpp translation unit already listed under
    # Stage S10 above (no new source file there -- ComputeMultiOp() is a
    # private method alongside S10's ComputeSingleOp()). HeckeReport is the
    # hand-rolled JSON writer for the full three-matrix-level report (plan
    # Sec. 3 "Output formats"); it consumes only HeckeModuleDecomposition's
    # public API plus ModArithmetic's Matrix, so it is compiled into
    # cohomology_app, NOT into the pure-integer HeckeCore static library.
    src/hacs/Hecke/HeckeReport.cpp

    # matrix-engine Phase 6 -- the model descriptor (audit item 7.18 / plan Q8).
    # The single place the "which simplicial model is this?" datum is written
    # down; consulted by HeckeOperatorOnK / HeckeCrossChecks (replacing their
    # literal 7/11/5 shape tests) and by the campaign driver, which records the
    # swept model in every job's JSON. Public-API only, so cohomology_app.
    src/hacs/Hecke/HeckeModelDescriptor.cpp
    src/hacs/Hecke/HeckeSimultaneousSmokeValidation.cpp

    # Stage S11V -- INDEPENDENT validation of the simultaneous MaxSpec(T)
    # decomposition + HeckeReport JSON export (see
    # Hecke/HeckeSimultaneousValidation.h for the independence statement).
    # Depends on the same layer as HeckeSimultaneousSmokeValidation above
    # (HeckeOperatorFamily / HeckeModuleDecomposition / HeckeReport public
    # API), so it is compiled into cohomology_app, NOT into the pure-integer
    # HeckeCore static library (same reasoning as S7-S11 above).
    src/hacs/Hecke/HeckeSimultaneousValidation.cpp

    # Stage S12 -- End-to-end validation battery + benchmark driver (the
    # ACCEPTANCE GATE for the whole integral-Hecke-action feature). Aggregates
    # the full S1-S11 + FIX-negI pipeline into three registered validations
    # (HeckeEndToEndValidation, HeckeBadPrimeValidation, HeckeBenchmarkSmoke);
    # depends only on the public Hecke pipeline / decomposition / induced-system
    # API plus ModArithmetic, so it is compiled into cohomology_app, NOT into
    # the pure-integer HeckeCore static library (same reasoning as S7-S11).
    src/hacs/Hecke/HeckeEndToEndValidation.cpp

    # Phase 0 (matrix-engine) -- golden-output regression harness
    # (GoldenRegressionValidation / GoldenEmit; docs/matrix-engine/00 Sec. 4.1).
    # Consumes only the public pipeline API, so it is compiled into cohomology_app.
    src/hacs/Golden/GoldenHarness.cpp
    # matrix-engine Phase 4 Stage B (consumer wiring) validations.
    src/hacs/EngineDeterminismValidation.cpp
    src/hacs/EngineTapeConsumerValidation.cpp

    # matrix-engine Phase 6 -- the campaign driver (`--campaign <spec-file>`),
    # docs/matrix-engine/campaign.md. Enumerates (n_sym, ell, p, r, q) jobs from
    # a hand-rolled spec file, groups them by shape signature (the Phase 4
    # batching hook), runs each through the SAME public API as `--bench hecke`,
    # and streams one JSON per job plus index.json / summary.tsv. Public API
    # only -- like BenchDriver and GoldenHarness it is a plain C++ TU of
    # cohomology_app, not an nvcc TU (kernel.cu only dispatches into it).
    src/hacs/Campaign/CampaignSpec.cpp
    src/hacs/Campaign/CampaignDriver.cpp
    src/hacs/Campaign/CampaignValidation.cpp

    # (hecke-extensions slot 5, WP-2 / E2 cont.) THE CAMPAIGN'S EXTENSION SEAM.
    # The ext record is per (n_sym, ell) CELL and a campaign job is per
    # (n_sym, ell, p, r) ([XA m14]), so the engine runs once per cell, on the
    # campaign's MAIN thread and before the job loop -- HeckeExtDriver builds
    # its tessellation tables outside hecke::tau_construction_mutex() and may
    # therefore not run on a job thread, while holding that lock around run()
    # would deadlock on the driver's own re-entry (CampaignExt.h states this).
    # This is the ONLY Campaign/ TU that includes a HeckeExt/ header, which
    # check_heckeext_includes.sh rule 1 admits by name.  `ext = false` is the
    # default and leaves every byte of the campaign's output unchanged.
    # Validations: HeckeExtCampaignValidation (this file's contract) and
    # CampaignSmokeValidation / CampaignCertificateValidation (that it did not
    # move).
    src/hacs/Campaign/CampaignExt.cpp
    src/hacs/Campaign/CampaignExtValidation.cpp

    # (hecke-extensions slot 5, WP-2 / E2 cont.) THE RECORD'S DETERMINISM GATE
    # ([X] Sec. 5): the FNV-1a-64 of a whole `hecke_ext/v2` record must be
    # identical under every ENGINE_THREADS x ENGINE_BATCH arm and on the
    # device, on the pattern of EngineDeterminismValidation.  It lives BESIDE
    # HeckeExt/ and not inside it, because installing a thread pool needs
    # src/Engine/, which check_heckeext_includes.sh rule 2 forbids
    # inside the layer; rule 1 admits a *Validation.* TU to include HeckeExt/.
    src/hacs/HeckeExtDeterminismValidation.cpp

    # Console phase C0 (WP-5) -- the language core's acceptance gate.  It is
    # compiled into cohomology_app rather than into ConsoleCore so that the
    # library stays STL-only (its tree-level arms walk directories through
    # POSIX <dirent.h> and shell out to check_console_includes.sh, the same way
    # MatrixRetirementValidation drives check_matrix_includes.sh).
    src/hacs/Console/ConsoleC0Validation.cpp

    # Console phase C1a (WP-7) -- the BINDINGS.  They are compiled into
    # cohomology_app and NOT into ConsoleCore, which is exactly what keeps the
    # library STL-only (plan v3 Sec. 9, Sec. 2 C8): they are the only place
    # where a console type and a pipeline type meet.  No `decompose` and no
    # `report` verb -- those objects are the ones the extension stage's single
    # golden re-baseline moves, so they are phase C1b's.
    src/hacs/Console/Bindings/Bindings.cpp
    src/hacs/Console/Bindings/CoreMath.cpp
    src/hacs/Console/Bindings/Model.cpp
    src/hacs/Console/Bindings/Cohomology.cpp
    src/hacs/Console/Bindings/Hecke.cpp
    src/hacs/Console/Bindings/Reproducibility.cpp
    src/hacs/Console/Bindings/RegisterPipelineVerbs.cpp

    # Console phase C1b (hecke-extensions slot 4, WP-5) -- the DECOMPOSITION
    # surface (`decompose` / `report` / `crosschecks`, the Decomposition /
    # Block / Report types) and the engine-owned mathematical settings, which
    # `set locality = strict` needs.  Same placement as every other binding TU:
    # cohomology_app, never ConsoleCore.
    src/hacs/Console/Bindings/Presets.cpp
    src/hacs/Console/Bindings/Decomposition.cpp
    src/hacs/Console/ConsoleValidation.cpp

    # Console phase C2a (hecke-extensions slot 6, WP-13) -- the ENGINE KNOBS as
    # `set` keys (threads / backend / device / the pass options / lean /
    # program_cache, plus the env-only ones), and ConsoleSettingsValidation.
    # Same placement as every other binding TU: cohomology_app, never
    # ConsoleCore.
    src/hacs/Console/Bindings/EngineSettings.cpp
    src/hacs/Console/ConsoleSettingsValidation.cpp

    # Console phase C2b (hecke-extensions slot 7, WP-8) -- the HARNESS verbs
    # (validations / run_validation / bench / micro / campaign_* /
    # golden_subset / timed / bench_line / rss) on the public bench runner
    # (bench::RunMode, gap G4) and Util/Rss.h (G4b), and
    # ConsoleDeterminismValidation.  Same placement as every other binding TU:
    # cohomology_app, never ConsoleCore.
    src/hacs/Console/Bindings/Harness.cpp
    src/hacs/Console/ConsoleDeterminismValidation.cpp
    # (hecke-extensions slot 7, WP-5 / phase E6 cont.) `--atlas`: the entry
    # point that builds Sigma(ell) outside the suite -- the ell = 691 build is
    # of the order of an hour and is NOT a registered validation (it is driven
    # by docs/matrix-engine/tools/atlas_build.sh under a `git status` guard).
    # kernel.cu dispatches into it, which check_heckeext_includes.sh rule 1
    # admits.  Its store is still written only under HECKE_ATLAS_OUT.
    src/hacs/HeckeExt/HeckeAtlasCli.cpp
    # (slot 7, WP-5) the per-rho-bar tangent pair ([E] Stage 6, cl:bk): dim m/m^2
    # and a generating pair of anemic operators at n = w - 2 + ell^2 - 1, for
    # the cuspidal systems of a store; deferred beyond n = 1000 (ell = 691).
    # Printed by HeckeAtlasValidation; enters no store, record or golden.
    src/hacs/HeckeExt/HeckeTangentPair.cpp
    # (hecke-extensions slot 8, WP-5; docs/KNOWN_ISSUES.md Sec. 4c) the
    # between-cells release of the engine's process-wide program caches, called
    # by Campaign/CampaignExt.cpp's ext loop and injected into `--atlas` by
    # kernel.cu.  Outside HeckeExt/ because check_heckeext_includes.sh rule 2
    # forbids Engine/ there; its header is engine-free so the nvcc TU can
    # include it.  ProgramCacheReleaseValidation (fast): the policy, the
    # recompile identity and the two-thread clear-while-running arm.
    src/hacs/Campaign/ProgramCacheRelease.cpp
    src/hacs/Campaign/ProgramCacheReleaseValidation.cpp
    # (hecke-extensions slot 8, WP-1 / E8 item 2) HeckeReductionValidation, the
    # name [X] Sec. 5 fixes: its ELIMINATION arms (ENGINE_API.md Sec. 13.9) --
    # the Z/ell^k ring of ModCtx::prime_power, the Sec. 13.3 guards and the
    # Sec. 13.7 replay check, engine::elim::howell / kernel_module /
    # image_module / solve_multi_rhs_module / length / invariant_factors
    # against the naive oracle engine::ref::howell / invariant_factors.  It
    # records a tape program (the replay arm), so it belongs to cohomology_app
    # (MatrixEngine), not to the ModArithmetic library.  E8 items 3-4 (WP-3 /
    # WP-4) append the pipeline, reduction and Bockstein arms.
    src/hacs/Hecke/HeckeReductionValidation.cpp

    # (hecke-extensions slot 8, WP-8; [IB] = docs/implementation/
    # integral_line_design_brief.md Sec. 6.2 V-B, Sec. 2.5(a)) the Bredon
    # tree-model identification: for ell in {2, 3}, dim H^q of the shipped
    # 7-vertex pipeline equals the independent Bass-Serre tree model
    # (V^S (+) V^U -> V^{-I}), checked against the frozen 16-golden-cell
    # oracle first, then over n in {0, ..., 40} (even and odd). New
    # `src/hacs/Integral/` (Line 2's placement, alongside
    # WP-7's IntegralCharpolyCrt): includes Hecke/ public headers indirectly
    # through LocalSystemSL2Z.h only, nothing from HeckeExt/.
    src/hacs/Integral/HeckeBredonTreeValidation.cpp

    # (hecke-extensions slot 8, WP-7; [IB] Sec. 6.2 M-0, Sec. 5.2, Sec. 3.4)
    # Line 2's first certified result: char_Z(T_p | Lambda^1) by CRT over
    # existing F_ell runs of the shipped pipeline, certified against the
    # frozen O5 table digit-wise mod each prime (Deligne bound as a bit
    # length); residue vectors at campaign n. No big integer, no type wider
    # than __int128 ([IB] OD-3 (c)); Hecke/ public headers and top-level
    # pipeline headers only, nothing from HeckeExt/.
    # HeckeIntegralCharpolyValidation (standard).
    src/hacs/Integral/IntegralCharpolyCrt.cpp
    src/hacs/Integral/HeckeIntegralCharpolyValidation.cpp
    # (hecke-extensions slot 8, WP-10 / console phase C4) ConsoleErrorValidation:
    # [P] Sec. 8's error model row by row, plus the post-throw integrity arm.
    # Same placement as every console validation: cohomology_app, never
    # ConsoleCore (it registers test-only verbs throwing pipeline types).
    src/hacs/Console/ConsoleErrorValidation.cpp
    # ... and the binding TU of `help_markdown()`, the generator of
    # docs/console/console.md ([P] Sec. 4.4; docs/console/gen_console_md.sh).
    src/hacs/Console/Bindings/Reference.cpp
    # (hecke-extensions slot 8, WP-3 / E8 item 3) The coefficient ring in force
    # and the refusal rule ([E] sec:ring (a)-(b)): the `coefficient_ring`
    # helpers, the composite honouring context, and the model's relation list
    # over Z/ell^k (the composite rho_n of FlatMatrix.cpp).  In cohomology_app
    # because it reads hecke::MathSettings and builds a LocalSystemSL2Z list.
    src/hacs/Hecke/CoefficientRing.cpp
    # (slot-8 WP-3) HeckeReductionValidation's E8 item 3 arms (not a registered
    # validation of its own; its name keeps it inside the *Validation.* globs of
    # the include guards and the raw-mod census, since it is test harness).
    src/hacs/Hecke/HeckeReductionPipelineValidation.cpp
    # (hecke-extensions slot 8, WP-4 / E8 item 4a) the reductions Z/ell^k ->
    # F_ell and Z/ell^{k+1} -> Z/ell^k on cohomology and the Bockstein of
    # 0 -> Z/ell -> Z/ell^2 -> Z/ell -> 0, as maps between stored bases, with
    # their certificates as map equalities ([X] F3); read by the golden
    # harness's E8 points (red_H^q, bockstein_rank^0, higher_torsion_count).
    src/hacs/Hecke/HeckeReduction.cpp
    # (slot-8 WP-4) HeckeReductionValidation's E8 item 4a arms and the `slow`
    # HeckeReductionCoverageSlowValidation (the E2 case list at ell >= 5).
    src/hacs/Hecke/HeckeReductionCoverageSlowValidation.cpp
)

# Slots 9..25 are closed: their source lists are folded per module into cmake/sources/*.cmake (S24 CL-10, CL-10b, S25 FOLD);
# registrations of slots 9..25 live in Registry/Slot<S>.cpp. The open slot (26) will keep one fragment per package.
include(cmake/sources/Registry.cmake)
include(cmake/sources/Integral.cmake)
include(cmake/sources/Root.cmake)
include(cmake/sources/Hecke.cmake)
include(cmake/sources/Campaign.cmake)
include(cmake/sources/Console.cmake)
include(cmake/sources/HeckeExt.cmake)
include(cmake/sources/Util.cmake)
include(cmake/sources/NumberField.cmake)
include(cmake/sources/Orbifold.cmake)
include(cmake/sources/Binary.cmake)
include(cmake/sources/Special.cmake)
include(cmake/slot26/HeckeZUp.cmake)   # slot 26 (open): one fragment per package
include(cmake/slot26/HeckeZLocalPar.cmake)
include(cmake/slot26/HeckeZEisUp.cmake)
include(cmake/slot26/HeckeHidaDiscLambda.cmake)


# Link the executable against the ModArithmetic and HeckeCore libraries and
# the CUDA runtime.
target_link_libraries(cohomology_app PRIVATE
    ModArithmetic
    MatrixEngine
    HeckeCore
    EngineCuda
    ConsoleCore
    NumberField
)
if(OC_ENABLE_CUDA)
    target_link_libraries(cohomology_app PRIVATE CUDA::cudart)
endif()

# --- Build switches into the binary (slot-19 NB-0) ----------------------
# Always defined, to 0 or 1, so a TU can test them with #if (never #ifdef);
# read through src/hacs/BuildSwitches.h, which refuses to
# compile when they are missing. OC_BUILD_NOTEBOOK=1 is the `--serve` compile
# flag NB-2 keys on.
target_compile_definitions(cohomology_app PRIVATE
    OC_ENABLE_CUDA=$<BOOL:${OC_ENABLE_CUDA}>
    OC_BUILD_DOCS=$<BOOL:${OC_BUILD_DOCS}>
    OC_BUILD_NOTEBOOK=$<BOOL:${OC_BUILD_NOTEBOOK}>
)

# (S24 CL-L0, CLEANUP plan Sec. 2.1) OC_ENABLE_LEGACY is ALWAYS defined, to 0 or 1, on every target that
# sees the engine headers -- PUBLIC on the libraries, PRIVATE on cohomology_app -- the ENGINE_SHADOW_LEGACY
# pattern below: the macro will guard class members, so every TU must agree (ODR).  Code tests it with
# `#if OC_ENABLE_LEGACY` / oc::legacy::kLegacyBuild (src/ModArithmetic/LegacyConfig.h, which
# #errors when it is undefined), never #ifdef.  ConsoleCore and NumberField (STL-only) never include it.
foreach(_oc_legacy_target ModArithmetic HeckeCore MatrixEngine EngineCuda)
    target_compile_definitions(${_oc_legacy_target} PUBLIC OC_ENABLE_LEGACY=$<BOOL:${OC_ENABLE_LEGACY}>)
endforeach()
target_compile_definitions(cohomology_app PRIVATE OC_ENABLE_LEGACY=$<BOOL:${OC_ENABLE_LEGACY}>)
include(cmake/legacy.cmake)   # the retired routes' sources, added only when OC_ENABLE_LEGACY=ON

# The optional components: included ONLY when switched on, so an OFF switch
# runs no find_program / find_package of its own (nothing required, nothing
# probed).
if(OC_BUILD_DOCS)
    include(cmake/OcDocs.cmake)
endif()
if(OC_BUILD_NOTEBOOK)
    include(cmake/OcNotebook.cmake)
endif()

target_compile_options(cohomology_app
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OC_CXX_WARNINGS}>
    # (slot-24 CL-M) -Wno-deprecated-declarations is gone (audit defect #26,
    # CODEAUDIT C-02): the `Matrix` / `Matrix64` aliases no longer carry
    # [[deprecated]] (direct includes of Matrix.h are policed by
    # check_matrix_includes.sh instead), and the deprecated free functions and
    # EquivariantCohomologyOnK compatibility accessors were deleted. A
    # -Wdeprecated-declarations warning is therefore a real diagnostic again.
)

# --- Hecke action options ---
# HECKE_PURE_INTEGER disables the certified floating-point fast path in the
# Hecke sign predicates (Stage S1), leaving only the exact integer backend.
# Both settings must produce bit-identical Hecke matrices ([C, obl:fp] in
# docs/latex/hecke_action.tex). Wired here (S0); consumed starting Stage S1.
option(HECKE_PURE_INTEGER "Disable the FP-filtered fast path; use only exact integer arithmetic in the Hecke pipeline." OFF)

if(HECKE_PURE_INTEGER)
    # PUBLIC on HeckeCore: propagates to every target that links HeckeCore
    # (including cohomology_app and the Hecke validation sources compiled into
    # it), so the FP fast path in the Hecke sign predicates is compiled out
    # everywhere consistently (Stage S1, [C, obl:fp]).
    target_compile_definitions(HeckeCore PUBLIC HECKE_PURE_INTEGER)
endif()

# --- Matrix-engine Phase 2 options -------------------------------------
# ENGINE_SHADOW_LEGACY compiles the retrofitted arithmetic paths so that BOTH
# the pre-Phase-2 legacy code and the new engine::kernels code run on every
# call and their results are memcmp'd; a single differing byte throws
# std::logic_error naming the operation (see
# docs/matrix-engine/00-implementation-plan.md Sec. 4.3). It is OFF by default
# -- it roughly doubles the arithmetic cost and is a debugging / acceptance
# instrument, not a shipping configuration. The complementary switch is the
# RUNTIME one, engine::backend_is_legacy() in
# src/ModArithmetic/EngineConfig.h: `ENGINE_BACKEND=legacy` sends
# every retrofitted path back to the legacy code with no rebuild.
# (S24 CL-L1, CLEANUP plan Sec. 2.1) The comparison code is a retired route: it
# lives in the legacy/ directories (src/ModArithmetic/legacy/
# ShadowLegacy.inc, src/hacs/legacy/
# EquivariantCohomologyOnKShadow.inc, Hecke/legacy/HeckeOperatorOnKShadow.inc),
# so ENGINE_SHADOW_LEGACY=ON needs OC_ENABLE_LEGACY=ON.
option(ENGINE_SHADOW_LEGACY "Run legacy and engine paths and memcmp (needs OC_ENABLE_LEGACY=ON)" OFF)

if(ENGINE_SHADOW_LEGACY AND NOT OC_ENABLE_LEGACY)
    message(FATAL_ERROR "ENGINE_SHADOW_LEGACY=ON needs OC_ENABLE_LEGACY=ON: the shadow comparison is compiled "
                        "from the legacy/ sources (configure with -DOC_ENABLE_LEGACY=ON -DENGINE_SHADOW_LEGACY=ON, "
                        "or drop ENGINE_SHADOW_LEGACY)")
endif()
if(ENGINE_SHADOW_LEGACY)
    # All three targets: the retrofits live in ModArithmetic, but the macro
    # must agree across every TU that includes the retrofitted headers, and
    # HeckeCore does not link ModArithmetic (S1 design statement), so it is
    # set on each target rather than propagated.
    target_compile_definitions(ModArithmetic PUBLIC ENGINE_SHADOW_LEGACY=1)
    target_compile_definitions(HeckeCore PUBLIC ENGINE_SHADOW_LEGACY=1)
    target_compile_definitions(cohomology_app PRIVATE ENGINE_SHADOW_LEGACY=1)
    message(STATUS "ENGINE_SHADOW_LEGACY=ON: retrofitted paths run legacy AND engine and compare.")
endif()

# --- Matrix-engine Phase 3 options --------------------------------------
# ENGINE_COUNT_OBJECTS turns on the FlatMatrixT / BlockFlatMatrixT construction
# census of src/ModArithmetic/ObjectCounters.h, reported by
# `--bench <mode> --count-allocs` next to the malloc count. OFF by default and
# then entirely absent: the counting member is inside an #ifdef, so layout,
# sizeof and code are unchanged. See docs/matrix-engine/03-profiling-baseline.md
# Sec. 11.
option(ENGINE_COUNT_OBJECTS "Count FlatMatrix / BlockFlatMatrix constructions (diagnostics)" OFF)

if(ENGINE_COUNT_OBJECTS)
    # Must agree across every TU that includes FlatMatrix.h -- the member is
    # part of the class layout when enabled, so a mismatch would be an ODR
    # violation. Set on both targets that see those headers.
    target_compile_definitions(ModArithmetic PUBLIC ENGINE_COUNT_OBJECTS=1)
    target_compile_definitions(cohomology_app PRIVATE ENGINE_COUNT_OBJECTS=1)
    message(STATUS "ENGINE_COUNT_OBJECTS=ON: FlatMatrix/BlockFlatMatrix constructions are counted.")
endif()

# OC_SOURCE_DIR lets ConsumerMigrationValidation locate the repository root so
# it can run the build-time grep scripts in docs/matrix-engine/tools/ (Phase 3
# "Validations": ConsumerMigrationValidation "asserts the raw-mod_*-loop count
# by a build-time grep script"). The validation falls back to walking up from
# /proc/self/exe when this path no longer exists (a moved or copied tree), so
# the definition is a fast path, not a hard dependency.
target_compile_definitions(cohomology_app PRIVATE OC_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# (R1 / rename S1+S2) The program is called `hacs`; the CMake TARGET keeps its name (every `target_sources(cohomology_app ...)`
# fragment and `make cohomology_app` in the kits stay valid).  The old name stays as a symlink next to the binary
# (cohomology_app -> hacs) for two slot closes and one cluster/census run.  /proc/self/exe resolves the link, so
# the frozen help text says `hacs` whichever name was typed.
set_target_properties(cohomology_app PROPERTIES OUTPUT_NAME hacs)
# Non-fatal (R1 audit N5): on a file system without symlinks the alias is skipped with a warning, the build succeeds.
add_custom_command(TARGET cohomology_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DOC_LINK_DIR=$<TARGET_FILE_DIR:cohomology_app> -P ${CMAKE_SOURCE_DIR}/cmake/OcAlias.cmake
    COMMENT "alias: cohomology_app -> hacs"
    VERBATIM)

# (hecke-extensions slot 4, WP-1 / E2a) OC_GIT_COMMIT -- the `commit` key of
# the extension record ([E] `sec:schema`, [X] Sec. 1.4: "the hash baked in
# by CMake, like OC_SOURCE_DIR").  Resolved at CONFIGURE time, so it moves only
# on a re-configure (a documents-only commit included) at a different HEAD --
# not strictly on every commit, since a build tree is not always reconfigured
# for one; hecke_ext::SourceCommit() returns "unknown" when the tool is
# absent or the tree is not a checkout.  It is metadata: no golden object and
# no campaign job id reads it, and nothing branches on it.
# (hecke-extensions slot 6, WP-6) It USED TO also feed
# HeckeExtDeterminismValidation's digest, making that digest non-reproducible
# across checkouts (slot-5 WP-3 finding 3); the validation now redacts
# `commit` before hashing (Golden/GoldenHarness.h's `GoldenRedactJsonKeys`),
# so the digest reads this value no longer.
find_package(Git QUIET)
# (slot-24 BUILD-SLIM, F7) The commit goes into ONE generated TU (oc::GitCommit(), src/hacs/BuildInfo.h),
# not into a compile definition: a definition (target-wide or per source) lands in flags.make, which every object of
# the Makefile generator depends on, so a reconfigure at a new HEAD recompiled all ~550 TUs.  configure_file rewrites
# the output only when its content changes, so an unchanged HEAD touches nothing.
# (S26 INTEG, R1 audit S1) The file is written by cmake/OcBuildInfo.cmake at configure time AND on every build (target
# oc_buildinfo), so the commit is the one BUILT, with `-dirty` in `--version` when a tracked file differs from HEAD.
# The binding contracts of the release, printed by `hacs --version` (metadata; bump with the signed versions).  Checked
# below against the \date lines of the three documents.  (S26 V011b, HACS 0.1.1) The title-page-only update, signed by
# the owner 2026-11-10 as [C] 1.26.4, [E] 1.25.3, [U] 0.7.3.
# (S26 C127, 2026-11-11) [C] 1.27.0, signed by the owner 2026-11-11 (= 1.27.0-pre.1, text unchanged).
# (S26 U081, 2026-11-11) [U] 0.8.1, signed by the owner 2026-11-11 (= 0.8.0, status sentences of part 7 only).
set(OC_CONTRACT_VERSIONS "[C] 1.27.0, [E] 1.25.3, [U] 0.8.1")
# (S26 INTEG, R1 audit M1) Guard: a document whose \date line names a RELEASE version (no -pre / -rc suffix) must name
# exactly that version in OC_CONTRACT_VERSIONS; a pre-release of the next version is accepted while the string names the
# last signed one.  (S26 V011) The entry is parsed per tag and compared whole (a FIND of "[C] 1.26.4" also matched
# "[C] 1.26.4-pre.1"); when the string names a pre-release, the document must name exactly that pre-release.
# A source tree without the documents skips the check.
foreach(oc_doc "C:gl2_contract" "E:hecke_extensions" "U:hida_hecke_algebra")
    string(REPLACE ":" ";" oc_doc_l "${oc_doc}")
    list(GET oc_doc_l 0 oc_tag)
    list(GET oc_doc_l 1 oc_file)
    set(oc_tex "${CMAKE_SOURCE_DIR}/docs/latex/${oc_file}.tex")
    if(EXISTS "${oc_tex}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${oc_tex}")
        file(STRINGS "${oc_tex}" oc_date REGEX "^\\\\date\\{Version " LIMIT_COUNT 1)
        string(REGEX MATCH "Version ([0-9]+\\.[0-9]+\\.[0-9]+)([-.][A-Za-z0-9.]*)?" oc_m "${oc_date}")
        set(oc_v "${CMAKE_MATCH_1}")
        set(oc_suffix "${CMAKE_MATCH_2}")
        if(oc_v STREQUAL "")
            message(FATAL_ERROR "contract guard: no \\date{Version ...} line in ${oc_tex}")
        endif()
        string(REGEX MATCH "\\[${oc_tag}\\] ([0-9]+\\.[0-9]+\\.[0-9]+)(-[A-Za-z0-9.]+)?" oc_s "${OC_CONTRACT_VERSIONS}")
        set(oc_sv "${CMAKE_MATCH_1}")
        set(oc_ssuffix "${CMAKE_MATCH_2}")
        if(oc_sv STREQUAL "")
            message(FATAL_ERROR "contract guard: OC_CONTRACT_VERSIONS '${OC_CONTRACT_VERSIONS}' has no [${oc_tag}] entry")
        endif()
        if(oc_suffix STREQUAL "")
            if(NOT (oc_sv STREQUAL oc_v AND oc_ssuffix STREQUAL ""))
                message(FATAL_ERROR "contract guard: ${oc_file}.tex is version ${oc_v} (a release) but OC_CONTRACT_VERSIONS is '${OC_CONTRACT_VERSIONS}'; update CMakeLists.txt")
            endif()
        elseif(NOT oc_ssuffix STREQUAL "" AND NOT "${oc_sv}${oc_ssuffix}" STREQUAL "${oc_v}${oc_suffix}")
            message(FATAL_ERROR "contract guard: OC_CONTRACT_VERSIONS names [${oc_tag}] ${oc_sv}${oc_ssuffix} (a pre-release) but ${oc_file}.tex is version ${oc_v}${oc_suffix}; update CMakeLists.txt")
        endif()
    endif()
endforeach()
set(OC_BUILDINFO_OUT ${CMAKE_BINARY_DIR}/generated/OcBuildInfo.cpp)
set(OC_BUILDINFO_ARGS -DOC_SRC=${CMAKE_SOURCE_DIR} -DOC_IN=${CMAKE_SOURCE_DIR}/cmake/OcBuildInfo.cpp.in
    -DOC_OUT=${OC_BUILDINFO_OUT} "-DOC_GIT=${GIT_EXECUTABLE}" -DOC_HACS_VERSION=${OC_HACS_VERSION}
    "-DOC_CONTRACT_VERSIONS=${OC_CONTRACT_VERSIONS}")
execute_process(COMMAND ${CMAKE_COMMAND} ${OC_BUILDINFO_ARGS} -P ${CMAKE_SOURCE_DIR}/cmake/OcBuildInfo.cmake
    RESULT_VARIABLE oc_bi_rc)
if(NOT oc_bi_rc EQUAL 0)
    message(FATAL_ERROR "cmake/OcBuildInfo.cmake failed (${oc_bi_rc})")
endif()
add_custom_target(oc_buildinfo
    COMMAND ${CMAKE_COMMAND} ${OC_BUILDINFO_ARGS} -P ${CMAKE_SOURCE_DIR}/cmake/OcBuildInfo.cmake
    BYPRODUCTS ${OC_BUILDINFO_OUT}
    COMMENT "build info: commit of the built tree"
    VERBATIM)
add_dependencies(cohomology_app oc_buildinfo)
target_sources(cohomology_app PRIVATE ${OC_BUILDINFO_OUT})
set_source_files_properties(${CMAKE_BINARY_DIR}/generated/OcBuildInfo.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
target_include_directories(cohomology_app PRIVATE ${CMAKE_SOURCE_DIR}/src/hacs)
