# SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
# SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

# glibc has malloc_trim; bionic (Termux/Android) and musl do not. Where it is missing, force-include a no-op shim.
include(CheckCXXSymbolExists)
check_cxx_symbol_exists(malloc_trim malloc.h OC_HAVE_MALLOC_TRIM)
if(NOT OC_HAVE_MALLOC_TRIM)
    message(STATUS "malloc_trim not available: using no-op shim")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-include${CMAKE_CURRENT_LIST_DIR}/malloc_trim_shim.h>")
endif()
