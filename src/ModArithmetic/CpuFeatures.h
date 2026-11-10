// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef ENGINE_CPU_FEATURES_H
#define ENGINE_CPU_FEATURES_H
// =====================================================================
// CpuFeatures.h -- ONE CPU-ISA probe and ONE IsaPath enum (S24 CL-1;
// CODEAUDIT C-06; docs/implementation/slot24/CLEANUP_plan_1030.md)
// =====================================================================
//
// Header-only, STL-only.  Nothing is migrated here (CL-5 does that).  Today
// eight translation units run their own __builtin_cpu_supports probe, and they
// test DIFFERENT feature sets for "AVX2" and "AVX-512".  The predicates below
// name each set, so a migration keeps its module's exact test:
//
//   avx2()              "avx2"                        Ring2k, BitMatrix, FlatMatrix, DescentSmithKernels,
//                                                     HeckeFormsHeckeAlgebra
//   avx2_fma()          "avx2" && "fma"               Kernels.cpp cpu_has_avx2 (the AVX2K microkernel)
//   avx512f()           "avx512f"                     BitMatrix, FlatMatrix, DescentSmithKernels
//   avx512bw()          "avx512bw"                    TritMatrix
//   avx512_f_bw_dq_vl() f && bw && dq && vl           Kernels.cpp cpu_has_avx512, Ring2k has_avx512
//   avx512_vnni()       the above && "avx512vnni"     Kernels.cpp (the VNNI route)
//
// These are CPU facts only.  Whether a module COMPILED its AVX body
// (ENGINE_HAVE_AVX512_TARGET, R2K_HAVE_X86, ...) stays the module's own
// condition: a module reports "available" = its build flag && the predicate.
// On a non-x86 build (aarch64: the Jetson, Termux) every predicate is false.
//
// IsaPath: the one enum for a forced path.  The existing enums differ in
// enumerator ORDER (kernels::KernelPath {Auto, Portable, Avx512, Avx2} vs
// ring2k::KernelPath {Auto, Portable, Avx2, Avx512}), so a migration converts
// by NAME (isa_path_name / isa_path_from_name), never by integer value.
// isa_path_from_name accepts exactly ENGINE_KERNEL's spellings: "portable"
// (alias "scalar"), "avx2", "avx512", "auto"; anything else is false.
// =====================================================================

#include <cstring>

namespace engine {
namespace cpu {

struct Features {
    bool avx2 = false, fma = false, avx512f = false, avx512bw = false, avx512dq = false, avx512vl = false, avx512vnni = false;
};

// The probe, run once per process.
inline const Features& features() {
    static const Features f = [] {
        Features r;
#if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();
        r.avx2 = __builtin_cpu_supports("avx2") != 0;
        r.fma = __builtin_cpu_supports("fma") != 0;
        r.avx512f = __builtin_cpu_supports("avx512f") != 0;
        r.avx512bw = __builtin_cpu_supports("avx512bw") != 0;
        r.avx512dq = __builtin_cpu_supports("avx512dq") != 0;
        r.avx512vl = __builtin_cpu_supports("avx512vl") != 0;
        r.avx512vnni = __builtin_cpu_supports("avx512vnni") != 0;
#endif
        return r;
    }();
    return f;
}

inline bool avx2() { return features().avx2; }
inline bool avx2_fma() { return features().avx2 && features().fma; }
inline bool avx512f() { return features().avx512f; }
inline bool avx512bw() { return features().avx512bw; }
inline bool avx512_f_bw_dq_vl() {
    const Features& f = features();
    return f.avx512f && f.avx512bw && f.avx512dq && f.avx512vl;
}
inline bool avx512_vnni() { return avx512_f_bw_dq_vl() && features().avx512vnni; }

enum class IsaPath : unsigned char { Auto, Portable, Avx2, Avx512 };

inline const char* isa_path_name(IsaPath p) {
    switch (p) {
    case IsaPath::Portable: return "portable";
    case IsaPath::Avx2: return "avx2";
    case IsaPath::Avx512: return "avx512";
    case IsaPath::Auto: break;
    }
    return "auto";
}

inline bool isa_path_from_name(const char* s, IsaPath& out) {
    if (s == nullptr) return false;
    if (std::strcmp(s, "portable") == 0 || std::strcmp(s, "scalar") == 0) out = IsaPath::Portable;
    else if (std::strcmp(s, "avx2") == 0) out = IsaPath::Avx2;
    else if (std::strcmp(s, "avx512") == 0) out = IsaPath::Avx512;
    else if (std::strcmp(s, "auto") == 0) out = IsaPath::Auto;
    else return false;
    return true;
}

} // namespace cpu
} // namespace engine

#endif // ENGINE_CPU_FEATURES_H
