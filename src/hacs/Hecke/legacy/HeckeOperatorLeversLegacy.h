// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Hecke/legacy/HeckeOperatorLeversLegacy.h -- the setters and Scoped* guards of the RETIRED slot-13/14 levers
// HECKE_U_ELL_LOW_RANK, HECKE_COSET_STREAM, HECKE_PARALLEL_SETUP, HECKE_RANK1_CLOSED_FORM (S24 CL-R1; owner ruling 2
// of 2026-10-30; CODEAUDIT-B B6).  Moved verbatim from HeckeOperatorOnK.h, where the levers' descriptions and getters
// stay.  Included at the end of HeckeOperatorOnK.h under `#if OC_ENABLE_LEGACY` only: in an OFF build nothing can
// force the old routes (the getters return the default route, and the env opt-out `=0` is refused).
#include "../HeckeOperatorOnK.h"

#if !OC_ENABLE_LEGACY
#error "HeckeOperatorLeversLegacy.h is an OC_ENABLE_LEGACY=ON header"
#endif

namespace hecke {
namespace HeckeOperatorOptions {
    void set_u_ell_low_rank_mode(UEllLowRankMode m);
    class ScopedUEllLowRank {
        UEllLowRankMode saved_;
    public:
        explicit ScopedUEllLowRank(bool on) : saved_(u_ell_low_rank_mode()) {
            set_u_ell_low_rank_mode(on ? UEllLowRankMode::On : UEllLowRankMode::Off);
        }
        explicit ScopedUEllLowRank(UEllLowRankMode m) : saved_(u_ell_low_rank_mode()) { set_u_ell_low_rank_mode(m); }
        ~ScopedUEllLowRank() { set_u_ell_low_rank_mode(saved_); }
        ScopedUEllLowRank(const ScopedUEllLowRank&) = delete;
        ScopedUEllLowRank& operator=(const ScopedUEllLowRank&) = delete;
    };

    void set_coset_streaming(bool on);
    class ScopedCosetStreaming {
        bool saved_;
    public:
        explicit ScopedCosetStreaming(bool on) : saved_(coset_streaming()) { set_coset_streaming(on); }
        ~ScopedCosetStreaming() { set_coset_streaming(saved_); }
        ScopedCosetStreaming(const ScopedCosetStreaming&) = delete;
        ScopedCosetStreaming& operator=(const ScopedCosetStreaming&) = delete;
    };

    void set_parallel_setup(bool on);
    class ScopedParallelSetup {
        bool saved_;
    public:
        explicit ScopedParallelSetup(bool on) : saved_(parallel_setup()) { set_parallel_setup(on); }
        ~ScopedParallelSetup() { set_parallel_setup(saved_); }
        ScopedParallelSetup(const ScopedParallelSetup&) = delete;
        ScopedParallelSetup& operator=(const ScopedParallelSetup&) = delete;
    };

    void set_rank_one_closed_form(bool on);
    class ScopedRankOneClosedForm {
        bool saved_;
    public:
        explicit ScopedRankOneClosedForm(bool on) : saved_(rank_one_closed_form()) { set_rank_one_closed_form(on); }
        ~ScopedRankOneClosedForm() { set_rank_one_closed_form(saved_); }
        ScopedRankOneClosedForm(const ScopedRankOneClosedForm&) = delete;
        ScopedRankOneClosedForm& operator=(const ScopedRankOneClosedForm&) = delete;
    };

    // (moved with the levers: its only reader is HeckeUEllStreamValidation, now an ON-build validation)
    // Validation hook: the fan-out threshold on the calling thread (default
    // kParallelSetupMinWork; 0 fans out every phase of >= 2 items), so the
    // small cells of HeckeUEllStreamValidation exercise the parallel code.
    class ScopedParallelSetupMinWork {
        uint64_t saved_;
    public:
        explicit ScopedParallelSetupMinWork(uint64_t w);
        ~ScopedParallelSetupMinWork();
        ScopedParallelSetupMinWork(const ScopedParallelSetupMinWork&) = delete;
        ScopedParallelSetupMinWork& operator=(const ScopedParallelSetupMinWork&) = delete;
    };
}
} // namespace hecke
