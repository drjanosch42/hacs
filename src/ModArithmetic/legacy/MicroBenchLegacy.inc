// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ModArithmetic/legacy/MicroBenchLegacy.inc -- the `legacy` column of the micro-benchmarks (`--bench micro
// --backend legacy`, console `micro(..., backend = "legacy")`, ENGINE_BACKEND=legacy PerformanceSmokeValidation):
// GemmLegacy, GemvLegacy, RrefLegacy, KernelLegacy, TransposeLegacy, each pinned to the pre-Phase-2 substrate by an
// engine::elim::ScopedLegacy guard.  Moved verbatim out of PerformanceSmokeValidation.cpp by S24 CL-L2 (CLEANUP plan
// Sec. 2.2 / 2.4; owner ruling 1 of 2026-10-30).  Included inside namespace micro's anonymous namespace under
// `#if OC_ENABLE_LEGACY`; in an OFF build the dispatchers return NotCompiledLegacy and ParseBackend refuses "legacy".
#if !OC_ENABLE_LEGACY
#error "MicroBenchLegacy.inc is compiled only with OC_ENABLE_LEGACY=ON"
#endif

// ---------------------------------------------------------------------------
// Legacy implementations -- FlatMatrix exactly as it stands today. These are
// the "before" side of the Phase-2 like-for-like comparison and must NOT be
// modified when the engine lands.
// ---------------------------------------------------------------------------

MicroResult GemmLegacy(uint32_t n, uint32_t k, uint32_t ell, uint32_t repeat,
                       const std::string& params) {
    MicroResult res;
    res.op = "gemm";
    res.params = params;
    res.work_modmul = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(k);
    // (Phase 2) FlatMatrix::operator* now multiplies through the engine by
    // default; the guard pins this arm to the PRE-Phase-2 loop, exactly as
    // RrefLegacy pins elimination.  See the block comment above the engine
    // implementations for what the two backend columns mean.
    engine::elim::ScopedLegacy legacy_guard;

    const FlatMatrix A = RandomMatrix(n, k, ell, 0x01);
    const FlatMatrix B = RandomMatrix(k, n, ell, 0x02);

    std::vector<double> ts;
    Fnv h;
    for (uint32_t i = 0; i < repeat; ++i) {
        const auto t0 = Clock::now();
        const FlatMatrix C = A * B;
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) h.feed_matrix(C);
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    return res;
}

MicroResult GemvLegacy(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "gemv";
    res.params = params;
    res.work_modmul = static_cast<double>(n) * static_cast<double>(n);
    engine::elim::ScopedLegacy legacy_guard;   // (Phase 2) see GemmLegacy

    const FlatMatrix A = RandomMatrix(n, n, ell, 0x03);
    const FlatMatrix x = RandomMatrix(n, 1, ell, 0x04);

    std::vector<double> ts;
    Fnv h;
    for (uint32_t i = 0; i < repeat; ++i) {
        const auto t0 = Clock::now();
        const FlatMatrix y = A * x;
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) h.feed_matrix(y);
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    return res;
}

MicroResult RrefLegacy(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                       uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "rref";
    res.params = params;
    // (Phase 2) FlatMatrix now eliminates through the engine by default; the
    // guard pins this arm to the PRE-Phase-2 scalar core so "legacy" stays the
    // before side of the comparison.
    engine::elim::ScopedLegacy legacy_guard;

    const FlatMatrix M0 = RankedMatrix(rows, cols, rank_target, ell);

    std::vector<double> ts;
    Fnv h;
    uint32_t rank_found = 0;
    for (uint32_t i = 0; i < repeat; ++i) {
        // Fresh copy: FlatMatrix's copy constructor deliberately does NOT copy
        // the RREF cache (FlatMatrix.cpp:252-258), so every repeat eliminates
        // from scratch.
        FlatMatrix M = M0;
        const auto t0 = Clock::now();
        const FlatMatrix R = M.get_row_echelon_form_as_flat_matrix();
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) {
            h.feed_matrix(R);
            rank_found = M.rank(); // cached by the call above; not timed
        }
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    res.rank = rank_found;
    res.work_modmul = ElimWork(rows, cols, rank_found);
    return res;
}

MicroResult KernelLegacy(uint32_t rows, uint32_t cols, uint32_t rank_target, uint32_t ell,
                         uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "kernel";
    res.params = params;
    engine::elim::ScopedLegacy legacy_guard;   // (Phase 2) see RrefLegacy

    const FlatMatrix M0 = RankedMatrix(rows, cols, rank_target, ell);

    std::vector<double> ts;
    Fnv h;
    uint32_t nullity = 0, rank_found = 0;
    for (uint32_t i = 0; i < repeat; ++i) {
        FlatMatrix M = M0;
        const auto t0 = Clock::now();
        const std::vector<FlatMatrix> ker = M.kernel();
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) {
            nullity = static_cast<uint32_t>(ker.size());
            h.feed(nullity);
            for (const auto& v : ker) h.feed_matrix(v);
            rank_found = cols - nullity;
        }
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    res.rank = rank_found;
    res.nullity = nullity;
    res.work_modmul = ElimWork(rows, cols, rank_found);
    return res;
}

MicroResult TransposeLegacy(uint32_t n, uint32_t ell, uint32_t repeat, const std::string& params) {
    MicroResult res;
    res.op = "transpose";
    res.params = params;
    res.work_modmul = 0.0; // data movement, not modmuls
    engine::elim::ScopedLegacy legacy_guard;   // (Phase 2) see GemmLegacy

    const FlatMatrix A = RandomMatrix(n, n, ell, 0x05);

    std::vector<double> ts;
    Fnv h;
    for (uint32_t i = 0; i < repeat; ++i) {
        const auto t0 = Clock::now();
        const FlatMatrix T = A.transpose();
        const auto t1 = Clock::now();
        ts.push_back(secs(t0, t1));
        if (i == 0) h.feed_matrix(T);
    }
    res.min_s = *std::min_element(ts.begin(), ts.end());
    res.median_s = Median(ts);
    res.checksum = h.value();
    std::ostringstream note;
    const double elems = static_cast<double>(n) * static_cast<double>(n);
    note << std::fixed << std::setprecision(2)
         << (res.min_s > 0.0 ? elems / res.min_s / 1e9 : 0.0) << " Gelem/s";
    res.note = note.str();
    return res;
}

