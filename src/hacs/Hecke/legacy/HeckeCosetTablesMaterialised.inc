// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Hecke/legacy/HeckeCosetTablesMaterialised.inc -- the MATERIALISED rho_n tables of HeckeCosetTablesT, the old route of
// the retired coset streaming lever HECKE_COSET_STREAM=0 / ScopedCosetStreaming(false) (slot-14 WP-10, 3580160a;
// retired by S24 CL-R1, owner ruling 2 of 2026-10-30; CODEAUDIT-B B6).  Extracted verbatim from HeckeCosetTables.cpp
// (the three `if (!streamed_)` fills became the three materialise_*() members; the dense reference accessors are
// unchanged).  Included at the end of HeckeCosetTables.cpp under `#if OC_ENABLE_LEGACY`, before the explicit
// instantiations; declarations in HeckeCosetTables.h under the same guard.  In an OFF build every table is streamed:
// the *_block() accessors evaluate rho_of(...) of the stored exact matrix (the same bytes the old table stored).

#if !OC_ENABLE_LEGACY
#error "HeckeCosetTablesMaterialised.inc is an OC_ENABLE_LEGACY=ON source"
#endif

// The rho_n tables, each block a pure function of its stored exact matrix, filled after the exact data (on the
// consumer pool under ScopedParallelSetup; not at all on a streamed table).
template<class Elt>
void HeckeCosetTablesT<Elt>::materialise_rho_t() {
    rho_t_.assign(static_cast<size_t>(m_) + 1, FlatMatrixT<Elt>());
    if (!streamed_) {
        HeckeOperatorOptions::for_each_index(m_, parallel_setup_, [&](size_t j) {
            const uint32_t i = static_cast<uint32_t>(j) + 1;
            rho_t_[i] = rho_of(t_[i]);
        }, static_cast<uint64_t>(n_sym_ + 1) * (n_sym_ + 1));
    }
}

template<class Elt>
void HeckeCosetTablesT<Elt>::materialise_rho_cocycle_upper() {
    rho_cocycle_upper_.assign(4, std::vector<FlatMatrixT<Elt>>(static_cast<size_t>(m_) + 1));
    // rho_n(h_i(gamma)) per (gen, coset): a pure function of the stored exact
    // cocycle (slot-14 WP-10: filled after the exact data, on the consumer
    // pool under ScopedParallelSetup; not at all on a streamed table).
    if (!streamed_) {
        HeckeOperatorOptions::for_each_index(static_cast<size_t>(4) * m_, parallel_setup_, [&](size_t j) {
            const uint32_t gen = static_cast<uint32_t>(j / m_), i = static_cast<uint32_t>(j % m_) + 1;
            rho_cocycle_upper_[gen][i] = rho_of(cocycle_upper_[gen][i]);
        }, static_cast<uint64_t>(n_sym_ + 1) * (n_sym_ + 1));
    }
}

template<class Elt>
void HeckeCosetTablesT<Elt>::materialise_rho_s_inv_cocycle() {
    // The rho_n tables, each block a pure function of its stored exact matrix
    // (slot-14 WP-10: filled after the exact data, on the consumer pool under
    // ScopedParallelSetup; not at all on a streamed table).
    if (!streamed_) {
        HeckeOperatorOptions::for_each_index(m_, parallel_setup_, [&](size_t j) {
            const uint32_t i = static_cast<uint32_t>(j) + 1;
            rho_s_inv_[i] = rho_of(s_inv_[i]);
        }, static_cast<uint64_t>(n_sym_ + 1) * (n_sym_ + 1));
        HeckeOperatorOptions::for_each_index(static_cast<size_t>(4) * m_, parallel_setup_, [&](size_t j) {
            const uint32_t gen = static_cast<uint32_t>(j / m_), i = static_cast<uint32_t>(j % m_) + 1;
            rho_cocycle_[gen][i] = rho_of(cocycle_[gen][i]);
        }, static_cast<uint64_t>(n_sym_ + 1) * (n_sym_ + 1));
    }
}

// The reference accessors of the rho_n tables REFUSE a streamed table (ENGINE_CHECK).
template<class Elt>
const FlatMatrixT<Elt>& HeckeCosetTablesT<Elt>::rho_cocycle(uint32_t gen, uint32_t i) const {
    check_gen(gen); check_index(i);
    ENGINE_CHECK_MSG(!streamed_, "HeckeCosetTables::rho_cocycle: this table was built under the coset streaming lever (ScopedCosetStreaming) and stores no rho_n table; read it through rho_cocycle_block().");
    return rho_cocycle_[gen][i];
}
template<class Elt>
const FlatMatrixT<Elt>& HeckeCosetTablesT<Elt>::rho_s_inv(uint32_t i) const {
    check_index(i);
    ENGINE_CHECK_MSG(!streamed_, "HeckeCosetTables::rho_s_inv: this table was built under the coset streaming lever (ScopedCosetStreaming) and stores no rho_n table; read it through rho_s_inv_block().");
    return rho_s_inv_[i];
}
template<class Elt>
const FlatMatrixT<Elt>& HeckeCosetTablesT<Elt>::rho_t(uint32_t i) const {
    check_index(i);
    ENGINE_CHECK_MSG(!streamed_, "HeckeCosetTables::rho_t: this table was built under the coset streaming lever (ScopedCosetStreaming) and stores no rho_n table; read it through rho_t_block().");
    return rho_t_[i];
}
template<class Elt>
const FlatMatrixT<Elt>& HeckeCosetTablesT<Elt>::rho_cocycle_upper(uint32_t gen, uint32_t i) const {
    check_gen(gen); check_index(i);
    ENGINE_CHECK_MSG(!streamed_, "HeckeCosetTables::rho_cocycle_upper: this table was built under the coset streaming lever (ScopedCosetStreaming) and stores no rho_n table; read it through rho_cocycle_upper_block().");
    return rho_cocycle_upper_[gen][i];
}
