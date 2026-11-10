// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "DenseBuf.h"

namespace engine {

// Explicit instantiations so the two supported widths are compiled once here
// (the class is header-only; this pins the symbols and catches template
// errors in the library build rather than in the first consumer).
template class DenseBuf<uint32_t>;
template class DenseBuf<uint64_t>;
template class DenseBuf<unsigned __int128>;   // (slot-11 WP-18a) the composite-only u128 element type

} // namespace engine
