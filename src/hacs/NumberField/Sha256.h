// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Sha256.h -- FIPS 180-4 SHA-256 (slot-19 WP-NF2), STL only. Used to compare
// the Coates--Wiles layer's lambda_m digit arrays with R19-7's numpy hashes
// (hashlib.sha256 of the (W, n) int64 little-endian digit array).
#include <cstddef>
#include <cstdint>
#include <string>

namespace nf {

class Sha256 {
public:
    Sha256();
    void update(const void* data, std::size_t len);
    void update_u64le(uint64_t v);
    std::string hex();  // finalises; the object must not be updated afterwards

private:
    void block(const uint8_t* p);
    uint32_t h_[8];
    uint8_t buf_[64];
    std::size_t buflen_ = 0;
    uint64_t total_ = 0;
    bool done_ = false;
};

}  // namespace nf
