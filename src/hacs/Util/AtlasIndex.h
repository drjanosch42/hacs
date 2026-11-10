// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// AtlasIndex.h -- the index sidecar `oc-atlas-index/1` of a `hecke_atlas/v1`
// store and the read-only mapping of the store ([P] v3.11 Sec. 18.16 (b);
// hecke-extensions slot 20, NB-4).
//
// One pure function of the store bytes, ScanStore, derives the byte offsets of
// the header, of every cell row and of every system row, plus the few system
// attributes the served filters need (r, type, collected, base, twist) and the
// occurrences of the collected systems.  The sidecar is that table serialised
// (Serialize / Parse), keyed to the store by its size and FNV-1a-64 and closed
// by an `end` line carrying the FNV of the index's own preceding bytes.  The
// writer (`cohomology_app --atlas --index`, HeckeExt/HeckeAtlasCli.cpp) and the
// reader (`cohomology_app --serve-atlas`, Console/Bindings/ServeAtlas.cpp) are
// the only users; the server re-derives the table at every open and compares
// it with the sidecar field by field (Equal), so a sidecar is never trusted.
//
// READ-ONLY BY CONSTRUCTION.  MappedFile opens O_RDONLY and maps PROT_READ /
// MAP_PRIVATE; nothing in this unit can write the store.
//
// Placement: cohomology_app, always compiled (no OC_BUILD_NOTEBOOK guard: the
// writer is a token of `--atlas`).  STL + POSIX + ModArithmetic/Fnv.h only; no
// Console/, no HeckeExt/ header (Console/Bindings/ may not include HeckeExt/,
// check_heckeext_includes.sh rule 1), so both sides can include it.
// ===========================================================================
#pragma once
#ifndef OC_UTIL_ATLAS_INDEX_H
#define OC_UTIL_ATLAS_INDEX_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace atlas_index {

constexpr const char* kIndexMagic = "#oc-atlas-index/1";
constexpr const char* kStoreSchema = "hecke_atlas/v1";
constexpr const char* kIndexSuffix = ".ocidx";

struct CellRow {
    std::uint32_t n = 0;
    std::uint64_t off = 0;
    std::uint64_t len = 0;
};

struct SysRow {
    std::uint64_t off = 0;
    std::uint64_t len = 0;
    std::uint32_t r = 0;
    char type = 'c';              // 'e' eisenstein, 'c' cuspidal
    std::uint8_t collected = 0;   // 0 / 1
    std::uint32_t base = 0;
    std::uint32_t twist = 0;
};

struct OccRow {
    std::uint32_t n = 0;
    std::uint32_t ideal = 0;
    std::uint32_t sid = 0;
};

struct Index {
    std::string store_name;
    std::uint64_t store_bytes = 0;
    std::uint64_t store_fnv = 0;
    std::string schema;
    std::uint32_t ell = 0;
    std::uint32_t n_max = 0;
    std::uint64_t head_off = 0, head_len = 0;
    std::uint64_t tail_off = 0, tail_len = 0;
    std::vector<CellRow> cells;     // ascending n
    std::vector<SysRow> systems;    // row index = system id
    std::vector<OccRow> occ;        // ascending (n, ideal, sid)
};

// FNV-1a-64 over a byte range, continuing `h` -- byte for byte the rule of
// engine::FnvBytes (ModArithmetic/Fnv.h), with its constants.
std::uint64_t FnvRange(std::uint64_t h, const char* p, std::size_t n);
std::uint64_t FnvOf(const char* p, std::size_t n);   // from the offset basis

// "0x" + 16 lower-case hex digits.
std::string Hex16(std::uint64_t v);
// Parses "0x" + 1..16 hex digits.
bool ParseHex64(const std::string& s, std::uint64_t& out);

// Derives the index of a store from its bytes.  `store_fnv` is recorded as
// given (the caller hashes once).  Returns false with a one-line reason when the
// bytes are not exactly the layout AtlasStore::to_json writes.
bool ScanStore(const char* data, std::size_t size, const std::string& store_name, std::uint64_t store_fnv,
               Index& out, std::string& error);

// The sidecar text (with its `end` line) and its reader.  Parse checks the
// magic, the field order and the `end` hash; it does not look at a store.
std::string Serialize(const Index& ix);
bool Parse(const std::string& text, Index& out, std::string& error);

// Field-by-field comparison; `diff` names the first difference.
bool Equal(const Index& a, const Index& b, std::string& diff);

// A read-only mapping of a regular file (O_RDONLY | O_CLOEXEC, PROT_READ,
// MAP_PRIVATE).  An empty file is refused (mmap of length 0).
class MappedFile {
public:
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();
    bool open(const std::string& path, std::string& error);
    const char* data() const { return data_; }
    std::size_t size() const { return size_; }
    int fd() const { return fd_; }
    // madvise hints (no effect on the bytes): sequential read-ahead, and the
    // release of the resident pages after a full pass.
    void advise_sequential() const;
    void release_resident() const;

private:
    int fd_ = -1;
    const char* data_ = nullptr;
    std::size_t size_ = 0;
};

// The basename of a path ("a/b/c.json" -> "c.json").
std::string BaseName(const std::string& path);

} // namespace atlas_index

#endif // OC_UTIL_ATLAS_INDEX_H
