// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ProgramCache.cpp -- see ProgramCache.h.
#include "ProgramCache.h"

#include <sstream>
#include <tuple>

namespace engine {
namespace tape {

bool ShapeKey::operator<(const ShapeKey& o) const {
    return std::tie(tag, elt_width, regime, d_V, extra, pattern_hash, row_dims, col_dims) <
           std::tie(o.tag, o.elt_width, o.regime, o.d_V, o.extra, o.pattern_hash, o.row_dims, o.col_dims);
}
bool ShapeKey::operator==(const ShapeKey& o) const { return !(*this < o) && !(o < *this); }

uint64_t ShapeKey::hash() const {
    uint64_t h = fnv_init();
    for (char c : tag) h = fnv_mix(h, static_cast<uint8_t>(c));
    h = fnv_mix(h, elt_width); h = fnv_mix(h, regime); h = fnv_mix(h, d_V); h = fnv_mix(h, extra);
    h = fnv_mix(h, pattern_hash);
    h = fnv_mix(h, row_dims.size()); for (uint32_t d : row_dims) h = fnv_mix(h, d);
    h = fnv_mix(h, col_dims.size()); for (uint32_t d : col_dims) h = fnv_mix(h, d);
    return h;
}

std::string ShapeKey::to_string() const {
    std::ostringstream s;
    s << tag << " grid " << row_dims.size() << "x" << col_dims.size() << " d_V=" << d_V
      << " regime=" << static_cast<int>(regime) << " w=" << static_cast<int>(elt_width)
      << " pattern=" << std::hex << pattern_hash << " extra=" << extra << std::dec;
    return s.str();
}

template<class Elt>
uint64_t pattern_hash_of(const BlockMat<Elt>& A) {
    uint64_t h = fnv_init();
    h = fnv_mix(h, A.nbr()); h = fnv_mix(h, A.nbc());
    for (uint32_t x : A.row_off()) h = fnv_mix(h, x);
    for (uint32_t x : A.col_off()) h = fnv_mix(h, x);
    std::map<uint32_t, uint32_t> ids;   // pool id -> first-occurrence number
    for (uint32_t br = 0; br < A.nbr(); ++br) {
        h = fnv_mix(h, A.row_end(br) - A.row_begin(br));
        for (uint32_t k = A.row_begin(br); k < A.row_end(br); ++k) {
            const BlockRef<Elt>& r = A.ref_at(k);
            h = fnv_mix(h, A.col_at(k));
            h = fnv_mix(h, static_cast<uint64_t>(r.kind));
            h = fnv_mix(h, r.alpha); h = fnv_mix(h, r.beta);
            h = fnv_mix(h, r.flags);
            if (r.has_data()) {
                auto it = ids.find(r.pool_id);
                if (it == ids.end()) it = ids.emplace(r.pool_id, static_cast<uint32_t>(ids.size())).first;
                h = fnv_mix(h, it->second + 1); h = fnv_mix(h, r.r0); h = fnv_mix(h, r.c0);
            } else h = fnv_mix(h, 0);
        }
    }
    return h;
}

template<class Elt>
std::shared_ptr<const Program<Elt>> ProgramCache<Elt>::find(const ShapeKey& key) {
    std::lock_guard<std::mutex> lk(m_);
    auto it = table_.find(key);
    if (it == table_.end()) { ++stats_.misses; return nullptr; }
    ++stats_.hits;
    return it->second;
}

template<class Elt>
std::shared_ptr<const Program<Elt>> ProgramCache<Elt>::insert(const ShapeKey& key, Program<Elt>&& prog) {
    ENGINE_CHECK_MSG(prog.compiled, "ProgramCache::insert: only compiled programs are cached");
    std::lock_guard<std::mutex> lk(m_);
    auto sp = std::make_shared<const Program<Elt>>(std::move(prog));
    table_[key] = sp;
    ++stats_.inserts;
    return sp;
}

template<class Elt> size_t ProgramCache<Elt>::size() const { std::lock_guard<std::mutex> lk(m_); return table_.size(); }
template<class Elt> void ProgramCache<Elt>::clear() { std::lock_guard<std::mutex> lk(m_); table_.clear(); }
template<class Elt> typename ProgramCache<Elt>::Stats ProgramCache<Elt>::stats() const { std::lock_guard<std::mutex> lk(m_); return stats_; }

template<class Elt>
ProgramCache<Elt>& ProgramCache<Elt>::global() { static ProgramCache<Elt> c; return c; }

template class ProgramCache<uint32_t>;
template class ProgramCache<uint64_t>;
template uint64_t pattern_hash_of<uint32_t>(const BlockMat<uint32_t>&);
template uint64_t pattern_hash_of<uint64_t>(const BlockMat<uint64_t>&);

} // namespace tape
} // namespace engine
