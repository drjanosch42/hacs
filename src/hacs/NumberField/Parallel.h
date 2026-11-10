// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
// Parallel.h -- a minimal std::thread parallel-for of the NumberField library
// (slot-19 WP-NF2). STL only. Every work item is a contiguous index range and
// writes only its own outputs, so the result never depends on the thread count.
#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace nf {

// f(begin, end, worker) over [0, n) in at most `threads` contiguous ranges.
template <class F>
void parallel_ranges(std::size_t n, unsigned threads, F&& f) {
    if (n == 0) return;
    const std::size_t T = std::max<std::size_t>(1, std::min<std::size_t>(threads ? threads : 1, n));
    if (T == 1) {
        f(std::size_t(0), n, 0u);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(T);
    std::exception_ptr err;
    std::mutex mu;
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t b = n * t / T, e = n * (t + 1) / T;
        pool.emplace_back([&, b, e, t] {
            try {
                f(b, e, static_cast<unsigned>(t));
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu);
                if (!err) err = std::current_exception();
            }
        });
    }
    for (auto& th : pool) th.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace nf
