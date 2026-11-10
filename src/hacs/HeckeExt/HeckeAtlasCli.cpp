// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// (hecke-extensions slot 7, WP-5 / phase E6 cont.) HeckeExt/HeckeAtlasCli --
// see the header.
// ===========================================================================

#include "HeckeAtlasCli.h"

#include "HeckeAtlas.h"

#include "../../ModArithmetic/Fnv.h"
#include "../Hecke/HeckeSettings.h"
#include "../Util/AtlasIndex.h"
#include "../Util/Rss.h"

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

namespace hecke_ext {

namespace {

using Clock = std::chrono::steady_clock;

double Secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

using oc_util::PeakRssMiB;

void Usage() {
    std::cout << "usage: hacs --atlas <ell> [--n-max N] [--collect-only] [--quiet]\n"
                 "       hacs --atlas --close <store.json>\n"
                 "       hacs --atlas --verify <store.json>\n"
                 "       hacs --atlas --index <store.json> [--out <path>]\n"
                 "       hacs --atlas --query <store.json> '<request JSON>'   (OC_BUILD_NOTEBOOK=ON)\n"
                 "The store is written only under HECKE_ATLAS_OUT=<dir> (never under\n"
                 "docs/matrix-engine/results/).  --index writes the oc-atlas-index/1\n"
                 "sidecar (default <store.json>.ocidx; never the store itself, never\n"
                 "inside the source tree).  See HeckeExt/HeckeAtlasCli.h.\n";
}

// realpath of an existing path, or "".
std::string RealPath(const std::string& p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) == nullptr) return std::string();
    return std::string(buf);
}

// (slot-20 NB-4, [P] v3.11 Sec. 18.16 (b)) `--atlas --index <store> [--out <path>]`:
// the one writer of the oc-atlas-index/1 sidecar.  The store is opened O_RDONLY
// and mapped PROT_READ (atlas_index::MappedFile); the index goes to
// <path>.tmp.<pid> (O_EXCL) and is renamed.  Exit 0 written, 1 refused, 3 the
// store is unreadable or not indexable.
int RunAtlasIndex(const std::string& store, const std::string& out_arg) {
    const std::string store_real = RealPath(store);
    if (store_real.empty()) {
        std::cout << "--atlas --index: cannot read " << store << ": " << std::strerror(errno) << std::endl;
        return 3;
    }
    // The default sits beside the REAL file (--serve-atlas looks for <realpath>.ocidx).
    const std::string out = out_arg.empty() ? store_real + atlas_index::kIndexSuffix : out_arg;
    const std::size_t slash = out.find_last_of('/');
    const std::string out_dir =
        slash == std::string::npos ? std::string(".") : (slash == 0 ? std::string("/") : out.substr(0, slash));
    const std::string out_base = slash == std::string::npos ? out : out.substr(slash + 1);
    const std::string dir_real = RealPath(out_dir);
    if (dir_real.empty() || out_base.empty() || out_base == "." || out_base == "..") {
        std::cout << "--atlas --index: refused: the output directory of " << out << " does not exist" << std::endl;
        return 1;
    }
    const std::string target = (dir_real == "/" ? std::string() : dir_real) + "/" + out_base;
    if (target == store_real || RealPath(target) == store_real) {
        std::cout << "--atlas --index: refused: the output " << out << " is the store itself (the store is never written)"
                  << std::endl;
        return 1;
    }
    if (target.find("/docs/matrix-engine/results/") != std::string::npos) {
        std::cout << "--atlas --index: refused: " << out << " lies under docs/matrix-engine/results/" << std::endl;
        return 1;
    }
#ifdef OC_SOURCE_DIR
    {
        const std::string src = RealPath(OC_SOURCE_DIR);
        if (!src.empty() && target.compare(0, src.size() + 1, src + "/") == 0) {
            std::cout << "--atlas --index: refused: " << out << " lies inside the source tree " << src
                      << " (the sidecar lives beside the store)" << std::endl;
            return 1;
        }
    }
#endif
    atlas_index::MappedFile mf;
    std::string err;
    if (!mf.open(store_real, err)) {
        std::cout << "--atlas --index: " << err << std::endl;
        return 3;
    }
    mf.advise_sequential();
    const std::uint64_t fnv = atlas_index::FnvOf(mf.data(), mf.size());
    atlas_index::Index ix;
    if (!atlas_index::ScanStore(mf.data(), mf.size(), atlas_index::BaseName(store_real), fnv, ix, err)) {
        std::cout << "--atlas --index: the store is not indexable: " << err << std::endl;
        return 3;
    }
    const std::string text = atlas_index::Serialize(ix);
    const std::string tmp = target + ".tmp." + std::to_string(::getpid());
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        std::cout << "--atlas --index: cannot create " << tmp << ": " << std::strerror(errno) << std::endl;
        return 1;
    }
    std::size_t off = 0;
    while (off < text.size()) {
        const ssize_t n = ::write(fd, text.data() + off, text.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            std::cout << "--atlas --index: write to " << tmp << " failed: " << std::strerror(errno) << std::endl;
            ::close(fd);
            ::unlink(tmp.c_str());
            return 1;
        }
        off += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0 || ::close(fd) != 0 || ::rename(tmp.c_str(), target.c_str()) != 0) {
        std::cout << "--atlas --index: cannot finish " << target << ": " << std::strerror(errno) << std::endl;
        ::unlink(tmp.c_str());
        return 1;
    }
    std::cout << "ATLAS_INDEX store_bytes=" << ix.store_bytes << " store_fnv=" << atlas_index::Hex16(ix.store_fnv)
              << " cells=" << ix.cells.size() << " systems=" << ix.systems.size() << " occurrences=" << ix.occ.size()
              << " index_bytes=" << text.size()
              << " index_fnv=" << atlas_index::Hex16(atlas_index::FnvOf(text.data(), text.size()))
              << " out=" << target << std::endl;
    return 0;
}

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("--atlas: cannot read " + path);
    std::ostringstream o;
    o << in.rdbuf();
    return o.str();
}

// The residue-degree profile "r:count,..." of a set of systems.
std::string Profile(const AtlasStore& s, bool collected_only) {
    std::map<uint32_t, uint64_t> by_r;
    for (const AtlasSystem& sys : s.systems)
        if (!collected_only || sys.collected) ++by_r[sys.r];
    std::string out;
    for (const auto& kv : by_r)
        out += (out.empty() ? "" : ",") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
    return out.empty() ? "-" : out;
}

std::string List(const std::vector<uint64_t>& v) {
    std::string out;
    for (uint64_t q : v) out += (out.empty() ? "" : ",") + std::to_string(q);
    return "{" + out + "}";
}

void Summary(const AtlasStore& s, bool closed) {
    uint64_t collected = 0, cusp = 0, eis = 0, rational = 0, in_h1 = 0;
    uint64_t a_known = 0, a_unknown = 0;
    std::map<std::string, uint64_t> reasons;
    for (const AtlasSystem& sys : s.systems) {
        if (sys.collected) ++collected;
        if (sys.type == "cuspidal") ++cusp;
        if (sys.type == "eisenstein") ++eis;
        if (sys.r == 1u) ++rational;
        if (sys.in_H1) ++in_h1;
        for (const AtlasOccurrence& o : sys.occurrences) {
            if (o.a_ell.known) ++a_known;
            else { ++a_unknown; ++reasons[o.a_ell.reason]; }
        }
    }
    std::string why;
    for (const auto& kv : reasons)
        why += (why.empty() ? "" : ",") + kv.first + ":" + std::to_string(kv.second);
    std::cout << "ATLAS_SUMMARY ell=" << s.ell << " n_max=" << s.n_max << " cells=" << s.cells.size()
              << " collected=" << collected << " sigma=" << s.systems.size()
              << " rational=" << rational << " cuspidal=" << cusp << " eisenstein=" << eis
              << " collected_in_H1=" << in_h1 << " fields=" << s.fields.size()
              << " collected_r_profile=" << Profile(s, true)
              << " sigma_r_profile=" << Profile(s, false)
              << " a_ell_known=" << a_known << " a_ell_unknown=" << a_unknown
              << " a_ell_unknown_reasons=" << (why.empty() ? "-" : why);
    if (closed) {
        std::cout << " chi_count=" << s.chi_count << " chi_in_sigma=" << s.chi_in_sigma
                  << " s_sep=" << List(s.s_sep.primes) << " s_sep_order=" << List(s.s_sep.greedy_order)
                  << " s_sep_separates=" << (s.s_sep.separates ? 1 : 0)
                  << " s_sep_targets=" << s.s_sep.targets << " s_sep_unseparated="
                  << s.s_sep.unseparated << " s_sep_undecided=" << s.s_sep.undecided
                  << " s0_anemic=" << List(s.s0_anemic.primes)
                  << " s0_separates=" << (s.s0_anemic.separates ? 1 : 0)
                  << " s0_unseparated=" << s.s0_anemic.unseparated;
    }
    std::cout << std::endl;
}

void StoreLine(const AtlasStore& s, const std::string& path) {
    const std::string json = s.to_json();
    std::cout << "ATLAS_STORE ell=" << s.ell << " bytes=" << json.size() << " fnv1a64=0x" << std::hex
              << engine::FnvBytes(engine::kFnvOffset64, json) << std::dec
              << " path=" << (path.empty() ? std::string("(not written: HECKE_ATLAS_OUT unset)") : path)
              << std::endl;
}

} // namespace

int RunAtlasCli(int argc, char** argv, std::function<void()> between_cells) {
    // (slot-8 WP-3) [E] sec:ring (b4): Sigma(ell) is an F_ell atlas.  Exit 5,
    // the NotImplementedInThisBuild code of every other surface.
    try {
        hecke::RequireFieldCoefficientRing("--atlas (HeckeAtlasCli)");
    } catch (const hecke::NotImplementedInThisBuild& e) {
        std::cout << "--atlas: " << e.what() << std::endl;
        return 5;
    }
    try {
        std::vector<std::string> a(argv, argv + argc);
        if (a.empty() || a[0] == "--help" || a[0] == "-h") {
            Usage();
            return a.empty() ? 1 : 0;
        }
        if (a[0] == "--query") {
            // (slot-22 VERB-3) served by main.cpp when OC_BUILD_NOTEBOOK=ON
            // (the request handler lives in Console/Bindings/ServeAtlas.cpp,
            // which this directory may not name); reaching here means OFF.
            std::cerr << "--atlas --query: needs a build with OC_BUILD_NOTEBOOK=ON (NotImplementedInThisBuild)" << std::endl;
            return 5;
        }
        if (a[0] == "--index") {
            // (slot-20 NB-4) the oc-atlas-index/1 sidecar writer.
            if (a.size() == 2) return RunAtlasIndex(a[1], std::string());
            if (a.size() == 4 && a[2] == "--out") return RunAtlasIndex(a[1], a[3]);
            Usage();
            return 1;
        }
        const Clock::time_point t0 = Clock::now();
        if (a[0] == "--close" || a[0] == "--verify") {
            if (a.size() != 2) { Usage(); return 1; }
            const std::string text = ReadFile(a[1]);
            AtlasStore s = ParseAtlasStore(text);
            CloseAtlas(s);
            const double close_s = Secs(t0, Clock::now());
            Summary(s, true);
            if (a[0] == "--verify") {
                const bool same = (s.to_json() == text);
                std::cout << "ATLAS_VERIFY ell=" << s.ell << " reclosed_identical=" << (same ? 1 : 0)
                          << " close_s=" << close_s << " peak_rss_mb=" << PeakRssMiB() << std::endl;
                StoreLine(s, "");
                return same ? 0 : 2;
            }
            const std::string path = WriteAtlasStoreIfRequested(s);
            std::cout << "ATLAS_TIME close_s=" << close_s << " peak_rss_mb=" << PeakRssMiB()
                      << std::endl;
            StoreLine(s, path);
            return 0;
        }

        char* end = nullptr;
        const unsigned long ell = std::strtoul(a[0].c_str(), &end, 10);
        if (end == a[0].c_str() || *end != '\0' || ell < 5u || ell > 100000u) { Usage(); return 1; }
        AtlasOptions opts;
        bool collect_only = false, quiet = false;
        for (size_t i = 1; i < a.size(); ++i) {
            if (a[i] == "--n-max" && i + 1 < a.size()) {
                opts.n_limit = static_cast<uint32_t>(std::strtoul(a[++i].c_str(), nullptr, 10));
            } else if (a[i] == "--collect-only") {
                collect_only = true;
            } else if (a[i] == "--quiet") {
                quiet = true;
            } else {
                Usage();
                return 1;
            }
        }
        if (!quiet) opts.progress = &std::cout;
        opts.between_cells = std::move(between_cells);   // slot-8 WP-5
        Atlas at = CollectAtlas(static_cast<uint32_t>(ell), opts);
        const double collect_s = Secs(t0, Clock::now());
        double close_s = 0.0;
        if (!collect_only) {
            const Clock::time_point c0 = Clock::now();
            CloseAtlas(at.store);
            close_s = Secs(c0, Clock::now());
        }
        double pipeline_s = 0.0, cf_s = 0.0;
        uint32_t r_max = 0;
        for (const AtlasCellTiming& t : at.timings) {
            pipeline_s += t.total_s;
            cf_s += t.canonical_field_s;
            if (t.r_max > r_max) r_max = t.r_max;
        }
        Summary(at.store, !collect_only);
        const std::string path = WriteAtlasStoreIfRequested(at.store, collect_only ? ".collected" : "");
        std::cout << "ATLAS_TIME ell=" << ell << " wall_s=" << Secs(t0, Clock::now())
                  << " collect_s=" << collect_s << " close_s=" << close_s
                  << " driver_total_s=" << pipeline_s << " canonical_field_s=" << cf_s
                  << " canonical_field_over_total=" << (pipeline_s > 0 ? cf_s / pipeline_s : 0.0)
                  << " r_max=" << r_max << " peak_rss_mb=" << PeakRssMiB() << std::endl;
        StoreLine(at.store, path);
        return 0;
    } catch (const std::exception& e) {
        std::cout << "--atlas: EXCEPTION: " << e.what() << std::endl;
        return 3;
    }
}

} // namespace hecke_ext
