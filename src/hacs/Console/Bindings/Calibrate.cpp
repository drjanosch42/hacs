// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Calibrate.cpp -- see Calibrate.h.  A port of docs/console/tools/nbconfirm_calibrate.py (main, run_one, startup, the profile
// writer), with the probes run as children of this binary and each child's peak RSS taken from wait4().

#include "Calibrate.h"

#include "HostProbe.h"
#include "../Core/CostModel.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace console {
namespace calibrate {

namespace {

struct Probe {
    const char* cls;
    bool bench;
    std::vector<std::string> arg;   // bench argv, or { OC code }
    const char* digest_re;          // nullptr = none
    bool serial;
};

// calib/v1 == calib/v2 probe set (S2 Sec. 4.2), ~25 s on the reference box at R = 1.
const std::vector<Probe>& Probes() {
    static const std::vector<Probe> P = {
        {"zdesc", false, {"set coefficient_ring = \"Z\"; r = descent(1, 60); print(r.digest)"}, "(0x[0-9a-f]+)", false},
        {"zrec", false, {"set coefficient_ring = \"Z\"; r = hecke_structure(12, 10, \"full\"); print(1)"}, nullptr, true},
        {"u128", false, {"x = level_n(30, 12, 7); print(1)"}, nullptr, true},
        {"heil", false, {"x = nebentypus(31, 1, 30, 7, [13]); print(1)"}, nullptr, true},
        {"ntt", false, {"for i in range(0, 60) { x = hida_lambda(101, 200, 6, [97, 89, 83, 79, 73, 71]) }; print(1)"}, nullptr, true},
        {"ladic", false, {"x = hida_tower(100, 5, 3); print(1)"}, nullptr, false},
        {"elim", true, {"--bench", "micro", "rref", "2048", "2048"}, "checksum=(0x[0-9a-f]+)", true},
        {"gemm", true, {"--bench", "micro", "gemm", "1024"}, "checksum=(0x[0-9a-f]+)", true},
        {"bw", true, {"--bench", "micro", "transpose", "4096"}, "checksum=(0x[0-9a-f]+)", true},
        {"hecke", true, {"--bench", "hecke", "192", "134217757", "2"}, nullptr, false},
    };
    return P;
}
const char* const kAnchors[] = {"zdesc", "zrec", "u128", "heil", "ntt", "ladic"};

struct Child {
    int rc = -1;
    double wall = 0.0, rss_mib = 0.0;
    std::string out;
};

bool ParseMask(const std::string& spec, cpu_set_t& set, unsigned& n) {
    CPU_ZERO(&set);
    n = 0;
    std::size_t a = 0;
    while (a <= spec.size()) {
        std::size_t b = spec.find(',', a);
        if (b == std::string::npos) b = spec.size();
        const std::string t = spec.substr(a, b - a);
        const std::size_t d = t.find('-');
        char* e = nullptr;
        const long lo = std::strtol(t.c_str(), &e, 10);
        long hi = lo;
        if (d != std::string::npos) hi = std::strtol(t.c_str() + d + 1, &e, 10);
        if (t.empty() || lo < 0 || hi < lo || hi >= CPU_SETSIZE) return false;
        for (long c = lo; c <= hi; ++c) {
            if (!CPU_ISSET(c, &set)) ++n;
            CPU_SET(c, &set);
        }
        a = b + 1;
    }
    return n > 0;
}

Child RunChild(const std::string& exe, const std::vector<std::string>& args, unsigned threads, const cpu_set_t* mask) {
    Child c;
    int fd[2];
    if (::pipe(fd) != 0) return c;
    std::vector<std::string> envs;
    for (char** e = environ; e && *e; ++e) {
        const std::string s = *e;
        if (s.rfind("ENGINE_CUDA_DISABLE=", 0) == 0 || s.rfind("ENGINE_THREADS=", 0) == 0) continue;
        envs.push_back(s);
    }
    envs.push_back("ENGINE_CUDA_DISABLE=1");
    envs.push_back("ENGINE_THREADS=" + std::to_string(threads));
    std::vector<char*> envp, argv;
    for (std::string& s : envs) envp.push_back(&s[0]);
    envp.push_back(nullptr);
    std::vector<std::string> av{exe};
    av.insert(av.end(), args.begin(), args.end());
    for (std::string& s : av) argv.push_back(&s[0]);
    argv.push_back(nullptr);
    const auto t0 = std::chrono::steady_clock::now();
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(fd[0]);
        ::close(fd[1]);
        return c;
    }
    if (pid == 0) {
        ::dup2(fd[1], 1);
        ::dup2(fd[1], 2);
        ::close(fd[0]);
        ::close(fd[1]);
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0) ::dup2(devnull, 0);
        if (mask != nullptr) ::sched_setaffinity(0, sizeof(cpu_set_t), mask);
        ::execve(exe.c_str(), argv.data(), envp.data());
        ::_exit(127);
    }
    ::close(fd[1]);
    char buf[4096];
    for (;;) {
        const ssize_t r = ::read(fd[0], buf, sizeof buf);
        if (r <= 0) break;
        c.out.append(buf, static_cast<std::size_t>(r));
    }
    ::close(fd[0]);
    int status = 0;
    struct rusage ru;
    std::memset(&ru, 0, sizeof ru);
    ::wait4(pid, &status, 0, &ru);
    c.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    c.rss_mib = static_cast<double>(ru.ru_maxrss) / 1024.0;   // Linux: KiB
    c.rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    return c;
}

std::string Fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

int Usage(const std::string& why) {
    std::cerr << "error: " << why << "\n"
              << "usage: hacs --calibrate[=thorough] [--repeat R] [--threads T] [--taskset CPUS]\n"
              << "                      [--profile PATH | --no-write] [--emit-reference]\n";
    return 1;
}

} // namespace

namespace {

// (S3 Fix round 1, audit m5) the scratch-emitter rule: the device profile is never written inside the source tree.  The path's
// longest existing ancestor is resolved (the profile and its parent directories may not exist yet), then the rest is appended.
std::string ResolveForWrite(const std::string& path) {
    std::string p = path;
    if (p.empty() || p[0] != '/') {
        char cwd[PATH_MAX];
        if (::getcwd(cwd, sizeof cwd) == nullptr) return std::string();
        p = std::string(cwd) + "/" + p;
    }
    std::string head = p, tail;
    for (;;) {
        char buf[PATH_MAX];
        if (::realpath(head.c_str(), buf) != nullptr) return std::string(buf) + tail;
        const std::size_t k = head.find_last_of('/');
        if (k == std::string::npos || head.empty()) return std::string();
        tail = head.substr(k) + tail;
        head = k == 0 ? std::string("/") : head.substr(0, k);
        if (head == "/" && tail.empty()) return std::string();
    }
}

bool InsideSourceTree(const std::string& path, std::string& src) {
#ifdef OC_SOURCE_DIR
    char buf[PATH_MAX];
    src = ::realpath(OC_SOURCE_DIR, buf) != nullptr ? std::string(buf) : std::string(OC_SOURCE_DIR);
#else
    src.clear();
    return false;
#endif
    const std::string r = ResolveForWrite(path);
    if (r.empty()) return true;   // unresolvable: refuse, as the emitters do
    return r == src || (r.size() > src.size() && r.compare(0, src.size(), src) == 0 && r[src.size()] == '/');
}

} // namespace

int RunCalibrate(int argc, char** argv) {
    const std::string tok = argc > 0 ? argv[0] : "--calibrate";
    int repeat = tok == "--calibrate=thorough" ? 3 : 1;
    unsigned ncpu = host::AffinityCpuCount();
    unsigned threads = std::min(16u, ncpu);
    bool threads_given = false, no_write = false, emit = false;
    std::string profile_path, taskset;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](std::string& out) {
            if (i + 1 >= argc) return false;
            out = argv[++i];
            return true;
        };
        std::string v;
        if (a == "--repeat" && val(v)) {
            repeat = std::atoi(v.c_str());
            if (repeat < 1) return Usage("--repeat needs R >= 1");
        } else if (a == "--threads" && val(v)) {
            const int t = std::atoi(v.c_str());
            if (t < 1 || t > 256) return Usage("--threads needs 1 <= T <= 256");
            threads = static_cast<unsigned>(t);
            threads_given = true;
        } else if (a == "--taskset" && val(taskset)) {
        } else if (a == "--profile" && val(profile_path)) {
        } else if (a == "--no-write") {
            no_write = true;
        } else if (a == "--emit-reference") {
            emit = true;
        } else {
            return Usage("unrecognised argument '" + a + "'");
        }
    }
    if (!no_write) {   // (S3 Fix round 1, audit m5) before any probe runs
        const std::string target = profile_path.empty() ? cost::DeviceProfilePath() : profile_path;
        std::string src;
        if (target != "none" && InsideSourceTree(target, src)) {
            std::cout << "calibrate: REFUSED: the device profile " << target << " lies inside the source tree (" << src
                      << "); write it outside the tree (--profile PATH) or use --no-write\n";
            return 1;
        }
    }
    cpu_set_t mask;
    const cpu_set_t* maskp = nullptr;
    if (!taskset.empty()) {
        unsigned n = 0;
        if (!ParseMask(taskset, mask, n)) return Usage("bad --taskset '" + taskset + "'");
        maskp = &mask;
        ncpu = n;
        if (!threads_given) threads = std::min(16u, n);
        threads = std::min(threads, n);   // the multi-threaded slot cannot use more CPUs than the mask holds
    }
    const std::string exe = host::SelfExe();
    if (exe.empty()) return Usage("cannot resolve /proc/self/exe");
    const cost::CostModels& M = cost::EmbeddedCostModels();
    const std::string cset = M.calib_set.empty() ? std::string("calib/v2") : M.calib_set;

    const Child su = RunChild(exe, {"-c", "set threads = 1; print(1)"}, 1, maskp);
    if (su.rc != 0) {
        std::cerr << "calibrate: the start-up probe failed (exit " << su.rc << "): " << su.out.substr(0, 300) << "\n";
        return 1;
    }
    struct Meas {
        double sec = 0.0, rss = 0.0;
        unsigned thr = 1;
    };
    std::map<std::pair<std::string, std::string>, Meas> meas;
    std::vector<std::pair<std::string, std::string>> order;
    std::map<std::string, std::string> digests;
    std::set<std::string> unavailable;
    const auto T0 = std::chrono::steady_clock::now();
    int nrun = 0, ntot = 0;
    for (const Probe& p : Probes()) ntot += p.serial ? 1 : 2;
    for (const Probe& p : Probes()) {
        for (const char* slot : {"1", "mt"}) {
            const unsigned thr = std::string(slot) == "1" ? 1u : threads;
            if (p.serial && std::string(slot) == "mt") {   // thread-independent class: the mt slot is the 1 slot
                auto it = meas.find({p.cls, "1"});
                if (it != meas.end()) {
                    Meas m = it->second;
                    m.thr = thr;
                    meas[{p.cls, "mt"}] = m;
                    order.push_back({p.cls, "mt"});
                }
                continue;
            }
            if (unavailable.count(p.cls)) continue;
            ++nrun;
            std::cerr << "calibrate: probe " << nrun << " of " << ntot << ": " << p.cls << " threads=" << thr << " ..." << std::endl;
            bool have = false;
            Meas best;
            best.thr = thr;
            for (int r = 0; r < repeat; ++r) {
                std::vector<std::string> args =
                    p.bench ? p.arg : std::vector<std::string>{"-c", "set threads = " + std::to_string(thr) + "; " + p.arg[0]};
                const Child c = RunChild(exe, args, thr, maskp);
                if (c.rc != 0) {
                    if (c.out.find("not implemented in this build") != std::string::npos ||
                        c.out.find("Unknown") != std::string::npos || c.rc == 5) {
                        unavailable.insert(p.cls);
                        break;
                    }
                    std::cerr << "calibrate: " << p.cls << ": exit " << c.rc << ": " << c.out.substr(c.out.size() > 300 ? c.out.size() - 300 : 0)
                              << "\n";
                    std::cout << "NBCONFIRM_CALIB set=" << cset << " verdict=probe_failed class=" << p.cls << "\n";
                    return 1;
                }
                double w = c.wall;
                if (p.bench) {   // the BENCH line's own timer (excludes process start-up)
                    std::smatch m;
                    if (std::regex_search(c.out, m, std::regex("median_s=([0-9.eE+-]+)")) ||
                        std::regex_search(c.out, m, std::regex("total_s=([0-9.eE+-]+)")))
                        w = std::atof(m[1].str().c_str());
                } else {
                    w = std::max(w - su.wall, 1e-4);
                }
                if (!have || w < best.sec) best.sec = w;
                if (!have || c.rss_mib < best.rss) best.rss = c.rss_mib;
                have = true;
                if (p.digest_re != nullptr) {
                    std::smatch m;
                    digests[p.cls] = std::regex_search(c.out, m, std::regex(p.digest_re)) ? m[1].str() : "?";
                }
            }
            if (have) {
                meas[{p.cls, slot}] = best;
                order.push_back({p.cls, slot});
            }
        }
    }
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
    if (emit) {
        for (const auto& k : order) {
            const Meas& m = meas[k];
            std::cout << "calib\tcalib/v2\t" << k.first << "\t" << k.second << "\t" << m.thr << "\t" << Fmt("%.4f", m.sec) << "\t"
                      << (digests.count(k.first) ? digests[k.first] : std::string("-")) << "\t" << Fmt("%.1f", m.rss) << "\n";
        }
        std::cerr << "# calib/v2 reference: total " << Fmt("%.1f", total) << " s at repeat " << repeat << ", startup "
                  << Fmt("%.3f", su.wall) << " s\n";
        return 0;
    }
    std::vector<std::string> bad;
    for (const auto& d : digests) {
        auto it = M.calib.find({d.first, "1"});
        if (it != M.calib.end() && it->second.digest != "-" && it->second.digest != d.second) bad.push_back(d.first);
    }
    auto ref_at = [&](const std::string& cls, unsigned T) {
        // the reference box's time at T threads: Amdahl through its 1- and 16-thread probe times
        auto r1 = M.calib.find({cls, "1"}), r16 = M.calib.find({cls, "mt"});
        if (r1 == M.calib.end()) return std::nan("");
        if (r16 == M.calib.end() || T <= 1) return r1->second.seconds;
        const double t16 = r16->second.threads;
        const double f = std::min(1.0, std::max(0.0, (r16->second.seconds / r1->second.seconds - 1 / t16) / (1 - 1 / t16)));
        return r1->second.seconds * (f + (1 - f) / T);
    };
    struct Row {
        std::string cls, slot;
        unsigned thr;
        double sec, ref, factor, rss, rss_ratio;
        bool fallback;
    };
    std::vector<Row> rows;
    for (const auto& k : order) {
        const Meas& m = meas[k];
        const double rs = ref_at(k.first, m.thr);
        auto rc = M.calib.find(k);
        const double rr = (rc != M.calib.end() && rc->second.rss_mib > 0) ? m.rss / rc->second.rss_mib : std::nan("");
        rows.push_back({k.first, k.second, m.thr, m.sec, rs, m.sec / rs, m.rss, rr, false});
    }
    for (const std::string& cls : unavailable) {   // the geometric mean of the available verb anchors of that slot
        for (const char* slot : {"1", "mt"}) {
            double s = 0;
            int n = 0;
            for (const Row& r : rows)
                if (r.slot == slot && !r.fallback && std::find(std::begin(kAnchors), std::end(kAnchors), r.cls) != std::end(kAnchors) &&
                    !unavailable.count(r.cls) && r.factor == r.factor) {
                    s += std::log(r.factor);
                    ++n;
                }
            auto rc = M.calib.find({cls, slot});
            rows.push_back({cls, slot, std::string(slot) == "1" ? 1u : threads, std::nan(""),
                            rc != M.calib.end() ? rc->second.seconds : std::nan(""), n ? std::exp(s / n) : std::nan(""),
                            std::nan(""), std::nan(""), true});
        }
    }
    std::cout << "CALIB class slot threads device_s ref_s factor rss_mib rss_ratio\n";
    for (const Row& r : rows) {
        char b[256];
        std::snprintf(b, sizeof b, "CALIB %-6s %-2s %3u %9.4f %9.4f %7.3f %8.1f %6.3f", r.cls.c_str(), r.slot.c_str(), r.thr, r.sec,
                      r.ref, r.factor, r.rss, r.rss_ratio);
        std::cout << b << "\n";
    }
    const std::string verdict = bad.empty() ? "OK" : "digest_mismatch";
    std::string un;
    for (const std::string& u : unavailable) un += (un.empty() ? "" : ",") + u;
    std::cout << "NBCONFIRM_CALIB set=" << cset << " threads=" << threads << " probes=" << rows.size() << " total_s=" << Fmt("%.1f", total)
              << " startup_s=" << Fmt("%.3f", su.wall) << " taskset=" << (taskset.empty() ? "-" : taskset)
              << " unavailable=" << (un.empty() ? "-" : un) << " verdict=" << verdict << "\n";
    // The drift gate (S2 FR.6): the verb anchors' wall factor and RSS ratio against [0.75, 1.33].
    {
        std::string ww = "-", wr = "-";
        double dw = 0, dr = 0;
        bool have_rss = false, drift = false;
        for (const Row& r : rows) {
            if (r.fallback || std::find(std::begin(kAnchors), std::end(kAnchors), r.cls) == std::end(kAnchors)) continue;
            const double lw = std::fabs(std::log(r.factor));
            if (lw > dw || ww == "-") {
                dw = lw;
                ww = r.cls + "/" + r.slot + ":" + Fmt("%.3f", r.factor);
            }
            if (r.factor < 0.75 || r.factor > 1.33) drift = true;
            if (r.rss_ratio == r.rss_ratio) {
                have_rss = true;
                const double lr = std::fabs(std::log(r.rss_ratio));
                if (lr > dr || wr == "-") {
                    dr = lr;
                    wr = r.cls + "/" + r.slot + ":" + Fmt("%.3f", r.rss_ratio);
                }
                if (r.rss_ratio < 0.75 || r.rss_ratio > 1.33) drift = true;
            }
        }
        const std::string bid = host::ElfBuildId(exe);
        std::cout << "COST_MODEL_DRIFT set=" << cset << " build_id=" << (bid.empty() ? "-" : bid) << " worst_wall=" << ww
                  << " worst_rss=" << wr << " verdict=" << (drift ? "DRIFT" : have_rss ? "OK" : "NO_REFERENCE_RSS")
                  << " (meaningful on the reference box only; the model's fitted_on = "
                  << (M.fitted_on.empty() ? std::string("-") : M.fitted_on.front()) << ")\n";
    }
    if (!bad.empty()) return 2;
    if (no_write) return 0;
    if (profile_path.empty()) profile_path = cost::DeviceProfilePath();
    if (profile_path == "none") return 0;
    {   // mkdir -p of the parent
        std::string dir = profile_path.substr(0, profile_path.find_last_of('/'));
        for (std::size_t i = 1; i <= dir.size(); ++i)
            if (i == dir.size() || dir[i] == '/') ::mkdir(dir.substr(0, i).c_str(), 0755);
    }
    const std::string tmp = profile_path + ".tmp";
    {
        std::ofstream o(tmp.c_str());
        if (!o) {
            std::cerr << "calibrate: cannot write " << tmp << "\n";
            return 1;
        }
        struct utsname un_;
        std::string node = "?", mach = "?";
        if (::uname(&un_) == 0) {
            node = un_.nodename;
            mach = un_.machine;
        }
        std::uint64_t size = 0;
        std::int64_t mtime = 0;
        host::FileStat(exe, size, mtime);
        const std::string bid = host::ElfBuildId(exe);
        char ts[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
        o << "# hacs device profile (NBCONFIRM S3 " << cset
          << "). Advisory only: scales cost ESTIMATES, never a value, record or hash.\n";
        o << "profile_version\t2\ncalib_set\t" << cset << "\nhost\t" << node << "\nmachine\t" << mach << "\n";
        o << "nproc\t" << ncpu << "\nthreads_mt\t" << threads << "\ntaskset\t" << (taskset.empty() ? "-" : taskset) << "\nstartup_s\t"
          << Fmt("%.4f", su.wall) << "\n";
        o << "binary\t" << exe << "\nbinary_size\t" << size << "\nbinary_mtime\t" << mtime << "\n";
        o << "binary_build_id\t" << (bid.empty() ? "-" : bid) << "\n";
        o << "unavailable\t" << (un.empty() ? "-" : un) << "\n";
        o << "created_utc\t" << ts << "\n";
        for (const Row& r : rows)
            o << "factor\t" << r.cls << "\t" << r.slot << "\t" << r.thr << "\t" << Fmt("%.4f", r.factor) << "\t" << Fmt("%.4f", r.sec)
              << "\t" << (r.fallback ? "fallback" : "measured") << "\t" << Fmt("%.1f", r.rss) << "\n";
    }
    if (std::rename(tmp.c_str(), profile_path.c_str()) != 0) {
        std::cerr << "calibrate: cannot rename " << tmp << "\n";
        return 1;
    }
    std::cout << "wrote " << profile_path << "\n";
    return 0;
}

} // namespace calibrate
} // namespace console
