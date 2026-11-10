// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// EngineElimPassOptionsValidation.cpp -- see the header.
//
// hecke-extensions slot 9, WP-6: engine findings 2 and 3 of the slot-8 WP-6b
// report.
//
//  (A) Nothing moves at the defaults: with no override of either kind
//      installed, elim_pass_options() == consumer_pass_options() ==
//      PassOptions::from_env(), field by field and by fingerprint.
//  (B) The fingerprint's device bit is the EFFECTIVE policy's.  For every
//      options value that sets the legacy pair and the policy together (every
//      from_env() mapping, ScopedThreadDevice, the console's `set device`)
//      the new fingerprint equals the pre-WP-6 formula, over all 2^7
//      combinations of the seven boolean pass options -- EXCEPT the `gpu`
//      mapping, which (slot-10 WP-8, decision 10 (A), finding 4) also carries
//      the ninth bit (256), the pre-WP-6 formula plus that bit exactly; an
//      options value that selects the device through `policy` alone
//      (HeckeExtDeterminismValidation's `gpu` arms) now carries bit 128
//      (finding 3) and, for `gpu` specifically, bit 256 as well (finding 4),
//      so `hybrid` and `gpu` no longer key the same slot.
//  (C) The elimination driver honours ScopedPassOptions (finding 2): under
//      batch = false every per-panel program is compiled with batch = false
//      (TapeElimStats::batch_off_programs == programs > 0), and the RREF is
//      entrywise the override-free one.  A ScopedThreadDevice still wins over
//      the process-wide override's device fields on its own thread.
//  (D) With a usable device: under a `policy = Gpu`-only override the
//      per-panel programs carry GPU-assigned nodes, and the RREF is
//      entrywise the CPU one.  [SKIP] without a device, and (slot-14 WP-2,
//      [C4]/[C7] Sec.2 (b)) also [SKIP] when the environment's OWN device
//      policy is already non-cpu (the "reference run: none" baseline this arm
//      compares against does not exist there).  Both (A) and (D) compute
//      their expectation from PassOptions::from_env() rather than assuming
//      it is cpu, so this validation is green under any ENGINE_DEVICE.
#include "EngineElimPassOptionsValidation.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../ModArithmetic/BlockElim.h"
#include "../Engine/DeviceBackend.h"
#include "../Engine/EngineRuntime.h"
#include "../Engine/EngineTestUtil.h"
#include "../Engine/Executor.h"
#include "../Engine/TapeElim.h"

namespace {

using namespace engine;
using namespace engine::tape;
using namespace engine::tape::test;

bool SameOptions(const PassOptions& a, const PassOptions& b) {
    return a.fold == b.fold && a.cse == b.cse && a.fuse == b.fuse && a.layout == b.layout &&
           a.ksplit == b.ksplit && a.batch == b.batch && a.arena == b.arena &&
           a.device_policy == b.device_policy && a.device_model == b.device_model && a.policy == b.policy;
}

// The pre-WP-6 fingerprint, verbatim (EngineRuntime.cpp before slot 9).
uint64_t OldFingerprint(const PassOptions& o) {
    uint64_t f = 0;
    f |= o.fold ? 1u : 0u;
    f |= o.cse ? 2u : 0u;
    f |= o.fuse ? 4u : 0u;
    f |= o.layout ? 8u : 0u;
    f |= o.ksplit ? 16u : 0u;
    f |= o.batch ? 32u : 0u;
    f |= o.arena ? 64u : 0u;
    f |= (o.device_model && o.device_policy == Device::GPU) ? 128u : 0u;
    return f;
}

struct ElimRun {
    blk::BlockRrefResult<uint32_t> res;
    TapeElimStats delta;
};

TapeElimStats Minus(const TapeElimStats& a, const TapeElimStats& b) {
    TapeElimStats d;
    d.programs = a.programs - b.programs;
    d.batch_off_programs = a.batch_off_programs - b.batch_off_programs;
    d.device_policy_programs = a.device_policy_programs - b.device_policy_programs;
    d.batched_programs = a.batched_programs - b.batched_programs;
    d.gpu_programs = a.gpu_programs - b.gpu_programs;
    d.gpu_nodes = a.gpu_nodes - b.gpu_nodes;
    return d;
}

ElimRun RunRref(Executor<uint32_t>& ex, const BlockMat<uint32_t>& A) {
    const TapeElimStats before = tape_elim_totals();
    ElimRun r;
    r.res = rref<uint32_t>(ex, A);
    r.delta = Minus(tape_elim_totals(), before);
    return r;
}

bool SameRref(const blk::BlockRrefResult<uint32_t>& a, const blk::BlockRrefResult<uint32_t>& b) {
    return a.rank == b.rank && a.pivots == b.pivots && same_entries(cview(a.R), cview(b.R));
}

} // namespace

uint32_t EngineElimPassOptionsValidation() {
    std::cout << "=============== EngineElimPassOptionsValidation ===============\n";
    Checker g;

    // ---- (A) the defaults ----------------------------------------------------
    const PassOptions env = PassOptions::from_env();
    g.check(!pass_options_overridden(), "(A) no process-wide override is installed on entry");
    g.check(thread_device_override() == ThreadDevice::Inherit, "(A) no thread-local device override on entry");
    g.check(SameOptions(consumer_pass_options(), env), "(A) consumer_pass_options() == from_env(), field by field");
    g.check(SameOptions(elim_pass_options(), env), "(A) elim_pass_options() == from_env(), field by field");
    // (slot-14 WP-2, [C4]/[C7] Sec.2 (b)) The pre-WP-6 formula omits the ninth
    // bit (256, slot-10 WP-8), which `pass_options_fingerprint()` sets iff the
    // EFFECTIVE policy is Gpu specifically (see (B) below): under a `cpu` or
    // `hybrid` environment `env`'s fingerprint still equals the pre-WP-6 one
    // unchanged, but under ENGINE_DEVICE=gpu it is the pre-WP-6 one WITH that
    // bit added. The check compares against the environment's own effective
    // policy rather than assuming cpu, so it passes in every environment.
    const uint64_t old_env_fp = OldFingerprint(env);
    const uint64_t want_env_fp =
        (env.effective_policy() == DevicePolicy::Gpu) ? (old_env_fp | 256u) : old_env_fp;
    g.check(pass_options_fingerprint(elim_pass_options()) == pass_options_fingerprint(env) &&
                pass_options_fingerprint(env) == want_env_fp,
            "(A) the defaults' fingerprint is the pre-WP-6 one (gpu: plus the ninth bit)");

    // ---- (B) the fingerprint -------------------------------------------------
    {
        uint32_t same_mapped = 0, total_mapped = 0;
        bool policy_only_keyed = true;
        bool ninth_bit_ok = true;   // slot-10 WP-8, finding 4: gpu carries bit 256, hybrid does not
        for (uint32_t bits = 0; bits < 128; ++bits) {
            PassOptions o;
            o.fold = bits & 1; o.cse = bits & 2; o.fuse = bits & 4; o.layout = bits & 8;
            o.ksplit = bits & 16; o.batch = bits & 32; o.arena = bits & 64;
            // from_env()'s three mappings (and ScopedThreadDevice's two, the console's three).
            // Since slot-10 WP-8 the `gpu` mapping ALSO carries bit 256 (the
            // ninth bit): its fingerprint is the pre-WP-6 one WITH that bit
            // added, not the pre-WP-6 one unchanged; `cpu` and `hybrid` are
            // untouched by this package.
            for (int m = 0; m < 3; ++m) {
                PassOptions q = o;
                if (m == 0) { q.device_policy = Device::CPU; q.device_model = false; q.policy = DevicePolicy::Cpu; }
                if (m == 1) { q.device_policy = Device::GPU; q.device_model = true; q.policy = DevicePolicy::Gpu; }
                if (m == 2) { q.device_policy = Device::GPU; q.device_model = true; q.policy = DevicePolicy::Hybrid; }
                ++total_mapped;
                const uint64_t want = (m == 1) ? (OldFingerprint(q) | 256u) : OldFingerprint(q);
                if (pass_options_fingerprint(q) == want) ++same_mapped;
            }
            // The Phase 4 legacy pair alone (device_model => Hybrid): keyed as before (no ninth bit).
            PassOptions lp = o;
            lp.device_policy = Device::GPU; lp.device_model = true; lp.policy = DevicePolicy::Cpu;
            ++total_mapped;
            if (pass_options_fingerprint(lp) == OldFingerprint(lp)) ++same_mapped;
            // `policy` alone (finding 3, slot 9): the device bit is set; the rest unchanged.
            // Since slot 10 (finding 4, decision 10 (A)) `gpu` ALSO carries
            // bit 256 and `hybrid` does not, so the two no longer key the
            // same slot (EngineFingerprintDeviceBitValidation tests this at
            // the ProgramCache level).
            for (DevicePolicy pol : {DevicePolicy::Gpu, DevicePolicy::Hybrid}) {
                PassOptions q = o;
                q.policy = pol;
                const uint64_t f = pass_options_fingerprint(q);
                policy_only_keyed = policy_only_keyed && (f & 128u) != 0 && (f & 127u) == (OldFingerprint(q) & 127u) &&
                                    (OldFingerprint(q) & 128u) == 0;
                const bool want256 = (pol == DevicePolicy::Gpu);
                ninth_bit_ok = ninth_bit_ok && (((f & 256u) != 0) == want256);
            }
        }
        std::cout << "  (B) fingerprint = pre-WP-6 fingerprint (gpu: plus bit 256, slot-10 WP-8) on " << same_mapped
                  << " / " << total_mapped << " pair-and-policy options values\n";
        g.check(same_mapped == total_mapped, "(B) every pair-and-policy options value keeps its fingerprint (gpu: plus the ninth bit)");
        g.check(policy_only_keyed, "(B) a `policy`-only gpu / hybrid options value carries the device bit (finding 3)");
        g.check(ninth_bit_ok, "(B) a `policy`-only gpu carries the ninth bit and hybrid does not (finding 4, slot-10 WP-8)");
    }

    // ---- (C) the elimination driver honours ScopedPassOptions ---------------
    const uint32_t p = 1000003;
    auto pool = std::make_shared<BlockPool<uint32_t>>();
    Rng rng(0x5EED6);
    const BlockMat<uint32_t> A = make_input<uint32_t>(rng, pool, p, Shape::ZqLike,
                                                      std::vector<uint32_t>(6, 17), std::vector<uint32_t>(7, 17), 30);
    Executor<uint32_t> ex;
    const ElimRun base = RunRref(ex, A);
    g.check(base.delta.programs > 0, "(C) the reference elimination records per-panel programs");
    g.check(base.delta.batch_off_programs == (env.batch ? 0u : base.delta.programs),
            "(C) with no override the panel programs compile under from_env()'s batch");
    {
        PassOptions o = PassOptions::from_env();
        o.batch = false;
        ScopedPassOptions so(o);
        g.check(!elim_pass_options().batch, "(C) under ScopedPassOptions(batch = false) elim_pass_options().batch == false");
        const ElimRun r = RunRref(ex, A);
        std::cout << "  (C) batch = false override: " << r.delta.programs << " panel programs, "
                  << r.delta.batch_off_programs << " compiled with batch = false (reference run: "
                  << base.delta.batch_off_programs << " of " << base.delta.programs << ")\n";
        g.check(r.delta.programs > 0 && r.delta.batch_off_programs == r.delta.programs,
                "(C) every panel program compiles under the override's batch = false");
        g.check(SameRref(r.res, base.res), "(C) the RREF is entrywise the override-free one");
        {
            // The thread-local device override still replaces only the device fields.
            ScopedThreadDevice sd(ThreadDevice::Cpu);
            const PassOptions e = elim_pass_options();
            g.check(!e.batch && e.effective_policy() == DevicePolicy::Cpu,
                    "(C) ScopedThreadDevice(Cpu) over the override: batch from the override, device from the thread");
        }
    }
    g.check(SameOptions(elim_pass_options(), env), "(C) the override is gone after its scope");

    // ---- (D) a `policy = Gpu`-only override reaches the panel programs --------
    // (slot-14 WP-2, [C4]/[C7] Sec.2 (b)) "the reference run: none" assumes
    // the environment's own default is cpu, so the unoverridden baseline
    // `base` carries zero device_policy_programs. Under ENGINE_DEVICE=gpu or
    // =hybrid that baseline is already non-cpu (from_env() itself selects the
    // device), so the differential this arm tests for does not exist -- skip
    // with a reason rather than assert a false expectation, exactly as the
    // no-device arm below skips.
    if (env.effective_policy() != DevicePolicy::Cpu) {
        std::cout << "  [SKIP] (D) the environment's own device policy is not cpu (ENGINE_DEVICE="
                  << (env.effective_policy() == DevicePolicy::Gpu ? "gpu" : "hybrid")
                  << "): the reference run already carries device_policy_programs, so the "
                     "cpu-baseline differential this arm tests does not exist here\n";
    } else if (DeviceBackend<uint32_t>::usable()) {
        PassOptions o = PassOptions::from_env();
        o.policy = DevicePolicy::Gpu;   // exactly HeckeExtDeterminismValidation's gpu arms
        ScopedPassOptions so(o);
        g.check(elim_pass_options().effective_policy() == DevicePolicy::Gpu,
                "(D) elim_pass_options() selects the gpu policy under the override");
        const ElimRun r = RunRref(ex, A);
        std::cout << "  (D) policy = gpu override: " << r.delta.programs << " panel programs, "
                  << r.delta.gpu_programs << " with GPU nodes (" << r.delta.gpu_nodes << " nodes; reference run: "
                  << base.delta.gpu_programs << ")\n";
        g.check(r.delta.device_policy_programs == r.delta.programs && base.delta.device_policy_programs == 0,
                "(D) every panel program compiles under the gpu policy (the reference run: none)");
        g.check(r.delta.gpu_programs > 0 && r.delta.gpu_nodes > 0, "(D) the panel programs carry GPU-assigned nodes");
        g.check(SameRref(r.res, base.res), "(D) the device RREF is entrywise the CPU one");
    } else {
        std::cout << "  [SKIP] (D) no usable CUDA device\n";
    }

    std::cout << "EngineElimPassOptionsValidation: " << g.checks << " checks, " << g.fails << " failing\n";
    return static_cast<uint32_t>(g.fails);
}
