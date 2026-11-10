// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// Payloads.h -- console phase C1a: the pipeline payloads the handles carry.
//
// One struct per plan Sec. 3.6 row.  Each OWNS its C++ object and holds an
// ObjectPtr to every handle that object BORROWS from, which is how the plan's
// Sec. 4.1 ownership graph is discharged:
//
//   Model ──> LocalSystem (copies K) ──> Cohomology (borrows lssc)
//   Cohomology + Model ──> HeckeOperator (owns cosets + tau; tau borrows K,
//                                         model, cosets; op borrows coh)
//   Cohomology + Model ──> Family        (borrows cohomL, K, model)
//
// Each payload also carries its own `ensure()`, which builds what it needs and
// is IDEMPOTENT.  It is deliberately independent of the console's Session: a
// handle's lazy work must not capture a Settings& that could outlive the
// session, and `ensure()` on a parent is exactly what a child's compute()
// needs.  The console's own state machine (declared -> computed | failed) sits
// ON TOP of it: Object::compute() calls ensure(), so a handle that a child
// forced is still reported `declared` until something reads one of ITS members,
// at which point its compute() is a no-op.
// ===========================================================================
#pragma once
#ifndef CONSOLE_BINDINGS_PAYLOADS_H
#define CONSOLE_BINDINGS_PAYLOADS_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../Core/Object.h"

#include "../../EquivariantCohomologyOnK.h"
#include "../../Hecke/HeckeCosetTables.h"
#include "../../Hecke/HeckeOperatorFamily.h"
#include "../../Hecke/HeckeOperatorOnK.h"
#include "../../Hecke/HeckeTauTable.h"
#include "../../Hecke/LadicCell.h"
#include "../../Hecke/SL2ZTessellationTables.h"
#include "../../LocalSystemSimplicialComplex.h"
#include "../../SimplicialCohomology.h"

namespace console {
namespace bind {

struct ComplexVal {
    SimplicialComplex K;
};

struct ModelVal {
    std::string name;
    SimplicialComplex K;                                        // owned: the tables borrow it
    std::unique_ptr<hecke::SL2ZTessellationTables> model;
};

struct LocalSystemVal {
    uint32_t n_sym = 0;
    uint32_t ell = 0;
    ObjectPtr model;                                            // keep-alive + the source of K
    std::unique_ptr<LocalSystemSimplicialComplex> lssc;
    // (C1b) Non-empty for a LocalSystemPresets system (S^1 / cylinder / Moebius
    // band): those carry NO SL(2,Z) tessellation model, so `model` is null,
    // `lssc` is filled eagerly by the preset verb and `ensure()` is a no-op.
    // `hecke` / `family` refuse such a handle with a named error -- the Hecke
    // action is defined on the 7-vertex model only.
    std::string preset;
    // (C5a, slot-9 WP-12; [E] sec:ring (h)) The coefficient ring the handle was
    // DISPATCHED under: 1 = F_ell (every pre-slot-9 handle), k >= 2 = Z/ell^k.
    // Read once, at the verb, from the MathSettings in force; ensure() installs
    // exactly this ring around the construction, so the object's ring never
    // depends on the handle (or child) that happens to force it.
    uint32_t ring_k = 1;
    // (slot-11 WP-18c; [E] v1.12.0-pre.2 (h)) ell^k >= 2^64 at ell >= 5: the
    // DIGIT route (Hecke/LadicCell); no local system object is built (lssc null).
    bool digit = false;
    // (slot-12 WP-17; [P] v3.9.3 Q9 decided (a)) 2^32 <= ell^k < 2^64 at ell >= 5:
    // the WIDE band -- the native pipeline's uint64_t instantiation, exactly as
    // the campaign job loop (Campaign/CampaignRing.cpp, RunRingPipeline<uint64_t>).
    // lssc stays null; lssc64 is the object.
    bool wide = false;
    std::unique_ptr<LocalSystemSimplicialComplex64> lssc64;
    void ensure();
};

struct CohomologyVal {
    ObjectPtr L;                                                // keep-alive: coh borrows its lssc
    std::unique_ptr<EquivariantCohomologyOnK> coh;
    // (C5a, slot-9 WP-12) The ring (LocalSystemVal::ring_k); equal to L's, by
    // the dispatch check of `cohomology(L)`.
    uint32_t ring_k = 1;
    // (C5a) The two COMPARISON objects of a Z/ell^k handle, built on first use
    // of `.red(q)` / `.bockstein_rank` / `.higher_torsion_count`, each with its
    // own ring installed for its construction only (the RunE8Point precedent,
    // [E] sec:ring (g)): the F_ell object at the same (n, ell), and -- at
    // k != 2 -- the Z/ell^2 object the (mod-ell) Bockstein is computed from.
    std::unique_ptr<LocalSystemSimplicialComplex> lssc_field, lssc_ring2;
    std::unique_ptr<EquivariantCohomologyOnK> coh_field, coh_ring2;
    // (slot-11 WP-18c) a digit handle: coh stays null, the route's record instead.
    bool digit = false;
    std::shared_ptr<const hecke::ladic_cell::CellRecord> digit_rec;
    // (slot-12 WP-17) a wide handle (2^32 <= ell^k < 2^64): coh stays null, the
    // uint64_t objects instead, the comparison objects in the same width.
    bool wide = false;
    std::unique_ptr<EquivariantCohomologyOnK64> coh64;
    std::unique_ptr<LocalSystemSimplicialComplex64> lssc_field64, lssc_ring264;
    std::unique_ptr<EquivariantCohomologyOnK64> coh_field64, coh_ring264;
    void ensure();
    const EquivariantCohomologyOnK& field_comparison();
    const EquivariantCohomologyOnK& ring2_comparison();
    const EquivariantCohomologyOnK64& field_comparison64();
    const EquivariantCohomologyOnK64& ring2_comparison64();
};

struct HeckeVal {
    ObjectPtr H;                                                // the Cohomology handle
    uint64_t p = 0;
    uint32_t r = 1;
    std::unique_ptr<hecke::HeckeCosetTables> cosets;
    std::unique_ptr<hecke::HeckeTauTable> tau;                  // borrows K, model, cosets
    std::unique_ptr<hecke::HeckeOperatorOnK> op;                // borrows coh, cosets, tau
    uint32_t ring_k = 1;                                        // (C5a) the Cohomology's ring
    bool digit = false;                                         // (slot-11 WP-18c) the digit route
    std::shared_ptr<const hecke::ladic_cell::CellRecord> digit_rec;
    // (slot-12 WP-17) the wide band: the uint64_t tables, tau and operator.
    bool wide = false;
    std::unique_ptr<hecke::HeckeCosetTables64> cosets64;
    std::unique_ptr<hecke::HeckeTauTable64> tau64;              // borrows K, model, cosets64
    std::unique_ptr<hecke::HeckeOperatorOnK64> op64;            // borrows coh64, cosets64, tau64
    void ensure();
};

struct FamilyVal {
    ObjectPtr H;                                                // the Cohomology handle
    std::unique_ptr<hecke::HeckeOperatorFamily> fam;            // borrows coh, K, model
    void ensure();
};

// The Model handle a Cohomology handle was ultimately built from (through its
// LocalSystem), i.e. the owner of the K and the tessellation tables every Hecke
// object borrows.  Never null for a handle the bindings created.
const ModelVal& ModelOf(const CohomologyVal& coh);

// (C5a, slot-9 WP-12; [E] sec:ring (h)) The ring exponent of a Cohomology /
// HeckeOperator handle (1 = F_ell).  A binding that is NOT on (h)'s honouring
// list and takes such a handle calls RequireFieldHandle at its dispatch, so a
// Z/ell^k handle is refused with hecke::NotImplementedInThisBuild (exit 5,
// naming `who` and the ring) instead of failing later on a field accessor.
uint32_t HandleRingK(const ObjectPtr& handle);
void RequireFieldHandle(const ObjectPtr& handle, const std::string& who);

} // namespace bind
} // namespace console

#endif // CONSOLE_BINDINGS_PAYLOADS_H
