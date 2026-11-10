// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_CONE_CROSS_ROUTE_RING_VALIDATION_H
#define HECKE_CONE_CROSS_ROUTE_RING_VALIDATION_H

#include <cstdint>

// hecke-extensions slot 10, WP-3 (slot-10 decision 3 (A)): [C, obl:coneexact]
// (vii) over Z/ell^k.  At the seven E8 points ((10, 691) Z/ell^2; (6, 5),
// (16, 5) Z/ell^2; (30, 5), (60, 5) Z/ell^2 and Z/ell^3 -- every one with
// ell^k < 2^32), for T_p, p in {2, 3} (and p = 5 at (10, 691), as E8 item 4b
// computes it), A_c^{(q)} from route 1 (ModuleConeActionT), the strict-pair
// route (HeckeConeStrictPairRoute) and route 3 (HeckeConeRelativeRoute on the
// relative sub-complex constructor over the ring) are entrywise equal in
// every degree.  Arms:
//   (R)  the seven points, three routes, every degree, every prime; route 3's
//        non-vacuity (E_nonzero, theta_prime_differs, !r_commutes_normative)
//        is ASSERTED at every cell, not just printed (slot 10, WP-8, m-12: a
//        route-3 build taking the trivial E == 0 / Theta' == Theta shortcut
//        would otherwise pass silently) -- a genuine negative seam on route 3
//        itself (e.g. E forced to 0) is left to slot 11, per the review;
//   (N)  negative control: a strict table perturbed after construction (one
//        tau^s coefficient + 1) is refused by the route (its re-run
//        certificate), and the unperturbed seam table agrees;
//   (W)  the regime above 2^32: at (10, 134217757) over Z/ell^2 route 1 runs
//        (the templated uint64_t ModuleConeActionT) and both cross-routes
//        refuse by the uint32_t regime's named refusal;
//   (X)  the three flipped members both ways: honour at ell >= 5 with the
//        ring in force; refuse on an F_ell normative action under the ring,
//        on a composite one under F_ell or under another k, on null, and the
//        relative sub-complex constructor at ell = 3.
// Registered `standard` in Registry/Slot10/WP3.cpp.
uint32_t HeckeConeCrossRouteRingValidation();

#endif // HECKE_CONE_CROSS_ROUTE_RING_VALIDATION_H
