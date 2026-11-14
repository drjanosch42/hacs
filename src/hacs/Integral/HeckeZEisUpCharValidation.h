// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef INTEGRAL_HECKEZ_EISUPCHAR_VALIDATION_H
#define INTEGRAL_HECKEZ_EISUPCHAR_VALIDATION_H

#include <cstdint>

// slot-27 EISUPb: HZ-EISUP (b), the Eisenstein U_p systems with characters psi != 1 (HeckeZEisUpChar.h; HZEISUPb_design.md v0.4 Sec. 7).
// The four in-scope record cells (9,0), (18,0), (25,0), (49,0); arms (a) the certificate C-B0 .. C-B8 with pinned digests, (b) the second
// route (the closed form's components; sum deg = e_k = c(N) - [k = 2]), (c) v_ell(disc f_theta) = the record's v_disc (design Sec. 3),
// (d) every g_{U_p,j} divides HZ-UTAG's chi_Q(U_p) over Z, (e) the record path (rows, C-B7, no blocked(U_p) left, final, pinned, scan
// clean, the Eisenstein--Eisenstein meetings), (f) bit identity (threads 1 / 4 / 16, reuse vs. recompute, ISA, two embeddings at q*; the
// direction unit check at o_N = 10), (f') the set S and the q* chooser (unit checks, the cap refusal), (g) tampers, (h) G10.contract per
// binary and the HZ hash invariance, (i) amendment A5 (schema tampers; edition 3's 226 records re-scanned), (j) the (a)-regression, (k)
// the seven refused(non_reduced) records of edition 3 unchanged.  Gate line `HECKEZ_EISUPB ... verdict=OK`.
uint32_t HeckeZEisUpCharValidation();

#endif
