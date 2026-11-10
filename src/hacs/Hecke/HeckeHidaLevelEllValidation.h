// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 13, WP-6) HeckeHidaLevelEllValidation -- Gamma_0(ell)
// on the fixed model by Shapiro (Hecke/HeckeLevelEll): [C] v1.13.0-pre.2
// prop:levelcorrespondence, prop:levelellU (eq:Ulevell, eq:Ulevellmodel -- their
// first numerical test), prop:oldspace, prop:levelorbit, obl:shapiro,
// lem:levelelltorsion, prop:stabilisation (the pair (e_ell o dg_ell,
// c_st^{-1} omega), never a trace inverse), and (Hi10).  An independent route:
// group 1-cocycles of Gamma_0(ell) through Shapiro on <S, T>, with the double
// coset [Gamma_0(ell) g Gamma_0(ell)] as cor o conj_g o res (no tau, no model).
// Standard tier; writes nothing; HIDA_LEVEL_ELL lines on stdout.
#pragma once
#ifndef HECKE_HIDA_LEVEL_ELL_VALIDATION_H
#define HECKE_HIDA_LEVEL_ELL_VALIDATION_H

#include <cstdint>

uint32_t HeckeHidaLevelEllValidation();

#endif
