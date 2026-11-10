// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef HECKE_EXT_LVALUE_BIRCH_ARMS_H
#define HECKE_EXT_LVALUE_BIRCH_ARMS_H

// (hecke-extensions slot 19, WP-LIN) The (BS-*) arms of HeckeLValueValidation: the linear instance
// BirchModularSymbolRoute ([E] rmk:lvshapes (B)) gated by R19-8 Sec. 6's G1-G5 and the isign facts at (258, 37);
// predictions docs/implementation/slot19/wplin/predictions.md (committed first).  Read-only; writes nothing.

#include <functional>
#include <string>

namespace hecke_ext {
namespace lvalue {

using BirchCheckFn = std::function<void(bool, const std::string&)>;
// heavy: the HECKE_LVALUE_HEAVY=1 additions ((258, 37) over Z/37^2 with the iota reading; G3 at the torsion cell (72, 37)).
void RunBirchArms(const BirchCheckFn& check, bool heavy);

} // namespace lvalue
} // namespace hecke_ext

#endif // HECKE_EXT_LVALUE_BIRCH_ARMS_H
