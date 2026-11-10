// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-LV) The L-value interface -- see LValueRoute.h.
#include "LValueRoute.h"

#include "../../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild

#include <sstream>
#include <stdexcept>

namespace hecke_ext {
namespace lvalue {

const char* ToString(RouteKind k) {
    switch (k) {
    case RouteKind::Linear: return "linear";
    case RouteKind::Bilinear: return "bilinear";
    case RouteKind::Congruence: return "congruence";
    }
    return "?";
}

const char* ToString(ModelId m) {
    switch (m) {
    case ModelId::SL2ZSevenVertex: return "SL2Z-7vertex";
    case ModelId::Gamma0NShapiro: return "Gamma0N-Shapiro";
    case ModelId::CharacterModel: return "character-model";
    case ModelId::MoerdijkPronk: return "Moerdijk-Pronk";
    }
    return "?";
}

std::string Domain::to_string() const {
    std::ostringstream os;
    os << "model=" << ToString(model) << " coh=" << cohomology << " n=" << n_sym << " N=" << level << " ell=" << ell
       << " R=" << (m == 0 ? std::string("Z_ell") : "Z/ell^" + std::to_string(m)) << " H'=" << subquotient;
    return os.str();
}

namespace {
std::string ConstText(const std::vector<NormalisingConstant>& cs) {
    if (cs.empty()) return "none";
    std::ostringstream os;
    for (size_t i = 0; i < cs.size(); ++i) {
        os << (i ? ";" : "") << cs[i].name << ":" << cs[i].value << "(v=";
        if (cs[i].valuation_known) os << cs[i].ell_valuation; else os << "?";
        os << ")";
    }
    return os.str();
}
} // namespace

std::string NormalisationRecord::to_string() const {
    std::ostringstream os;
    os << "L=" << l_function << " s0=" << s0 << " twist={" << twist << "} sign=" << sign_at_infinity
       << " euler_removed={" << ConstText(euler_factors_removed) << "} period=" << period
       << " constants={" << ConstText(constants) << "} ambiguity=" << ambiguity;
    return os.str();
}

std::string IndexStatement::to_string() const {
    std::ostringstream os;
    os << "label=" << label << " status=" << status << " Q=" << module_name << " L=" << l_value
       << " v(L)=" << l_valuation << (l_valuation_exact ? "" : "+") << " expected=[" << expected_lo << "," << expected_hi
       << "] found=" << found << " holds=" << (holds ? "yes" : "NO");
    return os.str();
}

LValueRoute::~LValueRoute() = default;

void LValueRoute::refuse_kind(const char* entry) const {
    throw std::logic_error(std::string("LValueRoute ") + name() + ": " + entry + " does not match kind=" + ToString(kind()) +
                           " ([E] def:lvroute(4): one evaluation per kind)");
}

std::vector<uint64_t> LValueRoute::evaluate_functional(const std::vector<CohomologyClass>&) const {
    refuse_kind("evaluate_functional (kind linear)");
}
uint64_t LValueRoute::evaluate_pairing(const CohomologyClass&, const CohomologyClass&) const {
    refuse_kind("evaluate_pairing (kind bilinear)");
}
std::string LValueRoute::adjunction() const { refuse_kind("adjunction (kind bilinear)"); }
CongruenceModule LValueRoute::congruence_module() const { refuse_kind("congruence_module (kind congruence)"); }

// ---- (B): BirchModularSymbol.cpp (slot-19 WP-LIN) -----------------------------------

// ---- (H): HidaPairingRoute.cpp (slot-19 WP-BIL) -----------------------------------------------------------

} // namespace lvalue
} // namespace hecke_ext
