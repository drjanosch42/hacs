// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Slot-21 registration (closed slot): the former per-package fragments Registry/Slot21/<pkg>.cpp folded verbatim
// (S24 CL-10), each package function kept; RegisterSlot21() calls them in the old RegisterAllValidations() order.

// ===== Slot 21 WPLIC ===========================================
// Slot-21 WPLIC registration fragment (hecke-extensions slot 21). Owned by
// package WPLIC ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot21/WPLIC.cmake.
//
// LicenceGuardValidation (fast, read-only): runs docs/licence/tools/check_licences.sh
// on the source tree (located through the path baked in by CMake, then by
// walking up from the executable / cwd, like ConsumerMigrationValidation): every
// header-able first-party file carries an SPDX header or is on the pending list,
// every identifier is on the allow-list, the vendored paths are byte-identical.
// Reads tracked files only; it never writes and consults no environment variable.
#include "../ValidationRegistry.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

bool FileExists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool LooksLikeRoot(const std::string& d) {
    return FileExists(d + "/docs/licence/tools/check_licences.sh") && FileExists(d + "/REUSE.toml");
}

std::string FindRoot() {
#ifdef OC_SOURCE_DIR
    {
        const std::string r(OC_SOURCE_DIR);
        if (!r.empty() && LooksLikeRoot(r)) return r;
    }
#endif
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            const size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
            if (LooksLikeRoot(dir)) return dir;
        }
    }
    if (::getcwd(buf, sizeof(buf)) != nullptr) {
        std::string dir(buf);
        for (int up = 0; up < 8; ++up) {
            if (LooksLikeRoot(dir)) return dir;
            const size_t slash = dir.find_last_of('/');
            if (slash == std::string::npos || slash == 0) break;
            dir = dir.substr(0, slash);
        }
    }
    return std::string();
}

uint32_t LicenceGuardValidation() {
    std::cout << "LicenceGuardValidation: SPDX headers, allow-listed identifiers, vendored paths untouched" << std::endl;
    const std::string root = FindRoot();
    if (root.empty()) {
        std::cout << "[SKIP] LicenceGuardValidation: source tree (docs/licence/tools/check_licences.sh) not found" << std::endl;
        return 0;
    }
    const std::string cmd = "cd '" + root + "' && bash ./docs/licence/tools/check_licences.sh 2>&1";
    std::cout << "  $ " << cmd << std::endl;
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        std::cout << "  [FAIL] cannot start check_licences.sh" << std::endl;
        return 1;
    }
    std::string last;
    bool saw_summary = false;
    char line[8192];
    while (std::fgets(line, sizeof(line), pipe) != nullptr) {
        std::cout << "  " << line;
        if (std::string(line).find("LICENCE_GUARD ") == 0) {
            saw_summary = true;
            last = line;
        }
    }
    const int status = ::pclose(pipe);
    const bool ran = status != -1 && WIFEXITED(status);
    const int rc = ran ? WEXITSTATUS(status) : -1;
    if (ran && rc == 2) {
        std::cout << "[SKIP] LicenceGuardValidation: environment (no git work tree or no python3)" << std::endl;
        return 0;
    }
    uint32_t failing = 0;
    if (!ran || rc != 0) {
        std::cout << "  [FAIL] check_licences.sh exit status " << rc << std::endl;
        ++failing;
    }
    if (!saw_summary || last.find("verdict=OK") == std::string::npos || last.find("failures=0") == std::string::npos) {
        std::cout << "  [FAIL] no LICENCE_GUARD ... failures=0 verdict=OK line" << std::endl;
        ++failing;
    }
    if (failing == 0) std::cout << "  [PASS] licence guard clean" << std::endl;
    return failing;
}

}  // namespace

void RegisterSlot21WPLIC() {
    // WP-LIC: the licence guard (fast, read-only; ~ 1 s).
    ValidationRegistry::register_validation("LicenceGuardValidation", &LicenceGuardValidation, ValidationTier::Fast);
}

// ===== Slot 21 WPNB ===========================================
// Slot-21 WPNB registration fragment (hecke-extensions slot 21). Owned by
// package WPNB ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot21/WPNB.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot21WPNB() {
    // Empty at WP-1 (day zero): no name registered, roster stays 182.
}

// ===== Slot 21 WPENG ===========================================
// Slot-21 WPENG registration fragment (hecke-extensions slot 21). Owned by
// package WPENG ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from
// RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot21/WPENG.cmake.
#include "../ValidationRegistry.h"

void RegisterSlot21WPENG() {
    // Empty at WP-1 (day zero): no name registered, roster stays 182.
}

// ===== Slot 21 T1_lll ===========================================
// Slot-21 T1_lll registration fragment (hecke-extensions slot 21). Owned by
// package T1 (the talk's in-house LLL) ONLY: register this package's
// validations here and nowhere else in the shared structure. Called explicitly
// from RegisterAllValidations() (never a static initialiser); its sources are
// listed in cmake/slot21/T1_lll.cmake. Created mid-slot with the two one-line
// shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../NumberField/LllValidation.h"

void RegisterSlot21T1LLL() {
    // (T1) the in-house exact LLL and POLRED (NumberField/Lll.h, Polred.h; Cohen
    // GTM 138 sec. 2.6, Alg. 4.4.11 / 4.4.12): scrambled Z^n / E_8 / A_2, integer
    // relations, random lattices against an own Gram--Schmidt, determinant and
    // brute-force lambda_1, Alg. 2.6.3 == Alg. 2.6.7, POLRED on Q(zeta_l)^+ and
    // Q(zeta_l) (l = 7, 11, 37) to known polynomials, negative controls.
    // Read-only, no env, ~ 2-4 s.
    ValidationRegistry::register_validation("NumberFieldLllValidation", &NumberFieldLllValidation, ValidationTier::Fast);
}

// ===== Slot 21 P7_qexp ===========================================
// Slot-21 P7_qexp registration fragment (hecke-extensions slot 21). Owned by
// package P7 (Lambda-adic q-expansions, the l-adic track) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot21/P7_qexp.cmake. Created mid-slot with the
// two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeLambdaQExpValidation.h"

void RegisterSlot21P7QExp() {
    // (P7) truncated Lambda-adic q-expansions sum_{n <= 30} a_n(X) q^n mod
    // (ell^3, (X - x_c)^m), m <= 3, per component of the cuspidal ordinary Hida
    // algebra and the Lambda-adic Eisenstein series ([U] sec:u-lambda), at
    // (11, 2), (13, 0), (37, 32), (103, 24): Hida's control at three further
    // weights against an independent q-expansion route, the Delta family at
    // weight 2 against 11a's eta product, eps_p against its closed form, the
    // Hecke relations, the Lambda-adic Eisenstein congruence mod (G~),
    // precision refusals and perturbed inputs caught. Read-only, no env.
    ValidationRegistry::register_validation("HeckeLambdaQExpValidation", &HeckeLambdaQExpValidation, ValidationTier::Standard);
}

// ===== Slot 21 T2_ribet ===========================================
// Slot-21 T2_ribet registration fragment (hecke-extensions slot 21). Owned by
// package T2 (the talk's degree-ell Ribet field polynomial) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot21/T2_ribet.cmake. Created mid-slot with the two
// one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../NumberField/RibetFieldValidation.h"

void RegisterSlot21T2Ribet() {
    // (T2) the degree-ell subfield of the Ribet field H = Q(zeta_ell, eta_i^{1/ell})
    // as a polynomial over Z (NumberField/RibetField.h, MpReal.h): at (37, 32) the
    // polynomial numerically at two precisions and exactly by CRT, the Frobenius
    // counts, disc, the maximal order (disc 37^35) and T1's POLRED (pinned). Claims
    // for the referee: docs/implementation/slot21/T2_claim_for_F4.md. Read-only; env arm
    // NUMBERFIELD_RIBET_CELLS="691:12,691:200" (EFFECTIVE line). ~1 min.
    ValidationRegistry::register_validation("NumberFieldRibetFieldValidation", &NumberFieldRibetFieldValidation,
                                            ValidationTier::Slow);  // slow since S24 VAL-IMPL (VALSTREAM_audit_1030.md row 20)
}

// ===== Slot 21 P8_msym ===========================================
// Slot-21 P8_msym registration fragment (hecke-extensions slot 21). Owned by
// package P8 (Lambda-adic modular symbols, the l-adic track) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot21/P8_msym.cmake. Created mid-slot with the
// two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeLambdaModSymValidation.h"

void RegisterSlot21P8MSym() {
    // (P8) ordinary cohomology along the tower as a Lambda-module, truncated mod
    // (ell^3, (X - x_c)^m), m <= 3 ([U] v0.5.0-pre.2 sec:u-msym, P8_spec v0.2):
    // per etale component of P7 and per [C]-sign, the ordinary modular symbols at
    // N = 5 consecutive weights + further weights: rank / cyclicity / Hecke
    // polynomials against P7's forms side (U15), (BF) over every b mod ell - 1,
    // the operator-valued masses against WP-MEAS, the normaliser of record and
    // the interpolated normalised masses (U16, findings), the gluing depth j_k
    // at m_Eis (U17), the level-ell arm, the lattice arm, negative controls.
    // Read-only; env arms HECKE_LAMBDA_MSYM_ALL / _CELLS with an EFFECTIVE line.
    ValidationRegistry::register_validation("HeckeLambdaModSymValidation", &HeckeLambdaModSymValidation, ValidationTier::Standard);
}

// ===== Slot 21 P9_twovar ===========================================
// Slot-21 P9_twovar registration fragment (hecke-extensions slot 21). Owned by
// package P9 (the two-variable measure, the l-adic track) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot21/P9_twovar.cmake. Created mid-slot with the
// two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/HeckeLambdaTwoVarValidation.h"

void RegisterSlot21P9TwoVar() {
    // (P9) the two-variable measure mu(X; b + ell^M) and L_a(X, T) mod (ell^K, (X - x_c)^m,
    // omega_{M-1}(T)) by interpolation of P8's normalised masses (P9_spec v0.2): assembly,
    // numerical control at further weights, P1's lfunction/v1 as the oracle, the two-variable
    // Eisenstein congruence at alpha_a (37, 32), and the weight-2 point (11a: control at k = 2,
    // the trivial zero, Greenberg-Stevens' derivative, the L-invariant against Tate's period).
    // Read-only; env arms HECKE_LAMBDA_TWOVAR_ALL / _CELLS with an EFFECTIVE line.
    ValidationRegistry::register_validation("HeckeLambdaTwoVarValidation", &HeckeLambdaTwoVarValidation, ValidationTier::Standard);
}

// ===== Slot 21 P3_coeffring ===========================================
// Slot-21 P3_coeffring registration fragment (hecke-extensions slot 21). Owned by
// package P3 (coefficient rings for the measure, the l-adic track) ONLY: register this
// package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot21/P3_coeffring.cmake. Created mid-slot with the
// two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/LValue/HeckeLValueCoefficientRingValidation.h"

void RegisterSlot21P3CoeffRing() {
    // (P3) the Mazur-Tate-Teitelbaum measure of an ordinary level-1 eigen-system with coefficients in
    // O = (Z/ell^R)[x]/(F): W(F_q) (residue degree 2, 3) and the totally ramified Z_ell[x]/(F) (S_28 at
    // 131, 139; route B through WP-LV3's m-block reader); Eichler-Shimura against q-expansions, prop:lvmtt
    // over O, Galois equivariance, Hida control mod m, the Python oracle's frozen table (P3_spec v0.1).
    // Read-only; env arm HECKE_LVALUE_COEFFRING_HEAVY with an EFFECTIVE line.
    ValidationRegistry::register_validation("HeckeLValueCoefficientRingValidation", &HeckeLValueCoefficientRingValidation, ValidationTier::Standard);
}

// ===== Slot 21 P2_eismeas ===========================================
// Slot-21 P2_eismeas registration fragment (hecke-extensions slot 21). Owned by
// package P2 (the Eisenstein measure exactly in Lambda, the l-adic track) ONLY: register
// this package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot21/P2_eismeas.cmake. Created mid-slot with the
// two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/LValue/HeckeLValueEisensteinMeasureValidation.h"

void RegisterSlot21P2EisMeas() {
    // (P2) the Mazur-Tate-Teitelbaum measure of Zagier's Eisenstein period polynomial n! p_k^- exactly: the Manin-path symbol is a
    // Dedekind-Rademacher sum, mu^Eis = -E_1 * iota E_{k-1} on units, branch series -G_{a+1} G_{k-1-a} in Lambda (x) Q, the pole
    // branches with the explicit polar term Res/(u(1+T)-1); WP-MEAS's polynomial symbol and the cusp forms at (30,37), (282,37),
    // (226,103) on every branch, pole branches included (P2_spec v0.1). Read-only; env arm HECKE_LVALUE_EISMEAS_HEAVY with an EFFECTIVE line.
    ValidationRegistry::register_validation("HeckeLValueEisensteinMeasureValidation", &HeckeLValueEisensteinMeasureValidation, ValidationTier::Standard);
}

// ===== Slot 21 P4_wild ===========================================
// Slot-21 P4_wild registration fragment (hecke-extensions slot 21). Owned by
// package P4 (wild characters on the L-side, the l-adic track) ONLY: register
// this package's validations here and nowhere else in the shared structure. Called
// explicitly from RegisterAllValidations() (never a static initialiser); its
// sources are listed in cmake/slot21/P4_wild.cmake. Created mid-slot with the
// two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/LValue/HeckeLValueWildCharacterValidation.h"

void RegisterSlot21P4Wild() {
    // (P4) characters chi = omega^i psi_t of conductor ell^m (m = 2, 3) with values in Z_ell[zeta_{ell^{m-1}}]: wild twisted Bernoulli numbers
    // against Kubota-Leopoldt at a wild point, [E] prop:lvmtt(4) at wild chi, branch elements at wild points and the lambda reading, P2's
    // Eisenstein measure and the pole branches at wild points, and the tau(chi) constant against complex L(f, chi, j+1) (P4_spec v0.1).
    // Read-only; env arm HECKE_LVALUE_WILD_HEAVY with an EFFECTIVE line.
    ValidationRegistry::register_validation("HeckeLValueWildCharacterValidation", &HeckeLValueWildCharacterValidation, ValidationTier::Standard);
}

// ===== Slot 21 P5_iwasawa ===========================================
// Slot-21 P5_iwasawa registration fragment (hecke-extensions slot 21). Owned by
// package P5 (b), (c) (Iwasawa invariants of the measure branches, the tame functional
// equation; the l-adic track) ONLY: register this package's validations here and nowhere
// else in the shared structure. Called explicitly from RegisterAllValidations() (never a
// static initialiser); its sources are listed in cmake/slot21/P5_iwasawa.cmake. Created
// mid-slot with the two one-line shared edits (slot-12 WP-14 precedent).
#include "../ValidationRegistry.h"
#include "../HeckeExt/LValue/HeckeLValueIwasawaInvariantsValidation.h"

void RegisterSlot21P5Iwasawa() {
    // (P5) (mu, lambda) of the MTT measure branches at the standard cells: invariants() against a ball-free values oracle (divided
    // differences of the critical values), Kubota-Leopoldt by the group ring and by exact Bernoulli numbers, level-1 Greenberg-Vatsal at
    // the Eisenstein cells, the tame functional equation and its branch / self-dual-parity consequences (P5_spec v0.1). Read-only; env arm
    // HECKE_LVALUE_IWASAWA_HEAVY with an EFFECTIVE line.
    ValidationRegistry::register_validation("HeckeLValueIwasawaInvariantsValidation", &HeckeLValueIwasawaInvariantsValidation, ValidationTier::Standard);
}

void RegisterSlot21() {
    RegisterSlot21WPLIC();
    RegisterSlot21WPNB();
    RegisterSlot21WPENG();
    RegisterSlot21T1LLL();
    RegisterSlot21P7QExp();
    RegisterSlot21T2Ribet();
    RegisterSlot21P8MSym();
    RegisterSlot21P9TwoVar();
    RegisterSlot21P3CoeffRing();
    RegisterSlot21P2EisMeas();
    RegisterSlot21P4Wild();
    RegisterSlot21P5Iwasawa();
}
