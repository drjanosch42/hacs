// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// (hecke-extensions slot 19, WP-BIL) HidaPairingRoute: the `bilinear` kind of [E] def:lvroute (rmk:lvshapes row (H)) on the
// core HidaPairing.{h,cpp}.  See LValueRoute.h for the lattices named by Domain::cohomology and
// docs/implementation/slot19/R19-9_report.md Sec. 6 for the SPEC (NormalisationRecord, IndexStatement H1-H3).
#include "HidaPairing.h"
#include "LValueRoute.h"
#include "MBlockReader.h"

#include "../../Hecke/HeckeSettings.h"   // hecke::NotImplementedInThisBuild
#include "../../../ModArithmetic/EngineCheck.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace hecke_ext {
namespace lvalue {

namespace {
enum class Lattice { Mixed, Hida, MixedShriek };

Lattice ParseLattice(const std::string& c) {
    if (c == "H^1_c(M_n) x H^1(D_n)" || c == "H^1" || c == "H^1_c") return Lattice::Mixed;
    if (c == "H^1_! x H^1_!" || c == "H^1_!") return Lattice::Hida;
    if (c == "H^1_c(M_n) x H^1_!(D_n)") return Lattice::MixedShriek;
    throw hecke::NotImplementedInThisBuild("HidaPairingRoute: the lattice '" + c +
                                           "' is not one of H^1_c(M_n) x H^1(D_n), H^1_! x H^1_!, H^1_c(M_n) x H^1_!(D_n)");
}

const char* LatticeName(Lattice L) {
    switch (L) {
    case Lattice::Mixed: return "H^1_c(M_n) x H^1(D_n)";
    case Lattice::Hida: return "H^1_! x H^1_!";
    case Lattice::MixedShriek: return "H^1_c(M_n) x H^1_!(D_n)";
    }
    return "?";
}

void RequireSupported(const Domain& d) {
    if (d.model != ModelId::SL2ZSevenVertex)
        throw hecke::NotImplementedInThisBuild(std::string("HidaPairingRoute: model ") + ToString(d.model) +
                                               " is not supported (the 7-vertex SL_2(Z) model only)");
    if (d.level != 1) throw hecke::NotImplementedInThisBuild("HidaPairingRoute: level N != 1 is not supported (level 1, no W_N)");
    if (d.ell < 5) throw hecke::NotImplementedInThisBuild("HidaPairingRoute: ell >= 5 (H^1(Gamma, M) = M/(M^S + M^U))");
    if (d.n_sym < 2 || d.n_sym % 2 != 0) throw hecke::NotImplementedInThisBuild("HidaPairingRoute: n even and >= 2");
    if (d.m == 0) throw hecke::NotImplementedInThisBuild("HidaPairingRoute: R = Z_ell (m = 0): give a precision m >= 1");
    if (hidapair::Zq::MaxPrecision(d.ell, d.m) < d.m)
        throw hecke::NotImplementedInThisBuild("HidaPairingRoute: ell^m >= 2^62 (uint64_t residues)");
    const Lattice L = ParseLattice(d.cohomology);
    if (L == Lattice::Hida && !(d.n_sym + 1 < d.ell))
        throw hecke::NotImplementedInThisBuild("HidaPairingRoute: Hida's H^1_! self-pairing needs n + 1 < ell (theta)");
}

// (slot-20 WP-LV3) the domain names an m-block when its subquotient says "m=" / "m_Eis" (or at (258, 37), WP-BIL's cell):
// then the generic reader (MBlockReader.h) reads it; otherwise WP-BIL's small-cell reading (all eigen-systems, dim S_k <= 2).
bool NamesBlock(const Domain& d) {
    return (d.n_sym == 258 && d.ell == 37) || d.subquotient.find("m=") != std::string::npos ||
           d.subquotient.find("m_Eis") != std::string::npos;
}
uint32_t BlockPrecision(const Domain& d) { return std::max<uint32_t>(d.m, (d.n_sym == 258 && d.ell == 37) ? 10u : 6u); }

int32_t VFactorial(uint64_t km1, uint64_t ell) {
    int32_t v = 0;
    for (uint64_t p = ell; p <= km1; p *= ell) v += static_cast<int32_t>(km1 / p);
    return v;
}
} // namespace

HidaPairingRoute::HidaPairingRoute(Domain d) : d_(std::move(d)) {}

const hidapair::HidaPairingCell& HidaPairingRoute::cell() const {
    RequireSupported(d_);
    if (!cell_) cell_ = std::make_shared<const hidapair::HidaPairingCell>(d_.n_sym, d_.ell, d_.m);
    return *cell_;
}

NormalisationRecord HidaPairingRoute::normalisation() const {
    NormalisationRecord N;
    const uint32_t k = d_.n_sym + 2;
    N.l_function = "L(s,Sym^2 f) (Hida (5.4), omega = 1; = L(1, Ad f) unitary)";
    N.s0 = "k=" + std::to_string(k);
    N.twist = "chi = 1; signs + x - (geometric; iota anti-isometric, [C] iota^* = -iota_geo pairs the same)";
    N.sign_at_infinity = '*';   // both: the pairing matches geo-even with geo-odd
    // no Euler factor removed: Hida Invent. 63 Thm 5.1 at N = 1 is the complete product (LIT p. 247)
    N.period = std::string("Omega_f^+ Omega_f^- with r_f^+- = Omega^+- delta^+-, delta^+- primitive in ") +
               ([&] {
                   try {
                       return std::string(LatticeName(ParseLattice(d_.cohomology)));
                   } catch (...) {
                       return std::string("(unsupported lattice)");
                   }
               })();
    N.constants.push_back({"Hida Thm 5.1 (N = 1)", "-(-1)^{k/2} i (2 pi)^{k+1} / (k-1)!", 0, false});
    N.constants.push_back({"(k-1)!", "(k-1)! = " + std::to_string(k - 1) + "!", VFactorial(k - 1, d_.ell), d_.ell >= 2});
    if (!(d_.n_sym + 1 < d_.ell)) {
        // (slot-20 WP-LV3) [E] def:lvlattice (1.20.0-pre.2), computed: c_f^eps = length_O(H^1(D)/tors[f] / theta_*(H^1(V)/tors[f])),
        // theta = Phi_n of [C] lem:symdual (HidaPairing::theta is Phi_n / n!); g^V_f primitive in H^1(V)/tors (read on the dual basis of
        // H^1_c(D)), h_f primitive in H^1(D)/tors, theta_*(g^V_f) = pi^{c_f} h_f up to a unit; c_f >= 0, = 0 for n < ell.
        // On a domain naming no m-block the constant is per eigen-system of the whole space: not computed (recorded so).
        if (NamesBlock(d_)) {
            mblock::MBlockOptions opt;
            opt.lattice_change = true;
            opt.hd_torsion = false;
            if (d_.subquotient.find("T_2") == std::string::npos) opt.joint = mblock::EisensteinJoint(hidapair::Zq(d_.ell, BlockPrecision(d_)), d_.n_sym);
            const uint32_t M = BlockPrecision(d_);
            const hidapair::Zq R(d_.ell, M);
            const mblock::MBlockReading B =
                mblock::ReadMBlockCell(d_.n_sym, d_.ell, M, mblock::Lambda0FromSubquotient(d_.subquotient, R, d_.n_sym), opt);
            const bool odd_sign = d_.subquotient.find("geo-odd") != std::string::npos;
            const mblock::MBlockSign& S = B.sign[odd_sign ? 1 : 0];
            std::ostringstream os;
            int32_t vmax = 0;
            uint32_t e = 1;
            bool known = S.complete && !S.forms.empty();
            os << "c_f = e v_ell(n!) - d_f (R20-3) on " << S.name << " =";
            for (const mblock::MBlockForm& f : S.forms) {
                os << " " << (f.have_lattice ? std::to_string(f.lattice_change) + "(=" + std::to_string(f.e * B.v_nfact) + "-" + std::to_string(f.d_lattice) + ")"
                                             : std::string("?"));
                known = known && f.have_lattice;
                vmax = std::max(vmax, f.lattice_change);
                e = f.e;
            }
            os << " (e = " << e << ", v_pi units; [E] def:lvlattice, theta = Phi_n; v_ell(n!) = " << B.v_nfact << ")";
            if (!S.complete) os << " [" << S.refusal << "]";
            N.constants.push_back({"change of lattice c_f (def:lvlattice; theta = Phi_n, n >= ell)", os.str(),
                                   e == 1 ? vmax : 0, known && e == 1});
        } else {
            N.constants.push_back({"change of lattice c_f (def:lvlattice; theta = Phi_n, n >= ell)",
                                   "not computed: the domain names no m-block (subquotient \"m=(ell,T_2-a)\" or \"m_Eis\")", 0, false});
        }
    }
    N.ambiguity = "O^x";
    return N;
}

uint64_t HidaPairingRoute::evaluate_pairing(const CohomologyClass& x, const CohomologyClass& y) const {
    const hidapair::HidaPairingCell& C = cell();
    const Lattice L = ParseLattice(d_.cohomology);
    const uint32_t r = C.ne() + C.no();
    const auto check_free = [&](const CohomologyClass& z, size_t rank, const char* which) {
        if (z.coordinates.size() != rank)
            throw std::logic_error(std::string("HidaPairingRoute::evaluate_pairing: ") + which + " has " +
                                   std::to_string(z.coordinates.size()) + " coordinates, the lattice has rank " + std::to_string(rank));
        for (uint32_t e : z.exponents)
            if (e != d_.m)
                throw std::logic_error(std::string("HidaPairingRoute::evaluate_pairing: ") + which +
                                       " is not on the free basis (Z/ell^m)^r (an exponent != m)");
    };
    const hidapair::FibreOps& F = C.F();
    check_free(x, r, "x");
    hidapair::Vec P(d_.n_sym + 1, 0);
    for (uint32_t i = 0; i < r; ++i) P = F.add(P, F.scale(x.coordinates[i] % C.R().q, C.W()[i]));
    switch (L) {
    case Lattice::Mixed: {
        check_free(y, r, "y");
        hidapair::Vec Y(d_.n_sym + 1, 0);
        for (uint32_t j = 0; j < r; ++j) Y = F.add(Y, F.scale(y.coordinates[j] % C.R().q, C.Y()[j]));
        return C.pair(P, Y);
    }
    case Lattice::Hida: {
        check_free(y, r, "y");
        hidapair::Vec Q(d_.n_sym + 1, 0);
        for (uint32_t j = 0; j < r; ++j) Q = F.add(Q, F.scale(y.coordinates[j] % C.R().q, C.W()[j]));
        return C.pair(P, C.theta(C.iota_shriek(hidapair::Fibre::V, Q)));
    }
    case Lattice::MixedShriek: {
        std::vector<hidapair::Vec> WD = C.sign_basis(hidapair::Fibre::D, 0);
        const std::vector<hidapair::Vec> WDo = C.sign_basis(hidapair::Fibre::D, 1);
        WD.insert(WD.end(), WDo.begin(), WDo.end());
        check_free(y, WD.size(), "y");
        hidapair::Vec Q(d_.n_sym + 1, 0);
        for (size_t j = 0; j < WD.size(); ++j) Q = F.add(Q, F.scale(y.coordinates[j] % C.R().q, WD[j]));
        return C.pair(P, C.iota_shriek(hidapair::Fibre::D, Q));
    }
    }
    return 0;
}

std::string HidaPairingRoute::adjunction() const {
    return "T_p^* = T_p (level 1, no W_N; A_D(T_p) = A_c(T_p)^T checked at p = 2, 3, 5); <iota x, iota y> = -<x, y> "
           "(geo-even pairs with geo-odd); <x, y> = <P, y> = -(Hecke/HeckePairing.h cone cup form P_1)";
}

IndexStatement HidaPairingRoute::index_statement() const {
    RequireSupported(d_);
    const Lattice L = ParseLattice(d_.cohomology);
    const bool odd_sign = d_.subquotient.find("geo-odd") != std::string::npos;
    const uint32_t k = d_.n_sym + 2;
    IndexStatement I;
    I.l_value = std::string("L(k, Sym^2 f) / (Omega^+ Omega^-) on ") + LatticeName(L) + " (norm over the eigenforms of H')";
    uint32_t found = 0, expected = 0, e = 1;
    bool all = true;
    if (NamesBlock(d_) && L != Lattice::Hida) {
        // (slot-20 WP-LV3) THE GENERIC READER: no frozen factor; at (258, 37) it reproduces WP-BIL's ReadCell258 bit for bit
        // (HeckeLValueValidation's (LV3-R) arm compares the two).
        mblock::MBlockOptions opt;
        opt.p9 = L == Lattice::MixedShriek;
        opt.hd_torsion = false;
        const uint32_t M = BlockPrecision(d_);
        const hidapair::Zq R(d_.ell, M);
        const uint64_t lambda0 = mblock::Lambda0FromSubquotient(d_.subquotient, R, d_.n_sym);
        if (d_.subquotient.find("T_2") == std::string::npos) opt.joint = mblock::EisensteinJoint(R, d_.n_sym);   // m_Eis: joint in T_2, T_3, T_5
        const mblock::MBlockReading C = mblock::ReadMBlockCell(d_.n_sym, d_.ell, M, lambda0, opt);
        const mblock::MBlockSign& S = C.sign[odd_sign ? 1 : 0];
        if (!S.complete) throw hecke::NotImplementedInThisBuild("HidaPairingRoute: the m-block reader refused: " + S.refusal);
        if (L == Lattice::MixedShriek && !S.have_p9)
            throw hecke::NotImplementedInThisBuild("HidaPairingRoute: P9 (H^1_!(D)_m) is not read on this block");
        for (const mblock::MBlockForm& f : S.forms) {
            e = f.e;
            const uint32_t val = L == Lattice::Mixed ? f.mixed : f.p9;
            const uint32_t exp = L == Lattice::Mixed ? f.cong : f.vpi_dP;
            found += val;
            expected += exp;
            all = all && val == exp;
        }
        for (const mblock::MBlockForm& f : S.forms) ENGINE_CHECK_MSG(f.e == e, "HidaPairingRoute: forms of different ramification in one norm");
        std::ostringstream lam;
        lam << lambda0 % d_.ell;
        I.label = L == Lattice::Mixed ? "[E] rmk:lvshapes (H2)/(H3) mixed at m = (" + std::to_string(d_.ell) + ", T_2 - " + lam.str() + ")"
                                      : "[E] rmk:lvshapes (H3) P9 through H^1_!(D)_m";
        I.status = L == Lattice::Mixed ? "\\PROVED (R19-9 Sec. 3 lemma) + COMP" : "COMP";
        I.module_name = std::string("congruence module of ") + (L == Lattice::Mixed ? "W" : "T_m (eta_f)") + "_m^{" + S.name +
                        "} (generic m-block reader, e = " + std::to_string(e) + ")";
    } else {
        if (d_.n_sym == 258 && d_.ell == 37) throw hecke::NotImplementedInThisBuild("HidaPairingRoute: H^1_! x H^1_! needs n + 1 < ell");
        if (L == Lattice::MixedShriek)
            throw hecke::NotImplementedInThisBuild("HidaPairingRoute: H^1_c(M_n) x H^1_!(D_n) is implemented at (258, 37) only");
        const hidapair::SmallCellReading C = hidapair::ReadSmallCell(d_.n_sym, d_.ell, d_.m);
        for (const hidapair::EigenReading& E : C.systems) {
            e = E.e;
            uint32_t val = 0, exp = 0;
            if (L == Lattice::Hida) {
                val = E.hida;
                exp = E.cong_odd;
            } else {
                val = odd_sign ? E.mixed_odd : E.mixed_even;
                exp = odd_sign ? E.cong_odd : E.cong_even;
            }
            found += val;
            expected += exp;
            all = all && val == exp;
        }
        I.label = L == Lattice::Hida ? "[E] rmk:lvshapes (H1) Hida H^1_!" : "[E] rmk:lvshapes (H2) mixed H^1_c x H^1(D)";
        I.status = "\\PROVED (R19-9 Sec. 3 lemma) + COMP";
        I.module_name = std::string("congruence module of ") + (L == Lattice::Hida ? "H^1_!" : "H^1_c(V)") + "^{" +
                        (L == Lattice::Hida ? std::string("geo-odd") : std::string(odd_sign ? "geo-odd" : "geo-even")) +
                        "} at each eigen-system (" + C.kind + ")";
    }
    ENGINE_CHECK_MSG(found % e == 0 && expected % e == 0, "HidaPairingRoute::index_statement: the norm valuation is not integral");
    I.found = found / e;
    I.expected_lo = I.expected_hi = expected / e;
    I.l_valuation = I.found;
    I.l_valuation_exact = d_.ell >= k;   // Hida's constant is an ell-unit exactly for ell >= k
    I.holds = all;
    return I;
}

} // namespace lvalue
} // namespace hecke_ext
