// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once

// The degree-1 amalgam reader on the Z/ell^k ladder (slot-22 BIN-3; docs/implementation/slot22/L23_master_plan.md
// Sec. 2.1 / 2.2 / 5, replacing census_l23_plan.md v0.3's L23-2; report docs/implementation/slot22/BIN3_report.md).
//
// Gamma = SL_2(Z) = <S> *_{<-I>} <U>, U = S T, S^4 = U^6 = 1, S^2 = U^3 = -I ([C] cl:amalgam); V = Sym^n Z^2 with
// rho_n of [C] def:rhon (column j = (a x + c y)^(n-j) (b x + d y)^j, i.e. f(x, y) -> f(a x + c y, b x + d y)).
// Inhomogeneous 1-cocycles are pairs (c(S), c(U)) in V^2 subject to the three relator conditions
//     (1 + S + S^2 + S^3) c(S) = 0,   (1 + U + ... + U^5) c(U) = 0,   (1 + S) c(S) = (1 + U + U^2) c(U),
// the 3d x 2d relation matrix A (d = n + 1); B^1 = {((S - 1) v, (U - 1) v)}.  T_p is the bar formula of
// cor o conj_g o res ([C] eq:corconjres, prop:barformula), g = diag(1, p), Gamma_0(p) \ Gamma by the bottom row,
// representatives r_k = [[1,0],[k,1]] (k < p) and r_inf = S, c(gamma) by the cocycle rule along the Euclid word of
// gamma in S, T, T^-1 -- the same conventions as docs/implementation/slot22/mp/mp4/mp4_oracle.py and its readers
// ladic2.py / oddn_l2.py, whose outputs the gate L23AmalgamReaderValidation reproduces bit for bit.
//
// Arithmetic: ell = 2 on engine::ring2k (BIN-2: rung R1 ... R5 by the working precision K; nothing ever reduces);
// ell = 3 on the engine's Z/3^k (engine::ladic digit planes, ModCtx::prime_power for one digit; F_3 by
// engine::elim).  Every operator on V is applied structurally, never as a dense matrix: rho(T)^{+-1} is a Taylor
// shift by +-1 (additions only), rho(S) a signed reversal, rho(diag(1, p)) a diagonal, and every other element
// through its word.
//
// What a lattice cell computes (read_lattice), with its precision bookkeeping (the plan's m = K - e_1 - e_2):
//   * sigma(A) over Z/ell^K with the rank over Q (two large primes) supplied as the rank hint (at ell = 2 through
//     ring2k::smith_adaptive); e_A := max sigma(A).  The saturated cocycle lattice L = ker_Z(A) (x) Z_ell is
//     red_{K - e_A} of ker(A mod ell^K) (generators: the Howell form of the kernel module), a free direct summand,
//     written in its unit-pivot reduced basis (z = 2d - rank_Q(A) rows, identity on the F_ell rank-profile columns;
//     NOT the Howell form, whose leading entries need not be units on a free summand).
//   * C = the coordinates of B^1 in that basis (z x d), sigma(C) over Z/ell^(K - e_A): the torsion exponents of
//     H^1(Gamma; V) (> 0), e_1 := max sigma(C); Lambda^1 = L / sat(B^1) via its dual, the left kernel of C reduced
//     to m := K - e_A - e_1 (rank_lambda rows, again in unit-pivot reduced form; the dual basis lambda_j of Lambda^1
//     is the L basis rows at its pivot columns).
//   * T_p (p in primes), and at ell = 3 the involution iota := -rho(eps) c(eps . eps), eps = diag(-1, 1) ([C] def:iota,
//     OQ3 sign), as rank_lambda x rank_lambda matrices over Z/ell^m; elementary divisors of T_p - (1 + p),
//     T_p - (1 + p^(n+1)) (p != ell), T_ell, T_ell^2 - T_ell, and at ell = 3 the same on Lambda^+ / Lambda^-.
//   Divisors are exact iff < m.  The cell reruns at a higher precision (one rung up at ell = 2, one digit up at
//   ell = 3; logged) until m >= ReaderOptions::m_target.
// h1_mod computes log_ell |H^1(Gamma; V / ell^k)| and the lengths of operator images on it (any k, any n);
// oddn_row the 22 columns of oddn_l2.py (odd n, ell = 2).
//
// Single-threaded; no setting, no golden object, no record.  The census hook is L23-1's (census_l23_plan.md Sec. 6).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../../ModArithmetic/EnvLever.h"

namespace l23 {

// Elementary divisors of an operator on a free Z_ell-module of rank r, read at precision cap (the census
// convention, census_l23_plan.md Sec. 2.3): `finite` ascending, 0 = a unit; `at_least` divisors are >= cap
// (zero diagonal entries included).  finite.size() + at_least == r.
struct Divisors {
    std::vector<uint32_t> finite;
    uint32_t at_least = 0;
    uint32_t cap = 0;
    uint32_t units() const;
    uint32_t rank() const { return static_cast<uint32_t>(finite.size()) + at_least; }
    std::string ladic2() const;   // ladic2.py's fmt(): the positive ones comma-joined (or "-"), then "+[>=cap]^c"
    std::string census() const;   // every divisor: "0,0,3,3,>=240" style
    bool operator==(const Divisors& o) const { return finite == o.finite && at_least == o.at_least && cap == o.cap; }
};

struct ReaderOptions {
    uint32_t ell = 2;                            // 2 or 3
    uint32_t k_min = 0;                          // first working precision K (0: 64 at ell = 2, 78 at ell = 3)
    uint32_t k_cap = 1024;                       // the largest K tried
    uint32_t m_target = 0;                       // rerun until the certified precision m >= m_target
    uint32_t cap = 0;                            // report threshold (0: the certified m); must be <= m
    std::vector<uint32_t> primes = {2, 3, 5, 7}; // T_p for these p (T_ell is always included when ell is listed)
    bool pm = true;                              // ell = 3: Lambda^+- and the ed_*_pm
    uint32_t mutate = 0;                         // TEST-ONLY negative controls (validation use): 0 = none
};

struct LatticeCell {
    uint32_t ell = 0, n = 0, d = 0;
    uint32_t K = 0, m = 0, e_A = 0, e_1 = 0;     // working / certified precision, the two losses
    std::string rung;                            // "R5(N=4)" (ell = 2) or "Z/3^K(L=5)" (ell = 3)
    uint32_t reruns = 0;
    uint32_t rank_q_A = 0, z = 0, rank_lambda = 0, h0 = 0;
    std::vector<uint32_t> torsion;               // exponents of H^1(Gamma; V)_ell, descending (ladic2's T1)
    std::map<uint32_t, Divisors> ed_tp, ed_eis; // p != ell
    Divisors ed_tell, ed_tell2;                  // T_ell, T_ell^2 - T_ell
    bool commute = false;                        // every pair of the T_p commutes mod ell^m
    // ell = 3 (pm):
    bool pm_done = false, iota_ok = false;       // iota^2 = 1 and iota T_p = T_p iota mod 3^m
    uint32_t dim_plus = 0, dim_minus = 0;
    std::map<uint32_t, Divisors> ed_tp_plus, ed_tp_minus, ed_eis_plus, ed_eis_minus;
    Divisors ed_tell_plus, ed_tell_minus;
    bool ok = false;                             // every internal check held (see `failures`)
    std::vector<std::string> failures;
    std::vector<std::string> log;                // BIN3_ADAPTIVE / RING2K_ADAPTIVE lines
    double seconds = 0;
    // "ell=2 n=10 rankL1=3 T1=[2, 1] p3:3,3,3 ... Tl:3,3 Tl2-Tl:3,3,11" -- ladic2.py's line without comm_v / s.
    std::string ladic2_line() const;
};
LatticeCell read_lattice(uint32_t n, const ReaderOptions& opt);

// H^1(Gamma; V / ell^k): log_ell of its order, of B^1-classes of the images of the listed operators, and the
// matrices' consistency; `ops` entries are (p, c): the operator T_p - c (c an integer; c = 0 gives T_p itself).
struct ModCell {
    uint32_t ell = 0, n = 0, k = 0;
    uint32_t h0 = 0;                             // dim H^0(Gamma; Sym^n F_ell^2)
    uint32_t length = 0;                         // log_ell |H^1(Gamma; V / ell^k)|
    std::vector<uint32_t> image_length;          // per ops entry: log_ell |(T_p - c) H^1(Gamma; V / ell^k)|
};
ModCell h1_mod(uint32_t ell, uint32_t n, uint32_t k, const std::vector<std::pair<uint32_t, int64_t>>& ops,
               uint32_t mutate = 0);

// (slot-23 TER-1) The F_3 route of h1_mod at ell = 3, k = 1: entries uint8_t in {0, 1, 2}, the structural rho_n on
// vectorised byte rows (binary::f3_add_u8 / f3_sub_u8 / f3_neg_u8) with rho(T)^{+-1} by the characteristic-3 Taylor
// shift ((t +- 1)^(3^j) = t^(3^j) +- 1: O(d log_3 d) row operations instead of d^2 / 2), and the eliminations on the
// bit-sliced binary::TritMatrix.  ON by default: it cannot change a value (reduction Z/9 -> F_3 is a ring map; RREF,
// rank and the D5b kernel basis are unique), and TernaryF3Validation compares it with the Ops3<1> route cell by cell.
// L23_TERNARY_F3=0 (read once per process) is the read-only rollback to the Ops3<1> route; ScopedTernaryF3Reader
// overrides both (process-wide; validation use).
// (S25 LEVERS-25) The env arm L23_TERNARY_F3=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool ternary_f3_reader_enabled();
// (TER-1 audit F4) "EFFECTIVE l23_ternary_f3=on|off source=default|env|scoped ...": print once per run wherever ell = 3, k = 1 cells run.
std::string ternary_f3_effective_line();
// (TER-1 audit F1, validation hook) Rho<Ops3F>::T / Ti (the characteristic-3 shift) against the d^2 / 2 row-operation shift on a
// random (n + 1) x r matrix over F_3, plus T^-1 T = id: the number of differing entries (0 = pass).
uint64_t taylor3_selfcheck(uint32_t n, uint32_t r, uint64_t seed);
struct ScopedTernaryF3Reader {
    engine::lever::EnvLever::Scoped guard_;   // S24 CL-7: the lever skeleton's RAII override
    explicit ScopedTernaryF3Reader(bool on);
    ScopedTernaryF3Reader(const ScopedTernaryF3Reader&) = delete;
    ScopedTernaryF3Reader& operator=(const ScopedTernaryF3Reader&) = delete;
};

// (slot-23 BIN3O1; docs/implementation/slot23/BIN3O1_report.md) BIN-3 open item O1, value-neutral, ON by default:
//   * the relation matrix and B^1 in O(d^2): rho(T) on the identity is the Pascal matrix, rho(U) = rho(S) P, rho(U)^2 a signed
//     column permutation of it, rho(U)^3 = rho(-I) = (-1)^n (instead of five O(d^3) Taylor shifts of I_d; every ring);
//   * the parity structure of A (checked entrywise): N_S = (1 + (-1)^n)(1 + S), N_U = (1 + (-1)^n)(1 + U + U^2).  The rank over Q
//     (rank_q_relation, F_p) from two d x d ranks (n even) or one d x 2d rank (n odd); read_lattice's L from two d x d kernel
//     modules (n even; accepted only when unit_basis finds a free summand of rank z, which then IS L, else the old route) or from
//     [1 + S | -(1 + U + U^2)] (n odd: the other rows are zero);
//   * on the native rungs R1 ... R3 the Taylor shifts through ring2k's kernel_shift_rows (the same sweep row j -= f row (j + 1),
//     as a register wavefront of 16 passes per column vector) and mat_add / mat_sub through its AXPY kernel; on the native,
//     F_2 and F_3 rings a run T^q of a word is one shift by q, and in the cocycle evaluation a run of q >= 4 letters is
//     c(T^q) = sum_{j<q} T^j c(T) by doubling (exact identities; R4, R5, Z/3^k (k >= 2) and F_p keep the letter-by-letter loop);
//   * h1_mod at ell = 2, k = 1 on F_2: uint8_t entries {0, 1}, the characteristic-2 Taylor shift (O(d log d) row XORs), the
//     eliminations on BIN-1's M4RI binary::BitMatrix (RREF and the D5b kernel basis are unique: the Ops2<uint8_t> objects), and
//     T_p Z computed once per distinct p.
// The blocked unit_basis of BIN-3 O1 is not built: unit_basis costs 0.1-0.2 % of an r3 cell (its input is a Howell form, already
// reduced above; BIN3O1_predictions.md Sec. 0).  L23_BIN3O1=0 (read once per process) is the read-only rollback to the old route;
// ScopedBin3o1 overrides both (process-wide; validation use).  L23Bin3o1Validation compares the routes.
// When the switch is read (BIN3O1 audit F4): a Rho object (the Taylor shifts, the relation matrix and B^1, the split of A) fixes its
// route at construction; mat_add / mat_sub and h1_mod's F_2 dispatch read it at call time.  A ScopedBin3o1 flipped while a cell is in
// flight can therefore mix the routes inside that cell; the values are the same either way (validation use only).
// (S25 LEVERS-25) The env arm L23_BIN3O1=0 is RETIRED in OC_ENABLE_LEGACY=OFF builds (refused, exit 1: EnvLever.h EnvArmRetiredLever); the old route stays via the Scoped override.
bool bin3o1_enabled();
// "EFFECTIVE l23_bin3o1=on|off source=default|env|scoped ...": deep_cell and a verbose orbifold campaign print it.
std::string bin3o1_effective_line();
struct ScopedBin3o1 {
    engine::lever::EnvLever::Scoped guard_;   // S24 CL-7: the lever skeleton's RAII override
    explicit ScopedBin3o1(bool on);
    ScopedBin3o1(const ScopedBin3o1&) = delete;
    ScopedBin3o1& operator=(const ScopedBin3o1&) = delete;
};
// Route counters since process start (validation: non-vacuity).
struct Bin3o1Counters {
    uint64_t split_accepted = 0, split_fallback = 0;   // read_lattice, n even: L from the two d x d kernels / the old route after it
    uint64_t odd_reduced = 0;                          // read_lattice, n odd: L from the d x 2d third block row
    uint64_t rankq_blocks = 0;                         // rank_q_relation primes read from the blocks
    uint64_t f2_cells = 0;                             // h1_mod calls on the F_2 route
    uint64_t hecke_reused = 0;                         // h1_mod ops entries that reused T_p Z
    // (BIN3O1 audit F1) the route on but the parity structure of A not found, so the old route ran (expected never):
    uint64_t rankq_fallback = 0;                       // rank_q_relation: rank of the full A
    uint64_t parity_fallback = 0;                      // read_lattice (no negative control): L from the full A, either parity
};
Bin3o1Counters bin3o1_counters();
// (BIN3O1 audit F1) "BIN3O1_ROUTE route=on|off split=<accepted>/<tried> odd=… rankq=… f2=… reused=… split_fallback=…
// rankq_fallback=… parity_fallback=… fallback=<their sum>" from the counters since process start: deep_cell prints it after its cell,
// a verbose orbifold campaign after its last cell.  fallback=0 is expected (a fallback is the old route: right values, old cost).
std::string bin3o1_route_line();
// Validation hooks (numbers of differing entries; 0 = pass):
//   taylor2_selfcheck: the characteristic-2 shift (Ops2F T / Ti) against the d^2 / 2 shift of Ops2<uint8_t> mod 2 on a random
//     (n + 1) x r 0/1 matrix, plus T^-1 T = id;
//   relation_selfcheck: the relation matrix and B^1, route on vs off, over Z/ell^k (ell = 2: the ring2k rung of k; ell = 3: Z/3^k),
//     and the parity structure on the on-route blocks (one count if it fails);
//   rows_selfcheck: rho(T), rho(T)^-1, rho(U), a word, mat_add, mat_sub on a random (n + 1) x r matrix over the ring2k rung of k
//     (the AXPY row kernels and kernel_shift_rows' wavefront on R1 ... R3), route on vs off.
uint64_t taylor2_selfcheck(uint32_t n, uint32_t r, uint64_t seed);
uint64_t relation_selfcheck(uint32_t ell, uint32_t n, uint32_t k);
uint64_t rows_selfcheck(uint32_t k, uint32_t n, uint32_t r, uint64_t seed);

// (slot-22 L23-3; L23_1_audit.md Sec. 7 items 4-6) The layers of H^1(Gamma; M), M = Sym^n F_ell^2, on the amalgam, over F_ell:
//   (B)  H^1_Br = ker(H^1 -> H^1(<S>; M) + H^1(<U>; M)) = {c : c(S) in (S - 1)M, c(U) in (U - 1)M} / B^1 ([R] Thm E: = im eps^*,
//        Hecke-stable; every finite subgroup is conjugate into <S> or <U>);
//   (U)  H^1(V)/ell = the cocycles mod ell^E reduced mod ell (E - 1 >= exp T^2, the UCT reading), and T^1/ell = the Bockstein image
//        {delta u / ell^a mod ell : delta u = 0 mod ell^a}, a = a_max >= exp T^1 (beta_a : H^0(V/ell^a) -> H^1(V)[ell^a] is onto);
//   (!)  H^1_! = ker(H^1 -> H^1(Gamma_inf; M)), Gamma_inf = +-<T> ([C] def:bdrygroup): c(T) = c(S^-1) + S^-1 c(U) and
//        c(-I) = (1 + S) c(S) must be a boundary coboundary ((T - 1) v, (-I - 1) v);
//   (L)  ell = 3: L = ker(H^1 -> H^1(Gamma(3); M)) = {c : c(g) = 0 on the four cusp parabolics of Gamma(3)} (Gamma(3) acts
//        trivially on M and is generated by them), [R] Thm C(3): H^1 = H^1_Br + L.
// The Jordan types of T_ell (eigenvalues 0 and 1, block sizes descending; `other` = the rest) on H^1, H^1_Br, H^1 / H^1_Br, H^1_!.
struct Jordan01 {
    uint32_t dim = 0, rank = 0;
    std::vector<uint32_t> eig0, eig1;
    uint32_t other = 0;
    bool valid = true;   // false: T_ell does not preserve the subquotient (the ranks of (T - lam)^j grew); eig* empty, other = dim
};
struct LayerCell {
    uint32_t ell = 0, n = 0, a_max = 0, E = 0;
    uint32_t h0 = 0, h1 = 0;
    uint32_t dim_br = 0;                         // H^1_Br
    uint32_t dim_sub = 0;                        // H^1(V)/ell
    uint32_t dim_t1 = 0;                         // T^1/ell
    uint32_t dim_t1_br = 0;                      // dim(T^1/ell cap H^1_Br)
    uint32_t dim_cusp = 0;                       // H^1_!
    uint32_t dim_cusp_br = 0, dim_cusp_sub = 0, dim_cusp_t1 = 0;
    bool br_stable = false, cusp_stable = false; // T_ell preserves H^1_Br / H^1_!
    bool t1_cocycles = false;                    // internal: the Bockstein columns are cocycles mod ell
    Jordan01 j_whole, j_br, j_quot, j_cusp;
    bool l_done = false;                         // ell = 3, n even
    uint32_t dim_L = 0, dim_L_br = 0;            // dim L, dim(L cap H^1_Br)
    bool L_in_T1 = false, L_in_sub = false, L_stable = false;
    double seconds = 0;
};
// The block sizes (descending) of a nilpotent part from its rank sequence r_j = rank (T - lam)^j, j = 0 .. m (r strictly
// decreasing, the last entry the stable rank). false (blocks empty) when r is not a rank sequence of an operator: r not strictly
// decreasing, or the differences r_{j-1} - r_j increase (L23_3_audit.md M1: e.g. r = (3, 2, 0) on a non-T-stable subquotient).
bool jordan_blocks_from_ranks(const std::vector<uint32_t>& r, std::vector<uint32_t>& blocks);
// a_max >= the largest exponent of T^1 (the reader's torsion; 0 = no torsion); E >= 1 + exp T^2 (4 at ell = 2, 2 at ell = 3).
// mutate (TEST-ONLY): 1 = drop r_inf from the bar formula (as read_lattice), 3 = take c(T) without the S^-1 c(U) term.
LayerCell h1_layers(uint32_t ell, uint32_t n, uint32_t a_max, uint32_t E, uint32_t mutate = 0);

// oddn_l2.py's 22 columns for odd n at ell = 2 (K = its argument, K <= 8): h0, h0_dickson, h1, h1_pred, d1, d2,
// exp2_K, dimK, T3_K, T5_K, T7_K, T3_M, T5_M, T7_M, rkT2, rkT2^2, rkT2-1, T2idem, rkT2|sub, T2sub_in_sub,
// imT2_in_sub, T2_K.
struct OddnRow {
    uint32_t n = 0;
    std::vector<uint32_t> v;   // 22 values
    std::string line() const;  // "n v0 ... v21" as oddn_l2.py prints them (without s)
};
OddnRow oddn_row(uint32_t n, uint32_t K, uint32_t mutate = 0);

// Internal consistency (validation use): the structural rho(g) of the reader against the dense Sym^n matrix
// built from [C] def:rhon, for g in {S, S^-1, T, T^-1, U, eps, diag(1, p), r^-1 diag(1, p), ...}; returns the
// number of mismatching (g, entry) pairs over Z/ell^k.
uint32_t rho_selfcheck(uint32_t ell, uint32_t n, uint32_t k);

// dim M_k (level 1, k even >= 0) and the rank of Lambda^1 = 2 dim M_{n+2} - 1 (n even >= 2), 0 for n odd.
uint32_t dim_modular_forms(uint32_t k);
uint32_t expected_rank_lambda(uint32_t n);

} // namespace l23
