// (hecke-extensions slot 20, stretch S2) the frozen values of the 691 arm of HeckeLValueMeasureValidation (env
// HECKE_LVALUE_MEASURE_691=1): the l-stabilised MTT measure of Delta at (n, ell) = (10, 691), R = 2, level M = 2, against
// Kubota-Leopoldt.  Sources, all committed before the C++ arm's first run:
//   * the theory predictions docs/implementation/slot20/s1s2/predictions.md (commit c361d65c): tau(p) mod 691^2 from the
//     q-expansion of Delta, the totals, the minus (mu, lambda) table (lambda = 1 at {199, 501}; the pole branches by analogy);
//   * R20-1's Python oracle (r201/measlib.py, unchanged) at (10, 691), R = 2, M = 2: s, mu^{+-}(2 + ell^2), the FNV-1a-64 of the
//     level-2 unit balls, and the plus / reduced-plus tables, which theory does not predict (s1s2/oracle691.md).
// NEVER re-pinned from this code's output: a disagreement is a finding against [E] chk:lvmeas (691 arm).  Included inside the
// anonymous namespace of the validation TU, after MeasureFrozenR201.inc (whose structs it reuses).

// tau(p) mod 691^2 (exact q-expansion, predictions.md); a_ell = tau(691); alpha == tau(691) mod 691^{11}.
struct FrozenTau691 {
    uint64_t p, tau_mod;
};
const FrozenTau691 kFrozenTau691[] = {{2, 477457ull}, {3, 252ull}, {5, 4830ull}, {7, 460737ull}, {691, 374523ull}};

// The oracle's values at R = 2, level 2 (r^- first unit coordinate 1, r^+_0 = 1; FNV as MeasureFrozenR201.inc's).
struct FrozenOracle691 {
    uint64_t s, mu_minus_2, fnv_minus, mu_plus_2, fnv_plus;
};
const FrozenOracle691 kFrozenOracle691 = {286647ull, 0ull, 0x93d02a7b4bf3e78cull, 172059ull, 0x5abd25b37b4c099dull};

// Totals at (10, 691): moments 6 x 11 per sign; 343 off-pole odd branches x 11 j; 2 pole branches x 11 j; 343 off-pole
// branches; 342 wrong-twist branches (a = 687 skipped).
const FrozenTotals kFrozenTotalsS2[] = {
    {10, 691, 66, 3773, 22, 343, 342},
};

const std::vector<FrozenInvariants>& FrozenInvariantTablesS2() {
    static const std::vector<FrozenInvariants> t = {
        // minus: theory (Q-691-8): (0,0) except lambda = 1 at 199, 501; the pole branches 11, 689 (0,0) by analogy with (30,37)
        {10, 691, -1, 0, 0, {{199, 0, 1}, {501, 0, 1}}},
        // plus and reduced plus: the Python oracle (Q-691-10)
        {10, 691, +1, 1, 0, {{22, 1, 1}, {350, 1, 1}, {678, 1, 1}}},
        {10, 691, 0, 0, 0, {{22, 0, 1}, {350, 0, 1}, {678, 0, 1}}},
    };
    return t;
}
