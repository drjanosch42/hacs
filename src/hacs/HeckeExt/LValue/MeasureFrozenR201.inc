// (hecke-extensions slot 20, WP-MEAS) R20-1's frozen values -- the numbers of record of HeckeLValueMeasureValidation.
// Source: docs/implementation/slot20/R20-1_measures.md Sec. 4.3 (r201/freeze.py -> r201/out/freeze.out), Sec. 4.5 (the
// (mu, lambda) tables, r201/out/c*_37*.out, c226_103*.out), Sec. 4.4 (the located pole points at 103, c226_103_full2.out);
// the 14 points at (282, 37) are WP-MEAS's prediction (docs/implementation/slot20/wpmeas/predictions.md Q4(d), committed
// before the first run).  [E] 1.20.0-pre.1 chk:lvmeas (M2), (M4d), (M5).  NEVER re-pinned from this code's output: a
// disagreement is a finding against [E] chk:lvmeas.  Included inside an anonymous namespace of the validation TU.

// (M2) mod ell^4, level 2; r^- scaled to first unit coordinate 1, r^+ to r^+_0 = 1; FNV-1a-64 of mu(a + ell^2 Z_ell),
// units a ascending, 8 LE bytes each.  s = -1: no Eisenstein cell.
struct FrozenMeasureCell {
    uint32_t n;
    uint64_t ell;
    uint32_t d;
    uint64_t a2, a3, a_ell, alpha;
    int64_t s;
    uint64_t mu_minus_2, fnv_minus, mu_plus_2, fnv_plus;
};
const FrozenMeasureCell kFrozenMeasure[] = {
    {10, 37, 0, 1874137ull, 252ull, 1454464ull, 1454464ull, -1, 272431ull, 0x721edc172f8e24bdull, 847858ull, 0xcc0c00522e7e0429ull},
    {16, 37, 0, 1873633ull, 1869877ull, 1454991ull, 1454991ull, -1, 57498ull, 0xed8c569e28ac74beull, 698836ull, 0x57a02c809eb73309ull},
    {30, 37, 1, 1622769ull, 150214ull, 270360ull, 270360ull, 922897, 1022643ull, 0xf6fef89b62e18907ull, 78662ull, 0x6ddd3cd8d0e6a6adull},
    {282, 37, 2, 1079424ull, 1744174ull, 1770118ull, 1770118ull, 575834, 1326561ull, 0xc1d253ded2d790cdull, 1815294ull, 0xa812126c962c7079ull},
    {226, 103, 2, 15697592ull, 81971822ull, 50764066ull, 50764066ull, 83864971, 44186485ull, 0x5fb753f5ed92f150ull, 95024813ull,
     0xde51edced2aacde1ull},
};

// (M5) branch invariants (mu, lambda) on the branches of the right parity: every branch reads (def_mu, def_lambda) except the
// listed ones (valuations capped at R; level 3 at 37, level 2 at 103).  sign: -1 = geometric-odd (r^-), +1 = geometric-even
// (r^+), 0 = the reduced plus measure ell^{-d} mu^+ (Eisenstein cells only).
struct FrozenInvariantEntry {
    uint32_t branch, mu, lambda;
};
struct FrozenInvariants {
    uint32_t n;
    uint64_t ell;
    int sign;
    uint32_t def_mu, def_lambda;
    std::vector<FrozenInvariantEntry> except;
};
const std::vector<FrozenInvariants>& FrozenInvariantTables() {
    static const std::vector<FrozenInvariants> t = {
        {10, 37, -1, 0, 0, {{21, 0, 1}, {25, 0, 1}}},
        {10, 37, +1, 0, 0, {}},
        {16, 37, -1, 0, 0, {{21, 0, 1}, {31, 0, 1}}},
        {16, 37, +1, 0, 0, {{8, 0, 1}, {22, 0, 1}, {26, 0, 1}, {30, 0, 1}}},
        {30, 37, -1, 0, 0, {}},
        {30, 37, +1, 1, 0, {{4, 1, 1}, {26, 1, 1}}},
        {30, 37, 0, 0, 0, {{4, 0, 1}, {26, 0, 1}}},
        {282, 37, -1, 0, 0, {}},
        {282, 37, +1, 2, 0, {{4, 2, 1}, {26, 2, 1}}},
        {282, 37, 0, 0, 0, {{4, 0, 1}, {26, 0, 1}}},
        {226, 103, -1, 0, 0, {}},
        {226, 103, +1, 2, 0, {{46, 2, 1}, {62, 2, 1}, {78, 2, 1}}},
        {226, 103, 0, 0, 0, {{46, 0, 1}, {62, 0, 1}, {78, 0, 1}}},
    };
    return t;
}

// (M4d) the pole-branch points where the naive product -s G_{a+1} G_{k-1-a} fails modulo ell^{d-1} (d = 2):
// (226, 103) located by R20-1; (282, 37) WP-MEAS's prediction by the rule (j == k-1 mod ell on the branch k-1, j == -1 mod ell
// on the branch -1).
struct FrozenPolePoint {
    uint32_t a, j;
};
const std::vector<FrozenPolePoint>& FrozenPolePoints(uint32_t n, uint64_t ell) {
    static const std::vector<FrozenPolePoint> p282 = {{31, 24}, {31, 61}, {31, 98}, {31, 135}, {31, 172}, {31, 209}, {31, 246},
                                                      {35, 36}, {35, 73}, {35, 110}, {35, 147}, {35, 184}, {35, 221}, {35, 258}};
    static const std::vector<FrozenPolePoint> p226 = {{23, 21}, {23, 124}, {101, 102}, {101, 205}};
    static const std::vector<FrozenPolePoint> none;
    if (n == 282 && ell == 37) return p282;
    if (n == 226 && ell == 103) return p226;
    return none;
}

// (M1) / (M4b) / (M4d) R20-1's totals: moment checks per sign (6 characters x (n+1)), the off-pole and the pole critical moments.
struct FrozenTotals {
    uint32_t n;
    uint64_t ell;
    uint32_t moments_per_sign, off_pole, pole, branches_off_pole, branches_wrong_twist;
};
const FrozenTotals kFrozenTotals[] = {
    {10, 37, 66, 0, 0, 0, 0},
    {16, 37, 102, 0, 0, 0, 0},
    {30, 37, 186, 496, 62, 16, 15},
    {282, 37, 1698, 4528, 566, 16, 15},
    {226, 103, 1362, 11123, 454, 49, 48},
};
