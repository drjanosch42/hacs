/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.K10

/-!
# L-K10: negative controls of K10′ and K10 (Mathlib-free, `decide +kernel`)

* **A7-M1, `(N, ℓ, q) = (61, 5, 37)`** ([C] `prop:heckeZeiscert`(3); 07 §5.10 v0.4.1): `q = 37` is good, and
  `h_37 = (x − 8)(x³ + 6x² − 36x − 108)` (the polynomial quoted in [C]; expanded `864 + 180x − 84x² − 2x³ + x⁴`). Its Hensel
  factor at `1 + q = 38` has degree 2: the rational newform `61a` (`a_37 = 8 ≡ 38 mod 5`) enters it. `F` is the C++ factor of the
  forced-`q` arm of `HeckeMazurEisValidation` (`F = [815452951765477150, 424546063561453435, 1]` in `y = x − 38`), shifted to `x` by
  `lean/tools/hz_mazur_to_lean.py`'s `shift`; `H = h_37 / F mod 5^26`. Goodness, the guards and the Hensel factorisation all pass
  (`ctl61_37_rest`), and the **valuation check rejects it**: `v₅(F(38)) = 2 > 1 = v₅(ν)` (`ctl61_37_val`, `ctl61_37_rejected`).
  This is the control that a goodness-only checker would accept.
* **Non-good `q`** at `(61, 5)`: `13` is a 5-th power mod 61 (`13^{12} ≡ 1`; fails (i)), `31 ≡ 1 mod 5` (fails (ii))
  (`good_61_5`, which isolates `goodB`). The accepted cell certificate with only `q` replaced by `13` is rejected
  (`ctl61_q13_rejected`), but by several checks at once (goodness, the shape `F ≡ (x − 14)^e`, the unit and the valuation), so it
  does not isolate goodness. **Goodness-only controls** (audit MINOR-5): `ctlGood13`, `ctlGood31` at `(61, 5)` with `q = 13`,
  resp. `31`, and a synthetic `h = F = x − a + 5` (`a = 1 + q`; `H = 1`): the guards, the Hensel factorisation and the
  valuation (`F(a) = 5`) pass, and only goodness fails. The checker treats `h` as data (TB-χ is outside it), so a synthetic `h`
  is a legitimate test of the checker; it is not `c_13` or `c_31`.
* **A tampered `F`**: the accepted `(61, 5, 2)` certificate with `F(0)` changed by `+1` (`ctl61_F_rejected`; it fails the shape
  `F ≡ (x − 3) mod 5` as well as the Hensel identity). **A tampered `h`** (`ctl61_h_rejected`, isolating the identity
  `h ≡ F·H mod ℓ^K`): `h(0)` changed by `+1`, with `F` and `H` kept; the guards, goodness, the shape, the unit and the valuation
  pass, and only the identity fails.
* **K10**: a tampered `P` and a tampered `e` (the Merel cell `(31, 5)`, `e = 2`, read as `e = 1`) are rejected.
-/

namespace GL2.HeckeZ.Cells.MazurControls

open GL2.HeckeZ.Exec.K10

/-- The accepted `(61, 5)` certificate (`q = 2`, part A of the data). -/
def cert61 : K10pCert :=
  { N := 61, ell := 5, q := 2, K := 26, v := 1,
    h := [1, (-2), (-4), 0, 1],
    flo := [210475224362432592],
    H := [1062370616098458553, 851895391736025960, 1279640895022333033, 1] }

theorem cert61_accepted : k10pCheck cert61 = true := by decide +kernel

/-- The A7-M1 counterexample: `(61, 5)` at the good prime `q = 37`. -/
def ctl61_37 : K10pCert :=
  { N := 61, ell := 5, q := 37, K := 26, v := 1,
    h := [864, 180, (-84), (-2), 1],
    flo := [1073979849662669939, 424546063561453359],
    H := [681794319630510201, 1065570055823312264, 1] }

/-- Everything but the valuation check passes: the guards, goodness, the Hensel factorisation (degree 2). -/
theorem ctl61_37_rest : (cellB ctl61_37 && goodB ctl61_37.N ctl61_37.ell ctl61_37.q && henselB ctl61_37) = true := by
  decide +kernel

/-- The valuation check fails: `5² ∣ F(38)`. -/
theorem ctl61_37_val : valB ctl61_37 = false := by decide +kernel

theorem ctl61_37_rejected : k10pCheck ctl61_37 = false := by decide +kernel

/-- `13` and `31` are not good at `(61, 5)`; `2` is. -/
theorem good_61_5 : (goodB 61 5 13, goodB 61 5 31, goodB 61 5 2) = (false, false, true) := by decide +kernel

theorem ctl61_q13_rejected : k10pCheck { cert61 with q := 13 } = false := by decide +kernel

/-- Rejected by several checks (goodness, shape, unit, valuation); see `ctlGood13` for the goodness-only control. -/
theorem ctl61_q13_several :
    (goodB 61 5 13, henselB { cert61 with q := 13 }, valB { cert61 with q := 13 }) = (false, false, false) := by
  decide +kernel

/-- Goodness-only control at `q = 13` (a 5-th power mod 61): synthetic `h = F = x − 9`, `F(14) = 5`. -/
def ctlGood13 : K10pCert :=
  { N := 61, ell := 5, q := 13, K := 26, v := 1, h := [(-9), 1], flo := [(-9)], H := [1] }

/-- Goodness-only control at `q = 31` (`≡ 1 mod 5`): synthetic `h = F = x − 27`, `F(32) = 5`. -/
def ctlGood31 : K10pCert :=
  { N := 61, ell := 5, q := 31, K := 26, v := 1, h := [(-27), 1], flo := [(-27)], H := [1] }

/-- Everything but goodness passes on `ctlGood13`, `ctlGood31`; goodness fails; so both are rejected. -/
theorem ctlGood_only :
    (cellB ctlGood13 && henselB ctlGood13 && valB ctlGood13, goodB ctlGood13.N ctlGood13.ell ctlGood13.q,
     k10pCheck ctlGood13,
     cellB ctlGood31 && henselB ctlGood31 && valB ctlGood31, goodB ctlGood31.N ctlGood31.ell ctlGood31.q,
     k10pCheck ctlGood31) = (true, false, false, true, false, false) := by
  decide +kernel

/-- Rejected by the shape `F ≡ (x − 3) (mod 5)` and the Hensel identity. -/
theorem ctl61_F_rejected : k10pCheck { cert61 with flo := [210475224362432593] } = false := by decide +kernel

/-- A tampered `h` (`h(0) + 1`): only the identity `h ≡ F·H (mod 5^26)` fails. -/
theorem ctl61_h_rejected :
    (cellB { cert61 with h := [2, (-2), (-4), 0, 1] } && goodB 61 5 2 &&
       peqMod 5 cert61.F (plin cert61.a cert61.e) && peval cert61.H cert61.a % 5 != 0 &&
       valB { cert61 with h := [2, (-2), (-4), 0, 1] },
     peqMod (5 ^ 26) [2, (-2), (-4), 0, 1] (pmul cert61.F cert61.H),
     k10pCheck { cert61 with h := [2, (-2), (-4), 0, 1] }) = (true, false, false) := by
  decide +kernel

/-- K10 at `(61, 5)` (`e = 1`) and at the Merel cell `(31, 5)` (`e = 2`), and their tampers. -/
theorem merel_controls :
    (merelCheck { N := 61, ell := 5, P := 42, e := 1 }, merelCheck { N := 61, ell := 5, P := 43, e := 1 },
     merelCheck { N := 31, ell := 5, P := 1, e := 2 }, merelCheck { N := 31, ell := 5, P := 1, e := 1 }) =
    (true, false, true, false) := by decide +kernel

end GL2.HeckeZ.Cells.MazurControls
