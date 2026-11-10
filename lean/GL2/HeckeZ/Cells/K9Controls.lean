/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Cells.K9Data

/-!
# K9 negative controls (L-K9; 07 §5.9 "a tampered B_k/k residue is rejected") — Mathlib-free

Each control is a real H1 certificate with one tampering. `checks c` lists the eight sub-checks of `k9Check` in order
`[guard, root, shape, val, bern, disc, sign, lam]`; each theorem states the **whole vector** (kernel), so it shows which checks fail.
Where the sub-checks are independent the control fails exactly one; where they are not (the shape at `g = 2` forces the
discriminant here; the guard), the control says so.
-/

namespace GL2.HeckeZ.Cells.K9Controls

open GL2.HeckeZ.Exec.K9 GL2.HeckeZ.Cells.K9Data

/-- The eight sub-checks of `k9Check`. -/
noncomputable def checks (c : K9Cert) : List Bool :=
  [guardB c, rootB c, shapeB c, valB c, bernB c, discB c, signB c, lamB c]

/-- `(547, 486)` as in `K9Data`. -/
def c547 : K9Cert :=
  { ell := 547, k := 486, flo := [135932, 62064], e := 76180, bk := 176134, gk := 123075, gnext := 47042, lam := 139 }

/-- `(691, 12)` as in `K9Data` (`f = x + 24`: `τ(2) = −24`). -/
def c691 : K9Cert :=
  { ell := 691, k := 12, flo := [24], e := 2049, bk := 202463, gk := 275018, gnext := 330298, lam := 611 }

/-- `c547` is the first certificate of `K9Data`. -/
theorem c547_eq : k9Cells[0]? = some c547 := rfl
/-- `c691` is the eleventh certificate of `K9Data`. -/
theorem c691_eq : k9Cells[10]? = some c691 := rfl
theorem c547_checks : checks c547 = [true, true, true, true, true, true, true, true] := by decide +kernel

/-- **The 07 control**: `B_k/k` a unit (`bk + 1`), with `gk`, `gnext` moved by `−1` so that the sign identity and the λ-test still
hold: **only** the Bernoulli valuation fails. -/
def ctlBern : K9Cert := { c547 with bk := 176135, gk := 123074, gnext := 47041 }
theorem ctlBern_only : checks ctlBern = [true, true, true, true, false, true, true, true] := by decide +kernel
theorem ctlBern_rejected : k9Check ctlBern = false := by decide +kernel

/-- `f(e) ≡ 0 (mod ℓ²)` (a valuation `≥ 2`, the rung case) at `(691, 12)`: `f = x − 2049`. Only the valuation fails. -/
def ctlVal : K9Cert := { c691 with flo := [475432] }
theorem ctlVal_only : checks ctlVal = [true, true, true, false, true, true, true, true] := by decide +kernel
theorem ctlVal_rejected : k9Check ctlVal = false := by decide +kernel

/-- A wrong `e` (`e + ℓ`: right mod `ℓ`, wrong mod `ℓ²`) at `(691, 12)`: only the recomputation of `1 + 2^{k−1}` fails. -/
def ctlRoot : K9Cert := { c691 with e := 2049 + 691 }
theorem ctlRoot_only : checks ctlRoot = [true, false, true, true, true, true, true, true] := by decide +kernel
theorem ctlRoot_rejected : k9Check ctlRoot = false := by decide +kernel

/-- `G_a(x_k) ≢ −B_k/k`: `gk + ℓ`, `gnext + ℓ` (the λ difference kept). Only the sign identity fails. -/
def ctlSign : K9Cert := { c547 with gk := 123075 + 547, gnext := 47042 + 547 }
theorem ctlSign_only : checks ctlSign = [true, true, true, true, true, true, false, true] := by decide +kernel
theorem ctlSign_rejected : k9Check ctlSign = false := by decide +kernel

/-- The λ-test failing: `G_a(x_{k+ℓ−1}) = G_a(x_k)`, λ-residue `0` (as if `λ(G_a) ≥ 2` were visible at this precision). Only the
λ-test fails. -/
def ctlLam : K9Cert := { c547 with gnext := 123075, lam := 0 }
theorem ctlLam_only : checks ctlLam = [true, true, true, true, true, true, true, false] := by decide +kernel
theorem ctlLam_rejected : k9Check ctlLam = false := by decide +kernel

/-- A non-Eisenstein shape at `(547, 486)`: `f + (x − e)`, so `f(e)` is unchanged but `f ≢ (x − e)² (mod ℓ)`. It fails the shape
**and** the discriminant (a quadratic with `f(e) = ℓ·unit` that is not `≡ (x − e)²` has a unit discriminant): the two are not
independent at `g = 2` (conversely, shape + valuation force `ℓ ∥ disc`). -/
def ctlShape : K9Cert := { c547 with flo := [135932 - 76180, 62064 + 1] }
theorem ctlShape_two : checks ctlShape = [true, true, false, true, true, false, true, true] := by decide +kernel
theorem ctlShape_rejected : k9Check ctlShape = false := by decide +kernel

/-- `ℓ = 549 = 9 · 61` (not prime): the guard fails (and with it the values computed at a wrong modulus). -/
def ctlPrime : K9Cert := { c547 with ell := 549 }
theorem ctlPrime_guard : guardB ctlPrime = false := by decide +kernel
theorem ctlPrime_rejected : k9Check ctlPrime = false := by decide +kernel

end GL2.HeckeZ.Cells.K9Controls
