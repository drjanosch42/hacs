/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Cells.FullNData

/-!
# M-HZ5 negative controls: every tampered matrix fails its kernel check (Mathlib-free)

One entry changed at a time; each theorem proves the check returns `false` (`decide +kernel`), so a tampered matrix cannot pass.
-/

namespace GL2.HeckeZ.Cells.FullNControls

open GL2.HeckeZ.Exec.TBInt GL2.HeckeZ.Cells.FullNData

/-- T4: the matrix of `U₂` with one entry changed (row 3, column 3: `128 → 127`). -/
def ctlU : IMat := [[1, 0, 0, 0], [-120, -8, 0, 120], [0, 0, -8, 0], [-64, 0, 0, 127]]

theorem ctl_eigen : ieigen 4 ctlU c2_6_vb (128) = false := by decide +kernel
theorem ctl_f : ieq 4 4 (imul 4 4 4 (imul 4 4 4 (ishift 4 ctlU 1) (ishift 4 ctlU 128)) (ishift 4 ctlU (-8))) [] = false := by decide +kernel

/-- T5: the first entry of `Q` changed by `1`. -/
def ctlQ : IMat :=
  match c8_10_Q with
  | [] => []
  | r :: rs => (match r with | [] => [] | x :: xs => (x + 1) :: xs) :: rs

theorem ctl_factor : ieq 22 22 (imul 22 22 22 c8_10_U c8_10_U) (imul 22 22 6 ctlQ c8_10_R) = false := by decide +kernel

/-- `(11, 0)`: `U₁₁` with one diagonal entry changed. -/
theorem ctl_identity : ieq 3 3 [[1, 0, 0], [0, 2, 0], [0, 0, 1]] c11_0_I = false := by decide +kernel

/-- `(23, 0)`: `T₂` with entry `(0, 0)` changed by `+1`. -/
def ctlT2 : IMat :=
  match c23_0_T2 with
  | [] => []
  | r :: rs => (match r with | [] => [] | x :: xs => (x + 1) :: xs) :: rs

/-- `(23, 0)`: the annihilation check `(T₂ − 3) c(T₂) = 0` fails for the tampered `T₂`, with `c(T₂)` recomputed from it (as the
data module computes `c23_0_C` from `c23_0_T2`). -/
theorem ctl_annih : ieq 5 5 (imul 5 5 5 (ishift 5 ctlT2 3) (iadd 5 5 (imul 5 5 5 ctlT2 ctlT2) (ishift 5 ctlT2 1))) [] = false := by
  decide +kernel

/-- `(23, 0)`: `T₂` with the entry `(0, 4)` changed by `+1` (`1` instead of `0`). -/
def ctlT2r : IMat := [[0, 0, -1, -1, 1], [1, 1, 2, 1, 0], [-1, -1, -2, 0, 0], [0, 1, 1, -1, 1], [0, 0, 0, 0, 3]]

/-- `(23, 0)`: the read-entry check `c(T₂)_{0,4} = −1` (`c23_0_Cent`) fails for the tampered `T₂` (the entry becomes `3`). -/
theorem ctl_read : (ient (iadd 5 5 (imul 5 5 5 ctlT2r ctlT2r) (ishift 5 ctlT2r 1)) 0 4 == -1) = false := by decide +kernel

/-- T4: `U₂` with entry `(3, 0)` changed (`−64 → −63`). -/
def ctlUp : IMat := [[1, 0, 0, 0], [-120, -8, 0, 120], [0, 0, -8, 0], [-63, 0, 0, 128]]

/-- T4: the pair-read entry check `((U₂ − 1)(U₂ + 8))_{1,0} = −7680` (`c2_6_Pent`) fails for the tampered `U₂` (`−7560`). -/
theorem ctl_pent : (ient (imul 4 4 4 (ishift 4 ctlUp 1) (ishift 4 ctlUp (-8))) 1 0 == -7680) = false := by decide +kernel

end GL2.HeckeZ.Cells.FullNControls
