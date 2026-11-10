/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Cells.BockData

/-!
# HeckeZ M-HZ3: negative controls for K5, K6, K7, K8 (L-HZ2) — Mathlib-free

[LP] §2.5 mutation discipline, 07 §10 ("one tampered-certificate negative control per checker"). Each control changes one
field of a certificate the checker accepts (`GL2.HeckeZ.Cells.BockData`) and is rejected by `decide +kernel`.
-/

namespace GL2.HeckeZ.Cells.BockControls

open GL2.HeckeZ.Exec GL2.HeckeZ.Cells.BockData

/-- K5: the emitted `h^1` at `(1, 10)`, `ℓ = 2`, shifted by one. -/
def ctl_k5 : K5Cert := { k5_l2_n10 with h := k5_l2_n10.h.set 1 (k5_l2_n10.h.getD 1 0 + 1) }

theorem ctl_k5_rejected : k5Check ctl_k5 = false := by decide +kernel

/-- K5: one torsion multiplicity `μ_1(1)` at `(1, 10)`, `ℓ = 2`, shifted by one (the antecedent no longer predicts `h`). -/
def ctl_k5_mu : K5Cert :=
  { k5_l2_n10 with mu := k5_l2_n10.mu.set 1 ((k5_l2_n10.mu.getD 1 []).set 0 ((k5_l2_n10.mu.getD 1 []).getD 0 0 + 1)) }

theorem ctl_k5_mu_rejected : k5Check ctl_k5_mu = false := by decide +kernel

/-- K6 / TB-χ: the constant term of `χ` changed to `−514` (`x − 514 ≡ x`, not coprime to `∏ P_j ≡ x²` modulo 2). -/
def ctl_k6_chi : K6Cert := { k6_n8 with chi := [-514, 1] }

theorem ctl_k6_chi_rejected : k6Check ctl_k6_chi = false := by decide +kernel

/-- K6 / TB-mat: the torsion matrix with `T_2 g_1 = 3 g_1` on `ℤ/8` (no longer nilpotent; `P` unchanged). -/
def ctl_k6_mat : K6Cert := { k6_n8 with C := [[0, 0], [0, 3]] }

theorem ctl_k6_mat_rejected : k6Check ctl_k6_mat = false := by decide +kernel

/-- K7: the section `h` at `(1, 26)` shifted in its first coordinate (the split identity fails). -/
def ctl_k7_split : K7Cert := { k7_n26 with h := k7_n26.h.set 0 (k7_n26.h.getD 0 0 + 1) }

theorem ctl_k7_split_rejected : k7Check ctl_k7_split = false := by decide +kernel

/-- K7: the status `zero` claimed at `(1, 26)` (all valuations `0`): the Smith certificate fails. -/
def ctl_k7_zero : K7Cert := { k7_n26 with vals := k7_n26.vals.map fun _ => 0 }

theorem ctl_k7_zero_rejected : k7Check ctl_k7_zero = false := by decide +kernel

/-- K7 / TB-mat: `T_5` acting on the torsion line of `(1, 18)` by `1` instead of `0` (the Hom bound `1` fails). -/
def ctl_k7_mat : K7Cert := { k7_n18 with c := k7_n18.c.set 2 1 }

theorem ctl_k7_mat_rejected : k7Check ctl_k7_mat = false := by decide +kernel

/-- K8: the emitted residue of `T_2` on the `ℤ/25` of `(1, 30)` changed from `21` to `22`. -/
def ctl_k8 : K8Item := { k8_l5_n30_s0 with res := [(2, 22), (3, 22), (7, 6)] }

theorem ctl_k8_rejected : k8Check ctl_k8 = false := by decide +kernel

end GL2.HeckeZ.Cells.BockControls
