/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Cells.S1SturmData

/-!
# HeckeZ S1: negative controls for the Prop. 5.6 checker (LC-11, slot 25) — Mathlib-free

[LP] §2.5 mutation discipline, 07 §10 ("one tampered-certificate negative control per checker"). Each control changes one field
of a certificate that `sturmCheck` accepts (T2 = `(1, 22)`: `sturm_n22` on `cert_n22`, reads at 103 and 2294797) and is
rejected by `decide +kernel`.
-/

namespace GL2.HeckeZ.Cells.S1SturmControls

open GL2.HeckeZ.Exec GL2.HeckeZ.Cells.S1DataB GL2.HeckeZ.Cells.S1SturmData

/-- T2 claimed complete by Prop. 5.6 without the read at 2294797 (`g ∤ cof56 · ∏ ℓ^ex`). -/
def ctl56_noread : SturmCert :=
  { sturm_n22 with reads := sturm_n22.reads.filter (fun r => r.ell != 2294797), cof56 := 1, cof := 1 }

theorem ctl56_noread_rejected : sturmCheck cert_n22 ctl56_noread = false := by decide +kernel

/-- A read claiming more than it reads (`v := 2` at 103: `103^{s+v} ∤ c(λ_p)`). -/
def ctl56_read_v : SturmCert :=
  { sturm_n22 with reads := sturm_n22.reads.map fun r => if r.ell == 103 then { r with v := 2 } else r }

theorem ctl56_read_v_rejected : sturmCheck cert_n22 ctl56_read_v = false := by decide +kernel

/-- A read at a composite modulus (`ℓ := 2 · 2294797`). -/
def ctl56_read_composite : SturmCert :=
  { sturm_n22 with reads := sturm_n22.reads.map fun r => if r.ell == 2294797 then { r with ell := 2 * 2294797 } else r }

theorem ctl56_read_composite_rejected : sturmCheck cert_n22 ctl56_read_composite = false := by decide +kernel

/-- An item's cuspidal polynomial shifted (`(x − λ_p) c ≠ f`). -/
def ctl56_item : SturmCert :=
  { sturm_n22 with items := sturm_n22.items.map fun it => { it with c := it.c.set 0 (it.c.getD 0 0 + 1) } }

theorem ctl56_item_rejected : sturmCheck cert_n22 ctl56_item = false := by decide +kernel

/-- The weight not that of the cell (`k := 36`: Sturm's bound would need `T_2, T_3` only, but `k ≠ n + 2`). -/
def ctl56_weight : SturmCert := { sturm_n22 with k := 36 }

theorem ctl56_weight_rejected : sturmCheck cert_n22 ctl56_weight = false := by decide +kernel

end GL2.HeckeZ.Cells.S1SturmControls
