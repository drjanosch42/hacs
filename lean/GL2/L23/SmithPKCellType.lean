/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.SmithPKCols
import GL2.L23.SmithPKType

/-!
# LV (slot 23, follow-up R1(a)): the Smith cell theorem with the type

`cell_type0/1/2`: a kernel-decided `sig K (cellPiv l K H n q) = (s, cs)` gives, for every `t`,
`#(ℓ^t · span over ZMod ℓ^K of the cell columns of d^q) = ℓ^{Σ_{v<K} cs_v (K − (v + t))}`. These numbers determine the
module up to isomorphism, so the valuation counts `cs` are certified (`card_smul_of_sig`). The cell columns are the block
formulas of `entN_colD0x` …; their identification with `treeD0` / `treeDS` is not proved here.
-/

noncomputable section

namespace GL2.L23.SmithPK

/-- The type form of `cell_card`. -/
theorem cell_type {l K H n q s : ℕ} {cs : List ℕ} (hq : q ≤ 2)
    (cx : Ctx l K H (if q = 0 then colsD0 (2 * H) (l ^ K) n else if q = 1 then colsD1 (2 * H) (l ^ K) n
      else colsD2 (2 * H) (l ^ K) n).length)
    (h : sig K (cellPiv l K H n q) = (s, cs)) (t : ℕ) :
    Nat.card ((spanK l K H (3 * (n + 1)) (if q = 0 then colsD0 (2 * H) (l ^ K) n
      else if q = 1 then colsD1 (2 * H) (l ^ K) n else colsD2 (2 * H) (l ^ K) n)).map
        ((l : ZMod (l ^ K)) ^ t • LinearMap.id)) = l ^ ∑ v ∈ Finset.range K, cs.getD v 0 * (K - (v + t)) := by
  rw [cellPiv_eq _ _ _ _ _ hq] at h
  exact card_smul_of_sig cx (fun r hr i _ => colsD_reduced cx.N_pos (le_trans cx.N_lt.le cx.H_le_F) n q r hr i) h t

theorem cell_type0 {l K H n s : ℕ} {cs : List ℕ} (cx : Ctx l K H (2 * (n + 1)))
    (h : sig K (cellPiv l K H n 0) = (s, cs)) (t : ℕ) :
    Nat.card ((spanK l K H (3 * (n + 1)) (colsD0 (2 * H) (l ^ K) n)).map ((l : ZMod (l ^ K)) ^ t • LinearMap.id)) =
      l ^ ∑ v ∈ Finset.range K, cs.getD v 0 * (K - (v + t)) :=
  cell_type (q := 0) (by norm_num) (by rw [if_pos rfl, length_colsD0]; exact cx) h t

theorem cell_type1 {l K H n s : ℕ} {cs : List ℕ} (cx : Ctx l K H (2 * (n + 1)))
    (h : sig K (cellPiv l K H n 1) = (s, cs)) (t : ℕ) :
    Nat.card ((spanK l K H (3 * (n + 1)) (colsD1 (2 * H) (l ^ K) n)).map ((l : ZMod (l ^ K)) ^ t • LinearMap.id)) =
      l ^ ∑ v ∈ Finset.range K, cs.getD v 0 * (K - (v + t)) :=
  cell_type (q := 1) (by norm_num) (by rw [if_neg (by norm_num), if_pos rfl, length_colsD1]; exact cx) h t

theorem cell_type2 {l K H n s : ℕ} {cs : List ℕ} (cx : Ctx l K H (3 * (n + 1)))
    (h : sig K (cellPiv l K H n 2) = (s, cs)) (t : ℕ) :
    Nat.card ((spanK l K H (3 * (n + 1)) (colsD2 (2 * H) (l ^ K) n)).map ((l : ZMod (l ^ K)) ^ t • LinearMap.id)) =
      l ^ ∑ v ∈ Finset.range K, cs.getD v 0 * (K - (v + t)) :=
  cell_type (q := 2) (by norm_num) (by rw [if_neg (by norm_num), if_neg (by norm_num), length_colsD2]; exact cx) h t

end GL2.L23.SmithPK

end
