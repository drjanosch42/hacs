/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.SmithPKCols
import GL2.L23.TreeF3V3

/-!
# LV (slot 23, follow-up of `LV_audit.md` M-3): the cell columns are the columns of `treeD0` / `treeDS`

`VN N n = Sym^n (ℤ/N)²` as a representation of `SL₂(ℤ)` on `Fin (n+1) → ZMod N`, `g ↦ ρ_n(g mod N)` in the basis
`u_t = x^{n−t} y^t` ([C] `def:rhon`, `eq:rhondef`; the pattern of `V2`, `V3`). `blk` reads a cochain
`((x, y), z) ∈ (A × A) × A` as one vector indexed by `t < 3(n+1)` (blocks at `0`, `n+1`, `2(n+1)`).

**`entN_colD0x_tree` … `entN_colD2z_tree`**: for every `t < 3(n+1)`, entry `t` of the packed cell column equals entry `t`
of `blk` of LR-Y's `treeD0` (`d⁰`), `treeDS 0` (`d¹`) and `treeDS 1` (`d²`) on `VN N n` applied to the basis vector
(`n` even for `d¹`, `d²`, where `−I` acts trivially). So the Smith cells of `GL2/L23/SmithPKCols.lean` /
`SmithPKCellType.lean` are about the actual tree differentials of `Sym^n (ℤ/N)²`. (The identification `VN N n ≅ VK
(ZMod N) n` with LG-1's `Sym^n` on homogeneous polynomials is F-LG R4, not done here.)
-/

noncomputable section

namespace GL2.L23.SmithPK

open CategoryTheory GL2.MP GL2.Exec Module GL2.L23.RankF3
open scoped MatrixGroups

variable (N : ℕ)

/-- Reduction mod `N` of an element of `SL₂(ℤ)`. -/
def modN (g : SL(2, ℤ)) : GL2.Mat2 (ZMod N) := (g : Matrix (Fin 2) (Fin 2) ℤ).map (Int.castRingHom (ZMod N))

lemma modN_mul (g h : SL(2, ℤ)) : modN N (g * h) = modN N g * modN N h := by
  simp only [modN, Matrix.SpecialLinearGroup.coe_mul, Matrix.map_mul]

lemma modN_one : modN N 1 = 1 := by
  simp only [modN, Matrix.SpecialLinearGroup.coe_one]; exact Matrix.map_one _ (map_zero _) (map_one _)

/-- **`Sym^n (ℤ/N)²`**, `g ↦ ρ_n(g mod N)` in the basis `u_t`. -/
def symRepN (n : ℕ) : Representation (ZMod N) SL(2, ℤ) (Fin (n + 1) → ZMod N) where
  toFun g := Matrix.toLin' (GL2.rhoMat n (modN N g))
  map_one' := by rw [modN_one, GL2.rhoMat_one, Matrix.toLin'_one]; rfl
  map_mul' g h := by rw [modN_mul, GL2.rhoMat_mul, Matrix.toLin'_mul]; rfl

/-- `VN N n = Sym^n (ℤ/N)²` in `Rep (ZMod N) SL₂(ℤ)`. -/
abbrev VN (n : ℕ) : Rep.{0} (ZMod N) SL(2, ℤ) := Rep.of (symRepN N n)

variable {N}

lemma VN_rho (n : ℕ) (g : SL(2, ℤ)) : (VN N n).ρ g = Matrix.toLin' (GL2.rhoMat n (modN N g)) := rfl

lemma modN_negOne : modN N (-1) = (-1 : ZMod N) • (1 : GL2.Mat2 (ZMod N)) := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [modN]

lemma VN_negOne_even {n : ℕ} (hn : Even n) : (VN N n).ρ (-1) = 1 := by
  show Matrix.toLin' (GL2.rhoMat n (modN N (-1))) = _
  rw [modN_negOne, GL2.rhoMat_smul_one, hn.neg_one_pow, one_smul, Matrix.toLin'_one]; rfl

lemma modN_sS : modN N sS = mSN := by
  rw [modN, sS, GL2.TauChk.M22.toSL_coe mS_det]
  ext i j; fin_cases i <;> fin_cases j <;> simp [GL2.TauChk.M22.toMat, mS, mSN]

lemma modN_sU : modN N sU = mUN := by
  rw [modN, sU, GL2.TauChk.M22.toSL_coe mU_det]
  ext i j; fin_cases i <;> fin_cases j <;> simp [GL2.TauChk.M22.toMat, mU, mUN]

lemma modN_sU_sq : modN N (sU ^ 2) = mUUN := by
  rw [pow_two, modN_mul, modN_sU]
  ext i j; fin_cases i <;> fin_cases j <;> simp [mUN, mUUN, Matrix.mul_apply, Fin.sum_univ_two]

lemma VN_rho_single (n : ℕ) (g : SL(2, ℤ)) (j : Fin (n + 1)) :
    (VN N n).ρ g (Pi.single j 1) = (GL2.rhoMat n (modN N g)).col j := by
  rw [VN_rho, Matrix.toLin'_apply, Matrix.mulVec_single_one]

/-! ### Reading cochains as one vector -/

/-- A vector of `Fin (n+1) → R` as a function on `ℕ` (zero beyond `n`). -/
def extv {R : Type} [Zero R] (n : ℕ) (v : Fin (n + 1) → R) (t : ℕ) : R := if h : t < n + 1 then v ⟨t, h⟩ else 0

/-- The cochain `((x, y), z)` as one vector, blocks at `0`, `n + 1`, `2(n + 1)`. -/
def blk (n : ℕ) (c : ((Fin (n + 1) → ZMod N) × (Fin (n + 1) → ZMod N)) × (Fin (n + 1) → ZMod N)) (t : ℕ) : ZMod N :=
  extv n c.1.1 t + shv (extv n c.1.2) (n + 1) t + shv (extv n c.2) (2 * (n + 1)) t

lemma extv_add (n : ℕ) (a b : Fin (n + 1) → ZMod N) (t : ℕ) : extv n (a + b) t = extv n a t + extv n b t := by
  unfold extv; split_ifs <;> simp

lemma extv_sub (n : ℕ) (a b : Fin (n + 1) → ZMod N) (t : ℕ) : extv n (a - b) t = extv n a t - extv n b t := by
  unfold extv; split_ifs <;> simp

lemma extv_smul (n : ℕ) (c : ZMod N) (a : Fin (n + 1) → ZMod N) (t : ℕ) : extv n (c • a) t = c * extv n a t := by
  unfold extv; split_ifs <;> simp

lemma extv_neg (n : ℕ) (a : Fin (n + 1) → ZMod N) (t : ℕ) : extv n (-a) t = -extv n a t := by
  unfold extv; split_ifs <;> simp

lemma extv_nsmul (n k : ℕ) (a : Fin (n + 1) → ZMod N) (t : ℕ) : extv n (k • a) t = k * extv n a t := by
  unfold extv; split_ifs <;> simp

lemma extv_zero (n t : ℕ) : extv n (0 : Fin (n + 1) → ZMod N) t = 0 := by
  unfold extv; split_ifs <;> simp

lemma extv_single (n : ℕ) (j : Fin (n + 1)) (t : ℕ) : extv n (Pi.single j (1 : ZMod N)) t = dl j t := by
  unfold extv dl
  split_ifs with h1 h2 h2
  · simp [h2]
  · rw [Pi.single_apply, if_neg (fun h => h2 (by rw [← h])), ]
  · exact absurd (h2 ▸ j.2) h1
  · rfl

lemma extv_rho (n : ℕ) (a b c d : ZMod N) (j : Fin (n + 1)) (t : ℕ) :
    extv n ((GL2.rhoMat n !![a, b; c, d]).col j) t = rhoEntry n a b c d t j := by
  unfold extv
  split_ifs with h
  · rw [Matrix.col_apply, rhoMat_apply']; simp
  · exact (rhoEntry_eq_zero_of_lt n a b c d (Nat.lt_succ_iff.mp j.2) (by omega)).symm

lemma extv_rhoS (n : ℕ) (j : Fin (n + 1)) (t : ℕ) :
    extv n ((VN N n).ρ sS (Pi.single j 1)) t = rhoEntry n (0 : ZMod N) (-1) 1 0 t j := by
  rw [VN_rho_single, modN_sS, mSN, extv_rho]

lemma extv_rhoU (n : ℕ) (j : Fin (n + 1)) (t : ℕ) :
    extv n ((VN N n).ρ sU (Pi.single j 1)) t = rhoEntry n (0 : ZMod N) (-1) 1 1 t j := by
  rw [VN_rho_single, modN_sU, mUN, extv_rho]

lemma extv_rhoUU (n : ℕ) (j : Fin (n + 1)) (t : ℕ) :
    extv n ((VN N n).ρ (sU ^ 2) (Pi.single j 1)) t = rhoEntry n (-1 : ZMod N) (-1) 1 0 t j := by
  rw [VN_rho_single, modN_sU_sq, mUUN, extv_rho]

lemma extv_rhoS' (n : ℕ) (j : Fin (n + 1)) (t : ℕ) :
    extv n (symRepN N n sS (Pi.single j 1)) t = rhoEntry n (0 : ZMod N) (-1) 1 0 t j := extv_rhoS n j t

lemma extv_rhoU' (n : ℕ) (j : Fin (n + 1)) (t : ℕ) :
    extv n (symRepN N n sU (Pi.single j 1)) t = rhoEntry n (0 : ZMod N) (-1) 1 1 t j := extv_rhoU n j t

lemma extv_rhoUU' (n : ℕ) (j : Fin (n + 1)) (t : ℕ) :
    extv n (symRepN N n (sU ^ 2) (Pi.single j 1)) t = rhoEntry n (-1 : ZMod N) (-1) 1 0 t j := extv_rhoUU n j t

lemma shv_extv_zero (n k t : ℕ) : shv (extv n (0 : Fin (n + 1) → ZMod N)) k t = 0 := by
  unfold shv; split_ifs <;> simp [extv_zero]

lemma shv_extv_single (n : ℕ) (j : Fin (n + 1)) (k t : ℕ) :
    shv (extv n (Pi.single j (1 : ZMod N))) k t = shv (dl j) k t := by
  unfold shv; split_ifs <;> simp [extv_single]

/-! ### The tree differentials on the basis vectors -/

section Tree

variable {n : ℕ}

lemma rho_negOne_mul_VN (hn : Even n) (g : SL(2, ℤ)) (x : VN N n) : (VN N n).ρ (-1 * g) x = (VN N n).ρ g x := by
  rw [map_mul, Module.End.mul_apply, VN_negOne_even hn]; rfl

lemma rho_negOne_VN (hn : Even n) (x : VN N n) : (VN N n).ρ (-1) x = x := by
  rw [VN_negOne_even hn]; rfl

/-- `d⁰ (u_j, 0) = ((1 − S)u_j, 0, u_j)`. -/
lemma treeD0_x (j : Fin (n + 1)) :
    treeD0 (VN N n) (Pi.single j 1, 0) = ((Pi.single j 1 - (VN N n).ρ sS (Pi.single j 1), 0), Pi.single j 1) := by
  simp [treeD0, vtxA, phiA]; rfl

/-- `d⁰ (0, u_j) = (0, (1 − U)u_j, −u_j)`. -/
lemma treeD0_y (j : Fin (n + 1)) :
    treeD0 (VN N n) (0, Pi.single j 1) = ((0, Pi.single j 1 - (VN N n).ρ sU (Pi.single j 1)), -Pi.single j 1) := by
  simp [treeD0, vtxA, phiA]; rfl

lemma treeDS_apply' {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (m : ℕ) (x y z : A) :
    treeDS A m ((x, y), z) = (vtxA A (m + 1) (x, y), phiA A (m + 1) (x, y) - edgA A m z) := rfl

lemma vtxA_one {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (x y : A) :
    vtxA A 1 (x, y) = (nrmA A sS 4 x, nrmA A sU 6 y) := by
  simp [vtxA]

lemma vtxA_two {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (x y : A) :
    vtxA A 2 (x, y) = (difA A sS x, difA A sU y) := by
  simp [vtxA]

lemma phiA_one {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (x y : A) :
    phiA A 1 (x, y) = nrmA A sS 2 x - nrmA A sU 3 y := by
  simp [phiA]

lemma phiA_two {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (x y : A) : phiA A 2 (x, y) = x - y := by
  simp [phiA]

lemma edgA_zero' {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (z : A) : edgA A 0 z = z - A.ρ (-1) z := by
  simp [edgA, difA_apply]

lemma edgA_one' {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (z : A) : edgA A 1 z = z + A.ρ (-1) z := by
  simp [edgA, nrmA_apply, Finset.sum_range_succ]

lemma nrmS4_VN (hn : Even n) (x : VN N n) : nrmA (VN N n) sS 4 x = 2 • (x + (VN N n).ρ sS x) := by
  rw [nrmA_apply]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one, Module.End.one_apply]
  rw [sS_sq, sS_pow3, rho_negOne_mul_VN hn, rho_negOne_VN hn, two_smul]; abel

lemma nrmS2_VN (x : VN N n) : nrmA (VN N n) sS 2 x = x + (VN N n).ρ sS x := by
  rw [nrmA_apply]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one, Module.End.one_apply]

lemma nrmU3_VN (x : VN N n) : nrmA (VN N n) sU 3 x = x + (VN N n).ρ sU x + (VN N n).ρ (sU ^ 2) x := by
  rw [nrmA_apply]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one, Module.End.one_apply]

lemma nrmU6_VN (hn : Even n) (x : VN N n) :
    nrmA (VN N n) sU 6 x = 2 • (x + (VN N n).ρ sU x + (VN N n).ρ (sU ^ 2) x) := by
  rw [nrmA_apply]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one, Module.End.one_apply]
  rw [sU_cube, sU_pow4, sU_pow5, rho_negOne_mul_VN hn, rho_negOne_mul_VN hn, rho_negOne_VN hn, two_smul]; abel

/-- `d¹ ((u_j, 0), 0) = (2(1 + S)u_j, 0, (1 + S)u_j)` (`n` even). -/
lemma treeDS0_x (hn : Even n) (j : Fin (n + 1)) :
    treeDS (VN N n) 0 ((Pi.single j 1, 0), 0) =
      ((2 • (Pi.single j 1 + (VN N n).ρ sS (Pi.single j 1)), 0), Pi.single j 1 + (VN N n).ρ sS (Pi.single j 1)) := by
  rw [treeDS_apply', vtxA_one, phiA_one, nrmS4_VN hn, nrmS2_VN, map_zero, map_zero, map_zero, sub_zero, sub_zero]

/-- `d¹ ((0, u_j), 0) = (0, 2N₃u_j, −N₃u_j)` (`n` even). -/
lemma treeDS0_y (hn : Even n) (j : Fin (n + 1)) :
    treeDS (VN N n) 0 ((0, Pi.single j 1), 0) =
      ((0, 2 • (Pi.single j 1 + (VN N n).ρ sU (Pi.single j 1) + (VN N n).ρ (sU ^ 2) (Pi.single j 1))),
        -(Pi.single j 1 + (VN N n).ρ sU (Pi.single j 1) + (VN N n).ρ (sU ^ 2) (Pi.single j 1))) := by
  rw [treeDS_apply', vtxA_one, phiA_one, nrmU6_VN hn, nrmU3_VN, map_zero, map_zero, map_zero, zero_sub, sub_zero]

/-- `d² ((u_j, 0), 0) = ((1 − S)u_j, 0, u_j)` (`n` even: `vtxA 2 = vtxA 0`). -/
lemma treeDS1_x (j : Fin (n + 1)) :
    treeDS (VN N n) 1 ((Pi.single j 1, 0), 0) =
      ((Pi.single j 1 - (VN N n).ρ sS (Pi.single j 1), 0), Pi.single j 1) := by
  rw [treeDS_apply', vtxA_two, phiA_two, map_zero, map_zero, sub_zero, sub_zero]; rfl

/-- `d² ((0, u_j), 0) = (0, (1 − U)u_j, −u_j)`. -/
lemma treeDS1_y (j : Fin (n + 1)) :
    treeDS (VN N n) 1 ((0, Pi.single j 1), 0) =
      ((0, Pi.single j 1 - (VN N n).ρ sU (Pi.single j 1)), -Pi.single j 1) := by
  rw [treeDS_apply', vtxA_two, phiA_two, map_zero, map_zero, zero_sub, sub_zero]; rfl

/-- `d² ((0, 0), u_j) = (0, 0, −2u_j)` (`n` even). -/
lemma treeDS1_z (hn : Even n) (j : Fin (n + 1)) :
    treeDS (VN N n) 1 ((0, 0), Pi.single j 1) = ((0, 0), -(2 • Pi.single j 1)) := by
  rw [treeDS_apply', vtxA_two, phiA_two, edgA_one', rho_negOne_VN hn, map_zero, map_zero, sub_zero, zero_sub, two_smul]

end Tree

/-! ### The cell columns are the columns of the tree differentials -/

section Entries

variable {F n : ℕ} [NeZero N] (hx : Fits F N)
include hx

/-- **`d⁰`, column `x = u_j`**: entry `t < 3(n+1)` of the packed column is entry `t` of `treeD0 (u_j, 0)`. -/
theorem entN_colD0x_tree (j : Fin (n + 1)) {t : ℕ} (ht : t < 3 * (n + 1)) :
    entN F N (colD0x F N n j) t = blk n (treeD0 (VN N n) (Pi.single j 1, 0)) t := by
  rw [entN_colD0x hx (Nat.lt_succ_iff.mp j.2), if_pos ht, treeD0_x]
  simp only [blk, shv]
  split_ifs <;> simp only [shv_extv_zero, extv_add, extv_sub, extv_neg, extv_nsmul, extv_zero, extv_single, extv_rhoS, extv_rhoS', mul_zero, zero_add, add_zero, Nat.cast_ofNat] <;> ring

/-- **`d⁰`, column `y = u_j`**. -/
theorem entN_colD0y_tree (j : Fin (n + 1)) {t : ℕ} (ht : t < 3 * (n + 1)) :
    entN F N (colD0y F N n j) t = blk n (treeD0 (VN N n) (0, Pi.single j 1)) t := by
  rw [entN_colD0y hx (Nat.lt_succ_iff.mp j.2), if_pos ht, treeD0_y]
  simp only [blk, shv]
  split_ifs <;> simp only [shv_extv_zero, extv_add, extv_sub, extv_neg, extv_nsmul, extv_zero, extv_single, extv_rhoU, extv_rhoU', mul_zero, zero_add, add_zero, Nat.cast_ofNat] <;> ring

/-- **`d¹`, column `x = u_j`** (`n` even). -/
theorem entN_colD1x_tree (hn : Even n) (j : Fin (n + 1)) {t : ℕ} (ht : t < 3 * (n + 1)) :
    entN F N (colD1x F N n j) t = blk n (treeDS (VN N n) 0 ((Pi.single j 1, 0), 0)) t := by
  rw [entN_colD1x hx (Nat.lt_succ_iff.mp j.2), if_pos ht, treeDS0_x hn]
  simp only [blk, shv]
  split_ifs <;> simp only [shv_extv_zero, extv_add, extv_sub, extv_neg, extv_nsmul, extv_zero, extv_single, extv_rhoS, extv_rhoS', mul_zero, zero_add, add_zero, Nat.cast_ofNat] <;> ring

/-- **`d¹`, column `y = u_j`** (`n` even). -/
theorem entN_colD1y_tree (hn : Even n) (j : Fin (n + 1)) {t : ℕ} (ht : t < 3 * (n + 1)) :
    entN F N (colD1y F N n j) t = blk n (treeDS (VN N n) 0 ((0, Pi.single j 1), 0)) t := by
  rw [entN_colD1y hx, if_pos ht, treeDS0_y hn]
  simp only [blk, shv, entN_colN3K hx (Nat.lt_succ_iff.mp j.2)]
  split_ifs <;> simp only [shv_extv_zero, extv_add, extv_sub, extv_neg, extv_nsmul, extv_zero, extv_single, extv_rhoU, extv_rhoU', extv_rhoUU, extv_rhoUU', mul_zero, zero_add, add_zero, Nat.cast_ofNat] <;> ring

/-- **`d²`, column `x = u_j`** (the packed column is `d⁰`'s). -/
theorem entN_colD2x_tree (j : Fin (n + 1)) {t : ℕ} (ht : t < 3 * (n + 1)) :
    entN F N (colD0x F N n j) t = blk n (treeDS (VN N n) 1 ((Pi.single j 1, 0), 0)) t := by
  rw [entN_colD0x_tree hx j ht, treeD0_x, treeDS1_x]

/-- **`d²`, column `y = u_j`**. -/
theorem entN_colD2y_tree (j : Fin (n + 1)) {t : ℕ} (ht : t < 3 * (n + 1)) :
    entN F N (colD0y F N n j) t = blk n (treeDS (VN N n) 1 ((0, Pi.single j 1), 0)) t := by
  rw [entN_colD0y_tree hx j ht, treeD0_y, treeDS1_y]

/-- **`d²`, column `z = u_j`** (`n` even). -/
theorem entN_colD2z_tree (hn : Even n) (j : Fin (n + 1)) {t : ℕ} (ht : t < 3 * (n + 1)) :
    entN F N (colD2z F N n j) t = blk n (treeDS (VN N n) 1 ((0, 0), Pi.single j 1)) t := by
  rw [entN_colD2z hx, if_pos ht, treeDS1_z hn]
  simp only [blk, shv]
  split_ifs <;> simp only [shv_extv_zero, extv_add, extv_sub, extv_neg, extv_nsmul, extv_zero, extv_single, mul_zero, zero_add, add_zero, Nat.cast_ofNat] <;> ring

end Entries

end GL2.L23.SmithPK

end
