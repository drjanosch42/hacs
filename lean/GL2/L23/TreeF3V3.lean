/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.TreeF3
import GL2.L23.TreeF3Cols
import GL2.Conventions.SymDual

/-!
# LV-2 (slot 22): `dim_{F₃} H^q(SL₂(ℤ); Sym^n F₃²)` — odd `n` in general, even `n` from four kernel ranks

`V3 n` is `Sym^n F₃²`, `g ↦ toLin' (GL2.rhoMat n (g mod 3))` ([C] `def:rhon`, `eq:rhondef`), as `V2 n` at ℓ = 2.

* **Odd `n`** (`V3_dims_odd`): `−I` acts by `(−1)^n = −1`, so `H^q = 0` for every `q` (`finrank_H_eq_zero_of_negOne`). No
  cells, no ranks: a theorem for every odd `n`.
* **Even `n`** (`V3_dims_of_ranks`): with `s = rankF3 (colsSig3 n)` (the stacked `[1 − ρ_n(S) ; 1 − ρ_n(U)]`),
  `p = rankF3 (colsPS n)` (`1 + ρ_n(S)`), `r = rankF3 (colsN3 n)` (`1 + ρ_n(U) + ρ_n(U²)`), `u = rankF3 (colsMU n)`
  (`1 − ρ_n(U)`), all mod 3: `dim H⁰ = (n+1) − s`, `dim H¹ = 2(n+1) − s − p − r`, `dim H^{q+2} = (n+1) − r − u`.
  The ranks are `rankF3` (`rankF3_eq_rank`, `GL2/L23/RankF3.lean`) of the columns of F1 (`GL2/L23/TreeF3Cols.lean`); the
  row bounds are theorems for every `n`, so a cell is four `decide +kernel` facts.
-/

noncomputable section

namespace GL2.L23.TreeF3

open CategoryTheory GL2.MP GL2.L23.RankF3 GL2.Exec Module
open scoped MatrixGroups

/-! ### The bridge: a linear map's rank is `rankF3` -/

theorem finrank_range_eq_rankF3 {V W I : Type} [AddCommGroup V] [Module (ZMod 3) V] [AddCommGroup W]
    [Module (ZMod 3) W] (L : V →ₗ[ZMod 3] W) (w : ℕ) (ι : W ≃ₗ[ZMod 3] (Fin w → ZMod 3))
    (b : Module.Basis I (ZMod 3) V) (rows : List (ℕ × ℕ)) (hrows : ∀ r ∈ rows, Disj r ∧ Low w r)
    (h : Set.range (fun i => ι (L (b i))) = vec3 w '' {r | r ∈ rows}) :
    finrank (ZMod 3) (LinearMap.range L) = rankF3 rows := by
  have hmap : Submodule.map (ι : W →ₗ[ZMod 3] (Fin w → ZMod 3)) (LinearMap.range L) =
      Submodule.span (ZMod 3) (Set.range fun i => ι (L (b i))) := by
    rw [LinearMap.range_eq_map, ← b.span_eq, Submodule.map_span, Submodule.map_span, ← Set.range_comp,
      ← Set.range_comp]
    rfl
  rw [rankF3_eq_rank w rows hrows, Matrix.rank_eq_finrank_span_row, range_toMat3, ← h, ← hmap, ι.finrank_map_eq]

lemma vec3_image_map_range (w : ℕ) (f : ℕ → ℕ × ℕ) (d : ℕ) :
    vec3 w '' {r | r ∈ (List.range d).map f} = Set.range fun j : Fin d => vec3 w (f j) := by
  ext v
  simp only [Set.mem_image, Set.mem_ofPred_eq, List.mem_map, List.mem_range, Set.mem_range]
  constructor
  · rintro ⟨r, ⟨j, hj, rfl⟩, rfl⟩; exact ⟨⟨j, hj⟩, rfl⟩
  · rintro ⟨j, rfl⟩; exact ⟨f j, ⟨j, j.2, rfl⟩, rfl⟩

/-! ### `Sym^n F₃²` -/

/-- Reduction mod 3 of an element of `SL₂(ℤ)`. -/
def mod3 (g : SL(2, ℤ)) : Mat2 (ZMod 3) := (g : Matrix (Fin 2) (Fin 2) ℤ).map (Int.castRingHom (ZMod 3))

lemma mod3_mul (g h : SL(2, ℤ)) : mod3 (g * h) = mod3 g * mod3 h := by
  simp only [mod3, Matrix.SpecialLinearGroup.coe_mul, Matrix.map_mul]

lemma mod3_one : mod3 1 = 1 := by
  simp only [mod3, Matrix.SpecialLinearGroup.coe_one]; exact Matrix.map_one _ (map_zero _) (map_one _)

/-- **`Sym^n F₃²`**, `g ↦ ρ_n(g mod 3)` in the basis `u_t = x^{n-t} y^t` ([C] `def:rhon`, `eq:rhondef`). -/
def symRep3 (n : ℕ) : Representation (ZMod 3) SL(2, ℤ) (Fin (n + 1) → ZMod 3) where
  toFun g := Matrix.toLin' (GL2.rhoMat n (mod3 g))
  map_one' := by rw [mod3_one, GL2.rhoMat_one, Matrix.toLin'_one]; rfl
  map_mul' g h := by rw [mod3_mul, GL2.rhoMat_mul, Matrix.toLin'_mul]; rfl

/-- `V₃(n) = Sym^n F₃²` as a representation of `SL₂(ℤ)`. -/
abbrev V3 (n : ℕ) : Rep.{0} (ZMod 3) SL(2, ℤ) := Rep.of (symRep3 n)

lemma V3_rho (n : ℕ) (g : SL(2, ℤ)) : (V3 n).ρ g = Matrix.toLin' (GL2.rhoMat n (mod3 g)) := rfl

instance (n : ℕ) : FiniteDimensional (ZMod 3) (V3 n) :=
  inferInstanceAs (FiniteDimensional (ZMod 3) (Fin (n + 1) → ZMod 3))

lemma finrank_V3 (n : ℕ) : finrank (ZMod 3) (V3 n) = n + 1 := Module.finrank_fin_fun (ZMod 3)

lemma two_ne_zero_zmod3 : (2 : ZMod 3) ≠ 0 := by decide

lemma mod3_negOne : mod3 (-1) = (-1 : ZMod 3) • (1 : Mat2 (ZMod 3)) := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [mod3]

lemma V3_negOne (n : ℕ) : (V3 n).ρ (-1) = (-1 : ZMod 3) ^ n • (1 : V3 n →ₗ[ZMod 3] V3 n) := by
  show Matrix.toLin' (GL2.rhoMat n (mod3 (-1))) = _
  rw [mod3_negOne, GL2.rhoMat_smul_one, map_smul, Matrix.toLin'_one]; rfl

lemma V3_negOne_even {n : ℕ} (hn : Even n) : (V3 n).ρ (-1) = 1 := by
  rw [V3_negOne, hn.neg_one_pow, one_smul]

lemma V3_negOne_odd {n : ℕ} (hn : ¬ Even n) : (V3 n).ρ (-1) = -1 := by
  rw [V3_negOne, (Nat.not_even_iff_odd.mp hn).neg_one_pow]
  refine LinearMap.ext fun x => ?_
  rw [LinearMap.smul_apply, Module.End.one_apply, LinearMap.neg_apply, Module.End.one_apply, neg_one_smul]

lemma mod3_sS : mod3 sS = mS3 := by
  rw [mod3, sS, GL2.TauChk.M22.toSL_coe mS_det]
  ext i j; fin_cases i <;> fin_cases j <;> simp [GL2.TauChk.M22.toMat, mS, mS3]

lemma mod3_sU : mod3 sU = mU3 := by
  rw [mod3, sU, GL2.TauChk.M22.toSL_coe mU_det]
  ext i j; fin_cases i <;> fin_cases j <;> simp [GL2.TauChk.M22.toMat, mU, mU3]

lemma mod3_sU_sq : mod3 (sU ^ 2) = mUU3 := by
  rw [pow_two, mod3_mul, mod3_sU]
  ext i j; fin_cases i <;> fin_cases j <;> simp [mU3, mUU3, Matrix.mul_apply, Fin.sum_univ_two]

/-! #### The columns, as vectors -/

lemma V3_rho_single (n : ℕ) (g : SL(2, ℤ)) (j : Fin (n + 1)) :
    (V3 n).ρ g (Pi.single j 1) = (GL2.rhoMat n (mod3 g)).col j := by
  rw [V3_rho, Matrix.toLin'_apply, Matrix.mulVec_single_one]

lemma vec3_e3_single (n : ℕ) (j : Fin (n + 1)) : vec3 (n + 1) (e3 j) = Pi.single j 1 := by
  funext t
  simp only [vec3, val3_e3, Pi.single_apply, Fin.ext_iff]

lemma vec3_colPS (n : ℕ) (j : Fin (n + 1)) :
    vec3 (n + 1) (colPS n j) = Pi.single j 1 + (V3 n).ρ sS (Pi.single j 1) := by
  rw [colPS, vec3_add3 (disj_e3 _) (disj_sgn3 (disj_e3 _) _), vec3_e3_single, vec3_colS, V3_rho_single, mod3_sS]

lemma vec3_colMS (n : ℕ) (j : Fin (n + 1)) :
    vec3 (n + 1) (colMS n j) = Pi.single j 1 - (V3 n).ρ sS (Pi.single j 1) := by
  rw [colMS, vec3_add3 (disj_e3 _) (disj_neg3 (disj_sgn3 (disj_e3 _) _)), vec3_neg3 (disj_sgn3 (disj_e3 _) _),
    vec3_e3_single, vec3_colS, V3_rho_single, mod3_sS, sub_eq_add_neg]

lemma vec3_colMU (n : ℕ) (j : Fin (n + 1)) :
    vec3 (n + 1) (colMU n j (qrow j)) = Pi.single j 1 - (V3 n).ρ sU (Pi.single j 1) := by
  have hj := Nat.lt_succ_iff.mp j.2
  rw [colMU, vec3_add3 (disj_e3 _) (disj_neg3 (shl3_qrow_ok n j hj).1), vec3_neg3 (shl3_qrow_ok n j hj).1,
    vec3_e3_single, vec3_colU, V3_rho_single, mod3_sU, sub_eq_add_neg]

lemma vec3_colN3 (n : ℕ) (j : Fin (n + 1)) :
    vec3 (n + 1) (colN3 n j (qrow j) (qrow (n - j))) =
      Pi.single j 1 + (V3 n).ρ sU (Pi.single j 1) + (V3 n).ρ (sU ^ 2) (Pi.single j 1) := by
  have hj := Nat.lt_succ_iff.mp j.2
  rw [colN3, vec3_add3 (disj_add3 (disj_e3 _) (shl3_qrow_ok n j hj).1) (disj_sgn3 (disj_low_qrow _).1 _),
    vec3_add3 (disj_e3 _) (shl3_qrow_ok n j hj).1, vec3_e3_single, vec3_colU, vec3_colU2, V3_rho_single,
    V3_rho_single, mod3_sU, mod3_sU_sq]

/-- `F₃^d × F₃^d ≃ F₃^{d+d}` (first factor in the low positions). -/
def ι3 (d : ℕ) : ((Fin d → ZMod 3) × (Fin d → ZMod 3)) ≃ₗ[ZMod 3] (Fin (d + d) → ZMod 3) :=
  (LinearEquiv.sumArrowLequivProdArrow (Fin d) (Fin d) (ZMod 3) (ZMod 3)).symm.trans
    (LinearEquiv.funCongrLeft (ZMod 3) (ZMod 3) finSumFinEquiv.symm)

lemma ι3_castAdd (d : ℕ) (x y : Fin d → ZMod 3) (t : Fin d) : ι3 d (x, y) (Fin.castAdd d t) = x t := by
  simp [ι3, LinearEquiv.funCongrLeft_apply]

lemma ι3_natAdd (d : ℕ) (x y : Fin d → ZMod 3) (t : Fin d) : ι3 d (x, y) (Fin.natAdd d t) = y t := by
  show (LinearEquiv.sumArrowLequivProdArrow (Fin d) (Fin d) (ZMod 3) (ZMod 3)).symm (x, y)
    (finSumFinEquiv.symm (Fin.natAdd d t)) = y t
  rw [finSumFinEquiv_symm_apply_natAdd]; rfl

lemma vec3_colSig3 (n : ℕ) (j : Fin (n + 1)) :
    vec3 ((n + 1) + (n + 1)) (colSig3 n j (qrow j)) =
      ι3 (n + 1) (vec3 (n + 1) (colMS n j), vec3 (n + 1) (colMU n j (qrow j))) := by
  have hlo := (colMS_ok n j (Nat.lt_succ_iff.mp j.2)).2
  funext i
  refine Fin.addCases (fun t => ?_) (fun t => ?_) i
  · rw [ι3_castAdd]
    exact val3_colSig3_lo n j _ hlo t t.2
  · rw [ι3_natAdd]
    exact val3_colSig3_hi n j _ hlo t

/-! #### The four ranks -/

lemma nrmA_two' {K : Type} [CommRing K] (A : Rep.{0} K SL(2, ℤ)) (t : SL(2, ℤ)) (x : A) :
    nrmA A t 2 x = x + A.ρ t x := by
  simp [nrmA, Finset.sum_range_succ]

lemma nrmA_three' {K : Type} [CommRing K] (A : Rep.{0} K SL(2, ℤ)) (t : SL(2, ℤ)) (x : A) :
    nrmA A t 3 x = x + A.ρ t x + A.ρ (t ^ 2) x := by
  simp [nrmA, Finset.sum_range_succ]

theorem finrank_range_sigma_V3 (n : ℕ) :
    finrank (ZMod 3) (LinearMap.range (sigmaK (V3 n))) = rankF3 (colsSig3 n) := by
  refine finrank_range_eq_rankF3 _ _ (show (V3 n × V3 n) ≃ₗ[ZMod 3] _ from ι3 (n + 1))
    (show Module.Basis _ (ZMod 3) (V3 n) from Pi.basisFun (ZMod 3) (Fin (n + 1))) _ (rows_colsSig3 n) ?_
  rw [colsSig3_eq, vec3_image_map_range]
  congr 1; funext j
  rw [vec3_colSig3, vec3_colMS, vec3_colMU, Pi.basisFun_apply, sigmaK_apply]

theorem finrank_range_PS_V3 (n : ℕ) :
    finrank (ZMod 3) (LinearMap.range (nrmA (V3 n) sS 2)) = rankF3 (colsPS n) := by
  refine finrank_range_eq_rankF3 _ _ (LinearEquiv.refl _ _)
    (show Module.Basis _ (ZMod 3) (V3 n) from Pi.basisFun (ZMod 3) (Fin (n + 1))) _ (rows_colsPS n) ?_
  rw [colsPS_eq, vec3_image_map_range]
  congr 1; funext j
  rw [vec3_colPS, Pi.basisFun_apply, LinearEquiv.refl_apply, nrmA_two']

theorem finrank_range_N3_V3 (n : ℕ) :
    finrank (ZMod 3) (LinearMap.range (nrmA (V3 n) sU 3)) = rankF3 (colsN3 n) := by
  refine finrank_range_eq_rankF3 _ _ (LinearEquiv.refl _ _)
    (show Module.Basis _ (ZMod 3) (V3 n) from Pi.basisFun (ZMod 3) (Fin (n + 1))) _ (rows_colsN3 n) ?_
  rw [colsN3_eq, vec3_image_map_range]
  congr 1; funext j
  rw [vec3_colN3, Pi.basisFun_apply, LinearEquiv.refl_apply, nrmA_three']

theorem finrank_range_MU_V3 (n : ℕ) :
    finrank (ZMod 3) (LinearMap.range (difA (V3 n) sU)) = rankF3 (colsMU n) := by
  refine finrank_range_eq_rankF3 _ _ (LinearEquiv.refl _ _)
    (show Module.Basis _ (ZMod 3) (V3 n) from Pi.basisFun (ZMod 3) (Fin (n + 1))) _ (rows_colsMU n) ?_
  rw [colsMU_eq, vec3_image_map_range]
  congr 1; funext j
  rw [vec3_colMU, Pi.basisFun_apply, LinearEquiv.refl_apply, difA_apply]

/-! ### The cell theorems -/

/-- **Odd `n`: `dim_{F₃} H^q(SL₂(ℤ); Sym^n F₃²) = 0` for every `q`.** -/
theorem V3_dims_odd (n : ℕ) (hn : n % 2 = 1) (q : ℕ) : finrank (ZMod 3) (groupCohomology (V3 n) q) = 0 :=
  finrank_H_eq_zero_of_negOne _ two_ne_zero_zmod3 (V3_negOne_odd (by rw [Nat.even_iff]; omega)) q

/-- **Even `n`: `dim H⁰`, `dim H¹`, `dim H^{q+2}` from four kernel ranks** (`s`: `[1 − S ; 1 − U]`, `p`: `1 + S`,
`r`: `1 + U + U²`, `u`: `1 − U`, all of `ρ_n` mod 3). -/
theorem V3_dims_of_ranks (n s p r u h0 h1 h2 : ℕ) (hn : n % 2 = 0) (hs : rankF3 (colsSig3 n) = s)
    (hp : rankF3 (colsPS n) = p) (hr : rankF3 (colsN3 n) = r) (hu : rankF3 (colsMU n) = u)
    (e0 : n + 1 = h0 + s) (e1 : 2 * (n + 1) = h1 + s + p + r) (e2 : n + 1 = h2 + r + u) :
    finrank (ZMod 3) (groupCohomology (V3 n) 0) = h0 ∧ finrank (ZMod 3) (groupCohomology (V3 n) 1) = h1 ∧
      ∀ q, finrank (ZMod 3) (groupCohomology (V3 n) (q + 2)) = h2 := by
  obtain ⟨d0, d1, d2⟩ := finrank_H_of_trivNegOne (V3 n) two_ne_zero_zmod3 (V3_negOne_even (Nat.even_iff.mpr hn))
  rw [finrank_range_sigma_V3, hs, finrank_V3] at d0 d1
  rw [finrank_range_PS_V3, hp, finrank_range_N3_V3, hr] at d1
  refine ⟨by omega, by omega, fun q => ?_⟩
  have := d2 q
  rw [finrank_range_N3_V3, hr, finrank_range_MU_V3, hu, finrank_V3] at this
  omega

end GL2.L23.TreeF3
