/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.TreeF2Cols
import GL2.MP.TreeCochains

/-!
# LV-1 (slot 22): `dim_{F₂} H^q(SL₂(ℤ); Sym^n F₂²)` in every degree from two ranks

`L23_master_plan.md` §4.2–4.3 (LV row), on LR-Y's explicit complex (`GL2.MP.treeCoch`, `model7TreeCochCoh`:
`H^q(SL₂(ℤ); A) ≅ H^q(A² → A³ → A³ → ⋯)` for every ring, module and `q`, `−I` included).

**Char 2 with `−I` trivial** (`TrivNegOne`: `A.ρ (−1) = 1` over `ZMod 2`). Then `S² = U³ = 1` on `A`, so `N_S = N_U = 0`
(`nrmA_sS_four`, `nrmA_sU_six`) and the edge block of every degree vanishes (`edgA_eq_zero`). Hence
* `d⁰ = treeD0`, `d^{2j+2} = treeD0 ∘ fst` (`treeDS_odd`): rank `dim A + rank σ`, `σ = (1 − S, 1 − U) : A → A²`
  (`finrank_range_treeD0`, through `ker treeD0 ≃ ker σ`, `kerD0Equiv`);
* `d^{2j+1} = inr ∘ ψ ∘ fst` (`treeDS_even`), `ψ = φ*_1 = (1 + S)∘fst − (1 + U + U²)∘snd : A² → A`.

**`finrank_H_zero`**: `dim H⁰ + rank σ = dim A`; **`finrank_H_succ`**: `dim H^{q+1} + rank ψ + rank σ = 2 dim A`, every `q`.

**`V2 n`** (`Sym^n F₂²`, `g ↦ toLin' (GL2.rhoMat n (g mod 2))`, [C] `def:rhon`): `rank ψ = rankF2 (colsA n)`, `rank σ =
rankF2 (colsSig n)` (`finrank_range_psi_V2`, `finrank_range_sigma_V2`) by `TreeF2Cols`' F1 lemmas; so the **cell theorem**
`V2_dims` turns two kernel-decided ranks (and two kernel-decided width bounds) into `dim H^q` for every `q`.
-/

noncomputable section

namespace GL2.L23.TreeF2

open CategoryTheory GL2.MP GL2.L23.RankF2 GL2.Exec Module
open scoped MatrixGroups

/-! ### Homology dimension of a short complex of vector spaces -/

section ShortCx

variable {K : Type} [Field K]

lemma finrank_homology_add (S : ShortComplex (ModuleCat.{0} K)) [FiniteDimensional K S.X₂] :
    finrank K S.homology + finrank K (LinearMap.range S.f.hom) + finrank K (LinearMap.range S.g.hom) =
      finrank K S.X₂ := by
  rw [S.moduleCatHomologyIso.toLinearEquiv.finrank_eq]
  show finrank K (LinearMap.ker S.g.hom ⧸ LinearMap.range S.moduleCatToCycles) + _ + _ = _
  have h1 := Submodule.finrank_quotient_add_finrank (LinearMap.range S.moduleCatToCycles)
  have h2 : finrank K (LinearMap.range S.moduleCatToCycles) = finrank K (LinearMap.range S.f.hom) := by
    have : LinearMap.range S.f.hom =
        (LinearMap.range S.moduleCatToCycles).map (LinearMap.ker S.g.hom).subtype := by
      rw [← LinearMap.range_comp]; rfl
    rw [this]
    exact (Submodule.equivMapOfInjective _ (Submodule.injective_subtype _) _).finrank_eq
  have h3 := LinearMap.finrank_range_add_finrank_ker S.g.hom
  omega

end ShortCx

/-! ### The tree complex in characteristic 2 with `−I` trivial -/

section Char2

variable (A : Rep.{0} (ZMod 2) SL(2, ℤ))

/-- `−I` acts trivially. -/
def TrivNegOne : Prop := A.ρ (-1) = 1

lemma add_self_end {M : Type} [AddCommGroup M] [Module (ZMod 2) M] (f : M →ₗ[ZMod 2] M) : f + f = 0 := by
  rw [← two_smul (ZMod 2) f, show (2 : ZMod 2) = 0 from rfl, zero_smul]

lemma sub_eq_add_zmod2 {M : Type} [AddCommGroup M] [Module (ZMod 2) M] (x y : M) : x - y = x + y := by
  rw [sub_eq_add_neg, ← neg_one_smul (ZMod 2) y, show (-1 : ZMod 2) = 1 from rfl, one_smul]

variable {A}

lemma rho_sS_sq (hI : TrivNegOne A) : A.ρ (sS ^ 2) = 1 := by rw [sS_sq]; exact hI

lemma rho_sS_cube (hI : TrivNegOne A) : A.ρ (sS ^ 3) = A.ρ sS := by
  rw [pow_succ, map_mul, rho_sS_sq hI, one_mul]

lemma rho_sU_cube (hI : TrivNegOne A) : A.ρ (sU ^ 3) = 1 := by rw [sU_cube]; exact hI

lemma rho_sU_four (hI : TrivNegOne A) : A.ρ (sU ^ 4) = A.ρ sU := by
  rw [pow_succ, map_mul, rho_sU_cube hI, one_mul]

lemma rho_sU_five (hI : TrivNegOne A) : A.ρ (sU ^ 5) = A.ρ (sU ^ 2) := by
  rw [show 5 = 3 + 2 from rfl, pow_add, map_mul, rho_sU_cube hI, one_mul]

lemma nrmA_sS_four (hI : TrivNegOne A) : nrmA A sS 4 = 0 := by
  simp only [nrmA, Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  rw [rho_sS_sq hI, rho_sS_cube hI, pow_zero, pow_one, map_one]
  rw [show (1 : A →ₗ[ZMod 2] A) + A.ρ sS + 1 + A.ρ sS = (1 + A.ρ sS) + (1 + A.ρ sS) by abel, add_self_end]

lemma nrmA_sU_six (hI : TrivNegOne A) : nrmA A sU 6 = 0 := by
  simp only [nrmA, Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  rw [rho_sU_cube hI, rho_sU_four hI, rho_sU_five hI, pow_zero, pow_one, map_one]
  rw [show (1 : A →ₗ[ZMod 2] A) + A.ρ sU + A.ρ (sU ^ 2) + 1 + A.ρ sU + A.ρ (sU ^ 2) =
    (1 + A.ρ sU + A.ρ (sU ^ 2)) + (1 + A.ρ sU + A.ρ (sU ^ 2)) by abel, add_self_end]

lemma edgA_eq_zero (hI : TrivNegOne A) (m : ℕ) : edgA A m = 0 := by
  unfold edgA
  split_ifs
  · simp only [difA, show A.ρ (-1) = 1 from hI]; exact sub_self _
  · simp only [nrmA, Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one,
      show A.ρ (-1) = 1 from hI]
    exact add_self_end _

lemma vtxA_odd (hI : TrivNegOne A) {m : ℕ} (hm : ¬ Even m) : vtxA A m = 0 := by
  rw [vtxA, if_neg hm, nrmA_sS_four hI, nrmA_sU_six hI]
  ext <;> simp

/-- `σ = (1 − S, 1 − U) : A → A × A`; `ker σ = A^Γ` once `−I` is trivial. -/
def sigmaA : A →ₗ[ZMod 2] A × A := (difA A sS).prod (difA A sU)

/-- `ψ = φ*_1 : (x, y) ↦ (1 + S)x − (1 + U + U²)y`. -/
abbrev psiA : A × A →ₗ[ZMod 2] A := phiA A 1

lemma phiA_odd_eq {m : ℕ} (hm : ¬ Even m) : phiA A m = psiA (A := A) := by
  unfold psiA phiA; rw [if_neg hm, if_neg (by decide)]

lemma vtxA_even_eq {m : ℕ} (hm : Even m) : vtxA A m = vtxA A 0 := by
  unfold vtxA; rw [if_pos hm, if_pos (by decide : Even 0)]

lemma phiA_even_eq {m : ℕ} (hm : Even m) : phiA A m = phiA A 0 := by
  unfold phiA; rw [if_pos hm, if_pos (by decide : Even 0)]

lemma treeDS_even (hI : TrivNegOne A) {m : ℕ} (hm : Even m) :
    treeDS A m = LinearMap.inr (ZMod 2) (A × A) A ∘ₗ psiA (A := A) ∘ₗ LinearMap.fst (ZMod 2) (A × A) A := by
  have hm1 : ¬ Even (m + 1) := by rw [Nat.even_add_one]; exact not_not.mpr hm
  refine LinearMap.ext fun v => ?_
  rw [treeDS, vtxA_odd hI hm1, edgA_eq_zero hI, phiA_odd_eq hm1]
  simp

lemma treeDS_odd (hI : TrivNegOne A) {m : ℕ} (hm : ¬ Even m) :
    treeDS A m = treeD0 A ∘ₗ LinearMap.fst (ZMod 2) (A × A) A := by
  have hm1 : Even (m + 1) := by rw [Nat.even_add_one]; exact hm
  refine LinearMap.ext fun v => ?_
  rw [treeDS, vtxA_even_eq hm1, phiA_even_eq hm1, edgA_eq_zero hI]
  simp [treeD0]

lemma treeD0_apply (x y : A) :
    treeD0 A (x, y) = ((x - A.ρ sS x, y - A.ρ sU y), x - y) := by
  simp [treeD0, vtxA, phiA, difA]

lemma sigmaA_apply (x : A) : sigmaA (A := A) x = (x - A.ρ sS x, x - A.ρ sU x) := rfl

/-- **`ker treeD0 ≃ ker σ`**, `(x, x) ↤ x` (any `A`; `x − y = 0` forces `y = x`). -/
def kerD0Equiv : LinearMap.ker (sigmaA (A := A)) ≃ₗ[ZMod 2] LinearMap.ker (treeD0 A) where
  toFun x := ⟨(x.1, x.1), by
    have h := x.2
    rw [LinearMap.mem_ker, sigmaA_apply, Prod.mk_eq_zero] at h
    rw [LinearMap.mem_ker, treeD0_apply, h.1, h.2, sub_self]; rfl⟩
  invFun v := ⟨v.1.1, by
    have h := v.2
    rw [LinearMap.mem_ker, show v.1 = (v.1.1, v.1.2) from rfl, treeD0_apply, Prod.mk_eq_zero, Prod.mk_eq_zero] at h
    rw [LinearMap.mem_ker, sigmaA_apply, Prod.mk_eq_zero]
    refine ⟨h.1.1, ?_⟩
    rw [sub_eq_zero.mp h.2]; exact h.1.2⟩
  map_add' x y := rfl
  map_smul' c x := rfl
  left_inv x := rfl
  right_inv v := by
    have h := v.2
    rw [LinearMap.mem_ker, show v.1 = (v.1.1, v.1.2) from rfl, treeD0_apply, Prod.mk_eq_zero, Prod.mk_eq_zero] at h
    apply Subtype.ext
    show (v.1.1, v.1.1) = v.1
    exact Prod.ext rfl (sub_eq_zero.mp h.2)

lemma finrank_range_comp_fst {M N P : Type} [AddCommGroup M] [Module (ZMod 2) M] [AddCommGroup N] [Module (ZMod 2) N]
    [AddCommGroup P] [Module (ZMod 2) P] (f : M →ₗ[ZMod 2] P) :
    LinearMap.range (f ∘ₗ LinearMap.fst (ZMod 2) M N) = LinearMap.range f :=
  LinearMap.range_comp_of_range_eq_top _ (LinearMap.range_eq_top.mpr LinearMap.fst_surjective)

lemma finrank_range_treeDS (hI : TrivNegOne A) (m : ℕ) :
    finrank (ZMod 2) (LinearMap.range (treeDS A m)) = if Even m then finrank (ZMod 2) (LinearMap.range (psiA (A := A)))
      else finrank (ZMod 2) (LinearMap.range (treeD0 A)) := by
  split_ifs with hm
  · rw [treeDS_even hI hm, LinearMap.range_comp, finrank_range_comp_fst]
    exact (Submodule.equivMapOfInjective _ LinearMap.inr_injective _).finrank_eq.symm
  · rw [treeDS_odd hI hm, finrank_range_comp_fst]

lemma finrank_coh (q : ℕ) : finrank (ZMod 2) (groupCohomology A q) = finrank (ZMod 2) ((treeCoch A).homology q) :=
  (model7TreeCochCoh A q).toLinearEquiv.finrank_eq

variable [FiniteDimensional (ZMod 2) A]

lemma finrank_range_treeD0 :
    finrank (ZMod 2) (LinearMap.range (treeD0 A)) =
      finrank (ZMod 2) A + finrank (ZMod 2) (LinearMap.range (sigmaA (A := A))) := by
  have h1 := LinearMap.finrank_range_add_finrank_ker (treeD0 A)
  have h2 := LinearMap.finrank_range_add_finrank_ker (sigmaA (A := A))
  rw [← (kerD0Equiv (A := A)).finrank_eq, finrank_prod] at h1
  omega

include A in
lemma finiteDimensional_X (n : ℕ) : FiniteDimensional (ZMod 2) ((treeCoch A).X n) := by
  cases n
  · show FiniteDimensional (ZMod 2) (A × A); infer_instance
  · show FiniteDimensional (ZMod 2) ((A × A) × A); infer_instance

omit [FiniteDimensional (ZMod 2) A] in
lemma finrank_range_tcD (m : ℕ) : finrank (ZMod 2) (LinearMap.range ((treeCoch A).d m (m + 1)).hom) =
    match m with
    | 0 => finrank (ZMod 2) (LinearMap.range (treeD0 A))
    | m + 1 => finrank (ZMod 2) (LinearMap.range (treeDS A m)) := by
  rw [treeCoch_d]
  cases m <;> rfl

/-- **`dim H⁰ + rank σ = dim A`.** -/
theorem finrank_H_zero :
    finrank (ZMod 2) (groupCohomology A 0) + finrank (ZMod 2) (LinearMap.range (sigmaA (A := A))) =
      finrank (ZMod 2) A := by
  have : FiniteDimensional (ZMod 2) ((treeCoch A).sc' 0 0 1).X₂ := finiteDimensional_X (A := A) 0
  rw [finrank_coh, ((treeCoch A).homologyIsoSc' 0 0 1 (by simp) (by simp)).toLinearEquiv.finrank_eq]
  have h := finrank_homology_add ((treeCoch A).sc' 0 0 1)
  have hf : finrank (ZMod 2) (LinearMap.range ((treeCoch A).sc' 0 0 1).f.hom) = 0 := by
    rw [show ((treeCoch A).sc' 0 0 1).f = 0 from (treeCoch A).shape 0 0 (by simp)]; simp
  have hg : finrank (ZMod 2) (LinearMap.range ((treeCoch A).sc' 0 0 1).g.hom) =
      finrank (ZMod 2) (LinearMap.range (treeD0 A)) := finrank_range_tcD (A := A) 0
  have hX : finrank (ZMod 2) ((treeCoch A).sc' 0 0 1).X₂ = finrank (ZMod 2) (A × A) := rfl
  rw [hf, hg, hX, finrank_range_treeD0, finrank_prod] at h
  omega

/-- **`dim H^{q+1} + rank ψ + rank σ = 2 dim A`**, every `q`. -/
theorem finrank_H_succ (hI : TrivNegOne A) (q : ℕ) :
    finrank (ZMod 2) (groupCohomology A (q + 1)) + finrank (ZMod 2) (LinearMap.range (psiA (A := A))) +
      finrank (ZMod 2) (LinearMap.range (sigmaA (A := A))) = 2 * finrank (ZMod 2) A := by
  have : FiniteDimensional (ZMod 2) ((treeCoch A).sc' q (q + 1) (q + 2)).X₂ := finiteDimensional_X (A := A) (q + 1)
  rw [finrank_coh, ((treeCoch A).homologyIsoSc' q (q + 1) (q + 2) (by simp) (by simp)).toLinearEquiv.finrank_eq]
  have h := finrank_homology_add ((treeCoch A).sc' q (q + 1) (q + 2))
  have hg : finrank (ZMod 2) (LinearMap.range ((treeCoch A).sc' q (q + 1) (q + 2)).g.hom) =
      finrank (ZMod 2) (LinearMap.range (treeDS A q)) := finrank_range_tcD (A := A) (q + 1)
  have hf : finrank (ZMod 2) (LinearMap.range ((treeCoch A).sc' q (q + 1) (q + 2)).f.hom) =
      finrank (ZMod 2) (LinearMap.range ((treeCoch A).d q (q + 1)).hom) := rfl
  have hX : finrank (ZMod 2) ((treeCoch A).sc' q (q + 1) (q + 2)).X₂ = finrank (ZMod 2) ((A × A) × A) := rfl
  rw [hg, hf, hX, finrank_range_tcD, finrank_range_treeDS hI, finrank_prod, finrank_prod] at h
  have hD0 := finrank_range_treeD0 (A := A)
  cases q with
  | zero =>
    rw [if_pos (by decide : Even 0)] at h
    simp only at h
    omega
  | succ q =>
    simp only at h
    rw [finrank_range_treeDS hI] at h
    by_cases hq : Even q
    · rw [if_pos hq, if_neg (by rw [Nat.even_add_one]; exact not_not.mpr hq)] at h; omega
    · rw [if_neg hq, if_pos (by rw [Nat.even_add_one]; exact hq)] at h; omega

end Char2

/-! ### From a linear map's rank to `rankF2` -/

/-- **The bridge**: if the images of a basis under `L`, read through a linear iso `ι` onto `F₂^w`, are exactly the bit
vectors of `rows` (as sets), then `rank L = rankF2 rows`. -/
theorem finrank_range_eq_rankF2 {V W I : Type} [AddCommGroup V] [Module (ZMod 2) V] [AddCommGroup W]
    [Module (ZMod 2) W] (L : V →ₗ[ZMod 2] W) (w : ℕ) (ι : W ≃ₗ[ZMod 2] (Fin w → ZMod 2))
    (b : Module.Basis I (ZMod 2) V) (rows : List ℕ) (hlt : rowsLt w rows = true)
    (h : Set.range (fun i => ι (L (b i))) = vec w '' {r | r ∈ rows}) :
    finrank (ZMod 2) (LinearMap.range L) = rankF2 rows := by
  have hmap : Submodule.map (ι : W →ₗ[ZMod 2] (Fin w → ZMod 2)) (LinearMap.range L) =
      Submodule.span (ZMod 2) (Set.range fun i => ι (L (b i))) := by
    rw [LinearMap.range_eq_map, ← b.span_eq, Submodule.map_span, Submodule.map_span, ← Set.range_comp,
      ← Set.range_comp]
    rfl
  rw [rankF2_eq_rank w rows (by simpa [rowsLt] using hlt), Matrix.rank_eq_finrank_span_row, range_toMat, ← h,
    ← hmap, ι.finrank_map_eq]

lemma vec_image_map_range (w : ℕ) (f : ℕ → ℕ) (d : ℕ) :
    vec w '' {r | r ∈ (List.range d).map f} = Set.range fun j : Fin d => vec w (f j) := by
  ext v
  simp only [Set.mem_image, Set.mem_ofPred_eq, List.mem_map, List.mem_range, Set.mem_range]
  constructor
  · rintro ⟨r, ⟨j, hj, rfl⟩, rfl⟩; exact ⟨⟨j, hj⟩, rfl⟩
  · rintro ⟨j, rfl⟩; exact ⟨f j, ⟨j, j.2, rfl⟩, rfl⟩

lemma vec_image_append (w : ℕ) (l₁ l₂ : List ℕ) :
    vec w '' {r | r ∈ l₁ ++ l₂} = vec w '' {r | r ∈ l₁} ∪ vec w '' {r | r ∈ l₂} := by
  rw [← Set.image_union]; congr 1; ext r; simp

lemma range_sum {α β γ : Type} (F : α ⊕ β → γ) :
    Set.range F = Set.range (F ∘ Sum.inl) ∪ Set.range (F ∘ Sum.inr) := by
  rw [← Set.Sum.elim_range]; congr 1; ext (i | i) <;> rfl

/-! ### `Sym^n F₂²` -/

/-- Reduction mod 2 of an element of `SL₂(ℤ)`. -/
def mod2 (g : SL(2, ℤ)) : Mat2 (ZMod 2) := (g : Matrix (Fin 2) (Fin 2) ℤ).map (Int.castRingHom (ZMod 2))

lemma mod2_mul (g h : SL(2, ℤ)) : mod2 (g * h) = mod2 g * mod2 h := by
  simp only [mod2, Matrix.SpecialLinearGroup.coe_mul, Matrix.map_mul]

lemma mod2_one : mod2 1 = 1 := by
  simp only [mod2, Matrix.SpecialLinearGroup.coe_one]; exact Matrix.map_one _ (map_zero _) (map_one _)

/-- **`Sym^n F₂²`**, `g ↦ ρ_n(g mod 2)` in the basis `u_t = x^{n-t} y^t` ([C] `def:rhon`, `eq:rhondef`; `GL2.rhoMat n A` is
the coordinate matrix of `GL2.rho n A`). -/
def symRep2 (n : ℕ) : Representation (ZMod 2) SL(2, ℤ) (Fin (n + 1) → ZMod 2) where
  toFun g := Matrix.toLin' (GL2.rhoMat n (mod2 g))
  map_one' := by rw [mod2_one, GL2.rhoMat_one, Matrix.toLin'_one]; rfl
  map_mul' g h := by rw [mod2_mul, GL2.rhoMat_mul, Matrix.toLin'_mul]; rfl

/-- `V₂(n) = Sym^n F₂²` as a representation of `SL₂(ℤ)`. -/
abbrev V2 (n : ℕ) : Rep.{0} (ZMod 2) SL(2, ℤ) := Rep.of (symRep2 n)

lemma V2_rho (n : ℕ) (g : SL(2, ℤ)) : (V2 n).ρ g = Matrix.toLin' (GL2.rhoMat n (mod2 g)) := rfl

lemma mod2_negOne : mod2 (-1) = 1 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [mod2]

lemma mod2_sS : mod2 sS = mS2 := by
  rw [mod2, sS, GL2.TauChk.M22.toSL_coe mS_det]
  ext i j; fin_cases i <;> fin_cases j <;> simp [GL2.TauChk.M22.toMat, mS, mS2]

lemma mod2_sU : mod2 sU = mU2 := by
  rw [mod2, sU, GL2.TauChk.M22.toSL_coe mU_det]
  ext i j; fin_cases i <;> fin_cases j <;> simp [GL2.TauChk.M22.toMat, mU, mU2]

lemma mod2_sU_sq : mod2 (sU ^ 2) = mUU2 := by
  rw [pow_two, mod2_mul, mod2_sU]
  ext i j; fin_cases i <;> fin_cases j <;> simp [mU2, mUU2, Matrix.mul_apply, Fin.sum_univ_two]
  all_goals decide

lemma V2_trivNegOne (n : ℕ) : TrivNegOne (V2 n) := by
  show Matrix.toLin' (GL2.rhoMat n (mod2 (-1))) = 1
  rw [mod2_negOne, GL2.rhoMat_one, Matrix.toLin'_one]; rfl

instance (n : ℕ) : FiniteDimensional (ZMod 2) (V2 n) :=
  inferInstanceAs (FiniteDimensional (ZMod 2) (Fin (n + 1) → ZMod 2))

lemma finrank_V2 (n : ℕ) : finrank (ZMod 2) (V2 n) = n + 1 :=
  Module.finrank_fin_fun (ZMod 2)

/-! #### The columns, as vectors -/

lemma vec_one_shiftLeft (n : ℕ) (j : Fin (n + 1)) : vec (n + 1) (Nat.shiftLeft 1 j) = Pi.single j 1 := by
  funext t
  change (if (1 <<< (j : ℕ)).testBit t then 1 else 0) = _
  rw [Nat.one_shiftLeft, Nat.testBit_two_pow, Pi.single_apply]
  by_cases h : t = j
  · subst h; simp
  · rw [if_neg h]
    have : ¬ ((j : ℕ) = t) := fun h' => h (Fin.ext h'.symm)
    simp [this]

lemma vec_col (n : ℕ) (j : Fin (n + 1)) (x : ℕ) (A : Mat2 (ZMod 2)) (a b c d : ℕ)
    (hA : A = !![(a : ZMod 2), b; c, d]) (hx : ∀ t, x.testBit t = decide (rhoEntry n a b c d t j % 2 = 1)) :
    vec (n + 1) x = (GL2.rhoMat n A).col j := by
  funext t
  rw [vec_of_testBit _ hx, Matrix.col_apply, rhoMat_entry n A a b c d hA]

lemma vec_colS (n : ℕ) (j : Fin (n + 1)) : vec (n + 1) (Nat.shiftLeft 1 (n - j)) = (GL2.rhoMat n mS2).col j :=
  vec_col n j _ mS2 0 1 1 0 (by simp [mS2]) (testBit_colS n j · (Nat.lt_succ_iff.mp j.2))

lemma vec_colU (n : ℕ) (j : Fin (n + 1)) :
    vec (n + 1) (Nat.shiftLeft (pascal j) (n - j)) = (GL2.rhoMat n mU2).col j :=
  vec_col n j _ mU2 0 1 1 1 (by simp [mU2]) (testBit_colU n j · (Nat.lt_succ_iff.mp j.2))

lemma vec_colU2 (n : ℕ) (j : Fin (n + 1)) : vec (n + 1) (pascal (n - j)) = (GL2.rhoMat n mUU2).col j :=
  vec_col n j _ mUU2 1 1 1 0 (by simp [mUU2]) (testBit_colU2 n j ·)

lemma V2_rho_single (n : ℕ) (g : SL(2, ℤ)) (j : Fin (n + 1)) :
    (V2 n).ρ g (Pi.single j 1) = (GL2.rhoMat n (mod2 g)).col j := by
  rw [V2_rho, Matrix.toLin'_apply, Matrix.mulVec_single_one]

lemma vec_colOnePlusS (n : ℕ) (j : Fin (n + 1)) :
    vec (n + 1) (colOnePlusS n j) = Pi.single j 1 + (V2 n).ρ sS (Pi.single j 1) := by
  rw [V2_rho_single, mod2_sS, ← vec_colS, ← vec_one_shiftLeft]
  exact vec_xor _ _

lemma vec_colOnePlusUp (n : ℕ) (j : Fin (n + 1)) :
    vec (n + 1) (colOnePlusUp n j (pascal j)) = Pi.single j 1 + (V2 n).ρ sU (Pi.single j 1) := by
  rw [V2_rho_single, mod2_sU, ← vec_colU, ← vec_one_shiftLeft]
  exact vec_xor _ _

lemma vec_colN3p (n : ℕ) (j : Fin (n + 1)) :
    vec (n + 1) (colN3p n j (pascal j) (pascal (n - j))) =
      Pi.single j 1 + (V2 n).ρ sU (Pi.single j 1) + (V2 n).ρ (sU ^ 2) (Pi.single j 1) := by
  rw [← vec_colOnePlusUp, V2_rho_single, mod2_sU_sq, ← vec_colU2]
  exact vec_xor _ _

/-! #### The two ranks -/

lemma sub_eq_add' {M : Type} [AddCommGroup M] [Module (ZMod 2) M] (x y : M) : x - y = x + y :=
  sub_eq_add_zmod2 x y

lemma nrmA_two (A : Rep.{0} (ZMod 2) SL(2, ℤ)) (t : SL(2, ℤ)) (x : A) : nrmA A t 2 x = x + A.ρ t x := by
  simp [nrmA, Finset.sum_range_succ]

lemma nrmA_three (A : Rep.{0} (ZMod 2) SL(2, ℤ)) (t : SL(2, ℤ)) (x : A) :
    nrmA A t 3 x = x + A.ρ t x + A.ρ (t ^ 2) x := by
  simp [nrmA, Finset.sum_range_succ]

lemma neg_eq_zmod2 {M : Type} [AddCommGroup M] [Module (ZMod 2) M] (x : M) : -x = x := by
  simpa using sub_eq_add_zmod2 (0 : M) x

/-- `ι : F₂^d × F₂^d ≃ F₂^{d+d}` (first factor in the low bits). -/
def ι2 (d : ℕ) : ((Fin d → ZMod 2) × (Fin d → ZMod 2)) ≃ₗ[ZMod 2] (Fin (d + d) → ZMod 2) :=
  (LinearEquiv.sumArrowLequivProdArrow (Fin d) (Fin d) (ZMod 2) (ZMod 2)).symm.trans
    (LinearEquiv.funCongrLeft (ZMod 2) (ZMod 2) finSumFinEquiv.symm)

lemma ι2_castAdd (d : ℕ) (x y : Fin d → ZMod 2) (t : Fin d) : ι2 d (x, y) (Fin.castAdd d t) = x t := by
  simp [ι2, LinearEquiv.funCongrLeft_apply]

lemma ι2_natAdd (d : ℕ) (x y : Fin d → ZMod 2) (t : Fin d) : ι2 d (x, y) (Fin.natAdd d t) = y t := by
  show (LinearEquiv.sumArrowLequivProdArrow (Fin d) (Fin d) (ZMod 2) (ZMod 2)).symm (x, y)
    (finSumFinEquiv.symm (Fin.natAdd d t)) = y t
  rw [finSumFinEquiv_symm_apply_natAdd]; rfl

lemma testBit_colOnePlusS_hi (n j k : ℕ) (hj : j ≤ n) (hk : n < k) : (colOnePlusS n j).testBit k = false := by
  change ((1 <<< j) ^^^ (1 <<< (n - j))).testBit k = false
  rw [Nat.testBit_xor, Nat.one_shiftLeft, Nat.one_shiftLeft, Nat.testBit_two_pow, Nat.testBit_two_pow]
  have h1 : j ≠ k := by omega
  have h2 : n - j ≠ k := by omega
  simp [h1, h2]

lemma vec_colSigp (n : ℕ) (j : Fin (n + 1)) :
    vec ((n + 1) + (n + 1)) (colSigp n j (pascal j)) =
      ι2 (n + 1) (vec (n + 1) (colOnePlusS n j), vec (n + 1) (colOnePlusUp n j (pascal j))) := by
  funext i
  refine Fin.addCases (fun t => ?_) (fun t => ?_) i
  · rw [ι2_castAdd]
    change (if ((colOnePlusS n j) ^^^ ((colOnePlusUp n j (pascal j)) <<< (n + 1))).testBit t then (1 : ZMod 2) else 0) =
      (if (colOnePlusS n j).testBit t then 1 else 0)
    rw [Nat.testBit_xor, Nat.testBit_shiftLeft]
    simp [show ¬ (n + 1 ≤ (t : ℕ)) from by omega]
  · rw [ι2_natAdd]
    change (if ((colOnePlusS n j) ^^^ ((colOnePlusUp n j (pascal j)) <<< (n + 1))).testBit (n + 1 + t) then
      (1 : ZMod 2) else 0) = (if (colOnePlusUp n j (pascal j)).testBit t then 1 else 0)
    rw [Nat.testBit_xor, Nat.testBit_shiftLeft, testBit_colOnePlusS_hi n j _ (Nat.lt_succ_iff.mp j.2) (by omega)]
    simp

/-- **`rank σ = rankF2 (colsSig n)`** on `V₂(n)`. -/
theorem finrank_range_sigma_V2 (n : ℕ) (hlt : rowsLt ((n + 1) + (n + 1)) (colsSig n) = true) :
    finrank (ZMod 2) (LinearMap.range (sigmaA (A := V2 n))) = rankF2 (colsSig n) := by
  refine finrank_range_eq_rankF2 _ _ (show (V2 n × V2 n) ≃ₗ[ZMod 2] _ from ι2 (n + 1))
    (show Module.Basis _ (ZMod 2) (V2 n) from Pi.basisFun (ZMod 2) (Fin (n + 1))) _ hlt ?_
  rw [colsSig_eq, vec_image_map_range]
  congr 1; funext j
  rw [vec_colSigp, vec_colOnePlusS, vec_colOnePlusUp, Pi.basisFun_apply, sigmaA_apply]
  simp only [sub_eq_add']

/-- **`rank ψ = rankF2 (colsA n)`** on `V₂(n)`. -/
theorem finrank_range_psi_V2 (n : ℕ) (hlt : rowsLt (n + 1) (colsA n) = true) :
    finrank (ZMod 2) (LinearMap.range (psiA (A := V2 n))) = rankF2 (colsA n) := by
  refine finrank_range_eq_rankF2 _ _ (LinearEquiv.refl _ _)
    (show Module.Basis _ (ZMod 2) (V2 n × V2 n) from
      (Pi.basisFun (ZMod 2) (Fin (n + 1))).prod (Pi.basisFun (ZMod 2) (Fin (n + 1)))) _ hlt ?_
  rw [colsA_eq, vec_image_append, vec_image_map_range, vec_image_map_range, range_sum]
  congr 1
  · congr 1; funext j
    simp only [Function.comp_apply, Module.Basis.prod_apply, Sum.elim_inl, LinearMap.coe_inl, Pi.basisFun_apply,
      LinearEquiv.refl_apply, psiA, phiA, if_neg (by decide : ¬ Even 1), LinearMap.sub_apply, LinearMap.comp_apply,
      LinearMap.fst_apply, LinearMap.snd_apply, map_zero, sub_zero]
    rw [vec_colOnePlusS, nrmA_two]
  · congr 1; funext j
    simp only [Function.comp_apply, Module.Basis.prod_apply, Sum.elim_inr, LinearMap.coe_inr, Pi.basisFun_apply,
      LinearEquiv.refl_apply, psiA, phiA, if_neg (by decide : ¬ Even 1), LinearMap.sub_apply, LinearMap.comp_apply,
      LinearMap.fst_apply, LinearMap.snd_apply, map_zero, zero_sub]
    rw [vec_colN3p, neg_eq_zmod2, nrmA_three]

/-! ### The cell theorem -/

/-- **`dim_{F₂} H^q(SL₂(ℤ); Sym^n F₂²)` for every `q`, from two ranks**: with `a = rankF2 (colsA n)` (the rank of
`[1 + ρ_n(S) | 1 + ρ_n(U) + ρ_n(U²)]`) and `s = rankF2 (colsSig n)` (the rank of `[1 + ρ_n(S) ; 1 + ρ_n(U)]`), both mod 2,
`dim H⁰ = (n + 1) − s` and `dim H^{q+1} = 2(n + 1) − a − s`. The four hypotheses are what a cell module decides in the
kernel. -/
theorem V2_dims (n a s : ℕ) (hlA : rowsLt (n + 1) (colsA n) = true)
    (hlS : rowsLt ((n + 1) + (n + 1)) (colsSig n) = true) (hA : rankF2 (colsA n) = a)
    (hS : rankF2 (colsSig n) = s) :
    finrank (ZMod 2) (groupCohomology (V2 n) 0) + s = n + 1 ∧
      ∀ q, finrank (ZMod 2) (groupCohomology (V2 n) (q + 1)) + a + s = 2 * (n + 1) := by
  have h0 := finrank_H_zero (A := V2 n)
  have h1 := finrank_H_succ (A := V2 n) (V2_trivNegOne n)
  rw [finrank_range_sigma_V2 n hlS, hS, finrank_V2] at h0
  refine ⟨h0, fun q => ?_⟩
  have := h1 q
  rw [finrank_range_sigma_V2 n hlS, hS, finrank_range_psi_V2 n hlA, hA, finrank_V2] at this
  exact this

/-- **The cell theorem with the census values**: `h0 = dim H⁰` and `h1 = dim H^q` for every `q ≥ 1`, from the kernel's
`rowsLtK` / `rankF2` facts and two arithmetic identities. -/
theorem V2_dims_eq (n a s h0 h1 : ℕ) (hlA : rowsLtK (n + 1) (colsA n) = true)
    (hlS : rowsLtK ((n + 1) + (n + 1)) (colsSig n) = true) (hA : rankF2 (colsA n) = a)
    (hS : rankF2 (colsSig n) = s) (e0 : n + 1 = h0 + s) (e1 : 2 * (n + 1) = h1 + a + s) :
    finrank (ZMod 2) (groupCohomology (V2 n) 0) = h0 ∧
      ∀ q, finrank (ZMod 2) (groupCohomology (V2 n) (q + 1)) = h1 := by
  obtain ⟨d0, d1⟩ := V2_dims n a s ((rowsLtK_eq _ _).symm.trans hlA) ((rowsLtK_eq _ _).symm.trans hlS) hA hS
  exact ⟨by omega, fun q => by have := d1 q; omega⟩

/-! ### The width bounds hold for every `n` (LV-1 audit NIT-4)

So a cell needs only the two rank decisions. -/

lemma pascal_lt (m : ℕ) : pascal m < 2 ^ (m + 1) := by
  induction m with
  | zero => rw [pascal_zero]; norm_num
  | succ m ih =>
    rw [pascal_succ]
    refine Nat.xor_lt_two_pow (lt_of_lt_of_le ih (Nat.pow_le_pow_right (by norm_num) (by omega))) ?_
    rw [Nat.shiftLeft_eq, pow_one, pow_succ]
    exact Nat.mul_lt_mul_of_pos_right ih (by norm_num)

lemma two_pow_lt {k w : ℕ} (h : k < w) : 2 ^ k < 2 ^ w := Nat.pow_lt_pow_right (by norm_num) h

lemma colOnePlusS_lt (n j : ℕ) (hj : j ≤ n) : colOnePlusS n j < 2 ^ (n + 1) := by
  change (1 <<< j) ^^^ (1 <<< (n - j)) < _
  rw [Nat.one_shiftLeft, Nat.one_shiftLeft]
  exact Nat.xor_lt_two_pow (two_pow_lt (by omega)) (two_pow_lt (by omega))

lemma colOnePlusUp_lt (n j : ℕ) (hj : j ≤ n) : colOnePlusUp n j (pascal j) < 2 ^ (n + 1) := by
  change (1 <<< j) ^^^ (pascal j <<< (n - j)) < _
  rw [Nat.one_shiftLeft, Nat.shiftLeft_eq]
  refine Nat.xor_lt_two_pow (two_pow_lt (by omega)) ?_
  calc pascal j * 2 ^ (n - j) < 2 ^ (j + 1) * 2 ^ (n - j) :=
        Nat.mul_lt_mul_of_pos_right (pascal_lt j) (Nat.two_pow_pos _)
    _ = 2 ^ (n + 1) := by rw [← pow_add]; congr 1; omega

lemma colN3p_lt (n j : ℕ) (hj : j ≤ n) : colN3p n j (pascal j) (pascal (n - j)) < 2 ^ (n + 1) :=
  Nat.xor_lt_two_pow (colOnePlusUp_lt n j hj)
    (lt_of_lt_of_le (pascal_lt _) (Nat.pow_le_pow_right (by norm_num) (by omega)))

lemma colSigp_lt (n j : ℕ) (hj : j ≤ n) : colSigp n j (pascal j) < 2 ^ ((n + 1) + (n + 1)) := by
  change colOnePlusS n j ^^^ (colOnePlusUp n j (pascal j) <<< (n + 1)) < _
  refine Nat.xor_lt_two_pow
    (lt_of_lt_of_le (colOnePlusS_lt n j hj) (Nat.pow_le_pow_right (by norm_num) (by omega))) ?_
  rw [Nat.shiftLeft_eq, show 2 ^ ((n + 1) + (n + 1)) = 2 ^ (n + 1) * 2 ^ (n + 1) from pow_add _ _ _]
  exact Nat.mul_lt_mul_of_pos_right (colOnePlusUp_lt n j hj) (Nat.two_pow_pos _)

/-- `rowsLt (n + 1) (colsA n)`, every `n`. -/
theorem rowsLt_colsA (n : ℕ) : rowsLt (n + 1) (colsA n) = true := by
  rw [colsA_eq]
  simp only [rowsLt, List.all_append, List.all_map, Bool.and_eq_true, List.all_eq_true, List.mem_range,
    Function.comp_apply, decide_eq_true_eq]
  exact ⟨fun j hj => colOnePlusS_lt n j (by omega), fun j hj => colN3p_lt n j (by omega)⟩

/-- `rowsLt (2(n + 1)) (colsSig n)`, every `n`. -/
theorem rowsLt_colsSig (n : ℕ) : rowsLt ((n + 1) + (n + 1)) (colsSig n) = true := by
  rw [colsSig_eq]
  simp only [rowsLt, List.all_map, List.all_eq_true, List.mem_range, Function.comp_apply, decide_eq_true_eq]
  exact fun j hj => colSigp_lt n j (by omega)

/-- **The cell theorem from the two ranks alone** (LV-2): `a = rankF2 (colsA n)`, `s = rankF2 (colsSig n)`, then
`h0 = dim H⁰` and `h1 = dim H^q` for every `q ≥ 1`. -/
theorem V2_dims_of_ranks (n a s h0 h1 : ℕ) (hA : rankF2 (colsA n) = a) (hS : rankF2 (colsSig n) = s)
    (e0 : n + 1 = h0 + s) (e1 : 2 * (n + 1) = h1 + a + s) :
    finrank (ZMod 2) (groupCohomology (V2 n) 0) = h0 ∧
      ∀ q, finrank (ZMod 2) (groupCohomology (V2 n) (q + 1)) = h1 := by
  obtain ⟨d0, d1⟩ := V2_dims n a s (rowsLt_colsA n) (rowsLt_colsSig n) hA hS
  exact ⟨by omega, fun q => by have := d1 q; omega⟩

end GL2.L23.TreeF2
