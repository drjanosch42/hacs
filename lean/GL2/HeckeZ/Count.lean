/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ M-HZ5: the counting criterion and the upper-bound lemma (L-HZ4, slot 26)

`docs/implementation/slot24/heckez/01-math-core.md` §4 (the two tags `non-reduced certified`, `upper bound certified`); [C]
`prop:heckeZcong`(2), the text of `thm:heckeZcert`'s "level N > 1, full" clause (the counting criterion, `N` a prime power).

**The counting bound** (`finrank_le_of_reduced`, with `U`, `m` free): `𝕋 ⊆ End_ℤ L` commutative and reduced, `𝕋^an ≤ 𝕋`,
`U ∈ 𝕋` with `𝕋 = 𝕋^an + U𝕋` (for `N = p^r` and `U = U_p`). Then `rank 𝕋 ≤ rank 𝕋^an + rank (U^m L)`.

Proof over `ℤ`: `U𝕋` meets the kernel of `𝕋 → End(U^m L)` trivially (`a = Uy`, `a U^m = 0` gives `a^{m+1} = 0`, `a = 0`), so
`rank U𝕋 ≤ rank B` for `B` the image of `𝕋` in `End(U^m L)`, reduced and faithful on `U^m L`, whence
`rank B ≤ rank U^m L` (`finrank_le_int`, the product-of-fields argument after `⊗ ℚ`); and `𝕋 = 𝕋^an + U𝕋` gives the sum.

This is the content of 02 §3.3 / 01 §4: at `(8, 10)` a reduced `𝕋` would need `d − d^an ≥ 7` systems with `U_2 ≠ 0`, but only
six non-zero roots are there. Taking `m = 2` and an explicit rank factorisation of `U²` is the cell's route (no characteristic polynomial).
-/

namespace GL2.HeckeZ.Count

open TensorProduct Polynomial

noncomputable section

/-! ### The field lemma: a faithful reduced commutative algebra embeds in its module -/

/-- **`dim_F A ≤ dim_F V`** for a reduced commutative `F`-algebra `A` of finite dimension acting faithfully on `V`. -/
theorem finrank_le_of_reduced_faithful {F A V : Type*} [Field F] [CommRing A] [Algebra F A] [FiniteDimensional F A]
    [IsReduced A] [AddCommGroup V] [Module F V] [FiniteDimensional F V] (ρ : A →ₐ[F] Module.End F V)
    (hρ : Function.Injective ρ) : Module.finrank F A ≤ Module.finrank F V := by
  classical
  haveI : IsArtinianRing A := IsArtinianRing.of_finite F A
  haveI : Fintype (MaximalSpectrum A) := Fintype.ofFinite _
  let e := IsArtinianRing.equivPi A
  let E : MaximalSpectrum A → A := fun m => e.symm (Pi.single m 1)
  have he : ∀ m, e (E m) = Pi.single m 1 := fun m => e.apply_symm_apply _
  have hEne : ∀ m, E m ≠ 0 := by
    intro m h
    have h1 := congrFun (he m) m
    rw [h, map_zero, Pi.single_eq_same] at h1
    haveI : Nontrivial (A ⧸ m.asIdeal) := Ideal.Quotient.nontrivial_iff.mpr m.isMaximal.ne_top
    exact zero_ne_one h1
  have hvm : ∀ m, ∃ v : V, ρ (E m) v ≠ 0 := by
    intro m
    by_contra hcon
    push Not at hcon
    exact hEne m (hρ (by rw [map_zero]; ext v; simpa using hcon v))
  choose v hv using hvm
  let w : V := ∑ m, ρ (E m) (v m)
  have hEE : ∀ m n, m ≠ n → E m * E n = 0 := by
    intro m n hmn
    apply e.injective
    rw [map_mul, he, he, map_zero]
    funext x
    by_cases hx : x = m
    · subst hx
      simp only [Pi.mul_apply, Pi.single_eq_same, Pi.zero_apply]
      rw [Pi.single_eq_of_ne hmn, mul_zero]
    · simp only [Pi.mul_apply, Pi.zero_apply]
      rw [Pi.single_eq_of_ne hx, zero_mul]
  have hEEs : ∀ m, E m * E m = E m := by
    intro m
    apply e.injective
    rw [map_mul, he]
    funext x
    by_cases hx : x = m
    · subst hx; simp only [Pi.mul_apply, Pi.single_eq_same, mul_one]
    · simp only [Pi.mul_apply]; rw [Pi.single_eq_of_ne hx, zero_mul]
  have hinj : ∀ a : A, ρ a w = 0 → a = 0 := by
    intro a ha
    apply e.injective
    rw [map_zero]
    funext n
    by_contra hne
    have h1 : ρ (a * E n) (v n) = 0 := by
      have h2 : ρ (E n) (ρ a w) = 0 := by rw [ha, map_zero]
      have h3 : ρ (E n) (ρ a w) = ρ (a * E n) (v n) := by
        have hc : ρ (E n) (ρ a w) = ρ a (ρ (E n) w) := by
          rw [← Module.End.mul_apply, ← Module.End.mul_apply, ← map_mul, ← map_mul, mul_comm]
        rw [hc]
        have : ρ (E n) w = ρ (E n) (v n) := by
          simp only [w, map_sum]
          rw [Finset.sum_eq_single n]
          · rw [← Module.End.mul_apply, ← map_mul, hEEs]
          · intro b _ hb
            rw [← Module.End.mul_apply, ← map_mul, hEE n b hb.symm, map_zero, LinearMap.zero_apply]
          · intro h; exact absurd (Finset.mem_univ _) h
        rw [this, ← Module.End.mul_apply, ← map_mul]
      rw [← h3, h2]
    -- `e a n` is a unit of the field `A ⧸ n`
    have hfield : IsField (A ⧸ n.asIdeal) := (Ideal.Quotient.maximal_ideal_iff_isField_quotient _).mp n.isMaximal
    obtain ⟨y, hy⟩ := hfield.mul_inv_cancel hne
    let b : A := e.symm (Pi.single n y)
    have hb : b * (a * E n) = E n := by
      apply e.injective
      rw [map_mul, map_mul, he, show e b = Pi.single n y from e.apply_symm_apply _]
      funext x
      by_cases hx : x = n
      · subst hx
        simp only [Pi.mul_apply, Pi.single_eq_same]
        rw [mul_one, mul_comm, hy]
      · simp [Pi.single_eq_of_ne hx]
    apply hv n
    rw [← hb, map_mul, Module.End.mul_apply, h1, map_zero]
  let ψ : A →ₗ[F] V := (LinearMap.applyₗ w).comp ρ.toLinearMap
  have hψ : Function.Injective ψ := by
    rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
    intro a ha
    exact hinj a ha
  exact LinearMap.finrank_le_finrank_of_injective hψ

/-! ### The lattice form -/

variable {W : Type*} [AddCommGroup W] [Module.Free ℤ W] [Module.Finite ℤ W]

/-- `f ↦ f ⊗ ℚ`, a ring map. -/
def bc : Module.End ℤ W →+* Module.End ℚ (ℚ ⊗[ℤ] W) := (Module.End.baseChangeHom ℤ ℚ W).toRingHom

lemma bc_apply (f : Module.End ℤ W) : bc f = f.baseChange ℚ := rfl

lemma bc_tmul (f : Module.End ℤ W) (v : W) : bc f ((1 : ℚ) ⊗ₜ v) = (1 : ℚ) ⊗ₜ f v := by
  simp [bc_apply, LinearMap.baseChange_tmul]

lemma embQ_injective : Function.Injective (fun v : W => (1 : ℚ) ⊗ₜ[ℤ] v) := by
  intro v w h
  let b := Module.Free.chooseBasis ℤ W
  refine b.ext_elem fun i => ?_
  have := congrArg (fun x => (Algebra.TensorProduct.basis ℚ b).repr x i) h
  simp only [Algebra.TensorProduct.basis_repr_tmul] at this
  simpa using this

lemma bc_injective : Function.Injective (bc (W := W)) := by
  rw [injective_iff_map_eq_zero]
  intro f h
  ext v
  apply embQ_injective
  simp only [← bc_tmul, h, LinearMap.zero_apply, TensorProduct.tmul_zero]

lemma bc_nsmul (n : ℤ) (f : Module.End ℤ W) : bc (n • f) = (n : ℚ) • bc f := by
  rw [map_zsmul]; exact (Int.cast_smul_eq_zsmul ℚ n _).symm

lemma lin_indep_bc {ι : Type*} [Fintype ι] {v : ι → Module.End ℤ W} (hv : LinearIndependent ℤ v) (g : ι → ℚ)
    (hg : ∑ i, g i • bc (v i) = 0) : g = 0 := by
  classical
  obtain ⟨D, hD⟩ := IsLocalization.exist_integer_multiples_of_finite (nonZeroDivisors ℤ) g
  choose n hn using hD
  have hD0 : ((D : ℤ) : ℚ) ≠ 0 := by exact_mod_cast nonZeroDivisors.ne_zero D.2
  have hkey : ∑ i, (n i) • v i = 0 := by
    apply bc_injective
    rw [map_zero, map_sum]
    calc ∑ i, bc (n i • v i)
        = ∑ i, ((D : ℤ) : ℚ) • (g i • bc (v i)) := by
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [bc_nsmul, smul_smul]
          congr 1
          have := hn i
          simp only [eq_intCast, zsmul_eq_mul] at this
          rw [this]
      _ = ((D : ℤ) : ℚ) • ∑ i, g i • bc (v i) := by rw [Finset.smul_sum]
      _ = 0 := by rw [hg, smul_zero]
  have hn0 := Fintype.linearIndependent_iff.mp hv n hkey
  funext i
  have := hn i
  simp only [eq_intCast, hn0, zsmul_eq_mul, Int.cast_zero] at this
  simpa [hD0] using this

/-- **`rank B ≤ rank W`** for a commutative reduced subring `B` of `End_ℤ W` (`W` free of finite rank). -/
theorem finrank_le_int (B : Subalgebra ℤ (Module.End ℤ W)) (hc : ∀ x ∈ B, ∀ y ∈ B, x * y = y * x)
    (hnil : ∀ x ∈ B, IsNilpotent x → x = 0) : Module.finrank ℤ B ≤ Module.finrank ℤ W := by
  classical
  have hcomm : ∀ a ∈ (bc '' (B : Set (Module.End ℤ W))), ∀ b ∈ (bc '' (B : Set (Module.End ℤ W))), a * b = b * a := by
    rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩
    rw [← map_mul, hc x hx y hy, map_mul]
  let A : Subalgebra ℚ (Module.End ℚ (ℚ ⊗[ℤ] W)) := Algebra.adjoin ℚ (bc '' (B : Set (Module.End ℤ W)))
  letI : CommRing A := Algebra.adjoinCommRingOfComm ℚ hcomm
  -- every element of `A` is `(1/c) • bc b`
  have hform : ∀ a ∈ A, ∃ (c : ℤ) (b : Module.End ℤ W), c ≠ 0 ∧ b ∈ B ∧ (c : ℚ) • a = bc b := by
    intro a ha
    induction ha using Algebra.adjoin_induction with
    | mem x hx =>
      obtain ⟨b, hb, rfl⟩ := hx
      exact ⟨1, b, one_ne_zero, hb, by simp⟩
    | algebraMap r =>
      refine ⟨(r.den : ℤ), (r.num : Module.End ℤ W), by exact_mod_cast r.den_nz, B.intCast_mem _, ?_⟩
      have h1 : ((r.den : ℤ) : ℚ) • (algebraMap ℚ (Module.End ℚ (ℚ ⊗[ℤ] W)) r) = (r.num : Module.End ℚ (ℚ ⊗[ℤ] W)) := by
        have hr : ((r.den : ℤ) : ℚ) * r = r.num := by push_cast; exact Rat.den_mul_eq_num r
        rw [← map_intCast (algebraMap ℚ (Module.End ℚ (ℚ ⊗[ℤ] W))), ← hr, map_mul, Algebra.smul_def]
      rw [h1, map_intCast]
    | add x y _ _ hx hy =>
      obtain ⟨c1, b1, hc1, hb1, e1⟩ := hx
      obtain ⟨c2, b2, hc2, hb2, e2⟩ := hy
      refine ⟨c1 * c2, c2 • b1 + c1 • b2, mul_ne_zero hc1 hc2,
        B.add_mem (B.zsmul_mem hb1 _) (B.zsmul_mem hb2 _), ?_⟩
      rw [map_add, bc_nsmul, bc_nsmul, ← e1, ← e2, smul_add, smul_smul, smul_smul]
      push_cast; ring_nf
    | mul x y _ _ hx hy =>
      obtain ⟨c1, b1, hc1, hb1, e1⟩ := hx
      obtain ⟨c2, b2, hc2, hb2, e2⟩ := hy
      refine ⟨c1 * c2, b1 * b2, mul_ne_zero hc1 hc2, B.mul_mem hb1 hb2, ?_⟩
      rw [map_mul, ← e1, ← e2, smul_mul_smul_comm]
      push_cast; ring_nf
  have hredA : IsReduced A := by
    refine ⟨fun a hn => ?_⟩
    obtain ⟨c, b, hc0, hb, e⟩ := hform a.1 a.2
    have hc0' : ((c : ℚ)) ≠ 0 := by exact_mod_cast hc0
    obtain ⟨k, hk⟩ := hn
    have hk' : (a : Module.End ℚ (ℚ ⊗[ℤ] W)) ^ k = 0 := by
      have := congrArg Subtype.val hk; exact this
    have hbk : b ^ k = 0 := by
      apply bc_injective
      rw [map_zero, map_pow, ← e, _root_.smul_pow, hk', smul_zero]
    have hb0 : b = 0 := hnil b hb ⟨k, hbk⟩
    apply Subtype.ext
    have h0 : (c : ℚ) • (a : Module.End ℚ (ℚ ⊗[ℤ] W)) = 0 := by rw [e, hb0, map_zero]
    exact (smul_eq_zero.mp h0).resolve_left hc0'
  let ρ : A →ₐ[ℚ] Module.End ℚ (ℚ ⊗[ℤ] W) := A.val
  have hle := finrank_le_of_reduced_faithful ρ Subtype.val_injective
  have hV : Module.finrank ℚ (ℚ ⊗[ℤ] W) = Module.finrank ℤ W := by
    rw [Module.finrank_eq_card_basis (Algebra.TensorProduct.basis ℚ (Module.Free.chooseBasis ℤ W)),
      Module.finrank_eq_card_chooseBasisIndex]
  have hBfin : Module.Finite ℤ B := Module.IsNoetherian.finite ℤ B.toSubmodule
  haveI : Module.Free ℤ B := Module.free_of_finite_type_torsion_free' (R := ℤ)
  let bB := Module.Free.chooseBasis ℤ B
  have hind : LinearIndependent ℚ (fun i => (⟨bc (bB i : Module.End ℤ W),
      Algebra.subset_adjoin ⟨_, (bB i).2, rfl⟩⟩ : A)) := by
    rw [Fintype.linearIndependent_iff]
    intro g hg i
    have h1 : ∑ j, g j • bc ((bB j : Module.End ℤ W)) = 0 := by
      have := congrArg Subtype.val hg
      simpa [AddSubmonoidClass.coe_finset_sum, Subalgebra.coe_smul] using this
    have hindZ : LinearIndependent ℤ (fun j => (bB j : Module.End ℤ W)) := by
      apply bB.linearIndependent.map' B.toSubmodule.subtype
      exact LinearMap.ker_eq_bot.mpr Subtype.val_injective
    exact congrFun (lin_indep_bc hindZ g h1) i
  have hc2 := hind.fintype_card_le_finrank
  have hcardB : Fintype.card (Module.Free.ChooseBasisIndex ℤ B) = Module.finrank ℤ B :=
    (Module.finrank_eq_card_chooseBasisIndex ℤ B).symm
  omega

/-! ### The counting bound -/

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]

set_option synthInstance.maxHeartbeats 400000 in
/-- **The counting bound** (01 §4): `𝕋` commutative and reduced, `𝕋 = 𝕋^an + U 𝕋`. Then `rank 𝕋 ≤ rank 𝕋^an + rank(U^m L)`. -/
theorem finrank_le_of_reduced (T Tan : Subalgebra ℤ (Module.End ℤ L)) (hc : ∀ x ∈ T, ∀ y ∈ T, x * y = y * x)
    (hTan : Tan ≤ T) (u : Module.End ℤ L) (hu : u ∈ T) (hgen : ∀ t ∈ T, ∃ s ∈ Tan, ∃ y ∈ T, t = s + u * y)
    (hred : ∀ t ∈ T, IsNilpotent t → t = 0) (m : ℕ) (hm : 1 ≤ m) :
    Module.finrank ℤ T ≤ Module.finrank ℤ Tan + Module.finrank ℤ (LinearMap.range (u ^ m)) := by
  classical
  haveI hTfin : Module.Finite ℤ T := Module.IsNoetherian.finite ℤ T.toSubmodule
  haveI : Module.Finite ℤ Tan := Module.IsNoetherian.finite ℤ Tan.toSubmodule
  have hpow : ∀ t ∈ T, ∀ n : ℕ, t * u ^ n = u ^ n * t := by
    intro t ht n
    induction n with
    | zero => simp
    | succ n ih =>
      have hn := hc t ht u hu
      rw [pow_succ, ← mul_assoc, ih, mul_assoc, hn, ← mul_assoc]
  -- restriction to `W = u^m L`
  set W := LinearMap.range (u ^ m) with hW
  haveI : Module.Finite ℤ W := Module.IsNoetherian.finite ℤ W
  haveI : Module.Free ℤ W := Module.free_of_finite_type_torsion_free' (R := ℤ)
  have htW : ∀ t ∈ T, ∀ x ∈ W, t x ∈ W := by
    rintro t ht _ ⟨y, rfl⟩
    refine ⟨t y, ?_⟩
    simpa [Module.End.mul_apply] using (congrArg (fun F : Module.End ℤ L => F y) (hpow t ht m)).symm
  let ρ : T →ₐ[ℤ] Module.End ℤ W :=
    { toFun := fun t => (t : Module.End ℤ L).restrict (htW t t.2)
      map_one' := by ext x; rfl
      map_mul' := by intro x y; ext w; rfl
      map_zero' := by ext x; rfl
      map_add' := by intro x y; ext w; rfl
      commutes' := by
        intro r; ext w
        simp [Algebra.algebraMap_eq_smul_one, Module.End.one_apply] }
  have hρ : ∀ (t : T) (x : L), (ρ t ⟨(u ^ m) x, x, rfl⟩ : L) = (u ^ m) ((t : Module.End ℤ L) x) := by
    intro t x
    change (t : Module.End ℤ L) ((u ^ m) x) = (u ^ m) ((t : Module.End ℤ L) x)
    simpa [Module.End.mul_apply] using congrArg (fun F : Module.End ℤ L => F x) (hpow t t.2 m)
  -- the image `B` of `𝕋`, reduced, faithful on `W`
  let B : Subalgebra ℤ (Module.End ℤ W) := ρ.range
  have hρ0 : ∀ t : T, ρ t = 0 → u ^ m * (t : Module.End ℤ L) = 0 := by
    intro t h
    ext x
    have h1 := hρ t x
    rw [h] at h1
    simpa [Module.End.mul_apply] using h1.symm
  have hB : Module.finrank ℤ B ≤ Module.finrank ℤ W := by
    apply finrank_le_int B
    · rintro _ ⟨x, rfl⟩ _ ⟨y, rfl⟩
      rw [← map_mul, ← map_mul]
      congr 1
      exact Subtype.ext (hc x x.2 y y.2)
    · rintro _ ⟨x, rfl⟩ ⟨k, hk⟩
      rw [← map_pow] at hk
      have h1 : u ^ m * (x : Module.End ℤ L) ^ k = 0 := by
        have := hρ0 (x ^ k) hk; simpa using this
      have hcomm' : Commute u (x : Module.End ℤ L) := (hc _ x.2 _ hu).symm
      have hux : (u * (x : Module.End ℤ L)) ^ (m + k) = 0 := by
        have e1 : (u * (x : Module.End ℤ L)) ^ (m + k) = (u ^ m * (x : Module.End ℤ L) ^ k) * (u ^ k * (x : Module.End ℤ L) ^ m) := by
          rw [hcomm'.mul_pow, pow_add u, pow_add (x : Module.End ℤ L)]
          have c1 : Commute (u ^ k) ((x : Module.End ℤ L) ^ k) := hcomm'.pow_pow _ _
          rw [((Commute.refl (x : Module.End ℤ L)).pow_pow m k).eq, mul_assoc (u ^ m), ← mul_assoc (u ^ k), c1.eq]
          simp only [mul_assoc]
        rw [e1, h1, zero_mul]
      have hux0 : u * (x : Module.End ℤ L) = 0 := hred _ (T.mul_mem hu x.2) ⟨_, hux⟩
      ext ⟨w, z, rfl⟩
      change ((ρ x) ⟨(u ^ m) z, z, rfl⟩ : L) = 0
      rw [hρ x z]
      have hmp : u ^ m = u ^ (m - 1) * u := by rw [← pow_succ]; congr 1; omega
      rw [hmp, Module.End.mul_apply, ← Module.End.mul_apply u, hux0, LinearMap.zero_apply, map_zero]
  -- the ideal `u𝕋` meets the kernel of `ρ` trivially
  have hu' : u ∈ T := hu
  let uT : T := ⟨u, hu⟩
  let φ : T →ₗ[ℤ] T := LinearMap.mulLeft ℤ uT
  let UT : Submodule ℤ T := LinearMap.range φ
  let inc : UT →ₗ[ℤ] B :=
    (ρ.rangeRestrict.toLinearMap).comp UT.subtype
  have hinc : Function.Injective inc := by
    rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
    rintro ⟨y, z, rfl⟩ h
    have h1 : ρ (φ z) = 0 := by
      have := congrArg Subtype.val h
      simpa [inc] using this
    have h2 := hρ0 _ h1
    have hy : ((φ z : T) : Module.End ℤ L) = u * (z : Module.End ℤ L) := rfl
    rw [hy] at h2
    -- `(u z)^{m+1} = (u^m · u z) · z^m = 0`
    have hcomm' : Commute u (z : Module.End ℤ L) := (hc _ z.2 _ hu).symm
    have h3 : (u * (z : Module.End ℤ L)) ^ (m + 1) = 0 := by
      have e2 : (u * (z : Module.End ℤ L)) ^ (m + 1) = (u ^ m * (u * (z : Module.End ℤ L))) * (z : Module.End ℤ L) ^ m := by
        rw [hcomm'.mul_pow, pow_succ u, pow_succ (z : Module.End ℤ L)]
        simp only [mul_assoc]
        rw [pow_mul_comm']
      rw [e2, h2, zero_mul]
    have hmemT : u * (z : Module.End ℤ L) ∈ T := T.mul_mem hu z.2
    have h4 := hred _ hmemT ⟨_, h3⟩
    apply Subtype.ext
    apply Subtype.ext
    exact h4
  haveI : Module.Finite ℤ B := Module.IsNoetherian.finite ℤ B.toSubmodule
  have hUT : Module.finrank ℤ UT ≤ Module.finrank ℤ B := LinearMap.finrank_le_finrank_of_injective hinc
  have hBW : Module.finrank ℤ UT ≤ Module.finrank ℤ W := hUT.trans hB
  -- `𝕋 = 𝕋^an + u𝕋`
  let Tan' : Submodule ℤ T := Submodule.comap T.val.toLinearMap Tan.toSubmodule
  have hsup : Tan' ⊔ UT = ⊤ := by
    rw [eq_top_iff]
    rintro ⟨t, ht⟩ -
    obtain ⟨s, hs, y, hy, rfl⟩ := hgen t ht
    refine Submodule.mem_sup.mpr ⟨⟨s, hTan hs⟩, hs, ⟨u * y, T.mul_mem hu hy⟩, ⟨⟨y, hy⟩, rfl⟩, ?_⟩
    exact Subtype.ext rfl
  let incT : Tan' →ₗ[ℤ] Tan :=
    { toFun := fun x => ⟨(x : T).1, x.2⟩
      map_add' := fun x y => rfl
      map_smul' := fun r x => rfl }
  have hincT : Function.Injective incT := fun x y h => by
    have h' : ((incT x : Tan) : Module.End ℤ L) = incT y := congrArg Subtype.val h
    exact Subtype.ext (Subtype.ext h')
  have hTanle : Module.finrank ℤ Tan' ≤ Module.finrank ℤ Tan :=
    LinearMap.finrank_le_finrank_of_injective hincT
  have hrk := Submodule.rank_add_le_rank_add_rank Tan' UT
  rw [hsup, rank_top] at hrk
  haveI : Module.Free ℤ T := Module.free_of_finite_type_torsion_free' (R := ℤ)
  haveI : Module.Finite ℤ Tan' := Module.IsNoetherian.finite ℤ Tan'
  haveI : Module.Finite ℤ UT := Module.IsNoetherian.finite ℤ UT
  haveI : Module.Free ℤ Tan' := Module.free_of_finite_type_torsion_free' (R := ℤ)
  haveI : Module.Free ℤ UT := Module.free_of_finite_type_torsion_free' (R := ℤ)
  have hrk' : Module.finrank ℤ T ≤ Module.finrank ℤ Tan' + Module.finrank ℤ UT := by
    rw [← Module.finrank_eq_rank, ← Module.finrank_eq_rank, ← Module.finrank_eq_rank] at hrk
    exact_mod_cast hrk
  omega

/-- **The counting criterion** (`non_reduced_certified`, 01 §4): if `rank 𝕋 > rank 𝕋^an + rank(U^m L)`, then `𝕋` is not reduced. -/
theorem not_isReduced_of_count (T Tan : Subalgebra ℤ (Module.End ℤ L)) (hc : ∀ x ∈ T, ∀ y ∈ T, x * y = y * x)
    (hTan : Tan ≤ T) (u : Module.End ℤ L) (hu : u ∈ T) (hgen : ∀ t ∈ T, ∃ s ∈ Tan, ∃ y ∈ T, t = s + u * y)
    (m : ℕ) (hm : 1 ≤ m)
    (hcount : Module.finrank ℤ Tan + Module.finrank ℤ (LinearMap.range (u ^ m)) < Module.finrank ℤ T) :
    ∃ t ∈ T, t ≠ 0 ∧ IsNilpotent t := by
  by_contra hcon
  push Not at hcon
  have hred : ∀ t ∈ T, IsNilpotent t → t = 0 := fun t ht hnil => by
    by_contra h
    exact hcon t ht h hnil
  exact absurd (finrank_le_of_reduced T Tan hc hTan u hu hgen hred m hm) (by omega)

/-! ### The upper-bound tag (`prop:heckeZcong`(2) for `θ = U_p`) -/

/-- **The upper bound** (01 Prop. 5.2(2)): `f(θ) ∈ I`, `g(θ) ∈ J`, `u f + v g = c` give `c ∈ I + J`: `c` kills `T/(I + J)`,
the congruence module `C_𝕋(S, S')`. Any `θ ∈ T` with `f = f_{θ,S}`, `g = f_{θ,S'}` may be used; `u, v` a Bezout pair. -/
theorem upper_bound_mem {R : Type*} [CommRing R] (I J : Ideal R) (θ : R) (f g u v : ℤ[X]) (c : ℤ)
    (hf : aeval θ f ∈ I) (hg : aeval θ g ∈ J) (h : u * f + v * g = C c) : (c : R) ∈ I ⊔ J := by
  have h1 : (c : R) = aeval θ u * aeval θ f + aeval θ v * aeval θ g := by
    have := congrArg (aeval θ) h
    simpa [map_add, map_mul, aeval_C] using this.symm
  rw [h1]
  exact Ideal.add_mem _ (Ideal.mem_sup_left (Ideal.mul_mem_left _ _ hf)) (Ideal.mem_sup_right (Ideal.mul_mem_left _ _ hg))

end

end GL2.HeckeZ.Count
