/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.Saturation.LevelOne
import GL2.HeckeZ.Cong

/-!
# HeckeZ HZ-0 (the η part, T1 shape): from Eichler–Shimura over `ℤ_ℓ` to an Eisenstein split over `ℤ` (M-HZ1)

`07-lean-plan.md` §2.1 (HZ-0), §7 ("`EichlerShimuraLevelOneZ`: the ℤ-form of the existing structure through HZ-0").
`L` is a free `ℤ`-module of finite rank with Hecke operators `X : ι → End_ℤ L` (interface data: `L = H¹(SL₂(ℤ); Sym^n ℤ)/tors`,
`X i` the `T_i`), `𝕋^fr = ℤ[X i] = Algebra.adjoin ℤ (range X)`. The literature input is the **existing** structure
`GL2.Literature.EichlerShimuraLevelOne` (`lit:eichler-shimura-levelone`), instantiated at `p`, the `ℤ_p`-lattice
`ℤ_p ⊗_ℤ L` with the operators `X i ⊗ ℤ_p`, and a field `E ⊇ ℚ_p`. No new literature row is introduced.

The bridge: `emb : L → V := E ⊗_{ℤ_p} (ℤ_p ⊗_ℤ L)`, `v ↦ 1 ⊗ 1 ⊗ v`, is injective with `ℤ`-coordinates (`repr_emb`), and
`bcH : End_ℤ L → End_E V` is a ring map with `bcH f ∘ emb = emb ∘ f`. In rank `3`, Eichler–Shimura forces **exactly two
systems**, of multiplicities `1` and `2` (`twoSys_of_ES`); a recorded integral characteristic polynomial
`χ(T_q) = (x − u)²(x − w)` (TB-χ) fixes their values at `q` by Cayley–Hamilton and the trace (`ident`); eigenvalues of
elements of `𝕋^fr` on the integral eigenvectors are rational algebraic integers, hence integers (`int_of_eigen`). The
outcome is a `GL2.HeckeZ.EisSplit` of `𝕋^fr` over `ℤ` (`eisSplit_of_ES`), the input of 01 Prop. 5.3 / 5.4.
-/

namespace GL2.HeckeZ.Bridge

open GL2.Literature GL2.Sat TensorProduct Polynomial

set_option linter.unusedSectionVars false

/-- `E ⊇ ℚ_p` has characteristic zero. -/
theorem charZeroE {p : ℕ} [Fact p.Prime] {E : Type*} [Field E] [Algebra ℚ_[p] E] : CharZero E :=
  charZero_of_injective_algebraMap (algebraMap ℚ_[p] E).injective

noncomputable section

variable (p : ℕ) [Fact p.Prime]
variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]
variable (E : Type*) [Field E] [Algebra ℤ_[p] E] [Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]

/-- The operators on the `ℤ_p`-lattice. -/
def Xp {ι : Type*} (X : ι → Module.End ℤ L) : ι → Module.End ℤ_[p] (ℤ_[p] ⊗[ℤ] L) := fun i => (X i).baseChange ℤ_[p]

/-- `f ↦ f ⊗ ℤ_p ⊗ E`, a ring map. -/
noncomputable def bcH : Module.End ℤ L →+* Module.End E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) :=
  (Module.End.baseChangeHom ℤ_[p] E (ℤ_[p] ⊗[ℤ] L)).toRingHom.comp (Module.End.baseChangeHom ℤ ℤ_[p] L).toRingHom

lemma bcH_apply (f : Module.End ℤ L) : bcH p E f = (f.baseChange ℤ_[p]).baseChange E := rfl

/-- `v ↦ 1 ⊗ 1 ⊗ v`. -/
noncomputable def emb : L →+ E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L) :=
  ((TensorProduct.mk ℤ_[p] E (ℤ_[p] ⊗[ℤ] L) 1).toAddMonoidHom).comp (TensorProduct.mk ℤ ℤ_[p] L 1).toAddMonoidHom

lemma emb_apply (v : L) : emb p E v = (1 : E) ⊗ₜ ((1 : ℤ_[p]) ⊗ₜ v) := rfl

lemma bcH_emb (f : Module.End ℤ L) (v : L) : bcH p E f (emb p E v) = emb p E (f v) := by
  simp [bcH_apply, emb_apply, LinearMap.baseChange_tmul]

lemma emb_zsmul (n : ℤ) (v : L) : emb p E (n • v) = (n : E) • emb p E v := by
  rw [map_zsmul]; exact (Int.cast_smul_eq_zsmul E n _).symm


/-- The `E`-basis of `V` induced by a `ℤ`-basis of `L`. -/
noncomputable def bE {κ : Type*} (b : Module.Basis κ ℤ L) : Module.Basis κ E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) :=
  Algebra.TensorProduct.basis E (Algebra.TensorProduct.basis ℤ_[p] b)

lemma repr_emb {κ : Type*} (b : Module.Basis κ ℤ L) (v : L) (i : κ) :
    (bE p E b).repr (emb p E v) i = ((b.repr v i : ℤ) : E) := by
  simp [bE, emb_apply, Algebra.TensorProduct.basis_repr_tmul]

lemma bE_apply {κ : Type*} (b : Module.Basis κ ℤ L) (i : κ) : bE p E b i = emb p E (b i) := by
  simp [bE, emb_apply, Algebra.TensorProduct.basis_apply]

lemma emb_injective : Function.Injective (emb p E (L := L)) := by
  haveI := charZeroE (p := p) (E := E)
  intro v w h
  let b := Module.Free.chooseBasis ℤ L
  refine b.ext_elem fun i => ?_
  have := congrArg (fun x => (bE p E b).repr x i) h
  simp only [repr_emb] at this
  exact_mod_cast this

lemma emb_ne_zero {v : L} (hv : v ≠ 0) : emb p E v ≠ 0 := by
  intro h; exact hv (emb_injective p E (by rw [h, map_zero]))

lemma bcH_eq_zero {g : Module.End ℤ L} (h : bcH p E g = 0) : g = 0 := by
  ext v
  apply emb_injective p E
  rw [← bcH_emb, h, LinearMap.zero_apply, LinearMap.zero_apply, map_zero]

lemma bcH_intCast_apply (n : ℤ) (w : E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) :
    bcH p E (n : Module.End ℤ L) w = (n : E) • w := by
  rw [map_intCast, Module.End.intCast_apply]; exact (Int.cast_smul_eq_zsmul E n w).symm

instance : FiniteDimensional E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) :=
  Module.Finite.of_basis (bE p E (Module.Free.chooseBasis ℤ L))

lemma finrank_V : Module.finrank E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) = Module.finrank ℤ L := by
  rw [Module.finrank_eq_card_basis (bE p E (Module.Free.chooseBasis ℤ L)),
    Module.finrank_eq_card_chooseBasisIndex]

/-! ### Two systems -/

variable {ι : Type*} (X : ι → Module.End ℤ L)

/-- Shorthand for the joint eigenspace of the system `χ`. -/
abbrev W (χ : ι → E) : Submodule E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) := jointEigen (Xp p X) E χ

/-- The two-system decomposition of `V` (rank three, level one): `V = W_e ⊕ W_c`, `dim W_e = 1`, `dim W_c = 2`. -/
structure TwoSys where
  χe : ι → E
  χc : ι → E
  hne : χe ≠ χc
  de : Module.finrank E (W p E X χe) = 1
  dc : Module.finrank E (W p E X χc) = 2
  top : W p E X χe ⊔ W p E X χc = ⊤
  disj : Disjoint (W p E X χe) (W p E X χc)

/-- **Eichler–Shimura in rank three forces two systems, of multiplicities `1` and `2`.** -/
theorem twoSys_of_ES (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (Xp p X) E) (hrk : Module.finrank ℤ L = 3) :
    Nonempty (TwoSys p E X) := by
  classical
  obtain ⟨S, mult, hm, hcard, hdim, htop⟩ := hES.decomp
  have hind := jointEigen_indep E (Xp p X) S htop
  have hsum : ∑ χ ∈ S, mult χ = 3 := by
    have := finrank_biSup_indep E (fun χ => jointEigen (Xp p X) E χ) hind S
    rw [htop, finrank_top, finrank_V, hrk] at this
    rw [this]; exact Finset.sum_congr rfl fun χ hχ => (hdim χ hχ).symm
  rw [← Finset.sum_filter_add_sum_filter_not S (fun χ => mult χ = 1)] at hsum
  have e1 : ∑ χ ∈ S.filter (fun χ => mult χ = 1), mult χ = (S.filter (fun χ => mult χ = 1)).card := by
    rw [Finset.card_eq_sum_ones]; exact Finset.sum_congr rfl fun χ hχ => (Finset.mem_filter.1 hχ).2
  have e2 : ∑ χ ∈ S.filter (fun χ => ¬ mult χ = 1), mult χ = 2 * (S.filter (fun χ => ¬ mult χ = 1)).card := by
    rw [Finset.card_eq_sum_ones, Finset.mul_sum]
    refine Finset.sum_congr rfl fun χ hχ => ?_
    have := Finset.mem_filter.1 hχ
    rcases hm χ this.1 with h | h
    · exact absurd h this.2
    · simp [h]
  rw [e1, e2] at hsum
  have hc1 : (S.filter (fun χ => mult χ = 1)).card = 1 := by omega
  have hc2 : (S.filter (fun χ => ¬ mult χ = 1)).card = 1 := by omega
  obtain ⟨χe, he⟩ := Finset.card_eq_one.1 hc1
  obtain ⟨χc, hc⟩ := Finset.card_eq_one.1 hc2
  have he' : χe ∈ S ∧ mult χe = 1 := Finset.mem_filter.1 (he ▸ Finset.mem_singleton_self χe)
  have hc' : χc ∈ S ∧ ¬ mult χc = 1 := Finset.mem_filter.1 (hc ▸ Finset.mem_singleton_self χc)
  have hc2' : mult χc = 2 := (hm χc hc'.1).resolve_left hc'.2
  have hne : χe ≠ χc := fun h => hc'.2 (h ▸ he'.2)
  have hS : S = {χe, χc} := by
    rw [← Finset.filter_union_filter_not_eq (p := fun χ => mult χ = 1) S, he, hc]; rfl
  rw [hS, Finset.iSup_insert, Finset.iSup_singleton] at htop
  exact ⟨⟨χe, χc, hne, by rw [hdim χe he'.1, he'.2], by rw [hdim χc hc'.1, hc2'], htop,
    hind.pairwiseDisjoint hne⟩⟩

variable {p E X}

lemma act_gen {χ : ι → E} {w : E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)} (hw : w ∈ W p E X χ) (i : ι) :
    bcH p E (X i) w = χ i • w :=
  (mem_jointEigen_iff (Xp p X) E).1 hw i

/-- Every element of `𝕋^fr = ℤ[X i]` acts on each joint eigenspace by a scalar. -/
lemma scalar_adj (χ : ι → E) {t : Module.End ℤ L} (ht : t ∈ Algebra.adjoin ℤ (Set.range X)) :
    ∃ c : E, ∀ w ∈ W p E X χ, bcH p E t w = c • w := by
  induction ht using Algebra.adjoin_induction with
  | mem x hx =>
    obtain ⟨i, rfl⟩ := hx
    exact ⟨χ i, fun w hw => act_gen hw i⟩
  | algebraMap r =>
    refine ⟨(r : E), fun w _ => ?_⟩
    rw [algebraMap_int_eq, eq_intCast, bcH_intCast_apply]
  | add x y _ _ hx hy =>
    obtain ⟨c, hc⟩ := hx
    obtain ⟨d, hd⟩ := hy
    exact ⟨c + d, fun w hw => by rw [map_add, LinearMap.add_apply, hc w hw, hd w hw, add_smul]⟩
  | mul x y _ _ hx hy =>
    obtain ⟨c, hc⟩ := hx
    obtain ⟨d, hd⟩ := hy
    refine ⟨c * d, fun w hw => ?_⟩
    rw [map_mul, Module.End.mul_apply, hd w hw, LinearMap.map_smul, hc w hw, smul_smul, mul_comm]

lemma ne_bot_of_finrank {Wx : Submodule E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L))} {k : ℕ} (h : Module.finrank E Wx = k)
    (hk : k ≠ 0) : Wx ≠ ⊥ := by
  rintro rfl; rw [finrank_bot] at h; exact hk h.symm

/-- Cayley–Hamilton on an eigenvector: an eigenvalue of `f ⊗ E` is a root of `χ_ℤ(f)`. -/
lemma aeval_charpoly_eq_zero (f : Module.End ℤ L) {a : E} {x : E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)}
    (hx : bcH p E f x = a • x) (hx0 : x ≠ 0) : aeval a f.charpoly = 0 := by
  have h0 : aeval (bcH p E f) f.charpoly = 0 := by
    rw [show bcH p E f = (bcH p E).toIntAlgHom f from rfl, Polynomial.aeval_algHom_apply,
      LinearMap.aeval_self_charpoly, map_zero]
  rw [← aeval_map_algebraMap E] at h0
  have hev : (bcH p E f).HasEigenvector a x := ⟨Module.End.mem_eigenspace_iff.2 hx, hx0⟩
  have := Module.End.aeval_apply_of_hasEigenvector (p := f.charpoly.map (algebraMap ℤ E)) hev
  rw [h0, LinearMap.zero_apply, eq_comm, smul_eq_zero] at this
  rw [← eval_map_algebraMap]
  exact this.resolve_right hx0

/-- `trace_ℤ f = 2u + w` from `χ_ℤ(f) = (x − u)²(x − w)` in rank three. -/
lemma trace_of_charpoly (hrk : Module.finrank ℤ L = 3) (f : Module.End ℤ L) (u w : ℤ)
    (hχ : f.charpoly = (Polynomial.X - C u) ^ 2 * (Polynomial.X - C w)) : LinearMap.trace ℤ L f = 2 * u + w := by
  classical
  let b := Module.Free.chooseBasis ℤ L
  have hcard : Fintype.card (Module.Free.ChooseBasisIndex ℤ L) = 3 := by
    rw [← Module.finrank_eq_card_chooseBasisIndex, hrk]
  haveI : Nonempty (Module.Free.ChooseBasisIndex ℤ L) := Fintype.card_pos_iff.1 (by omega)
  rw [LinearMap.trace_eq_matrix_trace ℤ b, Matrix.trace_eq_neg_charpoly_coeff, LinearMap.charpoly_toMatrix,
    hcard, hχ]
  have : ((Polynomial.X - C u) ^ 2 * (Polynomial.X - C w) : ℤ[X]) =
      Polynomial.X ^ 3 + C (-(2 * u + w)) * Polynomial.X ^ 2 + C (u ^ 2 + 2 * u * w) * Polynomial.X + C (-(u ^ 2 * w)) := by
    simp only [map_neg, map_add, map_mul, map_pow, map_ofNat]; ring
  rw [this]; simp only [coeff_add, coeff_C_mul_X_pow, coeff_C_mul_X, coeff_X_pow, coeff_C]; norm_num

variable (D : TwoSys p E X)

lemma trace_E (f : Module.End ℤ L) (a c : E) (ha : ∀ w ∈ W p E X D.χe, bcH p E f w = a • w)
    (hc : ∀ w ∈ W p E X D.χc, bcH p E f w = c • w) : ((LinearMap.trace ℤ L f : ℤ) : E) = a + 2 * c := by
  classical
  have h1 : LinearMap.trace E _ (bcH p E f) = ((LinearMap.trace ℤ L f : ℤ) : E) := by
    rw [bcH_apply, LinearMap.trace_baseChange, LinearMap.trace_baseChange]
    simp [algebraMap_int_eq]
  rw [← h1]
  let N : Fin 2 → Submodule E (E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)) := ![W p E X D.χe, W p E X D.χc]
  have hint : DirectSum.IsInternal N := by
    rw [DirectSum.isInternal_submodule_iff_isCompl N (i := 0) (j := 1) (by decide)
      (by ext k; fin_cases k <;> simp)]
    exact ⟨D.disj, codisjoint_iff.mpr D.top⟩
  have hf : ∀ i, Set.MapsTo (bcH p E f) (N i) (N i) := by
    intro i; fin_cases i
    · intro w hw; simp only [N] at hw ⊢; rw [ha w hw]; exact Submodule.smul_mem _ _ hw
    · intro w hw; simp only [N] at hw ⊢; rw [hc w hw]; exact Submodule.smul_mem _ _ hw
  haveI : ∀ i, Module.Free E (N i) := fun i => Module.Free.of_basis (Module.Basis.ofVectorSpace E (N i))
  haveI : ∀ i, Module.Finite E (N i) := fun i => inferInstance
  rw [LinearMap.trace_eq_sum_trace_restrict hint hf, Fin.sum_univ_two]
  have r0 : (bcH p E f).restrict (hf 0) = a • LinearMap.id := by
    ext w; simp only [LinearMap.coe_restrict_apply, LinearMap.smul_apply, LinearMap.id_apply, Submodule.coe_smul]
    exact ha w w.2
  have r1 : (bcH p E f).restrict (hf 1) = c • LinearMap.id := by
    ext w; simp only [LinearMap.coe_restrict_apply, LinearMap.smul_apply, LinearMap.id_apply, Submodule.coe_smul]
    exact hc w w.2
  have d0 : Module.finrank E (N 0) = 1 := D.de
  have d1 : Module.finrank E (N 1) = 2 := D.dc
  have t0 : LinearMap.trace E (N 0) LinearMap.id = (Module.finrank E (N 0) : E) := by
    convert LinearMap.trace_id E (N 0)
  have t1 : LinearMap.trace E (N 1) LinearMap.id = (Module.finrank E (N 1) : E) := by
    convert LinearMap.trace_id E (N 1)
  rw [r0, r1, map_smul, map_smul, t0, t1, d0, d1]; simp only [smul_eq_mul]; push_cast; ring

/-- **The values of the two systems at `f`** from `χ_ℤ(f) = (x − u)²(x − w)`, `u ≠ w`: the Eisenstein (simple)
system takes `w`, the double one `u`. -/
lemma ident (hrk : Module.finrank ℤ L = 3) (f : Module.End ℤ L) (u w : ℤ) (huw : u ≠ w)
    (hχ : f.charpoly = (Polynomial.X - C u) ^ 2 * (Polynomial.X - C w)) (a c : E)
    (ha : ∀ x ∈ W p E X D.χe, bcH p E f x = a • x) (hc : ∀ x ∈ W p E X D.χc, bcH p E f x = c • x) :
    a = w ∧ c = u := by
  haveI := charZeroE (p := p) (E := E)
  have huw' : (u : E) ≠ (w : E) := by exact_mod_cast huw
  have htr := trace_E D f a c ha hc
  rw [trace_of_charpoly hrk f u w hχ] at htr
  push_cast at htr
  have root : ∀ (Wx : Submodule E _) (k : ℕ), Module.finrank E Wx = k → k ≠ 0 → ∀ d : E,
      (∀ x ∈ Wx, bcH p E f x = d • x) → d = u ∨ d = w := by
    intro Wx k hk hk0 d hd
    obtain ⟨x, hx, hx0⟩ := (Submodule.ne_bot_iff Wx).1 (ne_bot_of_finrank hk hk0)
    have := aeval_charpoly_eq_zero f (hd x hx) hx0
    rw [hχ] at this
    simp only [map_mul, map_pow, map_sub, aeval_X, aeval_C] at this
    simp only [algebraMap_int_eq, eq_intCast] at this
    rcases mul_eq_zero.1 this with h | h
    · exact Or.inl (sub_eq_zero.1 (pow_eq_zero_iff two_ne_zero |>.1 h))
    · exact Or.inr (sub_eq_zero.1 h)
  rcases root _ 1 D.de one_ne_zero a ha with ra | ra <;>
  rcases root _ 2 D.dc two_ne_zero c hc with rc | rc
  · subst ra rc; exact absurd (show (u : E) = w by linear_combination -htr) huw'
  · subst ra rc; exact absurd (show (u : E) = w by linear_combination htr) huw'
  · exact ⟨ra, rc⟩
  · subst ra rc; exact absurd (show (u : E) = w by linear_combination htr / 2) huw'

/-- An eigenvalue of an integral endomorphism on an integral eigenvector is an integer. -/
lemma int_of_eigen (f : Module.End ℤ L) {v : L} (hv : v ≠ 0) {c : E} (h : emb p E (f v) = c • emb p E v) :
    ∃ n : ℤ, (n : E) = c := by
  haveI := charZeroE (p := p) (E := E)
  let b := Module.Free.chooseBasis ℤ L
  obtain ⟨i, hi⟩ : ∃ i, b.repr v i ≠ 0 := by
    by_contra hcon; push Not at hcon; exact hv (b.ext_elem fun i => by simp [hcon i])
  have hco : ((b.repr (f v) i : ℤ) : E) = c * ((b.repr v i : ℤ) : E) := by
    have := congrArg (fun x => (bE p E b).repr x i) h
    simpa only [repr_emb, LinearEquiv.map_smul, Finsupp.smul_apply, smul_eq_mul] using this
  have hi' : ((b.repr v i : ℤ) : E) ≠ 0 := by exact_mod_cast hi
  set q : ℚ := (b.repr (f v) i : ℚ) / (b.repr v i : ℚ) with hq
  have hqc : (q : E) = c := by
    rw [hq, Rat.cast_div, Rat.cast_intCast, Rat.cast_intCast, hco, mul_div_assoc, div_self hi', mul_one]
  have hint : IsIntegral ℤ c := by
    refine ⟨f.charpoly, LinearMap.charpoly_monic f, ?_⟩
    rw [← aeval_def]
    exact aeval_charpoly_eq_zero f (by rw [bcH_emb, h]) (emb_ne_zero p E hv)
  have hintq : IsIntegral ℤ q := by
    rw [← isIntegral_algHom_iff (Rat.castHom E).toIntAlgHom (Rat.castHom E).injective]
    simpa [hqc] using hint
  obtain ⟨y, hy⟩ := (IsIntegrallyClosed.isIntegral_iff (K := ℚ)).1 hintq
  refine ⟨y, ?_⟩
  rw [← hqc, ← hy]; simp

lemma int_eigen_vec (f : Module.End ℤ L) {v : L} (hv : v ≠ 0) {c : E} (h : emb p E (f v) = c • emb p E v) :
    ∃ n : ℤ, (n : E) = c ∧ f v = n • v := by
  obtain ⟨n, hn⟩ := int_of_eigen f hv h
  exact ⟨n, hn, emb_injective p E (by rw [h, emb_zsmul, hn])⟩

/-! ### The integer eigenvalue map on `𝕋^fr` -/

section EvHom

variable {T : Subalgebra ℤ (Module.End ℤ L)} {a0 : L}

lemma ev_uniq (ha0 : a0 ≠ 0) {n m : ℤ} {x : L} (hn : x = n • a0) (hm : x = m • a0) : n = m :=
  smul_left_injective ℤ ha0 (hn.symm.trans hm)

/-- The ring map `t ↦ (the integer by which t acts on a0)`. -/
noncomputable def evHom (ha0 : a0 ≠ 0) (hex : ∀ t : T, ∃ n : ℤ, (t : Module.End ℤ L) a0 = n • a0) : T →+* ℤ where
  toFun t := Classical.choose (hex t)
  map_one' := ev_uniq ha0 (Classical.choose_spec (hex 1)) (by simp)
  map_mul' t s := by
    refine ev_uniq ha0 (Classical.choose_spec (hex (t * s))) ?_
    have hs := Classical.choose_spec (hex s); have ht := Classical.choose_spec (hex t)
    set ns := Classical.choose (hex s); set nt := Classical.choose (hex t)
    rw [Subalgebra.coe_mul, Module.End.mul_apply, hs, map_zsmul, ht, smul_smul, mul_comm]
  map_zero' := ev_uniq ha0 (Classical.choose_spec (hex 0)) (by simp)
  map_add' t s := by
    refine ev_uniq ha0 (Classical.choose_spec (hex (t + s))) ?_
    have hs := Classical.choose_spec (hex s); have ht := Classical.choose_spec (hex t)
    set ns := Classical.choose (hex s); set nt := Classical.choose (hex t)
    rw [Subalgebra.coe_add, LinearMap.add_apply, hs, ht, add_smul]

lemma evHom_spec (ha0 : a0 ≠ 0) (hex : ∀ t : T, ∃ n : ℤ, (t : Module.End ℤ L) a0 = n • a0) (t : T) :
    (t : Module.End ℤ L) a0 = evHom ha0 hex t • a0 :=
  Classical.choose_spec (hex t)

lemma evHom_eq (ha0 : a0 ≠ 0) (hex : ∀ t : T, ∃ n : ℤ, (t : Module.End ℤ L) a0 = n • a0) (t : T) {n : ℤ}
    (h : (t : Module.End ℤ L) a0 = n • a0) : evHom ha0 hex t = n :=
  ev_uniq ha0 (evHom_spec ha0 hex t) h

end EvHom

/-! ### The Eisenstein split -/

/-- `𝕋^fr`. -/
abbrev Tfr (X : ι → Module.End ℤ L) : Subalgebra ℤ (Module.End ℤ L) := Algebra.adjoin ℤ (Set.range X)

/-- The generator `X i` as an element of `𝕋^fr`. -/
def gen (X : ι → Module.End ℤ L) (i : ι) : Tfr X := ⟨X i, Algebra.subset_adjoin ⟨i, rfl⟩⟩

/-- On `V = W_e ⊕ W_c`: an endomorphism that vanishes on both pieces is zero. -/
lemma bcH_zero_of (g : Module.End ℤ L) (he : ∀ w ∈ W p E X D.χe, bcH p E g w = 0)
    (hc : ∀ w ∈ W p E X D.χc, bcH p E g w = 0) : g = 0 := by
  apply bcH_eq_zero p E
  refine LinearMap.ext fun x => ?_
  have hx : x ∈ W p E X D.χe ⊔ W p E X D.χc := D.top ▸ Submodule.mem_top
  obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.1 hx
  rw [map_add, he y hy, hc z hz, add_zero, LinearMap.zero_apply]

/-- A vector of `V` killed by `f − d`, where `f` acts on `W_c` by `c ≠ d`, lies in `W_e`. -/
lemma mem_e_of (f : Module.End ℤ L) (a c d : E) (hcd : c ≠ d) (ha : ∀ w ∈ W p E X D.χe, bcH p E f w = a • w)
    (hc : ∀ w ∈ W p E X D.χc, bcH p E f w = c • w) {x : E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)}
    (hx : bcH p E f x = d • x) (had : a = d) : x ∈ W p E X D.χe := by
  have hx' : x ∈ W p E X D.χe ⊔ W p E X D.χc := D.top ▸ Submodule.mem_top
  obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.1 hx'
  rw [map_add, ha y hy, hc z hz, smul_add, had] at hx
  have hz' : c • z = d • z := add_left_cancel hx
  have h0 : (c - d) • z = 0 := (sub_smul c d z).trans (sub_eq_zero.mpr hz')
  rcases smul_eq_zero.1 h0 with h | h
  · exact absurd (sub_eq_zero.1 h) hcd
  · rw [h, add_zero]; exact hy

lemma mem_c_of (f : Module.End ℤ L) (a c d : E) (had : a ≠ d) (ha : ∀ w ∈ W p E X D.χe, bcH p E f w = a • w)
    (hc : ∀ w ∈ W p E X D.χc, bcH p E f w = c • w) {x : E ⊗[ℤ_[p]] (ℤ_[p] ⊗[ℤ] L)}
    (hx : bcH p E f x = d • x) (hcd : c = d) : x ∈ W p E X D.χc := by
  have hx' : x ∈ W p E X D.χe ⊔ W p E X D.χc := D.top ▸ Submodule.mem_top
  obtain ⟨y, hy, z, hz, rfl⟩ := Submodule.mem_sup.1 hx'
  rw [map_add, ha y hy, hc z hz, smul_add, hcd] at hx
  have hy' : a • y = d • y := add_right_cancel hx
  have h0 : (a - d) • y = 0 := (sub_smul a d y).trans (sub_eq_zero.mpr hy')
  rcases smul_eq_zero.1 h0 with h | h
  · exact absurd (sub_eq_zero.1 h) had
  · rw [h, zero_add]; exact hz

/-- **HZ-0 for T1** (07 §7): Eichler–Shimura at level one (`hES`, over `ℤ_p` and `E ⊇ ℚ_p`), rank `3`, and the two
recorded integral characteristic polynomials (TB-χ) `χ(X i₂) = (x − u₂)²(x − w₂)`, `χ(X i₃) = (x − u₃)²(x − w₃)` give an
Eisenstein split of `𝕋^fr = ℤ[X i]` over `ℤ` with `θ = X i₂`, `λ(X i_q) = w_q` (the simple, Eisenstein root) and
`μ(X i_q) = u_q` (the double, cuspidal root). -/
theorem eisSplit_of_ES (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (Xp p X) E) (hrk : Module.finrank ℤ L = 3)
    (i₂ i₃ : ι) (u₂ w₂ u₃ w₃ : ℤ) (h₂ : u₂ ≠ w₂) (h₃ : u₃ ≠ w₃)
    (hχ₂ : (X i₂).charpoly = (Polynomial.X - C u₂) ^ 2 * (Polynomial.X - C w₂))
    (hχ₃ : (X i₃).charpoly = (Polynomial.X - C u₃) ^ 2 * (Polynomial.X - C w₃)) :
    ∃ Es : EisSplit (Tfr X), Es.θ = gen X i₂ ∧ Es.lam (gen X i₂) = w₂ ∧ Es.mu (gen X i₂) = u₂ ∧
      Es.lam (gen X i₃) = w₃ ∧ Es.mu (gen X i₃) = u₃ := by
  haveI := charZeroE (p := p) (E := E)
  obtain ⟨D⟩ := twoSys_of_ES p E X hES hrk
  -- the values of the two systems at `X i₂`, `X i₃`
  obtain ⟨e2, c2⟩ := ident D hrk (X i₂) u₂ w₂ h₂ hχ₂ _ _ (fun x hx => act_gen hx i₂) (fun x hx => act_gen hx i₂)
  obtain ⟨e3, c3⟩ := ident D hrk (X i₃) u₃ w₃ h₃ hχ₃ _ _ (fun x hx => act_gen hx i₃) (fun x hx => act_gen hx i₃)
  have huw : (u₂ : E) ≠ (w₂ : E) := by exact_mod_cast h₂
  -- `P = X i₂ − u₂`, `Q = X i₂ − w₂`
  set Pm : Module.End ℤ L := X i₂ - (u₂ : Module.End ℤ L) with hPm
  set Qm : Module.End ℤ L := X i₂ - (w₂ : Module.End ℤ L) with hQm
  have bcP : ∀ x, bcH p E Pm x = bcH p E (X i₂) x - (u₂ : E) • x := fun x => by
    rw [hPm, map_sub, LinearMap.sub_apply, bcH_intCast_apply]
  have bcQ : ∀ x, bcH p E Qm x = bcH p E (X i₂) x - (w₂ : E) • x := fun x => by
    rw [hQm, map_sub, LinearMap.sub_apply, bcH_intCast_apply]
  have QP : Qm * Pm = 0 := by
    refine bcH_zero_of D _ (fun w hw => ?_) (fun w hw => ?_)
    · have hq : bcH p E Qm w = 0 := by rw [bcQ, act_gen hw, e2, sub_self]
      rw [map_mul, Module.End.mul_apply, bcP, act_gen hw, map_sub, map_smul, map_smul, hq]; simp
    · rw [map_mul, Module.End.mul_apply, bcP, act_gen hw, c2, sub_self, map_zero]
  have PQ : Pm * Qm = 0 := by
    have hcm : Commute Pm Qm := by
      rw [hPm, hQm]
      exact ((Commute.refl (X i₂)).sub_right (Int.cast_commute w₂ _).symm).sub_left
        ((Int.cast_commute u₂ (X i₂)).sub_right (Int.cast_commute u₂ _))
    rw [hcm.eq, QP]
  -- a nonzero integral vector in each piece
  let b := Module.Free.chooseBasis ℤ L
  have exP : ∃ j, Pm (b j) ≠ 0 := by
    by_contra hcon; push Not at hcon
    have h0 : bcH p E Pm = 0 := (bE p E b).ext fun j => by
      rw [bE_apply, bcH_emb, hcon j, map_zero, LinearMap.zero_apply]
    obtain ⟨x, hx, hx0⟩ := (Submodule.ne_bot_iff _).1 (ne_bot_of_finrank D.de one_ne_zero)
    have := congrArg (fun g => g x) h0
    simp only [LinearMap.zero_apply] at this
    rw [bcP, act_gen hx, e2] at this
    have h' : ((w₂ : E) - u₂) • x = 0 := (sub_smul _ _ x).trans this
    rcases smul_eq_zero.1 h' with h | h
    · exact huw (sub_eq_zero.1 h).symm
    · exact hx0 h
  have exQ : ∃ j, Qm (b j) ≠ 0 := by
    by_contra hcon; push Not at hcon
    have h0 : bcH p E Qm = 0 := (bE p E b).ext fun j => by
      rw [bE_apply, bcH_emb, hcon j, map_zero, LinearMap.zero_apply]
    obtain ⟨x, hx, hx0⟩ := (Submodule.ne_bot_iff _).1 (ne_bot_of_finrank D.dc two_ne_zero)
    have := congrArg (fun g => g x) h0
    simp only [LinearMap.zero_apply] at this
    rw [bcQ, act_gen hx, c2] at this
    have h' : ((u₂ : E) - w₂) • x = 0 := (sub_smul _ _ x).trans this
    rcases smul_eq_zero.1 h' with h | h
    · exact huw (sub_eq_zero.1 h)
    · exact hx0 h
  obtain ⟨j, hj⟩ := exP
  obtain ⟨j', hj'⟩ := exQ
  set a0 := Pm (b j) with ha0d
  set b0 := Qm (b j') with hb0d
  have hQa0 : Qm a0 = 0 := by rw [ha0d, ← Module.End.mul_apply, QP, LinearMap.zero_apply]
  have hPb0 : Pm b0 = 0 := by rw [hb0d, ← Module.End.mul_apply, PQ, LinearMap.zero_apply]
  -- integral vectors killed by `Q` (resp. `P`) lie in `W_e` (resp. `W_c`)
  have memE : ∀ a : L, Qm a = 0 → emb p E a ∈ W p E X D.χe := by
    intro a ha
    refine mem_e_of D (X i₂) _ _ (w₂ : E) (by rw [c2]; exact huw) (fun x hx => act_gen hx i₂)
      (fun x hx => act_gen hx i₂) ?_ e2
    have := congrArg (emb p E) ha
    rw [map_zero, ← bcH_emb, bcQ, sub_eq_zero] at this; exact this
  have memC : ∀ a : L, Pm a = 0 → emb p E a ∈ W p E X D.χc := by
    intro a ha
    refine mem_c_of D (X i₂) _ _ (u₂ : E) (by rw [e2]; exact huw.symm) (fun x hx => act_gen hx i₂)
      (fun x hx => act_gen hx i₂) ?_ c2
    have := congrArg (emb p E) ha
    rw [map_zero, ← bcH_emb, bcP, sub_eq_zero] at this; exact this
  -- integer eigenvalues on `a0`, `b0`
  have hexA : ∀ t : Tfr X, ∃ n : ℤ, (t : Module.End ℤ L) a0 = n • a0 := by
    intro t
    obtain ⟨c, hc⟩ := scalar_adj (p := p) D.χe t.2
    obtain ⟨n, -, hn⟩ := int_eigen_vec (p := p) (E := E) (t : Module.End ℤ L) hj (by rw [← bcH_emb, hc _ (memE a0 hQa0)])
    exact ⟨n, hn⟩
  have hexB : ∀ t : Tfr X, ∃ n : ℤ, (t : Module.End ℤ L) b0 = n • b0 := by
    intro t
    obtain ⟨c, hc⟩ := scalar_adj (p := p) D.χc t.2
    obtain ⟨n, -, hn⟩ := int_eigen_vec (p := p) (E := E) (t : Module.End ℤ L) hj' (by rw [← bcH_emb, hc _ (memC b0 hPb0)])
    exact ⟨n, hn⟩
  let lam := evHom hj hexA
  let mu := evHom hj' hexB
  -- every `t` acts on `ker Q` by `λ t`, on `ker P` by `μ t`
  have actA : ∀ (t : Tfr X) (a : L), Qm a = 0 → (t : Module.End ℤ L) a = lam t • a := by
    intro t a ha
    obtain ⟨c, hc⟩ := scalar_adj (p := p) D.χe t.2
    have h0 := hc _ (memE a0 hQa0)
    rw [bcH_emb, evHom_spec hj hexA t, emb_zsmul] at h0
    have hcl : c = (lam t : E) := by
      have h1 : ((lam t : E) - c) • emb p E a0 = 0 := (sub_smul (lam t : E) c (emb p E a0)).trans (sub_eq_zero.mpr h0)
      exact (sub_eq_zero.1 ((smul_eq_zero.1 h1).resolve_right (emb_ne_zero p E hj))).symm
    apply emb_injective p E
    rw [← bcH_emb, hc _ (memE a ha), hcl, emb_zsmul]
  have actB : ∀ (t : Tfr X) (a : L), Pm a = 0 → (t : Module.End ℤ L) a = mu t • a := by
    intro t a ha
    obtain ⟨c, hc⟩ := scalar_adj (p := p) D.χc t.2
    have h0 := hc _ (memC b0 hPb0)
    rw [bcH_emb, evHom_spec hj' hexB t, emb_zsmul] at h0
    have hcl : c = (mu t : E) := by
      have h1 : ((mu t : E) - c) • emb p E b0 = 0 := (sub_smul (mu t : E) c (emb p E b0)).trans (sub_eq_zero.mpr h0)
      exact (sub_eq_zero.1 ((smul_eq_zero.1 h1).resolve_right (emb_ne_zero p E hj'))).symm
    apply emb_injective p E
    rw [← bcH_emb, hc _ (memC a ha), hcl, emb_zsmul]
  -- the values at `X i₂`, `X i₃`
  have l2 : lam (gen X i₂) = w₂ := evHom_eq hj hexA _ (by
    have := hQa0; rw [hQm, LinearMap.sub_apply, Module.End.intCast_apply, sub_eq_zero] at this; exact this)
  have m2 : mu (gen X i₂) = u₂ := evHom_eq hj' hexB _ (by
    have := hPb0; rw [hPm, LinearMap.sub_apply, Module.End.intCast_apply, sub_eq_zero] at this; exact this)
  have l3 : lam (gen X i₃) = w₃ := evHom_eq hj hexA _ (by
    apply emb_injective p E
    rw [show ((gen X i₃ : Tfr X) : Module.End ℤ L) = X i₃ from rfl, ← bcH_emb, act_gen (memE a0 hQa0), e3,
      emb_zsmul])
  have m3 : mu (gen X i₃) = u₃ := evHom_eq hj' hexB _ (by
    apply emb_injective p E
    rw [show ((gen X i₃ : Tfr X) : Module.End ℤ L) = X i₃ from rfl, ← bcH_emb, act_gen (memC b0 hPb0), c3,
      emb_zsmul])
  have hθ : ((gen X i₂ : Tfr X) : Module.End ℤ L) = X i₂ := rfl
  refine ⟨{ lam := lam, mu := mu, θ := gen X i₂, hD := ?_, hQP := ?_, hA := ?_, hB := ?_ }, rfl, l2, m2, l3, m3⟩
  · rw [l2, m2]; exact h₂.symm
  · intro v
    rw [l2, m2, hθ]
    have := congrArg (fun g => g v) QP
    simpa [hPm, hQm, Module.End.mul_apply, Module.End.intCast_apply] using this
  · intro t a ha
    rw [l2, hθ] at ha
    exact actA t a (by rw [hQm, LinearMap.sub_apply, Module.End.intCast_apply]; exact ha)
  · intro t a ha
    rw [m2, hθ] at ha
    exact actB t a (by rw [hPm, LinearMap.sub_apply, Module.End.intCast_apply]; exact ha)

end

end GL2.HeckeZ.Bridge
