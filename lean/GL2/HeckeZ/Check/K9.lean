/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.K9
import GL2.HeckeZ.Check.K10
import Mathlib.RingTheory.Polynomial.Eisenstein.Basic
import Mathlib.RingTheory.Polynomial.GaussLemma
import Mathlib.RingTheory.Polynomial.Quotient
import Mathlib.RingTheory.DiscreteValuationRing.TFAE
import Mathlib.RingTheory.AdjoinRoot
import Mathlib.Algebra.Polynomial.Taylor
import Mathlib.Algebra.Group.Irreducible.Lemmas

/-!
# L-K9: K9 is sound, and the finite-weight theorem (07 §5.9 "soundness, now"; [U] `prop:u-H12`(4), `prop:u-H11`)

`docs/implementation/slot24/heckez/07-lean-plan.md` §5.9; brief `slot25/LK9_brief.md`.

* **`eis_finite_weight`** (pure mathematics, no literature input). Let `F ∈ ℤ_p[x]` be monic of degree `g ≥ 1` with
  `F ≡ (x − e)^g (mod p)` and `p ∥ F(e)`. Then `F(x + e)` is Eisenstein at `(p)`; `F` is irreducible over `ℤ_p` and over `ℚ_p`;
  `A = ℤ_p[x]/(F)` is a domain, of `ℤ_p`-rank (`finrank`) `g`, local with maximal ideal `(x − e)`, a **discrete valuation ring**,
  and the `T₂`-Eisenstein quotient `A/(x − e) ≅ ℤ/p`. (This quotient equals [U] `def:u-congmod`'s `C_Eis,k` only given the generator
  statement `I⁰ = (T₂ − e)` ([U] `prop:u-H11`, `prop:u-H12`(2); Emerton; L-H0); it is not proved here.) (Total ramification and Gorenstein are standard consequences of these facts and
  are not stated in Lean.)
* **`k9_sound`**: an accepted K9 certificate gives the guards, `e = 1 + 2^{k−1}` in `ℤ/ℓ²`, `f ≡ (x − e)^g (mod ℓ)`, and for
  **every** lift to `ℤ_ℓ` of the residues: `ℓ ∥ F̃(ẽ)`, `ℓ ∥ b̃` (so `v(f_k(e_k)) = v(B_k/k) = 1`), for `g = 2` `ℓ ∥ disc F̃`,
  `G̃_k ≡ −b̃ (mod ℓ²)` and `ℓ ∥ G̃_k − G̃_{k+ℓ−1}` (the λ-test).
* **`k9_cert`**: an accepted certificate and **every** monic lift `F̃ ∈ ℤ_ℓ[x]` of degree `g` of `f_k mod ℓ²` satisfy
  `eis_finite_weight` at `e_k = 1 + 2^{k−1} ∈ ℤ_ℓ`.

What the kernel does **not** derive: that `F̃` is the characteristic polynomial of `T₂` on the ordinary Eisenstein block (TB-χ,
block form) and that `bk`, `gnext` are `B_k/k`, `−B_{k+ℓ−1}/(k+ℓ−1)` mod `ℓ²` (TB-B). The theorems quantify over every lift of the
checked residues, so neither enters as a hypothesis; the reading of the conclusions as statements about Hecke algebras rests on them.
-/

namespace GL2.HeckeZ.K9

open Polynomial GL2.HeckeZ.Exec.K9 GL2.HeckeZ.Exec.K10 GL2.HeckeZ.K10 IsLocalRing

/-! ### The finite-weight theorem -/

section FiniteWeight

variable {p : ℕ} [Fact p.Prime]

lemma maxIdeal_eq : maximalIdeal ℤ_[p] = Ideal.span {(p : ℤ_[p])} := PadicInt.maximalIdeal_eq_span_p

lemma mem_maxIdeal_iff (x : ℤ_[p]) : x ∈ maximalIdeal ℤ_[p] ↔ PadicInt.toZMod x = 0 := by
  rw [← PadicInt.ker_toZMod, RingHom.mem_ker]

lemma map_taylor_toZMod (F : ℤ_[p][X]) (e : ℤ_[p]) :
    (taylor e F).map PadicInt.toZMod = taylor (PadicInt.toZMod e) (F.map PadicInt.toZMod) := by
  simp [taylor_apply, Polynomial.map_comp]

/-- The conclusions of the finite-weight theorem for `F` at `e` (07 §5.9). -/
structure EisFiniteWeight (F : ℤ_[p][X]) (e : ℤ_[p]) : Prop where
  /-- `F(x + e)` is Eisenstein at `(p)` -/
  eisenstein : (taylor e F).IsEisensteinAt (maximalIdeal ℤ_[p])
  /-- `F` is irreducible over `ℤ_p` -/
  irreducible : Irreducible F
  /-- `F` is irreducible over `ℚ_p` (the `g` roots are conjugate) -/
  irreducible_Qp : Irreducible (F.map (algebraMap ℤ_[p] ℚ_[p]))
  /-- `ℤ_p[x]/(F)` has `ℤ_p`-rank (`finrank`) `g = deg F` (freeness is in the proof, via the power basis, not a field) -/
  rank : Module.finrank ℤ_[p] (AdjoinRoot F) = F.natDegree
  /-- every maximal ideal of `ℤ_p[x]/(F)` is `(x − e)` -/
  maximal : ∀ M : Ideal (AdjoinRoot F), M.IsMaximal → M = Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e}
  /-- `ℤ_p[x]/(F)` is a discrete valuation ring -/
  dvr : ∃ _ : IsDomain (AdjoinRoot F), IsDiscreteValuationRing (AdjoinRoot F)
  /-- the `T₂`-Eisenstein quotient `ℤ_p[x]/(F, x − e) ≅ ℤ/p` (= [U]'s `C_Eis,k` only given `I⁰ = (T₂ − e)`, L-H0) -/
  cong : Nonempty ((AdjoinRoot F ⧸ Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e}) ≃+* ZMod p)

variable {F : ℤ_[p][X]} {e : ℤ_[p]}

lemma taylor_shape (hshape : F.map PadicInt.toZMod = (X - C (PadicInt.toZMod e)) ^ F.natDegree) :
    (taylor e F).map PadicInt.toZMod = X ^ F.natDegree := by
  rw [map_taylor_toZMod, hshape, taylor_pow, map_sub, taylor_X, taylor_C]; simp

lemma taylor_monic (hm : F.Monic) : (taylor e F).Monic := by
  unfold Monic; rw [leadingCoeff_taylor]; exact hm

theorem eis_taylor (hm : F.Monic)
    (hshape : F.map PadicInt.toZMod = (X - C (PadicInt.toZMod e)) ^ F.natDegree)
    (hval2 : ¬ (p : ℤ_[p]) ^ 2 ∣ F.eval e) : (taylor e F).IsEisensteinAt (maximalIdeal ℤ_[p]) := by
  refine (taylor_monic hm).isEisensteinAt_of_mem_of_notMem (maximalIdeal.isMaximal _).ne_top ?_ ?_
  · intro n hn
    rw [natDegree_taylor] at hn
    rw [mem_maxIdeal_iff, ← coeff_map, taylor_shape hshape, coeff_X_pow, if_neg (by omega)]
  · rw [taylor_coeff_zero, maxIdeal_eq, Ideal.span_singleton_pow, Ideal.mem_span_singleton]; exact hval2

theorem eis_irreducible (hm : F.Monic) (hg : 0 < F.natDegree)
    (hshape : F.map PadicInt.toZMod = (X - C (PadicInt.toZMod e)) ^ F.natDegree)
    (hval2 : ¬ (p : ℤ_[p]) ^ 2 ∣ F.eval e) : Irreducible F := by
  have h1 : Irreducible (taylor e F) :=
    (eis_taylor hm hshape hval2).irreducible (maximalIdeal.isMaximal _).isPrime (taylor_monic hm).isPrimitive
      (by rwa [natDegree_taylor])
  have h2 : Irreducible ((taylorEquiv e).toMulEquiv F) := h1
  exact (MulEquiv.irreducible_iff (f := (taylorEquiv e).toMulEquiv) (x := F)).mp h2

/-- `F(e) = p · u` with `u` a unit. -/
lemma eval_eq_p_mul_unit (hval : (p : ℤ_[p]) ∣ F.eval e) (hval2 : ¬ (p : ℤ_[p]) ^ 2 ∣ F.eval e) :
    ∃ u : ℤ_[p], IsUnit u ∧ F.eval e = p * u := by
  obtain ⟨u, hu⟩ := hval
  refine ⟨u, ?_, hu⟩
  by_contra hnu
  have : u ∈ maximalIdeal ℤ_[p] := hnu
  rw [maxIdeal_eq, Ideal.mem_span_singleton] at this
  obtain ⟨w, hw⟩ := this
  exact hval2 ⟨w, by rw [hu, hw]; ring⟩

/-- `F − (x − e)^g` has all coefficients divisible by `p`. -/
lemma sub_shape_dvd (hshape : F.map PadicInt.toZMod = (X - C (PadicInt.toZMod e)) ^ F.natDegree) :
    C (p : ℤ_[p]) ∣ F - (X - C e) ^ F.natDegree := by
  rw [C_dvd_iff_dvd_coeff]
  intro i
  have h0 : PadicInt.toZMod ((F - (X - C e) ^ F.natDegree).coeff i) = 0 := by
    rw [← coeff_map, Polynomial.map_sub, hshape, Polynomial.map_pow, Polynomial.map_sub, map_X, map_C, sub_self,
      coeff_zero]
  have := (mem_maxIdeal_iff _).mpr h0
  rwa [maxIdeal_eq, Ideal.mem_span_singleton] at this

/-- The ideal `(F, x − e)` of `ℤ_p[x]` is `(C F(e), x − e)`. -/
lemma span_F_sup (F : ℤ_[p][X]) (e : ℤ_[p]) :
    Ideal.span {F} ⊔ Ideal.span {X - C e} = Ideal.span {C (F.eval e), X - C e} := by
  obtain ⟨Q, hQ⟩ := X_sub_C_dvd_sub_C_eval (p := F) (a := e)
  generalize C (F.eval e) = c at hQ ⊢
  rw [Ideal.span_insert]
  have hF : F ∈ Ideal.span {c} ⊔ Ideal.span {X - C e} := by
    have h := Ideal.add_mem _ (Ideal.mem_sup_left (Ideal.mem_span_singleton_self c))
      (Ideal.mem_sup_right (Ideal.mul_mem_right Q _ (Ideal.mem_span_singleton_self (X - C e))))
    rwa [← hQ, add_sub_cancel] at h
  have hc : c ∈ Ideal.span {F} ⊔ Ideal.span {X - C e} := by
    have h := Ideal.sub_mem _ (Ideal.mem_sup_left (Ideal.mem_span_singleton_self F))
      (Ideal.mem_sup_right (Ideal.mul_mem_right Q _ (Ideal.mem_span_singleton_self (X - C e))))
    rwa [← hQ, sub_sub_cancel] at h
  apply le_antisymm
  · exact sup_le ((Ideal.span_singleton_le_iff_mem _).mpr hF) le_sup_right
  · exact sup_le ((Ideal.span_singleton_le_iff_mem _).mpr hc) le_sup_right

lemma root_sub_eq (F : ℤ_[p][X]) (e : ℤ_[p]) :
    AdjoinRoot.root F - AdjoinRoot.of F e = AdjoinRoot.mk F (X - C e) := by
  rw [map_sub, AdjoinRoot.mk_X, AdjoinRoot.mk_C]

/-- **The `T₂`-Eisenstein quotient**: `ℤ_p[x]/(F, x − e) ≅ ℤ/p` (equals [U] `def:u-congmod`'s `C_Eis,k` only given the generator statement, L-H0). -/
theorem cong_equiv (hval : (p : ℤ_[p]) ∣ F.eval e) (hval2 : ¬ (p : ℤ_[p]) ^ 2 ∣ F.eval e) :
    Nonempty ((AdjoinRoot F ⧸ Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e}) ≃+* ZMod p) := by
  obtain ⟨u, hu, hFe⟩ := eval_eq_p_mul_unit hval hval2
  have hspan : (Ideal.span {F.eval e} : Ideal ℤ_[p]) = RingHom.ker (PadicInt.toZMod : ℤ_[p] →+* ZMod p) := by
    rw [PadicInt.ker_toZMod, maxIdeal_eq, hFe, Ideal.span_singleton_mul_right_unit hu]
  have hI : (Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e} : Ideal (AdjoinRoot F)) =
      (Ideal.span {X - C e}).map (AdjoinRoot.mk F) := by
    rw [Ideal.map_span, Set.image_singleton, root_sub_eq]
  rw [hI]
  have e2 : ((ℤ_[p][X] ⧸ Ideal.span {F}) ⧸ (Ideal.span {X - C e}).map (Ideal.Quotient.mk (Ideal.span {F}))) ≃+*
      ℤ_[p][X] ⧸ (Ideal.span {F} ⊔ Ideal.span {X - C e}) :=
    DoubleQuot.quotQuotEquivQuotSup (Ideal.span {F}) (Ideal.span {X - C e})
  have e3 := Ideal.quotEquivOfEq (span_F_sup F e)
  have e4 := (Polynomial.quotientSpanCXSubCAlgEquiv (F.eval e) e).toRingEquiv
  have e5 := Ideal.quotEquivOfEq hspan
  have e6 := RingHom.quotientKerEquivOfSurjective (ZMod.ringHom_surjective (PadicInt.toZMod : ℤ_[p] →+* ZMod p))
  exact ⟨e2.trans (e3.trans (e4.trans (e5.trans e6)))⟩

theorem span_pi_maximal (hval : (p : ℤ_[p]) ∣ F.eval e) (hval2 : ¬ (p : ℤ_[p]) ^ 2 ∣ F.eval e) :
    (Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e} : Ideal (AdjoinRoot F)).IsMaximal := by
  obtain ⟨φ⟩ := cong_equiv hval hval2
  exact Ideal.Quotient.maximal_of_isField _ (MulEquiv.isField (Field.toIsField (ZMod p)) φ.toMulEquiv)

theorem p_mem_maximal (hm : F.Monic) (M : Ideal (AdjoinRoot F)) (hM : M.IsMaximal) :
    algebraMap ℤ_[p] (AdjoinRoot F) p ∈ M := by
  have : Module.Finite ℤ_[p] (AdjoinRoot F) := (AdjoinRoot.powerBasis' hm).finite
  have : Algebra.IsIntegral ℤ_[p] (AdjoinRoot F) := Algebra.IsIntegral.of_finite _ _
  have hc := Ideal.isMaximal_comap_of_isIntegral_of_isMaximal (R := ℤ_[p]) M
  have := IsLocalRing.eq_maximalIdeal hc
  have hp : (p : ℤ_[p]) ∈ M.comap (algebraMap ℤ_[p] (AdjoinRoot F)) := by
    rw [this, maxIdeal_eq]; exact Ideal.mem_span_singleton_self _
  exact hp

theorem maximal_eq (hm : F.Monic)
    (hshape : F.map PadicInt.toZMod = (X - C (PadicInt.toZMod e)) ^ F.natDegree)
    (hval : (p : ℤ_[p]) ∣ F.eval e) (hval2 : ¬ (p : ℤ_[p]) ^ 2 ∣ F.eval e)
    (M : Ideal (AdjoinRoot F)) (hM : M.IsMaximal) :
    M = Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e} := by
  have hpM := p_mem_maximal hm M hM
  obtain ⟨H, hH⟩ := sub_shape_dvd hshape
  have himg : AdjoinRoot.mk F (F - (X - C e) ^ F.natDegree) = - (AdjoinRoot.root F - AdjoinRoot.of F e) ^ F.natDegree := by
    rw [map_sub, AdjoinRoot.mk_self, map_pow, map_sub, AdjoinRoot.mk_X, AdjoinRoot.mk_C, zero_sub]
  have hpow : (AdjoinRoot.root F - AdjoinRoot.of F e) ^ F.natDegree ∈ M := by
    have h1 : AdjoinRoot.mk F (F - (X - C e) ^ F.natDegree) ∈ M := by
      rw [hH, map_mul, AdjoinRoot.mk_C]
      exact M.mul_mem_right _ hpM
    rw [himg] at h1
    exact (neg_mem_iff).mp h1
  have hπ : AdjoinRoot.root F - AdjoinRoot.of F e ∈ M := hM.isPrime.mem_of_pow_mem _ hpow
  have hle : (Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e} : Ideal (AdjoinRoot F)) ≤ M :=
    (Ideal.span_singleton_le_iff_mem _).mpr hπ
  exact ((span_pi_maximal hval hval2).eq_of_le hM.ne_top hle).symm

/-- **The finite-weight theorem** (07 §5.9 "soundness, now"; [U] `prop:u-H12`(4), `prop:u-H11`). -/
theorem eis_finite_weight (hm : F.Monic) (hg : 0 < F.natDegree)
    (hshape : F.map PadicInt.toZMod = (X - C (PadicInt.toZMod e)) ^ F.natDegree)
    (hval : (p : ℤ_[p]) ∣ F.eval e) (hval2 : ¬ (p : ℤ_[p]) ^ 2 ∣ F.eval e) : EisFiniteWeight F e := by
  have hirr := eis_irreducible hm hg hshape hval2
  have hF0 : F ≠ 0 := hm.ne_zero
  refine ⟨eis_taylor hm hshape hval2, hirr,
    (hm.irreducible_iff_irreducible_map_fraction_map (K := ℚ_[p])).mp hirr, ?_,
    maximal_eq hm hshape hval hval2, ?_, cong_equiv hval hval2⟩
  · rw [(AdjoinRoot.powerBasis' hm).finrank, AdjoinRoot.powerBasis'_dim]
  · have hprime : (Ideal.span {F}).IsPrime := (Ideal.span_singleton_prime hF0).mpr hirr.prime
    have hdom : IsDomain (AdjoinRoot F) := Ideal.Quotient.isDomain (Ideal.span {F})
    have : IsNoetherianRing (AdjoinRoot F) := inferInstanceAs (IsNoetherianRing (ℤ_[p][X] ⧸ Ideal.span {F}))
    have hmax := span_pi_maximal hval hval2 (F := F) (e := e)
    have : IsLocalRing (AdjoinRoot F) :=
      IsLocalRing.of_unique_max_ideal ⟨_, hmax, fun M hM => maximal_eq hm hshape hval hval2 M hM⟩
    have hme : Ideal.span {AdjoinRoot.root F - AdjoinRoot.of F e} = maximalIdeal (AdjoinRoot F) :=
      IsLocalRing.eq_maximalIdeal hmax
    have hnf : ¬ IsField (AdjoinRoot F) := by
      intro hf
      have h0 := (IsLocalRing.isField_iff_maximalIdeal_eq.mp hf)
      have hp := p_mem_maximal hm _ (maximalIdeal.isMaximal (AdjoinRoot F))
      rw [h0, Ideal.mem_bot] at hp
      have hinj := AdjoinRoot.of.injective_of_degree_ne_zero (f := F)
        (by rw [degree_eq_natDegree hF0]; exact_mod_cast hg.ne')
      have : (p : ℤ_[p]) = 0 := hinj (by rw [map_zero]; exact hp)
      exact (Nat.cast_ne_zero.mpr (Fact.out : p.Prime).ne_zero) this
    have hpr : (maximalIdeal (AdjoinRoot F)).IsPrincipal := ⟨⟨_, hme.symm⟩⟩
    exact ⟨hdom, ((IsDiscreteValuationRing.TFAE (AdjoinRoot F) hnf).out 4 0).mp hpr⟩

end FiniteWeight

/-! ### K9 soundness -/

/-- The polynomial `f_k` of a certificate, over `ℤ`. -/
noncomputable def _root_.GL2.HeckeZ.Exec.K9.K9Cert.Fp (c : K9Cert) : ℤ[X] := toPoly c.F

lemma Fp_eq (c : K9Cert) : c.Fp = toPoly c.flo + X ^ c.g := toPoly_append_one c.flo

lemma Fp_monic (c : K9Cert) : c.Fp.Monic := by
  rw [Fp_eq, add_comm]; exact monic_X_pow_add (degree_toPoly_lt c.flo)

lemma Fp_natDegree (c : K9Cert) : c.Fp.natDegree = c.g := by
  rw [Fp_eq, add_comm, natDegree_add_eq_left_of_degree_lt] <;> simp only [natDegree_X_pow, degree_X_pow]
  exact degree_toPoly_lt c.flo

/-- `ℓ ∥ z` read off a residue modulo `ℓ²`. -/
lemma v1_of_res {ℓ : ℕ} [Fact ℓ.Prime] (x : ℤ_[ℓ]) (z : ℤ) (hx : PadicInt.toZModPow 2 x = (z : ZMod (ℓ ^ 2)))
    (h : v1B ℓ z = true) : (ℓ : ℤ_[ℓ]) ∣ x ∧ ¬ (ℓ : ℤ_[ℓ]) ^ 2 ∣ x := by
  unfold v1B at h
  simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h
  refine ⟨?_, ?_⟩
  · have := (padic_dvd_iff_int (j := 1) (by norm_num) x z hx).mpr (by simpa using Int.dvd_of_emod_eq_zero h.1)
    simpa using this
  · rw [padic_dvd_iff_int (le_refl 2) x z hx]
    exact fun hd => h.2 (Int.emod_eq_zero_of_dvd hd)

/-- A residue modulo `ℓ²` determines the residue modulo `ℓ`. -/
lemma toZMod_of_res {ℓ : ℕ} [Fact ℓ.Prime] (x : ℤ_[ℓ]) (z : ℤ) (hx : PadicInt.toZModPow 2 x = (z : ZMod (ℓ ^ 2))) :
    PadicInt.toZMod x = (z : ZMod ℓ) := by
  have h2 : (ℓ : ℤ_[ℓ]) ^ 2 ∣ x - z := by
    rw [padic_dvd_iff, map_sub, hx, map_intCast, sub_self]
  have h1 : x - z ∈ maximalIdeal ℤ_[ℓ] := by
    rw [maxIdeal_eq, Ideal.mem_span_singleton]; exact (dvd_pow_self _ two_ne_zero).trans h2
  rw [mem_maxIdeal_iff, map_sub, map_intCast, sub_eq_zero] at h1
  exact h1

/-- What an accepted K9 certificate certifies. -/
structure K9Spec (c : K9Cert) : Prop where
  ell_prime : c.ell.Prime
  ell_five : 5 ≤ c.ell
  k_three : 3 ≤ c.k
  class_ne_zero : ¬ (c.ell - 1) ∣ c.k
  g_pos : 1 ≤ c.g
  monic : c.Fp.Monic
  natDegree : c.Fp.natDegree = c.g
  /-- `e = 1 + 2^{k−1}` in `ℤ/ℓ²` -/
  root : ((c.e : ℕ) : ZMod (c.ell ^ 2)) = 1 + 2 ^ (c.k - 1)
  /-- `f ≡ (x − e)^g (mod ℓ)` -/
  shape : c.Fp.map (Int.castRingHom (ZMod c.ell)) = (X - C ((c.e : ℕ) : ZMod c.ell)) ^ c.g
  /-- `ℓ ∥ F̃(ẽ)` for every lift -/
  val : ∀ [Fact c.ell.Prime] (F' : ℤ_[c.ell][X]) (e' : ℤ_[c.ell]),
    F'.map (PadicInt.toZModPow 2) = c.Fp.map (Int.castRingHom (ZMod (c.ell ^ 2))) →
    PadicInt.toZModPow 2 e' = ((c.e : ℕ) : ZMod (c.ell ^ 2)) →
    (c.ell : ℤ_[c.ell]) ∣ F'.eval e' ∧ ¬ (c.ell : ℤ_[c.ell]) ^ 2 ∣ F'.eval e'
  /-- `ℓ ∥ b̃` for every lift `b̃` of `B_k/k mod ℓ²` -/
  bern : ∀ [Fact c.ell.Prime] (b : ℤ_[c.ell]), PadicInt.toZModPow 2 b = ((c.bk : ℕ) : ZMod (c.ell ^ 2)) →
    (c.ell : ℤ_[c.ell]) ∣ b ∧ ¬ (c.ell : ℤ_[c.ell]) ^ 2 ∣ b
  /-- `g = 2`: `ℓ ∥ disc F̃` for every lift -/
  disc : c.g = 2 → ∀ [Fact c.ell.Prime] (F' : ℤ_[c.ell][X]),
    F'.map (PadicInt.toZModPow 2) = c.Fp.map (Int.castRingHom (ZMod (c.ell ^ 2))) →
    (c.ell : ℤ_[c.ell]) ∣ F'.coeff 1 ^ 2 - 4 * F'.coeff 0 ∧ ¬ (c.ell : ℤ_[c.ell]) ^ 2 ∣ F'.coeff 1 ^ 2 - 4 * F'.coeff 0
  /-- `G_a(x_k) ≡ −B_k/k (mod ℓ²)` -/
  sign : ((c.gk : ℕ) : ZMod (c.ell ^ 2)) = - ((c.bk : ℕ) : ZMod (c.ell ^ 2))
  /-- the λ-test: `ℓ ∥ G̃_k − G̃_{k+ℓ−1}` for every lift -/
  lam : ∀ [Fact c.ell.Prime] (G₁ G₂ : ℤ_[c.ell]), PadicInt.toZModPow 2 G₁ = ((c.gk : ℕ) : ZMod (c.ell ^ 2)) →
    PadicInt.toZModPow 2 G₂ = ((c.gnext : ℕ) : ZMod (c.ell ^ 2)) →
    (c.ell : ℤ_[c.ell]) ∣ G₁ - G₂ ∧ ¬ (c.ell : ℤ_[c.ell]) ^ 2 ∣ G₁ - G₂

/-- **K9 is sound.** -/
theorem k9_sound (c : K9Cert) (hc : k9Check c = true) : K9Spec c := by
  unfold k9Check at hc
  simp only [Bool.and_eq_true] at hc
  obtain ⟨⟨⟨⟨⟨⟨⟨hguard, hroot⟩, hshape⟩, hval⟩, hbern⟩, hdisc⟩, hsign⟩, hlam⟩ := hc
  unfold guardB at hguard
  simp only [Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq] at hguard
  obtain ⟨⟨⟨⟨⟨hpr, h5⟩, hk3⟩, hcls⟩, hg1⟩, hkr⟩ := hguard
  have pℓ := primeB_sound hpr
  have hM : 0 < c.ell ^ 2 := pow_pos pℓ.pos 2
  have hroot' : c.e = (1 + 2 ^ (c.k - 1)) % (c.ell ^ 2) := by
    unfold rootB eisRoot at hroot
    rw [beq_iff_eq, GL2.L23.SmithPK.powMod_eq hM _ _ _ hkr] at hroot
    rw [hroot, Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]
  -- the value `f(e)` as an integer
  have hFe : ∀ {R : Type} [CommRing R] (φ : ℤ →+* R), (c.Fp.map φ).eval (φ (c.e : ℤ)) = φ (peval c.F (c.e : ℤ)) := by
    intro R _ φ
    rw [eval_map, eval₂_at_apply, peval_eq]; rfl
  refine ⟨pℓ, h5, hk3, fun hd => hcls (Nat.mod_eq_zero_of_dvd hd), hg1, Fp_monic c, Fp_natDegree c, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- root
    rw [hroot', ZMod.natCast_mod]; push_cast; ring
  · -- shape
    have := peqMod_sound hshape
    rw [toPoly_plin] at this
    unfold K9Cert.Fp
    rw [this, Polynomial.map_pow, Polynomial.map_sub, map_X, map_C]; simp
  · -- val
    intro _ F' e' hF he
    apply v1_of_res _ (peval c.F (c.e : ℤ)) _ hval
    have h1 := congrArg (fun P => P.eval (PadicInt.toZModPow 2 e')) hF
    rw [eval_map, eval₂_at_apply] at h1
    rw [h1, he]
    have := hFe (Int.castRingHom (ZMod (c.ell ^ 2)))
    simpa using this
  · -- bern
    intro _ b hb
    exact v1_of_res b (c.bk : ℤ) (by rw [hb]; simp) hbern
  · -- disc
    intro hg2 _ F' hF
    unfold discB at hdisc
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq] at hdisc
    have hd := hdisc.resolve_left (fun h => h hg2)
    apply v1_of_res _ _ _ hd
    have h0 := congrArg (fun P => P.coeff 0) hF
    have h1 := congrArg (fun P => P.coeff 1) hF
    simp only [coeff_map] at h0 h1
    rw [map_sub, map_mul, map_pow, map_ofNat, h0, h1]
    simp [K9Cert.Fp, coeff_toPoly]
  · -- sign
    unfold signB at hsign
    rw [beq_iff_eq] at hsign
    have := (ZMod.natCast_eq_zero_iff _ _).mpr (Nat.dvd_of_mod_eq_zero hsign)
    push_cast at this
    exact eq_neg_of_add_eq_zero_left this
  · -- lam
    intro _ G₁ G₂ h1 h2
    unfold lamB at hlam
    simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at hlam
    obtain ⟨hl1, hl2⟩ := hlam
    have hres : PadicInt.toZModPow 2 (G₁ - G₂) = (((c.gk : ℤ) - c.gnext : ℤ) : ZMod (c.ell ^ 2)) := by
      rw [map_sub, h1, h2]; push_cast; rfl
    have hdv : ((c.ell : ℤ) ^ 2) ∣ ((c.gk : ℤ) - c.gnext) - c.ell * c.lam := Int.dvd_of_emod_eq_zero hl1
    have hv : v1B c.ell ((c.gk : ℤ) - c.gnext) = true := by
      unfold v1B
      simp only [Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq]
      obtain ⟨w, hw⟩ := hdv
      have heq : (c.gk : ℤ) - c.gnext = c.ell * (c.lam + c.ell * w) := by linear_combination hw
      refine ⟨by rw [heq]; exact Int.mul_emod_right _ _, ?_⟩
      intro h0
      have hd2 : (c.ell : ℤ) ^ 2 ∣ c.ell * (c.lam + c.ell * w) := heq ▸ Int.dvd_of_emod_eq_zero h0
      rw [pow_two] at hd2
      have hd3 : (c.ell : ℤ) ∣ c.lam + c.ell * w :=
        (Int.dvd_of_mul_dvd_mul_left (by exact_mod_cast pℓ.ne_zero) hd2)
      have hd4 : (c.ell : ℤ) ∣ (c.lam : ℤ) := by
        have := dvd_sub hd3 (dvd_mul_right (c.ell : ℤ) w); simpa using this
      exact hl2 (by exact_mod_cast Nat.mod_eq_zero_of_dvd (by exact_mod_cast hd4))
    exact v1_of_res _ _ hres hv

/-- **The K9 certificate theorem**: an accepted certificate and every monic lift `F̃ ∈ ℤ_ℓ[x]` of degree `g` of `f_k mod ℓ²`
satisfy the finite-weight theorem at `e_k = 1 + 2^{k−1}`. -/
theorem k9_cert (c : K9Cert) (hc : k9Check c = true) [Fact c.ell.Prime] (F' : ℤ_[c.ell][X]) (hm : F'.Monic)
    (hdeg : F'.natDegree = c.g)
    (hF : F'.map (PadicInt.toZModPow 2) = c.Fp.map (Int.castRingHom (ZMod (c.ell ^ 2)))) :
    EisFiniteWeight F' (1 + 2 ^ (c.k - 1)) := by
  have hs := k9_sound c hc
  have he : PadicInt.toZModPow 2 (1 + 2 ^ (c.k - 1) : ℤ_[c.ell]) = ((c.e : ℕ) : ZMod (c.ell ^ 2)) := by
    rw [hs.root]; simp [map_ofNat]
  have hv := hs.val F' _ hF he
  refine eis_finite_weight hm (by rw [hdeg]; exact hs.g_pos) ?_ hv.1 hv.2
  -- the shape modulo `ℓ`
  rw [hdeg]
  ext i
  have hi := congrArg (fun P => P.coeff i) hF
  simp only [coeff_map] at hi
  rw [coeff_map, toZMod_of_res _ (c.Fp.coeff i) (by rw [hi]; simp)]
  have := congrArg (fun P => P.coeff i) hs.shape
  simp only [coeff_map, eq_intCast] at this
  rw [this]
  have hee : PadicInt.toZMod (1 + 2 ^ (c.k - 1) : ℤ_[c.ell]) = ((c.e : ℕ) : ZMod c.ell) := by
    rw [toZMod_of_res _ (c.e : ℤ) (by rw [he]; simp)]; simp
  rw [hee]

end GL2.HeckeZ.K9
