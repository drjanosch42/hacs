/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Exec.K3
import GL2.HeckeZ.Cong

/-!
# HeckeZ K3: specification, soundness, and the bridge to the congruence module (L-HZ0, M-HZ1)

`07-lean-plan.md` §5 (common shape) and §5.3 (K3). The executable checker `GL2.HeckeZ.Exec.k3Check` is Mathlib-free;
here:
* `K3Spec c` — the integer facts the checker establishes (`k3Check_sound`);
* `k3_sound` — **the bridge**: for an Eisenstein split `E` of `T ⊆ End_ℤ L` (`GL2.HeckeZ.EisSplit`), if the systems
  of the certificate are those of `E` (`hsys`), `θ = T_{readP}`, and the dumped matrix is `T_{readP}` on `L/ℓ^K` in some
  basis (**TB-mat**: `U · [T_{readP}]_b ≡ D · U mod ℓ^K` with `U` invertible, `hmat`), then `g_P ∣ λ t − μ t` for every
  `t ∈ T` (the lower bound, 01 Prop. 5.4 read at a root) and `g_P` lies in the congruence ideal (the upper bound,
  Prop. 5.3); with `EisSplit.fibre_iff` this is `T ≅ ℤ ×_{ℤ/g_P} ℤ`, `C_T ≅ ℤ/g_P`.

The trust boundary is only `hmat` (and, through `hsys`, whatever fixes `λ`, `μ`: TB-χ in T1). The reduction of the
conjugation identity mod `ℓ^v` and `ℓ^{v+1}` (both divide `ℓ^K`) is what makes the read basis-independent.
-/

namespace GL2.HeckeZ.K3

open GL2.HeckeZ.Exec

/-- **`K3Spec`**: what `k3Check` establishes, as integer statements. -/
structure K3Spec (c : K3Cert) : Prop where
  hv : c.v + 1 ≤ c.K
  gP_eq : c.gP = c.ell ^ c.e
  gP_span : (c.gP : ℤ) ∈ Ideal.span {x : ℤ | ∃ s ∈ c.sys, x = s.2.1 - s.2.2}
  read : ∃ lam mu : ℤ, (c.readP, lam, mu) ∈ c.sys ∧
    (∀ i j : Fin c.f, (c.ell : ℤ) ^ c.v ∣ (entry c.dump i j : ℤ) - if i = j then mu else 0) ∧
    (∃ i j : Fin c.f, ¬ (c.ell : ℤ) ^ (c.v + 1) ∣ (entry c.dump i j : ℤ) - if i = j then mu else 0) ∧
    (c.ell : ℤ) ^ (c.v + c.e) ∣ lam - mu

/-! ### Soundness of the checker -/

lemma allFin_iff (f : ℕ) (P : ℕ → ℕ → Bool) : allFin f P = true ↔ ∀ i < f, ∀ j < f, P i j = true := by
  simp [allFin, List.all_eq_true, List.mem_range]

lemma gcdList_mem (l : List ℤ) : (gcdList l : ℤ) ∈ Ideal.span {x : ℤ | x ∈ l} := by
  induction l with
  | nil => simp [gcdList]
  | cons a l ih =>
    have hmono : Ideal.span {x : ℤ | x ∈ l} ≤ Ideal.span {x : ℤ | x ∈ a :: l} :=
      Ideal.span_mono fun x hx => List.mem_cons_of_mem a hx
    have ha : a ∈ Ideal.span {x : ℤ | x ∈ a :: l} := Ideal.subset_span (List.mem_cons_self ..)
    have e : (gcdList (a :: l) : ℤ) = (Int.gcd a (gcdList l : ℤ) : ℤ) := by
      simp [gcdList, Int.gcd]
    rw [e, Int.gcd_eq_gcd_ab]
    exact Ideal.add_mem _ (Ideal.mul_mem_right _ _ ha) (Ideal.mul_mem_right _ _ (hmono ih))

lemma cEntry_dvd (c : K3Cert) (h2 : 2 ≤ c.ell) (mu : ℤ) (i j : ℕ) {n : ℕ} (hn : n ∣ c.m) :
    (n ∣ cEntry c mu i j) ↔ (n : ℤ) ∣ (entry c.dump i j : ℤ) - if i = j then mu else 0 := by
  have hm : (0 : ℤ) < (c.m : ℤ) := by
    have : 0 < c.m := Nat.pow_pos (by omega)
    exact_mod_cast this
  have key : (c.m : ℤ) ∣ ((entry c.dump i j : ℤ) - if i = j then mu else 0) - (cEntry c mu i j : ℤ) := by
    unfold cEntry resid
    split_ifs with hij
    · rw [Int.natCast_mod, Nat.cast_add, Int.toNat_of_nonneg (Int.emod_nonneg _ hm.ne')]
      have h1 := Int.emod_emod_of_dvd ((entry c.dump i j : ℤ) + (-mu) % (c.m : ℤ)) (dvd_refl (c.m : ℤ))
      refine (Int.ModEq.dvd ?_)
      calc ((entry c.dump i j : ℤ) + (-mu) % (c.m : ℤ)) % (c.m : ℤ)
          ≡ (entry c.dump i j : ℤ) + (-mu) % (c.m : ℤ) [ZMOD c.m] := Int.mod_modEq _ _
        _ ≡ (entry c.dump i j : ℤ) + -mu [ZMOD c.m] := Int.ModEq.add_left _ (Int.mod_modEq _ _)
        _ = (entry c.dump i j : ℤ) - mu := by ring
    · rw [Nat.add_zero, Int.natCast_mod, sub_zero]
      exact Int.ModEq.dvd (Int.mod_modEq _ _)
  have hk : (n : ℤ) ∣ ((entry c.dump i j : ℤ) - if i = j then mu else 0) - (cEntry c mu i j : ℤ) :=
    (Int.natCast_dvd_natCast.mpr hn).trans key
  rw [← Int.natCast_dvd_natCast]
  constructor
  · intro h; have := dvd_add hk h; simpa using this
  · intro h; have := dvd_sub h hk; simpa using this

/-- **Soundness of the checker.** -/
theorem k3Check_sound (c : K3Cert) (h : k3Check c = true) : K3Spec c := by
  unfold k3Check at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨h2, _⟩, hvK⟩, _⟩, _⟩, _⟩, hgcd⟩, hpow⟩, hread⟩ := h
  refine ⟨hvK, hpow, ?_, ?_⟩
  · rw [hgcd]
    refine Ideal.span_mono ?_ (gcdList_mem _)
    intro x hx
    obtain ⟨s, hs, rfl⟩ := List.mem_map.mp hx
    exact ⟨s, hs, rfl⟩
  · split at hread
    · rename_i s hs
      have hsmem : s ∈ c.sys := List.mem_of_find?_eq_some hs
      have hs1 : s.1 = c.readP := by simpa using List.find?_some hs
      unfold readOK at hread
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, bne_iff_ne, ne_eq] at hread
      obtain ⟨⟨⟨⟨hall, hwit⟩, hw1⟩, hw2⟩, hlm⟩ := hread
      have hvm : c.ell ^ c.v ∣ c.m := pow_dvd_pow _ (by omega)
      have hv1m : c.ell ^ (c.v + 1) ∣ c.m := pow_dvd_pow _ hvK
      refine ⟨s.2.1, s.2.2, by rw [← hs1]; exact hsmem, ?_, ?_, ?_⟩
      · intro i j
        have := (allFin_iff _ _).mp hall i i.2 j j.2
        simp only [beq_iff_eq] at this
        have h' := (cEntry_dvd c h2 s.2.2 i j hvm).mp (Nat.dvd_of_mod_eq_zero this)
        simpa [Fin.ext_iff] using h'
      · refine ⟨⟨c.wit.1, hw1⟩, ⟨c.wit.2, hw2⟩, ?_⟩
        intro hd
        have := (cEntry_dvd c h2 s.2.2 c.wit.1 c.wit.2 hv1m).mpr (by push_cast at hd ⊢; simpa [Fin.ext_iff] using hd)
        exact hwit (Nat.mod_eq_zero_of_dvd this)
      · exact Int.dvd_of_emod_eq_zero hlm
    · exact absurd hread (by simp)

/-! ### The bridge: K3Spec ⇒ the congruence module -/

variable {L : Type*} [AddCommGroup L] {T : Subalgebra ℤ (Module.End ℤ L)}

/-- Two matrices conjugate by a unit modulo `m` vanish modulo `n ∣ m` together. -/
lemma zero_iff_of_conj {f m n : ℕ} (hn : n ∣ m) {A C U : Matrix (Fin f) (Fin f) (ZMod m)} (hU : IsUnit U)
    (h : U * A = C * U) :
    A.map (ZMod.castHom hn (ZMod n)) = 0 ↔ C.map (ZMod.castHom hn (ZMod n)) = 0 := by
  let φ : Matrix (Fin f) (Fin f) (ZMod m) →+* Matrix (Fin f) (Fin f) (ZMod n) := (ZMod.castHom hn (ZMod n)).mapMatrix
  have hφ : ∀ M, φ M = M.map (ZMod.castHom hn (ZMod n)) := fun _ => rfl
  have hU' : IsUnit (φ U) := hU.map φ
  have h' : φ U * φ A = φ C * φ U := by rw [← map_mul, ← map_mul, h]
  rw [← hφ, ← hφ]
  constructor
  · intro hA; rw [hA, mul_zero] at h'; exact (hU'.mul_left_eq_zero).mp h'.symm
  · intro hC; rw [hC, zero_mul] at h'; exact (hU'.mul_right_eq_zero).mp h'

/-- An integer matrix read in `ZMod q`. -/
def zmat {f : ℕ} (q : ℕ) (A : Fin f → Fin f → ℤ) : Matrix (Fin f) (Fin f) (ZMod q) := Matrix.of fun i j => (A i j : ZMod q)

lemma mapMatrix_zmat {f m n : ℕ} (hn : n ∣ m) (A : Fin f → Fin f → ℤ) :
    (zmat m A).map (ZMod.castHom hn (ZMod n)) = zmat n A := by
  ext i j; simp [zmat]

lemma zmat_eq_zero {f : ℕ} (q : ℕ) (A : Fin f → Fin f → ℤ) : zmat q A = 0 ↔ ∀ i j, (q : ℤ) ∣ A i j := by
  constructor
  · intro h i j; have := congrFun (congrFun h i) j
    simpa [zmat, ZMod.intCast_zmod_eq_zero_iff_dvd] using this
  · intro h; ext i j; simpa [zmat, ZMod.intCast_zmod_eq_zero_iff_dvd] using h i j

/-- **The K3 bridge** (01 Props. 5.3, 5.4 at `r = 1`; 07 §5.3 soundness). Under the trust boundary `hmat` (TB-mat), the
checked predicate `hc` gives the lower bound `g_P ∣ λ t − μ t` for all `t ∈ T` and the upper bound
`g_P ∈ congIdeal`. -/
theorem k3_sound (c : K3Cert) (hc : K3Spec c) (hℓ : c.ell.Prime) (E : EisSplit T) (g : ℕ → T)
    (hsys : ∀ s ∈ c.sys, E.lam (g s.1) = s.2.1 ∧ E.mu (g s.1) = s.2.2) (hθ : E.θ = g c.readP)
    (b : Module.Basis (Fin c.f) ℤ L) (U : Matrix (Fin c.f) (Fin c.f) (ZMod (c.ell ^ c.K))) (hU : IsUnit U)
    (hmat : U * (LinearMap.toMatrix b b (g c.readP : Module.End ℤ L)).map (Int.cast) =
      (Matrix.of fun (i j : Fin c.f) => (entry c.dump i j : ZMod (c.ell ^ c.K))) * U) :
    (∀ t : T, (c.gP : ℤ) ∣ E.lam t - E.mu t) ∧ (c.gP : ℤ) ∈ E.congIdeal := by
  obtain ⟨lam, mu, hmem, r0, r1, r2⟩ := hc.read
  obtain ⟨hl', hm'⟩ := hsys _ hmem
  simp only at hl' hm'
  have hl : E.lam E.θ = lam := by rw [hθ]; exact hl'
  have hm : E.mu E.θ = mu := by rw [hθ]; exact hm'
  -- the integer matrix of `P` in the basis `b`, and the dumped `c(T_p)`
  set A : Fin c.f → Fin c.f → ℤ := fun i j => b.repr (E.P (b j)) i with hA
  set C : Fin c.f → Fin c.f → ℤ := fun i j => (entry c.dump i j : ℤ) - if i = j then mu else 0 with hC
  have hAe : ∀ i j, A i j = LinearMap.toMatrix b b (g c.readP : Module.End ℤ L) i j - if i = j then mu else 0 := by
    intro i j
    simp only [hA, EisSplit.P, hθ, hm', map_sub, LinearEquiv.map_smul, Finsupp.sub_apply, Finsupp.smul_apply,
      Module.Basis.repr_self, LinearMap.toMatrix_apply, smul_eq_mul, Finsupp.single_apply]
    by_cases h : i = j
    · subst h; simp
    · simp [h, Ne.symm h]
  have hconj : U * zmat (c.ell ^ c.K) A = zmat (c.ell ^ c.K) C * U := by
    have e1 : zmat (c.ell ^ c.K) A = (LinearMap.toMatrix b b (g c.readP : Module.End ℤ L)).map (Int.cast) -
        (mu : ZMod (c.ell ^ c.K)) • (1 : Matrix (Fin c.f) (Fin c.f) (ZMod (c.ell ^ c.K))) := by
      ext i j; simp only [zmat, Matrix.of_apply, hAe, Matrix.sub_apply, Matrix.map_apply, Matrix.smul_apply,
        Matrix.one_apply, smul_eq_mul]; split_ifs <;> push_cast <;> ring
    have e2 : zmat (c.ell ^ c.K) C = (Matrix.of fun (i j : Fin c.f) => (entry c.dump i j : ZMod (c.ell ^ c.K))) -
        (mu : ZMod (c.ell ^ c.K)) • (1 : Matrix (Fin c.f) (Fin c.f) (ZMod (c.ell ^ c.K))) := by
      ext i j; simp only [zmat, hC, Matrix.of_apply, Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply,
        smul_eq_mul]; split_ifs <;> push_cast <;> ring
    rw [e1, e2, mul_sub, sub_mul, hmat, Matrix.mul_smul, Matrix.smul_mul, mul_one, one_mul]
  -- read modulo `ℓ^v` and `ℓ^{v+1}`
  have hvK := hc.hv
  have hdv : c.ell ^ c.v ∣ c.ell ^ c.K := pow_dvd_pow _ (by omega)
  have hdv1 : c.ell ^ (c.v + 1) ∣ c.ell ^ c.K := pow_dvd_pow _ hvK
  have hAv : ∀ i j, ((c.ell ^ c.v : ℕ) : ℤ) ∣ A i j := by
    have := (zero_iff_of_conj hdv hU hconj).mpr (by
      rw [mapMatrix_zmat, zmat_eq_zero]; intro i j; push_cast; exact r0 i j)
    rw [mapMatrix_zmat, zmat_eq_zero] at this; exact this
  have hAv1 : ∃ i j, ¬ ((c.ell ^ (c.v + 1) : ℕ) : ℤ) ∣ A i j := by
    by_contra hcon
    push Not at hcon
    have := (zero_iff_of_conj hdv1 hU hconj).mp (by rw [mapMatrix_zmat, zmat_eq_zero]; exact hcon)
    rw [mapMatrix_zmat, zmat_eq_zero] at this
    obtain ⟨i, j, hij⟩ := r1
    exact hij (by have := this i j; push_cast at this; exact this)
  obtain ⟨i, j, hij⟩ := hAv1
  have hD : (c.ell : ℤ) ^ (c.v + c.e) ∣ E.lam E.θ - E.mu E.θ := by rw [hl, hm]; exact r2
  refine ⟨fun t => ?_, ?_⟩
  · have := E.dvd_of_read b c.ell hℓ (c.v + c.e) c.v hD (b j) i (by have := hAv i j; push_cast at this; exact this)
      (by push_cast at hij; exact hij) t
    rw [hc.gP_eq]; push_cast; simpa using this
  · refine (Ideal.span_le.mpr ?_) hc.gP_span
    rintro x ⟨s, hs, rfl⟩
    obtain ⟨h1, h2⟩ := hsys s hs
    rw [← h1, ← h2]
    exact E.mem_congIdeal_sub _

end GL2.HeckeZ.K3
