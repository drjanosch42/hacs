import Mathlib

/-!
# G1 → M2 entry point (L17-3): the `Kᵘ` key model (L16-4's limit 1)

L16-4's `obl:tau` checker names an oriented simplex `γ · ι(σ)` of `Kᵘ` by its **canonical key** (P4):
`(σ̂, the lexicographically least γ t over t ∈ S(σ, σ̂))`.  Its trust boundary was that these keys are a
faithful coordinate system on the simplices of `Kᵘ`.  This file states and proves that, for an abstract
model datum: the simplices of `Kᵘ` are the pairs `(γ, σ)` modulo the **transporter groupoid**
`(γ, σ) ~ (γ', σ') ⟺ γ⁻¹ γ' ∈ S(σ, σ')` (where `S(σ, σ') = {t : ι(σ) = t ι(σ')}`), and the canonical key is a
complete invariant of that quotient.

What is assumed is exactly the groupoid structure of the transporter sets (identity, composition,
inverses), that each `σ` has a transporter to its orbit representative, and that the representative is
constant on orbits — the content of the C++ model certificate `validate_closure` ([C] `H4`).  The
orientation sign (the parity of the vertex correspondence) is not modelled here; it is the next layer of
M2's `X_bulk`.  Nothing hard-wires the 7-vertex model, a trivial centre, or `ℓ`.
-/

namespace GL2.Ku

open scoped MatrixGroups

/-- A transporter model on a type `V` of simplices (of one dimension). -/
structure KeyModel (V : Type*) where
  /-- the orbit representative `σ̂` -/
  rep : V → V
  /-- the transporter set `S(σ₁, σ₂) = {t ∈ Γ : ι(σ₁) = t ι(σ₂)}` (finite: the stabilisers are finite) -/
  S : V → V → Finset SL(2, ℤ)
  one_mem : ∀ σ, (1 : SL(2, ℤ)) ∈ S σ σ
  mul_mem : ∀ {a b c : V} {t u : SL(2, ℤ)}, t ∈ S a b → u ∈ S b c → t * u ∈ S a c
  inv_mem : ∀ {a b : V} {t : SL(2, ℤ)}, t ∈ S a b → t⁻¹ ∈ S b a
  rep_mem : ∀ σ, (S σ (rep σ)).Nonempty
  rep_eq : ∀ {a b : V}, (S a b).Nonempty → rep a = rep b

variable {V : Type*} (M : KeyModel V)

/-- The transporter groupoid relation: `(γ, σ)` and `(γ', σ')` name the same simplex `γ ι(σ) = γ' ι(σ')`. -/
def Rel (x y : SL(2, ℤ) × V) : Prop := x.1⁻¹ * y.1 ∈ M.S x.2 y.2

/-- The relation is an equivalence: `Kᵘ` is the quotient. -/
theorem rel_equivalence : Equivalence (Rel M) where
  refl x := by unfold Rel; rw [inv_mul_cancel]; exact M.one_mem _
  symm {x y} h := by
    unfold Rel at *; have := M.inv_mem h; rwa [mul_inv_rev, inv_inv] at this
  trans {x y z} h₁ h₂ := by
    unfold Rel at *; have := M.mul_mem h₁ h₂; rwa [mul_assoc, mul_inv_cancel_left] at this

/-- The lexicographic order on `(a, b, c, d)`. -/
abbrev KeyT := Lex (ℤ × Lex (ℤ × Lex (ℤ × ℤ)))

/-- The lexicographic key of `γ` on `(a, b, c, d)` (`hecke`'s `lex_less`). -/
def lexKey (γ : SL(2, ℤ)) : KeyT := toLex (γ 0 0, toLex (γ 0 1, toLex (γ 1 0, γ 1 1)))

lemma lexKey_injective : Function.Injective lexKey := by
  intro γ γ' h
  simp only [lexKey, toLex_inj, Prod.mk.injEq] at h
  ext i j; fin_cases i <;> fin_cases j <;> simp [h]

/-- The candidate keys `{lexKey (γ t) : t ∈ S(σ, σ̂)}`. -/
def cands (x : SL(2, ℤ) × V) : Finset (KeyT) :=
  (M.S x.2 (M.rep x.2)).image (fun t => lexKey (x.1 * t))

lemma cands_nonempty (x : SL(2, ℤ) × V) : (cands M x).Nonempty := (M.rep_mem x.2).image _

/-- **(P4) The canonical key** of `γ ι(σ)`: `(σ̂, min_lex γ t)`. -/
def canon (x : SL(2, ℤ) × V) : V × KeyT := (M.rep x.2, (cands M x).min' (cands_nonempty M x))

/-- Related pairs have the same candidate set (left translation by `γ⁻¹ γ'` is a bijection
`S(σ', ρ) → S(σ, ρ)`). -/
lemma cands_eq {x y : SL(2, ℤ) × V} (h : Rel M x y) (hr : M.rep x.2 = M.rep y.2) : cands M x = cands M y := by
  unfold cands Rel at *
  ext k
  simp only [Finset.mem_image]
  constructor
  · rintro ⟨t, ht, rfl⟩
    refine ⟨y.1⁻¹ * x.1 * t, ?_, by group⟩
    have h' := M.inv_mem h
    rw [mul_inv_rev, inv_inv] at h'
    rw [← hr]; exact M.mul_mem h' ht
  · rintro ⟨t, ht, rfl⟩
    refine ⟨x.1⁻¹ * y.1 * t, ?_, by group⟩
    rw [hr]; exact M.mul_mem h ht

/-- **The key model is faithful:** two pairs have the same canonical key iff they name the same simplex
of `Kᵘ`.  So canonical keys are a coordinate system on `Kᵘ` (L16-4's limit 1, discharged relative to the
groupoid axioms). -/
theorem canon_faithful (x y : SL(2, ℤ) × V) : canon M x = canon M y ↔ Rel M x y := by
  constructor
  · intro h
    simp only [canon, Prod.mk.injEq] at h
    obtain ⟨hr, hk⟩ := h
    obtain ⟨t, ht, htk⟩ := Finset.mem_image.1 ((cands M x).min'_mem (cands_nonempty M x))
    obtain ⟨u, hu, huk⟩ := Finset.mem_image.1 ((cands M y).min'_mem (cands_nonempty M y))
    rw [hk, ← huk] at htk
    have hxy : x.1 * t = y.1 * u := lexKey_injective htk
    unfold Rel
    have : x.1⁻¹ * y.1 = t * u⁻¹ := by
      have : y.1 = x.1 * t * u⁻¹ := by rw [hxy]; group
      rw [this]; group
    rw [this]
    have hu' := M.inv_mem hu
    rw [← hr] at hu'
    exact M.mul_mem ht hu'
  · intro h
    have hr : M.rep x.2 = M.rep y.2 := M.rep_eq ⟨_, h⟩
    simp only [canon, Prod.mk.injEq]
    refine ⟨hr, ?_⟩
    congr 1
    exact cands_eq M h hr

end GL2.Ku
