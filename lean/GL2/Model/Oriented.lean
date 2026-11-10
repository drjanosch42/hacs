import GL2.Cosets.KeyModel

/-!
# M2-b (L18-1): the oriented key model

[GT] `def:orkeymodel`, `thm:orkeymodel`.  L17-3's `KeyModel` names an *unoriented* simplex `γ · ι(σ)` of `Kᵘ` by its
canonical key.  An oriented simplex also needs a sign: a transporter `t ∈ S(σ, σ')` (so `ι(σ) = t · ι(σ')`) carries the
vertex order of `ι(σ')` to *some* vertex order of `ι(σ)`, and `sgn σ σ' t ∈ ℤˣ` is the parity of that correspondence.
`OrKeyModel` adds that sign with its two laws:

* `sgn_mul` -- it is multiplicative along the transporter groupoid (the parity of a composite correspondence);
* `noInv` -- **H4(i), no inversions**: a stabiliser element `t ∈ S(σ, σ)` preserves the orientation.

The oriented canonical key `orCanon` is `canon` plus the sign of the picked transporter, and `orCanon_faithful` says
it is a coordinate system on oriented simplices: the key part is a complete invariant of the simplex (L17-3's
`canon_faithful`), and for related pairs the sign parts differ by exactly the transporter's sign, so the oriented
generator `[γ, σ]` is `sign · [canonical representative]` independently of the pair chosen.  Nothing hard-wires the
7-vertex model, a trivial centre or `ℓ`.
-/

namespace GL2.Ku

open scoped MatrixGroups

/-- A transporter model with orientation signs ([C] `H4`(i), orientation half). -/
structure OrKeyModel (V : Type*) extends KeyModel V where
  /-- the parity of the vertex correspondence of `t ∈ S(σ, σ')` -/
  sgn : V → V → SL(2, ℤ) → ℤˣ
  /-- the sign is multiplicative along the groupoid -/
  sgn_mul : ∀ {a b c : V} {t u : SL(2, ℤ)}, t ∈ S a b → u ∈ S b c → sgn a c (t * u) = sgn a b t * sgn b c u
  /-- **H4(i), no inversions**: a stabiliser element preserves the orientation -/
  noInv : ∀ {σ : V} {t : SL(2, ℤ)}, t ∈ S σ σ → sgn σ σ t = 1

variable {V : Type*} (M : OrKeyModel V)

namespace OrKeyModel

lemma sgn_one (σ : V) : M.sgn σ σ 1 = 1 := M.noInv (M.one_mem σ)

/-- The sign of an inverse transporter. -/
lemma sgn_inv {a b : V} {t : SL(2, ℤ)} (h : t ∈ M.S a b) : M.sgn b a t⁻¹ = M.sgn a b t := by
  have h1 := M.sgn_mul h (M.inv_mem h)
  rw [mul_inv_cancel, M.sgn_one] at h1
  have h2 : M.sgn b a t⁻¹ = (M.sgn a b t)⁻¹ := by
    rw [eq_inv_iff_mul_eq_one, mul_comm]; exact h1.symm
  rw [h2]
  rcases Int.units_eq_one_or (M.sgn a b t) with e | e <;> rw [e] <;> decide

end OrKeyModel

/-- The transporter realising the minimum of `canon` (the minimum is attained). -/
lemma exists_pick (x : SL(2, ℤ) × V) :
    ∃ t ∈ M.S x.2 (M.rep x.2), lexKey (x.1 * t) = (canon M.toKeyModel x).2 := by
  obtain ⟨t, ht, hk⟩ := Finset.mem_image.1
    ((cands M.toKeyModel x).min'_mem (cands_nonempty M.toKeyModel x))
  exact ⟨t, ht, hk⟩

/-- The picked transporter `t ∈ S(σ, σ̂)` with `lexKey (γ t)` minimal. -/
noncomputable def pick (x : SL(2, ℤ) × V) : SL(2, ℤ) := Classical.choose (exists_pick M x)

lemma pick_mem (x : SL(2, ℤ) × V) : pick M x ∈ M.S x.2 (M.rep x.2) := (Classical.choose_spec (exists_pick M x)).1

lemma pick_key (x : SL(2, ℤ) × V) : lexKey (x.1 * pick M x) = (canon M.toKeyModel x).2 :=
  (Classical.choose_spec (exists_pick M x)).2

/-- **The oriented canonical key**: `(canon, sgn(σ, σ̂, t_min))`, i.e. `[γ, σ] = sign · [γ t_min, σ̂]`. -/
noncomputable def orCanon (x : SL(2, ℤ) × V) : (V × KeyT) × ℤˣ :=
  (canon M.toKeyModel x, M.sgn x.2 (M.rep x.2) (pick M x))

/-- **The oriented key model is faithful**: the key part is a complete invariant of the simplex, and on related pairs
`(γ, σ) ~ (γ', σ')` the signs differ by the transporter's sign `sgn(σ, σ', γ⁻¹γ')`. -/
theorem orCanon_faithful (x y : SL(2, ℤ) × V) :
    ((orCanon M x).1 = (orCanon M y).1 ↔ Rel M.toKeyModel x y) ∧
      (Rel M.toKeyModel x y → (orCanon M x).2 = M.sgn x.2 y.2 (x.1⁻¹ * y.1) * (orCanon M y).2) := by
  refine ⟨canon_faithful M.toKeyModel x y, fun h => ?_⟩
  have hc : canon M.toKeyModel x = canon M.toKeyModel y := (canon_faithful M.toKeyModel x y).2 h
  have hr : M.rep x.2 = M.rep y.2 := M.rep_eq ⟨_, h⟩
  -- equal minima: `x.1 * t_x = y.1 * t_y`
  have hk : x.1 * pick M x = y.1 * pick M y := by
    apply lexKey_injective; rw [pick_key, pick_key, hc]
  have htx : pick M x = (x.1⁻¹ * y.1) * pick M y := by
    rw [mul_assoc, ← hk]; group
  have hy := pick_mem M y
  rw [← hr] at hy
  show M.sgn x.2 (M.rep x.2) (pick M x) = _ * M.sgn y.2 (M.rep y.2) (pick M y)
  rw [htx, M.sgn_mul (t := x.1⁻¹ * y.1) h hy, hr]

end GL2.Ku
