import GL2.Model.Model7

/-!
# M3-b (L19-3): H4(ii) on the seven-vertex model -- `model7_acyclic`

[GT] `prop:model7h4-ii-0`, `prop:model7h4-ii-1`, `prop:model7h4-ii-2`, `prop:model7h4-ii`, mirroring [C] v1.17.0
`prop:model7h4`(ii) (contractibility, used through `eq:contractibleuse`).

`XBulk.Acyclic model7` is a statement about the **infinite** `Γ`-complex `X_bulk` (the chains are finitely supported
on `SL(2, ℤ) × V_q` modulo the oriented relation), so `decide` checks only the local tables of the dump; the global
steps are inductions over `SL(2, ℤ)` or over a height.

* **Coefficients (generic).** `coef q k : C_q → ℤ` reads the coefficient of a chain at the simplex class with key `k`
  (`Ku.orCanon`); `eq_zero_of_coef` (a chain with all coefficients `0` is `0`) and `isolate` (a functional that sees
  one class only reads its coefficient) make `C_q` usable as the free module on the simplex classes.
* **`model7_contract0`** (`H̃₀ = 0`): paths inside `K`, the identities `T v₀ = v₃`, `S v₆ = v₆`, and
  `Subgroup.closure_induction` over mathlib's `SL2Z_generators`.
* **`model7_contract1`** (`H₁ = 0`). Step A (`stepA`): across the five triangles every `1`-chain is homologous to a
  chain on `Y'`, the Serre tree `{γ e₉}` plus one hair `e₁, e₄, e₇` at each horocycle vertex.  Step B (`tree`): `Y'` carries
  no cycle.  The height (`Hn`) is Conway's topograph height for the hexagonal form `Q(x, y) = x² + xy + y²` on the
  columns `u, w` of `γ`: `2 (Q u + Q w + Q (w - u))` at `γ v₄` (a superbase), `4 (Q u + Q w)` at `γ v₆` (a base), one
  more at a hair vertex.  The link lemmas `tp4`, `tp6` say that at every vertex at most one edge class goes to a vertex
  not above it (two would force `Q = 0` on a column at `γ v₄`, and `2 B(u, w) = 0` at `γ v₆`, impossible since
  `2 B(u, w) ≡ det = 1 (mod 2)`).  So at the top vertex of a non-zero cycle only one edge class with a non-zero
  coefficient meets it, and the cycle condition there fails.
* **`model7_contract2`** (`Z₂ = 0`): the horocycle edges `e₀, e₂, e₅` lie in one triangle class each, which kills
  `f₀, f₁, f₃`; then `e₃ ⊂ f₀, f₂` and `e₆ ⊂ f₃, f₄` kill `f₂, f₄` (`kill`).
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.Ku

namespace XBulk

variable (D : XBulkDatum)

/-! ### Coefficients of a chain at a simplex (generic) -/

/-- The class key of `(γ, σ)`: equal keys are exactly related pairs (`orCanon_faithful`). -/
def okey (q : ℤ) (a : SL(2, ℤ) × D.V q) : D.V q × KeyT := (orCanon (D.M q) a).1

/-- The orientation of `(γ, σ)` against its class representative. -/
def osg (q : ℤ) (a : SL(2, ℤ) × D.V q) : ℤ := (((orCanon (D.M q) a).2 : ℤˣ) : ℤ)

lemma osg_mul_self (q : ℤ) (a : SL(2, ℤ) × D.V q) : osg D q a * osg D q a = 1 := by
  unfold osg; rw [← Units.val_mul, sgn_sq, Units.val_one]

lemma okey_eq_iff (q : ℤ) (a b : SL(2, ℤ) × D.V q) : okey D q a = okey D q b ↔ Rel (D.M q).toKeyModel a b :=
  (orCanon_faithful (D.M q) a b).1

lemma osg_rel (q : ℤ) {a b : SL(2, ℤ) × D.V q} (h : Rel (D.M q).toKeyModel a b) :
    osg D q a = (((D.M q).sgn a.2 b.2 (a.1⁻¹ * b.1) : ℤˣ) : ℤ) * osg D q b := by
  unfold osg; rw [(orCanon_faithful (D.M q) a b).2 h, Units.val_mul]

/-- The coefficient functional on the free module. -/
def coefF (q : ℤ) (k : D.V q × KeyT) : (SL(2, ℤ) × D.V q →₀ ℤ) →ₗ[ℤ] ℤ :=
  Finsupp.linearCombination ℤ (fun a => if okey D q a = k then osg D q a else 0)

lemma coefF_gen (q : ℤ) (k : D.V q × KeyT) (x : OrPt (D.V q)) :
    coefF D q k (gen x) = (x.2.2 : ℤ) * (if okey D q (x.1, x.2.1) = k then osg D q (x.1, x.2.1) else 0) := by
  simp [coefF, gen, Finsupp.linearCombination_single]

lemma relSub_le_coefF (q : ℤ) (k : D.V q × KeyT) : relSub (D.M q) ≤ LinearMap.ker (coefF D q k) := by
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, LinearMap.mem_ker, map_sub, coefF_gen, sub_eq_zero]
  have hr : Rel (D.M q).toKeyModel (x.1, x.2.1) (y.1, y.2.1) := h.1
  rw [(okey_eq_iff D q _ _).2 hr]
  split_ifs
  · rw [osg_rel D q hr, h.2]
    set s := (D.M q).sgn x.2.1 y.2.1 (x.1⁻¹ * y.1)
    have hs : (s : ℤ) * s = 1 := by rw [← Units.val_mul, sgn_sq, Units.val_one]
    simp only [Units.val_mul]
    linear_combination ((y.2.2 : ℤ) * osg D q (y.1, y.2.1)) * hs
  · simp

/-- **The coefficient** of a chain at the simplex class `k`. -/
def coef (q : ℤ) (k : D.V q × KeyT) : chains D q →ₗ[ℤ] ℤ :=
  (relSub (D.M q)).liftQ (coefF D q k) (relSub_le_coefF D q k)

lemma coef_genQ (q : ℤ) (k : D.V q × KeyT) (a : SL(2, ℤ) × D.V q) :
    coef D q k (genQ D q a) = if okey D q a = k then osg D q a else 0 := by
  show coefF D q k (gen _) = _
  rw [coefF_gen]; simp

lemma coef_mk (q : ℤ) (k : D.V q × KeyT) (F : SL(2, ℤ) × D.V q →₀ ℤ) :
    coef D q k (Submodule.Quotient.mk F) = ∑ a ∈ F.support with okey D q a = k, F a * osg D q a := by
  show coefF D q k F = _
  rw [coefF, Finsupp.linearCombination_apply, Finsupp.sum, Finset.sum_filter]
  refine Finset.sum_congr rfl fun a _ => ?_
  split_ifs <;> simp

/-- Related pairs have proportional generators. -/
lemma genQ_eq_of_okey (q : ℤ) {a b : SL(2, ℤ) × D.V q} (h : okey D q a = okey D q b) :
    genQ D q a = (osg D q a * osg D q b) • genQ D q b := by
  have hr := (okey_eq_iff D q a b).1 h
  set s := (D.M q).sgn a.2 b.2 (a.1⁻¹ * b.1)
  have hor : OrRel (D.M q) ((a.1, a.2, 1) : OrPt (D.V q)) (b.1, b.2, s) := ⟨hr, by simp only; rw [sgn_sq]⟩
  have e1 := mk_gen_eq (D.M q) hor
  rw [Submodule.mkQ_apply, Submodule.mkQ_apply] at e1
  have hos : osg D q a * osg D q b = s := by
    rw [osg_rel D q hr, mul_assoc, osg_mul_self, mul_one]
  rw [hos, genQ, e1, genQ, ← Submodule.Quotient.mk_smul]
  congr 1
  simp [gen, Finsupp.smul_single]

lemma mk_eq_sum (q : ℤ) (F : SL(2, ℤ) × D.V q →₀ ℤ) :
    (Submodule.Quotient.mk F : chains D q) = ∑ a ∈ F.support, F a • genQ D q a := by
  conv_lhs => rw [← Finsupp.sum_single F]
  rw [Finsupp.sum, quot_mk_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [genQ, ← Submodule.Quotient.mk_smul]; congr 1; simp [gen, Finsupp.smul_single]

variable {M : Type*} [AddCommGroup M]

/-- One fibre of the expansion: the summands of one simplex class. -/
lemma fibre_sum (q : ℤ) (F : SL(2, ℤ) × D.V q →₀ ℤ) (ψ : chains D q →+ M) (b : SL(2, ℤ) × D.V q) :
    ∑ a ∈ F.support with okey D q a = okey D q b, F a • ψ (genQ D q a) =
      (coef D q (okey D q b) (Submodule.Quotient.mk F) * osg D q b) • ψ (genQ D q b) := by
  rw [coef_mk, Finset.sum_mul, Finset.sum_smul]
  refine Finset.sum_congr rfl fun a ha => ?_
  rw [Finset.mem_filter] at ha
  rw [genQ_eq_of_okey D q ha.2, map_zsmul, smul_smul, mul_assoc]

lemma expand (q : ℤ) (F : SL(2, ℤ) × D.V q →₀ ℤ) (ψ : chains D q →+ M) :
    ψ (Submodule.Quotient.mk F) = ∑ k ∈ F.support.image (okey D q),
      ∑ a ∈ F.support with okey D q a = k, F a • ψ (genQ D q a) := by
  rw [Finset.sum_fiberwise_of_maps_to (fun a ha => Finset.mem_image_of_mem _ ha), mk_eq_sum, map_sum]
  simp [map_zsmul]

/-- A chain all of whose class coefficients vanish is `0`. -/
lemma eq_zero_of_coef (q : ℤ) (F : SL(2, ℤ) × D.V q →₀ ℤ)
    (h : ∀ a ∈ F.support, coef D q (okey D q a) (Submodule.Quotient.mk F) = 0) :
    (Submodule.Quotient.mk F : chains D q) = 0 := by
  have := expand D q F (AddMonoidHom.id _)
  rw [AddMonoidHom.id_apply] at this
  rw [this]
  refine Finset.sum_eq_zero fun k hk => ?_
  obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 hk
  rw [fibre_sum, h b hb, zero_mul, zero_smul]

/-- **Isolation**: if `a₀` is the only class with a non-zero coefficient that `ψ` sees, `ψ` reads that coefficient. -/
lemma isolate (q : ℤ) (F : SL(2, ℤ) × D.V q →₀ ℤ) (ψ : chains D q →+ ℤ) (a0 : SL(2, ℤ) × D.V q)
    (ha0 : a0 ∈ F.support)
    (h : ∀ a ∈ F.support, coef D q (okey D q a) (Submodule.Quotient.mk F) ≠ 0 → ψ (genQ D q a) ≠ 0 →
      okey D q a = okey D q a0) :
    ψ (Submodule.Quotient.mk F) =
      coef D q (okey D q a0) (Submodule.Quotient.mk F) * osg D q a0 * ψ (genQ D q a0) := by
  rw [expand, Finset.sum_eq_single (okey D q a0)]
  · rw [fibre_sum, smul_eq_mul]
  · intro k hk hne
    obtain ⟨b, hb, rfl⟩ := Finset.mem_image.1 hk
    rw [fibre_sum]
    by_cases hc : coef D q (okey D q b) (Submodule.Quotient.mk F) = 0
    · rw [hc, zero_mul, zero_smul]
    · by_cases hψ : ψ (genQ D q b) = 0
      · rw [hψ, smul_zero]
      · exact absurd (h b hb hc hψ) hne
  · intro h0; exact absurd (Finset.mem_image_of_mem _ ha0) h0

/-- Vertices carry no orientation: `osg = 1` in degree `0`. -/
lemma osg_zero (a : SL(2, ℤ) × D.V 0) : osg D 0 a = 1 := by
  unfold osg orCanon
  simp only
  rw [D.sgn0 (pick_mem (D.M 0) a), Units.val_one]

end XBulk

open GL2.TauChk GL2.TauChk.Data XBulk

namespace M7

lemma hX : tauP2.XAx := of_decide_eq_true tauP2_xbulkB

/-! ### Local tables of the dump (`decide +kernel`) -/

lemma mem_Sset_of_g {q : ℤ} {a b : ℕ} {g : M22} (h : g ∈ (tauP2.trSetZ q a b).map Tr.g) :
    M22.toSL g ∈ tauP2.Sset q a b := by
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 h
  exact (tauP2.mem_Sset).2 ⟨t, ht, rfl⟩

lemma Sset_eq_empty {q : ℤ} {a b : ℕ} (h : tauP2.trSetZ q a b = []) : tauP2.Sset q a b = ∅ := by
  simp [TauData.Sset, h]

lemma Sset_sub {q q' : ℤ} {a b a' b' : ℕ}
    (h : ∀ g ∈ (tauP2.trSetZ q a b).map Tr.g, g ∈ (tauP2.trSetZ q' a' b').map Tr.g) :
    tauP2.Sset q a b ⊆ tauP2.Sset q' a' b' := by
  intro γ hγ
  obtain ⟨t, ht, rfl⟩ := (tauP2.mem_Sset).1 hγ
  exact mem_Sset_of_g (h _ (List.mem_map_of_mem ht))

lemma Sset_cases {q : ℤ} {a b : ℕ} {γ : SL(2, ℤ)} {l : List M22} (hl : (tauP2.trSetZ q a b).map Tr.g = l)
    (h : γ ∈ tauP2.Sset q a b) : ∃ g ∈ l, γ = M22.toSL g := by
  obtain ⟨t, ht, rfl⟩ := (tauP2.mem_Sset).1 h
  exact ⟨t.g, hl ▸ List.mem_map_of_mem ht, rfl⟩

/-- `genQ (γ, σ) = genQ (γ g, σ')` for a stored transporter `g ∈ S(σ, σ')` of sign `+1`. -/
lemma genQ_tr (q : ℤ) (a b : model7.V q) (g : M22)
    (h : ∃ t ∈ tauP2.trSetZ q a.1 b.1, t.g = g ∧ permSign t.perm q.toNat = 1) (γ : SL(2, ℤ)) :
    genQ model7 q (γ, a) = genQ model7 q (γ * M22.toSL g, b) := by
  obtain ⟨t, ht, rfl, hs⟩ := h
  have hor : OrRel (model7.M q) ((γ, a, 1) : OrPt (model7.V q)) (γ * M22.toSL t.g, b, 1) := by
    refine ⟨?_, ?_⟩
    · show γ⁻¹ * (γ * M22.toSL t.g) ∈ tauP2.Sset q a.1 b.1
      rw [inv_mul_cancel_left]; exact (tauP2.mem_Sset).2 ⟨t, ht, rfl⟩
    · show (1 : ℤˣ) = tauP2.sgnZ q a.1 b.1 (γ⁻¹ * (γ * M22.toSL t.g)) * 1
      rw [inv_mul_cancel_left, TauData.sgnZ_eq (TauData.keyAx_all hX q) ht, hs, unitOf_one, mul_one]
  have := mk_gen_eq (model7.M q) hor
  rw [Submodule.mkQ_apply, Submodule.mkQ_apply] at this
  exact this

lemma toSL_T : M22.toSL ⟨1, 1, 0, 1⟩ = ModularGroup.T := by
  ext i j; rw [M22.toSL_coe (by decide)]; fin_cases i <;> fin_cases j <;> simp [M22.toMat, ModularGroup.coe_T]

lemma toSL_S : M22.toSL ⟨0, -1, 1, 0⟩ = ModularGroup.S := by
  ext i j; rw [M22.toSL_coe (by decide)]; fin_cases i <;> fin_cases j <;> simp [M22.toMat, ModularGroup.coe_S]

/-! ### Boundaries of the model edges and triangles -/

lemma bd_edge (γ : SL(2, ℤ)) (e : model7.V (0 + 1)) (h0 : tauP2.face 1 e.1 0 ∈ tauP2.idxF 0)
    (h1 : tauP2.face 1 e.1 1 ∈ tauP2.idxF 0) :
    bd model7 0 (genQ model7 (0 + 1) (γ, e)) =
      genQ model7 0 (γ, ⟨_, h0⟩) - genQ model7 0 (γ, ⟨_, h1⟩) := by
  show Submodule.Quotient.mk (bdF model7 0 (gen _)) = _
  rw [bdF_gen]
  show Submodule.Quotient.mk ((1 : ℤ) • ∑ j : Fin 2, _) = _
  rw [Fin.sum_univ_two, one_smul, Submodule.Quotient.mk_add]
  show (Submodule.Quotient.mk (gen ((γ * 1, ⟨tauP2.face 1 e.1 0, h0⟩, unitOf (sgnPow 0)) : OrPt (model7.V 0))) :
      chains model7 0) + Submodule.Quotient.mk (gen ((γ * 1, ⟨tauP2.face 1 e.1 1, h1⟩, unitOf (sgnPow 1)) :
      OrPt (model7.V 0))) = _
  have e0 : unitOf (sgnPow 0) = 1 := by decide
  have e1 : unitOf (sgnPow 1) = -1 := by decide
  rw [mul_one, e0, e1, sub_eq_add_neg, genQ, genQ, ← Submodule.Quotient.mk_neg, ← gen_neg]
  rfl

lemma bd_tri (γ : SL(2, ℤ)) (f : model7.V (0 + 1 + 1)) (h0 : tauP2.face 2 f.1 0 ∈ tauP2.idxF (0 + 1))
    (h1 : tauP2.face 2 f.1 1 ∈ tauP2.idxF (0 + 1)) (h2 : tauP2.face 2 f.1 2 ∈ tauP2.idxF (0 + 1)) :
    bd model7 (0 + 1) (genQ model7 (0 + 1 + 1) (γ, f)) =
      genQ model7 (0 + 1) (γ, ⟨_, h0⟩) - genQ model7 (0 + 1) (γ, ⟨_, h1⟩) + genQ model7 (0 + 1) (γ, ⟨_, h2⟩) := by
  show Submodule.Quotient.mk (bdF model7 (0 + 1) (gen _)) = _
  rw [bdF_gen]
  show Submodule.Quotient.mk ((1 : ℤ) • ∑ j : Fin 3, _) = _
  rw [Fin.sum_univ_three, one_smul, Submodule.Quotient.mk_add, Submodule.Quotient.mk_add]
  show (Submodule.Quotient.mk (gen ((γ * 1, ⟨tauP2.face 2 f.1 0, h0⟩, unitOf (sgnPow 0)) :
      OrPt (model7.V (0 + 1)))) : chains model7 (0 + 1)) +
      Submodule.Quotient.mk (gen ((γ * 1, ⟨tauP2.face 2 f.1 1, h1⟩, unitOf (sgnPow 1)) : OrPt (model7.V (0 + 1)))) +
      Submodule.Quotient.mk (gen ((γ * 1, ⟨tauP2.face 2 f.1 2, h2⟩, unitOf (sgnPow 2)) : OrPt (model7.V (0 + 1)))) = _
  have e0 : unitOf (sgnPow 0) = 1 := by decide
  have e1 : unitOf (sgnPow 1) = -1 := by decide
  have e2 : unitOf (sgnPow 2) = 1 := by decide
  rw [mul_one, e0, e1, e2, sub_eq_add_neg, genQ, genQ, genQ, ← Submodule.Quotient.mk_neg, ← gen_neg]
  rfl

/-- A model edge given by its index. -/
abbrev ed (n : ℕ) (h : n ∈ tauP2.idxF (0 + 1) := by decide +kernel) : model7.V (0 + 1) := ⟨n, h⟩
/-- A model vertex given by its index. -/
abbrev vx (n : ℕ) (h : n ∈ tauP2.idxF 0 := by decide +kernel) : model7.V 0 := ⟨n, h⟩
/-- A model triangle given by its index. -/
abbrev tri (n : ℕ) (h : n ∈ tauP2.idxF (0 + 1 + 1) := by decide +kernel) : model7.V (0 + 1 + 1) := ⟨n, h⟩

lemma bd_ed (γ : SL(2, ℤ)) (n a b : ℕ) (hn : n ∈ tauP2.idxF (0 + 1)) (ha : a ∈ tauP2.idxF 0)
    (hb : b ∈ tauP2.idxF 0) (h0 : tauP2.face 1 n 0 = b) (h1 : tauP2.face 1 n 1 = a) :
    bd model7 0 (genQ model7 (0 + 1) (γ, ⟨n, hn⟩)) = genQ model7 0 (γ, ⟨b, hb⟩) - genQ model7 0 (γ, ⟨a, ha⟩) := by
  subst h0 h1; exact bd_edge γ ⟨n, hn⟩ hb ha

lemma bd_tr (γ : SL(2, ℤ)) (n x y z : ℕ) (hn : n ∈ tauP2.idxF (0 + 1 + 1)) (hx : x ∈ tauP2.idxF (0 + 1))
    (hy : y ∈ tauP2.idxF (0 + 1)) (hz : z ∈ tauP2.idxF (0 + 1)) (h0 : tauP2.face 2 n 0 = x)
    (h1 : tauP2.face 2 n 1 = y) (h2 : tauP2.face 2 n 2 = z) :
    bd model7 (0 + 1) (genQ model7 (0 + 1 + 1) (γ, ⟨n, hn⟩)) =
      genQ model7 (0 + 1) (γ, ⟨x, hx⟩) - genQ model7 (0 + 1) (γ, ⟨y, hy⟩) + genQ model7 (0 + 1) (γ, ⟨z, hz⟩) := by
  subst h0 h1 h2; exact bd_tri γ ⟨n, hn⟩ hx hy hz

/-! ### `contract0`: `X_bulk` is connected -/

lemma aug_genQ (a : SL(2, ℤ) × model7.V 0) : aug model7 (genQ model7 0 a) = 1 := by
  show augF model7 (gen _) = 1
  rw [augF_gen]; rfl

lemma vtx_cases (σ : model7.V 0) : σ.1 = 0 ∨ σ.1 = 1 ∨ σ.1 = 2 ∨ σ.1 = 3 ∨ σ.1 = 4 ∨ σ.1 = 5 ∨ σ.1 = 6 := by
  have h : ∀ n ∈ tauP2.idxF 0, n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 := by decide +kernel
  exact h σ.1 σ.2

/-- The path inside `K` from `v₀` to every vertex. -/
lemma path_K (γ : SL(2, ℤ)) (σ : model7.V 0) :
    genQ model7 0 (γ, σ) - genQ model7 0 (γ, vx 0) ∈ LinearMap.range (bd model7 0) := by
  have E : ∀ n a b (hn : n ∈ tauP2.idxF (0 + 1)) (ha : a ∈ tauP2.idxF 0) (hb : b ∈ tauP2.idxF 0),
      tauP2.face 1 n 0 = b → tauP2.face 1 n 1 = a →
      genQ model7 0 (γ, ⟨b, hb⟩) - genQ model7 0 (γ, ⟨a, ha⟩) ∈ LinearMap.range (bd model7 0) :=
    fun n a b hn ha hb h0 h1 => ⟨_, bd_ed γ n a b hn ha hb h0 h1⟩
  have e0 := E 0 0 1 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have e1 := E 1 0 4 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have e2 := E 2 1 2 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have e4 := E 4 1 6 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have e5 := E 5 2 3 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have e6 := E 6 2 5 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
  obtain ⟨n, hn⟩ := σ
  rcases vtx_cases ⟨n, hn⟩ with h | h | h | h | h | h | h <;> simp only at h <;> subst h
  · rw [sub_self]; exact Submodule.zero_mem _
  · exact e0
  · have := Submodule.add_mem _ e2 e0; rwa [sub_add_sub_cancel] at this
  · have := Submodule.add_mem _ e5 (Submodule.add_mem _ e2 e0); rwa [sub_add_sub_cancel, sub_add_sub_cancel] at this
  · exact e1
  · have := Submodule.add_mem _ e6 (Submodule.add_mem _ e2 e0); rwa [sub_add_sub_cancel, sub_add_sub_cancel] at this
  · have := Submodule.add_mem _ e4 e0; rwa [sub_add_sub_cancel] at this

lemma range_smul {γ : SL(2, ℤ)} {z : chains model7 0} (h : z ∈ LinearMap.range (bd model7 0)) :
    γ • z ∈ LinearMap.range (bd model7 0) := by
  obtain ⟨y, rfl⟩ := h; exact ⟨γ • y, bd_equiv _ _ _ _⟩

/-- Every translate of `v₀` is joined to `v₀`: induction over the generators `S`, `T` (`SL2Z_generators`). -/
lemma path_v0 (γ : SL(2, ℤ)) :
    genQ model7 0 (γ, vx 0) - genQ model7 0 (1, vx 0) ∈ LinearMap.range (bd model7 0) := by
  have hγ : γ ∈ Subgroup.closure {ModularGroup.S, ModularGroup.T} := by
    rw [_root_.SpecialLinearGroup.SL2Z_generators]; trivial
  induction hγ using Subgroup.closure_induction with
  | mem x hx =>
    rcases hx with rfl | rfl
    · -- `S` fixes `v₆`
      have h6 : genQ model7 0 (1, vx 6) = genQ model7 0 (1 * ModularGroup.S, vx 6) := by
        rw [← toSL_S]; exact genQ_tr 0 (vx 6) (vx 6) _ (by decide +kernel) 1
      rw [one_mul] at h6
      have h1 := path_K ModularGroup.S (vx 6)
      have h2 := path_K 1 (vx 6)
      rw [← h6] at h1
      have := Submodule.sub_mem _ h2 h1
      rwa [sub_sub_sub_cancel_left] at this
    · -- `T v₀ = v₃`
      have h3 : genQ model7 0 (1, vx 3) = genQ model7 0 (1 * ModularGroup.T, vx 0) := by
        rw [← toSL_T]; exact genQ_tr 0 (vx 3) (vx 0) _ (by decide +kernel) 1
      rw [one_mul] at h3
      rw [← h3]; exact path_K 1 (vx 3)
  | one => rw [sub_self]; exact Submodule.zero_mem _
  | mul x y _ _ hx hy =>
    have := Submodule.add_mem _ (range_smul (γ := x) hy) hx
    rwa [smul_sub, smul_genQ, smul_genQ, mul_one, sub_add_sub_cancel] at this
  | inv x _ hx =>
    have := Submodule.neg_mem _ (range_smul (γ := x⁻¹) hx)
    rwa [smul_sub, smul_genQ, smul_genQ, inv_mul_cancel, mul_one, neg_sub] at this

/-- **`contract0` on the model**: `H̃₀(X_bulk) = 0`. -/
theorem _root_.GL2.Model.model7_contract0 : ∀ x : chains model7 0, aug model7 x = 0 → ∃ y, bd model7 0 y = x := by
  have key : ∀ x : chains model7 0, x - aug model7 x • genQ model7 0 (1, vx 0) ∈ LinearMap.range (bd model7 0) := by
    intro x
    induction x using Submodule.Quotient.induction_on with
    | _ F =>
      induction F using Finsupp.induction_linear with
      | zero => simp
      | add f g hf hg =>
        have := Submodule.add_mem _ hf hg
        rw [Submodule.Quotient.mk_add, map_add, add_smul]
        convert this using 1; abel
      | single a c =>
        have e : (Submodule.Quotient.mk (Finsupp.single a c) : chains model7 0) = c • genQ model7 0 a := by
          rw [genQ, ← Submodule.Quotient.mk_smul]; congr 1; simp [gen, Finsupp.smul_single]
        rw [e, map_zsmul, aug_genQ, smul_eq_mul, mul_one, ← smul_sub]
        refine Submodule.smul_of_tower_mem _ c ?_
        have := Submodule.add_mem _ (path_K a.1 a.2) (path_v0 a.1)
        rwa [sub_add_sub_cancel] at this
  intro x hx
  have := key x
  rwa [hx, zero_smul, sub_zero] at this

/-! ### `contract1`, step A: every `1`-chain is a chain on `Y'` modulo boundaries

`Y'` is the Serre tree (the edges `γ e₉ = [γ v₄, γ v₆]`) with one hair at each horocycle vertex (`e₁ = [v₀, v₄]`,
`e₄ = [v₁, v₆]`, `e₇ = [v₂, v₆]`); the other edges are pushed onto it across the five triangles. -/

/-- The edge indices of `Y'`. -/
def Yidx : Finset ℕ := {1, 4, 7, 9}

/-- The pairs `(γ, e)` with `e` an edge of `Y'`. -/
def E1 : Set (SL(2, ℤ) × model7.V (0 + 1)) := {a | a.2.1 ∈ Yidx}

/-- The chains on `Y'`. -/
def Yp : Submodule ℤ (chains model7 (0 + 1)) :=
  (Finsupp.supported ℤ ℤ E1).map (relSub (model7.M (0 + 1))).mkQ

lemma genQ_mem_Yp (a : SL(2, ℤ) × model7.V (0 + 1)) (ha : a.2.1 ∈ Yidx) : genQ model7 (0 + 1) a ∈ Yp :=
  ⟨Finsupp.single a 1, Finsupp.single_mem_supported ℤ _ ha, by simp [genQ, gen]⟩

lemma edge_cases (e : model7.V (0 + 1)) : e.1 = 0 ∨ e.1 = 1 ∨ e.1 = 2 ∨ e.1 = 3 ∨ e.1 = 4 ∨ e.1 = 5 ∨ e.1 = 6 ∨
    e.1 = 7 ∨ e.1 = 8 ∨ e.1 = 9 ∨ e.1 = 10 := by
  have h : ∀ n ∈ tauP2.idxF (0 + 1), n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨
    n = 9 ∨ n = 10 := by decide +kernel
  exact h e.1 e.2

lemma stepA_gen (γ : SL(2, ℤ)) (e : model7.V (0 + 1)) :
    genQ model7 (0 + 1) (γ, e) ∈ Yp ⊔ LinearMap.range (bd model7 (0 + 1)) := by
  set P := Yp ⊔ LinearMap.range (bd model7 (0 + 1))
  have hY : ∀ n (hn : n ∈ tauP2.idxF (0 + 1)), n ∈ Yidx → genQ model7 (0 + 1) (γ, ⟨n, hn⟩) ∈ P :=
    fun n hn h => Submodule.mem_sup_left (genQ_mem_Yp _ h)
  have hB : ∀ n x y z (hn : n ∈ tauP2.idxF (0 + 1 + 1)) (hx : x ∈ tauP2.idxF (0 + 1)) (hy : y ∈ tauP2.idxF (0 + 1))
      (hz : z ∈ tauP2.idxF (0 + 1)), tauP2.face 2 n 0 = x → tauP2.face 2 n 1 = y → tauP2.face 2 n 2 = z →
      genQ model7 (0 + 1) (γ, ⟨x, hx⟩) - genQ model7 (0 + 1) (γ, ⟨y, hy⟩) + genQ model7 (0 + 1) (γ, ⟨z, hz⟩) ∈ P :=
    fun n x y z hn hx hy hz h0 h1 h2 => Submodule.mem_sup_right ⟨_, bd_tr γ n x y z hn hx hy hz h0 h1 h2⟩
  have g1 := hY 1 (by decide +kernel) (by decide)
  have g4 := hY 4 (by decide +kernel) (by decide)
  have g7 := hY 7 (by decide +kernel) (by decide)
  have g9 := hY 9 (by decide +kernel) (by decide)
  -- `e₈ = T e₁`, `e₁₀ = S⁻¹ e₉`
  have g8 : genQ model7 (0 + 1) (γ, ed 8) ∈ P := by
    rw [genQ_tr (0 + 1) (ed 8) (ed 1) ⟨1, 1, 0, 1⟩ (by decide +kernel) γ]
    exact Submodule.mem_sup_left (genQ_mem_Yp _ (by simp [Yidx]))
  have g10 : genQ model7 (0 + 1) (γ, ed 10) ∈ P := by
    rw [genQ_tr (0 + 1) (ed 10) (ed 9) ⟨0, 1, -1, 0⟩ (by decide +kernel) γ]
    exact Submodule.mem_sup_left (genQ_mem_Yp _ (by simp [Yidx]))
  -- `∂f₂ = e₉ - e₄ + e₃`, `∂f₄ = e₁₀ - e₇ + e₆`, `∂f₀ = e₃ - e₁ + e₀`, `∂f₁ = e₇ - e₄ + e₂`, `∂f₃ = e₈ - e₆ + e₅`
  have f2 := hB 2 9 4 3 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have f4 := hB 4 10 7 6 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have f0 := hB 0 3 1 0 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have f1 := hB 1 7 4 2 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have f3 := hB 3 8 6 5 (by decide +kernel) (by decide +kernel) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel) (by decide +kernel)
  have g3 : genQ model7 (0 + 1) (γ, ed 3) ∈ P := by
    have := Submodule.sub_mem _ (Submodule.add_mem _ f2 g4) g9
    convert this using 1; abel
  have g6 : genQ model7 (0 + 1) (γ, ed 6) ∈ P := by
    have := Submodule.sub_mem _ (Submodule.add_mem _ f4 g7) g10
    convert this using 1; abel
  have g0 : genQ model7 (0 + 1) (γ, ed 0) ∈ P := by
    have := Submodule.sub_mem _ (Submodule.add_mem _ f0 g1) g3
    convert this using 1; abel
  have g2 : genQ model7 (0 + 1) (γ, ed 2) ∈ P := by
    have := Submodule.sub_mem _ (Submodule.add_mem _ f1 g4) g7
    convert this using 1; abel
  have g5 : genQ model7 (0 + 1) (γ, ed 5) ∈ P := by
    have := Submodule.sub_mem _ (Submodule.add_mem _ f3 g6) g8
    convert this using 1; abel
  obtain ⟨n, hn⟩ := e
  rcases edge_cases ⟨n, hn⟩ with h | h | h | h | h | h | h | h | h | h | h <;> simp only at h <;> subst h <;>
    assumption

lemma stepA (x : chains model7 (0 + 1)) : x ∈ Yp ⊔ LinearMap.range (bd model7 (0 + 1)) := by
  induction x using Submodule.Quotient.induction_on with
  | _ F =>
    induction F using Finsupp.induction_linear with
    | zero => exact Submodule.zero_mem _
    | add f g hf hg => rw [Submodule.Quotient.mk_add]; exact Submodule.add_mem _ hf hg
    | single a c =>
      have e : (Submodule.Quotient.mk (Finsupp.single a c) : chains model7 (0 + 1)) = c • genQ model7 (0 + 1) a := by
        rw [genQ, ← Submodule.Quotient.mk_smul]; congr 1; simp [gen, Finsupp.smul_single]
      rw [e]; exact Submodule.smul_of_tower_mem _ c (stepA_gen a.1 a.2)

/-! ### `contract1`, step B: `Y'` carries no cycle (the topograph height)

On the Serre tree the vertex `γ v₄` is the superbase `{u, w - u, -w}` and `γ v₆` the base `{u, w}` of the columns
`u, w` of `γ` (Conway's topograph).  With the hexagonal form `Q(x, y) = x² + xy + y²` the height
`2 (Q u + Q w + Q (w - u))` at `γ v₄` and `4 (Q u + Q w)` at `γ v₆` has, at every vertex, **at most one** neighbour
of smaller or equal height: at `γ v₄` two of the three would give `Q ≤ 0` on a column, at `γ v₆` both would give
`B(u, w) = 0`, impossible for a basis of `ℤ²` since `2 B(u, w) ≡ det = 1 (mod 2)`.  A hair vertex sits one above its
foot.  So the top vertex of a cycle on `Y'` meets a single edge class, whose coefficient must then vanish. -/

/-- The hexagonal form `Q(x, y) = x² + xy + y²`. -/
def Qf (x y : ℤ) : ℤ := x ^ 2 + x * y + y ^ 2

lemma Qf_pos {x y : ℤ} (h : x ≠ 0 ∨ y ≠ 0) : 0 < Qf x y := by
  unfold Qf
  rcases h with h | h
  · by_cases hy : y = 0
    · subst hy; have := sq_pos_of_ne_zero h; nlinarith
    · nlinarith [sq_nonneg (2 * x + y), sq_pos_of_ne_zero hy]
  · nlinarith [sq_nonneg (2 * x + y), sq_pos_of_ne_zero h]

/-- The height of the vertex `γ ι(v_n)` of `Y'` in terms of `γ = (a b; c d)`.  **Defined on `Y'`'s vertex set
`n ∈ {0, 1, 2, 4, 6}` only**: at `n ∈ {3, 5}` (the vertices `v₃`, `v₅`, not on `Y'`) and at junk indices it returns `0`,
which is never read (the incidences `inc` of `Y'` only involve `v₀, v₁, v₂, v₄, v₆`), so `Hn` is not a height on all
vertices.  At `v₄` / `v₆` it is invariant under the stored stabilisers (`Hn_inv4`, `Hn_inv6`); the hair stabilisers are
`{±I}` and `Hn` is even in `γ`. -/
def hgt (n : ℕ) (a b c d : ℤ) : ℤ :=
  if n = 4 then 2 * (Qf a c + Qf b d + Qf (b - a) (d - c))
  else if n = 6 then 4 * (Qf a c + Qf b d)
  else if n = 0 then 2 * (Qf a c + Qf b d + Qf (b - a) (d - c)) + 1
  else if n = 1 ∨ n = 2 then 4 * (Qf a c + Qf b d) + 1 else 0

/-- `Hn γ n`: the height of `γ ι(v_n)`, for `n ∈ {0, 1, 2, 4, 6}` (`0` elsewhere, see `hgt`). -/
def Hn (γ : SL(2, ℤ)) (n : ℕ) : ℤ :=
  hgt n ((γ : Matrix (Fin 2) (Fin 2) ℤ) 0 0) ((γ : Matrix (Fin 2) (Fin 2) ℤ) 0 1)
    ((γ : Matrix (Fin 2) (Fin 2) ℤ) 1 0) ((γ : Matrix (Fin 2) (Fin 2) ℤ) 1 1)

lemma Hn_0 (γ : SL(2, ℤ)) : Hn γ 0 = Hn γ 4 + 1 := by simp [Hn, hgt]
lemma Hn_1 (γ : SL(2, ℤ)) : Hn γ 1 = Hn γ 6 + 1 := by simp [Hn, hgt]
lemma Hn_2 (γ : SL(2, ℤ)) : Hn γ 2 = Hn γ 6 + 1 := by simp [Hn, hgt]

lemma Hn_mul (δ : SL(2, ℤ)) (g : M22) (hg : g.det = 1) (n : ℕ) :
    Hn (δ * M22.toSL g) n = hgt n
      ((δ : Matrix (Fin 2) (Fin 2) ℤ) 0 0 * g.a + (δ : Matrix (Fin 2) (Fin 2) ℤ) 0 1 * g.c)
      ((δ : Matrix (Fin 2) (Fin 2) ℤ) 0 0 * g.b + (δ : Matrix (Fin 2) (Fin 2) ℤ) 0 1 * g.d)
      ((δ : Matrix (Fin 2) (Fin 2) ℤ) 1 0 * g.a + (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 1 * g.c)
      ((δ : Matrix (Fin 2) (Fin 2) ℤ) 1 0 * g.b + (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 1 * g.d) := by
  unfold Hn
  rw [Matrix.SpecialLinearGroup.coe_mul, M22.toSL_coe hg]
  simp [Matrix.mul_apply, Fin.sum_univ_two, M22.toMat]

lemma det_ent (δ : SL(2, ℤ)) :
    (δ : Matrix (Fin 2) (Fin 2) ℤ) 0 0 * (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 1 -
      (δ : Matrix (Fin 2) (Fin 2) ℤ) 0 1 * (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 0 = 1 := by
  have := δ.2; rwa [Matrix.det_fin_two] at this

lemma L44 : (tauP2.trSetZ 0 4 4).map Tr.g =
    [⟨1, 0, 0, 1⟩, ⟨-1, 0, 0, -1⟩, ⟨0, -1, 1, 1⟩, ⟨-1, -1, 1, 0⟩, ⟨0, 1, -1, -1⟩, ⟨1, 1, -1, 0⟩] := by
  decide +kernel

lemma L66 : (tauP2.trSetZ 0 6 6).map Tr.g = [⟨1, 0, 0, 1⟩, ⟨-1, 0, 0, -1⟩, ⟨0, -1, 1, 0⟩, ⟨0, 1, -1, 0⟩] := by
  decide +kernel

lemma Hn_inv4 {γ δ : SL(2, ℤ)} (h : δ⁻¹ * γ ∈ tauP2.Sset 0 4 4) : Hn γ 4 = Hn δ 4 := by
  obtain ⟨g, hg, he⟩ := Sset_cases L44 h
  have hγ : γ = δ * M22.toSL g := by rw [← he]; group
  subst hγ
  have hdet : g.det = 1 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  rw [Hn_mul δ g hdet]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
  rcases hg with rfl | rfl | rfl | rfl | rfl | rfl <;> simp [Hn, hgt, Qf] <;> ring

lemma Hn_inv6 {γ δ : SL(2, ℤ)} (h : δ⁻¹ * γ ∈ tauP2.Sset 0 6 6) : Hn γ 6 = Hn δ 6 := by
  obtain ⟨g, hg, he⟩ := Sset_cases L66 h
  have hγ : γ = δ * M22.toSL g := by rw [← he]; group
  subst hγ
  have hdet : g.det = 1 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl <;> decide
  rw [Hn_mul δ g hdet]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
  rcases hg with rfl | rfl | rfl | rfl <;> simp [Hn, hgt, Qf] <;> ring

lemma cols_ne (δ : SL(2, ℤ)) :
    ((δ : Matrix (Fin 2) (Fin 2) ℤ) 0 0 ≠ 0 ∨ (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 0 ≠ 0) ∧
    ((δ : Matrix (Fin 2) (Fin 2) ℤ) 0 1 ≠ 0 ∨ (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 1 ≠ 0) := by
  have hd := det_ent δ
  constructor
  · by_contra h; push Not at h; rw [h.1, h.2] at hd; simp at hd
  · by_contra h; push Not at h; rw [h.1, h.2] at hd; simp at hd

/-- **The link of `γ v₄`**: at most one of its three tree neighbours is not above it. -/
lemma tp4 {γ δ : SL(2, ℤ)} (h : δ⁻¹ * γ ∈ tauP2.Sset 0 4 4) (h0 : Hn γ 6 ≤ Hn γ 4) (h1 : Hn δ 6 ≤ Hn γ 4) :
    δ⁻¹ * γ ∈ tauP2.Sset (0 + 1) 9 9 := by
  obtain ⟨g, hg, he⟩ := Sset_cases L44 h
  rw [he]
  have hγ : γ = δ * M22.toSL g := by rw [← he]; group
  subst hγ
  have hdet : g.det = 1 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  rw [Hn_mul δ g hdet, Hn_mul δ g hdet] at h0
  rw [Hn_mul δ g hdet] at h1
  have hu := Qf_pos (cols_ne δ).1
  have hw := Qf_pos (cols_ne δ).2
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
  rcases hg with rfl | rfl | rfl | rfl | rfl | rfl
  · exact mem_Sset_of_g (by decide +kernel)
  · exact mem_Sset_of_g (by decide +kernel)
  all_goals exfalso; simp [Hn, hgt, Qf] at h0 h1 hu hw; nlinarith

/-- **The link of `γ v₆`**: at most one of its two tree neighbours is not above it (`2 B(u, w)` is odd). -/
lemma tp6 {γ δ : SL(2, ℤ)} (h : δ⁻¹ * γ ∈ tauP2.Sset 0 6 6) (h0 : Hn γ 4 ≤ Hn γ 6) (h1 : Hn δ 4 ≤ Hn γ 6) :
    δ⁻¹ * γ ∈ tauP2.Sset (0 + 1) 9 9 := by
  obtain ⟨g, hg, he⟩ := Sset_cases L66 h
  rw [he]
  have hγ : γ = δ * M22.toSL g := by rw [← he]; group
  subst hγ
  have hdet : g.det = 1 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
    rcases hg with rfl | rfl | rfl | rfl <;> decide
  rw [Hn_mul δ g hdet, Hn_mul δ g hdet] at h0
  rw [Hn_mul δ g hdet] at h1
  have hd := det_ent δ
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hg
  rcases hg with rfl | rfl | rfl | rfl
  · exact mem_Sset_of_g (by decide +kernel)
  · exact mem_Sset_of_g (by decide +kernel)
  all_goals
    exfalso
    simp [Hn, hgt, Qf] at h0 h1
    set a := (δ : Matrix (Fin 2) (Fin 2) ℤ) 0 0
    set b := (δ : Matrix (Fin 2) (Fin 2) ℤ) 0 1
    set c := (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 0
    set d := (δ : Matrix (Fin 2) (Fin 2) ℤ) 1 1
    have hX : 2 * (a * b + b * c + c * d) + 1 = 0 := by nlinarith
    omega

/-- The incidences of `Y'`: `(edge, endpoint, other endpoint)`. -/
def inc : List (ℕ × ℕ × ℕ) :=
  [(1, 4, 0), (1, 0, 4), (4, 6, 1), (4, 1, 6), (7, 6, 2), (7, 2, 6), (9, 6, 4), (9, 4, 6)]

lemma inc_mem : ∀ p ∈ inc, p.2.1 ∈ tauP2.idxF 0 ∧ p.2.2 ∈ tauP2.idxF 0 := by decide +kernel

lemma inc_face : ∀ p ∈ inc, (tauP2.face 1 p.1 0 = p.2.1 ∧ tauP2.face 1 p.1 1 = p.2.2) ∨
    (tauP2.face 1 p.1 0 = p.2.2 ∧ tauP2.face 1 p.1 1 = p.2.1) := by decide +kernel

lemma inc_off : ∀ p ∈ inc, tauP2.trSetZ 0 p.2.2 p.2.1 = [] := by decide +kernel

lemma Yidx_inc : ∀ n ∈ Yidx, (n, tauP2.face 1 n 0, tauP2.face 1 n 1) ∈ inc ∧
    (n, tauP2.face 1 n 1, tauP2.face 1 n 0) ∈ inc := by decide +kernel

lemma Yidx_face : ∀ n ∈ Yidx, tauP2.face 1 n 0 ∈ tauP2.idxF 0 ∧ tauP2.face 1 n 1 ∈ tauP2.idxF 0 := by
  decide +kernel

/-- **The tree property** of `Y'` (the local link lemma): two `Y'`-edges at a common vertex `w`, both of whose
other endpoints are not above `w`, are the same edge. -/
lemma tp (p0 p : ℕ × ℕ × ℕ) (hp0 : p0 ∈ inc) (hp : p ∈ inc) {γ δ : SL(2, ℤ)}
    (hrel : δ⁻¹ * γ ∈ tauP2.Sset 0 p.2.1 p0.2.1) (h0 : Hn γ p0.2.2 ≤ Hn γ p0.2.1) (h1 : Hn δ p.2.2 ≤ Hn γ p0.2.1) :
    δ⁻¹ * γ ∈ tauP2.Sset (0 + 1) p.1 p0.1 := by
  simp only [inc, List.mem_cons, List.not_mem_nil, or_false] at hp0 hp
  rcases hp0 with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
  dsimp only at hrel h0 h1 ⊢ <;>
  first
  | (exfalso; rw [Sset_eq_empty (by decide +kernel)] at hrel; exact Finset.notMem_empty _ hrel)
  | exact Sset_sub (by decide +kernel) hrel
  | (exfalso; linarith [Hn_0 γ, Hn_1 γ, Hn_2 γ])
  | (exfalso; linarith [Hn_inv4 hrel, Hn_0 δ])
  | (exfalso; linarith [Hn_inv6 hrel, Hn_1 δ, Hn_2 δ])
  | exact tp4 hrel h0 h1
  | exact tp6 hrel h0 h1

lemma osg_ne_zero (q : ℤ) (a : SL(2, ℤ) × model7.V q) : osg model7 q a ≠ 0 := by
  intro h; have := osg_mul_self model7 q a; rw [h, zero_mul] at this; exact zero_ne_one this

/-- **Step B**: a cycle on `Y'` is zero. -/
theorem tree (F : SL(2, ℤ) × model7.V (0 + 1) →₀ ℤ) (hF : F ∈ Finsupp.supported ℤ ℤ E1)
    (hbd : bd model7 0 (Submodule.Quotient.mk F) = 0) : (Submodule.Quotient.mk F : chains model7 (0 + 1)) = 0 := by
  by_contra hne
  have hsupp : ∀ a ∈ F.support, a.2.1 ∈ Yidx := fun a ha => hF ha
  have hS : (F.support.filter fun a =>
      coef model7 (0 + 1) (okey model7 (0 + 1) a) (Submodule.Quotient.mk F) ≠ 0).Nonempty := by
    by_contra h
    rw [Finset.not_nonempty_iff_eq_empty, Finset.filter_eq_empty_iff] at h
    exact hne (eq_zero_of_coef model7 (0 + 1) F fun a ha => by simpa using h ha)
  let hm : SL(2, ℤ) × model7.V (0 + 1) → ℤ := fun a =>
    max (Hn a.1 (tauP2.face 1 a.2.1 0)) (Hn a.1 (tauP2.face 1 a.2.1 1))
  obtain ⟨a0, ha0S, hmax⟩ := Finset.exists_max_image _ hm hS
  obtain ⟨ha0, hc0⟩ := Finset.mem_filter.1 ha0S
  -- the top endpoint `w = (γ₀, v_{x₀})` of `a₀`
  obtain ⟨x0, y0, hp0, hy0, hx0⟩ : ∃ x0 y0, (a0.2.1, x0, y0) ∈ inc ∧ Hn a0.1 y0 ≤ Hn a0.1 x0 ∧
      hm a0 = Hn a0.1 x0 := by
    obtain ⟨i1, i2⟩ := Yidx_inc _ (hsupp a0 ha0)
    by_cases h : Hn a0.1 (tauP2.face 1 a0.2.1 1) ≤ Hn a0.1 (tauP2.face 1 a0.2.1 0)
    · exact ⟨_, _, i1, h, max_eq_left h⟩
    · push Not at h; exact ⟨_, _, i2, h.le, max_eq_right h.le⟩
  have hx0m := (inc_mem _ hp0).1
  let w : SL(2, ℤ) × model7.V 0 := (a0.1, ⟨x0, hx0m⟩)
  let ψ : chains model7 (0 + 1) →+ ℤ :=
    (coef model7 0 (okey model7 0 w)).toAddMonoidHom.comp (bd model7 0).toAddMonoidHom
  have hv : ∀ (δ : SL(2, ℤ)) (n : ℕ) (hn : n ∈ tauP2.idxF 0),
      coef model7 0 (okey model7 0 w) (genQ model7 0 (δ, ⟨n, hn⟩)) =
        if δ⁻¹ * a0.1 ∈ tauP2.Sset 0 n x0 then 1 else 0 := by
    intro δ n hn
    rw [coef_genQ, osg_zero]
    by_cases h : δ⁻¹ * a0.1 ∈ tauP2.Sset 0 n x0
    · rw [if_pos ((okey_eq_iff model7 0 (δ, ⟨n, hn⟩) w).2 h), if_pos h]
    · rw [if_neg (fun e => h ((okey_eq_iff model7 0 (δ, ⟨n, hn⟩) w).1 e)), if_neg h]
  have hψ : ∀ a ∈ F.support, ψ (genQ model7 (0 + 1) a) =
      (if a.1⁻¹ * a0.1 ∈ tauP2.Sset 0 (tauP2.face 1 a.2.1 0) x0 then 1 else 0) -
        (if a.1⁻¹ * a0.1 ∈ tauP2.Sset 0 (tauP2.face 1 a.2.1 1) x0 then 1 else 0) := by
    intro a ha
    obtain ⟨m0, m1⟩ := Yidx_face _ (hsupp a ha)
    show coef model7 0 (okey model7 0 w) (bd model7 0 (genQ model7 (0 + 1) (a.1, a.2))) = _
    rw [bd_edge a.1 a.2 m0 m1, map_sub, hv, hv]
  have hiso := isolate model7 (0 + 1) F ψ a0 ha0 (by
    intro a ha hca hψa
    rw [hψ a ha] at hψa
    have hRel : a.1⁻¹ * a0.1 ∈ tauP2.Sset 0 (tauP2.face 1 a.2.1 0) x0 ∨
        a.1⁻¹ * a0.1 ∈ tauP2.Sset 0 (tauP2.face 1 a.2.1 1) x0 := by
      by_contra hcon
      push Not at hcon
      rw [if_neg hcon.1, if_neg hcon.2] at hψa
      exact hψa rfl
    have hma : hm a ≤ hm a0 := hmax a (Finset.mem_filter.2 ⟨ha, hca⟩)
    obtain ⟨i1, i2⟩ := Yidx_inc _ (hsupp a ha)
    rw [okey_eq_iff]
    show a.1⁻¹ * a0.1 ∈ tauP2.Sset (0 + 1) a.2.1 a0.2.1
    rcases hRel with h | h
    · exact tp _ _ hp0 i1 h hy0 (by rw [← hx0]; exact le_trans (le_max_right _ _) hma)
    · exact tp _ _ hp0 i2 h hy0 (by rw [← hx0]; exact le_trans (le_max_left _ _) hma))
  have hψx : ψ (Submodule.Quotient.mk F) = 0 := by
    show coef model7 0 (okey model7 0 w) (bd model7 0 (Submodule.Quotient.mk F)) = 0
    rw [hbd, map_zero]
  have hψ0 : ψ (genQ model7 (0 + 1) a0) ≠ 0 := by
    rw [hψ a0 ha0]
    have h1 : (1 : SL(2, ℤ)) ∈ tauP2.Sset 0 x0 x0 := (model7.M 0).one_mem ⟨x0, hx0m⟩
    have h2 : (1 : SL(2, ℤ)) ∉ tauP2.Sset 0 y0 x0 := by
      rw [Sset_eq_empty (inc_off _ hp0)]; exact Finset.notMem_empty _
    rw [inv_mul_cancel]
    rcases inc_face _ hp0 with ⟨e0, e1⟩ | ⟨e0, e1⟩ <;> dsimp only at e0 e1 <;> rw [e0, e1, if_pos h1, if_neg h2] <;>
      decide
  rw [hiso] at hψx
  exact mul_ne_zero (mul_ne_zero hc0 (osg_ne_zero _ _)) hψ0 hψx

/-- **`contract1` on the model**: `H₁(X_bulk) = 0`. -/
theorem _root_.GL2.Model.model7_contract1 :
    ∀ x : chains model7 (0 + 1), bd model7 0 x = 0 → ∃ y, bd model7 (0 + 1) y = x := by
  intro x hx
  obtain ⟨y, hy, b, ⟨z, rfl⟩, rfl⟩ := Submodule.mem_sup.1 (stepA x)
  obtain ⟨F, hF, rfl⟩ := hy
  have hbF : bd model7 0 (Submodule.Quotient.mk F) = 0 := by
    rw [map_add, bd_bd, add_zero] at hx; exact hx
  refine ⟨z, ?_⟩
  rw [Submodule.mkQ_apply, tree F hF hbF]
  exact (zero_add _).symm

/-! ### `contract2`: no `2`-cycles

Each of `f₀, f₁, f₃` has a horocycle edge (`e₀, e₂, e₅`) lying in no other triangle class, so its coefficient in a
`2`-cycle vanishes; then `e₃` and `e₆` lie only in `{f₀, f₂}` and `{f₃, f₄}`, which kills `f₂` and `f₄`. -/

lemma tri_cases : ∀ f ∈ tauP2.idxF (0 + 1 + 1), f = 0 ∨ f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4 := by decide +kernel

lemma tri_face_mem : ∀ f ∈ tauP2.idxF (0 + 1 + 1), tauP2.face 2 f 0 ∈ tauP2.idxF (0 + 1) ∧
    tauP2.face 2 f 1 ∈ tauP2.idxF (0 + 1) ∧ tauP2.face 2 f 2 ∈ tauP2.idxF (0 + 1) := by decide +kernel

/-- **A triangle class with a private edge `e` has coefficient `0` in a `2`-cycle.** -/
lemma kill (F : SL(2, ℤ) × model7.V (0 + 1 + 1) →₀ ℤ)
    (hbd : bd model7 (0 + 1) (Submodule.Quotient.mk F) = 0) (a0 : SL(2, ℤ) × model7.V (0 + 1 + 1))
    (ha0 : a0 ∈ F.support) (e : ℕ) (he : e ∈ tauP2.idxF (0 + 1))
    (hoff : ∀ m ∈ tauP2.idxF (0 + 1), m ≠ e → tauP2.trSetZ (0 + 1) m e = [])
    (hsub : ∀ f ∈ tauP2.idxF (0 + 1 + 1), ∀ g ∈ (tauP2.trSetZ (0 + 1) e e).map Tr.g,
      g ∈ (tauP2.trSetZ (0 + 1 + 1) f f).map Tr.g)
    (honce : (tauP2.face 2 a0.2.1 0 = e ∧ tauP2.face 2 a0.2.1 1 ≠ e ∧ tauP2.face 2 a0.2.1 2 ≠ e) ∨
      (tauP2.face 2 a0.2.1 0 ≠ e ∧ tauP2.face 2 a0.2.1 1 = e ∧ tauP2.face 2 a0.2.1 2 ≠ e) ∨
      (tauP2.face 2 a0.2.1 0 ≠ e ∧ tauP2.face 2 a0.2.1 1 ≠ e ∧ tauP2.face 2 a0.2.1 2 = e))
    (hcof : ∀ a ∈ F.support, coef model7 (0 + 1 + 1) (okey model7 (0 + 1 + 1) a) (Submodule.Quotient.mk F) ≠ 0 →
      (tauP2.face 2 a.2.1 0 = e ∨ tauP2.face 2 a.2.1 1 = e ∨ tauP2.face 2 a.2.1 2 = e) → a.2.1 = a0.2.1) :
    coef model7 (0 + 1 + 1) (okey model7 (0 + 1 + 1) a0) (Submodule.Quotient.mk F) = 0 := by
  let k := okey model7 (0 + 1) (a0.1, ⟨e, he⟩)
  let ψ : chains model7 (0 + 1 + 1) →+ ℤ :=
    (coef model7 (0 + 1) k).toAddMonoidHom.comp (bd model7 (0 + 1)).toAddMonoidHom
  have hc : ∀ (δ : SL(2, ℤ)) (m : ℕ) (hm : m ∈ tauP2.idxF (0 + 1)),
      coef model7 (0 + 1) k (genQ model7 (0 + 1) (δ, ⟨m, hm⟩)) ≠ 0 →
        m = e ∧ δ⁻¹ * a0.1 ∈ tauP2.Sset (0 + 1) e e := by
    intro δ m hm h
    rw [coef_genQ] at h
    by_cases hk : okey model7 (0 + 1) (δ, ⟨m, hm⟩) = k
    · have hr : δ⁻¹ * a0.1 ∈ tauP2.Sset (0 + 1) m e := (okey_eq_iff model7 (0 + 1) (δ, ⟨m, hm⟩) (a0.1, ⟨e, he⟩)).1 hk
      by_cases hme : m = e
      · subst hme; exact ⟨rfl, hr⟩
      · rw [Sset_eq_empty (hoff m hm hme)] at hr; exact absurd hr (Finset.notMem_empty _)
    · rw [if_neg hk] at h; exact absurd rfl h
  have hψ : ∀ (a : SL(2, ℤ) × model7.V (0 + 1 + 1)), ψ (genQ model7 (0 + 1 + 1) a) =
      coef model7 (0 + 1) k (genQ model7 (0 + 1) (a.1, ⟨_, (tri_face_mem _ a.2.2).1⟩)) -
      coef model7 (0 + 1) k (genQ model7 (0 + 1) (a.1, ⟨_, (tri_face_mem _ a.2.2).2.1⟩)) +
      coef model7 (0 + 1) k (genQ model7 (0 + 1) (a.1, ⟨_, (tri_face_mem _ a.2.2).2.2⟩)) := by
    intro a
    show coef model7 (0 + 1) k (bd model7 (0 + 1) (genQ model7 (0 + 1 + 1) (a.1, a.2))) = _
    rw [bd_tri, map_add, map_sub]
  have hiso := isolate model7 (0 + 1 + 1) F ψ a0 ha0 (by
    intro a ha hca hψa
    rw [hψ a] at hψa
    have hj : ∃ m, ∃ hm : m ∈ tauP2.idxF (0 + 1),
        (m = tauP2.face 2 a.2.1 0 ∨ m = tauP2.face 2 a.2.1 1 ∨ m = tauP2.face 2 a.2.1 2) ∧
        coef model7 (0 + 1) k (genQ model7 (0 + 1) (a.1, ⟨m, hm⟩)) ≠ 0 := by
      by_contra hcon
      push Not at hcon
      rw [hcon _ _ (Or.inl rfl), hcon _ _ (Or.inr (Or.inl rfl)), hcon _ _ (Or.inr (Or.inr rfl))] at hψa
      exact hψa (by ring)
    obtain ⟨m, hm, hmf, hne⟩ := hj
    obtain ⟨rfl, hS⟩ := hc _ _ hm hne
    have hf : a.2.1 = a0.2.1 := hcof a ha hca (by
      rcases hmf with h | h | h <;> [left; (right; left); (right; right)] <;> exact h.symm)
    rw [okey_eq_iff]
    show a.1⁻¹ * a0.1 ∈ tauP2.Sset (0 + 1 + 1) a.2.1 a0.2.1
    rw [hf]
    exact Sset_sub (hsub _ a0.2.2) hS)
  have hψx : ψ (Submodule.Quotient.mk F) = 0 := by
    show coef model7 (0 + 1) k (bd model7 (0 + 1) (Submodule.Quotient.mk F)) = 0
    rw [hbd, map_zero]
  have hself : ∀ (hm : e ∈ tauP2.idxF (0 + 1)), coef model7 (0 + 1) k (genQ model7 (0 + 1) (a0.1, ⟨e, hm⟩)) =
      osg model7 (0 + 1) (a0.1, ⟨e, he⟩) := fun hm => by rw [coef_genQ, if_pos rfl]
  have hzero : ∀ m (hm : m ∈ tauP2.idxF (0 + 1)), m ≠ e →
      coef model7 (0 + 1) k (genQ model7 (0 + 1) (a0.1, ⟨m, hm⟩)) = 0 := by
    intro m hm hme
    by_contra h
    exact hme (hc _ _ hm h).1
  have hψ0 : ψ (genQ model7 (0 + 1 + 1) a0) ≠ 0 := by
    rw [hψ a0]
    have ho := osg_ne_zero (0 + 1) (a0.1, ⟨e, he⟩)
    rcases honce with ⟨h0, h1, h2⟩ | ⟨h0, h1, h2⟩ | ⟨h0, h1, h2⟩
    · rw [hzero _ _ h1, hzero _ _ h2]
      simp only [h0, hself, sub_zero, add_zero]; exact ho
    · rw [hzero _ _ h0, hzero _ _ h2]
      simp only [h1, hself, zero_sub, add_zero, neg_ne_zero]; exact ho
    · rw [hzero _ _ h0, hzero _ _ h1]
      simp only [h2, hself, sub_zero, zero_add]; exact ho
  rw [hiso] at hψx
  rcases mul_eq_zero.1 hψx with h | h
  · rcases mul_eq_zero.1 h with h | h
    · exact h
    · exact absurd h (osg_ne_zero _ _)
  · exact absurd h hψ0

/-- **`contract2` on the model**: `Z₂(X_bulk) = 0`. -/
theorem _root_.GL2.Model.model7_contract2 : ∀ x : chains model7 (0 + 1 + 1), bd model7 (0 + 1) x = 0 → x = 0 := by
  intro x hx
  induction x using Submodule.Quotient.induction_on with
  | _ F =>
    have p1 : ∀ a ∈ F.support, (a.2.1 = 0 ∨ a.2.1 = 1 ∨ a.2.1 = 3) →
        coef model7 (0 + 1 + 1) (okey model7 (0 + 1 + 1) a) (Submodule.Quotient.mk F) = 0 := by
      intro a ha h
      have cof : ∀ f ∈ tauP2.idxF (0 + 1 + 1), ∀ e ∈ [0, 2, 5],
          (tauP2.face 2 f 0 = e ∨ tauP2.face 2 f 1 = e ∨ tauP2.face 2 f 2 = e) →
            f = if e = 0 then 0 else if e = 2 then 1 else 3 := by decide +kernel
      rcases h with h | h | h
      · refine kill F hx a ha 0 (by decide +kernel) (by decide +kernel) (by decide +kernel)
          (by rw [h]; decide +kernel) fun b hb _ hf => ?_
        rw [cof _ b.2.2 0 (by decide) hf, h]; rfl
      · refine kill F hx a ha 2 (by decide +kernel) (by decide +kernel) (by decide +kernel)
          (by rw [h]; decide +kernel) fun b hb _ hf => ?_
        rw [cof _ b.2.2 2 (by decide) hf, h]; rfl
      · refine kill F hx a ha 5 (by decide +kernel) (by decide +kernel) (by decide +kernel)
          (by rw [h]; decide +kernel) fun b hb _ hf => ?_
        rw [cof _ b.2.2 5 (by decide) hf, h]; rfl
    have p2 : ∀ a ∈ F.support, (a.2.1 = 2 ∨ a.2.1 = 4) →
        coef model7 (0 + 1 + 1) (okey model7 (0 + 1 + 1) a) (Submodule.Quotient.mk F) = 0 := by
      intro a ha h
      have cof3 : ∀ f ∈ tauP2.idxF (0 + 1 + 1),
          (tauP2.face 2 f 0 = 3 ∨ tauP2.face 2 f 1 = 3 ∨ tauP2.face 2 f 2 = 3) → f = 0 ∨ f = 2 := by decide +kernel
      have cof6 : ∀ f ∈ tauP2.idxF (0 + 1 + 1),
          (tauP2.face 2 f 0 = 6 ∨ tauP2.face 2 f 1 = 6 ∨ tauP2.face 2 f 2 = 6) → f = 3 ∨ f = 4 := by decide +kernel
      rcases h with h | h
      · refine kill F hx a ha 3 (by decide +kernel) (by decide +kernel) (by decide +kernel)
          (by rw [h]; decide +kernel) fun b hb hcb hf => ?_
        rcases cof3 _ b.2.2 hf with h' | h'
        · exact absurd (p1 b hb (Or.inl h')) hcb
        · rw [h', h]
      · refine kill F hx a ha 6 (by decide +kernel) (by decide +kernel) (by decide +kernel)
          (by rw [h]; decide +kernel) fun b hb hcb hf => ?_
        rcases cof6 _ b.2.2 hf with h' | h'
        · exact absurd (p1 b hb (Or.inr (Or.inr h'))) hcb
        · rw [h', h]
    refine eq_zero_of_coef model7 (0 + 1 + 1) F fun a ha => ?_
    rcases tri_cases _ a.2.2 with h | h | h | h | h
    · exact p1 a ha (Or.inl h)
    · exact p1 a ha (Or.inr (Or.inl h))
    · exact p2 a ha (Or.inl h)
    · exact p1 a ha (Or.inr (Or.inr h))
    · exact p2 a ha (Or.inr h)

end M7

/-- **H4(ii) for the model** ([C] `prop:model7h4`(ii), [GT] `prop:model7h4-ii`): `X_bulk` of the seven-vertex model is
acyclic -- connected (`model7_contract0`), `H₁ = 0` (`model7_contract1`, the Serre tree with its topograph height) and
`Z₂ = 0` (`model7_contract2`).  Proved combinatorially; no `native_decide`, no topology. -/
theorem model7_acyclic : XBulk.Acyclic model7 := ⟨model7_contract0, model7_contract1, model7_contract2⟩


end GL2.Model
