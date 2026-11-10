import GL2.Model.H4
import GL2.Checkers.TauP2Key

/-!
# M2-c / M2-d (L18-1): an accepted τ-dump is an `X_bulk` descriptor; the 7-vertex model `model7`

[GT] `def:model7-lean`, `prop:model7h4-iv`, `prop:model7h4-ii`, mirroring [C] v1.17.0 `def:model7`, `def:model7rel`,
`lem:model7coincidences`(1), `prop:model7h4`.

* `TauData.XAx` -- the `X_bulk` axioms on the stored data (dimensions `≤ 2`, `KeyAx` in degrees `0, 1, 2`, faces
  closed, transporters permute faces through their stored vertex correspondence with the right signs, and the double
  faces cancel under the standard pairing `(j, i) ↦ (i, j - 1)` / `(i + 1, j)`); decidable (`xbulkB`).
* `TauData.toXBulk` -- an accepted dump is an `XBulkDatum` (faces `∂_j ι(τ) = (-1)^j ι(∂_j τ)` with identity
  transporters, as `CreateSL2ZFundamentalDomainTriangulation` stores them).
* `Model.model7` -- the instance on the real `p = 2` dump (`tauP2_xbulkB`, `decide +kernel`); its simplex list is the
  hand-written list of [C] `def:model7` (`tauP2_simplices`).
* `model7_stab` -- **H4(iv)** on the model: the stabiliser of `ι(σ)` is the stored `S(σ, σ)`; it contains `-I` for every
  simplex; it has order `6, 6, 4` at `v₄, v₅, v₆` (containing `ST`, `TS`, `S`) and order `2` (`= {±I}`) elsewhere.
* `model7_acyclic` -- **H4(ii) for the model** is proved in `GL2.Model.Acyclic` (L19-3, M3-b): `contract0` by the
  generators `S, T`, `contract1` by the Serre tree with its topograph height, `contract2` by local coface tables.
-/

namespace GL2.TauChk

open scoped MatrixGroups

/-! ### Generic finite-index helpers -/

/-- The face-slot count of a `(q+1)`-simplex: `q + 2` for `q ≥ 0`, none below. -/
def nf (q : ℤ) : ℕ := if 0 ≤ q then q.toNat + 2 else 0

lemma nf_pos {q : ℤ} (j : Fin (nf q)) : 0 ≤ q ∧ j.1 < q.toNat + 2 := by
  have := j.2; unfold nf at this; split_ifs at this with h
  · exact ⟨h, this⟩
  · exact absurd this (Nat.not_lt_zero _)

/-- The standard pairing of double faces: `(j, i) ↦ (i, j - 1)` if `i < j`, else `(i + 1, j)`. -/
def iotaN (j i : ℕ) : ℕ × ℕ := if i < j then (i, j - 1) else (i + 1, j)

/-- The pairing on `Fin a × Fin b`, `a = b + 1` (identity otherwise). -/
def iotaF (a b : ℕ) (p : Fin a × Fin b) : Fin a × Fin b :=
  if h : a = b + 1 then
    if hl : p.2.1 < p.1.1 then (⟨p.2.1, by have := p.2.2; omega⟩, ⟨p.1.1 - 1, by have := p.1.2; omega⟩)
    else (⟨p.2.1 + 1, by have := p.2.2; omega⟩, ⟨p.1.1, by have := p.2.2; omega⟩)
  else p

lemma iotaF_val (a b : ℕ) (h : a = b + 1) (p : Fin a × Fin b) :
    ((iotaF a b p).1.1, (iotaF a b p).2.1) = iotaN p.1.1 p.2.1 := by
  unfold iotaF iotaN; rw [dif_pos h]; split_ifs <;> rfl

lemma iotaN_iotaN (j i : ℕ) : iotaN (iotaN j i).1 (iotaN j i).2 = (j, i) := by
  unfold iotaN; split_ifs <;> simp_all <;> omega

lemma iotaF_involutive (a b : ℕ) : Function.Involutive (iotaF a b) := by
  intro p
  by_cases h : a = b + 1
  · have e1 := iotaF_val a b h (iotaF a b p)
    have e2 := iotaF_val a b h p
    have f1 : (iotaF a b p).1.1 = (iotaN p.1.1 p.2.1).1 := congrArg Prod.fst e2
    have f2 : (iotaF a b p).2.1 = (iotaN p.1.1 p.2.1).2 := congrArg Prod.snd e2
    rw [f1, f2, iotaN_iotaN] at e1
    exact Prod.ext (Fin.ext (congrArg Prod.fst e1)) (Fin.ext (congrArg Prod.snd e1))
  · unfold iotaF; simp only [dif_neg h]

/-- A vertex-slot list as a permutation of `Fin n` (`1` if it is not one). -/
noncomputable def listPerm (n : ℕ) (l : List ℕ) : Equiv.Perm (Fin n) :=
  if h : ∀ i : Fin n, l.getD i 0 < n then
    if hi : Function.Injective (fun i : Fin n => (⟨l.getD i 0, h i⟩ : Fin n)) then
      Equiv.ofBijective _ (Finite.injective_iff_bijective.mp hi)
    else 1
  else 1

lemma listPerm_apply (n : ℕ) (l : List ℕ) (h1 : ∀ j ∈ List.range n, l.getD j 0 < n)
    (h2 : ((List.range n).map (fun j => l.getD j 0)).Nodup) (i : Fin n) : (listPerm n l i : ℕ) = l.getD i 0 := by
  have h : ∀ i : Fin n, l.getD i 0 < n := fun i => h1 i (List.mem_range.2 i.2)
  have hi : Function.Injective (fun i : Fin n => (⟨l.getD i 0, h i⟩ : Fin n)) := by
    intro x y e
    simp only [Fin.mk.injEq] at e
    exact Fin.ext (List.inj_on_of_nodup_map h2 (List.mem_range.2 x.2) (List.mem_range.2 y.2) e)
  unfold listPerm; rw [dif_pos h, dif_pos hi]; rfl

lemma sgnPow_cases (j : ℕ) : sgnPow j = 1 ∨ sgnPow j = -1 := by unfold sgnPow; split <;> simp

lemma unitOf_neg {z : ℤ} (hz : z = 1 ∨ z = -1) : unitOf (-z) = -unitOf z := by
  rcases hz with rfl | rfl <;> decide

namespace TauData

variable (D : TauData)

/-- **The `X_bulk` axioms on the stored data** (decidable). -/
def XAx : Prop :=
  (∀ x ∈ D.simplices, x.1 ≤ 2) ∧ (∀ t ∈ D.trs, t.q ≤ 2) ∧ D.KeyAx 0 ∧ D.KeyAx 1 ∧ D.KeyAx 2 ∧
  (∀ x ∈ D.simplices, 1 ≤ x.1 → ∀ j ∈ List.range (x.1 + 1), D.face x.1 x.2.1 j ∈ D.idxL ((x.1 : ℤ) - 1)) ∧
  (∀ t ∈ D.trs, 1 ≤ t.q → (∀ j ∈ List.range (t.q + 1), t.perm.getD j 0 < t.q + 1) ∧
    ((List.range (t.q + 1)).map (fun j => t.perm.getD j 0)).Nodup ∧
    ∀ j ∈ List.range (t.q + 1), ∃ u ∈ D.trSetZ ((t.q : ℤ) - 1) (D.face t.q t.s1 j) (D.face t.q t.s2 (t.perm.getD j 0)),
      u.g = t.g ∧ sgnPow j = permSign u.perm (t.q - 1) * permSign t.perm t.q * sgnPow (t.perm.getD j 0)) ∧
  (∀ x ∈ D.simplices, 2 ≤ x.1 → ∀ j ∈ List.range (x.1 + 1), ∀ i ∈ List.range x.1,
    D.face (x.1 - 1) (D.face x.1 x.2.1 j) i = D.face (x.1 - 1) (D.face x.1 x.2.1 (iotaN j i).1) (iotaN j i).2 ∧
    sgnPow j * sgnPow i = -(sgnPow (iotaN j i).1 * sgnPow (iotaN j i).2))

instance : Decidable D.XAx := by unfold XAx; infer_instance

/-- Decides `XAx`. -/
def xbulkB : Bool := decide D.XAx

variable {D}

lemma mem_idxL {q : ℤ} {σ : ℕ} : σ ∈ D.idxL q ↔ ∃ x ∈ D.simplices, (x.1 : ℤ) = q ∧ x.2.1 = σ := by
  simp only [idxL, List.mem_map, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨x, ⟨hx, hq⟩, rfl⟩; exact ⟨x, hx, hq, rfl⟩
  · rintro ⟨x, hx, hq, rfl⟩; exact ⟨x, ⟨hx, hq⟩, rfl⟩

lemma keyAx_of_empty (q : ℤ) (h1 : ∀ x ∈ D.simplices, (x.1 : ℤ) ≠ q) (h2 : ∀ t ∈ D.trs, (t.q : ℤ) ≠ q) :
    D.KeyAx q := by
  have e1 : D.idxL q = [] := by
    apply List.eq_nil_iff_forall_not_mem.2
    intro σ hσ; obtain ⟨x, hx, hq, -⟩ := mem_idxL.1 hσ; exact h1 x hx hq
  have e2 : D.trZ q = [] := by
    apply List.eq_nil_iff_forall_not_mem.2
    intro t ht; rw [mem_trZ] at ht; exact h2 t ht.1 ht.2
  unfold KeyAx; simp [e1, e2]

lemma keyAx_all (h : D.XAx) (q : ℤ) : D.KeyAx q := by
  by_cases h0 : q = 0
  · subst h0; exact h.2.2.1
  by_cases h1 : q = 1
  · subst h1; exact h.2.2.2.1
  by_cases h2 : q = 2
  · subst h2; exact h.2.2.2.2.1
  apply keyAx_of_empty
  · intro x hx e; have := h.1 x hx; omega
  · intro t ht e; have := h.2.1 t ht; omega

lemma face_mem (h : D.XAx) {q : ℤ} (τ : D.Vz (q + 1)) (j : Fin (nf q)) :
    D.face (q + 1).toNat τ.1 j.1 ∈ D.idxF q := by
  obtain ⟨hq, hj⟩ := nf_pos j
  obtain ⟨x, hx, hxq, hxσ⟩ := mem_idxL.1 ((D.mem_idxF).1 τ.2)
  have hx1 : x.1 = (q + 1).toNat := by omega
  have := h.2.2.2.2.2.1 x hx (by omega) j.1 (List.mem_range.2 (by omega))
  rw [hx1, hxσ, show (((q + 1).toNat : ℕ) : ℤ) - 1 = q by omega] at this
  exact (D.mem_idxF).2 this

/-- The signed face `∂_j ι(τ) = (-1)^j ι(∂_j τ)` of the dump. -/
def faceZ (h : D.XAx) (q : ℤ) (τ : D.Vz (q + 1)) (j : Fin (nf q)) : GL2.Model.OrPt (D.Vz q) :=
  (1, ⟨D.face (q + 1).toNat τ.1 j.1, face_mem h τ j⟩, unitOf (sgnPow j.1))

lemma faceZ_transport (h : D.XAx) (q : ℤ) {τ τ' : D.Vz (q + 1)} {t : SL(2, ℤ)}
    (ht : t ∈ (D.toKeyModel (q + 1) (keyAx_all h (q + 1))).S τ τ') :
    ∃ π : Fin (nf q) ≃ Fin (nf q), ∀ j, GL2.Model.OrRel (D.toKeyModel q (keyAx_all h q)) (faceZ h q τ j)
      (GL2.Model.OrPt.tr t (faceZ h q τ' (π j)) ((D.toKeyModel (q + 1) (keyAx_all h (q + 1))).sgn τ τ' t)) := by
  by_cases hq : 0 ≤ q
  swap
  · refine ⟨Equiv.refl _, fun j => absurd (nf_pos j).1 hq⟩
  obtain ⟨t0, ht0, rfl⟩ := (D.mem_Sset).1 ht
  have ht0' := ht0
  rw [mem_trSetZ] at ht0'
  obtain ⟨ht0m, ht0q, ht0s1, ht0s2⟩ := ht0'
  have htq : t0.q = (q + 1).toNat := by omega
  obtain ⟨hp1, hp2, hc⟩ := h.2.2.2.2.2.2.1 t0 ht0m (by omega)
  have hn : nf q = t0.q + 1 := by unfold nf; rw [if_pos hq]; omega
  rw [← hn] at hp1 hp2
  refine ⟨listPerm (nf q) t0.perm, fun j => ?_⟩
  have hk : (listPerm (nf q) t0.perm j : ℕ) = t0.perm.getD j.1 0 := listPerm_apply (nf q) t0.perm hp1 hp2 j
  obtain ⟨u, hu, hug, hus⟩ := hc j.1 (List.mem_range.2 (by have := (nf_pos j).2; omega))
  rw [htq, ht0s1, ht0s2, show (((q + 1).toNat : ℕ) : ℤ) - 1 = q by omega, ← hk] at hu
  rw [htq, ← hk] at hus
  refine ⟨?_, ?_⟩
  · show (1 : SL(2, ℤ))⁻¹ * (M22.toSL t0.g * 1) ∈ D.Sset q _ _
    rw [inv_one, one_mul, mul_one]
    exact (D.mem_Sset).2 ⟨u, hu, by rw [hug]⟩
  · show unitOf (sgnPow j.1) = D.sgnZ q (D.face (q + 1).toNat τ.1 j.1)
        (D.face (q + 1).toNat τ'.1 (listPerm (nf q) t0.perm j : ℕ)) ((1 : SL(2, ℤ))⁻¹ * (M22.toSL t0.g * 1)) *
      (D.sgnZ (q + 1) τ.1 τ'.1 (M22.toSL t0.g) * unitOf (sgnPow (listPerm (nf q) t0.perm j : ℕ)))
    rw [inv_one, one_mul, mul_one, ← hug, sgnZ_eq (keyAx_all h q) hu, hug, sgnZ_eq (keyAx_all h (q + 1)) ht0, hus,
      show (q + 1).toNat - 1 = q.toNat by omega, ← mul_assoc,
      unitOf_mul (by rcases permSign_cases u.perm q.toNat with e | e <;> rcases permSign_cases t0.perm (q + 1).toNat
        with e' | e' <;> simp [e, e']) (sgnPow_cases _),
      unitOf_mul (permSign_cases _ _) (permSign_cases _ _)]

lemma faceZ_simplicial (h : D.XAx) (q : ℤ) (τ : D.Vz (q + 1 + 1)) :
    ∃ ι : Fin (nf (q + 1)) × Fin (nf q) → Fin (nf (q + 1)) × Fin (nf q), Function.Involutive ι ∧
      ∀ p, GL2.Model.OrRel (D.toKeyModel q (keyAx_all h q))
        (GL2.Model.OrPt.comp2 (faceZ h (q + 1) τ p.1) (faceZ h q (faceZ h (q + 1) τ p.1).2.1 p.2))
        (GL2.Model.OrPt.neg (GL2.Model.OrPt.comp2 (faceZ h (q + 1) τ (ι p).1)
          (faceZ h q (faceZ h (q + 1) τ (ι p).1).2.1 (ι p).2))) := by
  refine ⟨iotaF (nf (q + 1)) (nf q), iotaF_involutive _ _, fun p => ?_⟩
  obtain ⟨hq, hi⟩ := nf_pos p.2
  have hab : nf (q + 1) = nf q + 1 := by unfold nf; rw [if_pos hq, if_pos (by omega)]; omega
  have hv := iotaF_val _ _ hab p
  obtain ⟨x, hx, hxq, hxσ⟩ := mem_idxL.1 ((D.mem_idxF).1 τ.2)
  have hx1 : x.1 = (q + 1 + 1).toNat := by omega
  have hx2 : x.1 - 1 = (q + 1).toNat := by omega
  have hj := (nf_pos p.1).2
  obtain ⟨hc1, hc2⟩ := h.2.2.2.2.2.2.2 x hx (by omega) p.1.1 (List.mem_range.2 (by omega)) p.2.1
    (List.mem_range.2 (by omega))
  rw [hx2, hx1, hxσ, ← hv] at hc1
  rw [← hv] at hc2
  set M := D.toKeyModel q (keyAx_all h q)
  set c := (GL2.Model.OrPt.comp2 (faceZ h (q + 1) τ p.1) (faceZ h q (faceZ h (q + 1) τ p.1).2.1 p.2))
  set c' := (GL2.Model.OrPt.comp2 (faceZ h (q + 1) τ (iotaF _ _ p).1)
    (faceZ h q (faceZ h (q + 1) τ (iotaF _ _ p).1).2.1 (iotaF _ _ p).2))
  have hcc : c.2.1 = c'.2.1 := Subtype.ext hc1
  have h1 : c.1 = 1 := by simp [c, GL2.Model.OrPt.comp2, faceZ]
  have h1' : c'.1 = 1 := by simp [c', GL2.Model.OrPt.comp2, faceZ]
  have he : c.2.2 = -c'.2.2 := by
    simp only [c, c', GL2.Model.OrPt.comp2, faceZ]
    rw [← unitOf_mul (sgnPow_cases _) (sgnPow_cases _), ← unitOf_mul (sgnPow_cases _) (sgnPow_cases _),
      ← unitOf_neg (by rcases sgnPow_cases (iotaF _ _ p).1.1 with e | e <;>
        rcases sgnPow_cases (iotaF _ _ p).2.1 with e' | e' <;> simp [e, e']), hc2]
  refine ⟨?_, ?_⟩
  · show c.1⁻¹ * c'.1 ∈ M.S c.2.1 c'.2.1
    rw [h1, h1', ← hcc, inv_one, one_mul]; exact M.one_mem _
  · show c.2.2 = M.sgn c.2.1 c'.2.1 (c.1⁻¹ * c'.1) * -c'.2.2
    rw [h1, h1', ← hcc, inv_one, one_mul, M.sgn_one, one_mul, he]

lemma sgn0Z (h : D.XAx) {σ σ' : D.Vz 0} {t : SL(2, ℤ)} (ht : t ∈ (D.toKeyModel 0 (keyAx_all h 0)).S σ σ') :
    (D.toKeyModel 0 (keyAx_all h 0)).sgn σ σ' t = 1 := by
  obtain ⟨t0, ht0, rfl⟩ := (D.mem_Sset).1 ht
  show D.sgnZ 0 σ.1 σ'.1 _ = 1
  rw [sgnZ_eq (keyAx_all h 0) ht0]
  simp [permSign_zero, unitOf_one]

lemma dim2Z (h : D.XAx) (q : ℤ) (hq : q < 0 ∨ 2 < q) : IsEmpty (D.Vz q) := by
  refine ⟨fun σ => ?_⟩
  obtain ⟨x, hx, hxq, -⟩ := mem_idxL.1 ((D.mem_idxF).1 σ.2)
  have := h.1 x hx
  omega

variable (D)

/-- **An accepted dump is an `X_bulk` descriptor.** -/
noncomputable def toXBulk (h : D.XAx) : GL2.Model.XBulkDatum where
  V q := D.Vz q
  M q := D.toKeyModel q (keyAx_all h q)
  F q := Fin (nf q)
  face q τ j := faceZ h q τ j
  face_transport q _ _ _ ht := faceZ_transport h q ht
  face_simplicial q τ := faceZ_simplicial h q τ
  sgn0 ht := sgn0Z h ht
  aug_face τ := by
    show ∑ j : Fin 2, ((unitOf (sgnPow j.1) : ℤˣ) : ℤ) = 0
    decide
  dim2 q hq := dim2Z h q hq

lemma card_Sset {D : TauData} {q : ℤ} (hK : D.KeyAx q) (a b : ℕ) :
    (D.Sset q a b).card = ((D.trSetZ q a b).map Tr.g).toFinset.card := by
  have e : D.Sset q a b = ((D.trSetZ q a b).map Tr.g).toFinset.image M22.toSL := by
    ext γ; simp [Sset]
  rw [e]
  apply Finset.card_image_of_injOn
  intro x hx y hy hxy
  simp only [Finset.mem_coe, List.mem_toFinset, List.mem_map] at hx hy
  obtain ⟨t, ht, rfl⟩ := hx
  obtain ⟨u, hu, rfl⟩ := hy
  exact M22.toSL_injOn (hK.2.2.2.1 t (trZ_of_trSetZ D ht)).1 (hK.2.2.2.1 u (trZ_of_trSetZ D hu)).1 hxy

end TauData

/-! ### The real `p = 2` dump -/

namespace Data

/-- [C] `def:model7` by hand: `v₀ … v₆`, the eleven edges and five triangles, increasing vertex order, lexicographic. -/
def model7Simplices : List (ℕ × ℕ × List ℕ) :=
  ((List.range 7).map fun v => (0, v, [v])) ++
  ([[0, 1], [0, 4], [1, 2], [1, 4], [1, 6], [2, 3], [2, 5], [2, 6], [3, 5], [4, 6], [5, 6]].zipIdx.map
    fun e => (1, e.2, e.1)) ++
  ([[0, 1, 4], [1, 2, 6], [1, 4, 6], [2, 3, 5], [2, 5, 6]].zipIdx.map fun f => (2, f.2, f.1))

/-- The dump's simplex list is [C] `def:model7`'s. -/
theorem tauP2_simplices : tauP2.simplices = model7Simplices := by decide +kernel

theorem tauP2_xbulkB : tauP2.xbulkB = true := by decide +kernel

/-- The stabiliser orders of [C] `H4`(iv): `6, 6, 4` at `v₄, v₅, v₆`, `2` elsewhere. -/
def stabOrd (q : ℤ) (σ : ℕ) : ℕ := if q = 0 ∧ (σ = 4 ∨ σ = 5) then 6 else if q = 0 ∧ σ = 6 then 4 else 2

theorem tauP2_negI_check :
    ∀ x ∈ tauP2.simplices, ∃ t ∈ tauP2.trSetZ x.1 x.2.1 x.2.1, t.g = ⟨-1, 0, 0, -1⟩ := by decide +kernel

theorem tauP2_ord_check :
    ∀ x ∈ tauP2.simplices, ((tauP2.trSetZ x.1 x.2.1 x.2.1).map Tr.g).toFinset.card = stabOrd x.1 x.2.1 := by
  decide +kernel

theorem tauP2_gen_check :
    (∃ t ∈ tauP2.trSetZ 0 4 4, t.g = ⟨0, -1, 1, 1⟩) ∧ (∃ t ∈ tauP2.trSetZ 0 5 5, t.g = ⟨1, -1, 1, 0⟩) ∧
      (∃ t ∈ tauP2.trSetZ 0 6 6, t.g = ⟨0, -1, 1, 0⟩) := by decide +kernel

/-- [C] `lem:model7coincidences`(2): the ordered off-diagonal pairs `(q, σ, σ')` with `S(σ, σ') ≠ ∅` -- the four
unordered pairs `{v₀, v₃}`, `{v₄, v₅}`, `{e₀₄ = e₁, e₃₅ = e₈}`, `{e₄₆ = e₉, e₅₆ = e₁₀}` (edge indices lexicographic). -/
def coincPairs : List (ℕ × ℕ × ℕ) :=
  [(0, 0, 3), (0, 3, 0), (0, 4, 5), (0, 5, 4), (1, 1, 8), (1, 8, 1), (1, 9, 10), (1, 10, 9)]

/-- **`lem:model7coincidences`(2) on the dump** (WP-17, F3-18 MINOR 3): for distinct simplices of one dimension the
stored transporter set is non-empty exactly at `coincPairs`; every other simplex is alone in its orbit. -/
theorem tauP2_pairs_check :
    ∀ x ∈ tauP2.simplices, ∀ y ∈ tauP2.simplices, x.1 = y.1 → x.2.1 ≠ y.2.1 →
      (tauP2.trSetZ x.1 x.2.1 y.2.1 ≠ [] ↔ (x.1, x.2.1, y.2.1) ∈ coincPairs) := by decide +kernel

/-- [C] `def:model7rel`'s eight listed rows `(q, σ_lhs, σ_rhs, γ)`, read `ι(σ_lhs) = γ ι(σ_rhs)`, i.e.
`γ ∈ S(σ_lhs, σ_rhs)`: `T_v3_v0`, `T_v5_v4`, `T_e(0,4)_e(3,5)`, `S_v4_v5`, `S_v6_stabilizer`, `S_e(5,6)_e(4,6)`,
`ST_v4_stabilizer`, `TS_v5_stabilizer`. -/
def model7RelRows : List (ℕ × ℕ × ℕ × M22) :=
  [(0, 3, 0, ⟨1, 1, 0, 1⟩), (0, 5, 4, ⟨1, 1, 0, 1⟩), (1, 8, 1, ⟨1, 1, 0, 1⟩), (0, 4, 5, ⟨0, -1, 1, 0⟩),
    (0, 6, 6, ⟨0, -1, 1, 0⟩), (1, 9, 10, ⟨0, -1, 1, 0⟩), (0, 4, 4, ⟨0, -1, 1, 1⟩), (0, 5, 5, ⟨1, -1, 1, 0⟩)]

/-- **`def:model7rel`'s eight rows on the dump** (WP-17, F3-18 MINOR 3): each `γ` is a stored transporter
`S(σ_lhs, σ_rhs)` carrying the stored orientation to the stored orientation (`permSign = 1`), as
`lem:model7coincidences`(2) says. -/
theorem tauP2_rel_check :
    ∀ r ∈ model7RelRows, ∃ t ∈ tauP2.trSetZ r.1 r.2.1 r.2.2.1, t.g = r.2.2.2 ∧ permSign t.perm r.1 = 1 := by
  decide +kernel

end Data

end GL2.TauChk

namespace GL2.Model

open scoped MatrixGroups
open GL2.TauChk GL2.TauChk.Data

/-- **`Model.model7`**: the 7-vertex model as an `X_bulk` descriptor, built from the real `p = 2` dump. -/
noncomputable def model7 : XBulkDatum := tauP2.toXBulk (of_decide_eq_true tauP2_xbulkB)

/-- **H4(iv) on the model** ([C] `lem:model7coincidences`(1)): the stabiliser of `ι(σ)` is `S(σ, σ)`; it contains
`-I` for every simplex; its order is `6, 6, 4` at `v₄, v₅, v₆` (where it contains `ST`, `TS`, `S`) and `2` elsewhere
(so it is `{±I}` there). -/
theorem model7_stab :
    (∀ q (σ : model7.V q) (δ : SL(2, ℤ)),
      δ ∈ MulAction.stabilizer SL(2, ℤ) (XBulk.SimpM.mk (model7.M q) (1, σ)) ↔ δ ∈ (model7.M q).S σ σ) ∧
    (∀ q (σ : model7.V q), (-1 : SL(2, ℤ)) ∈ (model7.M q).S σ σ) ∧
    (∀ q (σ : model7.V q), ((model7.M q).S σ σ).card = stabOrd q σ.1) ∧
    M22.toSL ⟨0, -1, 1, 1⟩ ∈ tauP2.Sset 0 4 4 ∧ M22.toSL ⟨1, -1, 1, 0⟩ ∈ tauP2.Sset 0 5 5 ∧
      M22.toSL ⟨0, -1, 1, 0⟩ ∈ tauP2.Sset 0 6 6 := by
  have hX := of_decide_eq_true tauP2_xbulkB
  refine ⟨fun q σ δ => ?_, fun q σ => ?_, fun q σ => ?_, ?_⟩
  · rw [XBulk.mem_stab_iff model7 q (1, σ) δ]; simp
  · obtain ⟨x, hx, hxq, hxσ⟩ := TauData.mem_idxL.1 ((tauP2.mem_idxF).1 σ.2)
    obtain ⟨t, ht, htg⟩ := tauP2_negI_check x hx
    rw [hxq, hxσ] at ht
    show _ ∈ tauP2.Sset q σ.1 σ.1
    exact (tauP2.mem_Sset).2 ⟨t, ht, by rw [htg, M22.toSL_negOne]⟩
  · obtain ⟨x, hx, hxq, hxσ⟩ := TauData.mem_idxL.1 ((tauP2.mem_idxF).1 σ.2)
    have := tauP2_ord_check x hx
    rw [hxq, hxσ] at this
    show (tauP2.Sset q σ.1 σ.1).card = _
    rw [TauData.card_Sset (TauData.keyAx_all hX q), this]
  · obtain ⟨⟨t1, h1, e1⟩, ⟨t2, h2, e2⟩, ⟨t3, h3, e3⟩⟩ := tauP2_gen_check
    exact ⟨(tauP2.mem_Sset).2 ⟨t1, h1, by rw [e1]⟩, (tauP2.mem_Sset).2 ⟨t2, h2, by rw [e2]⟩,
      (tauP2.mem_Sset).2 ⟨t3, h3, by rw [e3]⟩⟩

/-- **[C] `lem:model7coincidences`(2) and `def:model7rel` on the model** (WP-17): in `S(σ, σ')` terms, the eight
listed rows hold, and distinct simplices are related exactly at `coincPairs`. -/
theorem model7_coincidences :
    (∀ r ∈ model7RelRows, M22.toSL r.2.2.2 ∈ tauP2.Sset r.1 r.2.1 r.2.2.1) ∧
    (∀ x ∈ tauP2.simplices, ∀ y ∈ tauP2.simplices, x.1 = y.1 → x.2.1 ≠ y.2.1 →
      ((tauP2.Sset x.1 x.2.1 y.2.1).Nonempty ↔ (x.1, x.2.1, y.2.1) ∈ coincPairs)) := by
  refine ⟨fun r hr => ?_, fun x hx y hy hq hne => ?_⟩
  · obtain ⟨t, ht, htg, -⟩ := tauP2_rel_check r hr
    exact (tauP2.mem_Sset).2 ⟨t, ht, by rw [htg]⟩
  · rw [← tauP2_pairs_check x hx y hy hq hne]
    constructor
    · rintro ⟨γ, hγ⟩ hnil
      obtain ⟨t, ht, -⟩ := (tauP2.mem_Sset).1 hγ
      rw [hnil] at ht; exact List.not_mem_nil ht
    · intro h
      obtain ⟨t, ht⟩ := List.exists_mem_of_ne_nil _ h
      exact ⟨_, (tauP2.mem_Sset).2 ⟨t, ht, rfl⟩⟩

/-- **H4(i) and (iii) on the model**: no inversions, finite stabilisers, fundamental domain (the generic theorems of
`GL2.Model.H4` at `model7`). -/
theorem model7_h4_i_iii :
    (∀ q (γ : SL(2, ℤ)) (x : SL(2, ℤ) × model7.V q),
      γ • XBulk.SimpM.mk (model7.M q) x = XBulk.SimpM.mk (model7.M q) x → γ • XBulk.genQ model7 q x = XBulk.genQ model7 q x) ∧
    (∀ q (x : SL(2, ℤ) × model7.V q),
      (MulAction.stabilizer SL(2, ℤ) (XBulk.SimpM.mk (model7.M q) x) : Set SL(2, ℤ)).Finite) ∧
    (∀ q (s : XBulk.Simp model7 q), ∃ (γ : SL(2, ℤ)) (σ : model7.V q),
      (model7.M q).rep σ = σ ∧ s = γ • XBulk.SimpM.mk (model7.M q) (1, σ)) :=
  ⟨XBulk.noInv model7, XBulk.finiteStab model7, XBulk.fundDomain model7⟩

end GL2.Model
