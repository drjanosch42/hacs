/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.SmithPK

/-!
# LV (slot 23, follow-up R1(a) of `LV_audit.md`): the valuation counts are certified

`smithPK_card` certifies only the order `#span = ℓ^{Σ (K − v)}`. Here we prove the **type**: for every `t`,

  `#(ℓ^t · span) = ℓ^{Σ_k (K − (v_k + t))}`   (truncated subtraction),

the sum over the pivots of `smithPK` (`smithPK_card_smul`). The numbers `#(ℓ^t · L)`, `t = 0, 1, …`, determine a finite
`ℤ/ℓ^K`-module `L` up to isomorphism, so the multiset `{v_k}` is the multiset of valuations of the elementary divisors
`≠ 0 mod ℓ^K` (an invariant of the span). In counts: `card_smul_of_sig` turns a kernel-decided `sig K … = (s, cs)` into
`#(ℓ^t · span) = ℓ^{Σ_{v<K} cs_v (K − (v + t))}` for every `t`.

Proof: an abstract echelon family of vectors over `ZMod ℓ^K` (`VEch`; valuation `v` may now be `≥ K`, the vector is then
`0`) spans `ℓ^{Σ (K − v)}` elements (`card_vspan`); multiplying the family by `ℓ^t` gives an echelon family with valuations
`v + t` (`VEch.smul`), whose span is `ℓ^t · span` (`map_vspan_smul`).
-/

noncomputable section

namespace GL2.L23.SmithPK

variable {l K w : ℕ}

/-- An abstract pivot `(v, j, p)` over `ZMod ℓ^K`: `p_j = ℓ^v u` with `u` a unit, every entry divisible by `ℓ^v`. -/
structure VPiv (l K w v j : ℕ) (p : Fin w → ZMod (l ^ K)) : Prop where
  hj : j < w
  unit : ∃ u : ZMod (l ^ K), IsUnit u ∧ p ⟨j, hj⟩ = (l : ZMod (l ^ K)) ^ v * u
  dvd : ∀ i, ∃ c : ZMod (l ^ K), p i = (l : ZMod (l ^ K)) ^ v * c

/-- An abstract echelon family in creation order: every later vector vanishes at the column of every earlier one. -/
def VEch (l K w : ℕ) : List (ℕ × ℕ × (Fin w → ZMod (l ^ K))) → Prop
  | [] => True
  | q :: C => VEch l K w C ∧ VPiv l K w q.1 q.2.1 q.2.2 ∧ ∀ q' ∈ C, ∀ h : q.2.1 < w, q'.2.2 ⟨q.2.1, h⟩ = 0

/-- The span of the vectors of a family. -/
def vspan (l K w : ℕ) (C : List (ℕ × ℕ × (Fin w → ZMod (l ^ K)))) : Submodule (ZMod (l ^ K)) (Fin w → ZMod (l ^ K)) :=
  Submodule.span _ {y | ∃ q ∈ C, q.2.2 = y}

lemma vspan_cons (q : ℕ × ℕ × (Fin w → ZMod (l ^ K))) (C : List (ℕ × ℕ × (Fin w → ZMod (l ^ K)))) :
    vspan l K w (q :: C) = Submodule.span _ {q.2.2} ⊔ vspan l K w C := by
  rw [vspan, vspan, ← Submodule.span_union]
  congr 1; ext y
  simp only [List.mem_cons, Set.mem_ofPred_eq, Set.mem_union, Set.mem_singleton_iff]
  constructor
  · rintro ⟨q', h | h, rfl⟩
    · exact Or.inl (h ▸ rfl)
    · exact Or.inr ⟨q', h, rfl⟩
  · rintro (rfl | ⟨q', h, rfl⟩)
    · exact ⟨q, Or.inl rfl, rfl⟩
    · exact ⟨q', Or.inr h, rfl⟩

lemma lpow_eq_zero {v : ℕ} (hv : K ≤ v) : (l : ZMod (l ^ K)) ^ v = 0 := by
  rw [show v = K + (v - K) by omega, pow_add, show ((l : ZMod (l ^ K)) ^ K) = ((l ^ K : ℕ) : ZMod (l ^ K)) by push_cast; rfl,
    ZMod.natCast_self, zero_mul]

theorem vsmul_eq_zero {v j : ℕ} {p : Fin w → ZMod (l ^ K)} (hq : VPiv l K w v j p) (a : ZMod (l ^ K)) :
    a • p = 0 ↔ (l : ZMod (l ^ K)) ^ v * a = 0 := by
  obtain ⟨u, hu, hpj⟩ := hq.unit
  constructor
  · intro h
    have := congrFun h ⟨j, hq.hj⟩
    simp only [Pi.smul_apply, smul_eq_mul, Pi.zero_apply, hpj] at this
    have h2 : (a * (l : ZMod (l ^ K)) ^ v) * u = 0 := by rw [← this]; ring
    rw [hu.mul_left_eq_zero] at h2
    rw [mul_comm]; exact h2
  · intro h
    funext i
    obtain ⟨c, hc⟩ := hq.dvd i
    simp only [Pi.smul_apply, smul_eq_mul, Pi.zero_apply, hc]
    rw [← mul_assoc, mul_comm a, h, zero_mul]

/-- One pivot vector of valuation `v` spans `ℓ^{K − v}` elements (`1` when `v ≥ K`). -/
theorem card_span_vpiv (hl : l.Prime) {v j : ℕ} {p : Fin w → ZMod (l ^ K)} (hq : VPiv l K w v j p) :
    Nat.card (Submodule.span (ZMod (l ^ K)) {p}) = l ^ (K - v) := by
  have : NeZero (l ^ K) := ⟨(pow_pos hl.pos K).ne'⟩
  by_cases hvK : K ≤ v
  · have hp : p = 0 := funext fun i => by
      obtain ⟨c, hc⟩ := hq.dvd i
      rw [hc, lpow_eq_zero hvK, zero_mul]; rfl
    rw [hp, Submodule.span_singleton_eq_bot.mpr rfl, Nat.sub_eq_zero_of_le hvK, pow_zero]
    simp
  push Not at hvK
  set m := l ^ (K - v) with hm
  have hlK : l ^ K = l ^ v * m := by rw [hm, ← pow_add]; congr 1; omega
  have hann := vsmul_eq_zero hq
  have hzero : ∀ t : ℕ, ((m * t : ℕ) : ZMod (l ^ K)) • p = 0 := fun t => by
    rw [hann]
    have : ((l ^ v * (m * t) : ℕ) : ZMod (l ^ K)) = 0 := by
      rw [← mul_assoc, ← hlK, Nat.cast_mul, ZMod.natCast_self, zero_mul]
    push_cast at this ⊢; exact this
  let f : Fin m → Submodule.span (ZMod (l ^ K)) {p} := fun t =>
    ⟨((t : ℕ) : ZMod (l ^ K)) • p, Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self _)⟩
  have hf : Function.Bijective f := by
    constructor
    · intro t1 t2 h
      have h' : ((t1 : ℕ) : ZMod (l ^ K)) • p = ((t2 : ℕ) : ZMod (l ^ K)) • p := congrArg Subtype.val h
      wlog hle : (t1 : ℕ) ≤ t2 generalizing t1 t2
      · exact (this h.symm h'.symm (by omega)).symm
      have hd : (((t2 : ℕ) - t1 : ℕ) : ZMod (l ^ K)) • p = 0 := by
        rw [Nat.cast_sub hle, sub_smul, h', sub_self]
      rw [hann] at hd
      have hd' : ((l ^ v * ((t2 : ℕ) - t1) : ℕ) : ZMod (l ^ K)) = 0 := by push_cast; exact hd
      rw [ZMod.natCast_eq_zero_iff, hlK] at hd'
      have hdvd := Nat.dvd_of_mul_dvd_mul_left (pow_pos hl.pos _) hd'
      have : (t2 : ℕ) - t1 = 0 := Nat.eq_zero_of_dvd_of_lt hdvd (by have := t2.2; omega)
      exact Fin.ext (by omega)
    · rintro ⟨x, hx⟩
      obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hx
      refine ⟨⟨a.val % m, Nat.mod_lt _ (pow_pos hl.pos _)⟩, Subtype.ext ?_⟩
      show ((a.val % m : ℕ) : ZMod (l ^ K)) • p = a • p
      have ha : a = ((a.val % m : ℕ) : ZMod (l ^ K)) + ((m * (a.val / m) : ℕ) : ZMod (l ^ K)) := by
        rw [← Nat.cast_add, Nat.mod_add_div, ZMod.natCast_zmod_val]
      conv_rhs => rw [ha]
      rw [add_smul, hzero, add_zero]
  rw [← Nat.card_eq_of_bijective f hf, Nat.card_eq_fintype_card, Fintype.card_fin]

/-- **The span of an abstract echelon family** has `ℓ^{Σ (K − v)}` elements. -/
theorem card_vspan (hl : l.Prime) : ∀ C : List (ℕ × ℕ × (Fin w → ZMod (l ^ K))), VEch l K w C →
    Nat.card (vspan l K w C) = l ^ (C.map fun q => K - q.1).sum
  | [], _ => by
    rw [show vspan l K w [] = ⊥ from by rw [vspan, Submodule.span_eq_bot]; rintro _ ⟨q, hq, _⟩; simp at hq]
    simp
  | q :: C, ⟨hC, hq, hz⟩ => by
    rw [vspan_cons, List.map_cons, List.sum_cons, pow_add, ← card_span_vpiv hl hq, ← card_vspan hl C hC]
    apply card_sup_of_inf_eq_bot
    rw [eq_bot_iff]
    rintro x ⟨hx1, hx2⟩
    obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hx1
    rw [Submodule.mem_bot, vsmul_eq_zero hq]
    have hvan : ∀ y ∈ vspan l K w C, y ⟨q.2.1, hq.hj⟩ = 0 := by
      intro y hy
      have : vspan l K w C ≤ LinearMap.ker (LinearMap.proj (R := ZMod (l ^ K))
          (φ := fun _ : Fin w => ZMod (l ^ K)) ⟨q.2.1, hq.hj⟩) := by
        rw [vspan, Submodule.span_le]
        rintro _ ⟨q', hq', rfl⟩
        simp only [SetLike.mem_coe, LinearMap.mem_ker, LinearMap.proj_apply]
        exact hz q' hq' hq.hj
      exact this hy
    have h0 := hvan _ hx2
    obtain ⟨u, hu, hpj⟩ := hq.unit
    simp only [Pi.smul_apply, smul_eq_mul, hpj] at h0
    have h2 : (a * (l : ZMod (l ^ K)) ^ q.1) * u = 0 := by rw [← h0]; ring
    rw [hu.mul_left_eq_zero] at h2
    rw [mul_comm]; exact h2

/-- Multiplying by `ℓ^t` raises every valuation by `t`. -/
lemma VPiv.smul {v j : ℕ} {p : Fin w → ZMod (l ^ K)} (hq : VPiv l K w v j p) (t : ℕ) :
    VPiv l K w (v + t) j ((l : ZMod (l ^ K)) ^ t • p) := by
  obtain ⟨u, hu, hpj⟩ := hq.unit
  refine ⟨hq.hj, ⟨u, hu, ?_⟩, fun i => ?_⟩
  · simp only [Pi.smul_apply, smul_eq_mul, hpj]; ring
  · obtain ⟨c, hc⟩ := hq.dvd i
    exact ⟨c, by simp only [Pi.smul_apply, smul_eq_mul, hc]; ring⟩

/-- The family multiplied by `ℓ^t`. -/
def vscale (t : ℕ) (C : List (ℕ × ℕ × (Fin w → ZMod (l ^ K)))) : List (ℕ × ℕ × (Fin w → ZMod (l ^ K))) :=
  C.map fun q => (q.1 + t, q.2.1, (l : ZMod (l ^ K)) ^ t • q.2.2)

theorem VEch.smul (t : ℕ) : ∀ C : List (ℕ × ℕ × (Fin w → ZMod (l ^ K))), VEch l K w C → VEch l K w (vscale t C)
  | [], _ => trivial
  | q :: C, ⟨hC, hq, hz⟩ => by
    refine ⟨VEch.smul t C hC, hq.smul t, fun q' hq' h => ?_⟩
    obtain ⟨q'', h'', rfl⟩ := List.mem_map.mp hq'
    simp only [Pi.smul_apply, smul_eq_mul, hz q'' h'' h, mul_zero]

/-- `ℓ^t · span = span of the scaled family`. -/
theorem map_vspan_smul (t : ℕ) (C : List (ℕ × ℕ × (Fin w → ZMod (l ^ K)))) :
    (vspan l K w C).map ((l : ZMod (l ^ K)) ^ t • LinearMap.id) = vspan l K w (vscale t C) := by
  rw [vspan, Submodule.map_span, vspan]
  congr 1; ext y
  simp only [Set.mem_image, Set.mem_ofPred_eq, vscale, List.mem_map]
  constructor
  · rintro ⟨_, ⟨q, hq, rfl⟩, rfl⟩
    exact ⟨_, ⟨q, hq, rfl⟩, rfl⟩
  · rintro ⟨_, ⟨q, hq, rfl⟩, rfl⟩
    exact ⟨_, ⟨q, hq, rfl⟩, rfl⟩

/-- **The type of an abstract echelon family**: `#(ℓ^t · span) = ℓ^{Σ (K − (v + t))}`. -/
theorem card_smul_vspan (hl : l.Prime) (t : ℕ) (C : List (ℕ × ℕ × (Fin w → ZMod (l ^ K)))) (hC : VEch l K w C) :
    Nat.card ((vspan l K w C).map ((l : ZMod (l ^ K)) ^ t • LinearMap.id)) = l ^ (C.map fun q => K - (q.1 + t)).sum := by
  rw [map_vspan_smul, card_vspan hl _ (hC.smul t), vscale, List.map_map]
  rfl

/-! ### From the packed pivots of `smithPK` -/

variable {H : ℕ}

/-- The packed pivots as vectors. -/
def vfam (l K H w : ℕ) (C : List Piv) : List (ℕ × ℕ × (Fin w → ZMod (l ^ K))) :=
  C.map fun q => (q.1, q.2.1, vecK l K H w q.2.2)

theorem ech_vech (hl : l.Prime) : ∀ C : List Piv, Ech l K H w C → VEch l K w (vfam l K H w C)
  | [], _ => trivial
  | q :: C, ⟨hC, ⟨_, hj, _, hdv, hnd⟩, hz⟩ => by
    refine ⟨ech_vech hl C hC, ⟨hj, ?_, fun i => ?_⟩, fun q' hq' h => ?_⟩
    · obtain ⟨u, hu⟩ := hdv _ hj
      have hlu : ¬ l ∣ u := fun h => hnd (by rw [hu, pow_succ]; exact Nat.mul_dvd_mul_left _ h)
      refine ⟨u, (ZMod.isUnit_iff_coprime u (l ^ K)).mpr
        (Nat.Coprime.pow_right _ ((Nat.Prime.coprime_iff_not_dvd hl).mpr hlu).symm), ?_⟩
      simp only [vecK, hu]; push_cast; rfl
    · obtain ⟨c, hc⟩ := hdv i i.2
      exact ⟨c, by simp only [vecK, hc]; push_cast; rfl⟩
    · obtain ⟨q'', h'', rfl⟩ := List.mem_map.mp hq'
      simp only [vecK, hz q'' h'', Nat.cast_zero]

lemma spanK_rowsOf_eq_vspan (C : List Piv) : spanK l K H w (rowsOf C) = vspan l K w (vfam l K H w C) := by
  rw [spanK, vspan]
  congr 1; ext y
  simp only [Set.mem_ofPred_eq, rowsOf, vfam, List.mem_map]
  constructor
  · rintro ⟨_, ⟨q, hq, rfl⟩, rfl⟩
    exact ⟨_, ⟨q, hq, rfl⟩, rfl⟩
  · rintro ⟨_, ⟨q, hq, rfl⟩, rfl⟩
    exact ⟨_, ⟨q, hq, rfl⟩, rfl⟩

/-- **`smithPK` certifies the type**: for every `t`, `#(ℓ^t · span of the rows) = ℓ^{Σ (K − (v + t))}`, the sum over the
pivots. -/
theorem smithPK_card_smul {rows : List ℕ} (cx : Ctx l K H rows.length)
    (hrows : ∀ r ∈ rows, ∀ i < w, fld (2 * H) r i < l ^ K) (t : ℕ) :
    Nat.card ((spanK l K H w rows).map ((l : ZMod (l ^ K)) ^ t • LinearMap.id)) =
      l ^ ((smithPK l K H w rows).map fun q => K - (q.1 + t)).sum := by
  obtain ⟨h1, h2⟩ := smithPK_spec (w := w) cx hrows
  rw [h2, spanK_congr (L' := rowsOf (smithPK l K H w rows).reverse) (fun r => by simp [rowsOf]),
    spanK_rowsOf_eq_vspan, card_smul_vspan cx.prime t _ (ech_vech cx.prime _ (pivOK_ech _ h1)), vfam,
    List.map_map, List.map_reverse, List.sum_reverse]
  rfl

/-! ### The counts of `sig` -/

lemma cntV_cons (v : ℕ) (q : Piv) (P : List Piv) : cntV v (q :: P) = cntV v P + if q.1 = v then 1 else 0 := by
  show @Bool.rec (fun _ => ℕ) (cntV v P) (Nat.succ (cntV v P)) (Nat.beq q.1 v) = _
  by_cases h : q.1 = v
  · rw [if_pos h, h, show Nat.beq v v = true by simp]
  · rw [if_neg h, show Nat.beq q.1 v = false from Bool.eq_false_iff.mpr fun h' => h (Nat.eq_of_beq_eq_true h')]; rfl

theorem sum_eq_cnt (f : ℕ → ℕ) : ∀ P : List Piv, (∀ q ∈ P, q.1 < K) →
    (P.map fun q => f q.1).sum = ∑ v ∈ Finset.range K, cntV v P * f v
  | [], _ => by simp [cntV]
  | q :: P, hP => by
    rw [List.map_cons, List.sum_cons, sum_eq_cnt f P fun q' h => hP q' (List.mem_cons_of_mem _ h)]
    simp only [cntV_cons, add_mul, Finset.sum_add_distrib, ite_mul, one_mul, zero_mul]
    rw [Finset.sum_ite_eq, if_pos (Finset.mem_range.mpr (hP q (List.mem_cons_self ..))), add_comm]

lemma sig_go (P : List Piv) : ∀ k v, @Nat.rec (fun _ => ℕ → List ℕ) (fun _ => [])
    (fun _ ih v => cntV v P :: ih (Nat.succ v)) k v = (List.range k).map fun i => cntV (v + i) P
  | 0, v => rfl
  | k + 1, v => by
    show cntV v P :: _ = _
    rw [sig_go P k (v + 1), List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply]; congr 1; omega

lemma sig_snd (K : ℕ) (P : List Piv) : (sig K P).2 = (List.range K).map fun v => cntV v P := by
  show @Nat.rec (fun _ => ℕ → List ℕ) _ _ K 0 = _
  rw [sig_go]; simp

lemma pivOK_lt : ∀ P : List Piv, PivOK l K H w P → ∀ q ∈ P, q.1 < K
  | [], _, q, h => by simp at h
  | q :: P, ⟨hP, hv, _⟩, q', h => by
    rcases List.mem_cons.mp h with rfl | h
    · exact hv
    · exact pivOK_lt P hP q' h

/-- **The cell form of the type**: a kernel-decided `sig K (smithPK …) = (s, cs)` gives, for every `t`,
`#(ℓ^t · span) = ℓ^{Σ_{v<K} cs_v (K − (v + t))}`; so the counts `cs` are certified. -/
theorem card_smul_of_sig {rows : List ℕ} (cx : Ctx l K H rows.length)
    (hrows : ∀ r ∈ rows, ∀ i < w, fld (2 * H) r i < l ^ K) {s : ℕ} {cs : List ℕ}
    (h : sig K (smithPK l K H w rows) = (s, cs)) (t : ℕ) :
    Nat.card ((spanK l K H w rows).map ((l : ZMod (l ^ K)) ^ t • LinearMap.id)) =
      l ^ ∑ v ∈ Finset.range K, cs.getD v 0 * (K - (v + t)) := by
  rw [smithPK_card_smul cx hrows t,
    sum_eq_cnt (fun v => K - (v + t)) _ (pivOK_lt _ (smithPK_spec (w := w) cx hrows).1)]
  have hcs : cs = (List.range K).map fun v => cntV v (smithPK l K H w rows) := by
    rw [← sig_snd, h]
  congr 1
  refine Finset.sum_congr rfl fun v hv => ?_
  rw [hcs]
  simp [List.getD_eq_getElem?_getD, Finset.mem_range.mp hv]

end GL2.L23.SmithPK

end
