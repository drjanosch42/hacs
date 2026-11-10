import Mathlib
import GL2.Cosets.DoubleCoset

/-!
# G1 (L17-3): the upper-triangular set `B_N` and its cosets ([C] `prop:bdrycosets`)

`Γ_∂ = {±T^k}` is the stabiliser of the cusp `∞`, i.e. the upper-triangular elements of `SL(2, ℤ)`
(`gammaBd_iff`); `B_N = {δ ∈ Γ g Γ : δ upper triangular}`.
* (1) at `ν = 1`: `B_p = δ^∂_∞ Γ_∂ ⊔ ⊔_{t < p} δ^∂_t Γ_∂`, each once, with the three `T`-identities;
* (2) for every `ν`: `δ Γ_∂ ↦ δ Γ` is a bijection `B_N / Γ_∂ → Γ g Γ / Γ`; for each `δ_k` the `γ` with
  `δ_k γ ∈ B_N` form one left coset `γ_k Γ_∂`, and the `δ_k γ_k` represent `B_N / Γ_∂` (so there are `m`);
* (3) at `ν = 1`: the two double cosets and the two stabiliser intersections;
* (4) at `ν = 1`: the right-coset representatives `β_b`, `β_∞` represent `Γ_∂ \ B_p`.
-/

namespace GL2.Cosets

open Matrix CongruenceSubgroup
open scoped MatrixGroups

/-- `Γ_∂`, the upper-triangular elements of `SL(2, ℤ)`. -/
def GammaBd : Subgroup SL(2, ℤ) where
  carrier := {γ | γ 1 0 = 0}
  mul_mem' := by
    intro a b ha hb
    simp only [Set.mem_ofPred_eq, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_apply,
      Fin.sum_univ_two] at *
    rw [ha, hb]; ring
  one_mem' := by simp
  inv_mem' := by
    intro a ha
    simp only [Set.mem_ofPred_eq, coe_inv_apply] at *
    simp [ha]

lemma mem_GammaBd (γ : SL(2, ℤ)) : γ ∈ GammaBd ↔ γ 1 0 = 0 := Iff.rfl

/-- `Γ_∂ = {±T^k}`. -/
theorem gammaBd_iff (γ : SL(2, ℤ)) :
    γ ∈ GammaBd ↔ ∃ k : ℤ, γ = ModularGroup.T ^ k ∨ γ = -(ModularGroup.T ^ k) := by
  rw [mem_GammaBd]
  constructor
  · intro h
    have hdet := det_SL γ
    rw [h, mul_zero, sub_zero] at hdet
    rcases Int.eq_one_or_neg_one_of_mul_eq_one hdet with h0 | h0
    · have h1 : γ 1 1 = 1 := by rw [h0] at hdet; linarith
      refine ⟨γ 0 1, Or.inl ?_⟩
      ext i j; rw [ModularGroup.coe_T_zpow]; fin_cases i <;> fin_cases j <;> simp [h, h0, h1]
    · have h1 : γ 1 1 = -1 := by rw [h0] at hdet; linarith
      refine ⟨-γ 0 1, Or.inr ?_⟩
      ext i j
      rw [Matrix.SpecialLinearGroup.coe_neg, ModularGroup.coe_T_zpow]
      fin_cases i <;> fin_cases j <;> simp [h, h0, h1]
  · rintro ⟨k, rfl | rfl⟩
    · rw [ModularGroup.coe_T_zpow]; simp
    · rw [Matrix.SpecialLinearGroup.coe_neg, ModularGroup.coe_T_zpow]; simp

lemma DC_det {N : ℤ} {δ : Mat2 ℤ} (h : δ ∈ DC N) : δ.det = N := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe, Matrix.SpecialLinearGroup.det_coe,
    gMat_det]; ring

/-- `B_N`, the upper-triangular elements of `Γ g Γ`. -/
def BN (N : ℤ) : Set (Mat2 ℤ) := {δ | δ ∈ DC N ∧ δ 1 0 = 0}

/-- **[C] `prop:bdrycosets`(2), injectivity:** if `δ, δ' ∈ B_N` and `δ' = δ γ` with `γ ∈ Γ`, then
`γ ∈ Γ_∂`. -/
theorem bd_inj {N : ℤ} (hN : N ≠ 0) {δ δ' : Mat2 ℤ} (hδ : δ ∈ BN N) (hδ' : δ' ∈ BN N) {γ : SL(2, ℤ)}
    (h : δ' = δ * γ) : γ ∈ GammaBd := by
  rw [mem_GammaBd]
  have hdet : δ 0 0 * δ 1 1 = N := by
    have := ((DC_det hδ.1)); rw [Matrix.det_fin_two, hδ.2] at this; linarith
  have h10 := congrFun (congrFun h 1) 0
  rw [hδ'.2, Matrix.mul_apply, Fin.sum_univ_two, hδ.2, zero_mul, zero_add] at h10
  have h11 : δ 1 1 ≠ 0 := by intro h0; rw [h0, mul_zero] at hdet; exact hN hdet.symm
  rcases mul_eq_zero.1 h10.symm with h0 | h0
  · exact absurd h0 h11
  · exact h0

/-- **[C] `prop:bdrycosets`(2), surjectivity:** every left coset `δ Γ ⊆ Γ g Γ` meets `B_N`. -/
theorem bd_surj {N : ℤ} {δ : Mat2 ℤ} (hδ : δ ∈ DC N) : ∃ γ : SL(2, ℤ), δ * (γ : Mat2 ℤ) ∈ BN N := by
  -- Bézout on the bottom row `(c, d)` of `δ`: find `γ` with `(δ γ)₁₀ = 0`.
  obtain ⟨γ, hγ⟩ : ∃ γ : SL(2, ℤ), (δ * (γ : Mat2 ℤ)) 1 0 = 0 := by
    by_cases hz : δ 1 0 = 0 ∧ δ 1 1 = 0
    · exact ⟨1, by simp [Matrix.mul_apply, Fin.sum_univ_two, hz.1]⟩
    · set c := δ 1 0 with hcdef
      set d := δ 1 1 with hddef
      have hne : d ≠ 0 ∨ c ≠ 0 := by tauto
      have hpos : 0 < Int.gcd d c := Int.gcd_pos_iff.2 hne
      obtain ⟨d', hd'⟩ := Int.gcd_dvd_left d c
      obtain ⟨c', hc'⟩ := Int.gcd_dvd_right d c
      have hb := Int.gcd_eq_gcd_ab d c
      have he : ((Int.gcd d c : ℕ) : ℤ) ≠ 0 := by exact_mod_cast hpos.ne'
      have h1 : d' * Int.gcdA d c + c' * Int.gcdB d c = 1 := by
        apply mul_left_cancel₀ he
        have : ((Int.gcd d c : ℕ) : ℤ) = ((Int.gcd d c : ℕ) : ℤ) * d' * Int.gcdA d c +
            ((Int.gcd d c : ℕ) : ℤ) * c' * Int.gcdB d c := by
          conv_lhs => rw [hb]
          rw [← hd', ← hc']
        linear_combination -this
      have key : c * d' = d * c' := by
        apply mul_left_cancel₀ he
        linear_combination (-c) * hd' + d * hc'
      refine ⟨mkSL d' (Int.gcdB d c) (-c') (Int.gcdA d c) (by linarith), ?_⟩
      simp only [mkSL_coe, Matrix.mul_apply, Fin.sum_univ_two]
      simp
      rw [← hcdef, ← hddef]
      linear_combination key
  exact ⟨γ, DC_right_mul hδ γ, hγ⟩

/-- **[C] `prop:bdrycosets`(2), the `γ_k`:** for each `δ ∈ Γ g Γ` (in particular `δ_k`) the set
`{γ ∈ Γ : δ γ ∈ B_N}` is a single left coset `γ₀ Γ_∂`. -/
theorem bd_fibre {N : ℤ} (hN : N ≠ 0) {δ : Mat2 ℤ} (hδ : δ ∈ DC N) :
    ∃ γ₀ : SL(2, ℤ), ∀ γ : SL(2, ℤ), δ * (γ : Mat2 ℤ) ∈ BN N ↔ γ₀⁻¹ * γ ∈ GammaBd := by
  obtain ⟨γ₀, h₀⟩ := bd_surj hδ
  refine ⟨γ₀, fun γ => ⟨fun h => ?_, fun h => ?_⟩⟩
  · apply bd_inj hN h₀ h
    rw [Matrix.SpecialLinearGroup.coe_mul, ← Matrix.mul_assoc, Matrix.mul_assoc δ, mul_inv_coe, Matrix.mul_one]
  · refine ⟨DC_right_mul hδ γ, ?_⟩
    have hγ : γ = γ₀ * (γ₀⁻¹ * γ) := by group
    rw [hγ, Matrix.SpecialLinearGroup.coe_mul, ← Matrix.mul_assoc, Matrix.mul_apply, Fin.sum_univ_two,
      h₀.2, (mem_GammaBd _).1 h]; ring

/-- **[C] `prop:bdrycosets`(2), the count:** the `δ_k γ_k`, `k < m`, represent `B_N / Γ_∂`, each class
once; so `B_N` is the disjoint union of exactly `m` left cosets of `Γ_∂` ([C] (1) for every `ν`). -/
theorem bd_leftCosets {p ν : ℕ} (hp : p.Prime) (hν : 1 ≤ ν) (γk : ℕ → SL(2, ℤ))
    (hγk : ∀ k, deltaMat p ν k * (γk k : Mat2 ℤ) ∈ BN ((p : ℤ) ^ ν)) {δ : Mat2 ℤ}
    (hδ : δ ∈ BN ((p : ℤ) ^ ν)) :
    ∃! k, k < mIdx p ν ∧ ∃ u ∈ GammaBd, δ = deltaMat p ν k * (γk k : Mat2 ℤ) * (u : Mat2 ℤ) := by
  have hN : ((p : ℤ) ^ ν) ≠ 0 := pow_ne_zero _ (by exact_mod_cast hp.ne_zero)
  obtain ⟨k, ⟨hk, γ, hγ⟩, huniq⟩ := leftCosets hp hν hδ.1
  refine ⟨k, ⟨hk, (γk k)⁻¹ * γ, ?_, ?_⟩, ?_⟩
  · apply bd_inj hN (hγk k) hδ
    rw [hγ, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc, mul_inv_coe_cancel_left]
  · rw [hγ, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc, mul_inv_coe_cancel_left]
  · rintro k' ⟨hk', u, _, hu⟩
    exact huniq k' ⟨hk', γk k' * u, by rw [hu, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc]⟩

/-! ## `ν = 1` -/

/-- `δ^∂_∞ = diag(1, p) = g` (index `p`) and `δ^∂_t = [[p, t], [0, 1]]` (`t < p`). -/
def dbd (p t : ℕ) : Mat2 ℤ := if t < p then !![(p : ℤ), (t : ℤ); 0, 1] else !![1, 0; 0, (p : ℤ)]

/-- **[C] `prop:bdrycosets`(1), the identities:** `T δ^∂_∞ = δ^∂_∞ T^p`, `T δ^∂_t = δ^∂_{t+1}`
(`t < p - 1`), `T δ^∂_{p-1} = δ^∂_0 T`. -/
theorem dbd_T_identities (p : ℕ) (hp : 2 ≤ p) :
    (ModularGroup.T : Mat2 ℤ) * dbd p p = dbd p p * ((ModularGroup.T ^ (p : ℤ) : SL(2, ℤ)) : Mat2 ℤ) ∧
    (∀ t, t + 1 < p → (ModularGroup.T : Mat2 ℤ) * dbd p t = dbd p (t + 1)) ∧
    (ModularGroup.T : Mat2 ℤ) * dbd p (p - 1) = dbd p 0 * (ModularGroup.T : Mat2 ℤ) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [ModularGroup.coe_T_zpow, ModularGroup.coe_T, dbd, if_neg (lt_irrefl _)]
    ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  · intro t ht
    rw [ModularGroup.coe_T, dbd, dbd, if_pos (by omega), if_pos ht]
    ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  · rw [ModularGroup.coe_T, dbd, dbd, if_pos (by omega), if_pos (by omega)]
    ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
    push_cast [Nat.cast_sub (by omega : 1 ≤ p)]; ring

lemma dbd_mem (p t : ℕ) (hp : p.Prime) (ht : t ≤ p) : dbd p t ∈ BN ((p : ℤ) ^ 1) := by
  unfold dbd
  split_ifs with h
  · refine ⟨(mem_DC_iff hp _).2 ⟨by simp [Matrix.det_fin_two_of], ?_⟩, rfl⟩
    intro e he; have := he 1 1; simp at this; exact isUnit_of_dvd_one this
  · refine ⟨(mem_DC_iff hp _).2 ⟨by simp [Matrix.det_fin_two_of], ?_⟩, rfl⟩
    intro e he; have := he 0 0; simp at this; exact isUnit_of_dvd_one this

/-- Uniqueness of the normal form `[[a, b], [0, d]]`, `0 ≤ b < a`, under right multiplication by `Γ_∂`. -/
theorem upper_right_unique {a b d a' b' d' : ℤ} {u : SL(2, ℤ)} (hu : u ∈ GammaBd)
    (h : !![a, b; 0, d] * (u : Mat2 ℤ) = !![a', b'; 0, d'])
    (ha : 0 < a) (hb : 0 ≤ b) (hba : b < a) (ha' : 0 < a') (hb' : 0 ≤ b') (hba' : b' < a') :
    a = a' ∧ b = b' ∧ d = d' := by
  rw [mem_GammaBd] at hu
  have hdet := det_SL u
  rw [hu, mul_zero, sub_zero] at hdet
  have e00 := congrFun (congrFun h 0) 0
  have e01 := congrFun (congrFun h 0) 1
  have e11 := congrFun (congrFun h 1) 1
  simp [Matrix.mul_apply, Fin.sum_univ_two, hu] at e00 e01 e11
  have hu0 : 0 < u 0 0 := by by_contra hc; nlinarith
  have hu1 : u 0 0 = 1 ∧ u 1 1 = 1 := by
    have : 0 < u 1 1 := by nlinarith
    constructor <;> nlinarith
  obtain ⟨h0', h1'⟩ := hu1
  rw [h1'] at e01 e11
  rw [h0'] at e00
  have hx : u 0 1 = 0 := by
    by_contra hne
    rcases lt_or_gt_of_ne hne with hl | hl <;> nlinarith
  rw [hx] at e01
  exact ⟨by linarith, by linarith, by linarith⟩

/-- **[C] `prop:bdrycosets`(1) at `ν = 1`:** `B_p = ⊔_{x ∈ {0, …, p-1, ∞}} δ^∂_x Γ_∂`, each once. -/
theorem bd_leftCosets_nu_one {p : ℕ} (hp : p.Prime) {δ : Mat2 ℤ} (hδ : δ ∈ BN ((p : ℤ) ^ 1)) :
    ∃! t, t ≤ p ∧ ∃ u ∈ GammaBd, δ = dbd p t * (u : Mat2 ℤ) := by
  have hp0 : (0 : ℤ) < p := by exact_mod_cast hp.pos
  have hdet := DC_det hδ.1
  rw [pow_one, Matrix.det_fin_two, hδ.2, mul_zero, sub_zero] at hdet
  set a := δ 0 0; set b := δ 0 1; set d := δ 1 1
  have hform : δ = !![a, b; 0, d] := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [a, b, d, hδ.2]
  -- normalise the sign: `a > 0`
  obtain ⟨σ, hσ, a₁, b₁, d₁, hσeq, ha₁, had₁⟩ : ∃ σ : SL(2, ℤ), σ ∈ GammaBd ∧ ∃ a₁ b₁ d₁ : ℤ,
      δ * (σ : Mat2 ℤ) = !![a₁, b₁; 0, d₁] ∧ 0 < a₁ ∧ a₁ * d₁ = p := by
    rcases lt_or_gt_of_ne (show a ≠ 0 by intro h0; rw [h0, zero_mul] at hdet; linarith) with hn | hpos
    · refine ⟨mkSL (-1) 0 0 (-1) (by ring), rfl, -a, -b, -d, ?_, by linarith, by linarith⟩
      rw [hform]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
    · exact ⟨1, one_mem _, a, b, d, by rw [hform]; simp, hpos, hdet⟩
  have hadvd : a₁.natAbs ∣ p := ⟨d₁.natAbs, by rw [← Int.natAbs_mul, had₁]; simp⟩
  -- reduce `b₁` modulo `a₁` with `T^{-q}` on the right
  set q := b₁ / a₁
  set τ : SL(2, ℤ) := mkSL 1 (-q) 0 1 (by ring)
  have hτ : τ ∈ GammaBd := rfl
  have hred : δ * ((σ * τ : SL(2, ℤ)) : Mat2 ℤ) = !![a₁, b₁ % a₁; 0, d₁] := by
    rw [Matrix.SpecialLinearGroup.coe_mul, ← Matrix.mul_assoc, hσeq, Int.emod_def]
    ext i j; fin_cases i <;> fin_cases j <;> simp [τ, Matrix.mul_apply, Fin.sum_univ_two]; ring
  have hb0 := Int.emod_nonneg b₁ ha₁.ne'
  have hb1 := Int.emod_lt_of_pos b₁ ha₁
  have hinv : δ = !![a₁, b₁ % a₁; 0, d₁] * (((σ * τ)⁻¹ : SL(2, ℤ)) : Mat2 ℤ) := by
    rw [← hred, Matrix.mul_assoc, mul_inv_coe, Matrix.mul_one]
  have hmem : (σ * τ)⁻¹ ∈ GammaBd := inv_mem (mul_mem hσ hτ)
  have exist : ∃ t, t ≤ p ∧ ∃ u ∈ GammaBd, δ = dbd p t * (u : Mat2 ℤ) := by
    rcases (Nat.dvd_prime hp).1 hadvd with h1 | h1
    · have ha1 : a₁ = 1 := by omega
      have hd1 : d₁ = p := by rw [ha1, one_mul] at had₁; exact had₁
      have hb : b₁ % a₁ = 0 := by omega
      refine ⟨p, le_refl _, _, hmem, ?_⟩
      rw [hinv, dbd, if_neg (lt_irrefl _), hb, ha1, hd1]
    · have hap : a₁ = p := by omega
      refine ⟨(b₁ % a₁).toNat, by omega, _, hmem, ?_⟩
      have hd1 : d₁ = 1 := by rw [hap] at had₁; nlinarith
      rw [hinv, dbd, if_pos (by omega), Int.toNat_of_nonneg hb0, hap, hd1]
  obtain ⟨t, ht, u, hu, hδu⟩ := exist
  refine ⟨t, ⟨ht, u, hu, hδu⟩, ?_⟩
  rintro t' ⟨ht', u', hu', hδu'⟩
  have hrel : dbd p t * ((u * u'⁻¹ : SL(2, ℤ)) : Mat2 ℤ) = dbd p t' := by
    rw [Matrix.SpecialLinearGroup.coe_mul, ← Matrix.mul_assoc, ← hδu, hδu', Matrix.mul_assoc, mul_inv_coe,
      Matrix.mul_one]
  have hm : u * u'⁻¹ ∈ GammaBd := mul_mem hu (inv_mem hu')
  unfold dbd at hrel
  split_ifs at hrel with h1 h2 h2
  · have := upper_right_unique hm hrel hp0 (by positivity) (by exact_mod_cast h1) hp0 (by positivity)
      (by exact_mod_cast h2)
    have := this.2.1; exact_mod_cast this.symm
  · have := upper_right_unique hm hrel hp0 (by positivity) (by exact_mod_cast h1) one_pos le_rfl one_pos
    have := hp.one_lt; omega
  · have := upper_right_unique hm hrel one_pos le_rfl one_pos hp0 (by positivity) (by exact_mod_cast h2)
    have := hp.one_lt; omega
  · omega

/-- **[C] `prop:bdrycosets`(3) at `ν = 1`, the stabilisers:** every `u ∈ Γ_∂` satisfies
`u δ^∂_∞ ∈ δ^∂_∞ Γ_∂` (the intersection is `Γ_∂`, one left coset), and `u δ^∂_0 ∈ δ^∂_0 Γ_∂` iff `p ∣ u₀₁`
(the intersection is `⟨T^p⟩ × ⟨-I⟩`, of index `p`). -/
theorem bd_stabilisers (p : ℕ) (hp : 0 < p) (u : SL(2, ℤ)) (hu : u ∈ GammaBd) :
    (∃ u' ∈ GammaBd, (u : Mat2 ℤ) * dbd p p = dbd p p * (u' : Mat2 ℤ)) ∧
    ((∃ u' ∈ GammaBd, (u : Mat2 ℤ) * dbd p 0 = dbd p 0 * (u' : Mat2 ℤ)) ↔ (p : ℤ) ∣ u 0 1) := by
  rw [mem_GammaBd] at hu
  have hdet := det_SL u
  rw [hu, mul_zero, sub_zero] at hdet
  refine ⟨⟨mkSL (u 0 0) (p * u 0 1) 0 (u 1 1) (by linarith), rfl, ?_⟩, ⟨?_, ?_⟩⟩
  · rw [dbd, if_neg (lt_irrefl _)]
    ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two, hu] <;> ring
  · rintro ⟨u', _, h⟩
    rw [dbd, if_pos hp] at h
    have := congrFun (congrFun h 0) 1
    simp [Matrix.mul_apply, Fin.sum_univ_two, hu] at this
    exact ⟨u' 0 1, this⟩
  · rintro ⟨f, hf⟩
    refine ⟨mkSL (u 0 0) f 0 (u 1 1) (by linarith), rfl, ?_⟩
    rw [dbd, if_pos hp]
    ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two, hu, hf] <;> ring

/-- **[C] `prop:bdrycosets`(3) at `ν = 1`, the two double cosets:** `B_p = Γ_∂ δ^∂_∞ Γ_∂ ⊔ Γ_∂ δ^∂_0 Γ_∂`. -/
theorem bd_doubleCosets {p : ℕ} (hp : p.Prime) {δ : Mat2 ℤ} (hδ : δ ∈ BN ((p : ℤ) ^ 1)) :
    ((∃ u ∈ GammaBd, ∃ u' ∈ GammaBd, δ = (u : Mat2 ℤ) * dbd p p * u') ∨
      (∃ u ∈ GammaBd, ∃ u' ∈ GammaBd, δ = (u : Mat2 ℤ) * dbd p 0 * u')) ∧
    ¬ ((∃ u ∈ GammaBd, ∃ u' ∈ GammaBd, δ = (u : Mat2 ℤ) * dbd p p * u') ∧
      (∃ u ∈ GammaBd, ∃ u' ∈ GammaBd, δ = (u : Mat2 ℤ) * dbd p 0 * u')) := by
  -- `|δ₀₀|` is `1` on the first and `p` on the second
  have key : ∀ (u u' : SL(2, ℤ)), u ∈ GammaBd → u' ∈ GammaBd → ∀ t, t ≤ p →
      ((u : Mat2 ℤ) * dbd p t * u') 0 0 = u 0 0 * (dbd p t) 0 0 * u' 0 0 := by
    intro u u' hu hu' t _
    rw [mem_GammaBd] at hu hu'
    unfold dbd; split_ifs <;> simp [Matrix.mul_apply, Fin.sum_univ_two, hu, hu']
  have unit : ∀ u : SL(2, ℤ), u ∈ GammaBd → |u 0 0| = 1 := by
    intro u hu; rw [mem_GammaBd] at hu
    have hdet := det_SL u; rw [hu, mul_zero, sub_zero] at hdet
    rcases Int.eq_one_or_neg_one_of_mul_eq_one hdet with h | h <;> simp [h]
  refine ⟨?_, ?_⟩
  · obtain ⟨t, ⟨ht, u, hu, hδu⟩, _⟩ := bd_leftCosets_nu_one hp hδ
    by_cases htp : t = p
    · subst htp; exact Or.inl ⟨1, one_mem _, u, hu, by simpa using hδu⟩
    · right
      refine ⟨ModularGroup.T ^ (t : ℤ), (gammaBd_iff _).2 ⟨t, Or.inl rfl⟩, u, hu, ?_⟩
      rw [hδu, ModularGroup.coe_T_zpow, dbd, dbd, if_pos (by omega), if_pos hp.pos]
      ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  · rintro ⟨⟨u₁, hu₁, u₁', hu₁', h₁⟩, ⟨u₂, hu₂, u₂', hu₂', h₂⟩⟩
    have e₁ := key u₁ u₁' hu₁ hu₁' p le_rfl
    have e₂ := key u₂ u₂' hu₂ hu₂' 0 (Nat.zero_le _)
    rw [← h₁] at e₁; rw [← h₂, e₁] at e₂
    have := congrArg abs e₂
    simp only [abs_mul, unit u₁ hu₁, unit u₁' hu₁', unit u₂ hu₂, unit u₂' hu₂', dbd,
      if_neg (lt_irrefl p), if_pos hp.pos] at this
    simp at this
    have := hp.one_lt; omega

/-- **[C] `prop:bdrycosets`(4) at `ν = 1`:** the right-coset representatives `β_b`, `β_∞` represent
`Γ_∂ \ B_p` (each `δ ∈ B_p` is `u β` for exactly one `β` and some `u ∈ Γ_∂`); they are not the left cosets. -/
theorem bd_rightCosets_nu_one {p : ℕ} (hp : p.Prime) {δ : Mat2 ℤ} (hδ : δ ∈ BN ((p : ℤ) ^ 1)) :
    ∃! b, b ≤ p ∧ ∃ u ∈ GammaBd, δ = (u : Mat2 ℤ) * betaMat p b := by
  obtain ⟨b, ⟨hb, γ, hγ⟩, huniq⟩ := rightCosets_nu_one hp hδ.1
  refine ⟨b, ⟨hb, γ, ?_, hγ⟩, fun b' ⟨hb', u, _, hu⟩ => huniq b' ⟨hb', u, hu⟩⟩
  -- `γ = δ β⁻¹` is upper triangular because `δ` and `β` are
  rw [mem_GammaBd]
  have h10 := congrFun (congrFun hγ 1) 0
  rw [hδ.2, Matrix.mul_apply, Fin.sum_univ_two] at h10
  unfold betaMat at h10
  split_ifs at h10 <;> simp at h10
  · exact h10.symm
  · rcases h10 with h | h
    · exact h
    · exact absurd h (by exact_mod_cast hp.ne_zero)

end GL2.Cosets
