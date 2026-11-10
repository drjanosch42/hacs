/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Check.Bock

/-!
# HeckeZ K7: split, Hom bound and the glue sandwich (L-HZ2, M-HZ3)

`07-lean-plan.md` §5.7; `03-bockstein-tor.md` B12(2), B13, §8.2 (K7); [C] `prop:heckeZglue`(2)(b), (3).

**TB-mat** (`K7Model`, the slot-25 EXPORT `DescentRecord.module(ell, K)` reduced modulo `q = ℓ^e`, `tors_ℓ = ℤ/ℓ^e` cyclic): in the
`ℤ`-module `H` (`= H¹(SL₂(ℤ); Sym^n ℤ²)`) with Hecke operators `X`, an element `g` (the export's torsion cocycle) and elements
`f_0, …, f_{r-1}` (its free cocycles) such that
* `q g = 0`, and `a g ∈ qH ⇒ q ∣ a` (`g` has order `q` in `H/qH`); every `ℓ`-power-torsion element is a multiple of `g` (`gen`);
* `H = ℤ g + Σ ℤ f_i + qH`, and `Σ v_i f_i ∈ ℤ g + qH ⇒ q ∣ v_i` (the `f_i` are a basis of `L/ℓ^e`);
* `T_p g = c_p g`, `T_p f_j ≡ Σ_i B_p[i][j] f_i + β_p[j] g (mod qH)` for `p ∈ P`.

With `S = ℤ g = tors_ℓ`, `𝕋' = ℤ[T_p : p ∈ P]`:
* `k7_hom_eq`, `k7_hom_zero`: an element of `Hom_{𝕋'}(H/S, S)` is determined by its Smith read `y ∈ (ℤ/q)^r`, which satisfies
  `ℓ^{v_k} y_k = 0`; all `v_k = 0` (status `zero`) forces `Hom_{𝕋'} = 0`;
* `k7_finite`, `k7_card_le`: `Hom_{𝕋'}(H/S, S)` is finite of order `≤ ℓ^{Σ v_k}` — **the upper bound** of B13's sandwich;
* `k7_split`: a section `h` gives a `𝕋'`-stable complement of `S` — **the lower bound** `J ∩ 𝕋' = 0` (B12(3));
* `k7_J_zero`, `k7_J_card_le`, `k7_J_split`: the same for the glue `J` of the full commutative algebra `𝕋 = ℤ[X_n]`.
Never `exact` (CB-3 is not a hypothesis, A3-M2).
-/

namespace GL2.HeckeZ.Check

open GL2.HeckeZ.Exec GL2.HeckeZ.Bock

section K7

variable {H : Type*} [AddCommGroup H]

/-- **TB-mat for K7** (see the module docstring). -/
structure K7Model (c : K7Cert) (X : ℕ → Module.End ℤ H) (g : H) (f : ℕ → H) : Prop where
  ord : (c.q : ℤ) • g = 0
  ordq : ∀ (a : ℤ) (y : H), a • g = (c.q : ℤ) • y → (c.q : ℤ) ∣ a
  gen : ∀ (x : H) (j : ℕ), ((c.ell : ℤ) ^ j) • x = 0 → x ∈ Submodule.span ℤ {g}
  span : ∀ x : H, ∃ (a : ℤ) (v : ℕ → ℤ) (y : H), x = a • g + ∑ i ∈ Finset.range c.r, v i • f i + (c.q : ℤ) • y
  indep : ∀ (v : ℕ → ℤ) (a : ℤ) (y : H), ∑ i ∈ Finset.range c.r, v i • f i = a • g + (c.q : ℤ) • y →
    ∀ i < c.r, (c.q : ℤ) ∣ v i
  actg : ∀ k < c.P.length, X (c.P.getD k 0) g = c.c.getD k 0 • g
  actf : ∀ k < c.P.length, ∀ j < c.r, ∃ y : H, X (c.P.getD k 0) (f j) =
    ∑ i ∈ Finset.range c.r, ientry (c.B.getD k []) i j • f i + (c.beta.getD k []).getD j 0 • g + (c.q : ℤ) • y

/-- `{T_p : p ∈ P}`. -/
def Pset (c : K7Cert) (X : ℕ → Module.End ℤ H) : Set (Module.End ℤ H) := X '' {p | p ∈ c.P}

lemma getD_mem_P {c : K7Cert} {k : ℕ} (hk : k < c.P.length) : c.P.getD k 0 ∈ c.P := by
  rw [List.getD_eq_getElem _ _ hk]; exact List.getElem_mem hk

/-- What a passing K7 certificate gives. -/
structure K7Spec (c : K7Cert) : Prop where
  ell2 : 2 ≤ c.ell
  split : c.isSplit = true → ∀ k < c.P.length, ∀ j < c.r, (c.q : ℤ) ∣
    (∑ i ∈ Finset.range c.r, c.h.getD i 0 * ientry (c.B.getD k []) i j) - c.c.getD k 0 * c.h.getD j 0
      - (c.beta.getD k []).getD j 0
  vlen : c.vals.length = c.r
  smith : ∀ i < c.r, ∀ k < c.r, (c.q : ℤ) ∣
    (∑ col ∈ Finset.range (c.r * c.P.length), c.A i col * ientry c.W col k)
      - ientry c.G i k * ((c.ell ^ c.vals.getD k 0 : ℕ) : ℤ)
  inv : ∀ i < c.r, ∀ k < c.r, (c.q : ℤ) ∣
    (∑ l ∈ Finset.range c.r, ientry c.G i l * ientry c.Gi l k) - (if i = k then 1 else 0)

/-- **Soundness of K7** (the checker's claims, read as divisibilities). -/
theorem k7Check_sound {c : K7Cert} (h : k7Check c = true) : K7Spec c := by
  simp only [k7Check, K7Cert.homOK, K7Cert.splitOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq,
    List.all_eq_true, List.mem_range, Bool.or_eq_true, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨⟨⟨⟨h2, _⟩, _⟩, _⟩, _⟩, hsp⟩, ⟨⟨⟨hvl, _⟩, hsm⟩, hinv⟩⟩ := h
  refine ⟨h2, ?_, hvl, ?_, ?_⟩
  · intro hs k hk j hj
    rcases hsp with hsp | hsp
    · rw [hs] at hsp; exact absurd hsp (by decide)
    · have := hsp k hk j hj
      rw [list_sum_range_map] at this
      exact Int.dvd_of_emod_eq_zero this
  · intro i hi k hk
    have := hsm i hi k hk
    rw [K7Cert.AW, list_sum_range_map] at this
    exact Int.dvd_of_emod_eq_zero this
  · intro i hi k hk
    have := hinv i hi k hk
    rw [list_sum_range_map] at this
    exact Int.dvd_of_emod_eq_zero this

variable {c : K7Cert} {X : ℕ → Module.End ℤ H} {g : H} {f : ℕ → H}

/-- The coefficient `h_i` of an element of `Hom(H/S, S)` at `f_i`: `t f_i = h_i g`. -/
noncomputable def coef {t : Module.End ℤ H} (ht : ∀ x, t x ∈ Submodule.span ℤ {g}) (i : ℕ) : ℤ :=
  Classical.choose (Submodule.mem_span_singleton.mp (ht (f i)))

lemma coef_spec {t : Module.End ℤ H} (ht : ∀ x, t x ∈ Submodule.span ℤ {g}) (i : ℕ) :
    coef (f := f) ht i • g = t (f i) :=
  Classical.choose_spec (Submodule.mem_span_singleton.mp (ht (f i)))

lemma map_qsmul_eq_zero (hM : K7Model c X g f) {t : Module.End ℤ H} (ht : ∀ x, t x ∈ Submodule.span ℤ {g}) (y : H) :
    t ((c.q : ℤ) • y) = 0 := by
  obtain ⟨a, ha⟩ := Submodule.mem_span_singleton.mp (ht y)
  rw [map_zsmul, ← ha, smul_comm, hM.ord, smul_zero]

/-- The linear equations of `Hom_{𝕋'}`: `q ∣ Σ_i h_i B_k[i][j] − c_k h_j`. -/
lemma coef_eq (hM : K7Model c X g f) {t : Module.End ℤ H} (ht : t ∈ HomP (Pset c X) (Submodule.span ℤ {g}))
    {k j : ℕ} (hk : k < c.P.length) (hj : j < c.r) :
    (c.q : ℤ) ∣ (∑ i ∈ Finset.range c.r, coef (f := f) ht.1 i * ientry (c.B.getD k []) i j)
      - c.c.getD k 0 * coef (f := f) ht.1 j := by
  obtain ⟨y, hy⟩ := hM.actf k hk j hj
  have hcomm := congrArg (· (f j)) (ht.2.2 _ ⟨c.P.getD k 0, getD_mem_P hk, rfl⟩).eq
  simp only [Module.End.mul_apply] at hcomm
  rw [hy, map_add, map_add, map_qsmul_eq_zero hM ht.1, map_sum, map_zsmul,
    ht.2.1 g (Submodule.mem_span_singleton_self g), smul_zero, add_zero, add_zero, ← coef_spec ht.1 j,
    map_zsmul, hM.actg k hk, smul_smul] at hcomm
  simp_rw [map_zsmul, ← coef_spec ht.1, smul_smul] at hcomm
  rw [← Finset.sum_smul] at hcomm
  refine hM.ordq _ 0 ?_
  rw [smul_zero, sub_smul, sub_eq_zero, Finset.sum_congr rfl fun i _ => mul_comm (coef (f := f) ht.1 i) _, hcomm,
    mul_comm]

/-- An element of `Hom(H/S, S)` vanishing on the `f_i` (modulo `q`) is zero. -/
lemma eq_of_coef (hM : K7Model c X g f) {t t' : Module.End ℤ H} (ht : ∀ x, t x ∈ Submodule.span ℤ {g})
    (ht' : ∀ x, t' x ∈ Submodule.span ℤ {g}) (hg : t g = 0) (hg' : t' g = 0)
    (hc : ∀ i < c.r, (c.q : ℤ) ∣ coef (f := f) ht i - coef (f := f) ht' i) : t = t' := by
  ext x
  obtain ⟨a, v, y, rfl⟩ := hM.span x
  simp only [map_add, map_sum, map_zsmul, hg, hg', map_qsmul_eq_zero hM ht, map_qsmul_eq_zero hM ht']
  congr 2
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [← coef_spec ht i, ← coef_spec ht' i]
  obtain ⟨m, hm⟩ := hc i (Finset.mem_range.mp hi)
  have : coef (f := f) ht i = coef (f := f) ht' i + c.q * m := by omega
  rw [this, add_smul, mul_comm, mul_smul, hM.ord, smul_zero, add_zero]

/-- The Smith read `y_k = Σ_i h_i G_ik ∈ ℤ/q`. -/
noncomputable def smithRead (c : K7Cert) {t : Module.End ℤ H} (ht : ∀ x, t x ∈ Submodule.span ℤ {g}) (k : Fin c.r) : ZMod c.q :=
  ∑ i ∈ Finset.range c.r, (coef (f := f) ht i : ZMod c.q) * (ientry c.G i k : ZMod c.q)

lemma cast_dvd {a b : ℤ} (h : (c.q : ℤ) ∣ a - b) : (a : ZMod c.q) = b := by
  rw [← sub_eq_zero, ← Int.cast_sub, ZMod.intCast_zmod_eq_zero_iff_dvd]; exact h

/-- **The read is killed by `ℓ^{v_k}`.** -/
lemma smithRead_ann (hs : K7Spec c) (hM : K7Model c X g f) {t : Module.End ℤ H}
    (ht : t ∈ HomP (Pset c X) (Submodule.span ℤ {g})) (k : Fin c.r) :
    ((c.ell ^ c.vals.getD k 0 : ℕ) : ZMod c.q) * smithRead c (f := f) ht.1 k = 0 := by
  set h : ℕ → ZMod c.q := fun i => (coef (f := f) ht.1 i : ZMod c.q)
  -- the equations, for every column of `A`
  have hA : ∀ col < c.r * c.P.length, ∑ i ∈ Finset.range c.r, h i * (c.A i col : ZMod c.q) = 0 := by
    intro col hcol
    have hr : 0 < c.r := Nat.pos_of_ne_zero fun h0 => by simp [h0] at hcol
    have hk : col / c.r < c.P.length := (Nat.div_lt_iff_lt_mul hr).mpr (by rw [mul_comm]; exact hcol)
    have hj : col % c.r < c.r := Nat.mod_lt _ hr
    have he := cast_dvd (coef_eq hM ht hk hj)
    push_cast at he
    simp only [K7Cert.A, Int.cast_sub, mul_sub, Finset.sum_sub_distrib, Int.cast_ite, Int.cast_zero, mul_ite,
      mul_zero, Finset.sum_ite_eq', Finset.mem_range, hj, if_true]
    rw [sub_eq_zero]
    convert he using 1 <;> first | rfl | ring
  have hsm : ∀ i < c.r, ((c.ell ^ c.vals.getD k 0 : ℕ) : ZMod c.q) * (ientry c.G i k : ZMod c.q) =
      ∑ col ∈ Finset.range (c.r * c.P.length), (c.A i col : ZMod c.q) * (ientry c.W col k : ZMod c.q) := by
    intro i hi
    have := cast_dvd (hs.smith i hi k k.2)
    push_cast at this ⊢
    rw [this]; ring
  unfold smithRead
  rw [Finset.mul_sum]
  calc ∑ i ∈ Finset.range c.r, ((c.ell ^ c.vals.getD k 0 : ℕ) : ZMod c.q) * (h i * (ientry c.G i k : ZMod c.q))
      = ∑ i ∈ Finset.range c.r, h i * ∑ col ∈ Finset.range (c.r * c.P.length),
          (c.A i col : ZMod c.q) * (ientry c.W col k : ZMod c.q) :=
        Finset.sum_congr rfl fun i hi => by rw [← hsm i (Finset.mem_range.mp hi)]; ring
    _ = ∑ col ∈ Finset.range (c.r * c.P.length),
          (∑ i ∈ Finset.range c.r, h i * (c.A i col : ZMod c.q)) * (ientry c.W col k : ZMod c.q) := by
        simp_rw [Finset.mul_sum, Finset.sum_mul]; rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring
    _ = 0 := Finset.sum_eq_zero fun col hcol => by rw [hA col (Finset.mem_range.mp hcol), zero_mul]

/-- **The read determines the coefficients**: `h_j = Σ_k y_k G'_kj`. -/
lemma coef_of_smithRead (hs : K7Spec c) {t : Module.End ℤ H} (ht : ∀ x, t x ∈ Submodule.span ℤ {g})
    {j : ℕ} (hj : j < c.r) :
    (coef (f := f) ht j : ZMod c.q) = ∑ k : Fin c.r, smithRead c (f := f) ht k * (ientry c.Gi k j : ZMod c.q) := by
  have hinv : ∀ i < c.r, ∑ k ∈ Finset.range c.r, (ientry c.G i k : ZMod c.q) * (ientry c.Gi k j : ZMod c.q) =
      if i = j then 1 else 0 := by
    intro i hi
    have := cast_dvd (hs.inv i hi j hj)
    push_cast at this
    exact this
  unfold smithRead
  rw [Fin.sum_univ_eq_sum_range (fun k => (∑ i ∈ Finset.range c.r, (coef (f := f) ht i : ZMod c.q) *
    (ientry c.G i k : ZMod c.q)) * (ientry c.Gi k j : ZMod c.q))]
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  simp_rw [mul_assoc, ← Finset.mul_sum]
  rw [Finset.sum_congr rfl fun i hi => by rw [hinv i (Finset.mem_range.mp hi)]]
  simp [Finset.mem_range, hj]

/-- **Injectivity of the Smith read** on `Hom_{𝕋'}(H/S, S)`. -/
theorem k7_hom_eq (hs : K7Spec c) (hM : K7Model c X g f) {t t' : Module.End ℤ H}
    (ht : t ∈ HomP (Pset c X) (Submodule.span ℤ {g})) (ht' : t' ∈ HomP (Pset c X) (Submodule.span ℤ {g}))
    (he : smithRead c (f := f) ht.1 = smithRead c (f := f) ht'.1) : t = t' := by
  refine eq_of_coef hM ht.1 ht'.1 (ht.2.1 g (Submodule.mem_span_singleton_self g))
    (ht'.2.1 g (Submodule.mem_span_singleton_self g)) fun i hi => ?_
  have h1 := coef_of_smithRead (f := f) hs ht.1 hi
  have h2 := coef_of_smithRead (f := f) hs ht'.1 hi
  rw [he] at h1
  rw [← ZMod.intCast_zmod_eq_zero_iff_dvd, Int.cast_sub, h1, h2, sub_self]

/-- **Status `zero`**: all `v_k = 0` forces `Hom_{𝕋'}(H/S, S) = 0`. -/
theorem k7_hom_zero (hs : K7Spec c) (hM : K7Model c X g f) (hz : ∀ k < c.r, c.vals.getD k 0 = 0) :
    ∀ t ∈ HomP (Pset c X) (Submodule.span ℤ {g}), t = 0 := by
  intro t ht
  have h0 : (0 : Module.End ℤ H) ∈ HomP (Pset c X) (Submodule.span ℤ {g}) :=
    ⟨fun _ => Submodule.zero_mem _, fun _ _ => rfl, fun _ _ => Commute.zero_left _⟩
  refine k7_hom_eq hs hM ht h0 (funext fun k => ?_)
  have a1 := smithRead_ann hs hM ht k
  have a2 := smithRead_ann hs hM h0 k
  rw [hz k k.2, pow_zero, Nat.cast_one, one_mul] at a1 a2
  rw [a1, a2]

/-- `|{x ∈ ℤ/ℓ^e : ℓ^v x = 0}| ≤ ℓ^v` for `v ≤ e`. -/
theorem card_ann_le {ℓ e v : ℕ} (hℓ : 2 ≤ ℓ) (hv : v ≤ e) :
    Nat.card {x : ZMod (ℓ ^ e) // ((ℓ ^ v : ℕ) : ZMod (ℓ ^ e)) * x = 0} ≤ ℓ ^ v := by
  have : NeZero (ℓ ^ e) := ⟨pow_ne_zero _ (by omega)⟩
  have hpos : 0 < ℓ ^ (e - v) := pow_pos (by omega) _
  have hlt : ∀ x : ZMod (ℓ ^ e), x.val / ℓ ^ (e - v) < ℓ ^ v := fun x => by
    rw [Nat.div_lt_iff_lt_mul hpos, ← pow_add, Nat.add_sub_cancel' hv]; exact ZMod.val_lt x
  let F : {x : ZMod (ℓ ^ e) // ((ℓ ^ v : ℕ) : ZMod (ℓ ^ e)) * x = 0} → Fin (ℓ ^ v) :=
    fun x => ⟨x.1.val / ℓ ^ (e - v), hlt x.1⟩
  have hdvd : ∀ x : {x : ZMod (ℓ ^ e) // ((ℓ ^ v : ℕ) : ZMod (ℓ ^ e)) * x = 0}, ℓ ^ (e - v) ∣ x.1.val := by
    rintro ⟨x, hx⟩
    have h1 : ((ℓ ^ v * x.val : ℕ) : ZMod (ℓ ^ e)) = 0 := by rw [Nat.cast_mul, ZMod.natCast_zmod_val]; exact hx
    rw [ZMod.natCast_eq_zero_iff] at h1
    have h2 : ℓ ^ v * ℓ ^ (e - v) ∣ ℓ ^ v * x.val := by rw [← pow_add, Nat.add_sub_cancel' hv]; exact h1
    exact Nat.dvd_of_mul_dvd_mul_left (pow_pos (by omega) _) h2
  have hinj : Function.Injective F := by
    intro x y hxy
    simp only [F, Fin.mk.injEq] at hxy
    apply Subtype.ext
    apply ZMod.val_injective
    rw [← Nat.mul_div_cancel' (hdvd x), ← Nat.mul_div_cancel' (hdvd y), hxy]
  calc Nat.card _ ≤ Nat.card (Fin (ℓ ^ v)) := Nat.card_le_card_of_injective F hinj
    _ = ℓ ^ v := Nat.card_fin _

/-- **`Hom_{𝕋'}(H/S, S)` is finite** (it embeds in `(ℤ/q)^r`). -/
theorem k7_finite (hs : K7Spec c) (hM : K7Model c X g f) : (HomP (Pset c X) (Submodule.span ℤ {g})).Finite := by
  have : NeZero c.q := ⟨pow_ne_zero _ (by have := hs.ell2; omega)⟩
  refine Set.finite_coe_iff.mp (Finite.of_injective (fun t : HomP (Pset c X) (Submodule.span ℤ {g}) =>
    smithRead c (f := f) t.2.1) fun t t' he => Subtype.ext (k7_hom_eq hs hM t.2 t'.2 he))

/-- **The upper bound**: `|Hom_{𝕋'}(H/S, S)| ≤ ℓ^{Σ_k v_k}`. -/
theorem k7_card_le (hs : K7Spec c) (hM : K7Model c X g f) (hv : ∀ k < c.r, c.vals.getD k 0 ≤ c.e) :
    Nat.card (HomP (Pset c X) (Submodule.span ℤ {g})) ≤ c.ell ^ (∑ k : Fin c.r, c.vals.getD k 0) := by
  have : NeZero c.q := ⟨pow_ne_zero _ (by have := hs.ell2; omega)⟩
  let Φ : HomP (Pset c X) (Submodule.span ℤ {g}) →
      ((k : Fin c.r) → {x : ZMod c.q // ((c.ell ^ c.vals.getD k 0 : ℕ) : ZMod c.q) * x = 0}) :=
    fun t k => ⟨smithRead c (f := f) t.2.1 k, smithRead_ann hs hM t.2 k⟩
  have hΦ : Function.Injective Φ := by
    intro t t' he
    refine Subtype.ext (k7_hom_eq hs hM t.2 t'.2 (funext fun k => ?_))
    have := congrFun he k
    simp only [Φ, Subtype.mk.injEq] at this
    exact this
  calc Nat.card _ ≤ Nat.card ((k : Fin c.r) → {x : ZMod c.q // ((c.ell ^ c.vals.getD k 0 : ℕ) : ZMod c.q) * x = 0}) :=
        Nat.card_le_card_of_injective Φ hΦ
    _ = ∏ k : Fin c.r, Nat.card {x : ZMod c.q // ((c.ell ^ c.vals.getD k 0 : ℕ) : ZMod c.q) * x = 0} :=
        Nat.card_pi
    _ ≤ ∏ k : Fin c.r, c.ell ^ c.vals.getD k 0 :=
        Finset.prod_le_prod' fun k _ => card_ann_le hs.ell2 (hv k k.2)
    _ = c.ell ^ (∑ k : Fin c.r, c.vals.getD k 0) := Finset.prod_pow_eq_pow_sum _ _ _

/-- The complement `C = {Σ v_i (f_i + h_i g) + q y}` of a split certificate. -/
def splitC (c : K7Cert) (g : H) (f : ℕ → H) : Submodule ℤ H where
  carrier := {x | ∃ (v : ℕ → ℤ) (y : H), x = ∑ i ∈ Finset.range c.r, v i • (f i + c.h.getD i 0 • g) + (c.q : ℤ) • y}
  add_mem' := by
    rintro _ _ ⟨v, y, rfl⟩ ⟨w, z, rfl⟩
    exact ⟨fun i => v i + w i, y + z, by simp only [add_smul, Finset.sum_add_distrib, smul_add]; abel⟩
  zero_mem' := ⟨fun _ => 0, 0, by simp⟩
  smul_mem' := by
    rintro a _ ⟨v, y, rfl⟩
    refine ⟨fun i => a * v i, a • y, ?_⟩
    rw [smul_add, Finset.smul_sum, smul_comm a (c.q : ℤ) y]
    congr 1
    exact Finset.sum_congr rfl fun i _ => by rw [smul_smul]

lemma sum_expand (v : ℕ → ℤ) (hv : ℕ → ℤ) :
    ∑ i ∈ Finset.range c.r, v i • (f i + hv i • g) =
      ∑ i ∈ Finset.range c.r, v i • f i + (∑ i ∈ Finset.range c.r, v i * hv i) • g := by
  rw [Finset.sum_smul, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by rw [smul_add, smul_smul]

/-- **The lower bound (B12(2), (3))**: a split certificate gives a `𝕋'`-stable complement of `S = ℤ g`. -/
theorem k7_split (hs : K7Spec c) (hM : K7Model c X g f) (hsp : c.isSplit = true) :
    IsCompl (Submodule.span ℤ {g}) (splitC c g f) ∧
      ∀ t ∈ Algebra.adjoin ℤ (Pset c X), ∀ x ∈ splitC c g f, t x ∈ splitC c g f := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · rw [Submodule.disjoint_def]
    intro x hx hC
    obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.mp hx
    obtain ⟨v, y, hxy⟩ := hC
    rw [sum_expand] at hxy
    have hv := hM.indep v (a - ∑ i ∈ Finset.range c.r, v i * c.h.getD i 0) (-y) (by
      rw [sub_smul, smul_neg, hxy]; abel)
    choose w hw using fun i : Fin c.r => hv i i.2
    have hsum : ∑ i ∈ Finset.range c.r, v i • f i = (c.q : ℤ) • ∑ i : Fin c.r, w i • f i := by
      rw [Finset.smul_sum, ← Fin.sum_univ_eq_sum_range (fun i => v i • f i)]
      exact Finset.sum_congr rfl fun i _ => by rw [hw i, mul_smul]
    have h1 := hM.ordq (a - ∑ i ∈ Finset.range c.r, v i * c.h.getD i 0)
      (∑ i : Fin c.r, w i • f i + y) (by rw [sub_smul, hxy, hsum, smul_add]; abel)
    have h2 : (c.q : ℤ) ∣ ∑ i ∈ Finset.range c.r, v i * c.h.getD i 0 :=
      Finset.dvd_sum fun i hi => Dvd.dvd.mul_right (hv i (Finset.mem_range.mp hi)) _
    have h3 : (c.q : ℤ) ∣ a := by simpa using dvd_add h1 h2
    obtain ⟨m, hm⟩ := h3
    rw [hm, mul_comm, mul_smul, hM.ord, smul_zero]
  · rw [codisjoint_iff, eq_top_iff]
    intro x _
    obtain ⟨a, v, y, rfl⟩ := hM.span x
    have hC : ∑ i ∈ Finset.range c.r, v i • (f i + c.h.getD i 0 • g) + (c.q : ℤ) • y ∈ splitC c g f := ⟨v, y, rfl⟩
    have hS : (a - ∑ i ∈ Finset.range c.r, v i * c.h.getD i 0) • g ∈ Submodule.span ℤ {g} :=
      Submodule.smul_mem _ _ (Submodule.mem_span_singleton_self g)
    have := Submodule.add_mem_sup hS hC
    convert this using 1
    rw [sum_expand, sub_smul]; abel
  · -- stability under each generator, then under the algebra
    have hgen : ∀ k < c.P.length, ∀ x ∈ splitC c g f, X (c.P.getD k 0) x ∈ splitC c g f := by
      intro k hk x hx
      obtain ⟨v, y, rfl⟩ := hx
      have hj : ∀ j < c.r, X (c.P.getD k 0) (f j + c.h.getD j 0 • g) ∈ splitC c g f := by
        intro j hj
        obtain ⟨y', hy'⟩ := hM.actf k hk j hj
        obtain ⟨m, hm⟩ := hs.split hsp k hk j hj
        refine ⟨fun i => ientry (c.B.getD k []) i j, y', ?_⟩
        rw [map_add, map_zsmul, hM.actg k hk, hy', sum_expand, smul_smul]
        have hco : ((c.beta.getD k []).getD j 0 + c.h.getD j 0 * c.c.getD k 0) • g =
            (∑ i ∈ Finset.range c.r, ientry (c.B.getD k []) i j * c.h.getD i 0) • g := by
          rw [← sub_eq_zero, ← sub_smul]
          have : ((c.beta.getD k []).getD j 0 + c.h.getD j 0 * c.c.getD k 0) -
              ∑ i ∈ Finset.range c.r, ientry (c.B.getD k []) i j * c.h.getD i 0 = -(c.q * m) := by
            rw [← hm, Finset.sum_congr rfl fun i _ => mul_comm _ _]; ring
          rw [this, neg_smul, mul_comm, mul_smul, hM.ord, smul_zero, neg_zero]
        rw [← hco, add_smul]; abel
      rw [map_add, map_sum]
      refine Submodule.add_mem _ (Submodule.sum_mem _ fun j hj' => ?_) ?_
      · rw [map_zsmul]; exact Submodule.smul_mem _ _ (hj j (Finset.mem_range.mp hj'))
      · rw [map_zsmul]; exact ⟨fun _ => 0, X (c.P.getD k 0) y, by simp⟩
    intro t ht
    induction ht using Algebra.adjoin_induction with
    | mem y hy =>
      obtain ⟨p, hp, rfl⟩ := hy
      obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hp
      intro x hx
      have := hgen k hk x hx
      rwa [List.getD_eq_getElem _ _ hk] at this
    | algebraMap r => intro x hx; simpa [Algebra.algebraMap_eq_smul_one] using Submodule.smul_mem _ r hx
    | add y z _ _ hy hz => intro x hx; exact Submodule.add_mem _ (hy x hx) (hz x hx)
    | mul y z _ _ hy hz => intro x hx; exact hy _ (hz x hx)

/-! ### The glue `J` of the full algebra -/

variable (hX : ∀ i j, Commute (X i) (X j))
include hX

lemma comm_Pset {t : Module.End ℤ H} (ht : t ∈ Algebra.adjoin ℤ (Set.range X)) :
    ∀ x ∈ Pset c X, Commute t x := by
  rintro _ ⟨p, _, rfl⟩
  exact commute_of_mem_adjoin hX ht p

/-- **K7, status `zero`**: `J_ℓ = 0` for `𝕋 = ℤ[X_n]` (relative to `tors_ℓ = ℤ g`). -/
theorem k7_J_zero (hs : K7Spec c) (hM : K7Model c X g f) (hz : ∀ k < c.r, c.vals.getD k 0 = 0) :
    ∀ t ∈ Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g}), t = 0 :=
  Jset_eq_zero_of_HomP (fun t ht => comm_Pset hX ht) (k7_hom_zero hs hM hz)

/-- **K7, the upper bound** for the glue of `𝕋 = ℤ[X_n]`: `J` is finite of order `≤ ℓ^{Σ v_k}`. -/
theorem k7_J_card_le (hs : K7Spec c) (hM : K7Model c X g f) (hv : ∀ k < c.r, c.vals.getD k 0 ≤ c.e) :
    (Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g})).Finite ∧
    Nat.card (Jset (Algebra.adjoin ℤ (Set.range X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g})) ≤
      c.ell ^ (∑ k : Fin c.r, c.vals.getD k 0) := by
  have hsub := Jset_subset_HomP (S := Submodule.span ℤ {g}) (fun t ht => comm_Pset (c := c) hX ht)
  exact ⟨(k7_finite hs hM).subset hsub, (Nat.card_mono (k7_finite hs hM) hsub).trans (k7_card_le hs hM hv)⟩

omit hX in
/-- **K7, the lower bound**: a split certificate gives `J ∩ 𝕋' = 0`. -/
theorem k7_J_split (hs : K7Spec c) (hM : K7Model c X g f) (hsp : c.isSplit = true) :
    ∀ t ∈ Jset (Algebra.adjoin ℤ (Pset c X) : Set (Module.End ℤ H)) (Submodule.span ℤ {g}), t = 0 :=
  Jset_eq_zero_of_compl (k7_split hs hM hsp).1 (k7_split hs hM hsp).2

end K7

end GL2.HeckeZ.Check
