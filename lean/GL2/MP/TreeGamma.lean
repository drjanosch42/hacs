/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.TreeRetract

/-!
# LR-Y phase 2a (slot 22): the tree complex as a complex of `ℤ[Γ]`-modules

[GT] `gt_part_MP.tex`, rows `def:mp-ys-action`, `thm:mp-tree-equivariant`, `thm:mp-model7-tree-orbits`, backing [C]
`eq:treeses` (1) (the terms as `ℤ[Γ]`-modules; plan of record `LV0_audit.md` §3.2, LR-2 input).

* `YS.mulAction`, `TS.mulAction`: `Γ` acts on the ordered simplices of `Y` and of the full subcomplex `𝒯` vertex-wise
  (`ord` is `Γ`-invariant, a translate of a maximal simplex is one).
* **Chain-level equivariance**: the boundaries of `Y` and of `𝒯` (`yBdS_smul`, `tBdS_smul`), the inclusion
  `incl : k[𝒯_•] → k[Y_•]` (`tIncl_smul`) and, for an equivariant vertex map, the retraction `r` (`tRetr_smul`) commute with
  `mapDomain (g • ·)`, i.e. are maps of permutation representations in every degree `≥ 1` (degree `0` is the trivial `k`).
* **The terms** (`M₇`, `m = 3`): `𝒯_1 ≃ Γ/⟨−I⟩` and `𝒯_0 ≃ Γ/⟨U⟩ ⊕ Γ/⟨S⟩` as `Γ`-sets (`model7EdgeEquiv`,
  `model7VtxEquiv`), hence `k[𝒯_1] ≅ k[Γ/⟨−I⟩]` and `k[𝒯_0] ≅ k[Γ/⟨U⟩ ⊔ Γ/⟨S⟩] = k[Γ/⟨U⟩] ⊕ k[Γ/⟨S⟩]` in `Rep k Γ`
  (`model7TreeRep1`, `model7TreeRep0`), with `Γ_i = ⟨S⟩` (order 4), `Γ_ρ = ⟨U⟩` (order 6), `Γ_{[ρ,i]} = {±I}`.
-/

namespace GL2.MP

open Finsupp

namespace Cover

variable {G : Type} [Group G] {C : Cover.{0} G} (H : C.Hyp)

lemma inS_smul_left (g : G) {o : G × C.T} {v : C.P} (h : InS o v) : InS (g * o.1, o.2) (g • v) := by
  unfold InS at *
  rwa [mul_inv_rev, mul_smul, inv_smul_smul]

/-- **`Γ` acts on the ordered simplices of `Y`**, vertex-wise. -/
instance YS.mulAction (s : ℕ) : MulAction G (YS H s) where
  smul g x := ⟨fun j => g • x.1 j, by
      have e : (H.ord ∘ fun j => g • x.1 j) = H.ord ∘ x.1 := funext fun j => H.ord_smul g _
      rw [e]; exact x.2.1,
    x.2.2.elim fun o ho => ⟨(g * o.1, o.2), fun j => inS_smul_left g (ho j)⟩⟩
  one_smul x := Subtype.ext (funext fun j => one_smul G (x.1 j))
  mul_smul g h x := Subtype.ext (funext fun j => mul_smul g h (x.1 j))

variable {H}

lemma YS.smul_val {s : ℕ} (g : G) (x : YS H s) (j : Fin (s + 1)) : (g • x).1 j = g • x.1 j := rfl

lemma YS.del_smul {s : ℕ} (j : Fin (s + 2)) (g : G) (x : YS H (s + 1)) : YS.del j (g • x) = g • YS.del j x := rfl

variable (H) (m : ℕ)

/-- **`Γ` acts on the ordered simplices of the full subcomplex on `{ord ≥ m}`.** -/
instance TS.mulAction (s : ℕ) : MulAction G (TS H m s) where
  smul g x := ⟨g • x.1, by show m ≤ H.ord (g • x.1.1 0); rw [H.ord_smul]; exact x.2⟩
  one_smul x := Subtype.ext (one_smul G x.1)
  mul_smul g h x := Subtype.ext (mul_smul g h x.1)

variable {H m}

lemma TS.smul_val {s : ℕ} (g : G) (x : TS H m s) : (g • x).1 = g • x.1 := rfl

lemma TS.del_smul {s : ℕ} (j : Fin (s + 2)) (g : G) (x : TS H m (s + 1)) : TS.del j (g • x) = g • TS.del j x := rfl

variable (k : Type) [CommRing k]

/-- A map given on bases that commutes with `g • ·` on basis elements commutes with `mapDomain (g • ·)`. -/
lemma lc_mapDomain_smul {α β : Type} [MulAction G α] [MulAction G β] (φ : α → (β →₀ k)) (g : G)
    (hφ : ∀ a, φ (g • a) = mapDomain (g • ·) (φ a)) (x : α →₀ k) :
    Finsupp.linearCombination k φ (mapDomain (g • ·) x) = mapDomain (g • ·) (Finsupp.linearCombination k φ x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [mapDomain_add, map_add, hx, hy, map_add, mapDomain_add]
  | single a c => rw [mapDomain_single, linearCombination_single, linearCombination_single, hφ, mapDomain_smul]

/-! ### Chain-level equivariance -/

/-- **The boundary of `Y` is equivariant.** -/
lemma yBdS_smul {s : ℕ} (g : G) (x : YS H (s + 1)) : yBdS H k (g • x) = mapDomain (g • ·) (yBdS H k x) := by
  rw [yBdS, yBdS, altSum_comp]; rfl

/-- **The boundary of the tree is equivariant.** -/
lemma tBdS_smul {s : ℕ} (g : G) (x : TS H m (s + 1)) :
    tBdS H m k (g • x) = mapDomain (g • ·) (tBdS H m k x) := by
  rw [tBdS, tBdS, altSum_comp]; rfl

/-- **The tree differential is equivariant** (degrees `≥ 1`). -/
lemma treeCx_d_smul (s : ℕ) (g : G) (x : TS H m (s + 1) →₀ k) :
    (treeCx H m k).d (s + 1) (mapDomain (fun y : TS H m (s + 1) => g • y) x) =
      mapDomain (fun y : TS H m s => g • y) ((treeCx H m k).d (s + 1) x) :=
  lc_mapDomain_smul k (tBdS H m k) g (tBdS_smul k g) x

/-- **The inclusion `k[𝒯_•] → k[Y_•]` is equivariant** (degrees `≥ 1`). -/
lemma tIncl_smul (s : ℕ) (g : G) (x : TS H m s →₀ k) :
    tIncl H m k (s + 1) (mapDomain (fun y : TS H m s => g • y) x) =
      mapDomain (fun y : YS H s => g • y) (tIncl H m k (s + 1) x) := by
  show mapDomain Subtype.val (mapDomain (g • ·) x) = mapDomain (g • ·) (mapDomain Subtype.val x)
  rw [← mapDomain_comp, ← mapDomain_comp]; rfl

lemma isE_smul (g : G) {u w : C.P} (h : IsE (H := H) u w) : IsE (H := H) (g • u) (g • w) := by
  obtain ⟨h1, o, hu, hw⟩ := isE_iff.1 h
  exact isE_iff.2 ⟨by rwa [H.ord_smul, H.ord_smul], (g * o.1, o.2), inS_smul_left g hu, inS_smul_left g hw⟩

lemma isE_smul_iff (g : G) {u w : C.P} : IsE (H := H) (g • u) (g • w) ↔ IsE (H := H) u w :=
  ⟨fun h => by simpa using isE_smul g⁻¹ h, isE_smul g⟩

/-- **The oriented tree edge is equivariant.** -/
lemma eCh_smul (g : G) (u w : C.P) : eCh H m k (g • u) (g • w) = mapDomain (g • ·) (eCh H m k u w) := by
  by_cases h : IsE (H := H) u w ∧ m ≤ H.ord u
  · have h' : IsE (H := H) (g • u) (g • w) ∧ m ≤ H.ord (g • u) := ⟨(isE_smul_iff g).2 h.1, by rw [H.ord_smul]; exact h.2⟩
    rw [eCh, dif_pos h', eCh, dif_pos h, mapDomain_single]
    congr 1
    exact TS.ext' (funext fun j => by fin_cases j <;> rfl)
  · have h' : ¬ (IsE (H := H) (g • u) (g • w) ∧ m ≤ H.ord (g • u)) := by rwa [isE_smul_iff, H.ord_smul]
    by_cases e : IsE (H := H) w u ∧ m ≤ H.ord w
    · have e' : IsE (H := H) (g • w) (g • u) ∧ m ≤ H.ord (g • w) := ⟨(isE_smul_iff g).2 e.1, by rw [H.ord_smul]; exact e.2⟩
      have hn : ∀ x : TS H m 1 →₀ k, mapDomain (fun y : TS H m 1 => g • y) (-x) = -mapDomain (g • ·) x := fun x => by
        rw [eq_neg_iff_add_eq_zero, ← mapDomain_add, neg_add_cancel, mapDomain_zero]
      rw [eCh, dif_neg h', dif_pos e', eCh, dif_neg h, dif_pos e, hn, mapDomain_single]
      congr 2
      exact TS.ext' (funext fun j => by fin_cases j <;> rfl)
    · have e' : ¬ (IsE (H := H) (g • w) (g • u) ∧ m ≤ H.ord (g • w)) := by rwa [isE_smul_iff, H.ord_smul]
      rw [eCh, dif_neg h', dif_neg e', eCh, dif_neg h, dif_neg e, mapDomain_zero]

variable {k}

/-- An **equivariant** retraction datum: the vertex map commutes with `Γ`. -/
def TreeRetr.Equivariant (R : TreeRetr H m) : Prop := ∀ (g : G) (v : C.P), R.f (g • v) = g • R.f v

variable (R : TreeRetr H m) (hR : R.Equivariant)
include hR

lemma fV_smul (g : G) (v : C.P) : fV R (g • v) = g • fV R v :=
  TS.ext' (funext fun j => by fin_cases j; exact hR g v)

variable (k)

/-- **`r` is equivariant on bases** (degrees `≥ 1`: vertices, edges, and `0` above). -/
lemma rB_smul : ∀ (s : ℕ) (g : G) (b : YS H s),
    rB k R (s + 1) (g • b : YS H s) = mapDomain (fun y : TS H m s => g • y) (rB k R (s + 1) b)
  | 0, g, b => by
    show single (fV R (g • b.1 0)) 1 = mapDomain (g • ·) (single (fV R (b.1 0)) 1)
    rw [mapDomain_single, fV_smul R hR]
  | 1, g, b => by
    show eCh H m k (R.f (g • b.1 0)) (R.f (g • b.1 1)) = mapDomain (g • ·) (eCh H m k (R.f (b.1 0)) (R.f (b.1 1)))
    rw [hR, hR, eCh_smul]
  | _ + 2, _, _ => (mapDomain_zero).symm

/-- **The retraction `r : k[Y_•] → k[𝒯_•]` is equivariant** (degrees `≥ 1`). -/
theorem tRetr_smul (s : ℕ) (g : G) (x : YS H s →₀ k) :
    tRetr k R (s + 1) (mapDomain (fun y : YS H s => g • y) x) =
      mapDomain (fun y : TS H m s => g • y) (tRetr k R (s + 1) x) :=
  lc_mapDomain_smul (α := YS H s) (β := TS H m s) k (rB k R (s + 1)) g (rB_smul k R hR s g) x

end Cover

/-! ### The decided-descriptor retraction is equivariant -/

namespace MPDatum

open GL2.TauChk
open scoped MatrixGroups

/-- **The orbit-map retraction of a decided descriptor is equivariant** (`fP_smul`). -/
theorem treeRetr_equivariant {D : MPDatum} (hW : D.WF) {m : ℕ} {τ φ : ℕ → ℕ} (hT : TreeChk D m τ φ)
    (hio : D.ioB = true) (hdim : D.dimB = true) : (treeRetr hW hT hio hdim).Equivariant := fun g v => fP_smul hW hT g v

end MPDatum

/-! ### `M₇`: the terms of the tree complex as permutation modules -/

open GL2.TauChk Cover
open scoped MatrixGroups

/-- `a^n` in `M22` (left-associated products). -/
def mpow (a : M22) : ℕ → M22
  | 0 => ⟨1, 0, 0, 1⟩
  | n + 1 => mpow a n * a

lemma toSL_mpow {a : M22} (ha : a.det = 1) : ∀ n, (mpow a n).det = 1 ∧ M22.toSL (mpow a n) = M22.toSL a ^ n
  | 0 => ⟨show (⟨1, 0, 0, 1⟩ : M22).det = 1 by decide, M22.toSL_one⟩
  | n + 1 => by
    obtain ⟨h1, h2⟩ := toSL_mpow ha n
    exact ⟨by rw [mpow, M22.det_mul, h1, ha]; norm_num, by rw [mpow, M22.toSL_mul h1 ha, h2, pow_succ]⟩

/-- `S = [[0, −1], [1, 0]]` (order 4, `S² = −I`). -/
def mS : M22 := ⟨0, -1, 1, 0⟩

/-- `U = [[0, −1], [1, 1]]` (order 6, `U³ = −I`). -/
def mU : M22 := ⟨0, -1, 1, 1⟩

/-- `S` in `SL₂(ℤ)`. -/
noncomputable def sS : SL(2, ℤ) := M22.toSL mS

/-- `U` in `SL₂(ℤ)`. -/
noncomputable def sU : SL(2, ℤ) := M22.toSL mU

/-- **The stored stabilisers of `i` (orbit 4) and `ρ` (orbit 3) are the powers of `S` and `U`**, `U` fixes `ρ`, and
the powers are distinct (`decide +kernel`). -/
theorem model7_stab_pows :
    (∀ a ∈ model7MP.st 4, ∃ j : Fin 4, a = mpow mS j) ∧ (∀ a ∈ model7MP.st 3, ∃ j : Fin 6, a = mpow mU j) ∧
    model7MP.maps mU 4 4 = true ∧ model7MP.maps mS 6 6 = true ∧ mS.det = 1 ∧ mU.det = 1 ∧
    mpow mS 2 = ⟨-1, 0, 0, -1⟩ ∧ mpow mU 3 = ⟨-1, 0, 0, -1⟩ ∧ mpow mS 4 = ⟨1, 0, 0, 1⟩ ∧ mpow mU 6 = ⟨1, 0, 0, 1⟩ ∧
    (∀ a b : Fin 4, mpow mS a = mpow mS b → a = b) ∧ (∀ a b : Fin 6, mpow mU a = mpow mU b → a = b) := by
  decide +kernel

lemma mS_det : mS.det = 1 := by decide
lemma mU_det : mU.det = 1 := by decide

lemma sS_pow (n : ℕ) : M22.toSL (mpow mS n) = sS ^ n := (toSL_mpow mS_det n).2
lemma sU_pow (n : ℕ) : M22.toSL (mpow mU n) = sU ^ n := (toSL_mpow mU_det n).2

/-- `S² = −I`, `U³ = −I`, `S⁴ = U⁶ = 1`. -/
theorem sS_sq : sS ^ 2 = -1 := by
  rw [← sS_pow, (model7_stab_pows).2.2.2.2.2.2.1, M22.toSL_negOne]
theorem sU_cube : sU ^ 3 = -1 := by
  rw [← sU_pow, (model7_stab_pows).2.2.2.2.2.2.2.1, M22.toSL_negOne]
theorem sS_four : sS ^ 4 = 1 := by
  rw [← sS_pow, (model7_stab_pows).2.2.2.2.2.2.2.2.1, M22.toSL_one]
theorem sU_six : sU ^ 6 = 1 := by
  rw [← sU_pow, (model7_stab_pows).2.2.2.2.2.2.2.2.2.1, M22.toSL_one]

theorem sS_pow_inj (a b : Fin 4) (h : sS ^ (a : ℕ) = sS ^ (b : ℕ)) : a = b := by
  rw [← sS_pow, ← sS_pow] at h
  exact (model7_stab_pows).2.2.2.2.2.2.2.2.2.2.1 a b
    (M22.toSL_injOn (toSL_mpow mS_det a).1 (toSL_mpow mS_det b).1 h)

theorem sU_pow_inj (a b : Fin 6) (h : sU ^ (a : ℕ) = sU ^ (b : ℕ)) : a = b := by
  rw [← sU_pow, ← sU_pow] at h
  exact (model7_stab_pows).2.2.2.2.2.2.2.2.2.2.2 a b
    (M22.toSL_injOn (toSL_mpow mU_det a).1 (toSL_mpow mU_det b).1 h)

lemma model7_sS_fix : sS • model7I = model7I :=
  (MPDatum.smul_pt_eq_iff model7_wf _ mS_det _ _).2 (model7_stab_pows).2.2.2.1

lemma model7_sU_fix : sU • model7Rho = model7Rho :=
  (MPDatum.smul_pt_eq_iff model7_wf _ mU_det _ _).2 (model7_stab_pows).2.2.1

lemma pow_smul_fix {X : Type} [MulAction SL(2, ℤ) X] {s : SL(2, ℤ)} {x : X} (h : s • x = x) (n : ℕ) : s ^ n • x = x := by
  induction n with
  | zero => rw [pow_zero, one_smul]
  | succ n ih => rw [pow_succ, mul_smul, h, ih]

/-- **`Γ_i = ⟨S⟩`**: `g` fixes `i` iff `g = S^j`, `j < 4`. -/
theorem model7_stab_I (g : SL(2, ℤ)) : g • model7I = model7I ↔ ∃ j : Fin 4, g = sS ^ (j : ℕ) := by
  obtain ⟨-, -, -, -, -, g6, -⟩ := model7_tree_data
  refine ⟨fun h => ?_, ?_⟩
  · obtain ⟨a, ha, rfl⟩ := (model7_fix_iff g ⟨6, by decide⟩ g6).1 h
    obtain ⟨j, rfl⟩ := (model7_stab_pows).1 a ha
    exact ⟨j, sS_pow j⟩
  · rintro ⟨j, rfl⟩
    exact pow_smul_fix model7_sS_fix _

/-- **`Γ_ρ = ⟨U⟩`**: `g` fixes `ρ` iff `g = U^j`, `j < 6`. -/
theorem model7_stab_Rho (g : SL(2, ℤ)) : g • model7Rho = model7Rho ↔ ∃ j : Fin 6, g = sU ^ (j : ℕ) := by
  obtain ⟨-, -, -, -, g4, -⟩ := model7_tree_data
  refine ⟨fun h => ?_, ?_⟩
  · obtain ⟨a, ha, rfl⟩ := (model7_fix_iff g ⟨4, by decide⟩ g4).1 h
    obtain ⟨j, rfl⟩ := (model7_stab_pows).2.1 a ha
    exact ⟨j, sU_pow j⟩
  · rintro ⟨j, rfl⟩
    exact pow_smul_fix model7_sU_fix _

/-- **`Γ_{[ρ,i]} = {±I}`**, as powers of `−I`. -/
theorem model7_stab_edge (g : SL(2, ℤ)) :
    (g • model7Rho = model7Rho ∧ g • model7I = model7I) ↔ ∃ j : Fin 2, g = (-1 : SL(2, ℤ)) ^ (j : ℕ) := by
  refine ⟨fun ⟨h1, h2⟩ => ?_, ?_⟩
  · rcases model7_tree_edge_stab g h1 h2 with rfl | rfl
    · exact ⟨0, by rw [M22.toSL_one]; exact (pow_zero _).symm⟩
    · exact ⟨1, by rw [M22.toSL_negOne]; exact (pow_one _).symm⟩
  · rintro ⟨j, rfl⟩
    obtain ⟨h1, h2⟩ := model7_negI_fixes
    rw [M22.toSL_negOne] at h1 h2
    exact ⟨pow_smul_fix h1 _, pow_smul_fix h2 _⟩

/-- The tree vertex `i` (as a `0`-simplex of `𝒯`). -/
noncomputable def model7IV : TS model7Hyp 3 0 := fV model7TreeRetr model7I

/-- The tree vertex `ρ`. -/
noncomputable def model7RhoV : TS model7Hyp 3 0 := fV model7TreeRetr model7Rho

lemma model7_ord_I : model7Hyp.ord model7I = 4 := by obtain ⟨-, -, -, h6, -⟩ := model7_tree_data; exact h6

lemma model7_ord_Rho : model7Hyp.ord model7Rho = 3 := by obtain ⟨-, -, h4, -⟩ := model7_tree_data; exact h4

lemma model7_f_I : model7TreeRetr.f model7I = model7I :=
  model7TreeRetr.f_id _ (by rw [model7_ord_I]; decide)

lemma model7_f_Rho : model7TreeRetr.f model7Rho = model7Rho :=
  model7TreeRetr.f_id _ (by rw [model7_ord_Rho])

lemma model7IV_val : model7IV.1.1 0 = model7I := model7_f_I

lemma model7RhoV_val : model7RhoV.1.1 0 = model7Rho := model7_f_Rho

lemma TS0_ext {x y : TS model7Hyp 3 0} (h : x.1.1 0 = y.1.1 0) : x = y :=
  TS.ext' (funext fun j => by fin_cases j; exact h)

/-! ### Orbit–stabiliser as `Γ`-sets and permutation representations -/

section OrbEquiv

variable {G : Type} [Group G] {X : Type} [MulAction G X]

/-- **Orbit–stabiliser**: for a transitive action with `Stab(x) = K`, `G ⧸ K ≃ X`, `gK ↦ g x`. -/
noncomputable def orbEquiv (x : X) (K : Subgroup G) (hK : MulAction.stabilizer G x = K) (hs : ∀ y, ∃ g : G, g • x = y) :
    G ⧸ K ≃ X :=
  (Subgroup.quotientEquivOfEq hK.symm).trans (Equiv.ofBijective (MulAction.ofQuotientStabilizer G x)
    ⟨MulAction.injective_ofQuotientStabilizer G x, fun y => by
      obtain ⟨g, h⟩ := hs y; exact ⟨QuotientGroup.mk g, h⟩⟩)

lemma orbEquiv_mk (x : X) (K : Subgroup G) (hK : MulAction.stabilizer G x = K) (hs : ∀ y, ∃ g : G, g • x = y) (g : G) :
    orbEquiv x K hK hs (QuotientGroup.mk g) = g • x := rfl

/-- The orbit–stabiliser bijection is `Γ`-equivariant. -/
lemma orbEquiv_smul (x : X) (K : Subgroup G) (hK : MulAction.stabilizer G x = K) (hs : ∀ y, ∃ g : G, g • x = y)
    (g : G) (q : G ⧸ K) : orbEquiv x K hK hs (g • q) = g • orbEquiv x K hK hs q := by
  induction q using QuotientGroup.induction_on with
  | H h => show (g * h) • x = g • h • x; rw [mul_smul]

variable (k : Type) [CommRing k]

/-- An equivariant bijection of `Γ`-sets is an isomorphism of permutation representations. -/
noncomputable def ofMulActionIso {Y : Type} [MulAction G Y] (e : X ≃ Y) (he : ∀ (g : G) x, e (g • x) = g • e x) :
    Rep.ofMulAction k G X ≅ Rep.ofMulAction k G Y :=
  Rep.mkIso (Representation.Equiv.mk
    ((MonoidAlgebra.coeffLinearEquiv k).trans ((Finsupp.domLCongr e).trans (MonoidAlgebra.coeffLinearEquiv k).symm))
    (fun g => by
      refine LinearMap.ext fun x => ?_
      simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, LinearEquiv.trans_apply,
        Representation.ofMulAction_def, LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply,
        Finsupp.domLCongr_apply, Finsupp.domCongr_apply, Finsupp.equivMapDomain_eq_mapDomain]
      congr 1
      rw [← Finsupp.mapDomain_comp, ← Finsupp.mapDomain_comp]
      congr 1
      funext y
      exact he g y))

end OrbEquiv

lemma model7_top2 : 2 < model7MP.top.length := by decide +kernel

lemma model7_inS (p : ℕ) (hp : p ∈ model7MP.tv 2) (hp' : p < model7MP.pts.length) :
    InS (C := model7Cover) ((1 : SL(2, ℤ)), (⟨2, model7_top2⟩ : Fin model7MP.top.length))
      (MPDatum.pt model7_wf ⟨p, hp'⟩) := by
  show (1 : SL(2, ℤ))⁻¹ • (MPDatum.pt model7_wf ⟨p, hp'⟩ : MPDatum.P model7MP model7_wf) ∈
    (MPDatum.cover model7_wf).vtx ⟨2, model7_top2⟩
  rw [inv_one, one_smul]
  exact MPDatum.pt_mem_vtx model7_wf (σ := ⟨2, model7_top2⟩) hp

/-- `[ρ, i]` is an edge of `Y` (inside the maximal simplex `[1, 4, 6]`). -/
lemma model7_isE : IsE (H := model7Hyp) model7Rho model7I :=
  isE_iff.2 ⟨by rw [model7_ord_Rho, model7_ord_I]; decide, _, model7_inS 4 (by decide) (by decide),
    model7_inS 6 (by decide) (by decide)⟩

/-- **The edge `[ρ, i]`** of the tree. -/
noncomputable def model7E : TS model7Hyp 3 1 :=
  ⟨⟨![model7Rho, model7I], model7_isE⟩, by show 3 ≤ model7Hyp.ord model7Rho; rw [model7_ord_Rho]⟩

lemma model7E_val : model7E.1.1 = ![model7Rho, model7I] := rfl

lemma TS1_ext {x y : TS model7Hyp 3 1} (h0 : x.1.1 0 = y.1.1 0) (h1 : x.1.1 1 = y.1.1 1) : x = y :=
  TS.ext' (funext fun j => by fin_cases j; exacts [h0, h1])

/-- **Every tree edge is a translate of `[ρ, i]`.** -/
theorem model7_cov_edge (x : TS model7Hyp 3 1) : ∃ g : SL(2, ℤ), g • model7E = x := by
  obtain ⟨g, h0, h1⟩ := model7_tree_edges x
  exact ⟨g, TS1_ext h0.symm h1.symm⟩

/-- **`Stab[ρ, i] = ⟨−I⟩`**, the powers `(−I)^j`, `j < 2`. -/
theorem model7_smul_E_iff (g : SL(2, ℤ)) : g • model7E = model7E ↔ ∃ j : Fin 2, g = (-1 : SL(2, ℤ)) ^ (j : ℕ) := by
  rw [← model7_stab_edge]
  constructor
  · intro h
    exact ⟨congrArg (fun x : TS model7Hyp 3 1 => x.1.1 0) h, congrArg (fun x : TS model7Hyp 3 1 => x.1.1 1) h⟩
  · rintro ⟨h0, h1⟩; exact TS1_ext h0 h1

/-- **Every tree vertex is a translate of `ρ` or of `i`.** -/
theorem model7_cov_vtx (x : TS model7Hyp 3 0) : ∃ g : SL(2, ℤ), g • model7RhoV = x ∨ g • model7IV = x := by
  obtain ⟨g, h | h⟩ := (model7_tree_vtx (x.1.1 0)).1 x.2
  · exact ⟨g, Or.inl (TS0_ext (by rw [TS.smul_val, YS.smul_val, model7RhoV_val, h]))⟩
  · exact ⟨g, Or.inr (TS0_ext (by rw [TS.smul_val, YS.smul_val, model7IV_val, h]))⟩

lemma model7_smul_IV_iff (g : SL(2, ℤ)) : g • model7IV = model7IV ↔ g • model7I = model7I :=
  ⟨fun h => by have := congrArg (fun x : TS model7Hyp 3 0 => x.1.1 0) h; simpa [TS.smul_val, YS.smul_val,
    model7IV_val] using this, fun h => TS0_ext (by rw [TS.smul_val, YS.smul_val, model7IV_val, h])⟩

lemma model7_smul_RhoV_iff (g : SL(2, ℤ)) : g • model7RhoV = model7RhoV ↔ g • model7Rho = model7Rho :=
  ⟨fun h => by have := congrArg (fun x : TS model7Hyp 3 0 => x.1.1 0) h; simpa [TS.smul_val, YS.smul_val,
    model7RhoV_val] using this, fun h => TS0_ext (by rw [TS.smul_val, YS.smul_val, model7RhoV_val, h])⟩

/-- The two vertex orbits are distinct (orders `3 ≠ 4`). -/
lemma model7_RhoV_ne_IV (g g' : SL(2, ℤ)) : g • model7RhoV ≠ g' • model7IV := by
  intro h
  have := congrArg (fun x : TS model7Hyp 3 0 => model7Hyp.ord (x.1.1 0)) h
  simp only [TS.smul_val, YS.smul_val, model7Hyp.ord_smul, model7RhoV_val, model7IV_val, model7_ord_Rho,
    model7_ord_I] at this
  exact absurd this (by decide)

lemma stabilizer_eq_zpowers {X : Type} [MulAction SL(2, ℤ) X] (x : X) (t : SL(2, ℤ)) (N : ℕ) (ht : t • x = x)
    (h : ∀ g : SL(2, ℤ), g • x = x → ∃ j : Fin N, g = t ^ (j : ℕ)) :
    MulAction.stabilizer SL(2, ℤ) x = Subgroup.zpowers t := by
  apply le_antisymm
  · intro g hg
    obtain ⟨j, rfl⟩ := h g hg
    exact Subgroup.pow_mem _ (Subgroup.mem_zpowers t) _
  · rw [Subgroup.zpowers_le]; exact ht

/-- `Stab[ρ, i] = ⟨−I⟩`. -/
theorem model7_stabilizer_E : MulAction.stabilizer SL(2, ℤ) model7E = Subgroup.zpowers (-1) :=
  stabilizer_eq_zpowers _ _ 2 ((model7_smul_E_iff _).2 ⟨1, (pow_one _).symm⟩) fun g h => (model7_smul_E_iff g).1 h

/-- `Γ_i = ⟨S⟩`. -/
theorem model7_stabilizer_IV : MulAction.stabilizer SL(2, ℤ) model7IV = Subgroup.zpowers sS :=
  stabilizer_eq_zpowers _ _ 4 ((model7_smul_IV_iff _).2 model7_sS_fix) fun g h =>
    (model7_stab_I g).1 ((model7_smul_IV_iff g).1 h)

/-- `Γ_ρ = ⟨U⟩`. -/
theorem model7_stabilizer_RhoV : MulAction.stabilizer SL(2, ℤ) model7RhoV = Subgroup.zpowers sU :=
  stabilizer_eq_zpowers _ _ 6 ((model7_smul_RhoV_iff _).2 model7_sU_fix) fun g h =>
    (model7_stab_Rho g).1 ((model7_smul_RhoV_iff g).1 h)

/-- **The tree edges as a `Γ`-set: `Γ/⟨−I⟩ ≃ 𝒯_1`**, `g⟨−I⟩ ↦ g·[ρ, i]`. -/
noncomputable def model7EdgeEquiv : SL(2, ℤ) ⧸ Subgroup.zpowers (-1 : SL(2, ℤ)) ≃ TS model7Hyp 3 1 :=
  orbEquiv model7E _ model7_stabilizer_E model7_cov_edge

/-- `Γ/⟨U⟩ → 𝒯_0`, `gU ↦ gρ`. -/
noncomputable def model7RhoEquiv : SL(2, ℤ) ⧸ Subgroup.zpowers sU → TS model7Hyp 3 0 :=
  MulAction.ofQuotientStabilizer SL(2, ℤ) model7RhoV ∘ Subgroup.quotientEquivOfEq model7_stabilizer_RhoV.symm

/-- `Γ/⟨S⟩ → 𝒯_0`, `gS ↦ g i`. -/
noncomputable def model7IEquiv : SL(2, ℤ) ⧸ Subgroup.zpowers sS → TS model7Hyp 3 0 :=
  MulAction.ofQuotientStabilizer SL(2, ℤ) model7IV ∘ Subgroup.quotientEquivOfEq model7_stabilizer_IV.symm

lemma model7RhoEquiv_mk (g : SL(2, ℤ)) : model7RhoEquiv (QuotientGroup.mk g) = g • model7RhoV := rfl
lemma model7IEquiv_mk (g : SL(2, ℤ)) : model7IEquiv (QuotientGroup.mk g) = g • model7IV := rfl

/-- The vertex map `Γ/⟨U⟩ ⊔ Γ/⟨S⟩ → 𝒯_0`. -/
noncomputable def model7VtxMap : (SL(2, ℤ) ⧸ Subgroup.zpowers sU) ⊕ (SL(2, ℤ) ⧸ Subgroup.zpowers sS) → TS model7Hyp 3 0 :=
  Sum.elim model7RhoEquiv model7IEquiv

lemma model7VtxMap_bij : Function.Bijective model7VtxMap := by
  refine ⟨?_, fun y => ?_⟩
  · rintro (a | a) (b | b) h
    · induction a using QuotientGroup.induction_on with
      | H a => induction b using QuotientGroup.induction_on with
        | H b =>
          have h' : a • model7RhoV = b • model7RhoV := h
          refine congrArg Sum.inl (QuotientGroup.eq.2 ?_)
          rw [← model7_stabilizer_RhoV, MulAction.mem_stabilizer_iff, mul_smul, ← h', inv_smul_smul]
    · induction a using QuotientGroup.induction_on with
      | H a => induction b using QuotientGroup.induction_on with
        | H b => exact absurd h (model7_RhoV_ne_IV a b)
    · induction a using QuotientGroup.induction_on with
      | H a => induction b using QuotientGroup.induction_on with
        | H b => exact absurd h.symm (model7_RhoV_ne_IV b a)
    · induction a using QuotientGroup.induction_on with
      | H a => induction b using QuotientGroup.induction_on with
        | H b =>
          have h' : a • model7IV = b • model7IV := h
          refine congrArg Sum.inr (QuotientGroup.eq.2 ?_)
          rw [← model7_stabilizer_IV, MulAction.mem_stabilizer_iff, mul_smul, ← h', inv_smul_smul]
  · obtain ⟨g, h | h⟩ := model7_cov_vtx y
    · exact ⟨Sum.inl (QuotientGroup.mk g), h⟩
    · exact ⟨Sum.inr (QuotientGroup.mk g), h⟩

/-- **The tree vertices as a `Γ`-set: `Γ/⟨U⟩ ⊔ Γ/⟨S⟩ ≃ 𝒯_0`**. -/
noncomputable def model7VtxEquiv :
    (SL(2, ℤ) ⧸ Subgroup.zpowers sU) ⊕ (SL(2, ℤ) ⧸ Subgroup.zpowers sS) ≃ TS model7Hyp 3 0 :=
  Equiv.ofBijective _ model7VtxMap_bij

lemma model7VtxEquiv_smul (g : SL(2, ℤ)) (q : (SL(2, ℤ) ⧸ Subgroup.zpowers sU) ⊕ (SL(2, ℤ) ⧸ Subgroup.zpowers sS)) :
    model7VtxEquiv (g • q) = g • model7VtxEquiv q := by
  rcases q with q | q <;> induction q using QuotientGroup.induction_on with
  | H h => exact mul_smul g h _

variable (k : Type) [CommRing k]

/-- **`k[𝒯_1] ≅ k[Γ/⟨−I⟩]`** in `Rep k Γ`. -/
noncomputable def model7TreeRep1 :
    Rep.ofMulAction k SL(2, ℤ) (SL(2, ℤ) ⧸ Subgroup.zpowers (-1 : SL(2, ℤ))) ≅ Rep.ofMulAction k SL(2, ℤ) (TS model7Hyp 3 1) :=
  ofMulActionIso k model7EdgeEquiv (orbEquiv_smul _ _ _ _)

/-- **`k[𝒯_0] ≅ k[Γ/⟨U⟩ ⊔ Γ/⟨S⟩] = k[Γ/⟨U⟩] ⊕ k[Γ/⟨S⟩]`** in `Rep k Γ`. -/
noncomputable def model7TreeRep0 :
    Rep.ofMulAction k SL(2, ℤ) ((SL(2, ℤ) ⧸ Subgroup.zpowers sU) ⊕ (SL(2, ℤ) ⧸ Subgroup.zpowers sS)) ≅
      Rep.ofMulAction k SL(2, ℤ) (TS model7Hyp 3 0) :=
  ofMulActionIso k model7VtxEquiv model7VtxEquiv_smul

/-- **For `M₇`, the retraction `r : k[Y_•] → k[𝒯_•]` is a map of permutation representations** (degrees `≥ 1`). -/
theorem model7_tRetr_smul (s : ℕ) (g : SL(2, ℤ)) (x : YS model7Hyp s →₀ k) :
    tRetr k model7TreeRetr (s + 1) (mapDomain (fun y : YS model7Hyp s => g • y) x) =
      mapDomain (fun y : TS model7Hyp 3 s => g • y) (tRetr k model7TreeRetr (s + 1) x) :=
  tRetr_smul k model7TreeRetr (MPDatum.treeRetr_equivariant _ _ _ _) s g x

end GL2.MP
