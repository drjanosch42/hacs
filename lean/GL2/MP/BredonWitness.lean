/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.MP.Bredon
import GL2.L23.OddExponent
import GL2.Model.XBdryModel

/-!
# C-w (slot 24, M6 close): the "not only if" witness of [C] `prop:mpbredon` (2)

[GT] `gt_part_MP.tex`, rows `thm:mp-bredon-zero`, `thm:mp-epsmp-witness`, backing [C] `prop:mpbredon` (2), second sentence:
"**Not only if**: for `R = ℤ[1/2]` and `n` odd both sides are acyclic …, while `3 ∉ R^×`".

* **Bredon side** (`bredonHom_eq_zero`, `bredon_isZero_homology`): if `−I` lies in every stabiliser of the descriptor
  (`model7_negI`, [C] (B3)), `−I` acts trivially on every permutation module `k[Simp_q]`; a `Γ`-map `f` into `A` with
  `ρ_A(−I) = −1` then satisfies `f = −f`, so `f = 0` once `2 ∈ k^×`.  Hence `C^q_eq(K, Rel; A) ≅ Hom_Γ(k[Simp_q], A) = 0`
  termwise and its cohomology vanishes in every degree ([C] `eq:negI` with `2` invertible).
* **Group / MP side** (`mp_isZero_homology`): `H^q(C^•(S; A)) ≅ H^q(Γ; A)` (`mpComparison`, (AC) over `k`) and
  `H^q(Γ; A) = 0` by LG-1 (`GL2.L23.OddExp.eq_zero_of_two_unit`: exponent `2` by centrality, `2 ∈ k^×`).
* **The witness** (`epsMP_quasiIso_of_two_unit`, `model7_epsMP_witness`, `rz_epsMP_witness`): both sides acyclic, so `ε^MP` is a
  quasi-isomorphism for every `k` with `2 ∈ k^×` and every `A` with `−I ↦ −1`, in particular `Sym^n k²` for odd `n`; at
  `k = ℤ[1/2]` (`ZHalf`) `3` and `6` are **not** units (`zHalf_three_not_isUnit`, via `ℤ[1/2] → 𝔽₃`), so the "if" of
  (2) (`6 ∈ k^×`) is not necessary.
-/

noncomputable section

namespace GL2.MP

open CategoryTheory Limits GL2.Model
open scoped MatrixGroups

/-! ### `−I` acts trivially on `k[Simp_q]` -/

section NegI

variable {k : Type} [CommRing k] (D : XBulkDatum)

/-- `−I` fixes every simplex of `X_bulk` when it lies in every stabiliser. -/
lemma negOne_smul_simp (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ) (q : ℤ) (s : XBulk.Simp D q) :
    (-1 : SL(2, ℤ)) • s = s := by
  induction s using Quot.ind with
  | _ x =>
    obtain ⟨γ, σ⟩ := x
    show XBulk.SimpM.mk _ (-1 * γ, σ) = XBulk.SimpM.mk _ (γ, σ)
    refine ((XBulk.simpMk_eq_iff _ _ _).2 ?_).symm
    have hc : (-1 : SL(2, ℤ)) * γ = γ * -1 := by
      ext i j; simp
    show γ⁻¹ * (-1 * γ) ∈ (D.M q).S σ σ
    rw [hc, inv_mul_cancel_left]
    exact hneg q σ

/-- `ρ(−I) = 1` on the permutation module `k[Simp_q]`. -/
lemma ofMulAction_negOne (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ) (q : ℤ)
    (x : Rep.ofMulAction k SL(2, ℤ) (XBulk.Simp D q)) :
    (Rep.ofMulAction k SL(2, ℤ) (XBulk.Simp D q)).ρ (-1) x = x := by
  have h : (fun s : XBulk.Simp D q => (-1 : SL(2, ℤ)) • s) = id := funext (negOne_smul_simp D hneg q)
  show Representation.ofMulAction k SL(2, ℤ) (XBulk.Simp D q) (-1) x = x
  rw [Representation.ofMulAction_def, h, Finsupp.lmapDomain_id]
  simp

variable {A : Rep.{0} k SL(2, ℤ)}

/-- **Every `Γ`-map `k[Simp_q] → A` vanishes** if `−I` fixes the simplices, acts by `−1` on `A`, and `2 ∈ k^×`. -/
theorem bredonHom_eq_zero (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ) (hI : L23.OddExp.NegOneActs A)
    (h2 : IsUnit (2 : k)) (q : ℤ) (f : Rep.ofMulAction k SL(2, ℤ) (XBulk.Simp D q) ⟶ A) : f = 0 := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  have e : f.hom x = -f.hom x := by
    conv_lhs => rw [← ofMulAction_negOne D hneg q x]
    rw [Rep.hom_comm_apply, hI]
  have e2 : (2 : k) • f.hom x = 0 := by rw [two_smul]; nth_rewrite 1 [e]; exact neg_add_cancel _
  obtain ⟨u, hu⟩ := h2
  show f.hom x = 0
  calc f.hom x = (↑u⁻¹ * (2 : k)) • f.hom x := by rw [← hu, Units.inv_mul, one_smul]
    _ = 0 := by rw [mul_smul, e2, smul_zero]

/-- **The Bredon complex is zero termwise**: `C^q_eq(K, Rel; A) = 0` for every `q`. -/
theorem bredon_isZero_X (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ) (hI : L23.OddExp.NegOneActs A)
    (h2 : IsUnit (2 : k)) (n : ℕ) : IsZero ((XBulk.relCxA D A).toNat.X n) := by
  have : Subsingleton (((XBulk.repCx D k).linearYonedaObj k A).X n) :=
    ⟨fun a b => (bredonHom_eq_zero D hneg hI h2 n a).trans (bredonHom_eq_zero D hneg hI h2 n b).symm⟩
  have hz : IsZero (((XBulk.repCx D k).linearYonedaObj k A).X n) := ModuleCat.isZero_of_subsingleton _
  exact IsZero.of_iso hz ((HomologicalComplex.eval _ _ n).mapIso (XBulk.homEqv_iso D A))

/-- **The Bredon side is acyclic**: `H^q(C^•_eq(K, Rel; A)) = 0` for every `q` ([C] `eq:negI`, `2 ∈ k^×`). -/
theorem bredon_isZero_homology (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hI : L23.OddExp.NegOneActs A) (h2 : IsUnit (2 : k)) (n : ℕ) :
    IsZero ((XBulk.relCxA D A).toNat.homology n) :=
  ShortComplex.isZero_homology_of_isZero_X₂ _ (bredon_isZero_X D hneg hI h2 n)

end NegI

/-! ### The MP (group) side and the witness -/

namespace Cover

variable {k : Type} [CommRing k] {C : Cover.{0} SL(2, ℤ)} (H : C.Hyp) (hAC : (yCx H k).toCx.Acyclic)
  {A : Rep.{0} k SL(2, ℤ)}

include H hAC in
/-- **The MP side is acyclic**: `H^q(C^•(S; A)) ≅ H^q(Γ; A) = 0` (LG-1, `2 ∈ k^×`). -/
theorem mp_isZero_homology (hI : L23.OddExp.NegOneActs A) (h2 : IsUnit (2 : k)) (n : ℕ) :
    IsZero ((C.mpComplex A).homology n) := by
  have : Subsingleton (groupCohomology A n) :=
    ⟨fun a b => (L23.OddExp.eq_zero_of_two_unit hI h2 n a).trans (L23.OddExp.eq_zero_of_two_unit hI h2 n b).symm⟩
  have hz : IsZero (groupCohomology A n) := ModuleCat.isZero_of_subsingleton _
  exact IsZero.of_iso hz (mpComparison H A hAC n).symm

variable (D : XBulkDatum) (hA : XBulk.Acyclic D) (v : D.V 0)

/-- **[C] `prop:mpbredon` (2), "not only if"**: with `−I` in every stabiliser, `2 ∈ k^×` and `−I ↦ −1` on `A`, both sides
are acyclic, so `ε^MP` is a quasi-isomorphism — whether or not `3 ∈ k^×`. -/
theorem epsMP_quasiIso_of_two_unit (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hI : L23.OddExp.NegOneActs A) (h2 : IsUnit (2 : k)) : QuasiIso (epsMP H hAC D hA v A) := by
  rw [quasiIso_iff]
  intro n
  rw [quasiIsoAt_iff_isIso_homologyMap]
  exact isIso_of_source_target_iso_zero _ (bredon_isZero_homology D hneg hI h2 n).isoZero
    (mp_isZero_homology H hAC hI h2 n).isoZero

end Cover

/-! ### `ℤ[1/2]` -/

/-- `ℤ[1/2]`. -/
abbrev ZHalf := Localization.Away (2 : ℤ)

lemma zHalf_two_isUnit : IsUnit (2 : ZHalf) := by
  simpa using IsLocalization.Away.algebraMap_isUnit (S := ZHalf) (2 : ℤ)

/-- The reduction `ℤ[1/2] → 𝔽₃` (`2 = −1` is a unit of `𝔽₃`). -/
def zHalfToF3 : ZHalf →+* ZMod 3 :=
  IsLocalization.Away.lift (2 : ℤ) (g := Int.castRingHom (ZMod 3)) ⟨⟨2, 2, by decide, by decide⟩, by simp⟩

/-- **`3 ∉ ℤ[1/2]^×`** (its image in `𝔽₃` is `0`). -/
theorem zHalf_three_not_isUnit : ¬ IsUnit (3 : ZHalf) := fun h => by
  have h' := h.map zHalfToF3
  rw [map_ofNat, show (3 : ZMod 3) = 0 from rfl] at h'
  exact not_isUnit_zero h'

/-- **`6 ∉ ℤ[1/2]^×`**, so the "if" hypothesis of [C] `prop:mpbredon` (2) fails at `ℤ[1/2]`. -/
theorem zHalf_six_not_isUnit : ¬ IsUnit (6 : ZHalf) := fun h =>
  zHalf_three_not_isUnit (isUnit_of_mul_isUnit_right (x := (2 : ZHalf)) (by norm_num; exact h))

/-! ### The instances: `M₇` and `R_ℤ`, `Sym^n ℤ[1/2]²`, `n` odd -/

/-- **The witness on `M₇`**: for `R = ℤ[1/2]` and `n` odd, `H^q` of the Bredon side and of the MP side of
`Sym^n R²` vanish in every degree, `ε^MP` is a quasi-isomorphism, and `6 ∉ R^×` ([C] `prop:mpbredon` (2), "not only if"). -/
theorem model7_epsMP_witness {n : ℕ} (hn : Odd n) :
    (∀ q, IsZero ((XBulk.relCxA model7 (L23.OddExp.VK ZHalf n)).toNat.homology q)) ∧
    (∀ q, IsZero ((model7Cover.mpComplex (L23.OddExp.VK ZHalf n)).homology q)) ∧
    QuasiIso (Cover.epsMP model7Hyp (model7_AC ZHalf) model7 model7_acyclic (M7.vx 0) (L23.OddExp.VK ZHalf n)) ∧
    ¬ IsUnit (6 : ZHalf) :=
  ⟨bredon_isZero_homology model7 model7_negI (L23.OddExp.VK_negOneActs hn) zHalf_two_isUnit,
    Cover.mp_isZero_homology model7Hyp (model7_AC ZHalf) (L23.OddExp.VK_negOneActs hn) zHalf_two_isUnit,
    Cover.epsMP_quasiIso_of_two_unit model7Hyp (model7_AC ZHalf) model7 model7_acyclic (M7.vx 0) model7_negI
      (L23.OddExp.VK_negOneActs hn) zHalf_two_isUnit,
    zHalf_six_not_isUnit⟩

/-- **The witness with the MP side on `R_ℤ`** (Bredon side `M₇`). -/
theorem rz_epsMP_witness {n : ℕ} (hn : Odd n) :
    QuasiIso (Cover.epsMP rzHyp (rz_AC ZHalf) model7 model7_acyclic (M7.vx 0) (L23.OddExp.VK ZHalf n)) ∧
    ¬ IsUnit (6 : ZHalf) :=
  ⟨Cover.epsMP_quasiIso_of_two_unit rzHyp (rz_AC ZHalf) model7 model7_acyclic (M7.vx 0) model7_negI
      (L23.OddExp.VK_negOneActs hn) zHalf_two_isUnit,
    zHalf_six_not_isUnit⟩

end GL2.MP
