import GL2.MP.MorphNatural
import GL2.MP.RZSubdiv

/-!
# MP-L3, rows L8 / L9 on `R_ℤ → M₇` (slot 22): the refinement with its label table

[GT] `gt_part_MP.tex`, rows `def:mp-orbmap`, `thm:mp-orbmap-morph`, `def:mp-rz-refine`, `thm:mp-rz-refine`,
`thm:mp-rz-untwisted`, backing [C] `def:mpmorphism` (2), [C] `thm:mpnatural` (3), (4) on `R_ℤ → M₇` and its
`\COMP` record (`OrbifoldMPRefinementValidation`).

**Decided vertex maps** (`MPDatum.OrbOk`, `MPDatum.LabOk`, `MPDatum.morph`).  An orbit table `o ↦ (o', t_o)`
(`Γ_o ⊆ t_o Γ'_{o'} t_o⁻¹`, decided) defines the `Γ`-equivariant vertex map `g ι(o) ↦ g t_o ι'(o')`; a label table
`σ ↦ (h_σ, σ')` with `h_σ⁻¹ f(p) ∈ ι'(σ')` for every vertex `p` of `ι(σ)` (decided) is then a morphism of MP data
(`Cover.Morph.ofVtx`).  For a refinement the vertex map is not part of [C]'s datum (a subdivision is not a simplicial map
`Ỹ → Y`); any equivariant vertex map with the containments certifies the one hypothesis `Morph.cov` that the induced maps
use, and the induced maps depend on the labels only.

**`R_ℤ → M₇`** (`rzRefine`): the collapse `7 ↦ 4`, `8 ↦ 5` of `GL2.MP.RZTau.cTab` as orbit table (orbit `5` of `R_ℤ`, the
new vertex `7` on the edge orbit `4–6`, goes to orbit `3` of `M₇`), and the seven labels of the subdivision record: each
`R_ℤ` triangle in one `M₇` triangle, every `h = I` (`0, 1 ↦ v₂v₅v₆`, `2 ↦ v₂v₃v₅`, `3, 4 ↦ v₁v₄v₆`, `5 ↦ v₁v₂v₆`,
`6 ↦ v₀v₁v₄`).  `rzRefineFlip`: the same with the label of triangle `0` replaced by `h · (−I)`, the other member of
its label coset.

* `rz_refine_quasiIso`: **[C] `thm:mpnatural` (3) on `R_ℤ → M₇`**: `u^* : C^•(S(M₇); A) → C^•(S(R_ℤ); A)` is a
  quasi-isomorphism for every commutative ring and every representation, inducing the identity of `H^•(SL₂(ℤ); A)`
  under the comparisons (`rz_refine_homology_eq`), and the same on cohomology for both label choices
  (`rz_refine_label_indep`).
* `rz_untwisted_const`: with all labels `I` the untwisted `f ↦ f ∘ u` is a cochain map (constant labels);
  `rz_untwisted_fails`: **with the mixed labels of `rzRefineFlip`, at `n = 1` (`A = 𝔽₃²`, the standard representation,
  `ρ(−I) = −1`), the untwisted map is not a cochain map** ([C] `thm:mpnatural` (4), `REFINE_TWIST`), while the twisted
  `u^*` is one for every label choice (`Morph.pull_dC`).
-/

noncomputable section

namespace GL2.MP

open GL2.TauChk CategoryTheory
open scoped MatrixGroups

namespace MPDatum

variable {D D' : MPDatum} (hW : D.WF) (hW' : D'.WF)

/-- `pt p' = (o', m)` from `samePt`. -/
lemma pt_eq_mkP_of_samePt (p' : Fin D'.pts.length) (o' : Fin D'.stab.length) (m : M22) (hm : m.det = 1)
    (h : D'.samePt (D'.orb p') (D'.gp p') o' m = true) : pt hW' p' = mkP hW' (o', M22.toSL m) := by
  simp only [samePt, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨e, hst⟩ := h
  have hg := hW'.pts_det p' p'.2
  have e' : (⟨D'.orb p', hW'.pts_orb p' p'.2⟩ : Fin D'.stab.length) = o' := Fin.ext e
  rw [pt, mkP_eq_iff]
  refine ⟨e', ?_⟩
  have hdet : ((D'.gp p').adj * m).det = 1 := by rw [M22.det_mul, M22.det_adj, hg, hm]; norm_num
  have := (mem_sub_iff hW' ⟨D'.orb p', hW'.pts_orb p' p'.2⟩ hdet).2 hst
  rwa [M22.toSL_mul (by rw [M22.det_adj]; exact hg) hm, M22.toSL_adj hg] at this

variable (D D') in
/-- The target orbit of an orbit table (reduced into range). -/
def omO (om : List (ℕ × M22)) (o : ℕ) : ℕ := (om.getD o (0, M22.I)).1 % D'.stab.length

/-- The translate `t_o` of an orbit table. -/
def omT (om : List (ℕ × M22)) (o : ℕ) : M22 := (om.getD o (0, M22.I)).2

variable (D D') in
/-- **The orbit table is admissible**: `det t_o = 1` and `t_o⁻¹ Γ_o t_o ⊆ Γ'_{o'}`. -/
def OrbOk (om : List (ℕ × M22)) : Bool :=
  (List.range D.stab.length).all fun o => (omT om o).det == 1 &&
    (D.st o).all fun h => decide ((omT om o).adj * h * omT om o ∈ D'.st (omO D' om o))

/-- The label `h_σ`. -/
def labH (lt : List (M22 × ℕ)) (σ : ℕ) : M22 := (lt.getD σ (M22.I, 0)).1

variable (D') in
/-- The target simplex `σ'` (reduced into range). -/
def labS (lt : List (M22 × ℕ)) (σ : ℕ) : ℕ := (lt.getD σ (M22.I, 0)).2 % D'.top.length

variable (D D') in
/-- **The label table is admissible**: `det h_σ = 1` and `h_σ⁻¹ f(p) ∈ ι'(σ')` for every vertex `p` of `ι(σ)`. -/
def LabOk (om : List (ℕ × M22)) (lt : List (M22 × ℕ)) : Bool :=
  (List.range D.top.length).all fun σ => (labH lt σ).det == 1 &&
    (D.tv σ).all fun p => (D'.tv (labS D' lt σ)).any fun p' =>
      D'.samePt (D'.orb p') (D'.gp p') (omO D' om (D.orb p)) ((labH lt σ).adj * D.gp p * omT om (D.orb p))

variable {om : List (ℕ × M22)} (hpos : 0 < D'.stab.length) (hom : OrbOk D D' om = true)

include hom in
lemma omT_det (o : ℕ) (ho : o < D.stab.length) : (omT om o).det = 1 := by
  have := (List.all_eq_true.1 hom) o (List.mem_range.2 ho)
  simp only [Bool.and_eq_true, beq_iff_eq] at this
  exact this.1

include hom in
lemma omT_conj (o : ℕ) (ho : o < D.stab.length) (h : M22) (hh : h ∈ D.st o) :
    (omT om o).adj * h * omT om o ∈ D'.st (omO D' om o) := by
  have := (List.all_eq_true.1 hom) o (List.mem_range.2 ho)
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, decide_eq_true_eq] at this
  exact this.2 h hh

/-- The target orbit as an element of `Fin`. -/
def omF (om : List (ℕ × M22)) (o : ℕ) : Fin D'.stab.length := ⟨omO D' om o, Nat.mod_lt _ hpos⟩

include hom in
/-- **The vertex map** `g ι(o) ↦ g t_o ι'(o')` of an admissible orbit table. -/
def vmap : P D hW → P D' hW' :=
  Quotient.lift (fun x : Fin D.stab.length × SL(2, ℤ) => mkP hW' (omF hpos om x.1, x.2 * M22.toSL (omT om x.1)))
    (by
      rintro ⟨o, g⟩ ⟨o₂, g₂⟩ ⟨e1, e2⟩
      dsimp only at e1 e2 ⊢
      subst e1
      obtain ⟨h, hh, eh⟩ := e2
      rw [mkP_eq_iff]
      refine ⟨rfl, ?_⟩
      have ht := omT_det hom o o.2
      have hhd := hW.stab_det o o.2 h hh
      have hc := omT_conj hom o o.2 h hh
      have hdet : ((omT om o).adj * h * omT om o).det = 1 := by
        rw [M22.det_mul, M22.det_mul, M22.det_adj, ht, hhd]; norm_num
      have hmem := (mem_sub_iff hW' (omF hpos om o) hdet).2 hc
      rw [M22.toSL_mul (by rw [M22.det_mul, M22.det_adj, ht, hhd]; norm_num) ht,
        M22.toSL_mul (by rw [M22.det_adj]; exact ht) hhd, M22.toSL_adj ht, eh] at hmem
      dsimp only
      convert hmem using 1
      group)

lemma vmap_mkP (x : Fin D.stab.length × SL(2, ℤ)) :
    vmap hW hW' hpos hom (mkP hW x) = mkP hW' (omF hpos om x.1, x.2 * M22.toSL (omT om x.1)) := rfl

lemma vmap_smul (g : SL(2, ℤ)) (q : P D hW) : vmap hW hW' hpos hom (g • q) = g • vmap hW hW' hpos hom q := by
  induction q using Quotient.ind with
  | _ x =>
    show vmap hW hW' hpos hom (mkP hW (x.1, g * x.2)) = g • vmap hW hW' hpos hom (mkP hW x)
    rw [vmap_mkP, vmap_mkP, smul_mkP, mul_assoc]

variable {lt : List (M22 × ℕ)} (hpos' : 0 < D'.top.length) (hlab : LabOk D D' om lt = true)

/-- The label table as `σ ↦ (h_σ, σ')`. -/
def labF (σ : Fin D.top.length) : SL(2, ℤ) × Fin D'.top.length :=
  (M22.toSL (labH lt σ), ⟨labS D' lt σ, Nat.mod_lt _ hpos'⟩)

include hlab in
/-- **The containments** `h_σ⁻¹ f(ι(σ)) ⊆ ι'(σ')`. -/
lemma labF_vtx (σ : Fin D.top.length) (q : P D hW) (hq : q ∈ (cover hW).vtx σ) :
    (labF hpos' σ (lt := lt)).1⁻¹ • vmap hW hW' hpos hom q ∈ (cover hW').vtx (labF hpos' σ (lt := lt)).2 := by
  obtain ⟨p, hp, rfl⟩ := hq
  have h1 := (List.all_eq_true.1 hlab) σ (List.mem_range.2 σ.2)
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.any_eq_true] at h1
  obtain ⟨hdetH, h2⟩ := h1
  obtain ⟨p', hp', hsame⟩ := h2 p hp
  have hp'lt : p' < D'.pts.length := hW'.top_pts _ (Nat.mod_lt _ hpos') p' hp'
  have hgp := hW.pts_det p p.2
  have hpo : D.orb p < D.stab.length := hW.pts_orb p p.2
  have ht := omT_det hom (D.orb p) hpo
  have hm : ((labH lt σ).adj * D.gp p * omT om (D.orb p)).det = 1 := by
    rw [M22.det_mul, M22.det_mul, M22.det_adj, hdetH, hgp, ht]; norm_num
  refine ⟨⟨p', hp'lt⟩, hp', ?_⟩
  rw [pt_eq_mkP_of_samePt hW' ⟨p', hp'lt⟩ (omF hpos om (D.orb p)) _ hm hsame]
  show (M22.toSL (labH lt σ))⁻¹ • vmap hW hW' hpos hom (mkP hW (⟨D.orb p, hpo⟩, M22.toSL (D.gp p))) = _
  rw [vmap_mkP, smul_mkP, M22.toSL_mul (by rw [M22.det_mul, M22.det_adj, hdetH, hgp]; norm_num) ht,
    M22.toSL_mul (by rw [M22.det_adj]; exact hdetH) hgp, M22.toSL_adj hdetH, mul_assoc]

/-- **A decided orbit table and label table give a morphism of MP data** (`φ = id`). -/
def morph : Cover.Morph (MonoidHom.id SL(2, ℤ)) (cover hW) (cover hW') :=
  Cover.Morph.ofVtx (vmap hW hW' hpos hom) (vmap_smul hW hW' hpos hom) (labF hpos' (lt := lt))
    (labF_vtx hW hW' hpos hom hpos' hlab)

end MPDatum

/-! ### `R_ℤ → M₇` -/

/-- The orbit table of the collapse `7 ↦ 4`, `8 ↦ 5` (orbits `0, …, 4` to themselves, orbit `5` to orbit `3`). -/
def rzOrb : List (ℕ × M22) := [(0, M22.I), (1, M22.I), (2, M22.I), (3, M22.I), (4, M22.I), (3, M22.I)]

/-- **The label table of the subdivision `R_ℤ → M₇`**: every `R_ℤ` triangle in one `M₇` triangle, `h = I`. -/
def rzLab : List (M22 × ℕ) := [(M22.I, 4), (M22.I, 4), (M22.I, 3), (M22.I, 2), (M22.I, 2), (M22.I, 1), (M22.I, 0)]

/-- `−I`. -/
def M22.negI : M22 := ⟨-1, 0, 0, -1⟩

/-- The label table with the label of triangle `0` replaced by `h · (−I)` (the other member of its label coset). -/
def rzLabFlip : List (M22 × ℕ) := [(M22.negI, 4), (M22.I, 4), (M22.I, 3), (M22.I, 2), (M22.I, 2), (M22.I, 1), (M22.I, 0)]

theorem rz_orbOk : MPDatum.OrbOk rzMP model7MP rzOrb = true := by decide +kernel
theorem rz_labOk : MPDatum.LabOk rzMP model7MP rzOrb rzLab = true := by decide +kernel
theorem rz_labFlipOk : MPDatum.LabOk rzMP model7MP rzOrb rzLabFlip = true := by decide +kernel
theorem model7_pos : 0 < model7MP.stab.length ∧ 0 < model7MP.top.length := by decide +kernel

/-- **The refinement `R_ℤ → M₇`** with the labels of the subdivision record. -/
def rzRefine : Cover.Morph (MonoidHom.id SL(2, ℤ)) rzCover model7Cover :=
  MPDatum.morph rz_wf model7_wf model7_pos.1 rz_orbOk model7_pos.2 rz_labOk

/-- **The refinement `R_ℤ → M₇`** with one label flipped by `−I`. -/
def rzRefineFlip : Cover.Morph (MonoidHom.id SL(2, ℤ)) rzCover model7Cover :=
  MPDatum.morph rz_wf model7_wf model7_pos.1 rz_orbOk model7_pos.2 rz_labFlipOk

/-- **[C] `thm:mpnatural` (3) on `R_ℤ → M₇`**: `H^n(u^*)` is the composite of the comparison isomorphisms (the identity of
`H^n(SL₂(ℤ); A)`), for every commutative ring and every representation. -/
theorem rz_refine_homology_eq (k : Type) [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    HomologicalComplex.homologyMap (rzRefine.pullMap A) n =
      (Cover.mpComparisonNat model7Hyp (model7_AC k) A n).inv ≫ (Cover.mpComparisonNat rzHyp (rz_AC k) A n).hom :=
  Cover.refine_homology_eq rzRefine (rz_AC k) (model7_AC k) A n

/-- **Refinement invariance on `R_ℤ → M₇`**: `u^* : C^•(S(M₇); A) → C^•(S(R_ℤ); A)` is a quasi-isomorphism. -/
theorem rz_refine_quasiIso (k : Type) [CommRing k] (A : Rep.{0} k SL(2, ℤ)) : QuasiIso (rzRefine.pullMap A) :=
  Cover.refine_quasiIso rzRefine (rz_AC k) (model7_AC k) A

/-- **Label independence on `R_ℤ → M₇`**: both label choices induce the same map on cohomology. -/
theorem rz_refine_label_indep (k : Type) [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    HomologicalComplex.homologyMap (rzRefine.pullMap A) n = HomologicalComplex.homologyMap (rzRefineFlip.pullMap A) n :=
  Cover.pull_label_indep rzRefine rzRefineFlip (rz_AC k) (model7_AC k) A n

/-! ### The label factor on `R_ℤ → M₇` ([C] `thm:mpnatural` (4)) -/

lemma rzRefine_lab (σ : Fin rzMP.top.length) : (rzRefine.lab σ).1 = 1 := by
  show M22.toSL (MPDatum.labH rzLab σ) = 1
  have : MPDatum.labH rzLab σ = M22.I := by
    fin_cases σ <;> rfl
  rw [this]; exact M22.toSL_one

/-- **With the labels of the subdivision record (all `I`) the untwisted map is a cochain map.** -/
theorem rz_untwisted_const {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) (f : model7Cover.Coch A n) :
    rzRefine.untw A (n + 1) (model7Cover.dC A n f) = rzCover.dC (Rep.res (MonoidHom.id _) A) n (rzRefine.untw A n f) :=
  rzRefine.untw_dC_of_const A (fun σ τ => by rw [rzRefine_lab σ, rzRefine_lab τ])
    (fun σ g => by rw [rzRefine_lab σ, map_one, one_mul, mul_one]) n f

/-- **The standard representation** `𝔽₃²` of `SL₂(ℤ)` (`Sym¹`, `ρ(−I) = −1`). -/
def stdRep3 : Representation (ZMod 3) SL(2, ℤ) (Fin 2 → ZMod 3) where
  toFun g := Matrix.toLin' ((g : Matrix (Fin 2) (Fin 2) ℤ).map (Int.castRingHom (ZMod 3)))
  map_one' := by simp; rfl
  map_mul' g h := by
    simp only [Matrix.SpecialLinearGroup.coe_mul, Matrix.map_mul, Matrix.toLin'_mul]
    rfl

/-- `A = 𝔽₃²` as a representation. -/
abbrev A3 : Rep (ZMod 3) SL(2, ℤ) := Rep.of stdRep3

/-- The string `(v₁v₂v₆ ←1— v₂v₆v₈)` of `S_1(R_ℤ)` (triangles `5` and `0` of the dump share the vertex `2`). -/
def rzStr : rzCover.MPSet 1 :=
  ⟨![(1, ⟨5, by decide⟩), (1, ⟨0, by decide⟩)],
    ⟨MPDatum.pt rz_wf ⟨2, by decide⟩, fun i => by
      fin_cases i
      · exact ⟨⟨2, by decide⟩, by decide, by
        show (1 : SL(2, ℤ))⁻¹ • (MPDatum.pt rz_wf ⟨2, by decide⟩ : rzCover.P) = _
        rw [inv_one]; exact one_smul _ _⟩
      · exact ⟨⟨2, by decide⟩, by decide, by
        show (1 : SL(2, ℤ))⁻¹ • (MPDatum.pt rz_wf ⟨2, by decide⟩ : rzCover.P) = _
        rw [inv_one]; exact one_smul _ _⟩⟩, rfl⟩

lemma rzRefineFlip_lab0 : (rzRefineFlip.lab ⟨0, by decide⟩).1 = M22.toSL M22.negI := rfl
lemma rzRefineFlip_lab5 : (rzRefineFlip.lab ⟨5, by decide⟩).1 = 1 := M22.toSL_one

/-- **[C] `thm:mpnatural` (4) on `R_ℤ → M₇`**: with the mixed labels `I, −I` of `rzRefineFlip`, at `n = 1` over `𝔽₃`, the
untwisted `f ↦ f ∘ u` is **not** a cochain map `C^0(S(M₇); 𝔽₃²) → C^1(S(R_ℤ); 𝔽₃²)` (the twisted `u^*` is, `Morph.pull_dC`). -/
theorem rz_untwisted_fails :
    ∃ f : model7Cover.Coch A3 0,
      rzRefineFlip.untw A3 1 (model7Cover.dC A3 0 f) ≠ rzCover.dC (Rep.res (MonoidHom.id _) A3) 0 (rzRefineFlip.untw A3 0 f) := by
  refine rzRefineFlip.untw_not_cochain A3 rzStr ![1, 0] ?_
  show stdRep3 1 ![1, 0] ≠ stdRep3 ((rzRefineFlip.lab ⟨5, by decide⟩).1⁻¹ * 1 * (rzRefineFlip.lab ⟨0, by decide⟩).1) ![1, 0]
  rw [rzRefineFlip_lab0, rzRefineFlip_lab5]
  have hneg : ((M22.toSL M22.negI : SL(2, ℤ)) : Matrix (Fin 2) (Fin 2) ℤ) = !![-1, 0; 0, -1] := by
    rw [M22.toSL_coe (by decide)]
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  simp only [inv_one, one_mul, map_one, Module.End.one_apply]
  show ![1, 0] ≠ Matrix.toLin' (((M22.toSL M22.negI : SL(2, ℤ)) : Matrix (Fin 2) (Fin 2) ℤ).map
    (Int.castRingHom (ZMod 3))) ![1, 0]
  rw [hneg, Matrix.toLin'_apply]
  decide

end GL2.MP
