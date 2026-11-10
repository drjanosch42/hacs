import GL2.MP.Comparison
import GL2.MP.Models

/-!
# MP-L2, row L0b (slot 22): a decided descriptor is a cover datum; `M₇`, `R_ℤ` as MP covers

[GT] `gt_part_MP.tex`, rows `def:mp-datum-cover`, `thm:mp-datum-hyp`, `thm:mp-models-comparison`,
`thm:mp-rz-counts`, backing [C] `def:mptri`, `def:mpset`, `thm:mpcomparison` (5) (hypothesis form) and the counts of
[C] `rmk:mpwhat`.

For a well-formed `MPDatum` (`wfB = true`): the vertex-orbit stabilisers are subgroups `Γ_o ≤ SL₂(ℤ)` (`sub`), the
vertices of `Y` are `Σ o, SL₂(ℤ) ⧸ Γ_o` (a `g ι(o)` is the coset `g Γ_o`), and the maximal simplices are the point sets of
the stored lists (`cover`).  With (IO) (`ioB`) and `dim ≤ 2` (`dimB`) this is a `Cover.Hyp` (`hyp`; the order is the orbit
index).  The decidable arrows are the arrows of the cover (`mem_Ar_iff`), so the C++/Lean counts are the cardinalities of
the abstract MP sets (`card_mpSet_zero`, `card_mpSet_one`): `|S_0| = 7`, `|S_1| = 172` on `R_ℤ`, `5, 132` on `M₇`.
`rz_comparison`, `model7_comparison`: [C] `thm:mpcomparison` for `R_ℤ` and `M₇`, with (AC) over `k` as the remaining
hypothesis.
-/

namespace GL2.MP

open GL2.TauChk
open scoped MatrixGroups

namespace MPDatum

variable (D : MPDatum)

/-- **`dim ≤ 2`**: every maximal simplex has at most three vertices. -/
def dimB : Bool := D.top.all (·.length ≤ 3)

/-- The well-formedness of `wfB`, as propositions. -/
structure WF : Prop where
  stab_det : ∀ o < D.stab.length, ∀ h ∈ D.st o, h.det = 1
  stab_one : ∀ o < D.stab.length, M22.I ∈ D.st o
  stab_mul : ∀ o < D.stab.length, ∀ h ∈ D.st o, ∀ h' ∈ D.st o, h * h' ∈ D.st o
  stab_adj : ∀ o < D.stab.length, ∀ h ∈ D.st o, h.adj ∈ D.st o
  pts_orb : ∀ p < D.pts.length, D.orb p < D.stab.length
  pts_det : ∀ p < D.pts.length, (D.gp p).det = 1
  top_pts : ∀ σ < D.top.length, ∀ p ∈ D.tv σ, p < D.pts.length
  top_ne : ∀ σ < D.top.length, D.tv σ ≠ []

theorem wf (h : D.wfB = true) : D.WF := by
  simp only [wfB, stabGroupB, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq,
    Bool.not_eq_true', List.isEmpty_eq_false_iff] at h
  obtain ⟨⟨⟨hst, hpts⟩, -⟩, htop⟩ := h
  have st_eq : ∀ o (ho : o < D.stab.length), D.st o = D.stab[o] := fun o ho => List.getD_eq_getElem _ _ ho
  have pt_eq : ∀ p (hp : p < D.pts.length), D.pts.getD p (0, M22.I) = D.pts[p] := fun p hp => List.getD_eq_getElem _ _ hp
  have tv_eq : ∀ σ (hσ : σ < D.top.length), D.tv σ = D.top[σ] := fun σ hσ => List.getD_eq_getElem _ _ hσ
  refine ⟨fun o ho => ?_, fun o ho => ?_, fun o ho => ?_, fun o ho => ?_, fun p hp => ?_, fun p hp => ?_,
    fun σ hσ => ?_, fun σ hσ => ?_⟩
  · rw [st_eq o ho]; exact (hst _ (List.getElem_mem ho)).1.1.1
  · rw [st_eq o ho]; exact (hst _ (List.getElem_mem ho)).1.1.2
  · rw [st_eq o ho]; exact (hst _ (List.getElem_mem ho)).1.2
  · rw [st_eq o ho]; exact (hst _ (List.getElem_mem ho)).2
  · show (D.pts.getD p (0, M22.I)).1 < _; rw [pt_eq p hp]; exact (hpts _ (List.getElem_mem hp)).1
  · show (D.pts.getD p (0, M22.I)).2.det = 1; rw [pt_eq p hp]; exact (hpts _ (List.getElem_mem hp)).2
  · rw [tv_eq σ hσ]; exact (htop _ (List.getElem_mem hσ)).2
  · rw [tv_eq σ hσ]; exact (htop _ (List.getElem_mem hσ)).1

variable {D} (hW : D.WF)

/-- **The stabiliser** `Γ_o ≤ SL₂(ℤ)` of a vertex-orbit representative. -/
def sub (o : Fin D.stab.length) : Subgroup SL(2, ℤ) where
  carrier := {g | ∃ h ∈ D.st o, M22.toSL h = g}
  one_mem' := ⟨M22.I, hW.stab_one o o.2, M22.toSL_one⟩
  mul_mem' := by
    rintro _ _ ⟨h, hh, rfl⟩ ⟨h', hh', rfl⟩
    exact ⟨h * h', hW.stab_mul o o.2 h hh h' hh',
      M22.toSL_mul (hW.stab_det o o.2 h hh) (hW.stab_det o o.2 h' hh')⟩
  inv_mem' := by
    rintro _ ⟨h, hh, rfl⟩
    exact ⟨h.adj, hW.stab_adj o o.2 h hh, M22.toSL_adj (hW.stab_det o o.2 h hh)⟩

lemma mem_sub_iff (o : Fin D.stab.length) {m : M22} (hm : m.det = 1) : M22.toSL m ∈ sub hW o ↔ m ∈ D.st o := by
  constructor
  · rintro ⟨h, hh, e⟩
    rwa [M22.toSL_injOn (hW.stab_det o o.2 h hh) hm e] at hh
  · exact fun h => ⟨m, h, rfl⟩

/-- The relation `(o, g) ∼ (o', g')` iff `o = o'` and `g⁻¹ g' ∈ Γ_o`: the same vertex `g ι(o) = g' ι(o')` of `Y`. -/
def ptRel : Setoid (Fin D.stab.length × SL(2, ℤ)) where
  r x y := x.1 = y.1 ∧ x.2⁻¹ * y.2 ∈ sub hW x.1
  iseqv := by
    refine ⟨fun x => ⟨rfl, by rw [inv_mul_cancel]; exact (sub hW x.1).one_mem⟩, ?_, ?_⟩
    · rintro x y ⟨e, h⟩
      refine ⟨e.symm, ?_⟩
      rw [← e]
      have := (sub hW x.1).inv_mem h
      rwa [mul_inv_rev, inv_inv] at this
    · rintro x y z ⟨e, h⟩ ⟨e', h'⟩
      refine ⟨e.trans e', ?_⟩
      rw [← e] at h'
      have := (sub hW x.1).mul_mem h h'
      rwa [mul_assoc, mul_inv_cancel_left] at this

variable (D) in
/-- **The vertices of `Y`**: pairs `(o, g)` up to `ptRel` (`g ι(o)`). -/
def P := Quotient (ptRel hW)

instance : MulAction SL(2, ℤ) (P D hW) where
  smul γ := Quotient.map (fun x => (x.1, γ * x.2)) (fun x y h => ⟨h.1, by
    show (γ * x.2)⁻¹ * (γ * y.2) ∈ sub hW x.1
    rw [mul_inv_rev, mul_assoc, inv_mul_cancel_left]; exact h.2⟩)
  one_smul x := by
    induction x using Quotient.ind with
    | _ x => show Quotient.mk _ (x.1, 1 * x.2) = _; rw [one_mul]
  mul_smul γ δ x := by
    induction x using Quotient.ind with
    | _ x => show Quotient.mk _ (x.1, γ * δ * x.2) = Quotient.mk _ (x.1, γ * (δ * x.2)); rw [mul_assoc]

/-- The class of `(o, g)`. -/
def mkP (x : Fin D.stab.length × SL(2, ℤ)) : P D hW := Quotient.mk (ptRel hW) x

lemma smul_mkP (γ : SL(2, ℤ)) (x : Fin D.stab.length × SL(2, ℤ)) : γ • mkP hW x = mkP hW (x.1, γ * x.2) := rfl

lemma mkP_eq_iff (x y : Fin D.stab.length × SL(2, ℤ)) :
    mkP hW x = mkP hW y ↔ x.1 = y.1 ∧ x.2⁻¹ * y.2 ∈ sub hW x.1 := Quotient.eq

/-- The vertex `g_p ι(o_p)` of a lifted point. -/
def pt (p : Fin D.pts.length) : P D hW := mkP hW (⟨D.orb p, hW.pts_orb p p.2⟩, M22.toSL (D.gp p))

/-- **The MP cover datum** of a well-formed descriptor. -/
def cover : Cover SL(2, ℤ) where
  P := P D hW
  T := Fin D.top.length
  vtx σ := {x | ∃ p : Fin D.pts.length, (p : ℕ) ∈ D.tv σ ∧ x = pt hW p}

/-- **`γ • pt w = pt p` iff `maps γ w p`** (`γ ∈ g_p Γ_o g_w⁻¹`, `o = o_p = o_w`). -/
lemma smul_pt_eq_iff (γ : M22) (hγ : γ.det = 1) (w p : Fin D.pts.length) :
    M22.toSL γ • pt hW w = pt hW p ↔ D.maps γ w p = true := by
  have hp := hW.pts_det p p.2
  have hw := hW.pts_det w w.2
  have hm : (M22.adj (D.gp p) * (γ * D.gp w)).det = 1 := by
    rw [M22.det_mul, M22.det_adj, M22.det_mul, hp, hγ, hw]; norm_num
  have hsl : M22.toSL (M22.adj (D.gp p) * (γ * D.gp w)) = (M22.toSL (D.gp p))⁻¹ * (M22.toSL γ * M22.toSL (D.gp w)) := by
    rw [M22.toSL_mul (by rw [M22.det_adj]; exact hp) (by rw [M22.det_mul, hγ, hw]; norm_num), M22.toSL_adj hp,
      M22.toSL_mul hγ hw]
  simp only [maps, samePt, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  rw [pt, pt, smul_mkP, mkP_eq_iff]
  dsimp only
  constructor
  · rintro ⟨e, h⟩
    refine ⟨(congrArg Fin.val e).symm, ?_⟩
    rw [e] at h
    have h' := (sub hW _).inv_mem h
    rw [mul_inv_rev, inv_inv, ← hsl] at h'
    exact (mem_sub_iff hW _ hm).1 h'
  · rintro ⟨e, h⟩
    have e' : (⟨D.orb w, hW.pts_orb w w.2⟩ : Fin D.stab.length) = ⟨D.orb p, hW.pts_orb p p.2⟩ := Fin.ext e.symm
    refine ⟨e', ?_⟩
    have h' := (mem_sub_iff hW ⟨D.orb p, hW.pts_orb p p.2⟩ hm).2 h
    rw [hsl] at h'
    have h'' := (sub hW _).inv_mem h'
    rw [mul_inv_rev, inv_inv] at h''
    rw [e']; exact h''

/-! ### `SL₂(ℤ)` elements as integer matrices -/

/-- The entries of an element of `SL₂(ℤ)`. -/
def ofSL (g : SL(2, ℤ)) : M22 := ⟨g 0 0, g 0 1, g 1 0, g 1 1⟩

lemma ofSL_det (g : SL(2, ℤ)) : (ofSL g).det = 1 := by
  have := g.2
  rw [Matrix.det_fin_two] at this
  simpa [ofSL, M22.det] using this

lemma toSL_ofSL (g : SL(2, ℤ)) : M22.toSL (ofSL g) = g := by
  apply Subtype.ext
  rw [M22.toSL_coe (ofSL_det g)]
  ext i j; fin_cases i <;> fin_cases j <;> rfl

/-! ### The hypotheses (IO), finiteness, `dim ≤ 2` -/

/-- The orbit index of a vertex (the `Γ`-invariant order of (IO)). -/
def ordP : P D hW → ℕ := Quotient.lift (fun x : Fin D.stab.length × SL(2, ℤ) => (x.1 : ℕ)) (fun _ _ h => congrArg Fin.val h.1)

lemma ordP_pt (p : Fin D.pts.length) : ordP hW (pt hW p) = D.orb p := rfl

lemma tv_mem (σ : Fin D.top.length) : D.tv σ ∈ D.top := by
  rw [tv, List.getD_eq_getElem _ _ σ.2]; exact List.getElem_mem σ.2

open Classical in
/-- **(IO), finiteness and `dim ≤ 2`** for the cover of a decided descriptor ([C] `def:mptri` (3)). -/
noncomputable def hyp (hio : D.ioB = true) (hdim : D.dimB = true) : (cover hW).Hyp where
  ord := ordP hW
  ord_smul g x := by
    induction x using Quotient.ind with
    | _ x => rfl
  fin σ := (Set.finite_range (pt hW)).subset (by rintro _ ⟨p, -, rfl⟩; exact ⟨p, rfl⟩)
  inj σ := by
    change Fin D.top.length at σ
    rintro _ ⟨p, hp, rfl⟩ _ ⟨q, hq, rfl⟩ e
    have hn : ((D.tv σ).map D.orb).Nodup := by
      have := (List.all_eq_true.1 hio) _ (tv_mem σ)
      simpa using this
    have : (p : ℕ) = q := List.inj_on_of_nodup_map hn hp hq e
    rw [Fin.ext this]
  dim σ := by
    change Fin D.top.length at σ
    let S : Finset (Fin D.pts.length) := Finset.univ.filter fun p => (p : ℕ) ∈ D.tv σ
    have hsub : (cover hW).vtx σ = (pt hW) '' (S : Set (Fin D.pts.length)) := by
      ext x; constructor
      · rintro ⟨p, hp, rfl⟩; exact ⟨p, by simpa [S] using hp, rfl⟩
      · rintro ⟨p, hp, rfl⟩; exact ⟨p, by simpa [S] using hp, rfl⟩
    have hS : S.card ≤ (D.tv σ).length := by
      calc S.card ≤ (D.tv σ).toFinset.card :=
            Finset.card_le_card_of_injOn (fun p => (p : ℕ)) (fun p hp => by simpa [S] using hp)
              (fun a _ b _ e => Fin.ext e)
        _ ≤ (D.tv σ).length := List.toFinset_card_le _
    have hl : (D.tv σ).length ≤ 3 := by
      have := (List.all_eq_true.1 hdim) _ (tv_mem σ)
      simpa using this
    rw [hsub]
    calc ((pt hW) '' (S : Set (Fin D.pts.length))).ncard ≤ (S : Set (Fin D.pts.length)).ncard := Set.ncard_image_le
      _ = S.card := Set.ncard_coe_finset S
      _ ≤ 3 := hS.trans hl

/-! ### The arrows of the cover are the decided arrows -/

/-- **`Ar(σ, σ')` of the cover is `arr σ σ'`** ([C] `rmk:mpwhat` (1)): `γ` carries a vertex of `ι(σ')` to a vertex of
`ι(σ)` iff `γ ∈ ⋃ g_p Γ_o g_w⁻¹`. -/
theorem mem_Ar_iff (σ σ' : Fin D.top.length) (γ : M22) (hγ : γ.det = 1) :
    (∃ q : P D hW, q ∈ (cover hW).vtx σ ∧ (M22.toSL γ)⁻¹ • q ∈ (cover hW).vtx σ') ↔ γ ∈ D.arr σ σ' := by
  simp only [arr, List.mem_dedup, List.mem_flatMap]
  constructor
  · rintro ⟨_, ⟨p, hp, rfl⟩, ⟨w, hw, hpw⟩⟩
    change (M22.toSL γ)⁻¹ • pt hW p = pt hW w at hpw
    have hmap : D.maps γ w p = true := by
      rw [← smul_pt_eq_iff hW γ hγ w p, ← hpw, smul_inv_smul]
    simp only [maps, samePt, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hmap
    obtain ⟨e, hst⟩ := hmap
    refine ⟨p, hp, w, hw, ?_⟩
    rw [if_pos e]
    refine List.mem_map.2 ⟨_, hst, ?_⟩
    have hgp := hW.pts_det p p.2
    have hgw := hW.pts_det w w.2
    have h1 : ((D.gp p).adj * (γ * D.gp w)).det = 1 := by
      rw [M22.det_mul, M22.det_adj, M22.det_mul, hgp, hγ, hgw]; norm_num
    apply M22.toSL_injOn (by rw [M22.det_mul, M22.det_mul, hgp, h1, M22.det_adj, hgw]; norm_num) hγ
    rw [M22.toSL_mul (by rw [M22.det_mul, hgp, h1]; norm_num) (by rw [M22.det_adj]; exact hgw),
      M22.toSL_mul hgp h1, M22.toSL_adj hgw, M22.toSL_mul (by rw [M22.det_adj]; exact hgp)
        (by rw [M22.det_mul, hγ, hgw]; norm_num), M22.toSL_adj hgp, M22.toSL_mul hγ hgw]
    group
  · rintro ⟨p, hp, w, hw, hmem⟩
    have hpl : p < D.pts.length := hW.top_pts σ σ.2 p hp
    have hwl : w < D.pts.length := hW.top_pts σ' σ'.2 w hw
    split_ifs at hmem with e
    · obtain ⟨h, hh, rfl⟩ := List.mem_map.1 hmem
      refine ⟨pt hW ⟨p, hpl⟩, ⟨⟨p, hpl⟩, hp, rfl⟩, ⟨w, hwl⟩, hw, ?_⟩
      show (M22.toSL (D.gp p * h * (D.gp w).adj))⁻¹ • pt hW ⟨p, hpl⟩ = pt hW ⟨w, hwl⟩
      have hgp := hW.pts_det p hpl
      have hgw := hW.pts_det w hwl
      have hhd := hW.stab_det _ (hW.pts_orb p hpl) h hh
      have hγ' : (D.gp p * h * (D.gp w).adj).det = 1 := by
        rw [M22.det_mul, M22.det_mul, hgp, hhd, M22.det_adj, hgw]; norm_num
      rw [inv_smul_eq_iff]
      symm
      rw [smul_pt_eq_iff hW _ hγ' ⟨w, hwl⟩ ⟨p, hpl⟩]
      simp only [maps, samePt, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
      refine ⟨e, ?_⟩
      have : (D.gp p).adj * (D.gp p * h * (D.gp w).adj * D.gp w) = h := by
        apply M22.toSL_injOn _ hhd
        · rw [M22.toSL_mul (by rw [M22.det_adj]; exact hgp) (by rw [M22.det_mul, hγ', hgw]; norm_num),
            M22.toSL_adj hgp, M22.toSL_mul hγ' hgw, M22.toSL_mul (by rw [M22.det_mul, hgp, hhd]; norm_num)
              (by rw [M22.det_adj]; exact hgw), M22.toSL_adj hgw, M22.toSL_mul hgp hhd]
          group
        · rw [M22.det_mul, M22.det_adj, M22.det_mul, hγ', hgp, hgw]; norm_num
      show (D.gp p).adj * (D.gp p * h * (D.gp w).adj * D.gp w) ∈ D.st (D.orb p)
      rw [this]; exact hh
    · exact absurd hmem (List.not_mem_nil)

/-! ### The counts `|S_0|`, `|S_1|` of the abstract MP set -/

lemma exists_vtx (σ : Fin D.top.length) : ∃ q : P D hW, q ∈ (cover hW).vtx σ := by
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ (hW.top_ne σ σ.2)
  exact ⟨pt hW ⟨p, hW.top_pts σ σ.2 p hp⟩, ⟨_, hp, rfl⟩⟩

/-- **`S_0` is the set of maximal-simplex representatives.** -/
def mpSetZeroEquiv : (cover hW).MPSet 0 ≃ Fin D.top.length where
  toFun s := (s.1 0).2
  invFun σ := ⟨fun _ => (1, σ), by
    obtain ⟨q, hq⟩ := exists_vtx hW σ
    refine ⟨q, fun _ => ?_⟩
    show (1 : SL(2, ℤ))⁻¹ • (q : (cover hW).P) ∈ (cover hW).vtx σ
    rw [inv_one, one_smul]; exact hq, rfl⟩
  left_inv s := by
    apply Subtype.ext; funext i
    rw [Fin.eq_zero i]
    exact Prod.ext s.2.2.symm rfl
  right_inv σ := rfl

/-- **`|S_0|` = the number of maximal-simplex representatives.** -/
theorem card_mpSet_zero : Nat.card ((cover hW).MPSet 0) = D.top.length := by
  rw [Nat.card_congr (mpSetZeroEquiv hW), Nat.card_eq_fintype_card, Fintype.card_fin]

/-- The arrow set `Ar(σ, σ')` of the cover. -/
def ArS (σ σ' : Fin D.top.length) : Type :=
  {γ : SL(2, ℤ) // ∃ q : P D hW, q ∈ (cover hW).vtx σ ∧ γ⁻¹ • q ∈ (cover hW).vtx σ'}

lemma arr_det (hW : D.WF) {σ σ' : Fin D.top.length} {m : M22} (hm : m ∈ D.arr σ σ') : m.det = 1 := by
  simp only [arr, List.mem_dedup, List.mem_flatMap] at hm
  obtain ⟨p, hp, w, hw, hmem⟩ := hm
  have hpl : p < D.pts.length := hW.top_pts σ σ.2 p hp
  have hwl : w < D.pts.length := hW.top_pts σ' σ'.2 w hw
  split_ifs at hmem with e
  · obtain ⟨h, hh, rfl⟩ := List.mem_map.1 hmem
    rw [M22.det_mul, M22.det_mul, hW.pts_det p hpl, hW.stab_det _ (hW.pts_orb p hpl) h hh, M22.det_adj,
      hW.pts_det w hwl]; norm_num
  · exact absurd hmem List.not_mem_nil

/-- **`Ar(σ, σ')` of the cover is the decided list `arr σ σ'`** (as a type). -/
def arSEquiv (σ σ' : Fin D.top.length) : ArS hW σ σ' ≃ {m : M22 // m ∈ D.arr σ σ'} where
  toFun γ := ⟨ofSL γ.1, (mem_Ar_iff hW σ σ' (ofSL γ.1) (ofSL_det _)).1 (by rw [toSL_ofSL]; exact γ.2)⟩
  invFun m := ⟨M22.toSL m.1, (mem_Ar_iff hW σ σ' m.1 (arr_det hW m.2)).2 m.2⟩
  left_inv γ := Subtype.ext (toSL_ofSL γ.1)
  right_inv m := Subtype.ext (M22.toSL_injOn (ofSL_det _) (arr_det hW m.2) (toSL_ofSL _))

lemma card_arS (σ σ' : Fin D.top.length) : Nat.card (ArS hW σ σ') = (D.arr σ σ').length := by
  classical
  rw [Nat.card_congr ((arSEquiv hW σ σ').trans (Equiv.subtypeEquivRight fun m => List.mem_toFinset.symm)),
    Nat.card_eq_fintype_card, Fintype.card_coe, List.toFinset_card_of_nodup (by unfold arr; exact List.nodup_dedup _)]

instance (σ σ' : Fin D.top.length) : Finite (ArS hW σ σ') := Finite.of_equiv _ (arSEquiv hW σ σ').symm

/-- `S_1` is the set of arrows `(σ ←γ— σ')`. -/
def mpSetOneEquiv : (cover hW).MPSet 1 ≃ Σ σ : Fin D.top.length, Σ σ' : Fin D.top.length, ArS hW σ σ' where
  toFun s := ⟨(s.1 0).2, (s.1 1).2, ⟨(s.1 1).1, by
    obtain ⟨q, hq⟩ := s.2.1
    refine ⟨q, ?_, hq 1⟩
    have := hq 0
    rwa [s.2.2, inv_one, one_smul] at this⟩⟩
  invFun x := ⟨![(1, x.1), (x.2.2.1, x.2.1)], by
    obtain ⟨q, hq0, hq1⟩ := x.2.2.2
    refine ⟨q, fun i => ?_⟩
    fin_cases i
    · show (1 : SL(2, ℤ))⁻¹ • (q : (cover hW).P) ∈ (cover hW).vtx x.1
      rw [inv_one, one_smul]; exact hq0
    · show (x.2.2.1)⁻¹ • (q : (cover hW).P) ∈ (cover hW).vtx x.2.1
      exact hq1, rfl⟩
  left_inv s := by
    apply Subtype.ext; funext i
    fin_cases i
    · exact Prod.ext s.2.2.symm rfl
    · rfl
  right_inv x := rfl

/-- **`|S_1| = ∑_{σ, σ'} |Ar(σ, σ')|`** (all strings, degenerate included). -/
theorem card_mpSet_one :
    Nat.card ((cover hW).MPSet 1) = ∑ σ : Fin D.top.length, ∑ σ' : Fin D.top.length, (D.arr σ σ').length := by
  rw [Nat.card_congr (mpSetOneEquiv hW), Nat.card_sigma]
  refine Finset.sum_congr rfl fun σ _ => ?_
  rw [Nat.card_sigma]
  exact Finset.sum_congr rfl fun σ' _ => card_arS hW σ σ'

end MPDatum

/-! ### `M₇` and `R_ℤ` as MP covers -/

theorem rz_wf : rzMP.WF := rzMP.wf rz_rules.1
theorem rz_dimB : rzMP.dimB = true := by decide +kernel
theorem model7_wf : model7MP.WF := model7MP.wf model7_rules.1
theorem model7_dimB : model7MP.dimB = true := by decide +kernel

/-- **`R_ℤ` as an MP cover datum** (from the C++ dump). -/
noncomputable def rzCover : Cover SL(2, ℤ) := MPDatum.cover rz_wf
/-- **`M₇` as an MP cover datum** (from the dump that defines `model7`). -/
noncomputable def model7Cover : Cover SL(2, ℤ) := MPDatum.cover model7_wf

/-- (IO), finiteness, `dim ≤ 2` for `R_ℤ`. -/
noncomputable def rzHyp : rzCover.Hyp := MPDatum.hyp rz_wf rz_rules.2.1 rz_dimB
/-- (IO), finiteness, `dim ≤ 2` for `M₇`. -/
noncomputable def model7Hyp : model7Cover.Hyp := MPDatum.hyp model7_wf model7_rules.2.1 model7_dimB

/-- **[C] `thm:mpcomparison` for `R_ℤ`** (unnormalised cochains), with (AC) over `k` for the vertex scheme of the dump as
the remaining hypothesis: `H^n(SL₂(ℤ); A) ≅ H^n(C^•(S(R_ℤ); A))` for every ring `k`, every `A`, every `n`. -/
noncomputable def rz_comparison {k : Type} [CommRing k] (hAC : (Cover.yCx rzHyp k).toCx.Acyclic)
    (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) : groupCohomology A n ≅ (rzCover.mpComplex A).homology n :=
  Cover.mpComparison rzHyp A hAC n

/-- **[C] `thm:mpcomparison` for `M₇`** (unnormalised cochains), with (AC) over `k` as the remaining hypothesis. -/
noncomputable def model7_comparison {k : Type} [CommRing k] (hAC : (Cover.yCx model7Hyp k).toCx.Acyclic)
    (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) : groupCohomology A n ≅ (model7Cover.mpComplex A).homology n :=
  Cover.mpComparison model7Hyp A hAC n

/-- **The abstract MP sets of `R_ℤ` and `M₇` have the C++ counts** ([C] `rmk:mpwhat`, all strings):
`|S_0|, |S_1| = 7, 172` on `R_ℤ` and `5, 132` on `M₇`. -/
theorem mpSet_counts :
    Nat.card (rzCover.MPSet 0) = 7 ∧ Nat.card (rzCover.MPSet 1) = 172 ∧
      Nat.card (model7Cover.MPSet 0) = 5 ∧ Nat.card (model7Cover.MPSet 1) = 132 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [rzCover, MPDatum.card_mpSet_zero]; decide +kernel
  · rw [rzCover, MPDatum.card_mpSet_one]; decide +kernel
  · rw [model7Cover, MPDatum.card_mpSet_zero]; decide +kernel
  · rw [model7Cover, MPDatum.card_mpSet_one]; decide +kernel

end GL2.MP
