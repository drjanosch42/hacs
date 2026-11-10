import GL2.MP.ACBridge

/-!
# MP-L2, open item 1 (slot 22): certified table chain maps between `X_bulk` complexes

[GT] `gt_part_MP.tex`, rows `def:mp-tab`, `thm:mp-tab-same`, `def:mp-tab-map`, backing [C] `thm:mpcomparison` (5) for
`R_ℤ` (the subdivision argument).

**Formal sums.**  A term `(c, g, τ)` (`Tm`) stands for `c · [γ g, τ]` in `C_q(X_bulk)` of a dump `D`; a list of terms is a
formal sum, `val q γ L = ∑ c · [γ g, τ]`.  The boundary of a formal sum is a formal sum (`bdL`, faces with transporter `1` and
sign `(-1)^j`), and so is the image under a **table map** (`appL`): a table `T σ` gives the image `T σ` of `[1, σ]`, and
`Fmap` extends it equivariantly (`[γ, σ] ↦ val γ (T σ)`), well defined on the oriented relation when the table is
transporter-compatible (`TrChk`).

**Deciding equalities** (`Same`, `same_sound`).  When the dump is bridged to an MP descriptor (`ACBridge.Checks`), the sign
bridge `Ξ` is injective and sends `c · [γ g, τ]` to `c · eps τ · e_{γ g · srt τ}`; two formal sums have the same value if,
for every term `z` occurring, the summed coefficients of the terms whose ordered simplex `g · srt τ` equals that of `z` agree
(`coefSum`).  Equality of ordered simplices is decided slot by slot by `MPDatum.maps` (`ΘeqB`).  So every identity between
table maps is a finite check, decided by the kernel.
-/

namespace GL2.MP

open GL2.TauChk GL2.Model GL2.Model.XBulk Finsupp
open scoped MatrixGroups

namespace TabMap

/-- A term `c · [g, τ]`. -/
abbrev Tm := ℤ × M22 × ℕ

/-! ### Formal sums -/

section Val

variable (D : TauData) (hX : D.XAx)

/-- `[γ, τ]`, or `0` if `τ` is not a simplex of degree `q`. -/
noncomputable def gq (q : ℤ) (γ : SL(2, ℤ)) (τ : ℕ) : chains (D.toXBulk hX) q :=
  if h : τ ∈ D.idxF q then genQ (D.toXBulk hX) q (γ, ⟨τ, h⟩) else 0

/-- **The value** `∑ c · [γ g, τ]` of a formal sum. -/
noncomputable def val (q : ℤ) (γ : SL(2, ℤ)) (L : List Tm) : chains (D.toXBulk hX) q :=
  (L.map fun x => x.1 • gq D hX q (γ * M22.toSL x.2.1) x.2.2).sum

lemma val_nil (q : ℤ) (γ : SL(2, ℤ)) : val D hX q γ [] = 0 := rfl

lemma val_cons (q : ℤ) (γ : SL(2, ℤ)) (x : Tm) (L : List Tm) :
    val D hX q γ (x :: L) = x.1 • gq D hX q (γ * M22.toSL x.2.1) x.2.2 + val D hX q γ L := by
  simp [val]

lemma val_append (q : ℤ) (γ : SL(2, ℤ)) (L L' : List Tm) : val D hX q γ (L ++ L') = val D hX q γ L + val D hX q γ L' := by
  simp [val]

lemma val_flatMap (q : ℤ) (γ : SL(2, ℤ)) {α : Type} (l : List α) (f : α → List Tm) :
    val D hX q γ (l.flatMap f) = (l.map fun a => val D hX q γ (f a)).sum := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.flatMap_cons, val_append, ih, List.map_cons, List.sum_cons]

/-- Scale a formal sum. -/
def scaleL (c : ℤ) (L : List Tm) : List Tm := L.map fun x => (c * x.1, x.2.1, x.2.2)

/-- Translate a formal sum by `g`. -/
def transL (g : M22) (L : List Tm) : List Tm := L.map fun x => (x.1, g * x.2.1, x.2.2)

/-- `-L`. -/
def negL (L : List Tm) : List Tm := scaleL (-1) L

lemma val_scaleL (q : ℤ) (γ : SL(2, ℤ)) (c : ℤ) (L : List Tm) : val D hX q γ (scaleL c L) = c • val D hX q γ L := by
  induction L with
  | nil => simp [val, scaleL]
  | cons x L ih =>
    simp only [scaleL, List.map_cons] at ih ⊢
    rw [val_cons, val_cons, ih, smul_add, mul_smul]

lemma val_negL (q : ℤ) (γ : SL(2, ℤ)) (L : List Tm) : val D hX q γ (negL L) = -val D hX q γ L := by
  rw [negL, val_scaleL, neg_one_smul]

/-- All matrices of a formal sum have determinant one. -/
def DetOne (L : List Tm) : Prop := ∀ x ∈ L, x.2.1.det = 1

lemma val_transL (q : ℤ) (γ : SL(2, ℤ)) {g : M22} (hg : g.det = 1) {L : List Tm} (hL : DetOne L) :
    val D hX q γ (transL g L) = val D hX q (γ * M22.toSL g) L := by
  induction L with
  | nil => simp [val, transL]
  | cons x L ih =>
    simp only [transL, List.map_cons] at ih ⊢
    rw [val_cons, val_cons, ih (fun y hy => hL y (List.mem_cons_of_mem _ hy)),
      M22.toSL_mul hg (hL x (List.mem_cons_self ..)), mul_assoc]

lemma gq_smul (q : ℤ) (γ δ : SL(2, ℤ)) (τ : ℕ) : δ • gq D hX q γ τ = gq D hX q (δ * γ) τ := by
  unfold gq
  split_ifs
  · rw [smul_genQ]
  · rw [smul_zero]

lemma val_smul (q : ℤ) (γ δ : SL(2, ℤ)) (L : List Tm) : δ • val D hX q γ L = val D hX q (δ * γ) L := by
  induction L with
  | nil => simp [val]
  | cons x L ih => rw [val_cons, val_cons, smul_add, ih, smul_comm, gq_smul, mul_assoc]

/-- The formal sum `[1, σ]`. -/
def unitL (σ : ℕ) : List Tm := [(1, M22.I, σ)]

lemma genQ_eq_val (q : ℤ) (a : SL(2, ℤ) × (D.toXBulk hX).V q) :
    genQ (D.toXBulk hX) q a = val D hX q a.1 (unitL a.2.1) := by
  rw [unitL, val_cons, val_nil, add_zero, one_smul, show M22.toSL M22.I = 1 from M22.toSL_one, mul_one, gq, dif_pos a.2.2]
  rfl

lemma listSum_range {A : Type} [AddCommMonoid A] (f : ℕ → A) (n : ℕ) :
    ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := by
  induction n with
  | zero => simp
  | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

/-- The faces of a formal sum of degree `n + 1`: `∑ c (-1)^j [g, ∂_j τ]`. -/
def bdL (n : ℕ) (L : List Tm) : List Tm :=
  L.flatMap fun x => (List.range (n + 2)).map fun j => (x.1 * sgnPow j, x.2.1, D.face (n + 1) x.2.2 j)

lemma bd_genQ (n : ℕ) (a : SL(2, ℤ) × (D.toXBulk hX).V ((n : ℤ) + 1)) :
    bd (D.toXBulk hX) (n : ℤ) (genQ (D.toXBulk hX) ((n : ℤ) + 1) a) =
      val D hX (n : ℤ) a.1 ((List.range (n + 2)).map fun j => (sgnPow j, M22.I, D.face (n + 1) a.2.1 j)) := by
  rw [genQ, bd_mk, bdF_gen, Units.val_one, one_smul, quot_mk_sum, val, List.map_map, listSum_range, Finset.sum_range]
  have hn := ACBridge.nf_nat n
  refine Fintype.sum_equiv (finCongr hn) _ _ fun j => ?_
  show (Submodule.Quotient.mk (gen (a.1 * 1, ⟨D.face ((n : ℤ) + 1).toNat a.2.1 j.1, TauData.face_mem hX a.2 j⟩,
    unitOf (sgnPow j.1))) : chains (D.toXBulk hX) (n : ℤ)) = sgnPow j.1 • gq D hX (n : ℤ) (a.1 * M22.toSL M22.I) _
  have hmem : D.face (n + 1) a.2.1 j.1 ∈ D.idxF (n : ℤ) := TauData.face_mem hX a.2 j
  rw [gq]
  split_ifs with h
  swap; · exact absurd hmem h
  rw [show M22.toSL M22.I = 1 from M22.toSL_one, genQ, ← Submodule.Quotient.mk_smul]
  congr 1
  simp only [gen, smul_single, smul_eq_mul, mul_one]
  rw [unitOf_val (sgnPow_cases _), Units.val_one, mul_one]
  rfl

lemma smul_val_map (q : ℤ) (γ : SL(2, ℤ)) (g : M22) (c : ℤ) {α : Type} (l : List α) (sg : α → ℤ) (f : α → ℕ) :
    c • val D hX q (γ * M22.toSL g) (l.map fun j => (sg j, M22.I, f j)) =
      val D hX q γ (l.map fun j => (c * sg j, g, f j)) := by
  induction l with
  | nil => simp [val]
  | cons j l ih =>
    rw [List.map_cons, List.map_cons, val_cons, val_cons, smul_add, ih, mul_smul,
      show M22.toSL M22.I = 1 from M22.toSL_one, mul_one]

lemma bd_val' (n : ℕ) (γ : SL(2, ℤ)) (L : List Tm) (hL : ∀ x ∈ L, x.2.2 ∈ D.idxF ((n : ℤ) + 1)) :
    bd (D.toXBulk hX) (n : ℤ) (val D hX ((n : ℤ) + 1) γ L) = val D hX (n : ℤ) γ (bdL D n L) := by
  induction L with
  | nil => simp [val, bdL]
  | cons x L ih =>
    rw [val_cons, map_add, map_zsmul, bdL, List.flatMap_cons, val_append, ← bdL,
      ih (fun y hy => hL y (List.mem_cons_of_mem _ hy))]
    congr 1
    rw [gq, dif_pos (hL x (List.mem_cons_self ..)), bd_genQ]
    dsimp only
    rw [smul_val_map]

/-- **The boundary of a formal sum** is the formal sum of the faces (degree `q = n`, `q + 1 = n + 1`). -/
lemma bd_val (q : ℤ) (n : ℕ) (hq : q = n) (γ : SL(2, ℤ)) (L : List Tm) (hL : ∀ x ∈ L, x.2.2 ∈ D.idxL ((n + 1 : ℕ) : ℤ)) :
    bd (D.toXBulk hX) q (val D hX (q + 1) γ L) = val D hX q γ (bdL D n L) := by
  subst hq
  exact bd_val' D hX n γ L fun x hx => (D.mem_idxF).2 (by have := hL x hx; push_cast at this; exact this)

/-- The augmentation of a formal sum of degree `0`. -/
def augL (L : List Tm) : ℤ := (L.map (·.1)).sum

lemma aug_val (γ : SL(2, ℤ)) (L : List Tm) (hL : ∀ x ∈ L, x.2.2 ∈ D.idxF 0) :
    aug (D.toXBulk hX) (val D hX 0 γ L) = augL L := by
  induction L with
  | nil => simp [val, augL]
  | cons x L ih =>
    rw [val_cons, map_add, map_zsmul, ih (fun y hy => hL y (List.mem_cons_of_mem _ hy)), gq,
      dif_pos (hL x (List.mem_cons_self ..)), aug_genQ]
    simp only [augL, List.map_cons, List.sum_cons, smul_eq_mul, mul_one]

end Val

/-! ### Deciding equalities of formal sums through the sign bridge -/

/-- **A dump bridged to an MP descriptor** on the same labels (the hypotheses of `ACBridge.yCx_acyclic_of_bridge`). -/
structure Br where
  D : TauData
  M : MPDatum
  hX : D.XAx
  hW : M.WF
  hio : M.ioB = true
  hdim : M.dimB = true
  hC : ACBridge.Checks D M

namespace Br

variable (B : Br)

/-- `[g, τ]` and `[g', τ']` have the same ordered simplex `g · srt τ = g' · srt τ'` (slot by slot). -/
def ΘeqB (s : ℕ) (x z : Tm) : Prop :=
  ∀ j ∈ List.range (s + 1), B.M.maps (z.2.1.adj * x.2.1) (ACBridge.sv B.D B.M s x.2.2 j) (ACBridge.sv B.D B.M s z.2.2 j) = true

instance (s : ℕ) (x z : Tm) : Decidable (B.ΘeqB s x z) := by unfold ΘeqB; infer_instance

/-- The summed coefficient (with orientation signs) of the terms of `L` on the ordered simplex of `z`. -/
def coefSum (s : ℕ) (L : List Tm) (z : Tm) : ℤ :=
  ((L.filter fun x => decide (B.ΘeqB s x z)).map fun x => x.1 * ACBridge.eps B.D B.M s x.2.2).sum

/-- **`Same s L L'`**: the formal sums `L`, `L'` of degree `s` have the same coefficient on every ordered simplex that
occurs (and all their simplices exist, all their matrices have determinant one). -/
def Same (s : ℕ) (L L' : List Tm) : Prop :=
  (∀ x ∈ L ++ L', x.2.2 ∈ B.D.idxL (s : ℤ) ∧ x.2.1.det = 1) ∧ ∀ z ∈ L ++ L', B.coefSum s L z = B.coefSum s L' z

instance (s : ℕ) (L L' : List Tm) : Decidable (B.Same s L L') := by unfold Same; infer_instance

/-- The hypotheses of the bridge, unpacked. -/
lemma hS : ACBridge.chkSimp B.D B.M := B.hC.1
lemma hT : ACBridge.chkTr B.D B.M := B.hC.2.1
lemma hI : ACBridge.chkInj B.D B.M := B.hC.2.2.1
lemma hU : ACBridge.chkSurj B.D B.M := B.hC.2.2.2.1

/-- `Θ` of a term. -/
noncomputable def θ {s : ℕ} (γ : SL(2, ℤ)) (x : Tm) (h : x.2.2 ∈ B.D.idxF (s : ℤ)) : Cover.YS (ACBridge.Hy B.hW B.hio B.hdim) s :=
  ACBridge.Θ B.hX B.hW B.hio B.hdim B.hS (s := s) (γ * M22.toSL x.2.1, ⟨x.2.2, h⟩)

lemma θ_eq_iff {s : ℕ} (γ : SL(2, ℤ)) {x z : Tm} (hx : x.2.2 ∈ B.D.idxF (s : ℤ)) (hz : z.2.2 ∈ B.D.idxF (s : ℤ))
    (dx : x.2.1.det = 1) (dz : z.2.1.det = 1) : B.θ γ x hx = B.θ γ z hz ↔ B.ΘeqB s x z := by
  have hm : (z.2.1.adj * x.2.1).det = 1 := by rw [M22.det_mul, M22.det_adj, dz, dx, one_mul]
  have hsl : M22.toSL (z.2.1.adj * x.2.1) = (γ * M22.toSL z.2.1)⁻¹ * (γ * M22.toSL x.2.1) := by
    rw [M22.toSL_mul (by rw [M22.det_adj]; exact dz) dx, M22.toSL_adj dz]; group
  let vx : (B.D.toXBulk B.hX).V (s : ℤ) := show B.D.Vz (s : ℤ) from ⟨x.2.2, hx⟩
  let vz : (B.D.toXBulk B.hX).V (s : ℤ) := show B.D.Vz (s : ℤ) from ⟨z.2.2, hz⟩
  constructor
  · intro h j hj
    have := congrFun (congrArg Subtype.val h) ⟨j, List.mem_range.1 hj⟩
    rw [θ, θ, ACBridge.Θ_apply, ACBridge.Θ_apply] at this
    dsimp only at this
    exact (MPDatum.smul_pt_eq_iff B.hW _ hm (ACBridge.vt B.hX B.hS vx ⟨j, List.mem_range.1 hj⟩)
      (ACBridge.vt B.hX B.hS vz ⟨j, List.mem_range.1 hj⟩)).1 (by rw [hsl, mul_smul, inv_smul_eq_iff]; exact this)
  · intro h
    apply Subtype.ext
    funext j
    rw [θ, θ, ACBridge.Θ_apply, ACBridge.Θ_apply]
    have := (MPDatum.smul_pt_eq_iff B.hW _ hm (ACBridge.vt B.hX B.hS vx j) (ACBridge.vt B.hX B.hS vz j)).2
      (h j (List.mem_range.2 j.2))
    rw [hsl, mul_smul] at this
    exact inv_smul_eq_iff.1 this

/-- The sign-bridge image of one term. -/
noncomputable def tF {s : ℕ} (γ : SL(2, ℤ)) (x : Tm) : Cover.YS (ACBridge.Hy B.hW B.hio B.hdim) s →₀ ℤ :=
  if h : x.2.2 ∈ B.D.idxF (s : ℤ) then single (B.θ γ x h) (x.1 * ACBridge.eps B.D B.M s x.2.2) else 0

lemma Ξ_term {s : ℕ} (γ : SL(2, ℤ)) (x : Tm) :
    ACBridge.Ξ B.hX B.hW B.hio B.hdim B.hS B.hT s (x.1 • gq B.D B.hX (s : ℤ) (γ * M22.toSL x.2.1) x.2.2) = B.tF γ x := by
  unfold tF gq
  split_ifs with h
  · rw [map_zsmul, ACBridge.Ξ_genQ, smul_single, smul_eq_mul]; rfl
  · rw [smul_zero, map_zero]

lemma Ξ_val {s : ℕ} (γ : SL(2, ℤ)) (L : List Tm) :
    ACBridge.Ξ B.hX B.hW B.hio B.hdim B.hS B.hT s (val B.D B.hX (s : ℤ) γ L) = (L.map (B.tF γ)).sum := by
  rw [val, map_list_sum, List.map_map]
  congr 1
  exact List.map_congr_left fun x _ => B.Ξ_term γ x

lemma sum_ite_filter (L : List Tm) (p : Tm → Prop) [DecidablePred p] (f : Tm → ℤ) :
    (L.map fun x => if p x then f x else 0).sum = ((L.filter fun x => decide (p x)).map f).sum := by
  induction L with
  | nil => rfl
  | cons x L ih =>
    by_cases hp : p x
    · simp [hp, ih]
    · simp [hp, ih]

open Classical in
lemma tF_apply {s : ℕ} (γ : SL(2, ℤ)) (x : Tm) (hx : x.2.2 ∈ B.D.idxF (s : ℤ)) (y : Cover.YS (ACBridge.Hy B.hW B.hio B.hdim) s) :
    (B.tF (s := s) γ x) y = if B.θ γ x hx = y then x.1 * ACBridge.eps B.D B.M s x.2.2 else 0 := by
  rw [tF, dif_pos hx, single_apply]

/-- **Soundness of `Same`**: equal coefficients on the ordered simplices give equal values. -/
theorem same_sound' {s : ℕ} {L L' : List Tm} (h : B.Same s L L') (γ : SL(2, ℤ)) :
    val B.D B.hX (s : ℤ) γ L = val B.D B.hX (s : ℤ) γ L' := by
  obtain ⟨hw, hc⟩ := h
  have inj := fun u v (e : ACBridge.Ξ B.hX B.hW B.hio B.hdim B.hS B.hT s u = ACBridge.Ξ B.hX B.hW B.hio B.hdim B.hS B.hT s v) => by
    have := congrArg (ACBridge.ΞI B.hX B.hW B.hio B.hdim B.hS B.hU s) e
    rwa [ACBridge.ΞI_Ξ B.hX B.hW B.hio B.hdim B.hS B.hT B.hU B.hI, ACBridge.ΞI_Ξ B.hX B.hW B.hio B.hdim B.hS B.hT B.hU B.hI]
      at this
  apply inj
  rw [B.Ξ_val, B.Ξ_val]
  ext y
  show Finsupp.applyAddHom y (List.sum _) = Finsupp.applyAddHom y (List.sum _)
  rw [map_list_sum, map_list_sum, List.map_map, List.map_map]
  have mem : ∀ x ∈ L ++ L', x.2.2 ∈ B.D.idxF (s : ℤ) := fun x hx => (B.D.mem_idxF).2 (hw x hx).1
  -- the coefficient of `y` in a sublist, against a witness `z`
  have key : ∀ (K : List Tm), (∀ x ∈ K, x ∈ L ++ L') → ∀ z (hz : z ∈ L ++ L'), B.θ γ z (mem z hz) = y →
      (K.map ((Finsupp.applyAddHom y) ∘ B.tF (s := s) γ)).sum = B.coefSum s K z := by
    intro K hK z hz hzy
    induction K with
    | nil => rfl
    | cons x K ih =>
      have hxK : x ∈ L ++ L' := hK x (List.mem_cons_self ..)
      rw [List.map_cons, List.sum_cons, ih (fun u hu => hK u (List.mem_cons_of_mem _ hu))]
      unfold coefSum
      rw [List.filter_cons]
      simp only [Function.comp, Finsupp.applyAddHom_apply]
      rw [B.tF_apply γ x (mem x hxK) y]
      have hiff := B.θ_eq_iff γ (mem x hxK) (mem z hz) (hw x hxK).2 (hw z hz).2
      by_cases hq : B.ΘeqB s x z
      · rw [if_pos ((hiff.2 hq).trans hzy), if_pos (decide_eq_true hq)]
        simp [coefSum]
      · rw [if_neg (fun e => hq (hiff.1 (e.trans hzy.symm))), if_neg (by simpa using hq)]
        simp [coefSum]
  by_cases hex : ∃ z, ∃ hz : z ∈ L ++ L', B.θ γ z (mem z hz) = y
  · obtain ⟨z, hz, hzy⟩ := hex
    rw [key L (fun x hx => List.mem_append_left _ hx) z hz hzy, key L' (fun x hx => List.mem_append_right _ hx) z hz hzy]
    exact hc z hz
  · have zero : ∀ (K : List Tm), (∀ x ∈ K, x ∈ L ++ L') → (K.map ((Finsupp.applyAddHom y) ∘ B.tF (s := s) γ)).sum = 0 := by
      intro K hK
      induction K with
      | nil => rfl
      | cons x K ih =>
        have hxK : x ∈ L ++ L' := hK x (List.mem_cons_self ..)
        rw [List.map_cons, List.sum_cons, ih (fun u hu => hK u (List.mem_cons_of_mem _ hu))]
        simp only [Function.comp, Finsupp.applyAddHom_apply]
        rw [B.tF_apply γ x (mem x hxK) y, if_neg (fun e => hex ⟨x, hxK, e⟩), add_zero]
    rw [zero L (fun x hx => List.mem_append_left _ hx), zero L' (fun x hx => List.mem_append_right _ hx)]

/-- **Soundness of `Same`**, in any degree `q = s`. -/
theorem same_sound {s : ℕ} {L L' : List Tm} (h : B.Same s L L') (q : ℤ) (hq : q = s) (γ : SL(2, ℤ)) :
    val B.D B.hX q γ L = val B.D B.hX q γ L' := by
  subst hq; exact B.same_sound' h γ

end Br

/-! ### Table maps -/

/-- **The image of a formal sum under a table**: `∑ c · [g, τ] ↦ ∑ c · g · T τ`. -/
def appL (T : ℕ → List Tm) (L : List Tm) : List Tm := L.flatMap fun x => scaleL x.1 (transL x.2.1 (T x.2.2))

namespace Br

/-- **An admissible table** of degree `s → s'` from `B1` to `B2`: its entries exist, have determinant one, and it is
compatible with every stored transporter `t ∈ S(σ, σ')` of `B1`: `T σ = sgn(t) · t · T σ'` (decided by `Same`). -/
def TabOk (B1 B2 : Br) (s s' : ℕ) (T : ℕ → List Tm) : Prop :=
  (∀ σ ∈ B1.D.idxL (s : ℤ), ∀ x ∈ T σ, x.2.1.det = 1 ∧ x.2.2 ∈ B2.D.idxL (s' : ℤ)) ∧
  ∀ t ∈ B1.D.trZ (s : ℤ), B2.Same s' (T t.s1) (scaleL (permSign t.perm s) (transL t.g (T t.s2)))

instance (B1 B2 : Br) (s s' : ℕ) (T : ℕ → List Tm) : Decidable (TabOk B1 B2 s s' T) := by
  unfold TabOk; infer_instance

variable {B1 B2 : Br} {s s' : ℕ} {T : ℕ → List Tm}

/-- The table map on the free module. -/
noncomputable def FF (B1 B2 : Br) (T : ℕ → List Tm) (q q' : ℤ) :
    (SL(2, ℤ) × (B1.D.toXBulk B1.hX).V q →₀ ℤ) →ₗ[ℤ] chains (B2.D.toXBulk B2.hX) q' :=
  Finsupp.linearCombination ℤ (fun a => val B2.D B2.hX q' a.1 (T a.2.1))

lemma relSub_le_FF (hT : TabOk B1 B2 s s' T) (q q' : ℤ) (hq : q = s) (hq' : q' = s') :
    relSub ((B1.D.toXBulk B1.hX).M q) ≤ LinearMap.ker (FF B1 B2 T q q') := by
  subst hq
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, LinearMap.mem_ker, map_sub, sub_eq_zero]
  obtain ⟨h1, h2⟩ := h
  obtain ⟨t, ht, hg⟩ := (TauData.mem_Sset B1.D).1 (show x.1⁻¹ * y.1 ∈ B1.D.Sset (s : ℤ) x.2.1.1 y.2.1.1 from h1)
  have htz := TauData.trZ_of_trSetZ B1.D ht
  have ht' := (TauData.mem_trSetZ B1.D).1 ht
  have hdet : t.g.det = 1 := ((TauData.keyAx_all B1.hX (s : ℤ)).2.2.2.1 t htz).1
  have hy : y.1 = x.1 * M22.toSL t.g := by rw [hg, mul_inv_cancel_left]
  have hsame := hT.2 t htz
  rw [ht'.2.2.1, ht'.2.2.2] at hsame
  have hdetL : DetOne (T y.2.1.1) := fun z hz => (hT.1 _ ((B1.D.mem_idxF).1 y.2.1.2) z hz).1
  simp only [FF, gen, Finsupp.linearCombination_single]
  rw [B2.same_sound hsame q' hq' x.1, val_scaleL, val_transL _ _ _ _ hdet hdetL, ← hy, smul_smul]
  congr 1
  have : x.2.2 = B1.D.sgnZ (s : ℤ) x.2.1.1 y.2.1.1 (x.1⁻¹ * y.1) * y.2.2 := h2
  rw [this, ← hg, TauData.sgnZ_eq (TauData.keyAx_all B1.hX (s : ℤ)) ht, Units.val_mul, unitOf_val (permSign_cases _ _),
    Int.toNat_natCast]
  have := ACBridge.permSign_sq t.perm s
  linear_combination (y.2.2 : ℤ) * this

/-- **The table map** `C_q(X_bulk(B1)) → C_{q'}(X_bulk(B2))`, `[γ, σ] ↦ val γ (T σ)`. -/
noncomputable def Fmap (hT : TabOk B1 B2 s s' T) (q q' : ℤ) (hq : q = s) (hq' : q' = s') :
    chains (B1.D.toXBulk B1.hX) q →ₗ[ℤ] chains (B2.D.toXBulk B2.hX) q' :=
  (relSub ((B1.D.toXBulk B1.hX).M q)).liftQ (FF B1 B2 T q q') (relSub_le_FF hT q q' hq hq')

lemma Fmap_genQ (hT : TabOk B1 B2 s s' T) (q q' : ℤ) (hq : q = s) (hq' : q' = s') (a : SL(2, ℤ) × (B1.D.toXBulk B1.hX).V q) :
    Fmap hT q q' hq hq' (genQ (B1.D.toXBulk B1.hX) q a) = val B2.D B2.hX q' a.1 (T a.2.1) := by
  show FF B1 B2 T q q' (gen _) = _
  simp [FF, gen, Finsupp.linearCombination_single]

/-- **The table map on a formal sum** is the formal sum of the table images. -/
lemma Fmap_val (hT : TabOk B1 B2 s s' T) (q q' : ℤ) (hq : q = s) (hq' : q' = s') (γ : SL(2, ℤ)) (L : List Tm)
    (hL : ∀ x ∈ L, x.2.2 ∈ B1.D.idxL (s : ℤ) ∧ x.2.1.det = 1) :
    Fmap hT q q' hq hq' (val B1.D B1.hX q γ L) = val B2.D B2.hX q' γ (appL T L) := by
  induction L with
  | nil => simp [val, appL]
  | cons x L ih =>
    rw [val_cons, map_add, map_zsmul, ih (fun y hy => hL y (List.mem_cons_of_mem _ hy)),
      show appL T (x :: L) = scaleL x.1 (transL x.2.1 (T x.2.2)) ++ appL T L by simp [appL], val_append]
    congr 1
    have hx := hL x (List.mem_cons_self ..)
    have hmem : x.2.2 ∈ B1.D.idxF q := (B1.D.mem_idxF).2 (by rw [hq]; exact hx.1)
    rw [gq, dif_pos hmem, Fmap_genQ, val_scaleL, val_transL _ _ _ _ hx.2 (fun z hz => (hT.1 _ hx.1 z hz).1)]

end Br

/-! ### Acyclicity through a deformation retraction given by tables -/

/-- A formal sum is well formed in degree `s`. -/
def WFL (B : Br) (s : ℕ) (L : List Tm) : Prop := ∀ x ∈ L, x.2.2 ∈ B.D.idxL (s : ℤ) ∧ x.2.1.det = 1

instance (B : Br) (s : ℕ) (L : List Tm) : Decidable (WFL B s L) := by unfold WFL; infer_instance

/-- **The decided data of a deformation retraction** `C(B2) → C(B1) → C(B2)`: chain maps `c` (`B2 → B1`) and `sd`
(`B1 → B2`) and a homotopy `h` with `id - sd ∘ c = d h + h d` on `C(B2)`, all given by tables. -/
structure RetrChk (B1 B2 : Br) (sd c : ℕ → ℕ → List Tm) (h0 h1 : ℕ → List Tm) : Prop where
  sd0 : Br.TabOk B1 B2 0 0 (sd 0)
  sd1 : Br.TabOk B1 B2 1 1 (sd 1)
  sd2 : Br.TabOk B1 B2 2 2 (sd 2)
  c0 : Br.TabOk B2 B1 0 0 (c 0)
  c1 : Br.TabOk B2 B1 1 1 (c 1)
  c2 : Br.TabOk B2 B1 2 2 (c 2)
  hk0 : Br.TabOk B2 B2 0 1 h0
  hk1 : Br.TabOk B2 B2 1 2 h1
  wf1 : ∀ σ ∈ B1.D.idxL ((1 : ℕ) : ℤ), WFL B1 0 (bdL B1.D 0 (unitL σ))
  wf2 : ∀ σ ∈ B1.D.idxL ((2 : ℕ) : ℤ), WFL B1 1 (bdL B1.D 1 (unitL σ))
  wf1' : ∀ σ ∈ B2.D.idxL ((1 : ℕ) : ℤ), WFL B2 0 (bdL B2.D 0 (unitL σ))
  wf2' : ∀ σ ∈ B2.D.idxL ((2 : ℕ) : ℤ), WFL B2 1 (bdL B2.D 1 (unitL σ))
  S1 : ∀ σ ∈ B1.D.idxL ((1 : ℕ) : ℤ), B2.Same 0 (bdL B2.D 0 (sd 1 σ)) (appL (sd 0) (bdL B1.D 0 (unitL σ)))
  S2 : ∀ σ ∈ B1.D.idxL ((2 : ℕ) : ℤ), B2.Same 1 (bdL B2.D 1 (sd 2 σ)) (appL (sd 1) (bdL B1.D 1 (unitL σ)))
  C1 : ∀ σ ∈ B2.D.idxL ((1 : ℕ) : ℤ), B1.Same 0 (bdL B1.D 0 (c 1 σ)) (appL (c 0) (bdL B2.D 0 (unitL σ)))
  C2 : ∀ σ ∈ B2.D.idxL ((2 : ℕ) : ℤ), B1.Same 1 (bdL B1.D 1 (c 2 σ)) (appL (c 1) (bdL B2.D 1 (unitL σ)))
  Ca : ∀ σ ∈ B2.D.idxL ((0 : ℕ) : ℤ), augL (c 0 σ) = 1
  H0 : ∀ σ ∈ B2.D.idxL ((0 : ℕ) : ℤ), B2.Same 0 (unitL σ ++ negL (appL (sd 0) (c 0 σ))) (bdL B2.D 0 (h0 σ))
  H1 : ∀ σ ∈ B2.D.idxL ((1 : ℕ) : ℤ),
    B2.Same 1 (unitL σ ++ negL (appL (sd 1) (c 1 σ))) (bdL B2.D 1 (h1 σ) ++ appL h0 (bdL B2.D 0 (unitL σ)))
  H2 : ∀ σ ∈ B2.D.idxL ((2 : ℕ) : ℤ), B2.Same 2 (unitL σ ++ negL (appL (sd 2) (c 2 σ))) (appL h1 (bdL B2.D 1 (unitL σ)))

section Retract

variable {B1 B2 : Br} {sd c : ℕ → ℕ → List Tm} {h0 h1 : ℕ → List Tm} (R : RetrChk B1 B2 sd c h0 h1)

lemma unitL_wf {B : Br} {s : ℕ} {σ : ℕ} (h : σ ∈ B.D.idxL (s : ℤ)) : ∀ x ∈ unitL σ, x.2.2 ∈ B.D.idxL (s : ℤ) ∧ x.2.1.det = 1 := by
  intro x hx
  simp only [unitL, List.mem_singleton] at hx
  subst hx
  exact ⟨h, show M22.I.det = 1 by decide⟩

lemma memL {B : Br} {q : ℤ} (σ : (B.D.toXBulk B.hX).V q) : σ.1 ∈ B.D.idxL q := (B.D.mem_idxF).1 σ.2

include R in
/-- **(AC) through a deformation retraction**: if `X_bulk(B1)` is acyclic, so is `X_bulk(B2)`. -/
theorem acyclic_of_retr (hA : XBulk.Acyclic (B1.D.toXBulk B1.hX)) : XBulk.Acyclic (B2.D.toXBulk B2.hX) := by
  set X1 := B1.D.toXBulk B1.hX
  set X2 := B2.D.toXBulk B2.hX
  let sd0 := Br.Fmap R.sd0 0 0 rfl rfl
  let sd1 := Br.Fmap R.sd1 (0 + 1) (0 + 1) rfl rfl
  let sd2 := Br.Fmap R.sd2 (0 + 1 + 1) (0 + 1 + 1) rfl rfl
  let c0 := Br.Fmap R.c0 0 0 rfl rfl
  let c1 := Br.Fmap R.c1 (0 + 1) (0 + 1) rfl rfl
  let c2 := Br.Fmap R.c2 (0 + 1 + 1) (0 + 1 + 1) rfl rfl
  let k0 := Br.Fmap R.hk0 0 (0 + 1) rfl rfl
  let k1 := Br.Fmap R.hk1 (0 + 1) (0 + 1 + 1) rfl rfl
  -- the chain-map and homotopy identities, on generators
  have eS1 : ∀ x : chains X1 (0 + 1), bd X2 0 (sd1 x) = sd0 (bd X1 0 x) := by
    intro x
    have := chains_hom_ext X1 (q := 0 + 1) (u := ((bd X2 0) ∘ₗ sd1).toAddMonoidHom)
      (v := (sd0 ∘ₗ (bd X1 0)).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply]
        rw [Br.Fmap_genQ, bd_val B2.D B2.hX 0 0 rfl _ _ (fun x hx => (R.sd1.1 _ (memL a.2) x hx).2),
          genQ_eq_val, bd_val B1.D B1.hX 0 0 rfl _ _ (fun x hx => (unitL_wf (memL a.2)) x hx |>.1),
          Br.Fmap_val _ _ _ _ _ _ _ (R.wf1 _ (memL a.2))]
        exact B2.same_sound (R.S1 _ (memL a.2)) 0 rfl a.1)
    exact DFunLike.congr_fun this x
  have eS2 : ∀ x : chains X1 (0 + 1 + 1), bd X2 (0 + 1) (sd2 x) = sd1 (bd X1 (0 + 1) x) := by
    intro x
    have := chains_hom_ext X1 (q := 0 + 1 + 1) (u := ((bd X2 (0 + 1)) ∘ₗ sd2).toAddMonoidHom)
      (v := (sd1 ∘ₗ (bd X1 (0 + 1))).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply]
        rw [Br.Fmap_genQ, bd_val B2.D B2.hX (0 + 1) 1 rfl _ _ (fun x hx => (R.sd2.1 _ (memL a.2) x hx).2),
          genQ_eq_val, bd_val B1.D B1.hX (0 + 1) 1 rfl _ _ (fun x hx => (unitL_wf (memL a.2)) x hx |>.1),
          Br.Fmap_val _ _ _ _ _ _ _ (R.wf2 _ (memL a.2))]
        exact B2.same_sound (R.S2 _ (memL a.2)) (0 + 1) rfl a.1)
    exact DFunLike.congr_fun this x
  have eC1 : ∀ x : chains X2 (0 + 1), bd X1 0 (c1 x) = c0 (bd X2 0 x) := by
    intro x
    have := chains_hom_ext X2 (q := 0 + 1) (u := ((bd X1 0) ∘ₗ c1).toAddMonoidHom)
      (v := (c0 ∘ₗ (bd X2 0)).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply]
        rw [Br.Fmap_genQ, bd_val B1.D B1.hX 0 0 rfl _ _ (fun x hx => (R.c1.1 _ (memL a.2) x hx).2),
          genQ_eq_val, bd_val B2.D B2.hX 0 0 rfl _ _ (fun x hx => (unitL_wf (memL a.2)) x hx |>.1),
          Br.Fmap_val _ _ _ _ _ _ _ (R.wf1' _ (memL a.2))]
        exact B1.same_sound (R.C1 _ (memL a.2)) 0 rfl a.1)
    exact DFunLike.congr_fun this x
  have eC2 : ∀ x : chains X2 (0 + 1 + 1), bd X1 (0 + 1) (c2 x) = c1 (bd X2 (0 + 1) x) := by
    intro x
    have := chains_hom_ext X2 (q := 0 + 1 + 1) (u := ((bd X1 (0 + 1)) ∘ₗ c2).toAddMonoidHom)
      (v := (c1 ∘ₗ (bd X2 (0 + 1))).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply]
        rw [Br.Fmap_genQ, bd_val B1.D B1.hX (0 + 1) 1 rfl _ _ (fun x hx => (R.c2.1 _ (memL a.2) x hx).2),
          genQ_eq_val, bd_val B2.D B2.hX (0 + 1) 1 rfl _ _ (fun x hx => (unitL_wf (memL a.2)) x hx |>.1),
          Br.Fmap_val _ _ _ _ _ _ _ (R.wf2' _ (memL a.2))]
        exact B1.same_sound (R.C2 _ (memL a.2)) (0 + 1) rfl a.1)
    exact DFunLike.congr_fun this x
  have eCa : ∀ x : chains X2 0, aug X1 (c0 x) = aug X2 x := by
    intro x
    have := chains_hom_ext X2 (q := 0) (u := ((aug X1) ∘ₗ c0).toAddMonoidHom) (v := (aug X2).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply]
        rw [Br.Fmap_genQ, aug_val B1.D B1.hX _ _ (fun x hx => (B1.D.mem_idxF).2 (R.c0.1 _ (memL a.2) x hx).2),
          R.Ca _ (memL a.2), aug_genQ])
    exact DFunLike.congr_fun this x
  have eH0 : ∀ x : chains X2 0, x - sd0 (c0 x) = bd X2 0 (k0 x) := by
    intro x
    have := chains_hom_ext X2 (q := 0) (u := (LinearMap.id - sd0 ∘ₗ c0).toAddMonoidHom)
      (v := ((bd X2 0) ∘ₗ k0).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply, LinearMap.sub_apply,
          LinearMap.id_apply]
        rw [Br.Fmap_genQ, Br.Fmap_val _ _ _ _ _ _ _ (fun x hx => ⟨(R.c0.1 _ (memL a.2) x hx).2, (R.c0.1 _ (memL a.2) x hx).1⟩),
          Br.Fmap_genQ, bd_val B2.D B2.hX 0 0 rfl _ _ (fun x hx => (R.hk0.1 _ (memL a.2) x hx).2), genQ_eq_val,
          sub_eq_add_neg, ← val_negL, ← val_append]
        exact B2.same_sound (R.H0 _ (memL a.2)) 0 rfl a.1)
    exact DFunLike.congr_fun this x
  have eH1 : ∀ x : chains X2 (0 + 1), x - sd1 (c1 x) = bd X2 (0 + 1) (k1 x) + k0 (bd X2 0 x) := by
    intro x
    have := chains_hom_ext X2 (q := 0 + 1) (u := (LinearMap.id - sd1 ∘ₗ c1).toAddMonoidHom)
      (v := ((bd X2 (0 + 1)) ∘ₗ k1 + k0 ∘ₗ (bd X2 0)).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply, LinearMap.sub_apply,
          LinearMap.id_apply, LinearMap.add_apply]
        rw [Br.Fmap_genQ, Br.Fmap_val _ _ _ _ _ _ _ (fun x hx => ⟨(R.c1.1 _ (memL a.2) x hx).2, (R.c1.1 _ (memL a.2) x hx).1⟩),
          Br.Fmap_genQ, bd_val B2.D B2.hX (0 + 1) 1 rfl _ _ (fun x hx => (R.hk1.1 _ (memL a.2) x hx).2), genQ_eq_val,
          bd_val B2.D B2.hX 0 0 rfl _ _ (fun x hx => (unitL_wf (memL a.2)) x hx |>.1),
          Br.Fmap_val _ _ _ _ _ _ _ (R.wf1' _ (memL a.2)),
          sub_eq_add_neg, ← val_negL, ← val_append, ← val_append]
        exact B2.same_sound (R.H1 _ (memL a.2)) (0 + 1) rfl a.1)
    exact DFunLike.congr_fun this x
  have eH2 : ∀ x : chains X2 (0 + 1 + 1), x - sd2 (c2 x) = k1 (bd X2 (0 + 1) x) := by
    intro x
    have := chains_hom_ext X2 (q := 0 + 1 + 1) (u := (LinearMap.id - sd2 ∘ₗ c2).toAddMonoidHom)
      (v := (k1 ∘ₗ (bd X2 (0 + 1))).toAddMonoidHom) (fun a => by
        simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply, LinearMap.sub_apply,
          LinearMap.id_apply]
        rw [Br.Fmap_genQ, Br.Fmap_val _ _ _ _ _ _ _ (fun x hx => ⟨(R.c2.1 _ (memL a.2) x hx).2, (R.c2.1 _ (memL a.2) x hx).1⟩),
          genQ_eq_val, bd_val B2.D B2.hX (0 + 1) 1 rfl _ _ (fun x hx => (unitL_wf (memL a.2)) x hx |>.1),
          Br.Fmap_val _ _ _ _ _ _ _ (R.wf2' _ (memL a.2)), sub_eq_add_neg, ← val_negL, ← val_append]
        exact B2.same_sound (R.H2 _ (memL a.2)) (0 + 1 + 1) rfl a.1)
    exact DFunLike.congr_fun this x
  -- the acyclicity of `X_bulk(B2)`
  refine ⟨fun x hx => ?_, fun x hx => ?_, fun x hx => ?_⟩
  · obtain ⟨y, hy⟩ := hA.contract0 (c0 x) (by rw [eCa, hx])
    refine ⟨sd1 y + k0 x, ?_⟩
    rw [map_add, eS1, hy, ← eH0]
    abel
  · obtain ⟨y, hy⟩ := hA.contract1 (c1 x) (by rw [eC1, hx, map_zero])
    refine ⟨sd2 y + k1 x, ?_⟩
    have := eH1 x
    rw [hx, map_zero, add_zero] at this
    rw [map_add, eS2, hy, ← this]
    abel
  · have h0 := hA.contract2 (c2 x) (by rw [eC2, hx, map_zero])
    have := eH2 x
    rw [hx, map_zero, h0, map_zero, sub_zero] at this
    exact this

end Retract

end TabMap

end GL2.MP
