import GL2.MP.DatumCover
import GL2.Model.Bridge

/-!
# MP-L2, open item 1 (slot 22): the (AC) bridge -- ordered chains of `Y` = the oriented chains of `X_bulk`

[GT] `gt_part_MP.tex`, rows `def:mp-ac-checks`, `thm:mp-ac-bridge`, backing [C] `thm:mpcomparison` (5) ((AC) for the
vertex scheme of an MP datum whose simplices are those of a `τ`-dump).

**Setting.**  A `τ`-dump `D` with the `X_bulk` axioms (`TauData.XAx`; `X = D.toXBulk`, the oriented simplicial chains
`C_•(X_bulk)` over `ℤ`, faces `∂_j ι(τ) = (-1)^j ι(∂_j τ)`, transporters with stored vertex correspondences) and an MP
descriptor `M` (`MPDatum`, well formed, (IO), `dim ≤ 2`; the cover `M.cover` and its `Cover.Hyp`), **on the same point
labels**: the vertex `v` of the dump is the point `v` of `M`.  The ordered `s`-simplices `YS s` of the MP vertex scheme `Y`
are the strings of vertices of `Y` strictly increasing in the orbit index (the (IO) order) inside a translate of a maximal
simplex.

**The sign bridge.**  `srt σ` is the stored vertex list of the dump simplex `σ` sorted by orbit index, `eps σ = ±1` the
sign of the sorting permutation (`permSign` of the position list `posL σ`).  `Θ (γ, σ) = γ · srt σ` is an ordered simplex
of `Y`, and

  `Ξ : C_s(X_bulk) → ℤ[YS s]`,  `[γ, σ] ↦ eps σ · e_{γ · srt σ}`

is well defined on the oriented relation, bijective and a chain map commuting with the augmentations, given the finite
checks `Checks D M` (decided by the kernel on the data):

* `chkSimp` -- every sorted list has the right length, is strictly increasing in orbit index, lies in a maximal simplex;
* `chkTr` -- every stored transporter `t ∈ S(σ, σ')` carries `srt σ'` onto `srt σ` slot by slot, and
  `eps σ = permSign(t) · eps σ'`;
* `chkInj` -- ((SC) for `Ξ`) every `γ` carrying `srt σ'` onto `srt σ` slot by slot is a stored transporter (only the
  coset `g_p Γ_o g_w⁻¹` of the first vertices has to be searched);
* `chkSurj` -- every orbit-increasing string of vertices of a maximal simplex is `srt σ` of a dump simplex;
* `chkBd` -- the faces: deleting slot `i` of `srt τ` is `srt` of the stored face `∂_{π i} τ`, `π` the position
  permutation, with `(-1)^{π i} eps(∂_{π i} τ) = eps τ (-1)^i`.

**Consequence** (`yCx_acyclic_of_bridge`): if `X_bulk` is acyclic (H4(ii), `XBulk.Acyclic`) and has a vertex, the
augmented ordered chain complex `k[Y_•] → k` is acyclic over **every** commutative ring `k`: the `ℤ`-contraction of
`GL2.Model.Bridge` (`contractZ`) transported through `Ξ` is a contraction of `ℤ[Y_•] → ℤ`, and its base change is one of
`k[Y_•] → k`.
-/

namespace GL2.MP

open GL2.TauChk GL2.Model GL2.Model.XBulk Finsupp
open scoped MatrixGroups

namespace ACBridge

variable (D : TauData) (M : MPDatum)

/-! ### The finite data and the checks -/

/-- The stored vertex list of the dump simplex `σ`, sorted by orbit index (the (IO) order). -/
def srt (q σ : ℕ) : List ℕ := (D.verts q σ).insertionSort fun p p' => M.orb p ≤ M.orb p'

/-- Slot `j` of `srt σ`. -/
def sv (q σ j : ℕ) : ℕ := (srt D M q σ).getD j 0

/-- The position permutation: slot `i` of `srt σ` is slot `posL σ i` of the stored list. -/
def posL (q σ : ℕ) : List ℕ := (srt D M q σ).map fun p => (D.verts q σ).idxOf p

/-- **The orientation sign** of `σ`: the sign of the sorting permutation. -/
def eps (q σ : ℕ) : ℤ := permSign (posL D M q σ) q

/-- An increasing string of length `n` (in orbit index). -/
def Incr (n : ℕ) (l : List ℕ) : Prop :=
  ∀ i ∈ List.range n, ∀ j ∈ List.range n, i < j → M.orb (l.getD i 0) < M.orb (l.getD j 0)

instance (n : ℕ) (l : List ℕ) : Decidable (Incr M n l) := by unfold Incr; infer_instance

/-- All strings of length `n` with entries in `l`. -/
def tuples (l : List ℕ) : ℕ → List (List ℕ)
  | 0 => [[]]
  | n + 1 => l.flatMap fun a => (tuples l n).map (a :: ·)

/-- (1) the sorted lists: length, inside the points, increasing, inside a maximal simplex. -/
def chkSimp : Prop :=
  ∀ q ∈ ([0, 1, 2] : List ℕ), ∀ σ ∈ D.idxL (q : ℤ), (srt D M q σ).length = q + 1 ∧
    (∀ j ∈ List.range (q + 1), sv D M q σ j < M.pts.length) ∧ Incr M (q + 1) (srt D M q σ) ∧
    ∃ t ∈ List.range M.top.length, ∀ j ∈ List.range (q + 1), sv D M q σ j ∈ M.tv t

/-- (2) stored transporters respect the sorted lists and the signs. -/
def chkTr : Prop :=
  ∀ q ∈ ([0, 1, 2] : List ℕ), ∀ t ∈ D.trZ (q : ℤ), (∀ j ∈ List.range (q + 1), M.maps t.g (sv D M q t.s2 j) (sv D M q t.s1 j) = true) ∧
    eps D M q t.s1 = permSign t.perm q * eps D M q t.s2

/-- (3) every slot-wise transporter of sorted lists is stored ((SC) for `Ξ`). -/
def chkInj : Prop :=
  ∀ q ∈ ([0, 1, 2] : List ℕ), ∀ σ ∈ D.idxL (q : ℤ), ∀ σ' ∈ D.idxL (q : ℤ),
    ∀ m ∈ M.coset (M.gp (sv D M q σ 0)) (M.orb (sv D M q σ 0)) (M.gp (sv D M q σ' 0)),
      (∀ j ∈ List.range (q + 1), M.maps m (sv D M q σ' j) (sv D M q σ j) = true) →
        ∃ t ∈ D.trSetZ (q : ℤ) σ σ', t.g = m

/-- (4) every increasing string of vertices of a maximal simplex is a sorted dump simplex. -/
def chkSurj : Prop :=
  ∀ q ∈ ([0, 1, 2] : List ℕ), ∀ t ∈ List.range M.top.length, ∀ u ∈ tuples (M.tv t) (q + 1), Incr M (q + 1) u →
    ∃ σ ∈ D.idxL (q : ℤ), srt D M q σ = u

/-- The slot `i.succAbove m` in `ℕ`. -/
def skip (i m : ℕ) : ℕ := if m < i then m else m + 1

/-- (5) the faces: deleting slot `i` of `srt τ` is `srt (∂_{π i} τ)`, with the signs. -/
def chkBd : Prop :=
  ∀ s ∈ ([0, 1] : List ℕ), ∀ τ ∈ D.idxL ((s + 1 : ℕ) : ℤ),
    (∀ i ∈ List.range (s + 2), (posL D M (s + 1) τ).getD i 0 < s + 2) ∧
    ((List.range (s + 2)).map fun i => (posL D M (s + 1) τ).getD i 0).Nodup ∧
    ∀ i ∈ List.range (s + 2),
      (∀ m ∈ List.range (s + 1),
        sv D M s (D.face (s + 1) τ ((posL D M (s + 1) τ).getD i 0)) m = sv D M (s + 1) τ (skip i m)) ∧
      sgnPow ((posL D M (s + 1) τ).getD i 0) * eps D M s (D.face (s + 1) τ ((posL D M (s + 1) τ).getD i 0)) =
        eps D M (s + 1) τ * sgnPow i

/-- **The finite checks** of the bridge. -/
def Checks : Prop := chkSimp D M ∧ chkTr D M ∧ chkInj D M ∧ chkSurj D M ∧ chkBd D M

instance : Decidable (chkSimp D M) := by unfold chkSimp; infer_instance
instance : Decidable (chkTr D M) := by unfold chkTr; infer_instance
instance : Decidable (chkInj D M) := by unfold chkInj; infer_instance
instance : Decidable (chkSurj D M) := by unfold chkSurj; infer_instance
instance : Decidable (chkBd D M) := by unfold chkBd; infer_instance
instance : Decidable (Checks D M) := by unfold Checks; infer_instance


/-! ### Generic facts -/

lemma eps_cases (q σ : ℕ) : eps D M q σ = 1 ∨ eps D M q σ = -1 := permSign_cases _ _

lemma eps_sq (q σ : ℕ) : eps D M q σ * eps D M q σ = 1 := by
  rcases eps_cases D M q σ with e | e <;> rw [e] <;> norm_num

lemma eps_zero (σ : ℕ) : eps D M 0 σ = 1 := permSign_zero _

lemma sgnPow_eq (j : ℕ) : sgnPow j = (-1 : ℤ) ^ j := by
  unfold sgnPow
  rcases Nat.even_or_odd j with h | h
  · rw [if_pos (Nat.even_iff.1 h), h.neg_one_pow]
  · rw [if_neg (by have := Nat.odd_iff.1 h; omega), h.neg_one_pow]

lemma val_succAbove {n : ℕ} (i : Fin (n + 2)) (m : Fin (n + 1)) : (i.succAbove m : ℕ) = skip i m := by
  by_cases h : (m : ℕ) < i
  · rw [Fin.succAbove_of_castSucc_lt _ _ (by rw [Fin.lt_def]; simpa using h)]; simp [skip, h]
  · rw [Fin.succAbove_of_le_castSucc _ _ (by rw [Fin.le_def]; simp; omega)]; simp [skip, h]

lemma mem_tuples (l : List ℕ) : ∀ (u : List ℕ), (∀ a ∈ u, a ∈ l) → u ∈ tuples l u.length
  | [], _ => by simp [tuples]
  | a :: u, h => by
    simp only [tuples, List.length_cons, List.mem_flatMap, List.mem_map]
    exact ⟨a, h a (by simp), u, mem_tuples l u (fun b hb => h b (by simp [hb])), rfl⟩

variable {D M}

/-! ### The bridge map `Θ` and the sign bridge `Ξ` -/

section Bridge

variable (hX : D.XAx) (hW : M.WF) (hio : M.ioB = true) (hdim : M.dimB = true)

/-- The cover's hypotheses. -/
noncomputable abbrev Hy : (M.cover hW).Hyp := MPDatum.hyp hW hio hdim

lemma mem_idxL_of {s : ℕ} (σ : (D.toXBulk hX).V (s : ℤ)) : σ.1 ∈ D.idxL (s : ℤ) := (D.mem_idxF).1 σ.2

include hX in
lemma mem_dims {s : ℕ} (σ : (D.toXBulk hX).V (s : ℤ)) : s ∈ ([0, 1, 2] : List ℕ) := by
  obtain ⟨x, hx, hxq, -⟩ := TauData.mem_idxL.1 (mem_idxL_of hX σ)
  have := hX.1 x hx
  have hs : s ≤ 2 := by omega
  interval_cases s <;> simp

variable (hS : chkSimp D M)

include hX hS in
lemma sv_lt {s : ℕ} (σ : (D.toXBulk hX).V (s : ℤ)) (j : Fin (s + 1)) : sv D M s σ.1 j < M.pts.length :=
  (hS s (mem_dims hX σ) σ.1 (mem_idxL_of hX σ)).2.1 j (List.mem_range.2 j.2)

/-- The slot `j` vertex of `srt σ`, as a point. -/
def vt {s : ℕ} (σ : (D.toXBulk hX).V (s : ℤ)) (j : Fin (s + 1)) : Fin M.pts.length := ⟨sv D M s σ.1 j, sv_lt hX hS σ j⟩

lemma ord_pt (γ : SL(2, ℤ)) (p : Fin M.pts.length) : (Hy hW hio hdim).ord (γ • MPDatum.pt hW p) = M.orb p := by
  rfl

/-- **`Θ (γ, σ) = γ · srt σ`**, an ordered simplex of `Y`. -/
def Θ {s : ℕ} (a : SL(2, ℤ) × (D.toXBulk hX).V (s : ℤ)) : Cover.YS (Hy hW hio hdim) s :=
  ⟨fun j => a.1 • MPDatum.pt hW (vt hX hS a.2 j), by
    intro i j hij
    simp only [Function.comp, ord_pt]
    exact (hS s (mem_dims hX a.2) a.2.1 (mem_idxL_of hX a.2)).2.2.1 i (List.mem_range.2 i.2) j (List.mem_range.2 j.2) hij, by
    obtain ⟨t, ht, htv⟩ := (hS s (mem_dims hX a.2) a.2.1 (mem_idxL_of hX a.2)).2.2.2
    refine ⟨(a.1, ⟨t, List.mem_range.1 ht⟩), fun j => ?_⟩
    show a.1⁻¹ • a.1 • MPDatum.pt hW (vt hX hS a.2 j) ∈ (M.cover hW).vtx ⟨t, List.mem_range.1 ht⟩
    rw [inv_smul_smul]
    exact ⟨vt hX hS a.2 j, htv j (List.mem_range.2 j.2), rfl⟩⟩

lemma Θ_apply {s : ℕ} (a : SL(2, ℤ) × (D.toXBulk hX).V (s : ℤ)) (j : Fin (s + 1)) :
    (Θ hX hW hio hdim hS a).1 j = a.1 • MPDatum.pt hW (vt hX hS a.2 j) := rfl

variable (hT : chkTr D M)

include hX in
lemma det_tr {q : ℤ} {t : Tr} (ht : t ∈ D.trZ q) : t.g.det = 1 := ((TauData.keyAx_all hX q).2.2.2.1 t ht).1

include hT in
/-- A stored transporter does not move `Θ`. -/
lemma Θ_tr {s : ℕ} (a b : SL(2, ℤ) × (D.toXBulk hX).V (s : ℤ)) {t : Tr} (ht : t ∈ D.trSetZ (s : ℤ) a.2.1 b.2.1)
    (hb : b.1 = a.1 * M22.toSL t.g) : Θ hX hW hio hdim hS a = Θ hX hW hio hdim hS b := by
  have ht' := (TauData.mem_trSetZ D).1 ht
  have htz : t ∈ D.trZ (s : ℤ) := TauData.trZ_of_trSetZ D ht
  have hc := (hT s (mem_dims hX a.2) t htz).1
  apply Subtype.ext
  funext j
  rw [Θ_apply, Θ_apply, hb, mul_smul]
  congr 1
  symm
  rw [MPDatum.smul_pt_eq_iff hW t.g (det_tr hX htz)]
  have := hc j (List.mem_range.2 j.2)
  rw [ht'.2.2.1, ht'.2.2.2] at this
  exact this

include hX in
lemma mem_dims_tr {s : ℕ} {t : Tr} (ht : t ∈ D.trZ (s : ℤ)) : s ∈ ([0, 1, 2] : List ℕ) := by
  have h1 := hX.2.1 t ((TauData.mem_trZ D).1 ht).1
  have h2 := ((TauData.mem_trZ D).1 ht).2
  have hs : s ≤ 2 := by omega
  interval_cases s <;> simp

include hX hT in
/-- The sign of a stored transporter is the quotient of the orientation signs. -/
lemma permSign_tr {s : ℕ} {t : Tr} (ht : t ∈ D.trZ (s : ℤ)) :
    eps D M s t.s1 = permSign t.perm s * eps D M s t.s2 := (hT s (mem_dims_tr hX ht) t ht).2

/-! ### `Ξ` on the free module and on the chains -/

/-- `Ξ` on the free module: `e_(γ, σ) ↦ eps σ · e_{Θ (γ, σ)}`. -/
noncomputable def ΞF (s : ℕ) : (SL(2, ℤ) × (D.toXBulk hX).V (s : ℤ) →₀ ℤ) →ₗ[ℤ] (Cover.YS (Hy hW hio hdim) s →₀ ℤ) :=
  Finsupp.linearCombination ℤ (fun a => single (Θ hX hW hio hdim hS a) (eps D M s a.2.1))

lemma ΞF_gen {s : ℕ} (x : OrPt ((D.toXBulk hX).V (s : ℤ))) :
    ΞF hX hW hio hdim hS s (gen x) = single (Θ hX hW hio hdim hS (x.1, x.2.1)) ((x.2.2 : ℤ) * eps D M s x.2.1.1) := by
  simp [ΞF, gen, Finsupp.linearCombination_single, smul_single]

lemma permSign_sq (l : List ℕ) (q : ℕ) : permSign l q * permSign l q = 1 := by
  rcases permSign_cases l q with e | e <;> rw [e] <;> norm_num

include hT in
lemma relSub_le_ΞF (s : ℕ) :
    relSub ((D.toXBulk hX).M (s : ℤ)) ≤ LinearMap.ker (ΞF hX hW hio hdim hS s) := by
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, LinearMap.mem_ker, map_sub, sub_eq_zero]
  obtain ⟨h1, h2⟩ := h
  obtain ⟨t0, ht0, hg⟩ := (TauData.mem_Sset D).1 (show x.1⁻¹ * y.1 ∈ D.Sset (s : ℤ) x.2.1.1 y.2.1.1 from h1)
  have htz : t0 ∈ D.trZ (s : ℤ) := TauData.trZ_of_trSetZ D ht0
  have ht0' := (TauData.mem_trSetZ D).1 ht0
  rw [ΞF_gen, ΞF_gen]
  have hy : y.1 = x.1 * M22.toSL t0.g := by rw [hg, mul_inv_cancel_left]
  rw [Θ_tr hX hW hio hdim hS hT (x.1, x.2.1) (y.1, y.2.1) ht0 hy]
  congr 1
  have h2' : (x.2.2 : ℤ) = permSign t0.perm s * (y.2.2 : ℤ) := by
    have : x.2.2 = D.sgnZ (s : ℤ) x.2.1.1 y.2.1.1 (x.1⁻¹ * y.1) * y.2.2 := h2
    rw [this, ← hg, TauData.sgnZ_eq (TauData.keyAx_all hX (s : ℤ)) ht0, Units.val_mul,
      unitOf_val (permSign_cases _ _), Int.toNat_natCast]
  have he := permSign_tr hX hT htz
  rw [ht0'.2.2.1, ht0'.2.2.2] at he
  rw [h2', he]
  have := permSign_sq t0.perm s
  linear_combination ((y.2.2 : ℤ) * eps D M s y.2.1.1) * this

/-- **`Ξ : C_s(X_bulk) → ℤ[YS s]`**, `[γ, σ] ↦ eps σ · e_{γ · srt σ}`. -/
noncomputable def Ξ (s : ℕ) : chains (D.toXBulk hX) (s : ℤ) →ₗ[ℤ] (Cover.YS (Hy hW hio hdim) s →₀ ℤ) :=
  (relSub ((D.toXBulk hX).M (s : ℤ))).liftQ (ΞF hX hW hio hdim hS s) (relSub_le_ΞF hX hW hio hdim hS hT s)

lemma Ξ_genQ {s : ℕ} (a : SL(2, ℤ) × (D.toXBulk hX).V (s : ℤ)) :
    Ξ hX hW hio hdim hS hT s (genQ (D.toXBulk hX) (s : ℤ) a) = single (Θ hX hW hio hdim hS a) (eps D M s a.2.1) := by
  show ΞF hX hW hio hdim hS s (gen _) = _
  rw [ΞF_gen]; simp

/-! ### Surjectivity of `Θ` -/

variable (hU : chkSurj D M)

include hU in
lemma exists_pre {s : ℕ} (y : Cover.YS (Hy hW hio hdim) s) : ∃ a, Θ hX hW hio hdim hS a = y := by
  obtain ⟨o, ho⟩ := y.2.2
  obtain ⟨g, t⟩ := o
  change Fin M.top.length at t
  have hp : ∀ j, ∃ p : Fin M.pts.length, (p : ℕ) ∈ M.tv t ∧ g⁻¹ • y.1 j = MPDatum.pt hW p := fun j => ho j
  choose p hpt hpe using hp
  have hs3 : s + 1 ≤ 3 := Cover.card_le_three (Hy hW hio hdim) y.1 (Cover.injective_of_strictMono y.2.1) (g, t) ho
  have hs2 : s ≤ 2 := by omega
  have hs : s ∈ ([0, 1, 2] : List ℕ) := by interval_cases s <;> simp
  set u : List ℕ := List.ofFn fun j => (p j : ℕ)
  have hu : u ∈ tuples (M.tv t) (s + 1) := by
    have := mem_tuples (M.tv t) u (by
      intro a ha
      obtain ⟨j, rfl⟩ := List.mem_ofFn.1 ha
      exact hpt j)
    rwa [List.length_ofFn] at this
  have hord : ∀ j, M.orb (p j) = (Hy hW hio hdim).ord (y.1 j) := by
    intro j
    have : (Hy hW hio hdim).ord (g⁻¹ • y.1 j) = (Hy hW hio hdim).ord (y.1 j) := (Hy hW hio hdim).ord_smul _ _
    rw [← this, hpe j]; rfl
  have hgetD : ∀ j : Fin (s + 1), u.getD j 0 = p j := by
    intro j
    have hl : (j : ℕ) < u.length := by rw [List.length_ofFn]; exact j.2
    rw [List.getD_eq_getElem _ _ hl, List.getElem_ofFn]
  have hinc : Incr M (s + 1) u := by
    intro i hi j hj hij
    rw [List.mem_range] at hi hj
    rw [hgetD ⟨i, hi⟩, hgetD ⟨j, hj⟩, hord, hord]
    exact y.2.1 (show (⟨i, hi⟩ : Fin (s + 1)) < ⟨j, hj⟩ from hij)
  obtain ⟨σ, hσ, hsrt⟩ := hU s hs t (List.mem_range.2 t.2) u hu hinc
  let σ' : (D.toXBulk hX).V (s : ℤ) := show D.Vz (s : ℤ) from ⟨σ, (D.mem_idxF).2 hσ⟩
  refine ⟨(g, σ'), Subtype.ext (funext fun j => ?_)⟩
  rw [Θ_apply]
  have : vt hX hS σ' j = p j := by
    apply Fin.ext
    show sv D M s σ j = p j
    rw [sv, hsrt, hgetD]
  show g • MPDatum.pt hW (vt hX hS σ' j) = y.1 j
  rw [this, ← hpe j]
  exact smul_inv_smul (α := (M.cover hW).P) g (y.1 j)

/-- A chosen preimage under `Θ`. -/
noncomputable def pre {s : ℕ} (y : Cover.YS (Hy hW hio hdim) s) : SL(2, ℤ) × (D.toXBulk hX).V (s : ℤ) :=
  (exists_pre hX hW hio hdim hS hU y).choose

lemma Θ_pre {s : ℕ} (y : Cover.YS (Hy hW hio hdim) s) : Θ hX hW hio hdim hS (pre hX hW hio hdim hS hU y) = y :=
  (exists_pre hX hW hio hdim hS hU y).choose_spec

/-! ### Injectivity of `Θ` (with signs) -/

include hW in
lemma maps_coset {m : M22} (hm : m.det = 1) {w p : ℕ} (hw : w < M.pts.length) (hp : p < M.pts.length)
    (h : M.maps m w p = true) : m ∈ M.coset (M.gp p) (M.orb p) (M.gp w) := by
  simp only [MPDatum.maps, MPDatum.samePt, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨-, hst⟩ := h
  refine List.mem_map.2 ⟨_, hst, ?_⟩
  have hgp := hW.pts_det p hp
  have hgw := hW.pts_det w hw
  have h1 : ((M.gp p).adj * (m * M.gp w)).det = 1 := by
    rw [M22.det_mul, M22.det_adj, M22.det_mul, hgp, hm, hgw]; norm_num
  apply M22.toSL_injOn (by rw [M22.det_mul, M22.det_mul, hgp, h1, M22.det_adj, hgw]; norm_num) hm
  rw [M22.toSL_mul (by rw [M22.det_mul, hgp, h1]; norm_num) (by rw [M22.det_adj]; exact hgw),
    M22.toSL_mul hgp h1, M22.toSL_adj hgw, M22.toSL_mul (by rw [M22.det_adj]; exact hgp)
      (by rw [M22.det_mul, hm, hgw]; norm_num), M22.toSL_adj hgp, M22.toSL_mul hm hgw]
  group

variable (hI : chkInj D M)

include hT hI in
/-- **`Θ` is injective on classes, with signs**: `Θ a = Θ b` forces `[a] = eps a · eps b · [b]`. -/
lemma genQ_of_Θ {s : ℕ} {a b : SL(2, ℤ) × (D.toXBulk hX).V (s : ℤ)} (h : Θ hX hW hio hdim hS a = Θ hX hW hio hdim hS b) :
    genQ (D.toXBulk hX) (s : ℤ) a = (eps D M s a.2.1 * eps D M s b.2.1) • genQ (D.toXBulk hX) (s : ℤ) b := by
  set δ := a.1⁻¹ * b.1
  have hδ : ∀ j, M.maps (MPDatum.ofSL δ) (vt hX hS b.2 j) (vt hX hS a.2 j) = true := by
    intro j
    rw [← MPDatum.smul_pt_eq_iff hW _ (MPDatum.ofSL_det δ), MPDatum.toSL_ofSL]
    have := congrFun (congrArg Subtype.val h) j
    rw [Θ_apply, Θ_apply] at this
    rw [mul_smul, ← this, inv_smul_smul]
  have hc := maps_coset hW (MPDatum.ofSL_det δ) (vt hX hS b.2 0).2 (vt hX hS a.2 0).2 (hδ 0)
  obtain ⟨t, ht, htg⟩ := hI s (mem_dims hX a.2) a.2.1 (mem_idxL_of hX a.2) b.2.1 (mem_idxL_of hX b.2) _ hc
    (fun j hj => hδ ⟨j, List.mem_range.1 hj⟩)
  have htz : t ∈ D.trZ (s : ℤ) := TauData.trZ_of_trSetZ D ht
  have ht' := (TauData.mem_trSetZ D).1 ht
  have hδt : M22.toSL t.g = δ := by rw [htg, MPDatum.toSL_ofSL]
  have hsg : ((D.sgnZ (s : ℤ) a.2.1 b.2.1 δ : ℤˣ) : ℤ) = eps D M s a.2.1 * eps D M s b.2.1 := by
    rw [← hδt, TauData.sgnZ_eq (TauData.keyAx_all hX (s : ℤ)) ht, unitOf_val (permSign_cases _ _), Int.toNat_natCast]
    have he := permSign_tr hX hT htz
    rw [ht'.2.2.1, ht'.2.2.2] at he
    rw [he]
    have := eps_sq D M s b.2.1
    linear_combination (-(permSign t.perm s)) * this
  have hr : OrRel ((D.toXBulk hX).M (s : ℤ)) ((a.1, a.2, 1) : OrPt ((D.toXBulk hX).V (s : ℤ)))
      (b.1, b.2, D.sgnZ (s : ℤ) a.2.1 b.2.1 δ) := by
    refine ⟨(TauData.mem_Sset D).2 ⟨t, ht, hδt⟩, ?_⟩
    show (1 : ℤˣ) = D.sgnZ (s : ℤ) a.2.1 b.2.1 δ * D.sgnZ (s : ℤ) a.2.1 b.2.1 δ
    rw [sgn_sq]
  have := mk_gen_eq ((D.toXBulk hX).M (s : ℤ)) hr
  rw [Submodule.mkQ_apply, Submodule.mkQ_apply] at this
  rw [genQ, this, genQ, ← hsg, ← Submodule.Quotient.mk_smul]
  congr 1
  simp [gen, smul_single]

/-! ### `Ξ` is an isomorphism -/

/-- The inverse of `Ξ`: `e_y ↦ eps · [pre y]`. -/
noncomputable def ΞI (s : ℕ) : (Cover.YS (Hy hW hio hdim) s →₀ ℤ) →ₗ[ℤ] chains (D.toXBulk hX) (s : ℤ) :=
  Finsupp.linearCombination ℤ (fun y => eps D M s (pre hX hW hio hdim hS hU y).2.1 •
    genQ (D.toXBulk hX) (s : ℤ) (pre hX hW hio hdim hS hU y))

lemma Ξ_ΞI (s : ℕ) (x : Cover.YS (Hy hW hio hdim) s →₀ ℤ) :
    Ξ hX hW hio hdim hS hT s (ΞI hX hW hio hdim hS hU s x) = x := by
  have : Ξ hX hW hio hdim hS hT s ∘ₗ ΞI hX hW hio hdim hS hU s = LinearMap.id := by
    apply Finsupp.lhom_ext
    intro y c
    simp only [LinearMap.comp_apply, LinearMap.id_apply, ΞI, Finsupp.linearCombination_single, map_smul, Ξ_genQ,
      Θ_pre, smul_single, smul_eq_mul]
    congr 1
    linear_combination c * eps_sq D M s (pre hX hW hio hdim hS hU y).2.1
  exact LinearMap.congr_fun this x

include hI in
lemma ΞI_Ξ (s : ℕ) (x : chains (D.toXBulk hX) (s : ℤ)) :
    ΞI hX hW hio hdim hS hU s (Ξ hX hW hio hdim hS hT s x) = x := by
  have := chains_hom_ext (D.toXBulk hX) (q := (s : ℤ))
    (u := (ΞI hX hW hio hdim hS hU s ∘ₗ Ξ hX hW hio hdim hS hT s).toAddMonoidHom) (v := AddMonoidHom.id _) (fun a => by
      simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply, AddMonoidHom.id_apply, Ξ_genQ,
        ΞI, Finsupp.linearCombination_single]
      have hb := genQ_of_Θ hX hW hio hdim hS hT hI (Θ_pre hX hW hio hdim hS hU (Θ hX hW hio hdim hS a))
      rw [hb, smul_smul, smul_smul]
      convert one_smul ℤ (genQ (D.toXBulk hX) (s : ℤ) a) using 2
      have h1 := eps_sq D M s a.2.1
      have h2 := eps_sq D M s (pre hX hW hio hdim hS hU (Θ hX hW hio hdim hS a)).2.1
      linear_combination (eps D M s a.2.1 * eps D M s a.2.1) * h2 + h1)
  exact DFunLike.congr_fun this x

/-- **`Ξ` is an isomorphism** `C_s(X_bulk) ≃ ℤ[YS s]`. -/
noncomputable def Ξe (s : ℕ) : chains (D.toXBulk hX) (s : ℤ) ≃ₗ[ℤ] (Cover.YS (Hy hW hio hdim) s →₀ ℤ) :=
  LinearEquiv.ofLinearMap (Ξ hX hW hio hdim hS hT s) (ΞI hX hW hio hdim hS hU s)
    (LinearMap.ext fun x => Ξ_ΞI hX hW hio hdim hS hT hU s x) (LinearMap.ext fun x => ΞI_Ξ hX hW hio hdim hS hT hU hI s x)

/-! ### `Ξ` is a chain map -/

include hS in
lemma Ξ_aug (x : chains (D.toXBulk hX) 0) :
    Finsupp.linearCombination ℤ (Cover.yBd0 (Hy hW hio hdim) ℤ) (Ξ hX hW hio hdim hS hT 0 x) =
      single () (aug (D.toXBulk hX) x) := by
  have := chains_hom_ext (D.toXBulk hX) (q := 0)
    (u := (Finsupp.linearCombination ℤ (Cover.yBd0 (Hy hW hio hdim) ℤ) ∘ₗ Ξ hX hW hio hdim hS hT 0).toAddMonoidHom)
    (v := (Finsupp.lsingle () ∘ₗ aug (D.toXBulk hX)).toAddMonoidHom) (fun a => by
      simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply, Finsupp.lsingle_apply]
      show Finsupp.linearCombination ℤ _ (Ξ hX hW hio hdim hS hT 0 (genQ (D.toXBulk hX) ((0 : ℕ) : ℤ) a)) = _
      rw [Ξ_genQ, Finsupp.linearCombination_single, aug_genQ, eps_zero]
      simp [Cover.yBd0])
  exact DFunLike.congr_fun this x

variable (hB : chkBd D M)

lemma nf_nat (s : ℕ) : nf (s : ℤ) = s + 2 := by unfold nf; simp

include hB in
lemma Ξ_bd_genQ {s : ℕ} (hs : s ∈ ([0, 1] : List ℕ)) (a : SL(2, ℤ) × (D.toXBulk hX).V ((s : ℤ) + 1)) :
    Ξ hX hW hio hdim hS hT s (bd (D.toXBulk hX) (s : ℤ) (genQ (D.toXBulk hX) ((s : ℤ) + 1) a)) =
      Finsupp.linearCombination ℤ (Cover.yBdS (Hy hW hio hdim) ℤ (s := s))
        (Ξ hX hW hio hdim hS hT (s + 1) (genQ (D.toXBulk hX) ((s + 1 : ℕ) : ℤ) a)) := by
  have hmem : a.2.1 ∈ D.idxL ((s + 1 : ℕ) : ℤ) := by
    have := (D.mem_idxF).1 a.2.2
    push_cast; exact this
  obtain ⟨hB1, hB2, hB3⟩ := hB s hs a.2.1 hmem
  rw [Ξ_genQ (s := s + 1), Finsupp.linearCombination_single, genQ, bd_mk, bdF_gen, Units.val_one, one_smul]
  show ΞF hX hW hio hdim hS s _ = _
  rw [map_sum]
  simp only [ΞF_gen]
  rw [Cover.yBdS, Cover.altSum, Finset.smul_sum]
  have hn := nf_nat s
  set π := listPerm (nf (s : ℤ)) (posL D M (s + 1) a.2.1)
  have hπ : ∀ i : Fin (nf (s : ℤ)), (π i : ℕ) = (posL D M (s + 1) a.2.1).getD i 0 :=
    listPerm_apply _ _ (by rw [hn]; exact hB1) (by rw [hn]; exact hB2)
  let e : Fin (s + 2) ≃ Fin (nf (s : ℤ)) := (finCongr hn.symm).trans π
  refine (Fintype.sum_equiv e _ _ fun i => ?_).symm
  have hei : ((e i : Fin (nf (s : ℤ))) : ℕ) = (posL D M (s + 1) a.2.1).getD i 0 := by
    rw [show e i = π (finCongr hn.symm i) from rfl, hπ]; rfl
  obtain ⟨hv, hsg⟩ := hB3 i (List.mem_range.2 i.2)
  rw [smul_single, smul_eq_mul]
  congr 1
  · apply Subtype.ext
    funext m
    rw [Θ_apply]
    show a.1 • MPDatum.pt hW (vt hX hS a.2 (i.succAbove m)) = (a.1 * 1) • MPDatum.pt hW (vt hX hS _ m)
    rw [mul_one]
    congr 2
    apply Fin.ext
    show sv D M (s + 1) a.2.1 (i.succAbove m) = sv D M s (D.face (s + 1) a.2.1 (e i : ℕ)) m
    rw [hei, hv m (List.mem_range.2 m.2), val_succAbove]
  · show eps D M (s + 1) a.2.1 * (-1 : ℤ) ^ (i : ℕ) =
      ((unitOf (sgnPow (e i : ℕ)) : ℤˣ) : ℤ) * eps D M s (D.face (s + 1) a.2.1 (e i : ℕ))
    rw [unitOf_val (sgnPow_cases _), hei, hsg, sgnPow_eq]

include hB in
/-- **`Ξ` is a chain map** in degrees `s + 1 → s`, `s = 0, 1`. -/
lemma Ξ_bd {s : ℕ} (hs : s ∈ ([0, 1] : List ℕ)) (x : chains (D.toXBulk hX) ((s : ℤ) + 1)) :
    Ξ hX hW hio hdim hS hT s (bd (D.toXBulk hX) (s : ℤ) x) =
      Finsupp.linearCombination ℤ (Cover.yBdS (Hy hW hio hdim) ℤ (s := s)) (Ξ hX hW hio hdim hS hT (s + 1) x) := by
  have := chains_hom_ext (D.toXBulk hX) (q := (s : ℤ) + 1)
    (u := (Ξ hX hW hio hdim hS hT s ∘ₗ bd (D.toXBulk hX) (s : ℤ)).toAddMonoidHom)
    (v := (Finsupp.linearCombination ℤ (Cover.yBdS (Hy hW hio hdim) ℤ (s := s)) ∘ₗ
      Ξ hX hW hio hdim hS hT (s + 1)).toAddMonoidHom)
    (fun a => Ξ_bd_genQ hX hW hio hdim hS hT hB hs a)
  exact DFunLike.congr_fun this x

/-! ### The contraction over `ℤ` and its base change -/

/-- `E₀ : ℤ[Simp_0] ≃ C_0(X_bulk) ≃ ℤ[YS 0]`. -/
noncomputable def E0 : (Simp (D.toXBulk hX) 0 →₀ ℤ) ≃ₗ[ℤ] (Cover.YS (Hy hW hio hdim) 0 →₀ ℤ) :=
  (chainsEquivPerm (D.toXBulk hX) 0).symm.trans (Ξe hX hW hio hdim hS hT hU hI 0)

/-- `E₁ : ℤ[Simp_1] ≃ ℤ[YS 1]`. -/
noncomputable def E1 : (Simp (D.toXBulk hX) (0 + 1) →₀ ℤ) ≃ₗ[ℤ] (Cover.YS (Hy hW hio hdim) 1 →₀ ℤ) :=
  (chainsEquivPerm (D.toXBulk hX) (0 + 1)).symm.trans (Ξe hX hW hio hdim hS hT hU hI 1)

/-- `E₂ : ℤ[Simp_2] ≃ ℤ[YS 2]`. -/
noncomputable def E2 : (Simp (D.toXBulk hX) (0 + 1 + 1) →₀ ℤ) ≃ₗ[ℤ] (Cover.YS (Hy hW hio hdim) 2 →₀ ℤ) :=
  (chainsEquivPerm (D.toXBulk hX) (0 + 1 + 1)).symm.trans (Ξe hX hW hio hdim hS hT hU hI 2)

/-- The differentials of `ℤ[Y_•] → ℤ` on the concretely typed bases. -/
noncomputable def dY0 : (Cover.YS (Hy hW hio hdim) 0 →₀ ℤ) →ₗ[ℤ] (Unit →₀ ℤ) := (Cover.yCx (Hy hW hio hdim) ℤ).d 0
/-- `∂ : ℤ[YS 1] → ℤ[YS 0]`. -/
noncomputable def dY1 : (Cover.YS (Hy hW hio hdim) 1 →₀ ℤ) →ₗ[ℤ] (Cover.YS (Hy hW hio hdim) 0 →₀ ℤ) := (Cover.yCx (Hy hW hio hdim) ℤ).d 1
/-- `∂ : ℤ[YS 2] → ℤ[YS 1]`. -/
noncomputable def dY2 : (Cover.YS (Hy hW hio hdim) 2 →₀ ℤ) →ₗ[ℤ] (Cover.YS (Hy hW hio hdim) 1 →₀ ℤ) := (Cover.yCx (Hy hW hio hdim) ℤ).d 2

lemma d0_E (f : Simp (D.toXBulk hX) 0 →₀ ℤ) :
    dY0 hW hio hdim (E0 hX hW hio hdim hS hT hU hI f) = single () (augZ (D.toXBulk hX) f) :=
  Ξ_aug hX hW hio hdim hS hT (toChains (D.toXBulk hX) 0 f)

include hB in
lemma d1_E (f : Simp (D.toXBulk hX) (0 + 1) →₀ ℤ) :
    dY1 hW hio hdim (E1 hX hW hio hdim hS hT hU hI f) = E0 hX hW hio hdim hS hT hU hI (dZ (D.toXBulk hX) 0 f) := by
  show _ = Ξ hX hW hio hdim hS hT 0 (toChains (D.toXBulk hX) 0 (ofChains (D.toXBulk hX) 0
    (bd (D.toXBulk hX) 0 (toChains (D.toXBulk hX) (0 + 1) f))))
  rw [toChains_ofChains]
  exact (Ξ_bd hX hW hio hdim hS hT hB (s := 0) (by simp) _).symm

include hB in
lemma d2_E (f : Simp (D.toXBulk hX) (0 + 1 + 1) →₀ ℤ) :
    dY2 hW hio hdim (E2 hX hW hio hdim hS hT hU hI f) = E1 hX hW hio hdim hS hT hU hI (dZ (D.toXBulk hX) (0 + 1) f) := by
  show _ = Ξ hX hW hio hdim hS hT 1 (toChains (D.toXBulk hX) (0 + 1) (ofChains (D.toXBulk hX) (0 + 1)
    (bd (D.toXBulk hX) (0 + 1) (toChains (D.toXBulk hX) (0 + 1 + 1) f))))
  rw [toChains_ofChains]
  exact (Ξ_bd hX hW hio hdim hS hT hB (s := 1) (by simp) _).symm

/-! ### The contraction of `ℤ[Y_•] → ℤ` -/

omit hX in
lemma lc_eq_of {B N : Type} [AddCommGroup N] (L : (B →₀ ℤ) →ₗ[ℤ] N) :
    Finsupp.linearCombination ℤ (fun y => L (single y 1)) = L :=
  Finsupp.lhom_ext fun y n => by rw [Finsupp.linearCombination_single, ← map_smul, smul_single, smul_eq_mul, mul_one]

variable (c : ContractZ (D.toXBulk hX))

/-- The contraction on bases: `E c.z`, `E s₀ E⁻¹`, `E s₁ E⁻¹`, `0`. -/
noncomputable def hbZ : ∀ m, (Cover.yCx (Hy hW hio hdim) ℤ).B m → ((Cover.yCx (Hy hW hio hdim) ℤ).B (m + 1) →₀ ℤ)
  | 0 => fun _ => E0 hX hW hio hdim hS hT hU hI c.z
  | 1 => fun y => E1 hX hW hio hdim hS hT hU hI (c.s0 ((E0 hX hW hio hdim hS hT hU hI).symm (single y 1)))
  | 2 => fun y => E2 hX hW hio hdim hS hT hU hI (c.s1 ((E1 hX hW hio hdim hS hT hU hI).symm (single y 1)))
  | _ + 3 => fun _ => 0

lemma lc_hbZ1 (w : (Cover.yCx (Hy hW hio hdim) ℤ).B 1 →₀ ℤ) :
    Finsupp.linearCombination ℤ (hbZ hX hW hio hdim hS hT hU hI c 1) w = E1 hX hW hio hdim hS hT hU hI (c.s0 ((E0 hX hW hio hdim hS hT hU hI).symm w)) :=
  LinearMap.congr_fun (lc_eq_of (B := (Cover.yCx (Hy hW hio hdim) ℤ).B 1)
    ((E1 hX hW hio hdim hS hT hU hI).toLinearMap ∘ₗ c.s0 ∘ₗ (E0 hX hW hio hdim hS hT hU hI).symm.toLinearMap)) w

lemma lc_hbZ2 (w : (Cover.yCx (Hy hW hio hdim) ℤ).B 2 →₀ ℤ) :
    Finsupp.linearCombination ℤ (hbZ hX hW hio hdim hS hT hU hI c 2) w = E2 hX hW hio hdim hS hT hU hI (c.s1 ((E1 hX hW hio hdim hS hT hU hI).symm w)) :=
  LinearMap.congr_fun (lc_eq_of (B := (Cover.yCx (Hy hW hio hdim) ℤ).B 2)
    ((E2 hX hW hio hdim hS hT hU hI).toLinearMap ∘ₗ c.s1 ∘ₗ (E1 hX hW hio hdim hS hT hU hI).symm.toLinearMap)) w

omit hX in
lemma isEmpty_YS (m : ℕ) : IsEmpty (Cover.YS (Hy hW hio hdim) (m + 3)) := by
  refine ⟨fun x => ?_⟩
  obtain ⟨o, ho⟩ := x.2.2
  have := Cover.card_le_three (Hy hW hio hdim) x.1 (Cover.injective_of_strictMono x.2.1) o ho
  omega

omit hX in
lemma d_single_B {m : ℕ} (z : (Cover.yCx (Hy hW hio hdim) ℤ).B (m + 1)) : (Cover.yCx (Hy hW hio hdim) ℤ).d m (single z 1) = (Cover.yCx (Hy hW hio hdim) ℤ).bd m z := by
  rw [BCx.d_single, one_smul]

omit hX in
lemma yd0_single (y : Cover.YS (Hy hW hio hdim) 0) : dY0 hW hio hdim (single y 1) = single () 1 :=
  d_single_B hW hio hdim (m := 0) y

omit hX in
lemma yd1_single (y : Cover.YS (Hy hW hio hdim) 1) : dY1 hW hio hdim (single y 1) = (Cover.yCx (Hy hW hio hdim) ℤ).bd 1 y :=
  d_single_B hW hio hdim (m := 1) y

omit hX in
lemma yd2_single (y : Cover.YS (Hy hW hio hdim) 2) : dY2 hW hio hdim (single y 1) = (Cover.yCx (Hy hW hio hdim) ℤ).bd 2 y :=
  d_single_B hW hio hdim (m := 2) y

lemma lc_hbZ0 (u : (Cover.yCx (Hy hW hio hdim) ℤ).B 0) : Finsupp.linearCombination ℤ (hbZ hX hW hio hdim hS hT hU hI c 0) (single u 1) = hbZ hX hW hio hdim hS hT hU hI c 0 u := by
  rw [Finsupp.linearCombination_single, one_smul]

lemma lc_hbZ0' (b : (Cover.yCx (Hy hW hio hdim) ℤ).B 1) : Finsupp.linearCombination ℤ (hbZ hX hW hio hdim hS hT hU hI c 0) ((Cover.yCx (Hy hW hio hdim) ℤ).bd 0 b) = E0 hX hW hio hdim hS hT hU hI c.z :=
  lc_hbZ0 hX hW hio hdim hS hT hU hI c ()

include hB in
/-- **The contraction of `ℤ[Y_•] → ℤ`**, transported from `X_bulk` through `Ξ`. -/
noncomputable def contrZ : BContr ℤ (Cover.yCx (Hy hW hio hdim) ℤ) where
  hb := hbZ hX hW hio hdim hS hT hU hI c
  h0 b := by
    change Unit at b
    cases b
    exact (d0_E hX hW hio hdim hS hT hU hI c.z).trans (by rw [c.aug_z]; rfl)
  hs m b := by
    match m, b with
    | 0, b => -- degree `YS 0`
      let y : Cover.YS (Hy hW hio hdim) 0 := b
      have haug : augZ (D.toXBulk hX) ((E0 hX hW hio hdim hS hT hU hI).symm (single y 1)) = 1 := by
        have h := d0_E hX hW hio hdim hS hT hU hI ((E0 hX hW hio hdim hS hT hU hI).symm (single y 1))
        rw [LinearEquiv.apply_symm_apply, yd0_single] at h
        exact (Finsupp.single_injective () h).symm
      have key : dY1 hW hio hdim (E1 hX hW hio hdim hS hT hU hI (c.s0 ((E0 hX hW hio hdim hS hT hU hI).symm (single y 1)))) + E0 hX hW hio hdim hS hT hU hI c.z = single y 1 := by
        rw [d1_E hX hW hio hdim hS hT hU hI hB]
        have := LinearMap.congr_fun c.d_s0 ((E0 hX hW hio hdim hS hT hU hI).symm (single y 1))
        simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.sub_apply, LinearMap.id_apply,
          LinearMap.smulRight_apply, haug, one_smul] at this
        rw [this, map_sub, LinearEquiv.apply_symm_apply, sub_add_cancel]
      rw [lc_hbZ0']
      exact key
    | 1, b => -- degree `YS 1`
      let y : Cover.YS (Hy hW hio hdim) 1 := b
      have hd : (E0 hX hW hio hdim hS hT hU hI).symm (dY1 hW hio hdim (single y 1)) = dZ (D.toXBulk hX) 0 ((E1 hX hW hio hdim hS hT hU hI).symm (single y 1)) := by
        have h := d1_E hX hW hio hdim hS hT hU hI hB ((E1 hX hW hio hdim hS hT hU hI).symm (single y 1))
        rw [LinearEquiv.apply_symm_apply] at h
        rw [h, LinearEquiv.symm_apply_apply]
      have key : dY2 hW hio hdim (E2 hX hW hio hdim hS hT hU hI (c.s1 ((E1 hX hW hio hdim hS hT hU hI).symm (single y 1)))) +
          E1 hX hW hio hdim hS hT hU hI (c.s0 ((E0 hX hW hio hdim hS hT hU hI).symm (dY1 hW hio hdim (single y 1)))) = single y 1 := by
        rw [d2_E hX hW hio hdim hS hT hU hI hB, hd]
        have := LinearMap.congr_fun c.d_s1 ((E1 hX hW hio hdim hS hT hU hI).symm (single y 1))
        simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.sub_apply, LinearMap.id_apply] at this
        rw [this, map_sub, LinearEquiv.apply_symm_apply, sub_add_cancel]
      rw [yd1_single] at key
      rw [lc_hbZ1]
      exact key
    | 2, b => -- degree `YS 2`
      let y : Cover.YS (Hy hW hio hdim) 2 := b
      have hd : (E1 hX hW hio hdim hS hT hU hI).symm (dY2 hW hio hdim (single y 1)) =
          dZ (D.toXBulk hX) (0 + 1) ((E2 hX hW hio hdim hS hT hU hI).symm (single y 1)) := by
        have h := d2_E hX hW hio hdim hS hT hU hI hB ((E2 hX hW hio hdim hS hT hU hI).symm (single y 1))
        rw [LinearEquiv.apply_symm_apply] at h
        rw [h, LinearEquiv.symm_apply_apply]
      have key : E2 hX hW hio hdim hS hT hU hI (c.s1 ((E1 hX hW hio hdim hS hT hU hI).symm (dY2 hW hio hdim (single y 1)))) = single y 1 := by
        rw [hd]
        have := LinearMap.congr_fun c.s1_d ((E2 hX hW hio hdim hS hT hU hI).symm (single y 1))
        simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.id_apply] at this
        rw [this, LinearEquiv.apply_symm_apply]
      rw [yd2_single] at key
      rw [lc_hbZ2]
      have h0 : (Cover.yCx (Hy hW hio hdim) ℤ).d (2 + 1) (hbZ hX hW hio hdim hS hT hU hI c (2 + 1) b) = 0 := map_zero _
      rw [h0]
      exact (zero_add _).trans key
    | m + 3, b => exact (isEmpty_YS hW hio hdim m).elim b

end Bridge

/-! ### Base change `ℤ → k` of a complex on bases -/

section BaseChange

variable {k : Type} [CommRing k]

/-- The base change `C ⊗ k` of a complex of free `ℤ`-modules on bases. -/
noncomputable abbrev bcBCx (C : BCx ℤ) (k : Type) [CommRing k] : BCx k where
  B := C.B
  bd m b := castF (C.bd m b)
  bdd m b := by
    have h := C.bdd m b
    have e : ∀ x : C.B (m + 1) →₀ ℤ, Finsupp.linearCombination k (fun b => castF (R := k) (C.bd m b)) (castF x) =
        castF (Finsupp.linearCombination ℤ (C.bd m) x) := by
      intro x
      induction x using Finsupp.induction_linear with
      | zero => simp
      | add x y hx hy => rw [map_add, map_add, hx, hy, map_add, map_add]
      | single b n => rw [castF_single, Finsupp.linearCombination_single, Finsupp.linearCombination_single, castF_zsmul]
    rw [e, h, map_zero]

lemma bc_d (C : BCx ℤ) (m : ℕ) (x : C.B (m + 1) →₀ ℤ) : (bcBCx C k).d m (castF x) = castF (C.d m x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, map_add, map_add]
  | single b n =>
    rw [castF_single, BCx.d_single, BCx.d_single, castF_zsmul]

lemma bc_lc {B B' : Type} (h : B → (B' →₀ ℤ)) (x : B →₀ ℤ) :
    Finsupp.linearCombination k (fun b => castF (R := k) (h b)) (castF x) = castF (Finsupp.linearCombination ℤ h x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, map_add, map_add]
  | single b n => rw [castF_single, Finsupp.linearCombination_single, Finsupp.linearCombination_single, castF_zsmul]

/-- **A contraction over `ℤ` base-changes to one over `k`.** -/
noncomputable def BContr.bc {C : BCx ℤ} (c : BContr ℤ C) (k : Type) [CommRing k] : BContr k (bcBCx C k) where
  hb m b := castF (c.hb m b)
  h0 b := by
    rw [bc_d, c.h0, castF_single, Int.cast_one]
  hs m b := by
    show (bcBCx C k).d (m + 1) (castF (c.hb (m + 1) b)) +
      Finsupp.linearCombination k (fun b => castF (R := k) (c.hb m b)) (castF (C.bd m b)) = single b 1
    rw [bc_d, bc_lc, ← map_add, c.hs, castF_single, Int.cast_one]

variable {G : Type} [Group G] {C : Cover.{0} G} (H : C.Hyp)

/-- `k[Y_•] → k` is the base change of `ℤ[Y_•] → ℤ`. -/
lemma yCx_eq_bc (k : Type) [CommRing k] : Cover.yCx H k = bcBCx (Cover.yCx H ℤ) k := by
  have e : ∀ m (b : Cover.yB H (m + 1)), Cover.yBd H k m b = castF (Cover.yBd H ℤ m b) := by
    intro m b
    rcases m with _ | m
    · show single () 1 = castF (single () 1); rw [castF_single, Int.cast_one]
    · show Cover.altSum k _ = castF (Cover.altSum ℤ _)
      simp only [Cover.altSum, map_sum, castF_single, Int.cast_pow, Int.cast_neg, Int.cast_one]
  unfold Cover.yCx bcBCx
  congr 1
  funext m b
  exact e m b

end BaseChange

/-! ### The bridge theorem -/

/-- **(AC) from `X_bulk`** ([C] `thm:mpcomparison` (5)): if the finite checks hold, `X_bulk` (the oriented chains of the
dump) is acyclic and has a vertex, then the augmented ordered chain complex `k[Y_•] → k` of the MP vertex scheme is
acyclic over every commutative ring `k`. -/
theorem yCx_acyclic_of_bridge (hX : D.XAx) (hW : M.WF) (hio : M.ioB = true) (hdim : M.dimB = true)
    (hC : Checks D M) (hA : XBulk.Acyclic (D.toXBulk hX)) (v : (D.toXBulk hX).V 0) (k : Type) [CommRing k] :
    (Cover.yCx (MPDatum.hyp hW hio hdim) k).toCx.Acyclic := by
  obtain ⟨hS, hT, hI, hU, hB⟩ := hC
  obtain ⟨c⟩ := contractZ hA v
  rw [yCx_eq_bc]
  exact (BContr.bc (contrZ hX hW hio hdim hS hT hU hI hB c) k).acyclic

end ACBridge

end GL2.MP
