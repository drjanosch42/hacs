import GL2.Model.Oriented
import GL2.Checkers.TauCheck

/-!
# M2-a / M2-b (L18-1): the `TauData` tie -- the checker's keys are the key model's canonical keys

[GT] `thm:keymodel-tie`, `thm:permsign`.  L16-4's `obl:tau` checker (`GL2.TauChk`) names an oriented simplex
`γ · ι(σ)` by `TauData.canon`, computed on the dumped integer data (`M22`, transporter lists with vertex
correspondences).  L17-3 proved that the abstract canonical key `Ku.canon` of a `KeyModel` is a faithful coordinate
system.  This file closes the gap (L16-4's limit 1):

* `M22.toSL`, `M22.lexLt_iff` -- the dumped matrices *are* elements of `SL(2, ℤ)` (determinant one), and `lexLt` is
  the order of `Ku.lexKey`;
* `TauData.KeyAx D q` -- the oriented-key-model axioms on the stored finite data in degree `q` (groupoid laws of the
  transporter lists, constancy of the orbit representative, sign multiplicativity, sign consistency, no inversions);
  `TauData.keyModelB` decides it;
* `TauData.toKeyModel`, `TauData.keyModelB_sound` -- an accepted dump **is** an `OrKeyModel` whose transporter sets,
  representatives and signs are exactly the stored ones;
* `TauData.canon_eq_canon` -- `D.canon γ q σ` is `Ku.orCanon` of that model at `(γ, σ)`: the same representative,
  the same lexicographic minimum, and the same orientation sign;
* `permSign_eq_sign`, `permSign_mul` -- `permSign` is `Equiv.Perm.sign` (in the model's dimensions `q ≤ 2`), hence
  multiplicative; `TauData.permSign_eq_sgn` -- the model's sign of a stored transporter is its `permSign`.

The hypotheses are the decidable `KeyAx`, discharged on the real `p = 2` dump by `decide +kernel`
(`GL2.TauChk.Data.tauP2_keyModelB`).  Nothing hard-wires the 7-vertex model.
-/

namespace GL2.TauChk

open scoped MatrixGroups

/-! ### `M22` and `SL(2, ℤ)` -/

namespace M22

/-- The integer matrix of an `M22`. -/
def toMat (x : M22) : Matrix (Fin 2) (Fin 2) ℤ := !![x.a, x.b; x.c, x.d]

lemma det_toMat (x : M22) : (toMat x).det = x.det := by
  simp [toMat, Matrix.det_fin_two_of, det]

lemma mul_def (x y : M22) : x * y =
    ⟨x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d⟩ := rfl

lemma toMat_mul (x y : M22) : toMat (x * y) = toMat x * toMat y := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [toMat, mul_def, Matrix.mul_apply, Fin.sum_univ_two]

lemma det_mul (x y : M22) : (x * y).det = x.det * y.det := by
  rw [← det_toMat, toMat_mul, Matrix.det_mul, det_toMat, det_toMat]

lemma toMat_injective : Function.Injective toMat := by
  rintro ⟨a, b, c, d⟩ ⟨a', b', c', d'⟩ h
  have h1 := congrFun (congrFun h 0) 0
  have h2 := congrFun (congrFun h 0) 1
  have h3 := congrFun (congrFun h 1) 0
  have h4 := congrFun (congrFun h 1) 1
  simp [toMat] at h1 h2 h3 h4
  simp [h1, h2, h3, h4]

/-- **`M22.toSL`**: a determinant-one `M22` as an element of `SL(2, ℤ)` (`1` off determinant one). -/
def toSL (x : M22) : SL(2, ℤ) := if h : x.det = 1 then ⟨toMat x, by rw [det_toMat, h]⟩ else 1

lemma toSL_coe {x : M22} (h : x.det = 1) : ((toSL x : SL(2, ℤ)) : Matrix (Fin 2) (Fin 2) ℤ) = toMat x := by
  simp [toSL, h]

lemma toSL_mul {x y : M22} (hx : x.det = 1) (hy : y.det = 1) : toSL (x * y) = toSL x * toSL y := by
  have hxy : (x * y).det = 1 := by rw [det_mul, hx, hy, one_mul]
  apply Subtype.ext
  rw [Matrix.SpecialLinearGroup.coe_mul, toSL_coe hxy, toSL_coe hx, toSL_coe hy, toMat_mul]

lemma toSL_injOn {x y : M22} (hx : x.det = 1) (hy : y.det = 1) (h : toSL x = toSL y) : x = y := by
  apply toMat_injective
  rw [← toSL_coe hx, ← toSL_coe hy, h]

lemma det_adj (x : M22) : x.adj.det = x.det := by simp [adj, det]; ring

lemma toSL_adj {x : M22} (hx : x.det = 1) : toSL x.adj = (toSL x)⁻¹ := by
  apply Subtype.ext
  rw [Matrix.SpecialLinearGroup.coe_inv, toSL_coe ((det_adj x).trans hx), toSL_coe hx]
  simp [toMat, adj, Matrix.adjugate_fin_two_of]

lemma toSL_one : toSL ⟨1, 0, 0, 1⟩ = 1 := by
  apply Subtype.ext
  rw [toSL_coe (by decide)]
  simp [toMat, Matrix.one_fin_two]

lemma toSL_negOne : toSL ⟨-1, 0, 0, -1⟩ = -1 := by
  apply Subtype.ext
  rw [toSL_coe (by decide), Matrix.SpecialLinearGroup.coe_neg, Matrix.SpecialLinearGroup.coe_one]
  simp [toMat, Matrix.one_fin_two]

/-- The lexicographic key of an `M22` on `(a, b, c, d)`. -/
def lexKeyM (x : M22) : Ku.KeyT := toLex (x.a, toLex (x.b, toLex (x.c, x.d)))

lemma lexKey_toSL {x : M22} (hx : x.det = 1) : Ku.lexKey (toSL x) = lexKeyM x := by
  simp [Ku.lexKey, lexKeyM, toSL_coe hx, toMat]

lemma lexKeyM_injective : Function.Injective lexKeyM := by
  rintro ⟨a, b, c, d⟩ ⟨a', b', c', d'⟩ h
  simp only [lexKeyM, toLex_inj, Prod.mk.injEq] at h
  simp [h]

/-- **`lexLt` is the order of `lexKey`** (`hecke`'s `lex_less`). -/
lemma lexLt_iff (x y : M22) : lexLt x y = true ↔ lexKeyM x < lexKeyM y := by
  simp only [lexLt, lexKeyM, Prod.Lex.toLex_lt_toLex]
  split_ifs <;> simp_all

end M22

/-! ### Signs -/

/-- `±1 : ℤ` as a unit (`-1` off `1`). -/
def unitOf (z : ℤ) : ℤˣ := if z = 1 then 1 else -1

lemma unitOf_val {z : ℤ} (h : z = 1 ∨ z = -1) : ((unitOf z : ℤˣ) : ℤ) = z := by
  rcases h with rfl | rfl <;> decide

lemma unitOf_mul {z w : ℤ} (hz : z = 1 ∨ z = -1) (hw : w = 1 ∨ w = -1) : unitOf (z * w) = unitOf z * unitOf w := by
  rcases hz with rfl | rfl <;> rcases hw with rfl | rfl <;> decide

lemma unitOf_one : unitOf 1 = 1 := by decide

lemma permSign_cases (l : List ℕ) (q : ℕ) : permSign l q = 1 ∨ permSign l q = -1 := by
  unfold permSign; dsimp only; split <;> simp

lemma permSign_zero (l : List ℕ) : permSign l 0 = 1 := by
  simp [permSign]

/-- A permutation of `Fin n` as a vertex-slot list. -/
def ofPerm {n : ℕ} (σ : Equiv.Perm (Fin n)) : List ℕ := List.ofFn (fun i => (σ i : ℕ))

/-- **`permSign` is the sign of the permutation** in the model's dimensions `q ≤ 2`. -/
theorem permSign_eq_sign (q : ℕ) (hq : q ≤ 2) (σ : Equiv.Perm (Fin (q + 1))) :
    permSign (ofPerm σ) q = (Equiv.Perm.sign σ : ℤ) := by
  interval_cases q
  · revert σ; decide
  · revert σ; decide
  · revert σ; decide

/-- **`permSign` is multiplicative** (`Equiv.Perm.sign_mul`). -/
theorem permSign_mul (q : ℕ) (hq : q ≤ 2) (σ τ : Equiv.Perm (Fin (q + 1))) :
    permSign (ofPerm (σ * τ)) q = permSign (ofPerm σ) q * permSign (ofPerm τ) q := by
  rw [permSign_eq_sign q hq, permSign_eq_sign q hq, permSign_eq_sign q hq, Equiv.Perm.sign_mul, Units.val_mul]

/-! ### The stored model in degree `q` -/

namespace TauData

variable (D : TauData)

/-- The stored `q`-simplex indices. -/
def idxL (q : ℤ) : List ℕ := (D.simplices.filter (fun x => (x.1 : ℤ) = q)).map (·.2.1)

/-- The stored `q`-simplex indices as a `Finset`. -/
def idxF (q : ℤ) : Finset ℕ := (D.idxL q).toFinset

/-- The model simplices of degree `q` (empty outside the stored degrees, in particular for `q < 0`). -/
abbrev Vz (q : ℤ) := ↥(D.idxF q)

/-- The stored transporters of degree `q`. -/
def trZ (q : ℤ) : List Tr := D.trs.filter (fun t => (t.q : ℤ) = q)

/-- The stored transporter list `S(a, b)` of degree `q`. -/
def trSetZ (q : ℤ) (a b : ℕ) : List Tr := D.trs.filter (fun t => (t.q : ℤ) = q ∧ t.s1 = a ∧ t.s2 = b)

/-- The stored orbit representative. -/
def repZ (q : ℤ) (σ : ℕ) : ℕ := D.rep q.toNat σ

lemma mem_idxF {q : ℤ} {σ : ℕ} : σ ∈ D.idxF q ↔ σ ∈ D.idxL q := List.mem_toFinset

lemma mem_trSetZ {q : ℤ} {a b : ℕ} {t : Tr} :
    t ∈ D.trSetZ q a b ↔ t ∈ D.trs ∧ (t.q : ℤ) = q ∧ t.s1 = a ∧ t.s2 = b := by
  simp [trSetZ]

lemma mem_trZ {q : ℤ} {t : Tr} : t ∈ D.trZ q ↔ t ∈ D.trs ∧ (t.q : ℤ) = q := by
  simp [trZ]

lemma trZ_of_trSetZ {q : ℤ} {a b : ℕ} {t : Tr} (h : t ∈ D.trSetZ q a b) : t ∈ D.trZ q := by
  rw [mem_trSetZ] at h; rw [mem_trZ]; exact ⟨h.1, h.2.1⟩

/-- **The oriented-key-model axioms on the stored data in degree `q`** (decidable). -/
def KeyAx (q : ℤ) : Prop :=
  (∀ σ ∈ D.idxL q, D.repZ q σ ∈ D.idxL q) ∧
  (∀ σ ∈ D.idxL q, ∃ t ∈ D.trSetZ q σ σ, t.g = ⟨1, 0, 0, 1⟩) ∧
  (∀ σ ∈ D.idxL q, D.trSetZ q σ (D.repZ q σ) ≠ []) ∧
  (∀ t ∈ D.trZ q, t.g.det = 1 ∧ D.repZ q t.s1 = D.repZ q t.s2 ∧ ∃ u ∈ D.trSetZ q t.s2 t.s1, u.g = t.g.adj) ∧
  (∀ t ∈ D.trZ q, ∀ u ∈ D.trZ q, t.s2 = u.s1 → ∃ v ∈ D.trSetZ q t.s1 u.s2,
    v.g = t.g * u.g ∧ permSign v.perm q.toNat = permSign t.perm q.toNat * permSign u.perm q.toNat) ∧
  (∀ t ∈ D.trZ q, ∀ u ∈ D.trZ q, t.s1 = u.s1 → t.s2 = u.s2 → t.g = u.g →
    permSign t.perm q.toNat = permSign u.perm q.toNat) ∧
  (∀ t ∈ D.trZ q, t.s1 = t.s2 → permSign t.perm q.toNat = 1)

instance (q : ℤ) : Decidable (D.KeyAx q) := by unfold KeyAx; infer_instance

/-- **`TauData.keyModelB`**: decides the key-model axioms of degree `q` on the stored data. -/
def keyModelB (q : ℤ) : Bool := decide (D.KeyAx q)

/-- The stored transporter set `S(a, b)` in `SL(2, ℤ)`. -/
def Sset (q : ℤ) (a b : ℕ) : Finset SL(2, ℤ) := ((D.trSetZ q a b).map (fun t => M22.toSL t.g)).toFinset

/-- The stored sign of `γ ∈ S(a, b)`: the `permSign` of the stored transporter with matrix `γ`. -/
def sgnZ (q : ℤ) (a b : ℕ) (γ : SL(2, ℤ)) : ℤˣ :=
  match (D.trSetZ q a b).find? (fun t => decide (M22.toSL t.g = γ)) with
  | some t => unitOf (permSign t.perm q.toNat)
  | none => 1

lemma mem_Sset {q : ℤ} {a b : ℕ} {γ : SL(2, ℤ)} : γ ∈ D.Sset q a b ↔ ∃ t ∈ D.trSetZ q a b, M22.toSL t.g = γ := by
  simp [Sset]

variable {D}

lemma sgnZ_eq {q : ℤ} (hK : D.KeyAx q) {a b : ℕ} {t : Tr} (ht : t ∈ D.trSetZ q a b) :
    D.sgnZ q a b (M22.toSL t.g) = unitOf (permSign t.perm q.toNat) := by
  unfold sgnZ
  cases hf : (D.trSetZ q a b).find? (fun u => decide (M22.toSL u.g = M22.toSL t.g)) with
  | none =>
    rw [List.find?_eq_none] at hf
    exact absurd (by simp) (hf t ht)
  | some f =>
    have hfm := List.mem_of_find?_eq_some hf
    have hfp := List.find?_some hf
    simp only [decide_eq_true_eq] at hfp
    have hdf := (hK.2.2.2.1 f (trZ_of_trSetZ D hfm)).1
    have hdt := (hK.2.2.2.1 t (trZ_of_trSetZ D ht)).1
    have hg := M22.toSL_injOn hdf hdt hfp
    rw [mem_trSetZ] at hfm ht
    simp only
    rw [hK.2.2.2.2.2.1 f (by rw [mem_trZ]; exact ⟨hfm.1, hfm.2.1⟩) t (by rw [mem_trZ]; exact ⟨ht.1, ht.2.1⟩)
      (hfm.2.2.1.trans ht.2.2.1.symm) (hfm.2.2.2.trans ht.2.2.2.symm) hg]

/-- **`TauData.toKeyModel`**: an accepted dump is an oriented key model on its `q`-simplices. -/
def toKeyModel (q : ℤ) (hK : D.KeyAx q) : Ku.OrKeyModel (D.Vz q) where
  rep σ := ⟨D.repZ q σ.1, (D.mem_idxF).2 (hK.1 σ.1 ((D.mem_idxF).1 σ.2))⟩
  S a b := D.Sset q a.1 b.1
  one_mem σ := by
    obtain ⟨t, ht, hg⟩ := hK.2.1 σ.1 ((D.mem_idxF).1 σ.2)
    exact (D.mem_Sset).2 ⟨t, ht, by rw [hg, M22.toSL_one]⟩
  mul_mem {a b c t u} ht hu := by
    obtain ⟨t0, ht0, rfl⟩ := (D.mem_Sset).1 ht
    obtain ⟨u0, hu0, rfl⟩ := (D.mem_Sset).1 hu
    have e : t0.s2 = u0.s1 := by
      rw [mem_trSetZ] at ht0 hu0; rw [ht0.2.2.2, hu0.2.2.1]
    obtain ⟨v, hv, hvg, -⟩ := hK.2.2.2.2.1 t0 (trZ_of_trSetZ D ht0) u0 (trZ_of_trSetZ D hu0) e
    have hv' : v ∈ D.trSetZ q a.1 c.1 := by
      rw [mem_trSetZ] at hv ht0 hu0 ⊢; exact ⟨hv.1, hv.2.1, hv.2.2.1.trans ht0.2.2.1, hv.2.2.2.trans hu0.2.2.2⟩
    refine (D.mem_Sset).2 ⟨v, hv', ?_⟩
    rw [hvg, M22.toSL_mul (hK.2.2.2.1 t0 (trZ_of_trSetZ D ht0)).1 (hK.2.2.2.1 u0 (trZ_of_trSetZ D hu0)).1]
  inv_mem {a b t} ht := by
    obtain ⟨t0, ht0, rfl⟩ := (D.mem_Sset).1 ht
    obtain ⟨hd, -, u, hu, hug⟩ := hK.2.2.2.1 t0 (trZ_of_trSetZ D ht0)
    have hu' : u ∈ D.trSetZ q b.1 a.1 := by
      rw [mem_trSetZ] at hu ht0 ⊢; exact ⟨hu.1, hu.2.1, hu.2.2.1.trans ht0.2.2.2, hu.2.2.2.trans ht0.2.2.1⟩
    exact (D.mem_Sset).2 ⟨u, hu', by rw [hug, M22.toSL_adj hd]⟩
  rep_mem σ := by
    obtain ⟨t, ht⟩ := List.exists_mem_of_ne_nil _ (hK.2.2.1 σ.1 ((D.mem_idxF).1 σ.2))
    exact ⟨_, (D.mem_Sset).2 ⟨t, ht, rfl⟩⟩
  rep_eq {a b} h := by
    obtain ⟨γ, hγ⟩ := h
    obtain ⟨t0, ht0, -⟩ := (D.mem_Sset).1 hγ
    have := (hK.2.2.2.1 t0 (trZ_of_trSetZ D ht0)).2.1
    rw [mem_trSetZ] at ht0
    apply Subtype.ext
    simp only
    rw [← ht0.2.2.1, ← ht0.2.2.2, this]
  sgn a b γ := D.sgnZ q a.1 b.1 γ
  sgn_mul {a b c t u} ht hu := by
    obtain ⟨t0, ht0, rfl⟩ := (D.mem_Sset).1 ht
    obtain ⟨u0, hu0, rfl⟩ := (D.mem_Sset).1 hu
    have e : t0.s2 = u0.s1 := by
      rw [mem_trSetZ] at ht0 hu0; rw [ht0.2.2.2, hu0.2.2.1]
    obtain ⟨v, hv, hvg, hvs⟩ := hK.2.2.2.2.1 t0 (trZ_of_trSetZ D ht0) u0 (trZ_of_trSetZ D hu0) e
    have hv' : v ∈ D.trSetZ q a.1 c.1 := by
      rw [mem_trSetZ] at hv ht0 hu0 ⊢; exact ⟨hv.1, hv.2.1, hv.2.2.1.trans ht0.2.2.1, hv.2.2.2.trans hu0.2.2.2⟩
    rw [← M22.toSL_mul (hK.2.2.2.1 t0 (trZ_of_trSetZ D ht0)).1 (hK.2.2.2.1 u0 (trZ_of_trSetZ D hu0)).1, ← hvg,
      sgnZ_eq hK hv', sgnZ_eq hK ht0, sgnZ_eq hK hu0, hvs, unitOf_mul (permSign_cases _ _) (permSign_cases _ _)]
  noInv {σ t} ht := by
    obtain ⟨t0, ht0, rfl⟩ := (D.mem_Sset).1 ht
    show D.sgnZ q σ.1 σ.1 _ = 1
    rw [sgnZ_eq hK ht0]
    have := ht0
    rw [mem_trSetZ] at this
    rw [hK.2.2.2.2.2.2 t0 (trZ_of_trSetZ D ht0) (this.2.2.1.trans this.2.2.2.symm), unitOf_one]

variable (D)

/-- **`TauData.keyModelB_sound`**: if the checker accepts degree `q`, the stored representatives, transporter sets and
signs **are** an oriented key model (the `KeyModel` axioms, `sgn_mul`, `noInv`). -/
theorem keyModelB_sound (q : ℤ) (h : D.keyModelB q = true) :
    ∃ M : Ku.OrKeyModel (D.Vz q), (∀ a b, M.S a b = D.Sset q a.1 b.1) ∧ (∀ σ, (M.rep σ).1 = D.repZ q σ.1) ∧
      ∀ a b γ, M.sgn a b γ = D.sgnZ q a.1 b.1 γ :=
  ⟨D.toKeyModel q (of_decide_eq_true h), fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

/-- **`TauData.permSign_eq_sgn`**: the model's sign of a stored transporter is its `permSign`. -/
theorem permSign_eq_sgn (q : ℤ) (hK : D.KeyAx q) (a b : D.Vz q) {t : Tr} (ht : t ∈ D.trSetZ q a.1 b.1) :
    ((D.toKeyModel q hK).sgn a b (M22.toSL t.g) : ℤ) = permSign t.perm q.toNat := by
  show ((D.sgnZ q a.1 b.1 (M22.toSL t.g) : ℤˣ) : ℤ) = _
  rw [sgnZ_eq hK ht, unitOf_val (permSign_cases _ _)]

/-! ### `canon = canon` -/

/-- One step of `TauData.canon`'s fold. -/
def pickStep (γ : M22) (q : ℕ) (acc : Option (M22 × ℤ)) (t : Tr) : Option (M22 × ℤ) :=
  let cand := γ * t.g
  match acc with
  | none => some (cand, permSign t.perm q)
  | some (best, sg) => if M22.lexLt cand best then some (cand, permSign t.perm q) else some (best, sg)

/-- The fold invariant: the accumulator is the lexicographic minimum of what has been read. -/
def PickInv (γ : M22) (q : ℕ) (acc : Option (M22 × ℤ)) (S : List Tr) : Prop :=
  (acc = none ∧ S = []) ∨
    ∃ t ∈ S, acc = some (γ * t.g, permSign t.perm q) ∧ ∀ u ∈ S, M22.lexKeyM (γ * t.g) ≤ M22.lexKeyM (γ * u.g)

lemma pickInv_foldl (γ : M22) (q : ℕ) :
    ∀ (l : List Tr) (acc : Option (M22 × ℤ)) (S : List Tr), PickInv γ q acc S →
      PickInv γ q (l.foldl (pickStep γ q) acc) (S ++ l) := by
  intro l
  induction l with
  | nil => intro acc S h; simpa using h
  | cons a l ih =>
    intro acc S h
    rw [List.foldl_cons, ← List.singleton_append, ← List.append_assoc]
    apply ih
    rcases h with ⟨rfl, rfl⟩ | ⟨t, ht, rfl, hmin⟩
    · right; exact ⟨a, by simp, rfl, by simp⟩
    · right
      simp only [pickStep]
      split_ifs with hlt
      · rw [M22.lexLt_iff] at hlt
        refine ⟨a, by simp, rfl, ?_⟩
        intro u hu
        rcases List.mem_append.1 hu with hu | hu
        · exact le_of_lt (lt_of_lt_of_le hlt (hmin u hu))
        · simp at hu; rw [hu]
      · rw [M22.lexLt_iff, not_lt] at hlt
        refine ⟨t, List.mem_append_left _ ht, rfl, ?_⟩
        intro u hu
        rcases List.mem_append.1 hu with hu | hu
        · exact hmin u hu
        · simp at hu; rw [hu]; exact hlt

lemma canon_eq_fold (γ : M22) (q σ : ℕ) :
    D.canon γ q σ =
      match (D.trSet q σ (D.rep q σ)).foldl (pickStep γ q) none with
      | some (best, sg) => ((q, D.rep q σ, best), sg)
      | none => ((q, σ, γ), 1) := rfl

lemma trSet_eq (q a b : ℕ) : D.trSet q a b = D.trSetZ q a b := by
  unfold trSet trSetZ; congr 1; funext t; simp

/-- **`TauData.canon_eq_canon`**: on an accepted degree, the checker's canonical key of `γ · ι(σ)` is the key model's
oriented canonical key: the same orbit representative, the same lexicographic minimum (`lexKeyM` of the stored
matrix is `Ku.lexKey` of the `SL(2, ℤ)` element), and the same orientation sign. -/
theorem canon_eq_canon (q : ℕ) (hK : D.KeyAx q) (γ : M22) (hγ : γ.det = 1) (σ : D.Vz q) :
    (D.canon γ q σ.1).1.1 = q ∧
    (D.canon γ q σ.1).1.2.1 = (Ku.orCanon (D.toKeyModel q hK) (M22.toSL γ, σ)).1.1.1 ∧
    M22.lexKeyM (D.canon γ q σ.1).1.2.2 = (Ku.orCanon (D.toKeyModel q hK) (M22.toSL γ, σ)).1.2 ∧
    (D.canon γ q σ.1).2 = ((Ku.orCanon (D.toKeyModel q hK) (M22.toSL γ, σ)).2 : ℤ) := by
  set M := D.toKeyModel q hK
  have hrep : D.repZ (q : ℤ) σ.1 = D.rep q σ.1 := by simp [repZ]
  set l := D.trSetZ (q : ℤ) σ.1 (D.rep q σ.1)
  have hne : l ≠ [] := by
    have := hK.2.2.1 σ.1 ((D.mem_idxF).1 σ.2); rwa [hrep] at this
  have hinv := pickInv_foldl γ q l none [] (Or.inl ⟨rfl, rfl⟩)
  rw [List.nil_append] at hinv
  rcases hinv with ⟨-, hl⟩ | ⟨t, ht, hfold, hmin⟩
  · exact absurd hl hne
  have hcanon : D.canon γ q σ.1 = ((q, D.rep q σ.1, γ * t.g), permSign t.perm q) := by
    rw [canon_eq_fold, trSet_eq, hfold]
  have hdet : ∀ u ∈ l, u.g.det = 1 := fun u hu => (hK.2.2.2.1 u (trZ_of_trSetZ D hu)).1
  have hS : M.S σ (M.rep σ) = D.Sset q σ.1 (D.rep q σ.1) := by
    show D.Sset q σ.1 (D.repZ q σ.1) = _; rw [hrep]
  -- the key: `min' cands = lexKeyM (γ * t.g)`
  have hkey : ∀ u ∈ l, Ku.lexKey (M22.toSL γ * M22.toSL u.g) = M22.lexKeyM (γ * u.g) := by
    intro u hu
    rw [← M22.toSL_mul hγ (hdet u hu), M22.lexKey_toSL (by rw [M22.det_mul, hγ, hdet u hu, one_mul])]
  have hmin' : (Ku.canon M.toKeyModel (M22.toSL γ, σ)).2 = M22.lexKeyM (γ * t.g) := by
    apply le_antisymm
    · apply Finset.min'_le
      refine Finset.mem_image.2 ⟨M22.toSL t.g, ?_, hkey t ht⟩
      show _ ∈ M.S σ (M.rep σ); rw [hS]; exact (D.mem_Sset).2 ⟨t, ht, rfl⟩
    · refine Finset.le_min' _ (Ku.cands_nonempty _ _) _ ?_
      intro k hk
      obtain ⟨s, hs, rfl⟩ := Finset.mem_image.1 hk
      have hs' : s ∈ D.Sset q σ.1 (D.rep q σ.1) := by rw [← hS]; exact hs
      obtain ⟨u, hu, rfl⟩ := (D.mem_Sset).1 hs'
      show _ ≤ Ku.lexKey (M22.toSL γ * M22.toSL u.g)
      rw [hkey u hu]; exact hmin u hu
  refine ⟨by rw [hcanon], by rw [hcanon]; show _ = D.repZ q σ.1; rw [hrep], by rw [hcanon]; exact hmin'.symm, ?_⟩
  -- the sign: the picked transporter is `toSL t.g`
  have hpick : Ku.pick M (M22.toSL γ, σ) = M22.toSL t.g := by
    have h1 := Ku.pick_key M (M22.toSL γ, σ)
    rw [hmin', ← hkey t ht] at h1
    exact mul_left_cancel (Ku.lexKey_injective h1)
  rw [hcanon]
  show permSign t.perm q = ((D.sgnZ q σ.1 (D.repZ q σ.1) (Ku.pick M (M22.toSL γ, σ)) : ℤˣ) : ℤ)
  rw [hpick, hrep, sgnZ_eq hK ht, unitOf_val (permSign_cases _ _)]
  simp

end TauData

end GL2.TauChk
