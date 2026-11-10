import GL2.Checkers.TauCheck

/-!
# L18-3a: the `obl:taubdry` checker

[GT] `chk:taubd`, mirroring [C] `def:taubdry` (T1∂)–(T3∂), `lem:bdrycomponent` (the forced component
map `c ↦ g⁻¹ c`), `lem:taubdryexists`(2) and `obl:taubdry` (i)–(iv).  Clause (v) (the Shapiro
transcription) is a validation *route*, not a property of the table, and stays out.

**What is modelled.** A `TauBdData` is the `taubd_p<p>_r<ν>.txt` dump of `HeckeTauDumpValidation`
(`HECKE_DUMP_TAU_PAIR`, L18-3): the boundary simplices `K_bd` of the 7-vertex model by degree and one
stored chain `τ_∂(s_j · ι(σ))` per slot `(j, q, σ)`, `σ ∈ K_bd`.  The model (simplex lists, orbit
representatives, transporter sets with their vertex correspondences) and the coset datum
(`N`, `g`, `s_j`) are those of a `TauData` `base` -- the bulk τ-dump of the same `(p, ν)`.  So
canonical keys (P4), the (P1) split and the (T1) helper are **literally the bulk ones**
(`TauData.canon`, `classOf`, `conj`, `tauT1`), applied to the view `B.view` whose slots are the τ_∂
slots: [C] `obl:taubdry`(iii) is `obl:tau`'s clause "with `X_bd` in place of `Kᵘ`".

**Components** ([C] `lem:bdrycomponent`, `obl:taubdry`(i)).  `K_bd` is the path `v₀ – v₁ – v₂ – v₃`
on the horocycle at `∞`, so the component (cusp) of `γ · ι(σ)`, `σ ∈ K_bd`, is `γ · ∞ = (a : c)`.
The slot `(j, σ)` lies in the component `s_j · ∞`; the forced target is `g⁻¹ s_j ∞ = (N a_j : c_j)`
(through `adj g = diag(N, 1)`, projective, no rationals).  Projective equality of two primitive
vectors is the vanishing of the cross product, so `compOk` needs no `gcd`.  That `K_bd` lies at `∞`
is the model fact ([C] (B1)–(B3), `SL2ZTessellationTables::boundary_simplices`), an input like the
transporter sets in `TauCheck`'s trust boundary.

`TauBdSpec B` is the predicate; `taubdCheck` decides it (`taubdCheck_sound`, `taubdCheck_complete`).
-/

namespace GL2.TauChk

/-- The `τ_∂` dump over the model and coset datum of `base`. `bdry` lists the boundary simplices
`(q, σ)` in the C++ order (`boundary_simplices(q)`, `q` ascending). -/
structure TauBdData where
  base : TauData
  bdry : List (ℕ × ℕ)
  slots : List Slot
  regime : String
  fnv : ℕ
  deriving Repr

namespace TauBdData

variable (B : TauBdData)

/-- The `τ_∂` table as a `TauData` over the base's model and coset datum: the same canonical keys,
(P1) split and (T1) helper, now reading the `τ_∂` slots. -/
def view : TauData := { B.base with slots := B.slots, regime := B.regime, fnv := B.fnv }

/-- The slots the table must have: `j ∈ [1, m]` times every boundary simplex. -/
def wantKeys : List (ℕ × ℕ × ℕ) :=
  ((List.range B.base.m).map (· + 1)).flatMap fun i => B.bdry.map fun x => (i, x.1, x.2)

def slotKeys : List (ℕ × ℕ × ℕ) := B.slots.map fun s => (s.i, s.q, s.sigma)

/-- The term `γ · ι(σ)` of slot `j` lies in the forced component `g⁻¹ (s_j · ∞)`:
`(a : c) = (N a_j : c_j)` projectively, i.e. `a c_j = N a_j c`. -/
def compOk (j : ℕ) (γ : M22) : Prop :=
  γ.a * (B.base.sAt j).c = (B.base.N : ℤ) * (B.base.sAt j).a * γ.c

instance (j : ℕ) (γ : M22) : Decidable (B.compOk j γ) := by unfold compOk; infer_instance

/-- `K_bd` is a face-closed set of simplices of `K` of dimension `≤ 1` ((B2)), without repetition. -/
def bdryOk : Prop :=
  B.bdry.Nodup ∧
  (∀ x ∈ B.bdry, (B.base.simplices.any fun y => y.1 = x.1 ∧ y.2.1 = x.2) = true) ∧
  (∀ x ∈ B.bdry, x.1 ≤ 1) ∧
  (∀ x ∈ B.bdry, x.1 = 1 → ∀ f ∈ List.range 2, (0, B.base.face 1 x.2 f) ∈ B.bdry)

instance : Decidable B.bdryOk := by unfold bdryOk; infer_instance

end TauBdData

open TauBdData in
/-- **The `obl:taubdry` predicate on generators** ([C] `obl:taubdry` (i)–(iv); (T1∂)–(T3∂) of
`def:taubdry`). -/
structure TauBdSpec (B : TauBdData) : Prop where
  /-- the coset datum of the base (as `TauSpec.cosets`): `g = diag(1, N)`, `m` pairwise
  `Γ₀(N)`-inequivalent `s_j` of determinant one, nonempty `S(σ, σ̂)`. -/
  cosets : B.base.g = ⟨1, 0, 0, B.base.N⟩ ∧ 0 < B.base.N ∧ B.base.s.length = B.base.m ∧
    (∀ x ∈ B.base.s, x.det = 1) ∧
    (∀ i ∈ List.range B.base.m, ∀ j ∈ List.range B.base.m, i ≠ j →
      (B.base.sAt (i + 1) * (B.base.sAt (j + 1)).adj).c % (B.base.N : ℤ) ≠ 0) ∧
    (∀ x ∈ B.base.simplices, B.base.trSet x.1 x.2.1 (B.base.rep x.1 x.2.1) ≠ [])
  /-- the model is closed under faces and `K_bd` is a face-closed set of simplices of dimension `≤ 1`. -/
  closed : B.base.closedB = true ∧ B.bdryOk
  /-- completeness: exactly one stored chain per slot `(j, q, σ)`, `σ ∈ K_bd`. -/
  complete : B.slotKeys.Nodup ∧ B.slotKeys.Perm B.wantKeys
  /-- structure: non-zero coefficient, `det γ_t = 1`, `σ_t ∈ K_bd` of degree `q`, canonical form
  with sign `+1`. -/
  struct : ∀ s ∈ B.slots, ∀ t ∈ s.terms,
    t.coeff ≠ 0 ∧ t.g.det = 1 ∧ (s.q, t.sigma) ∈ B.bdry ∧
      B.view.canon t.g s.q t.sigma = ((s.q, t.sigma, t.g), 1)
  /-- (i) (T3∂): in degree `0` every chain is a single simplex with coefficient `1`. -/
  T3 : ∀ s ∈ B.slots, s.q = 0 → s.terms.map Term.coeff = [1]
  /-- (i) components: every term of slot `(j, σ)` lies in the forced component `g⁻¹ (s_j · ∞)`
  ([C] `lem:bdrycomponent`, recomputed from `g` alone); hence every chain is supported in a single
  component (`TauBdSpec.single_component`). -/
  comp : ∀ s ∈ B.slots, ∀ t ∈ s.terms, B.compOk s.i t.g
  /-- (ii) (T2∂): `∂ τ_∂(s_j σ) = τ_∂(∂(s_j σ))` on every stored slot of degree `1`, the right-hand
  side through the (T1∂) helper only. -/
  T2 : ∀ s ∈ B.slots, s.q = 1 →
    Chain.eval (B.view.bdry 1 s.terms) = Chain.eval (B.view.tauBdry s.i 1 s.sigma)
  /-- (iii) (T1∂) bookkeeping: for every transporter `ι(σ) = t · ι(σ')` between boundary simplices and
  every `j`, the slot `(j, σ)` is the sign-tracked (T1∂)-translate of `(s_j t) · ι(σ')`. -/
  T1 : ∀ s ∈ B.slots, ∀ t ∈ B.base.trs, t.q = s.q → t.s1 = s.sigma → (t.q, t.s2) ∈ B.bdry →
    Chain.eval (B.view.canonTerms s.q 1 s.terms) =
      Chain.eval (B.view.tauT1 (permSign t.perm s.q) (B.base.sAt s.i * t.g) s.q t.s2)
  /-- (iv) integrality is unconditional: every coefficient is `±1` and the regime is `Z`. -/
  integral : (∀ s ∈ B.slots, ∀ t ∈ s.terms, t.coeff = 1 ∨ t.coeff = -1) ∧ B.regime = "Z"
  /-- the int32 bound `max(‖γ_t‖, 4N²) < 2^31` ([C] `obl:backend`; the C++ certifies the int64 tier). -/
  bounds : (∀ s ∈ B.slots, ∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31) ∧ 4 * B.base.N ^ 2 < 2 ^ 31

/-! ### The checker -/

/-- **The checker.** Each conjunct decides the corresponding field of `TauBdSpec`. -/
def taubdCheck (B : TauBdData) : Bool :=
  decide (B.base.g = ⟨1, 0, 0, B.base.N⟩ ∧ 0 < B.base.N ∧ B.base.s.length = B.base.m ∧
    (∀ x ∈ B.base.s, x.det = 1) ∧
    (∀ i ∈ List.range B.base.m, ∀ j ∈ List.range B.base.m, i ≠ j →
      (B.base.sAt (i + 1) * (B.base.sAt (j + 1)).adj).c % (B.base.N : ℤ) ≠ 0) ∧
    (∀ x ∈ B.base.simplices, B.base.trSet x.1 x.2.1 (B.base.rep x.1 x.2.1) ≠ [])) &&
  (B.base.closedB && decide B.bdryOk) &&
  decide (B.slotKeys.Nodup ∧ B.slotKeys.Perm B.wantKeys) &&
  decide (∀ s ∈ B.slots, ∀ t ∈ s.terms,
    t.coeff ≠ 0 ∧ t.g.det = 1 ∧ (s.q, t.sigma) ∈ B.bdry ∧
      B.view.canon t.g s.q t.sigma = ((s.q, t.sigma, t.g), 1)) &&
  decide (∀ s ∈ B.slots, s.q = 0 → s.terms.map Term.coeff = [1]) &&
  decide (∀ s ∈ B.slots, ∀ t ∈ s.terms, B.compOk s.i t.g) &&
  B.slots.all (fun s => !(s.q = 1) || chainEqB (B.view.bdry 1 s.terms) (B.view.tauBdry s.i 1 s.sigma)) &&
  B.slots.all (fun s => B.base.trs.all fun t => !(t.q = s.q ∧ t.s1 = s.sigma ∧ (t.q, t.s2) ∈ B.bdry) ||
    chainEqB (B.view.canonTerms s.q 1 s.terms)
      (B.view.tauT1 (permSign t.perm s.q) (B.base.sAt s.i * t.g) s.q t.s2)) &&
  decide ((∀ s ∈ B.slots, ∀ t ∈ s.terms, t.coeff = 1 ∨ t.coeff = -1) ∧ B.regime = "Z") &&
  decide ((∀ s ∈ B.slots, ∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31) ∧ 4 * B.base.N ^ 2 < 2 ^ 31)

/-- **Soundness**: an accepted table satisfies the `obl:taubdry` predicate. -/
theorem taubdCheck_sound (B : TauBdData) (h : taubdCheck B = true) : TauBdSpec B := by
  simp only [taubdCheck, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, ⟨h0a, h0b⟩⟩, h2⟩, h3⟩, h4⟩, h4c⟩, h5⟩, h6⟩, h7⟩, h8⟩ := h
  refine ⟨h1, ⟨h0a, h0b⟩, h2, h3, h4, h4c, ?_, ?_, h7, h8⟩
  · intro s hs hq
    have := List.all_eq_true.mp h5 s hs
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at this
    rcases this with hn | hc
    · exact absurd hq hn
    · exact (chainEqB_iff _ _).mp hc
  · intro s hs t ht hq h1' hb
    have := List.all_eq_true.mp (List.all_eq_true.mp h6 s hs) t ht
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at this
    rcases this with hn | hc
    · exact absurd ⟨hq, h1', hb⟩ hn
    · exact (chainEqB_iff _ _).mp hc

/-- **Completeness**: the checker accepts every table satisfying the predicate. -/
theorem taubdCheck_complete (B : TauBdData) (h : TauBdSpec B) : taubdCheck B = true := by
  obtain ⟨h1, ⟨h0a, h0b⟩, h2, h3, h4, h4c, h5, h6, h7, h8⟩ := h
  simp only [taubdCheck, Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, ⟨h0a, h0b⟩⟩, h2⟩, h3⟩, h4⟩, h4c⟩, ?_⟩, ?_⟩, h7⟩, h8⟩
  · refine List.all_eq_true.mpr fun s hs => ?_
    by_cases hq : s.q = 1
    · simp only [hq, Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
      exact Or.inr ((chainEqB_iff _ _).mpr (h5 s hs hq))
    · simp [hq]
  · refine List.all_eq_true.mpr fun s hs => List.all_eq_true.mpr fun t ht => ?_
    by_cases hq : t.q = s.q ∧ t.s1 = s.sigma ∧ (t.q, t.s2) ∈ B.bdry
    · simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
      exact Or.inr ((chainEqB_iff _ _).mpr (h6 s hs t ht hq.1 hq.2.1 hq.2.2))
    · simp only [hq, decide_false, Bool.not_false, Bool.true_or]

/-- `TauBdSpec` is a decidable `Prop`: `taubdCheck` decides it. -/
instance instDecidableTauBdSpec (B : TauBdData) : Decidable (TauBdSpec B) :=
  decidable_of_iff (taubdCheck B = true) ⟨taubdCheck_sound B, taubdCheck_complete B⟩

/-- `det s_j = 1` for every `j` (a stored representative, or the default identity). -/
theorem TauData.sAt_det (D : TauData) (h : ∀ x ∈ D.s, x.det = 1) (i : ℕ) : (D.sAt i).det = 1 := by
  unfold TauData.sAt
  rw [List.getD_eq_getElem?_getD]
  cases hx : D.s[i - 1]? with
  | none => simp [M22.det]
  | some x => simpa using h x (List.mem_of_getElem? hx)

/-- **(i) single component.** Under `TauBdSpec`, any two terms `γ₁ · ι(σ₁)`, `γ₂ · ι(σ₂)` of one stored
chain lie in the same component: `γ₁ · ∞ = γ₂ · ∞` projectively (`a₁ c₂ = a₂ c₁`). -/
theorem TauBdSpec.single_component {B : TauBdData} (h : TauBdSpec B) (s : Slot) (hs : s ∈ B.slots)
    (t₁ t₂ : Term) (h₁ : t₁ ∈ s.terms) (h₂ : t₂ ∈ s.terms) :
    t₁.g.a * t₂.g.c = t₂.g.a * t₁.g.c := by
  have c₁ := h.comp s hs t₁ h₁
  have c₂ := h.comp s hs t₂ h₂
  have hd := TauData.sAt_det B.base h.cosets.2.2.2.1 s.i
  have hN : (B.base.N : ℤ) ≠ 0 := by exact_mod_cast (Nat.pos_iff_ne_zero.mp h.cosets.2.1)
  simp only [TauBdData.compOk] at c₁ c₂
  simp only [M22.det] at hd
  generalize B.base.sAt s.i = sj at c₁ c₂ hd
  -- `a_k c_j = (N a_j) c_k`: cancel `c_j` if it is non-zero; otherwise `a_j ≠ 0` (det) and `c_k = 0`.
  by_cases hx : sj.c = 0
  · have ha : (B.base.N : ℤ) * sj.a ≠ 0 := by
      refine mul_ne_zero hN fun ha => ?_
      rw [ha, hx] at hd; simp at hd
    rw [hx, mul_zero] at c₁ c₂
    have e₁ : t₁.g.c = 0 := (mul_eq_zero.mp c₁.symm).resolve_left ha
    have e₂ : t₂.g.c = 0 := (mul_eq_zero.mp c₂.symm).resolve_left ha
    rw [e₁, e₂]; ring
  · apply mul_right_cancel₀ hx
    calc t₁.g.a * t₂.g.c * sj.c = t₁.g.a * sj.c * t₂.g.c := by ring
      _ = (B.base.N : ℤ) * sj.a * t₁.g.c * t₂.g.c := by rw [c₁]
      _ = t₂.g.a * sj.c * t₁.g.c := by rw [c₂]; ring
      _ = t₂.g.a * t₁.g.c * sj.c := by ring


/-! ### The golden tie: FNV-1a-64 exactly as `HeckeBoundaryTauTable::table_fnv64` -/

/-- The table hash: `N, m`, then per boundary `(q, σ)` (C++ order) and `j`: `q, σ, j` and the terms. -/
def bdTableFnv (B : TauBdData) : ℕ :=
  B.bdry.foldl (fun h x => ((List.range B.base.m).map (· + 1)).foldl (fun h j =>
      fnvTerms ([x.1, x.2, j].foldl fnvU64 h) (B.view.lookup j x.1 x.2)) h)
    (fnvU64 (fnvU64 fnvOffset B.base.N) B.base.m)

end GL2.TauChk
