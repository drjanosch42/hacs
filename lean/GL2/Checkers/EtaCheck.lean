import GL2.Checkers.TauBdCheck

/-!
# L18-3b: the `obl:eta` checker

[GT] `chk:eta`, mirroring [C] `def:pairhomotopy` (`eq:pairhomotopy`, sign convention OP6) and
`obl:eta` (i)–(iv).  Clauses (v) (the derived `h̃` is the transport of the stored η: a statement about
cochain matrices over `R`, not about the table) and (vi) (diagnostics, never branched on) stay out.

**What is modelled.** An `EtaData` is the `eta_p<p>_r<ν>.txt` dump of `HeckeTauDumpValidation`
(`HECKE_DUMP_TAU_PAIR`, L18-3): one stored `(q+1)`-chain `η(s_j · ι(σ))` of `Kᵘ` per slot `(j, q, σ)`,
`σ` a boundary `q`-simplex, over the `τ_∂`-dump `bd` (a `TauBdData`) whose `base` is the bulk τ-dump
(a `TauData`) of the same `(p, ν)`.  So the model, the coset datum and the (P1) split are **one**,
by construction -- which is `obl:eta`(i)'s "the tables it refers to are the certified τ and `τ_∂` of the
same `g`, `Γ''` and `s_j`" -- and `EtaSpec` *contains* `TauSpec` of the base and `TauBdSpec` of `bd`.

* the (T1) helper of η, `EtaData.etaT1`: `η(γ · ι(σ)) = (g⁻¹ h g) · η(s_i · ι(σ))` for `γ = h s_i`, the
  same `classOf` / `conj` as τ and `τ_∂`, canonicalised in degree `q + 1`;
* the defect `(j τ_∂ − τ j)(s_j σ)` (`EtaData.defect`): `τ_∂` through **its** helper at `γ = s_j`, τ the
  stored slot, both as `q`-chains of `Kᵘ` (a boundary simplex of `K` is a simplex of `K`);
* (ii) the homotopy identity: `∂ η(s_j σ) = defect` on vertex slots and
  `∂ η(s_j σ) + η(∂(s_j σ)) = defect` on edge slots, `η(∂ ·)` through the helper only.

`etaCheck` decides `EtaSpec` (`etaCheck_sound`, `etaCheck_complete`); it is the conjunction
`tauCheck base && taubdCheck bd && etaCoreCheck E`, so a data module can kernel-check the three
pieces as three declarations (`etaCheck_of`).
-/

namespace GL2.TauChk

/-- The η dump over the `τ_∂`-dump `bd` (and through it over the bulk τ-dump `bd.base`). -/
structure EtaData where
  bd : TauBdData
  slots : List Slot
  regime : String
  fnv : ℕ
  deriving Repr

namespace EtaData

variable (E : EtaData)

/-- The bulk τ-dump: the model, the coset datum and the stored τ. -/
abbrev base : TauData := E.bd.base

/-- `dim X`: the largest simplex dimension of the model (`2` on the 7-vertex model). -/
def dimX : ℕ := E.base.simplices.foldl (fun a x => max a x.1) 0

/-- The slots the table must have: `j ∈ [1, m]` times every boundary `q`-simplex with `q + 1 ≤ dim X`
(`num_degrees() = min(τ_∂.num_degrees(), dim X)`, [C] `open:ell2` rider (i)). -/
def wantKeys : List (ℕ × ℕ × ℕ) :=
  ((List.range E.base.m).map (· + 1)).flatMap fun i =>
    (E.bd.bdry.filter fun x => x.1 + 1 ≤ E.dimX).map fun x => (i, x.1, x.2)

def slotKeys : List (ℕ × ℕ × ℕ) := E.slots.map fun s => (s.i, s.q, s.sigma)

def lookup (i q σ : ℕ) : List Term :=
  match E.slots.find? (fun s => s.i = i ∧ s.q = q ∧ s.sigma = σ) with
  | some s => s.terms
  | none => []

/-- **The (T1) helper of η**: `η(γ · ι(σ)) = (g⁻¹ h g) · η(s_i · ι(σ))`, `γ = h s_i` by the one (P1)
split, canonicalised in degree `q + 1`, scaled by `e`; the poison chain if `γ` is in no class. -/
def etaT1 (e : ℤ) (γ : M22) (q σ : ℕ) : Chain :=
  if E.base.classOf γ = 0 then [((3, 0, ⟨0, 0, 0, 0⟩), 1)] else
  E.base.canonTerms (q + 1) e ((E.lookup (E.base.classOf γ) q σ).map fun t => { t with g := E.base.conj γ * t.g })

/-- `η(∂(s_j · ι(σ)))` through the helper (zero in degree `0`). -/
def etaBdry (j q σ : ℕ) : Chain :=
  if q = 0 then [] else
  ((List.range (q + 1)).map fun f => E.etaT1 (sgnPow f) (E.base.sAt j) (q - 1) (E.base.face q σ f)).flatten

/-- The right-hand side of `eq:pairhomotopy` on a stored slot: `(j τ_∂ − τ j)(s_j · ι(σ))`, `τ_∂` through
its (T1∂) helper, `τ` the stored slot. -/
def defect (j q σ : ℕ) : Chain :=
  E.bd.view.tauT1 1 (E.base.sAt j) q σ ++ E.base.canonTerms q (-1) (E.base.lookup j q σ)

/-- The left-hand side: `∂ η(s_j σ) + η(∂(s_j σ))`. -/
def homLhs (s : Slot) : Chain := E.base.bdry (s.q + 1) s.terms ++ E.etaBdry s.i s.q s.sigma

/-- `σ_t` is a `(q+1)`-simplex of `K`. -/
def isSimplex (q σ : ℕ) : Bool := E.base.simplices.any fun y => y.1 = q ∧ y.2.1 = σ

end EtaData

open EtaData in
/-- **The `obl:eta` predicate on generators** ([C] `obl:eta` (i)–(iv)). -/
structure EtaSpec (E : EtaData) : Prop where
  /-- (i) the base τ-table is certified (`obl:tau`) … -/
  tau : TauSpec E.base
  /-- … and so is the `τ_∂`-table (`obl:taubdry`), on the same model and coset datum (by construction). -/
  taubd : TauBdSpec E.bd
  /-- completeness: exactly one stored chain per slot `(j, q, σ)`, `σ ∈ K_bd`, `q + 1 ≤ dim X`. -/
  complete : E.slotKeys.Nodup ∧ E.slotKeys.Perm E.wantKeys
  /-- (i) structure: non-zero coefficient, `det γ_t = 1`, `σ_t` a `(q+1)`-simplex of `K`, canonical form
  with sign `+1`. -/
  struct : ∀ s ∈ E.slots, ∀ t ∈ s.terms,
    t.coeff ≠ 0 ∧ t.g.det = 1 ∧ E.isSimplex (s.q + 1) t.sigma = true ∧
      E.base.canon t.g (s.q + 1) t.sigma = ((s.q + 1, t.sigma, t.g), 1)
  /-- (ii) the homotopy identity `eq:pairhomotopy`: `∂ η(s_j σ) + η(∂(s_j σ)) = (j τ_∂ − τ j)(s_j σ)`
  (the middle term absent in degree `0`), every value through its table's helper. -/
  homotopy : ∀ s ∈ E.slots, Chain.eval (E.homLhs s) = Chain.eval (E.defect s.i s.q s.sigma)
  /-- (iii) (T1) bookkeeping: for every transporter `ι(σ) = t · ι(σ')` between boundary simplices and
  every `j`, the slot `(j, σ)` is the sign-tracked (T1)-translate of `(s_j t) · ι(σ')`. -/
  T1 : ∀ s ∈ E.slots, ∀ t ∈ E.base.trs, t.q = s.q → t.s1 = s.sigma → (t.q, t.s2) ∈ E.bd.bdry →
    Chain.eval (E.base.canonTerms (s.q + 1) 1 s.terms) =
      Chain.eval (E.etaT1 (permSign t.perm s.q) (E.base.sAt s.i * t.g) s.q t.s2)
  /-- (iii) the entry bound recounted on the η-chains: `max(‖γ_t‖, 4N²) < 2^31`. -/
  bounds : (∀ s ∈ E.slots, ∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31) ∧ 4 * E.base.N ^ 2 < 2 ^ 31
  /-- (iv) the solve regime is `Z`. -/
  integral : E.regime = "Z"

/-! ### The checker -/

/-- The η-specific part of the checker (everything but `TauSpec base` and `TauBdSpec bd`). -/
def etaCoreCheck (E : EtaData) : Bool :=
  decide (E.slotKeys.Nodup ∧ E.slotKeys.Perm E.wantKeys) &&
  decide (∀ s ∈ E.slots, ∀ t ∈ s.terms,
    t.coeff ≠ 0 ∧ t.g.det = 1 ∧ E.isSimplex (s.q + 1) t.sigma = true ∧
      E.base.canon t.g (s.q + 1) t.sigma = ((s.q + 1, t.sigma, t.g), 1)) &&
  E.slots.all (fun s => chainEqB (E.homLhs s) (E.defect s.i s.q s.sigma)) &&
  E.slots.all (fun s => E.base.trs.all fun t => !(t.q = s.q ∧ t.s1 = s.sigma ∧ (t.q, t.s2) ∈ E.bd.bdry) ||
    chainEqB (E.base.canonTerms (s.q + 1) 1 s.terms)
      (E.etaT1 (permSign t.perm s.q) (E.base.sAt s.i * t.g) s.q t.s2)) &&
  decide ((∀ s ∈ E.slots, ∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31) ∧ 4 * E.base.N ^ 2 < 2 ^ 31) &&
  decide (E.regime = "Z")

/-- **The checker.** -/
def etaCheck (E : EtaData) : Bool := tauCheck E.base && taubdCheck E.bd && etaCoreCheck E

/-- The three pieces, checked separately (one kernel declaration each in a data module). -/
theorem etaCheck_of {E : EtaData} (h₁ : tauCheck E.base = true) (h₂ : taubdCheck E.bd = true)
    (h₃ : etaCoreCheck E = true) : etaCheck E = true := by
  simp [etaCheck, h₁, h₂, h₃]

/-- **Soundness**: an accepted table satisfies the `obl:eta` predicate. -/
theorem etaCheck_sound (E : EtaData) (h : etaCheck E = true) : EtaSpec E := by
  simp only [etaCheck, etaCoreCheck, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hτ, hb⟩, ⟨⟨⟨⟨⟨h2, h3⟩, h5⟩, h6⟩, h7⟩, h8⟩⟩ := h
  refine ⟨tauCheck_sound _ hτ, taubdCheck_sound _ hb, h2, h3, ?_, ?_, h7, h8⟩
  · intro s hs
    exact (chainEqB_iff _ _).mp (List.all_eq_true.mp h5 s hs)
  · intro s hs t ht hq h1' hb'
    have := List.all_eq_true.mp (List.all_eq_true.mp h6 s hs) t ht
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at this
    rcases this with hn | hc
    · exact absurd ⟨hq, h1', hb'⟩ hn
    · exact (chainEqB_iff _ _).mp hc

/-- **Completeness**: the checker accepts every table satisfying the predicate. -/
theorem etaCheck_complete (E : EtaData) (h : EtaSpec E) : etaCheck E = true := by
  obtain ⟨hτ, hb, h2, h3, h5, h6, h7, h8⟩ := h
  simp only [etaCheck, etaCoreCheck, Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨tauCheck_complete _ hτ, taubdCheck_complete _ hb⟩, ⟨⟨⟨⟨⟨h2, h3⟩, ?_⟩, ?_⟩, h7⟩, h8⟩⟩
  · exact List.all_eq_true.mpr fun s hs => (chainEqB_iff _ _).mpr (h5 s hs)
  · refine List.all_eq_true.mpr fun s hs => List.all_eq_true.mpr fun t ht => ?_
    by_cases hq : t.q = s.q ∧ t.s1 = s.sigma ∧ (t.q, t.s2) ∈ E.bd.bdry
    · simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
      exact Or.inr ((chainEqB_iff _ _).mpr (h6 s hs t ht hq.1 hq.2.1 hq.2.2))
    · simp only [hq, decide_false, Bool.not_false, Bool.true_or]

/-- `EtaSpec` is a decidable `Prop`: `etaCheck` decides it. -/
instance instDecidableEtaSpec (E : EtaData) : Decidable (EtaSpec E) :=
  decidable_of_iff (etaCheck E = true) ⟨etaCheck_sound E, etaCheck_complete E⟩

/-! ### The golden tie: FNV-1a-64 exactly as `HeckePairHomotopy::table_fnv64` -/

/-- The table hash: `N, m`, then per boundary `(q, σ)` with `q + 1 ≤ dim X` (C++ order) and `j`:
`q, σ, j` and the terms. -/
def etaTableFnv (E : EtaData) : ℕ :=
  (E.bd.bdry.filter fun x => x.1 + 1 ≤ E.dimX).foldl (fun h x =>
      ((List.range E.base.m).map (· + 1)).foldl (fun h j =>
        fnvTerms ([x.1, x.2, j].foldl fnvU64 h) (E.lookup j x.1 x.2)) h)
    (fnvU64 (fnvU64 fnvOffset E.base.N) E.base.m)

end GL2.TauChk
