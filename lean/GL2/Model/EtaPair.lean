import GL2.Model.XBdryModel
import GL2.Checkers.EtaP2

/-!
# M3 / C2 (L19-4): `EtaSpec → IsPairHomotopy` -- the checked η-table is a pair homotopy

[GT] `thm:etaspec-pairhomotopy`, `cor:etap2-pairhomotopy`, mirroring [C] `def:pairhomotopy` (`eq:pairhomotopy`) and
`obl:eta` (i)–(iv), on the descriptor of an accepted dump.

**Reading a table as chains.** For an accepted τ-dump `D` (`D.XAx`) the abstract chains are those of `D.toXBulk`.
`genAt q γ σ = [γ, σ]` (zero if `σ` is not a stored `q`-simplex), `absT q ts = ∑ c_t [γ_t, σ_t]` reads a stored term
list, and `absC q l = ∑ c [γ_k, σ_k]` reads a checker chain by its keys.  Two facts make the checker's arithmetic the
abstract one:

* `absC_eq_of_eval` -- chains with equal `Chain.eval` have equal readings (the checker's equality is keywise);
* `genAt_canon` -- the canonical key of `γ · ι(σ)` with its sign reads back to `[γ, σ]` (`canon_spec`: the canonical
  matrix is `γ t` for a stored transporter `t ∈ S(σ, σ̂)`, the sign its `permSign`), so `canonTerms` reads as `absT`;
  and `bd_genQ` -- the descriptor's boundary of `[γ, τ]` is the signed face sum the checker's `bdry` walks.

**The theorem** (`EtaSpec.toIsPairHomotopy`): let `E` be an η-table with `EtaSpec E`, `X_∂` a boundary descriptor whose
simplices are `E`'s boundary list, `g = diag(1, N) ∈ GL(2, ℚ)`, and suppose the `s_j` cover `Γ₀(N)\SL(2, ℤ)`.  Then for
**any** `g`-twisted chain data `τ`, `τ_∂` and any (T1) map `η` which *realise* the three tables on the generators
`s_j · ι(σ)` (`RealisesTau`, `RealisesTauBd`, `RealisesEta`), `η` is a pair homotopy:
`∂η + η∂ = j τ_∂ − τ j` on all of `C_•(X_∂)`.  The identity is checked by `EtaSpec.homotopy` on the generators, read
through `absC`, and spread to every simplex by (T1) and the coset covering (`T1_ext_gen`).

*Not claimed.*  That a `τ` (resp. `τ_∂`, `η`) realising a given table exists -- the construction of a twisted chain
map from its table, [C] `alg:tau` read backwards -- is not formalised: the theorem is about any realisation.
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.TauChk GL2.TauChk.Data GL2.Cone

namespace EtaRead

variable (D : TauData) (hX : D.XAx)

/-- `[γ, σ]` in `C_q`, zero if `σ` is not a stored `q`-simplex. -/
def genAt (q : ℤ) (γ : M22) (σ : ℕ) : XBulk.chains (D.toXBulk hX) q :=
  if h : σ ∈ D.idxF q then XBulk.genQ (D.toXBulk hX) q (M22.toSL γ, ⟨σ, h⟩) else 0

/-- A stored term list read as a chain: `∑ c_t [γ_t, σ_t]`. -/
def absT (q : ℤ) (ts : List Term) : XBulk.chains (D.toXBulk hX) q :=
  (ts.map fun t => t.coeff • genAt D hX q t.g t.sigma).sum

/-- A checker chain read by its keys: `∑ c [γ_k, σ_k]`. -/
def absC (q : ℤ) (l : Chain) : XBulk.chains (D.toXBulk hX) q :=
  (l.map fun p => p.2 • genAt D hX q p.1.2.2 p.1.2.1).sum

lemma absC_nil (q : ℤ) : absC D hX q [] = 0 := rfl

lemma absC_append (q : ℤ) (l l' : Chain) : absC D hX q (l ++ l') = absC D hX q l + absC D hX q l' := by
  simp [absC, List.map_append, List.sum_append]

lemma absC_flatten (q : ℤ) (L : List Chain) : absC D hX q L.flatten = (L.map (absC D hX q)).sum := by
  induction L with
  | nil => rfl
  | cons l L ih => rw [List.flatten_cons, absC_append, ih, List.map_cons, List.sum_cons]

lemma absT_append (q : ℤ) (l l' : List Term) : absT D hX q (l ++ l') = absT D hX q l + absT D hX q l' := by
  simp [absT, List.map_append, List.sum_append]

lemma eval_cons (p : Key × ℤ) (l : Chain) (k : Key) :
    Chain.eval (p :: l) k = (if p.1 = k then p.2 else 0) + Chain.eval l k := by
  unfold Chain.eval
  by_cases h : p.1 = k <;> simp [h]

/-- The reading through the coefficient function, on any key set containing the chain's keys. -/
lemma absC_eq_sum (q : ℤ) (l : Chain) (S : Finset Key) (hS : ∀ p ∈ l, p.1 ∈ S) :
    absC D hX q l = ∑ k ∈ S, Chain.eval l k • genAt D hX q k.2.2 k.2.1 := by
  induction l with
  | nil => simp [absC_nil, Chain.eval]
  | cons p l ih =>
    have hp : p.1 ∈ S := hS p (by simp)
    rw [show absC D hX q (p :: l) = p.2 • genAt D hX q p.1.2.2 p.1.2.1 + absC D hX q l by simp [absC],
      ih (fun r hr => hS r (by simp [hr]))]
    simp only [eval_cons, add_smul, Finset.sum_add_distrib, ite_smul, zero_smul]
    rw [Finset.sum_ite_eq S p.1, if_pos hp]

/-- **Equal values, equal readings**: the checker's keywise equality is equality of chains. -/
theorem absC_eq_of_eval (q : ℤ) {l l' : Chain} (h : Chain.eval l = Chain.eval l') :
    absC D hX q l = absC D hX q l' := by
  classical
  set S : Finset Key := ((l ++ l').map Prod.fst).toFinset
  rw [absC_eq_sum D hX q l S (fun p hp => by simp [S]; exact Or.inl ⟨p.2, hp⟩),
    absC_eq_sum D hX q l' S (fun p hp => by simp [S]; exact Or.inr ⟨p.2, hp⟩), h]

/-- A term list already in canonical form (sign `+1`) reads as itself. -/
theorem absC_canonTerms_self (n : ℕ) (e : ℤ) (ts : List Term)
    (h : ∀ t ∈ ts, D.canon t.g n t.sigma = ((n, t.sigma, t.g), 1)) :
    absC D hX n (D.canonTerms n e ts) = e • absT D hX n ts := by
  induction ts with
  | nil => simp [absC, absT, TauData.canonTerms]
  | cons t ts ih =>
    have ht := h t (by simp)
    have ih' := ih (fun u hu => h u (by simp [hu]))
    simp only [TauData.canonTerms, List.map_cons] at ih' ⊢
    simp only [absC, absT, List.map_cons, List.sum_cons] at ih' ⊢
    rw [ih', ht, smul_add, smul_smul]
    simp

/-! ### The canonical key reads back -/

/-- `canon γ n σ` is `(σ̂, γ t)` with its `permSign`, for a stored transporter `t ∈ S(σ, σ̂)`. -/
lemma canon_spec (n : ℕ) (hK : D.KeyAx n) (γ : M22) (σ : ℕ) (hσ : σ ∈ D.idxL n) :
    ∃ t ∈ D.trSetZ n σ (D.rep n σ), D.canon γ n σ = ((n, D.rep n σ, γ * t.g), permSign t.perm n) := by
  have hrep : D.repZ (n : ℤ) σ = D.rep n σ := by simp [TauData.repZ]
  have hne : D.trSetZ (n : ℤ) σ (D.rep n σ) ≠ [] := by have := hK.2.2.1 σ hσ; rwa [hrep] at this
  have hinv := TauData.pickInv_foldl γ n (D.trSetZ n σ (D.rep n σ)) none [] (Or.inl ⟨rfl, rfl⟩)
  rw [List.nil_append] at hinv
  rcases hinv with ⟨-, hl⟩ | ⟨t, ht, hfold, -⟩
  · exact absurd hl hne
  · exact ⟨t, ht, by rw [D.canon_eq_fold, D.trSet_eq, hfold]⟩

/-- **The canonical key reads back**: `[γ, σ] = sgn · [γ_canon, σ̂]`. -/
theorem genAt_canon (n : ℕ) (γ : M22) (hγ : γ.det = 1) (σ : ℕ) (hσ : σ ∈ D.idxF n) :
    ((D.canon γ n σ).2) • genAt D hX n (D.canon γ n σ).1.2.2 (D.canon γ n σ).1.2.1 = genAt D hX n γ σ := by
  have hK := TauData.keyAx_all hX n
  obtain ⟨t, ht, hc⟩ := canon_spec D n hK γ σ ((D.mem_idxF).1 hσ)
  have hrep : D.repZ (n : ℤ) σ = D.rep n σ := by simp [TauData.repZ]
  have hrepm : D.rep n σ ∈ D.idxF n := by
    rw [← hrep]; exact (D.mem_idxF).2 (hK.1 σ ((D.mem_idxF).1 hσ))
  have htd : t.g.det = 1 := (hK.2.2.2.1 t (TauData.trZ_of_trSetZ D ht)).1
  rw [hc]
  simp only [genAt, dif_pos hσ, dif_pos hrepm]
  set M := (D.toXBulk hX).M n
  have hS : M22.toSL t.g ∈ M.S ⟨σ, hσ⟩ ⟨D.rep n σ, hrepm⟩ := by
    show _ ∈ D.Sset n σ (D.rep n σ)
    exact (D.mem_Sset).2 ⟨t, ht, rfl⟩
  have hsg : M.sgn ⟨σ, hσ⟩ ⟨D.rep n σ, hrepm⟩ (M22.toSL t.g) = unitOf (permSign t.perm n) := by
    show D.sgnZ n σ (D.rep n σ) (M22.toSL t.g) = _
    rw [TauData.sgnZ_eq hK ht]; simp
  have hr : OrRel M ((M22.toSL γ, ⟨σ, hσ⟩, 1) : OrPt _)
      ((M22.toSL γ * M22.toSL t.g, ⟨D.rep n σ, hrepm⟩, unitOf (permSign t.perm n)) : OrPt _) := by
    refine ⟨by simpa using hS, ?_⟩
    show (1 : ℤˣ) = M.sgn ⟨σ, hσ⟩ ⟨D.rep n σ, hrepm⟩ ((M22.toSL γ)⁻¹ * (M22.toSL γ * M22.toSL t.g)) * _
    rw [inv_mul_cancel_left, hsg, XBulk.sgn_sq]
  have h1 := XBulk.mk_gen_eq M hr
  simp only [Submodule.mkQ_apply] at h1
  have e1 : XBulk.genQ (D.toXBulk hX) n (M22.toSL γ, ⟨σ, hσ⟩) =
      Submodule.Quotient.mk (XBulk.gen ((M22.toSL γ, ⟨σ, hσ⟩, 1) : OrPt _)) := rfl
  rw [e1, h1, XBulk.mk_gen_eq_smul, M22.toSL_mul hγ htd]
  simp only
  rw [XBulk.smul_genQ, mul_one, unitOf_val (permSign_cases _ _)]

/-- `canonTerms` reads as `absT`, for terms of determinant one on stored simplices. -/
theorem absC_canonTerms (n : ℕ) (e : ℤ) (ts : List Term) (h : ∀ t ∈ ts, t.g.det = 1 ∧ t.sigma ∈ D.idxF n) :
    absC D hX n (D.canonTerms n e ts) = e • absT D hX n ts := by
  induction ts with
  | nil => simp [absC, absT, TauData.canonTerms]
  | cons t ts ih =>
    have ht := h t (by simp)
    have ih' := ih (fun u hu => h u (by simp [hu]))
    simp only [TauData.canonTerms, List.map_cons] at ih' ⊢
    simp only [absC, absT, List.map_cons, List.sum_cons] at ih' ⊢
    rw [ih', smul_add]
    congr 1
    rw [← genAt_canon D hX n t.g ht.1 t.sigma ht.2, smul_smul, smul_smul]


/-! ### Boundaries of generators -/

/-- The descriptor's boundary of a generator: `∂[γ, τ] = ∑_j ε_j [γ h_j, σ_j]` (any descriptor). -/
lemma bd_genQ_gen (D' : XBulkDatum) (q : ℤ) (a : SL(2, ℤ) × D'.V (q + 1)) :
    XBulk.bd D' q (XBulk.genQ D' (q + 1) a) =
      ∑ j, (((D'.face q a.2 j).2.2 : ℤˣ) : ℤ) • XBulk.genQ D' q (a.1 * (D'.face q a.2 j).1, (D'.face q a.2 j).2.1) := by
  have e1 : XBulk.genQ D' (q + 1) a = Submodule.Quotient.mk (XBulk.gen ((a.1, a.2, 1) : OrPt _)) := rfl
  rw [e1, XBulk.bd_mk, XBulk.bdF_gen]
  simp only [Units.val_one, one_smul]
  rw [quot_mk_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [XBulk.mk_gen_eq_smul, XBulk.smul_genQ, mul_one]
  rfl

lemma nf_natCast (n : ℕ) : nf (n : ℤ) = n + 2 := by unfold nf; simp

lemma sum_fin_nf {M : Type*} [AddCommMonoid M] (n : ℕ) (F : ℕ → M) :
    ∑ j : Fin (nf (n : ℤ)), F j.1 = ∑ f ∈ Finset.range (n + 2), F f := by
  rw [Fin.sum_univ_eq_sum_range (fun f => F f) (nf (n : ℤ)), nf_natCast]

/-- **The model's boundary of `[γ, τ]`** is the checker's signed face walk. -/
theorem bd_genAt (q : ℤ) (n : ℕ) (hqn : q = n) (γ : M22) (τ : ℕ) (hτ : τ ∈ D.idxF (q + 1)) :
    XBulk.bd (D.toXBulk hX) q (genAt D hX (q + 1) γ τ) =
      ∑ f ∈ Finset.range (n + 2), sgnPow f • genAt D hX q γ (D.face (n + 1) τ f) := by
  subst hqn
  rw [genAt, dif_pos hτ, bd_genQ_gen]
  have e : ((n : ℤ) + 1).toNat = n + 1 := by omega
  have key : ∀ j : Fin (nf (n : ℤ)), ((((D.toXBulk hX).face (n : ℤ) ⟨τ, hτ⟩ j).2.2 : ℤˣ) : ℤ) •
      XBulk.genQ (D.toXBulk hX) (n : ℤ) (M22.toSL γ * ((D.toXBulk hX).face (n : ℤ) ⟨τ, hτ⟩ j).1,
        ((D.toXBulk hX).face (n : ℤ) ⟨τ, hτ⟩ j).2.1) =
      sgnPow j.1 • genAt D hX (n : ℤ) γ (D.face (n + 1) τ j.1) := by
    intro j
    have hm : D.face ((n : ℤ) + 1).toNat τ j.1 ∈ D.idxF (n : ℤ) := TauData.face_mem hX ⟨τ, hτ⟩ j
    show ((unitOf (sgnPow j.1) : ℤˣ) : ℤ) • XBulk.genQ (D.toXBulk hX) (n : ℤ) (M22.toSL γ * 1,
      ⟨D.face ((n : ℤ) + 1).toNat τ j.1, hm⟩) = _
    rw [unitOf_val (sgnPow_cases _), mul_one]
    have hm' : D.face (n + 1) τ j.1 ∈ D.idxF (n : ℤ) := by rw [← e]; exact hm
    rw [genAt, dif_pos hm']
    congr 3
  exact (Finset.sum_congr rfl (fun j _ => key j)).trans
    (sum_fin_nf n (fun f => sgnPow f • genAt D hX (n : ℤ) γ (D.face (n + 1) τ f)))

lemma list_sum_range {M : Type*} [AddCommMonoid M] (k : ℕ) (F : ℕ → M) :
    ((List.range k).map F).sum = ∑ f ∈ Finset.range k, F f := by
  induction k with
  | zero => simp
  | succ k ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

lemma list_finset_comm {M : Type*} [AddCommMonoid M] {α : Type*} (l : List α) (S : Finset ℕ) (F : α → ℕ → M) :
    (l.map fun t => ∑ f ∈ S, F t f).sum = ∑ f ∈ S, (l.map fun t => F t f).sum := by
  induction l with
  | nil => simp
  | cons t l ih => simp only [List.map_cons, List.sum_cons, ih, Finset.sum_add_distrib]

/-- **`∂` of a stored term list is the checker's `bdry`**, read. -/
theorem bd_absT (q : ℤ) (n : ℕ) (hqn : q = n) (ts : List Term)
    (h : ∀ t ∈ ts, t.g.det = 1 ∧ t.sigma ∈ D.idxF (q + 1)) :
    XBulk.bd (D.toXBulk hX) q (absT D hX (q + 1) ts) = absC D hX q (D.bdry (n + 1) ts) := by
  subst hqn
  have lhs : XBulk.bd (D.toXBulk hX) (n : ℤ) (absT D hX ((n : ℤ) + 1) ts) =
      (ts.map fun t => ∑ f ∈ Finset.range (n + 2), sgnPow f • t.coeff • genAt D hX (n : ℤ) t.g (D.face (n + 1) t.sigma f)).sum := by
    unfold absT
    rw [map_list_sum, List.map_map]
    congr 1
    apply List.map_congr_left
    intro t ht
    simp only [Function.comp_apply, map_zsmul]
    rw [bd_genAt D hX (n : ℤ) n rfl t.g t.sigma (h t ht).2, Finset.smul_sum]
    refine Finset.sum_congr rfl fun f _ => ?_
    rw [smul_comm]
  rw [lhs, list_finset_comm]
  unfold TauData.bdry
  rw [absC_flatten, List.map_map, list_sum_range]
  refine Finset.sum_congr rfl fun f hf => ?_
  simp only [Function.comp_apply, Nat.add_sub_cancel]
  have hface : ∀ t ∈ ts.map (fun t => { t with sigma := D.face (n + 1) t.sigma f }),
      t.g.det = 1 ∧ t.sigma ∈ D.idxF (n : ℕ) := by
    intro t' ht'
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 ht'
    refine ⟨(h t ht).1, ?_⟩
    have hτ := (h t ht).2
    have hj : f < nf (n : ℤ) := by rw [nf_natCast]; exact Finset.mem_range.1 hf
    have := TauData.face_mem hX (q := (n : ℤ)) ⟨t.sigma, hτ⟩ ⟨f, hj⟩
    have e : ((n : ℤ) + 1).toNat = n + 1 := by omega
    rw [e] at this; exact this
  rw [absC_canonTerms D hX n (sgnPow f) _ hface]
  unfold absT
  rw [List.smul_sum, List.map_map, List.map_map]
  congr 1

/-- `⟨1,0,0,1⟩ * x = x` for the checker's matrices. -/
lemma M22.one_mul' (x : M22) : (⟨1, 0, 0, 1⟩ : M22) * x = x := by
  rw [M22.mul_def]; cases x; simp

/-- The (T1) helper at a coset representative is a plain lookup. -/
lemma tauT1_sAt (D' : TauData) (e : ℤ) (j q σ : ℕ) (hj : D'.classOf (D'.sAt j) = j ∧ D'.conj (D'.sAt j) = ⟨1, 0, 0, 1⟩)
    (hj0 : j ≠ 0) : D'.tauT1 e (D'.sAt j) q σ = D'.canonTerms q e (D'.lookup j q σ) := by
  unfold TauData.tauT1
  rw [if_neg (by rw [hj.1]; exact hj0), hj.1, hj.2]
  congr 1
  conv_rhs => rw [← List.map_id (D'.lookup j q σ)]
  apply List.map_congr_left
  intro t _
  cases t; simp [M22.one_mul']

/-- A term of a looked-up slot lies in a stored slot with those keys. -/
lemma mem_lookup (D' : TauData) {j q σ : ℕ} {t : Term} (ht : t ∈ D'.lookup j q σ) :
    ∃ s ∈ D'.slots, s.i = j ∧ s.q = q ∧ s.sigma = σ ∧ t ∈ s.terms := by
  unfold TauData.lookup at ht
  split at ht
  · rename_i s hs
    have hm := List.mem_of_find?_eq_some hs
    have hp := List.find?_some hs
    simp only [decide_eq_true_eq] at hp
    exact ⟨s, hm, hp.1, hp.2.1, hp.2.2, ht⟩
  · simp at ht

/-- A term of a looked-up η-slot lies in a stored slot with those keys. -/
lemma mem_lookupE (E : EtaData) {j q σ : ℕ} {t : Term} (ht : t ∈ E.lookup j q σ) :
    ∃ s ∈ E.slots, s.i = j ∧ s.q = q ∧ s.sigma = σ ∧ t ∈ s.terms := by
  unfold EtaData.lookup at ht
  split at ht
  · rename_i s hs
    have hm := List.mem_of_find?_eq_some hs
    have hp := List.find?_some hs
    simp only [decide_eq_true_eq] at hp
    exact ⟨s, hm, hp.1, hp.2.1, hp.2.2, ht⟩
  · simp at ht

/-- If a slot with the keys exists, `lookup` reads one. -/
lemma exists_lookupE (E : EtaData) {j q σ : ℕ} (h : ∃ s ∈ E.slots, s.i = j ∧ s.q = q ∧ s.sigma = σ) :
    ∃ s ∈ E.slots, s.i = j ∧ s.q = q ∧ s.sigma = σ ∧ E.lookup j q σ = s.terms := by
  unfold EtaData.lookup
  cases hf : E.slots.find? (fun s => s.i = j ∧ s.q = q ∧ s.sigma = σ) with
  | none =>
    obtain ⟨s, hs, h1, h2, h3⟩ := h
    rw [List.find?_eq_none] at hf
    exact absurd (by simp [h1, h2, h3]) (hf s hs)
  | some s =>
    have hm := List.mem_of_find?_eq_some hf
    have hp := List.find?_some hf
    simp only [decide_eq_true_eq] at hp
    exact ⟨s, hm, hp.1, hp.2.1, hp.2.2, rfl⟩

/-- The η helper at a coset representative is a plain lookup. -/
lemma etaT1_sAt (E : EtaData) (e : ℤ) (j q σ : ℕ)
    (hj : E.base.classOf (E.base.sAt j) = j ∧ E.base.conj (E.base.sAt j) = ⟨1, 0, 0, 1⟩) (hj0 : j ≠ 0) :
    E.etaT1 e (E.base.sAt j) q σ = E.base.canonTerms (q + 1) e (E.lookup j q σ) := by
  unfold EtaData.etaT1
  rw [if_neg (by rw [hj.1]; exact hj0), hj.1, hj.2]
  congr 1
  conv_rhs => rw [← List.map_id (E.lookup j q σ)]
  apply List.map_congr_left
  intro t _
  cases t; simp [M22.one_mul']

/-! ### The boundary descriptor's generators -/

section Bdry

variable (Xb : XBdryDatum (D.toXBulk hX))

/-- `[γ, σ]` in `C_q(X_∂)`, zero unless `σ` is a stored boundary `q`-simplex. -/
def genAtB (q : ℤ) (γ : M22) (σ : ℕ) : XBulk.chains Xb.toXBulk q := by
  classical
  exact if h : ∃ hσ : σ ∈ D.idxF q, Xb.B q ⟨σ, hσ⟩ then XBulk.genQ Xb.toXBulk q (M22.toSL γ, ⟨⟨σ, h.1⟩, h.2⟩)
    else 0

lemma genAtB_pos {q : ℤ} {γ : M22} {σ : ℕ} (h : ∃ hσ : σ ∈ D.idxF q, Xb.B q ⟨σ, hσ⟩) :
    genAtB D hX Xb q γ σ = XBulk.genQ Xb.toXBulk q (M22.toSL γ, ⟨⟨σ, h.1⟩, h.2⟩) := by
  unfold genAtB; simp only [dif_pos h]

/-- A stored term list read in `C_q(X_∂)`. -/
def absTB (q : ℤ) (ts : List Term) : XBulk.chains Xb.toXBulk q :=
  (ts.map fun t => t.coeff • genAtB D hX Xb q t.g t.sigma).sum

lemma inc_genAtB {q : ℤ} {γ : M22} {σ : ℕ} (h : ∃ hσ : σ ∈ D.idxF q, Xb.B q ⟨σ, hσ⟩) :
    Xb.inc q (genAtB D hX Xb q γ σ) = genAt D hX q γ σ := by
  rw [genAtB_pos D hX Xb h, XBdryDatum.inc_genQ, genAt, dif_pos h.1]

lemma inc_absTB {q : ℤ} (ts : List Term) (h : ∀ t ∈ ts, ∃ hσ : t.sigma ∈ D.idxF q, Xb.B q ⟨t.sigma, hσ⟩) :
    Xb.inc q (absTB D hX Xb q ts) = absT D hX q ts := by
  unfold absTB absT
  rw [map_list_sum, List.map_map]
  congr 1
  apply List.map_congr_left
  intro t ht
  simp only [Function.comp_apply, map_zsmul, inc_genAtB D hX Xb (h t ht)]

/-- **The boundary of a boundary generator** is the signed face walk, inside `X_∂`. -/
theorem bdB_genAtB (q : ℤ) (n : ℕ) (hqn : q = n) (γ : M22) (σ : ℕ)
    (h : ∃ hσ : σ ∈ D.idxF (q + 1), Xb.B (q + 1) ⟨σ, hσ⟩) :
    XBulk.bd Xb.toXBulk q (genAtB D hX Xb (q + 1) γ σ) =
      ∑ f ∈ Finset.range (n + 2), sgnPow f • genAtB D hX Xb q γ (D.face (n + 1) σ f) := by
  subst hqn
  rw [genAtB_pos D hX Xb h, bd_genQ_gen]
  have e : ((n : ℤ) + 1).toNat = n + 1 := by omega
  have key : ∀ j : Fin (nf (n : ℤ)), (((Xb.toXBulk.face (n : ℤ) ⟨⟨σ, h.1⟩, h.2⟩ j).2.2 : ℤˣ) : ℤ) •
      XBulk.genQ Xb.toXBulk (n : ℤ) (M22.toSL γ * (Xb.toXBulk.face (n : ℤ) ⟨⟨σ, h.1⟩, h.2⟩ j).1,
        (Xb.toXBulk.face (n : ℤ) ⟨⟨σ, h.1⟩, h.2⟩ j).2.1) =
      sgnPow j.1 • genAtB D hX Xb (n : ℤ) γ (D.face (n + 1) σ j.1) := by
    intro j
    have hm : D.face ((n : ℤ) + 1).toNat σ j.1 ∈ D.idxF (n : ℤ) := TauData.face_mem hX ⟨σ, h.1⟩ j
    have hB := Xb.closed_face (n : ℤ) ⟨σ, h.1⟩ j h.2
    show ((unitOf (sgnPow j.1) : ℤˣ) : ℤ) • XBulk.genQ Xb.toXBulk (n : ℤ) (M22.toSL γ * 1,
      ⟨⟨D.face ((n : ℤ) + 1).toNat σ j.1, hm⟩, hB⟩) = _
    rw [unitOf_val (sgnPow_cases _), mul_one]
    have hm' : D.face (n + 1) σ j.1 ∈ D.idxF (n : ℤ) := by rw [← e]; exact hm
    have hB' : Xb.B (n : ℤ) ⟨D.face (n + 1) σ j.1, hm'⟩ := by
      have : (⟨D.face (n + 1) σ j.1, hm'⟩ : D.Vz (n : ℤ)) = ⟨D.face ((n : ℤ) + 1).toNat σ j.1, hm⟩ := by
        apply Subtype.ext; simp only; rw [e]
      rw [this]; exact hB
    rw [genAtB_pos D hX Xb ⟨hm', hB'⟩]
    congr 3
  exact (Finset.sum_congr rfl (fun j _ => key j)).trans
    (sum_fin_nf n (fun f => sgnPow f • genAtB D hX Xb (n : ℤ) γ (D.face (n + 1) σ f)))

end Bdry

/-- Free-degree forms of the reading lemmas (the degree as any integer equal to `n`). -/
lemma absC_canonTerms_self' (q : ℤ) (n : ℕ) (hqn : q = n) (e : ℤ) (ts : List Term)
    (h : ∀ t ∈ ts, D.canon t.g n t.sigma = ((n, t.sigma, t.g), 1)) :
    absC D hX q (D.canonTerms n e ts) = e • absT D hX q ts := by
  subst hqn; exact absC_canonTerms_self D hX n e ts h

lemma view_canonTerms (B : TauBdData) : B.view.canonTerms = B.base.canonTerms := rfl

/-- A stored simplex (`isSimplex`) is in `idxF`. -/
lemma mem_idxF_of_any (q : ℕ) (σ : ℕ) (h : (D.simplices.any fun y => y.1 = q ∧ y.2.1 = σ) = true) :
    σ ∈ D.idxF (q : ℤ) := by
  rw [List.any_eq_true] at h
  obtain ⟨y, hy, hp⟩ := h
  simp only [decide_eq_true_eq] at hp
  exact (D.mem_idxF).2 (TauData.mem_idxL.2 ⟨y, hy, by rw [hp.1], hp.2⟩)

end EtaRead

/-! ### The theorem -/

namespace EtaRead

variable {E : EtaData} (hE : EtaSpec E) (hX : E.base.XAx) (Xb : XBdryDatum (E.base.toXBulk hX))

/-- `τ` realises the bulk τ-table on the generators `s_j · ι(σ)`. -/
def RealisesTau {g : GLQ} (τ : TwistedChainDatum ΓQ g (Xb.pairVia eQ.toMonoidHom).bulk) : Prop :=
  ∀ (q : ℤ) (j σ : ℕ), 0 ≤ q → 1 ≤ j → j ≤ E.base.m → σ ∈ E.base.idxF q →
    τ.τ q (genAt E.base hX q (E.base.sAt j) σ) = absT E.base hX q (E.base.lookup j q.toNat σ)

/-- `τ_∂` realises the `τ_∂`-table on the boundary generators `s_j · ι(σ)`. -/
def RealisesTauBd {g : GLQ} (τb : TwistedChainDatum ΓQ g (Xb.pairVia eQ.toMonoidHom).bdry) : Prop :=
  ∀ (q : ℤ) (j σ : ℕ), 0 ≤ q → 1 ≤ j → j ≤ E.base.m → (q.toNat, σ) ∈ E.bd.bdry →
    τb.τ q (genAtB E.base hX Xb q (E.base.sAt j) σ) = absTB E.base hX Xb q (E.bd.view.lookup j q.toNat σ)

/-- `η` realises the η-table on the boundary generators `s_j · ι(σ)`. -/
def RealisesEta (η : ∀ q, (Xb.pairVia eQ.toMonoidHom).bdry.X q →+ (Xb.pairVia eQ.toMonoidHom).bulk.X (q + 1)) :
    Prop :=
  ∀ (q : ℤ) (j σ : ℕ), 0 ≤ q → 1 ≤ j → j ≤ E.base.m → (q.toNat, σ) ∈ E.bd.bdry →
    η q (genAtB E.base hX Xb q (E.base.sAt j) σ) = absT E.base hX (q + 1) (E.lookup j q.toNat σ)

/-- **Spreading by (T1)**: two `g`-twisted maps out of `C_n(X_∂)` that agree on the generators `s_j · ι(σ)` agree,
when the `s_j` cover `Γ₀(N)\SL(2, ℤ)`. -/
theorem T1_ext_gen {N : ℤ} (hN : N ≠ 0) (m : ℕ) (s : ℕ → M22)
    (hcov : ∀ γ : SL(2, ℤ), ∃ j, 1 ≤ j ∧ j ≤ m ∧ N ∣ (γ * (M22.toSL (s j))⁻¹) 1 0)
    {n k : ℤ} (F F' : (Xb.pairVia eQ.toMonoidHom).bdry.X n →+ (Xb.pairVia eQ.toMonoidHom).bulk.X k)
    (hF : T1 ΓQ (gQ N hN) F) (hF' : T1 ΓQ (gQ N hN) F')
    (hgen : ∀ j σ, 1 ≤ j → j ≤ m → (∃ hσ : σ ∈ E.base.idxF n, Xb.B n ⟨σ, hσ⟩) →
      F (genAtB E.base hX Xb n (s j) σ) = F' (genAtB E.base hX Xb n (s j) σ)) : F = F' := by
  have ext := XBulk.chains_hom_ext (q := n) (A := XBulk.chains (E.base.toXBulk hX) k) Xb.toXBulk (u := F) (v := F')
  apply ext
  intro z
  obtain ⟨j, hj1, hjm, hdvd⟩ := hcov z.1
  set δ := z.1 * (M22.toSL (s j))⁻¹ with hδ
  have hmem : (gQ N hN)⁻¹ * ((eQ.symm δ : ΓQ) : GLQ) * gQ N hN ∈ ΓQ := by
    rw [coe_eQ_symm]; exact (conj_mem_ΓQ_iff N hN δ).2 hdvd
  let h' : ΓQ := ⟨_, hmem⟩
  have hσ : ∃ hσ : z.2.1.1 ∈ E.base.idxF n, Xb.B n ⟨z.2.1.1, hσ⟩ := ⟨z.2.1.2, z.2.2⟩
  have hz : XBulk.genQ Xb.toXBulk n z = eQ.toMonoidHom (eQ.symm δ) • genAtB E.base hX Xb n (s j) z.2.1.1 := by
    rw [genAtB_pos E.base hX Xb hσ, MulEquiv.coe_toMonoidHom, MulEquiv.apply_symm_apply, XBulk.smul_genQ]
    congr 1
    rw [hδ, inv_mul_cancel_right]
    rfl
  rw [hz]
  have e1 := hF (eQ.symm δ) h' rfl (genAtB E.base hX Xb n (s j) z.2.1.1 : (Xb.pairVia eQ.toMonoidHom).bdry.X n)
  have e2 := hF' (eQ.symm δ) h' rfl (genAtB E.base hX Xb n (s j) z.2.1.1 : (Xb.pairVia eQ.toMonoidHom).bdry.X n)
  rw [hgen j _ hj1 hjm hσ] at e1
  exact e1.trans e2.symm

include hE in
/-- The η-slot of a boundary generator exists, and `lookup` reads it. -/
lemma slot_of (hdim : 2 ≤ E.dimX) {j n σ : ℕ} (hj1 : 1 ≤ j) (hjm : j ≤ E.base.m) (hb : (n, σ) ∈ E.bd.bdry)
    (hn : n ≤ 1) : ∃ s ∈ E.slots, s.i = j ∧ s.q = n ∧ s.sigma = σ ∧ E.lookup j n σ = s.terms := by
  apply exists_lookupE
  have hw : (j, n, σ) ∈ E.wantKeys := by
    simp only [EtaData.wantKeys, List.mem_flatMap, List.mem_map, List.mem_range, List.mem_filter,
      decide_eq_true_eq]
    exact ⟨j, ⟨j - 1, by omega, by omega⟩, (n, σ), ⟨hb, by simp only; omega⟩, rfl⟩
  have := hE.complete.2.mem_iff.2 hw
  simp only [EtaData.slotKeys, List.mem_map] at this
  obtain ⟨s, hs, he⟩ := this
  simp only [Prod.mk.injEq] at he
  exact ⟨s, hs, he.1, he.2.1, he.2.2⟩

include hE in
lemma base_self {j n σ : ℕ} : ∀ t ∈ E.base.lookup j n σ, E.base.canon t.g n t.sigma = ((n, t.sigma, t.g), 1) := by
  intro t ht
  obtain ⟨s, hs, -, hq, -, hts⟩ := mem_lookup E.base ht
  have := (hE.tau.struct s hs t hts).2.2
  rwa [hq] at this

include hE in
lemma view_self {j n σ : ℕ} :
    ∀ t ∈ E.bd.view.lookup j n σ, E.base.canon t.g n t.sigma = ((n, t.sigma, t.g), 1) := by
  intro t ht
  obtain ⟨s, hs, -, hq, -, hts⟩ := mem_lookup E.bd.view ht
  have := (hE.taubd.struct s hs t hts).2.2.2
  rw [hq] at this; exact this

include hE in
lemma eta_self {j n σ : ℕ} :
    ∀ t ∈ E.lookup j n σ, E.base.canon t.g (n + 1) t.sigma = ((n + 1, t.sigma, t.g), 1) := by
  intro t ht
  obtain ⟨s, hs, -, hq, -, hts⟩ := mem_lookupE E ht
  have := (hE.struct s hs t hts).2.2.2
  rwa [hq] at this

include hE in
lemma eta_detIdx {s : Slot} (hs : s ∈ E.slots) :
    ∀ t ∈ s.terms, t.g.det = 1 ∧ t.sigma ∈ E.base.idxF ((s.q + 1 : ℕ) : ℤ) := by
  intro t ht
  obtain ⟨-, hd, hsimp, -⟩ := hE.struct s hs t ht
  exact ⟨hd, mem_idxF_of_any E.base (s.q + 1) t.sigma hsimp⟩

include hE in
/-- The `τ_∂`-terms are boundary simplices. -/
lemma view_bdry (hXb : ∀ q σ, Xb.B q σ ↔ (0 ≤ q ∧ (q.toNat, σ.1) ∈ E.bd.bdry)) {q : ℤ} (hq : 0 ≤ q) {j σ : ℕ} :
    ∀ t ∈ E.bd.view.lookup j q.toNat σ, ∃ hσ : t.sigma ∈ E.base.idxF q, Xb.B q ⟨t.sigma, hσ⟩ := by
  intro t ht
  obtain ⟨s, hs, -, hsq, -, hts⟩ := mem_lookup E.bd.view ht
  have hb := (hE.taubd.struct s hs t hts).2.2.1
  rw [hsq] at hb
  have hany := (hE.taubd.closed.2).2.1 _ hb
  have hm : t.sigma ∈ E.base.idxF ((q.toNat : ℕ) : ℤ) := mem_idxF_of_any E.base q.toNat t.sigma hany
  rw [Int.toNat_of_nonneg hq] at hm
  exact ⟨hm, (hXb _ _).2 ⟨hq, hb⟩⟩

/-- The `τ_∂` helper at `s_j`, through the view. -/
lemma tauT1_view (e : ℤ) {j : ℕ} (q σ : ℕ)
    (hj : E.base.classOf (E.base.sAt j) = j ∧ E.base.conj (E.base.sAt j) = ⟨1, 0, 0, 1⟩) (hj0 : j ≠ 0) :
    E.bd.view.tauT1 e (E.base.sAt j) q σ = E.base.canonTerms q e (E.bd.view.lookup j q σ) :=
  tauT1_sAt E.bd.view e j q σ hj hj0

include hE in
/-- **`thm:etaspec-pairhomotopy`** -- `EtaSpec → IsPairHomotopy`.  For an η-table accepted by the checker over a
boundary descriptor `X_∂` with `E`'s boundary simplices, `g = diag(1, N)`, and coset representatives `s_j` covering
`Γ₀(N)\SL(2, ℤ)` (each in its own class, `hcl`): every (T1) map `η` realising the η-table is a pair homotopy for every
`τ`, `τ_∂` realising the τ- and `τ_∂`-tables -- `∂η + η∂ = j τ_∂ − τ j` on all of `C_•(X_∂)`. -/
theorem _root_.GL2.TauChk.EtaSpec.toIsPairHomotopy
    (hXb : ∀ q σ, Xb.B q σ ↔ (0 ≤ q ∧ (q.toNat, σ.1) ∈ E.bd.bdry))
    (hN : (E.base.N : ℤ) ≠ 0) (hdim : 2 ≤ E.dimX)
    (hcl : ∀ j, 1 ≤ j → j ≤ E.base.m →
      E.base.classOf (E.base.sAt j) = j ∧ E.base.conj (E.base.sAt j) = ⟨1, 0, 0, 1⟩)
    (hcov : ∀ γ : SL(2, ℤ), ∃ j, 1 ≤ j ∧ j ≤ E.base.m ∧ (E.base.N : ℤ) ∣ (γ * (M22.toSL (E.base.sAt j))⁻¹) 1 0)
    (τ : TwistedChainDatum ΓQ (gQ E.base.N hN) (Xb.pairVia eQ.toMonoidHom).bulk)
    (τb : TwistedChainDatum ΓQ (gQ E.base.N hN) (Xb.pairVia eQ.toMonoidHom).bdry)
    (η : ∀ q, (Xb.pairVia eQ.toMonoidHom).bdry.X q →+ (Xb.pairVia eQ.toMonoidHom).bulk.X (q + 1))
    (hηT1 : ∀ q, T1 ΓQ (gQ E.base.N hN) (η q))
    (hτ : RealisesTau hX Xb τ) (hτb : RealisesTauBd hX Xb τb) (hη : RealisesEta hX Xb η) :
    IsPairHomotopy ΓQ (Xb.pairVia eQ.toMonoidHom) τ τb η where
  t1 := hηT1
  htpy q y := by
    by_cases hlo : q + 1 < 0
    · have : Subsingleton ((Xb.pairVia eQ.toMonoidHom).bulk.X (q + 1)) :=
        XBulk.subsingleton_neg (E.base.toXBulk hX) (q + 1) hlo
      exact Subsingleton.elim _ _
    by_cases hhi : 2 ≤ q + 1
    · have : Subsingleton ((Xb.pairVia eQ.toMonoidHom).bdry.X (q + 1)) :=
        XBulk.subsingleton_of_isEmpty Xb.toXBulk _ (Xb.isEmpty_dim _ hhi)
      rw [Subsingleton.elim y 0]
      simp only [map_zero, add_zero, sub_zero]
    have hL : T1 ΓQ (gQ E.base.N hN)
        (((Xb.pairVia eQ.toMonoidHom).bulk.bd (q + 1)).comp (η (q + 1)) + (η q).comp ((Xb.pairVia eQ.toMonoidHom).bdry.bd q)) :=
      ((hηT1 (q + 1)).comp_left ((Xb.pairVia eQ.toMonoidHom).bulk.bd_equiv (q + 1))).add ((hηT1 q).comp_right ((Xb.pairVia eQ.toMonoidHom).bdry.bd_equiv q))
    have hR : T1 ΓQ (gQ E.base.N hN) (((Xb.pairVia eQ.toMonoidHom).j (q + 1)).comp (τb.τ (q + 1)) -
        (τ.τ (q + 1)).comp ((Xb.pairVia eQ.toMonoidHom).j (q + 1))) :=
      ((τb.t1 (q + 1)).comp_left ((Xb.pairVia eQ.toMonoidHom).j_equiv (q + 1))).sub ((τ.t1 (q + 1)).comp_right ((Xb.pairVia eQ.toMonoidHom).j_equiv (q + 1)))
    suffices key : ((Xb.pairVia eQ.toMonoidHom).bulk.bd (q + 1)).comp (η (q + 1)) +
        (η q).comp ((Xb.pairVia eQ.toMonoidHom).bdry.bd q) =
        ((Xb.pairVia eQ.toMonoidHom).j (q + 1)).comp (τb.τ (q + 1)) -
        (τ.τ (q + 1)).comp ((Xb.pairVia eQ.toMonoidHom).j (q + 1)) from DFunLike.congr_fun key y
    refine T1_ext_gen hX Xb hN E.base.m E.base.sAt hcov _ _ hL hR (fun j σ hj1 hjm hσ => ?_)
    have hj0 : j ≠ 0 := by omega
    -- chain-typed copies of `η`, `τ`, `τ_∂` (the same maps)
    let η' : ∀ r, XBulk.chains Xb.toXBulk r →+ XBulk.chains (E.base.toXBulk hX) (r + 1) :=
      fun r => AddMonoidHom.mk' (fun x => η r x) (fun a b => map_add (η r) a b)
    let τ' : ∀ r, XBulk.chains (E.base.toXBulk hX) r →+ XBulk.chains (E.base.toXBulk hX) r :=
      fun r => AddMonoidHom.mk' (fun x => τ.τ r x) (fun a b => map_add (τ.τ r) a b)
    let τb' : ∀ r, XBulk.chains Xb.toXBulk r →+ XBulk.chains Xb.toXBulk r :=
      fun r => AddMonoidHom.mk' (fun x => τb.τ r x) (fun a b => map_add (τb.τ r) a b)
    have hη' : ∀ (r : ℤ) (j σ : ℕ), 0 ≤ r → 1 ≤ j → j ≤ E.base.m → (r.toNat, σ) ∈ E.bd.bdry →
        η' r (genAtB E.base hX Xb r (E.base.sAt j) σ) = absT E.base hX (r + 1) (E.lookup j r.toNat σ) :=
      fun r j σ h1 h2 h3 h4 => hη r j σ h1 h2 h3 h4
    have hτ' : ∀ (r : ℤ) (j σ : ℕ), 0 ≤ r → 1 ≤ j → j ≤ E.base.m → σ ∈ E.base.idxF r →
        τ' r (genAt E.base hX r (E.base.sAt j) σ) = absT E.base hX r (E.base.lookup j r.toNat σ) :=
      fun r j σ h1 h2 h3 h4 => hτ r j σ h1 h2 h3 h4
    have hτb' : ∀ (r : ℤ) (j σ : ℕ), 0 ≤ r → 1 ≤ j → j ≤ E.base.m → (r.toNat, σ) ∈ E.bd.bdry →
        τb' r (genAtB E.base hX Xb r (E.base.sAt j) σ) = absTB E.base hX Xb r (E.bd.view.lookup j r.toNat σ) :=
      fun r j σ h1 h2 h3 h4 => hτb r j σ h1 h2 h3 h4
    show XBulk.bd (E.base.toXBulk hX) (q + 1) (η' (q + 1) (genAtB E.base hX Xb (q + 1) (E.base.sAt j) σ)) +
        η' q (XBulk.bd Xb.toXBulk q (genAtB E.base hX Xb (q + 1) (E.base.sAt j) σ)) =
      Xb.inc (q + 1) (τb' (q + 1) (genAtB E.base hX Xb (q + 1) (E.base.sAt j) σ)) -
        τ' (q + 1) (Xb.inc (q + 1) (genAtB E.base hX Xb (q + 1) (E.base.sAt j) σ))
    obtain rfl | rfl : q = -1 ∨ q = 0 := by omega
    · -- vertices: `∂ η(s_j v) = (j τ_∂ − τ j)(s_j v)`
      have hb' := ((hXb _ _).1 hσ.2).2
      have h0 : ((-1 : ℤ) + 1).toNat = 0 := by norm_num
      have hb : (0, σ) ∈ E.bd.bdry := by rw [h0] at hb'; exact hb'
      obtain ⟨s, hs, hsi, hsq, hss, hlook⟩ := slot_of hE hdim hj1 hjm hb (by norm_num)
      have H := absC_eq_of_eval E.base hX (-1 + 1) (hE.homotopy s hs)
      unfold EtaData.homLhs EtaData.defect at H
      have hdi := eta_detIdx hE hs
      have hcast : ((s.q + 1 : ℕ) : ℤ) = -1 + 1 + 1 := by rw [hsq]; norm_num
      rw [hcast] at hdi
      rw [absC_append, absC_append, hsi, hsq, hss] at H
      have hnil : E.etaBdry j 0 σ = [] := by simp [EtaData.etaBdry]
      rw [hnil, absC_nil, add_zero, tauT1_view 1 0 σ (hcl j hj1 hjm) hj0,
        absC_canonTerms_self' E.base hX (-1 + 1) 0 (by norm_num) 1 _ (view_self hE),
        absC_canonTerms_self' E.base hX (-1 + 1) 0 (by norm_num) (-1) _ (base_self hE),
        ← bd_absT E.base hX (-1 + 1) 0 (by norm_num) s.terms hdi] at H
      have hz : XBulk.bd Xb.toXBulk (-1) (genAtB E.base hX Xb (-1 + 1) (E.base.sAt j) σ) = 0 := by
        have : Subsingleton (XBulk.chains Xb.toXBulk (-1)) := XBulk.subsingleton_neg Xb.toXBulk (-1) (by norm_num)
        exact Subsingleton.elim _ _
      rw [hz, map_zero, add_zero, hη' (-1 + 1) j σ (by norm_num) hj1 hjm hb', h0, hlook,
        inc_genAtB E.base hX Xb hσ, hτ' (-1 + 1) j σ (by norm_num) hj1 hjm hσ.1, h0,
        hτb' (-1 + 1) j σ (by norm_num) hj1 hjm hb', inc_absTB E.base hX Xb _ (view_bdry hE hX Xb hXb (by norm_num)), h0,
        H, one_smul, neg_one_zsmul, sub_eq_add_neg]
    · -- edges: `∂ η(s_j e) + η(∂(s_j e)) = (j τ_∂ − τ j)(s_j e)`
      have hb' := ((hXb _ _).1 hσ.2).2
      have h1 : ((0 : ℤ) + 1).toNat = 1 := by norm_num
      have hb : (1, σ) ∈ E.bd.bdry := by rw [h1] at hb'; exact hb'
      obtain ⟨s, hs, hsi, hsq, hss, hlook⟩ := slot_of hE hdim hj1 hjm hb le_rfl
      have H := absC_eq_of_eval E.base hX (0 + 1) (hE.homotopy s hs)
      unfold EtaData.homLhs EtaData.defect at H
      have hdi := eta_detIdx hE hs
      have hcast : ((s.q + 1 : ℕ) : ℤ) = 0 + 1 + 1 := by rw [hsq]; norm_num
      rw [hcast] at hdi
      rw [absC_append, absC_append, hsi, hsq, hss] at H
      -- `η(∂(s_j e))` on the checker side
      have hfaces : ∀ f ∈ List.range 2, (0, E.base.face 1 σ f) ∈ E.bd.bdry :=
        (hE.taubd.closed.2).2.2.2 _ hb rfl
      have hetaB : absC E.base hX (0 + 1) (E.etaBdry j 1 σ) =
          ∑ f ∈ Finset.range 2, sgnPow f • absT E.base hX (0 + 1) (E.lookup j 0 (E.base.face 1 σ f)) := by
        unfold EtaData.etaBdry
        rw [if_neg one_ne_zero, absC_flatten, List.map_map, list_sum_range]
        refine Finset.sum_congr rfl fun f _ => ?_
        simp only [Function.comp_apply]
        rw [etaT1_sAt E (sgnPow f) j (1 - 1) _ (hcl j hj1 hjm) hj0,
          absC_canonTerms_self' E.base hX (0 + 1) (1 - 1 + 1) (by norm_num) _ _ (eta_self hE)]
      rw [hetaB, tauT1_view 1 1 σ (hcl j hj1 hjm) hj0,
        absC_canonTerms_self' E.base hX (0 + 1) 1 (by norm_num) 1 _ (view_self hE),
        absC_canonTerms_self' E.base hX (0 + 1) 1 (by norm_num) (-1) _ (base_self hE),
        ← bd_absT E.base hX (0 + 1) 1 (by norm_num) s.terms hdi] at H
      -- `η(∂(s_j e))` on the abstract side
      have hbd := bdB_genAtB E.base hX Xb 0 0 (by norm_num) (E.base.sAt j) σ hσ
      have hηsum : η' 0 (XBulk.bd Xb.toXBulk 0 (genAtB E.base hX Xb (0 + 1) (E.base.sAt j) σ)) =
          ∑ f ∈ Finset.range 2, sgnPow f • absT E.base hX (0 + 1) (E.lookup j 0 (E.base.face 1 σ f)) := by
        rw [hbd, map_sum]
        refine Finset.sum_congr rfl fun f hf => ?_
        rw [map_zsmul, hη' 0 j _ le_rfl hj1 hjm (by simpa using hfaces f (by simpa using hf))]
        rfl
      rw [hηsum, hη' (0 + 1) j σ (by norm_num) hj1 hjm hb', h1, hlook,
        inc_genAtB E.base hX Xb hσ, hτ' (0 + 1) j σ (by norm_num) hj1 hjm hσ.1, h1,
        hτb' (0 + 1) j σ (by norm_num) hj1 hjm hb', inc_absTB E.base hX Xb _ (view_bdry hE hX Xb hXb (by norm_num)), h1,
        H, one_smul, neg_one_zsmul, sub_eq_add_neg]

end EtaRead

end GL2.Model

/-! ### The real `p = 2` table -/

namespace GL2.Model

open scoped MatrixGroups
open GL2.TauChk GL2.TauChk.Data GL2.Cone EtaRead

/-- `(γ s⁻¹)₂₁ = c d_s − d c_s` for `s` of determinant one. -/
lemma entry10 (γ : SL(2, ℤ)) (x : M22) (hx : x.det = 1) :
    (γ * (M22.toSL x)⁻¹ : SL(2, ℤ)) 1 0 = γ 1 0 * x.d - γ 1 1 * x.c := by
  rw [← M22.toSL_adj hx, Matrix.SpecialLinearGroup.coe_mul, M22.toSL_coe (by rw [M22.det_adj, hx])]
  simp [Matrix.mul_apply, Fin.sum_univ_two, M22.toMat, M22.adj]
  ring

/-- The three representatives `I, [[1,0],[1,1]], S` of `tauP2` cover `Γ₀(2)\SL(2, ℤ)`. -/
theorem tauP2_cover (γ : SL(2, ℤ)) :
    ∃ j, 1 ≤ j ∧ j ≤ tauP2.m ∧ (tauP2.N : ℤ) ∣ (γ * (M22.toSL (tauP2.sAt j))⁻¹ : SL(2, ℤ)) 1 0 := by
  have hdet : γ 0 0 * γ 1 1 - γ 0 1 * γ 1 0 = 1 := by
    have := γ.2; rw [Matrix.det_fin_two] at this; exact this
  have s1 : tauP2.sAt 1 = ⟨1, 0, 0, 1⟩ := rfl
  have s2 : tauP2.sAt 2 = ⟨1, 0, 1, 1⟩ := rfl
  have s3 : tauP2.sAt 3 = ⟨0, -1, 1, 0⟩ := rfl
  have hN : (tauP2.N : ℤ) = 2 := rfl
  rcases Int.emod_two_eq_zero_or_one (γ 1 0) with hc | hc
  · refine ⟨1, le_rfl, by decide, ?_⟩
    rw [s1, entry10 γ _ (by decide), hN]; simp only [mul_one, mul_zero, sub_zero]; omega
  rcases Int.emod_two_eq_zero_or_one (γ 1 1) with hd | hd
  · refine ⟨3, by decide, le_rfl, ?_⟩
    rw [s3, entry10 γ _ (by decide), hN]; simp only [mul_zero, mul_one, zero_sub]; omega
  · refine ⟨2, by decide, by decide, ?_⟩
    rw [s2, entry10 γ _ (by decide), hN]; simp only [mul_one]; omega

theorem tauP2_classes : ∀ j, 1 ≤ j → j ≤ tauP2.m →
    tauP2.classOf (tauP2.sAt j) = j ∧ tauP2.conj (tauP2.sAt j) = ⟨1, 0, 0, 1⟩ := by
  intro j h1 h2
  have : j = 1 ∨ j = 2 ∨ j = 3 := by have : tauP2.m = 3 := rfl; omega
  rcases this with rfl | rfl | rfl <;> decide

theorem etaP2_dimX : 2 ≤ etaP2R1.dimX := by decide

theorem tauP2_N_ne : ((tauP2.N : ℕ) : ℤ) ≠ 0 := by decide

/-- **`cor:etap2-pairhomotopy`**: on the 7-vertex model with its horocycle boundary (`model7Bd`) and
`g = diag(1, 2) ∈ GL(2, ℚ)`, every (T1) map `η` realising the checked `p = 2` η-table (`etaP2_spec`) is a pair
homotopy for every `τ`, `τ_∂` realising the checked τ- and `τ_∂`-tables. -/
theorem etaP2_pairHomotopy
    (τ : TwistedChainDatum ΓQ (gQ (tauP2.N : ℤ) tauP2_N_ne) (model7Bd.pairVia eQ.toMonoidHom).bulk)
    (τb : TwistedChainDatum ΓQ (gQ (tauP2.N : ℤ) tauP2_N_ne) (model7Bd.pairVia eQ.toMonoidHom).bdry)
    (η : ∀ q, (model7Bd.pairVia eQ.toMonoidHom).bdry.X q →+ (model7Bd.pairVia eQ.toMonoidHom).bulk.X (q + 1))
    (hηT1 : ∀ q, T1 ΓQ (gQ (tauP2.N : ℤ) tauP2_N_ne) (η q))
    (hτ : RealisesTau (E := etaP2R1) (of_decide_eq_true tauP2_xbulkB) model7Bd τ)
    (hτb : RealisesTauBd (E := etaP2R1) (of_decide_eq_true tauP2_xbulkB) model7Bd τb)
    (hη : RealisesEta (E := etaP2R1) (of_decide_eq_true tauP2_xbulkB) model7Bd η) :
    IsPairHomotopy ΓQ (model7Bd.pairVia eQ.toMonoidHom) τ τb η :=
  etaP2_spec.toIsPairHomotopy (of_decide_eq_true tauP2_xbulkB) model7Bd (fun _ _ => Iff.rfl) tauP2_N_ne etaP2_dimX
    tauP2_classes tauP2_cover τ τb η hηT1 hτ hτb hη

end GL2.Model

end
