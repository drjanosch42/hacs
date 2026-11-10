import GL2.Model.XBulk

/-!
# M2-d (L18-1): H4 over the combinatorial `X_bulk`

[GT] `prop:model7h4-i`, `prop:model7h4-iii`, `def:xbulk-acyclic`, `thm:pairhyp-of-acyclic`, mirroring [C] v1.17.0
`H4`(i)--(iii) and `prop:model7h4`.

For **every** `X_bulk` descriptor (no 7-vertex model, no `ℓ`):

* **H4(i)** -- `noInv`: a group element fixing a simplex fixes its oriented generator (no inversions);
  `finiteStab`: every simplex stabiliser is finite (`mem_stab_iff`: the stabiliser of `ι(σ)` is the transporter set
  `S(σ, σ)`, conjugated by `γ` at `γ ι(σ)`).
* **H4(iii)** -- `fundDomain` (FD2): every simplex is `γ · ι(σ)` with `σ` an orbit representative of the model;
  `fundDomain_strict` / `fundDomain_unique` (FD1, WP-17): distinct representatives are inequivalent.
* **H4(ii)** -- *not* combinatorial: `Acyclic` is exactly the three contraction clauses of the cone layer's `PairHyp`
  (`eq:contractibleuse`) for `toGChainCx`; the dimension clauses are *proved* from the descriptor (`subsingleton_neg`,
  `subsingleton_dim`), and `pairHyp_of_acyclic` is the bridge: `Acyclic` plus the boundary's own fields give `PairHyp`
  for any pair whose bulk is `X_bulk`.
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.Ku

namespace XBulk

variable (D : XBulkDatum)

/-- The oriented generator `[γ, σ] = +1 · e_(γ, σ)` in `C_q`. -/
def genQ (q : ℤ) (x : SL(2, ℤ) × D.V q) : chains D q := Submodule.Quotient.mk (gen ((x.1, x.2, 1) : OrPt (D.V q)))

lemma smul_genQ (q : ℤ) (γ : SL(2, ℤ)) (x : SL(2, ℤ) × D.V q) : γ • genQ D q x = genQ D q (γ * x.1, x.2) := by
  simp only [genQ, smul_mk, actF_gen]; rfl

/-- **H4(i), no inversions**: an element fixing the simplex `γ ι(σ)` fixes its oriented generator. -/
theorem noInv (q : ℤ) (γ : SL(2, ℤ)) (x : SL(2, ℤ) × D.V q)
    (h : γ • SimpM.mk (D.M q) x = SimpM.mk (D.M q) x) : γ • genQ D q x = genQ D q x := by
  rw [smul_simpMk, simpMk_eq_iff] at h
  rw [smul_genQ, genQ, genQ]
  have hr : OrRel (D.M q) ((γ * x.1, x.2, 1) : OrPt (D.V q)) (x.1, x.2, 1) :=
    ⟨h, by simp only; rw [(D.M q).noInv h, one_mul]⟩
  have := mk_gen_eq (D.M q) hr
  simpa only [Submodule.mkQ_apply] using this

/-- The stabiliser of `γ ι(σ)` is `γ S(σ, σ) γ⁻¹`. -/
lemma mem_stab_iff (q : ℤ) (x : SL(2, ℤ) × D.V q) (δ : SL(2, ℤ)) :
    δ ∈ MulAction.stabilizer SL(2, ℤ) (SimpM.mk (D.M q) x) ↔ x.1⁻¹ * δ * x.1 ∈ (D.M q).S x.2 x.2 := by
  rw [MulAction.mem_stabilizer_iff, smul_simpMk, simpMk_eq_iff]
  constructor
  · intro h
    have := (D.M q).inv_mem h
    convert this using 1; group
  · intro h
    have := (D.M q).inv_mem h
    unfold Ku.Rel; convert this using 1; group

/-- **H4(i), finite stabilisers.** -/
theorem finiteStab (q : ℤ) (x : SL(2, ℤ) × D.V q) :
    (MulAction.stabilizer SL(2, ℤ) (SimpM.mk (D.M q) x) : Set SL(2, ℤ)).Finite := by
  apply ((D.M q).S x.2 x.2 |>.finite_toSet.image (fun t => x.1 * t * x.1⁻¹)).subset
  intro δ hδ
  refine ⟨x.1⁻¹ * δ * x.1, (mem_stab_iff D q x δ).1 hδ, by group⟩

/-- **H4(iii), fundamental domain**: every simplex is `γ · ι(σ)` with `σ` an orbit representative (`σ̂ = σ`). -/
theorem fundDomain (q : ℤ) (s : Simp D q) :
    ∃ (γ : SL(2, ℤ)) (σ : D.V q), (D.M q).rep σ = σ ∧ s = γ • SimpM.mk (D.M q) (1, σ) := by
  induction s using Quot.ind with
  | _ x =>
    obtain ⟨t, ht⟩ := (D.M q).rep_mem x.2
    have hrr : (D.M q).rep ((D.M q).rep x.2) = (D.M q).rep x.2 := ((D.M q).rep_eq ⟨t, ht⟩).symm
    refine ⟨x.1 * t, (D.M q).rep x.2, hrr, ?_⟩
    rw [smul_simpMk, mul_one]
    show SimpM.mk _ x = _
    rw [simpMk_eq_iff]
    unfold Ku.Rel; simpa using ht

/-- **H4(iii), FD1** (WP-17, F3-18 MINOR 4): distinct orbit representatives are inequivalent --
`S(σ, σ') ≠ ∅` between representatives forces `σ = σ'`. -/
theorem fundDomain_strict (q : ℤ) {σ σ' : D.V q} (hσ : (D.M q).rep σ = σ) (hσ' : (D.M q).rep σ' = σ')
    (h : ((D.M q).S σ σ').Nonempty) : σ = σ' := by
  rw [← hσ, ← hσ', (D.M q).rep_eq h]

/-- **H4(iii), FD1 on simplices**: `γ ι(σ) = γ' ι(σ')` for representatives `σ, σ'` forces `σ = σ'`. -/
theorem fundDomain_unique (q : ℤ) {σ σ' : D.V q} (hσ : (D.M q).rep σ = σ) (hσ' : (D.M q).rep σ' = σ')
    (γ γ' : SL(2, ℤ)) (h : γ • SimpM.mk (D.M q) (1, σ) = γ' • SimpM.mk (D.M q) (1, σ')) : σ = σ' := by
  rw [smul_simpMk, smul_simpMk, simpMk_eq_iff] at h
  exact fundDomain_strict D q hσ hσ' ⟨_, h⟩

/-! ### H4(ii) and the bridge to the cone layer -/

/-- The chains vanish where the model has no simplices. -/
lemma subsingleton_of_isEmpty (q : ℤ) (h : IsEmpty (D.V q)) : Subsingleton (chains D q) := by
  have : Subsingleton (SL(2, ℤ) × D.V q →₀ ℤ) := inferInstance
  exact ⟨fun a b => by
    induction a using Submodule.Quotient.induction_on with
    | _ f => induction b using Submodule.Quotient.induction_on with
      | _ g => rw [Subsingleton.elim f g]⟩

lemma subsingleton_neg (q : ℤ) (hq : q < 0) : Subsingleton (chains D q) :=
  subsingleton_of_isEmpty D q (D.dim2 q (Or.inl hq))

/-- **H4(ii), the dimension half (proved)**: `X_bulk` is at most `2`-dimensional. -/
lemma subsingleton_dim (q : ℤ) (hq : 3 ≤ q) : Subsingleton (chains D q) :=
  subsingleton_of_isEmpty D q (D.dim2 q (Or.inr (by omega)))

/-- **H4(ii), the acyclicity half** as the cone layer uses it (`eq:contractibleuse`): exactly `PairHyp`'s
`contract0/1/2` for `toGChainCx`.  [C] proves it topologically (`prop:model7h4`(ii)). -/
structure Acyclic : Prop where
  contract0 : ∀ x : chains D 0, aug D x = 0 → ∃ y, bd D 0 y = x
  contract1 : ∀ x : chains D (0 + 1), bd D 0 x = 0 → ∃ y, bd D (0 + 1) y = x
  contract2 : ∀ x : chains D (0 + 1 + 1), bd D (0 + 1) x = 0 → x = 0

/-- **The bridge**: for any subgroup `H` and any boundary complex with its own `PairHyp` fields (negative degrees,
dimension `≤ 1`, the (T1)-extension property), an acyclic `X_bulk` gives `PairHyp` for the pair
`(X_bulk, X_∂)`.  So [C]'s H4(ii) enters the cone chapter exactly as `Acyclic`. -/
theorem pairHyp_of_acyclic (hA : Acyclic D) (H : Subgroup SL(2, ℤ)) (g : SL(2, ℤ))
    (bdry : GL2.Cone.GChainCx.{0} H) (j : ∀ q, bdry.X q →+ (toGChainCxOn D H).X q)
    (j_equiv : ∀ q (γ : H) (x : bdry.X q), j q (γ • x) = γ • j q x)
    (j_bd : ∀ q (x : bdry.X (q + 1)), (toGChainCxOn D H).bd q (j (q + 1) x) = j q (bdry.bd q x))
    (hneg : ∀ q, q < 0 → Subsingleton (bdry.X q)) (hdim : ∀ q, 2 ≤ q → Subsingleton (bdry.X q))
    (hfree : ∀ q, ∃ (I : Type) (rep : I → bdry.X q), ∀ b (val : I → (toGChainCxOn D H).X b),
      ∃! F : bdry.X q →+ (toGChainCxOn D H).X b, GL2.Cone.T1 H g F ∧ ∀ i, F (rep i) = val i) :
    GL2.Cone.PairHyp H ⟨toGChainCxOn D H, bdry, j, j_equiv, j_bd⟩ g where
  neg q hq := ⟨subsingleton_neg D q hq, hneg q hq⟩
  contract0 := hA.contract0
  contract1 := hA.contract1
  contract2 := hA.contract2
  bulk_dim q hq := subsingleton_dim D q hq
  bdry_dim := hdim
  free_twisted := hfree

end XBulk

end GL2.Model

end
