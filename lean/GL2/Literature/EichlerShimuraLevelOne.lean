import GL2.Literature.Basic

/-!
# L1: Eichler–Shimura at level one, as used by [C] `sec:saturation` (L16-6)

[C] `lem:heckerankone` and `prop:cohforms`(3) rest on the `\STD` inputs named in their proofs:
Eichler–Shimura with its Hecke equivariance (Shimura, *Introduction to the arithmetic theory of automorphic functions*,
Ch. 8; [C] `thm:classical`, `cl:dimH1`(1), `obl:oracle`) and the basis of normalised eigenforms of `M_k` with pairwise
distinct systems (Serre, *Cours d'arithmétique*, Ch. VII, §5).  **Locator status (L17-5 repairs, 2026-10-02):
every source is verified in the library**, so neither structure is `UNVERIFIED` any longer; both remain *derived
consequences* (no single quotation states them), and deriving them from the quoted structures is an L2 proof.
* **Eichler-Shimura with Hecke equivariance:** Diamond-Im, *Modular forms and modular curves*, Theorem 12.2.2
  (printed p. 108) and Proposition 12.4.10 (printed p. 117), quoted and guarded in
  `GL2.Literature.DiamondImEichlerShimura` (`Γ = Γ₀(1) = SL₂(ℤ)`, `k = n + 2 ≥ 2`: `M_k ⊕ S̄_k ≅ H¹(Γ, V_k)`).
* **Eigenbasis on cusp forms and multiplicity one:** Diamond-Im §3.5-§3.6 (printed pp. 54-55),
  `DiamondImEigenspaces`; multiplicity one also Serre, *A Course in Arithmetic* (GTM 7), §5.4, Thm 7 and Cor. 1
  (printed p. 102), `SerreCoursThm7`.
* **The Eisenstein system, once:** `M_k = M_k^0 ⊕ C.G_k (for k ≥ 2)` (Serre §3.2, printed p. 88) and Serre §5.5,
  Prop. 13 (printed p. 104, `SerreCoursProp13`: `G_k` an eigenform with eigenvalues `σ_{k-1}(n)`).
Shimura's book (Ch. 3, Ch. 8) is not in the library and is **no longer needed** for these two structures.
These two structures state exactly the *consequence* [C]'s proofs draw from those sources, over **interface data**:

* `L'` stands for `𝕃' = H¹(SL₂(ℤ); Sym^n ⊗ ℤ_ℓ)/torsion`, `TL` for its Hecke algebra `T^𝕃 ⊆ End(𝕃')`;
* `Mk` for `M_k(ℤ_ℓ)` (`k = n + 2`), `Tf` for `T^f ⊆ End(M_k(ℤ_ℓ))`;
* `E` for `\bar ℚ_ℓ ≅ ℂ`; a *system* is an `ℤ_ℓ`-algebra map `χ : TL → E`.  **`E ⊇ ℚ_ℓ`** (characteristic `0`):
  the consumers (`GL2.Sat.heckerankone`, `GL2.Sat.cohforms_iso`) carry `[Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]`
  (F3 audit finding 1: over a field of characteristic `ℓ` the consequence drawn is false).

The L2 layer must supply the real objects and prove the identification.
-/

namespace GL2.Literature

open TensorProduct

variable {p : ℕ} [Fact p.Prime]

/-- The joint eigenspace of a family `X : ι → End(L')` on `E ⊗ L'` for the system `χ : ι → E`. -/
noncomputable def jointEigen {ι : Type*} {L' : Type*} [AddCommGroup L'] [Module ℤ_[p] L']
    (X : ι → Module.End ℤ_[p] L') (E : Type*) [Field E] [Algebra ℤ_[p] E] (χ : ι → E) :
    Submodule E (E ⊗[ℤ_[p]] L') :=
  ⨅ i, LinearMap.ker (LinearMap.baseChange E (X i) - χ i • LinearMap.id)

open Classical in
/-- **Eichler–Shimura at level one, Hecke-module form** (sources `VERIFIED`, a derived consequence: Diamond-Im
Thm 12.2.2 + Prop. 12.4.10 (`DiamondImEichlerShimura`), Diamond-Im §3.5-§3.6 (`DiamondImEigenspaces`), Serre GTM 7
§3.2, §5.4 Thm 7 / Cor. 1, §5.5 Prop. 13 (`SerreCoursThm7`, `SerreCoursProp13`)).  [C] `lem:heckerankone`'s proof: `𝕃' ⊗ ℂ ≅ M_k(ℂ) ⊕ \bar S_k(ℂ)`
as Hecke modules, and `M_k` has a basis of normalised eigenforms with pairwise distinct systems, the Eisenstein one
once: so the Hecke operators `X i` (`i` running over the primes) act **semisimply** on `E ⊗ L'`, with finitely many
systems `S`, each of multiplicity `2` (cuspidal) except at most one of multiplicity `1` (Eisenstein).  Interface data:
`L'`, `X`, `E` (see the module docstring).  No decidable finite instance exists, so no sanity instance is given. -/
structure EichlerShimuraLevelOne {ι : Type*} (L' : Type*) [AddCommGroup L'] [Module ℤ_[p] L']
    (X : ι → Module.End ℤ_[p] L') (E : Type*) [Field E] [Algebra ℤ_[p] E] : Prop where
  decomp : ∃ (S : Finset (ι → E)) (mult : (ι → E) → ℕ),
    (∀ χ ∈ S, mult χ = 1 ∨ mult χ = 2) ∧ (S.filter fun χ => mult χ = 1).card ≤ 1 ∧
    (∀ χ ∈ S, Module.finrank E (jointEigen X E χ) = mult χ) ∧
    (⨆ χ ∈ S, jointEigen X E χ) = ⊤

/-- **The forms side carries the same systems, each once** (sources `VERIFIED`, a derived consequence of the same
quoted structures as `EichlerShimuraLevelOne`;
[C] `prop:cohforms`(3)'s proof).  The classical `T_p` (`XM i`) act semisimply on `E ⊗ M_k` with multiplicity one, and
a system occurs on `E ⊗ 𝕃'` iff it occurs on `E ⊗ M_k`. -/
structure FormsSameSystems {ι : Type*} (L' Mk : Type*) [AddCommGroup L'] [Module ℤ_[p] L'] [AddCommGroup Mk]
    [Module ℤ_[p] Mk] (XL : ι → Module.End ℤ_[p] L') (XM : ι → Module.End ℤ_[p] Mk) (E : Type*) [Field E]
    [Algebra ℤ_[p] E] : Prop where
  semisimple_forms : ∃ S : Finset (ι → E), (∀ χ ∈ S, Module.finrank E (jointEigen XM E χ) = 1) ∧
    (⨆ χ ∈ S, jointEigen XM E χ) = ⊤
  same_systems : ∀ χ : ι → E, jointEigen XL E χ ≠ ⊥ ↔ jointEigen XM E χ ≠ ⊥

end GL2.Literature
