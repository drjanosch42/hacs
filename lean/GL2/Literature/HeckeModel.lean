import GL2.M4.Data

/-!
# L1: the Hecke-ring relations on the model, transported to `H^n` (slot 20, L20-2, branch H)

[GT] Part L rows `lit:hecke-Tp-prim`, `lit:hecke-coprime`, `lit:hecke-commute` (L20-0 §3.2). Each structure states a
**transported** identity: an identity of Shimura's abstract Hecke ring `R(Γ, Δ)` (Shimura, *Introduction to the arithmetic
theory of automorphic functions*, Ch. 3), moved to operators on `H^n` by the multiplicativity of the action of `R(Γ, Δ)` on
cohomology (Ash–Stevens, Crelle 365 (1986), §1; Rhie–Whaples). Neither the abstract ring nor the action exists in mathlib
`db584cd`, so the identities are stated **directly on the model's operators** (`GL2.M4.Tdc`, `Tprim`, `TdcOf`), in the
project's notation (`Tprim` = the primitive `T_{Δ^prim_{p^μ}}`, `c` = the scalar of `ρ(pI)`). Shimura's symbols `T(a, d)`,
`T(m)` are deliberately not used (the notation question of [M4] §5.1 was settled by R20-L).

**Locators: verified in R20-L (2026-10-05, the owner's Shimura Ch. 3 scan).** `hTp`: Shimura Thm 3.24(5), p. 63, with
Prop 3.17, p. 58; `hcop`: Prop 3.16, p. 57 (Thm 3.24(3) is the full-operator form); `hcomm`: Prop 3.8, p. 54, as applied in
§3.2, p. 56. The transport is Ash–Stevens §1.1, p. 194. Shimura's notation `T(1, m)` (primitive), `T(m)` (full) is confirmed
(Thm 3.24(1), (2)). No quotation is given.

**Guards** (the transport's standing hypotheses): `6 ∈ R^×` (the operators are those of [C] `thm:welldef`(b)); `z = p·I`
with `ρ(z) = c • 1` (the action of `T(p, p)`); `M, N > 0` coprime. Interface data: none, the operators are the model's.
-/

noncomputable section

namespace GL2.Literature

open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets GL2.M4

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]

/-- **`lit:hecke-Tp-prim`** (`hTp`; verified in R20-L: Shimura Thm 3.24(5), p. 63, with Prop 3.17, p. 58;
transported by Ash–Stevens §1.1, p. 194). The product `T_{Δ^prim_p} T_{Δ^prim_{p^{μ+1}}}` by double cosets, on `H^n`:
`Tprim 1 · Tprim 1 = Tprim 2 + (p + 1) c` (`μ = 0`) and `Tprim 1 · Tprim (μ+1) = Tprim (μ+2) + p c Tprim μ` (`μ ≥ 1`),
for every family of `τ`s. -/
structure HeckeTpPrim (ρ : ΔQ →* Module.End R V) : Prop where
  sq : ∀ (p : ℕ) (hp : p.Prime) (z : GLQ) (hz : z ∈ ΔQ) (c : R), IsScalarQ z p → ρ ⟨z, hz⟩ = c • 1 →
    IsUnit (6 : R) → ∀ (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (n : ℕ),
      Tprim ρ p hp τ n 1 * Tprim ρ p hp τ n 1 = Tprim ρ p hp τ n 2 + ((p : R) + 1) • c • (1 : Module.End R (HM ρ n))
  succ : ∀ (p : ℕ) (hp : p.Prime) (z : GLQ) (hz : z ∈ ΔQ) (c : R), IsScalarQ z p → ρ ⟨z, hz⟩ = c • 1 →
    IsUnit (6 : R) → ∀ (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (n μ : ℕ), 1 ≤ μ →
      Tprim ρ p hp τ n 1 * Tprim ρ p hp τ n (μ + 1) = Tprim ρ p hp τ n (μ + 2) + ((p : R) * c) • Tprim ρ p hp τ n μ

/-- `diag(1, N)` is integral (F3-20 MINOR 10(c): the membership binders of `HeckeCoprime.mul` are discharged by this). -/
lemma gQ_mem_ΔQ (N : ℤ) (hN : N ≠ 0) : gQ N hN ∈ ΔQ := ⟨gMat N, coe_gQ N hN⟩

/-- **`lit:hecke-coprime`** (`hcop`; verified in R20-L: Shimura Prop 3.16, p. 57; transported by
Ash–Stevens §1.1, p. 194). For coprime `M, N > 0`: `T_{Γ diag(1,M) Γ} T_{Γ diag(1,N) Γ} = T_{Γ diag(1,MN) Γ}` on `H^n`, stated on the
choice-free operators `TdcOf` (one coset datum and one `τ` each). -/
structure HeckeCoprime (ρ : ΔQ →* Module.End R V) : Prop where
  mul : ∀ (h6 : IsUnit (6 : R)) (M N : ℤ) (hM : M ≠ 0) (hN : N ≠ 0), 0 < M → 0 < N → IsCoprime M N →
    ∀ (n : ℕ),
      TdcOf ρ h6 (gQ M hM) (gQ_mem_ΔQ M hM) n * TdcOf ρ h6 (gQ N hN) (gQ_mem_ΔQ N hN) n =
        TdcOf ρ h6 (gQ (M * N) (mul_ne_zero hM hN)) (gQ_mem_ΔQ (M * N) (mul_ne_zero hM hN)) n

/-- **`lit:hecke-commute`** (`hcomm`; verified in R20-L: Shimura Prop 3.8, p. 54, as applied in §3.2, p. 56;
transported by Ash–Stevens §1.1, p. 194). Any two double-coset operators commute on `H^n`, stated on the choice-free
operators `TdcOf`. This is [C] `obl:algebra`'s "whence the `{T_N}` commute" (the image of the commutative `𝓗(Δ, Γ)`), of
which `thm:relations`(1)'s first clause is an instance. `ΔQ` includes `det < 0`; Prop 3.8 applies since every double coset
has a symmetric (diagonal) representative (F3-20 MINOR 4). -/
structure HeckeCommute (ρ : ΔQ →* Module.End R V) : Prop where
  comm : ∀ (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (g' : GLQ) (hg' : g' ∈ ΔQ) (n : ℕ),
    Commute (TdcOf ρ h6 g hg n) (TdcOf ρ h6 g' hg' n)

end GL2.Literature

end
