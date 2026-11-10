// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ===========================================================================
// CoreMath.cpp -- console phase C1a: the arithmetic types of plan Sec. 3.4 and
// the linear-algebra surface of plan Sec. 3.5 that needs no new substrate.
//
// Everything here is F_ell-linear algebra over FlatMatrixT<uint32_t> /
// FpPolyT<uint32_t>, using the tree's own operations -- `Hecke/SubspaceOps.h`
// (SolveColumnsExact, HStack, KernelAsMatrix, algebra_closure, RrefRowSpan,
// VecOfMatrix / MatrixOfVec, SplitMix64) and `Hecke/FpPoly.h`
// (CharPolyBerkowitz, MinPolyKrylov, EvalPolyAtMatrix, FactorPoly).  Nothing is
// re-implemented: [I Sec. 4.3]'s recommendation ("do not introduce an F_{ell^r}
// Elt; implement the extension-field needs inside the small algebra layer as
// F_ell-linear algebra") is what the whole surface follows.
//
// NOT bound in C1a, each with its reason (WP-7 report Sec. "deviations"):
//   bernoulli_mod                  plan Sec. 3.5 lists it, but it is phase E6's
//                                  ([S Stage 6.4]); slot-3 WP-7 scope says
//                                  explicitly "not bound in C1a".
//   nilpotent_part / radical /     the local-algebra invariants, which plan
//   tangent_space / socle /        Sec. 3.5 defers to C5 "when [I G5] exists".
//   nilpotency_index / loewy /
//   socle_series
//   check_identities               wraps [I G2], which does not exist yet.
// ===========================================================================

#include "Bindings.h"

#include "../../Hecke/FpPoly.h"
#include "../../Hecke/SubspaceOps.h"
#include "../../Hecke/HeckeLinAlg.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace console {
namespace bind {

namespace {

using Flat = FlatMatrixT<uint32_t>;
using Poly = hecke::FpPolyT<uint32_t>;

// --- small conversions -----------------------------------------------------

MatrixT<uint32_t> ToMatrix(const Flat& m) { return m.to_matrix(); }
Flat FromMatrix(const MatrixT<uint32_t>& m) { return Flat(m); }

// The columns of a list of one-column matrices, side by side.
Flat HStackFlat(const std::vector<Flat>& parts, uint32_t rows, uint32_t ell,
                const SourcePos& pos, const char* who) {
    uint32_t cols = 0;
    for (const Flat& f : parts) {
        if (f.get_rows() != rows) {
            throw TypeError(std::string(who) + ": the parts do not agree in rows (" +
                                std::to_string(rows) + " vs " + std::to_string(f.get_rows()) + ")",
                            pos);
        }
        cols += f.get_cols();
    }
    Flat out(rows, cols, ell);
    uint32_t off = 0;
    for (const Flat& f : parts) {
        for (uint32_t c = 0; c < f.get_cols(); ++c) {
            for (uint32_t r = 0; r < rows; ++r) out.set_entry(r, off + c, f.get_entry(r, c));
        }
        off += f.get_cols();
    }
    return out;
}

Flat BasisAsMatrix(const std::vector<Flat>& basis, uint32_t rows, uint32_t ell) {
    // FlatMatrix::kernel() / image() hand back one column vector per basis
    // element; the console's Matrix type is a single matrix, so they are laid
    // out side by side, which is the convention every consumer below uses.
    if (basis.empty()) return Flat(rows, 0, ell);
    Flat out(rows, static_cast<uint32_t>(basis.size()), ell);
    for (std::size_t j = 0; j < basis.size(); ++j) {
        for (uint32_t r = 0; r < rows; ++r)
            out.set_entry(r, static_cast<uint32_t>(j), basis[j].get_entry(r, 0));
    }
    return out;
}

// --- printers --------------------------------------------------------------

std::string MatrixText(const Flat& m, std::int64_t max_entries, bool full) {
    const std::int64_t entries = static_cast<std::int64_t>(m.get_rows()) *
                                 static_cast<std::int64_t>(m.get_cols());
    if (!full && entries > max_entries) {
        std::ostringstream os;
        os << "Matrix " << m.get_rows() << " x " << m.get_cols() << " over F_" << m.get_p()
           << " (hash 0x" << std::hex << HashMatrix(m) << std::dec << ")";
        return os.str();
    }
    std::ostringstream os;
    os << "[";
    for (uint32_t r = 0; r < m.get_rows(); ++r) {
        if (r != 0) os << " ";
        os << "[";
        for (uint32_t c = 0; c < m.get_cols(); ++c) {
            if (c != 0) os << " ";
            os << m.get_entry(r, c);
        }
        os << "]";
    }
    os << "]";
    return os.str();
}

std::string PolyText(const Poly& f) {
    // (C1b) hecke::FpPolyT::to_string() ALREADY ends in " (mod ell)", so the
    // C1a form appended the modulus twice ("... (mod 691) mod 691").  Fixed
    // here rather than in the printer because this is the only producer; it
    // changes printed text only -- no computed value, no golden object, and no
    // C1a assertion reads this string (C1a ships no .expected at all).
    return f.to_string();
}

// --- Jordan type -----------------------------------------------------------
// The partition of a NILPOTENT operator from the nullity sequence:
// #{parts >= i} = nullity(N^i) - nullity(N^{i-1}), and the partition is its
// conjugate -- hecke::PartitionFromNullities (Hecke/HeckeLinAlg.h, gap G14).
std::vector<uint32_t> NilpotentJordanType(const Flat& N, const SourcePos& pos, const char* who) {
    if (N.get_rows() != N.get_cols()) {
        throw TypeError(std::string(who) + ": expected a square matrix", pos);
    }
    const uint32_t d = N.get_rows();
    if (d == 0) return {};
    std::vector<uint32_t> nullity;      // nullity(N^i), i = 0 .. r
    Flat P(d, N.get_p());               // N^0 = I
    nullity.push_back(0);
    for (uint32_t i = 1; i <= d; ++i) {
        P = P * N;
        const uint32_t n = d - P.rank();
        nullity.push_back(n);
        if (n == d) break;
    }
    if (nullity.back() != d) {
        throw RuntimeError(std::string(who) +
                               ": the matrix is not nilpotent, so it has no Jordan type over "
                               "F_ell without first splitting its characteristic polynomial "
                               "(the general pi-primary form is phase C1b's `decompose`)",
                           pos);
    }
    // (slot-7 WP-8, gap G14) The conjugate-partition arithmetic is the
    // decomposition's own, now public: hecke::PartitionFromNullities with
    // deg(pi) = 1.  It replaced a console-side copy of the same arithmetic.
    return hecke::PartitionFromNullities(nullity, 1u, who);
}

// ===========================================================================
// Verbs
// ===========================================================================

// --- Fp --------------------------------------------------------------------

Value VerbFp(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "Fp(p, a)");
    if (p < 2) throw TypeError("Fp(p, a): the modulus must be at least 2", ctx.pos);
    const std::int64_t x = a[1].is_int() ? a[1].int_value() : 0;
    if (!a[1].is_int()) throw TypeError("Fp(p, a): a must be Int", ctx.pos);
    // Canonical residue in [0, p), negatives included.
    const std::int64_t m = x % static_cast<std::int64_t>(p);
    return FpValue(ctx, p, static_cast<uint32_t>(m < 0 ? m + static_cast<std::int64_t>(p) : m));
}

// --- Poly ------------------------------------------------------------------

Value VerbPoly(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "poly(p, coeffs)");
    if (p < 2) throw TypeError("poly(p, coeffs): the modulus must be at least 2", ctx.pos);
    std::vector<uint32_t> cs;
    for (const Value& v : a[1].list_value()) {
        if (!v.is_int()) throw TypeError("poly(p, coeffs): every coefficient must be Int", ctx.pos);
        std::int64_t m = v.int_value() % static_cast<std::int64_t>(p);
        if (m < 0) m += static_cast<std::int64_t>(p);
        cs.push_back(static_cast<uint32_t>(m));
    }
    return PolyValue(ctx, Poly(p, std::move(cs)));
}

Value VerbX(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "X(p)");
    if (p < 2) throw TypeError("X(p): the modulus must be at least 2", ctx.pos);
    return PolyValue(ctx, Poly::X(p));
}

Value VerbCoeffs(CallContext& ctx, const std::vector<Value>& a) {
    // LOW -> HIGH, the machine order of the report JSON (plan Sec. 2 C5).
    const PolyVal& f = AsPoly(a[0], ctx.pos, "coeffs(f)");
    return IntList(f.f.coeffs);
}

Value VerbDegree(CallContext& ctx, const std::vector<Value>& a) {
    const PolyVal& f = AsPoly(a[0], ctx.pos, "degree(f)");
    return Value::from_int(f.f.degree());
}

Value VerbFactor(CallContext& ctx, const std::vector<Value>& a) {
    const PolyVal& f = AsPoly(a[0], ctx.pos, "factor(f)");
    if (f.f.is_zero()) throw RuntimeError("factor(f): the zero polynomial has no factorisation", ctx.pos);
    // FactorPoly ALWAYS passes its result through CertifyFactorization before
    // returning it (FpPoly.h:170), so a console factorisation is certified by
    // construction -- never a randomised answer taken on trust.
    const hecke::FpPolyFactorizationT<uint32_t> fac = hecke::FactorPoly(f.f);
    Record out;
    out.set("ell", Value::from_int(static_cast<std::int64_t>(fac.ell)));
    out.set("leading_coeff", Value::from_int(static_cast<std::int64_t>(fac.leading_coeff)));
    std::vector<Value> items;
    for (const auto& fr : fac.factors) {
        Record r;
        r.set("factor", PolyValue(ctx, fr.pi));
        r.set("multiplicity", Value::from_int(static_cast<std::int64_t>(fr.multiplicity)));
        r.set("degree", Value::from_int(fr.pi.degree()));
        items.push_back(Value::from_record(std::move(r)));
    }
    out.set("factors", Value::from_list(std::move(items)));
    out.set("certified", Value::from_bool(true));
    return Value::from_record(std::move(out));
}

Value VerbGcd(CallContext& ctx, const std::vector<Value>& a) {
    const PolyVal& f = AsPoly(a[0], ctx.pos, "gcd(f, g)");
    const PolyVal& g = AsPoly(a[1], ctx.pos, "gcd(f, g)");
    if (f.f.ell != g.f.ell) throw TypeError("gcd(f, g): different moduli", ctx.pos);
    return PolyValue(ctx, Poly::gcd(f.f, g.f));
}

Value VerbPowmod(CallContext& ctx, const std::vector<Value>& a) {
    const PolyVal& g = AsPoly(a[0], ctx.pos, "powmod(g, e, f)");
    const uint64_t e = U64Arg(a[1], ctx.pos, "powmod(g, e, f)");
    const PolyVal& f = AsPoly(a[2], ctx.pos, "powmod(g, e, f)");
    return PolyValue(ctx, Poly::powmod(g.f, e, f.f));
}

Value VerbEval(CallContext& ctx, const std::vector<Value>& a) {
    const PolyVal& f = AsPoly(a[0], ctx.pos, "eval(f, x)");
    const uint32_t x = U32Arg(a[1], ctx.pos, "eval(f, x)");
    return Value::from_int(static_cast<std::int64_t>(f.f.eval(x % f.f.ell)));
}

Value VerbCharPoly(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "charpoly(A)");
    if (A.m.get_rows() != A.m.get_cols()) {
        throw TypeError("charpoly(A): expected a square matrix", ctx.pos);
    }
    // det of the 0x0 matrix is 1 -- the golden harness's own convention
    // (GoldenHarness.cpp:363-366), kept so that `hash(charpoly(T.A(q)))` is
    // comparable with `golden(point, "charpoly_A(T_p)^q")` at an empty degree.
    if (A.m.get_rows() == 0) {
        return PolyValue(ctx, Poly(A.m.get_p(), std::vector<uint32_t>{1u}));
    }
    return PolyValue(ctx, hecke::CharPolyBerkowitz(ToMatrix(A.m)));
}

Value VerbTrace(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "trace(A)");
    const uint32_t p = A.m.get_p();
    uint64_t t = 0;
    for (uint32_t i = 0; i < A.m.get_rows() && i < A.m.get_cols(); ++i)
        t = (t + A.m.get_entry(i, i)) % p;
    return Value::from_int(static_cast<std::int64_t>(t));
}

Value VerbMinPoly(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "minpoly(A)");
    if (A.m.get_rows() != A.m.get_cols()) {
        throw TypeError("minpoly(A): expected a square matrix", ctx.pos);
    }
    return PolyValue(ctx, hecke::MinPolyKrylov(ToMatrix(A.m)));
}

Value VerbEvalAt(CallContext& ctx, const std::vector<Value>& a) {
    const PolyVal& f = AsPoly(a[0], ctx.pos, "eval_at(f, A)");
    const MatrixVal& A = AsMatrix(a[1], ctx.pos, "eval_at(f, A)");
    if (A.m.get_rows() != A.m.get_cols()) {
        throw TypeError("eval_at(f, A): expected a square matrix", ctx.pos);
    }
    return MatrixValue(ctx, FromMatrix(hecke::EvalPolyAtMatrix(f.f, ToMatrix(A.m))));
}

// --- Matrix construction ---------------------------------------------------

Value VerbMatrix(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "matrix(p, rows)");
    if (p < 2) throw TypeError("matrix(p, rows): the modulus must be at least 2", ctx.pos);
    const std::vector<Value>& rows = a[1].list_value();
    uint32_t cols = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (!rows[i].is_list()) {
            throw TypeError("matrix(p, rows): every row must be a List of Int", ctx.pos);
        }
        const std::size_t n = rows[i].list_value().size();
        if (i == 0) cols = static_cast<uint32_t>(n);
        else if (n != cols) {
            throw TypeError("matrix(p, rows): the rows do not all have the same length", ctx.pos);
        }
    }
    Flat m(static_cast<uint32_t>(rows.size()), cols, p);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const std::vector<Value>& row = rows[r].list_value();
        for (std::size_t c = 0; c < row.size(); ++c) {
            if (!row[c].is_int()) {
                throw TypeError("matrix(p, rows): every entry must be Int", ctx.pos);
            }
            std::int64_t v = row[c].int_value() % static_cast<std::int64_t>(p);
            if (v < 0) v += static_cast<std::int64_t>(p);
            m.set_entry(static_cast<uint32_t>(r), static_cast<uint32_t>(c),
                        static_cast<uint32_t>(v));
        }
    }
    return MatrixValue(ctx, std::move(m));
}

Value VerbIdentity(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "identity(p, n)");
    const uint32_t n = U32Arg(a[1], ctx.pos, "identity(p, n)");
    if (p < 2) throw TypeError("identity(p, n): the modulus must be at least 2", ctx.pos);
    return MatrixValue(ctx, Flat(n, p));
}

Value VerbZero(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "zero(p, rows, cols)");
    const uint32_t r = U32Arg(a[1], ctx.pos, "zero(p, rows, cols)");
    const uint32_t c = U32Arg(a[2], ctx.pos, "zero(p, rows, cols)");
    if (p < 2) throw TypeError("zero(p, rows, cols): the modulus must be at least 2", ctx.pos);
    return MatrixValue(ctx, Flat(r, c, p));
}

// --- Matrix arithmetic -----------------------------------------------------

Value VerbMul(CallContext& ctx, const std::vector<Value>& a) {
    if (a[0].is_object() && a[0].object_value() && a[0].object_value()->type() &&
        a[0].object_value()->type()->name == kTypePoly) {
        const PolyVal& f = AsPoly(a[0], ctx.pos, "mul(f, g)");
        const PolyVal& g = AsPoly(a[1], ctx.pos, "mul(f, g)");
        return PolyValue(ctx, f.f * g.f);
    }
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "mul(A, B)");
    const MatrixVal& B = AsMatrix(a[1], ctx.pos, "mul(A, B)");
    RequireSameModulus(A.m, B.m, ctx.pos, "mul(A, B)");
    if (A.m.get_cols() != B.m.get_rows()) {
        throw TypeError("mul(A, B): shapes do not compose (" + std::to_string(A.m.get_rows()) +
                            "x" + std::to_string(A.m.get_cols()) + " * " +
                            std::to_string(B.m.get_rows()) + "x" + std::to_string(B.m.get_cols()) +
                            ")",
                        ctx.pos);
    }
    return MatrixValue(ctx, A.m * B.m);
}

Value VerbAdd(CallContext& ctx, const std::vector<Value>& a) {
    if (a[0].is_object() && a[0].object_value() && a[0].object_value()->type() &&
        a[0].object_value()->type()->name == kTypePoly) {
        const PolyVal& f = AsPoly(a[0], ctx.pos, "add(f, g)");
        const PolyVal& g = AsPoly(a[1], ctx.pos, "add(f, g)");
        return PolyValue(ctx, f.f + g.f);
    }
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "add(A, B)");
    const MatrixVal& B = AsMatrix(a[1], ctx.pos, "add(A, B)");
    RequireSameModulus(A.m, B.m, ctx.pos, "add(A, B)");
    if (A.m.get_rows() != B.m.get_rows() || A.m.get_cols() != B.m.get_cols()) {
        throw TypeError("add(A, B): shapes differ", ctx.pos);
    }
    return MatrixValue(ctx, A.m + B.m);
}

Value VerbSub(CallContext& ctx, const std::vector<Value>& a) {
    if (a[0].is_object() && a[0].object_value() && a[0].object_value()->type() &&
        a[0].object_value()->type()->name == kTypePoly) {
        const PolyVal& f = AsPoly(a[0], ctx.pos, "sub(f, g)");
        const PolyVal& g = AsPoly(a[1], ctx.pos, "sub(f, g)");
        return PolyValue(ctx, f.f - g.f);
    }
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "sub(A, B)");
    const MatrixVal& B = AsMatrix(a[1], ctx.pos, "sub(A, B)");
    RequireSameModulus(A.m, B.m, ctx.pos, "sub(A, B)");
    if (A.m.get_rows() != B.m.get_rows() || A.m.get_cols() != B.m.get_cols()) {
        throw TypeError("sub(A, B): shapes differ", ctx.pos);
    }
    return MatrixValue(ctx, A.m - B.m);
}

Value VerbScale(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "scale(A, c)");
    std::int64_t c = 0;
    if (!a[1].is_int()) throw TypeError("scale(A, c): c must be Int", ctx.pos);
    c = a[1].int_value() % static_cast<std::int64_t>(A.m.get_p());
    if (c < 0) c += static_cast<std::int64_t>(A.m.get_p());
    return MatrixValue(ctx, A.m * static_cast<uint32_t>(c));
}

// --- Matrix linear algebra --------------------------------------------------

Value VerbRank(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "rank(A)");
    if (A.m.get_rows() == 0 || A.m.get_cols() == 0) return Value::from_int(0);
    return Value::from_int(static_cast<std::int64_t>(A.m.rank()));
}

Value VerbNullity(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "nullity(A)");
    const uint32_t r = (A.m.get_rows() == 0 || A.m.get_cols() == 0) ? 0u : A.m.rank();
    return Value::from_int(static_cast<std::int64_t>(A.m.get_cols() - r));
}

Value VerbRref(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "rref(A)");
    return MatrixValue(ctx, A.m.get_row_echelon_form_as_flat_matrix());
}

Value VerbKernel(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "kernel(A)");
    return MatrixValue(ctx, BasisAsMatrix(A.m.kernel(), A.m.get_cols(), A.m.get_p()));
}

Value VerbImage(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "image(A)");
    return MatrixValue(ctx, BasisAsMatrix(A.m.image(), A.m.get_rows(), A.m.get_p()));
}

Value VerbInverse(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "inverse(A)");
    if (A.m.get_rows() != A.m.get_cols()) {
        throw TypeError("inverse(A): expected a square matrix", ctx.pos);
    }
    if (!A.m.is_invertible()) throw RuntimeError("inverse(A): the matrix is singular", ctx.pos);
    return MatrixValue(ctx, A.m.inverse());
}

Value VerbTranspose(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "transpose(A)");
    return MatrixValue(ctx, A.m.transpose());
}

Value VerbPow(CallContext& ctx, const std::vector<Value>& a) {
    // slot-21 verbs package: A^e by square-and-multiply over A.m * B.m (e >= 0; A^0 = I).
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "pow(A, e)");
    if (A.m.get_rows() != A.m.get_cols()) throw TypeError("pow(A, e): expected a square matrix", ctx.pos);
    if (!a[1].is_int() || a[1].int_value() < 0) throw TypeError("pow(A, e): e must be a non-negative Int", ctx.pos);
    std::uint64_t e = static_cast<std::uint64_t>(a[1].int_value());
    const uint32_t n = A.m.get_rows(), p = A.m.get_p();
    Flat result(n, p);   // the identity
    Flat base = A.m;
    while (e > 0) {
        if (e & 1u) result = result * base;
        e >>= 1;
        if (e > 0) base = base * base;
    }
    return MatrixValue(ctx, result);
}

Value VerbSolve(CallContext& ctx, const std::vector<Value>& a) {
    // Exact X with A*X = B, through the pipeline's own invariance certificate
    // (SolveColumnsExact, HeckeModuleDecomposition.cpp:68-94 -- gap G14 closed
    // by SubspaceOps.h).  It requires A to have full COLUMN rank, which is
    // exactly the case in which X is unique; otherwise the error names it.
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "solve(A, B)");
    const MatrixVal& B = AsMatrix(a[1], ctx.pos, "solve(A, B)");
    RequireSameModulus(A.m, B.m, ctx.pos, "solve(A, B)");
    if (A.m.get_rows() != B.m.get_rows()) {
        throw TypeError("solve(A, B): A and B must have the same number of rows", ctx.pos);
    }
    if (A.m.get_cols() != 0 && A.m.rank() != A.m.get_cols()) {
        throw RuntimeError("solve(A, B): A does not have full column rank, so the solution is not "
                           "unique; restrict to an independent column set first",
                           ctx.pos);
    }
    return MatrixValue(ctx, FromMatrix(hecke::SolveColumnsExact(ToMatrix(A.m), ToMatrix(B.m),
                                                               "console solve(A, B)")));
}

Value VerbHStack(CallContext& ctx, const std::vector<Value>& a) {
    const std::vector<Value>& items = a[0].list_value();
    if (items.empty()) throw TypeError("hstack(list): the list is empty", ctx.pos);
    std::vector<Flat> parts;
    for (const Value& v : items) parts.push_back(AsMatrix(v, ctx.pos, "hstack(list)").m);
    return MatrixValue(ctx, HStackFlat(parts, parts[0].get_rows(), parts[0].get_p(), ctx.pos,
                                       "hstack(list)"));
}

Value VerbVStack(CallContext& ctx, const std::vector<Value>& a) {
    const std::vector<Value>& items = a[0].list_value();
    if (items.empty()) throw TypeError("vstack(list): the list is empty", ctx.pos);
    std::vector<Flat> parts;
    for (const Value& v : items) parts.push_back(AsMatrix(v, ctx.pos, "vstack(list)").m);
    const uint32_t cols = parts[0].get_cols();
    const uint32_t ell = parts[0].get_p();
    uint32_t rows = 0;
    for (const Flat& f : parts) {
        if (f.get_cols() != cols) throw TypeError("vstack(list): the parts differ in columns", ctx.pos);
        rows += f.get_rows();
    }
    Flat out(rows, cols, ell);
    uint32_t off = 0;
    for (const Flat& f : parts) {
        for (uint32_t r = 0; r < f.get_rows(); ++r)
            for (uint32_t c = 0; c < cols; ++c) out.set_entry(off + r, c, f.get_entry(r, c));
        off += f.get_rows();
    }
    return MatrixValue(ctx, std::move(out));
}

Value VerbSubmatrix(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "submatrix(A, r0, r1, c0, c1)");
    const uint32_t r0 = U32Arg(a[1], ctx.pos, "submatrix");
    const uint32_t r1 = U32Arg(a[2], ctx.pos, "submatrix");
    const uint32_t c0 = U32Arg(a[3], ctx.pos, "submatrix");
    const uint32_t c1 = U32Arg(a[4], ctx.pos, "submatrix");
    if (r0 > r1 || r1 > A.m.get_rows() || c0 > c1 || c1 > A.m.get_cols()) {
        throw RuntimeError("submatrix(A, r0, r1, c0, c1): the half-open ranges are out of bounds",
                           ctx.pos);
    }
    Flat out(r1 - r0, c1 - c0, A.m.get_p());
    for (uint32_t r = r0; r < r1; ++r)
        for (uint32_t c = c0; c < c1; ++c) out.set_entry(r - r0, c - c0, A.m.get_entry(r, c));
    return MatrixValue(ctx, std::move(out));
}

Value VerbColumns(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "columns(A, idx)");
    const std::vector<Value>& idx = a[1].list_value();
    Flat out(A.m.get_rows(), static_cast<uint32_t>(idx.size()), A.m.get_p());
    for (std::size_t j = 0; j < idx.size(); ++j) {
        const uint32_t c = U32Arg(idx[j], ctx.pos, "columns(A, idx)");
        if (c >= A.m.get_cols()) {
            throw RuntimeError("columns(A, idx): column " + std::to_string(c) + " is out of range",
                               ctx.pos);
        }
        for (uint32_t r = 0; r < A.m.get_rows(); ++r)
            out.set_entry(r, static_cast<uint32_t>(j), A.m.get_entry(r, c));
    }
    return MatrixValue(ctx, std::move(out));
}

Value VerbVec(CallContext& ctx, const std::vector<Value>& a) {
    // End(V) -> F_ell^{d^2}, ROW-MAJOR: entry (i, j) at coordinate i*cols + j
    // -- SubspaceOps.h's stated convention, shared with algebra_closure.
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "vec(A)");
    Flat out(A.m.get_rows() * A.m.get_cols(), 1, A.m.get_p());
    for (uint32_t r = 0; r < A.m.get_rows(); ++r)
        for (uint32_t c = 0; c < A.m.get_cols(); ++c)
            out.set_entry(r * A.m.get_cols() + c, 0, A.m.get_entry(r, c));
    return MatrixValue(ctx, std::move(out));
}

Value VerbUnvec(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& v = AsMatrix(a[0], ctx.pos, "unvec(v, rows, cols)");
    const uint32_t rows = U32Arg(a[1], ctx.pos, "unvec(v, rows, cols)");
    const uint32_t cols = U32Arg(a[2], ctx.pos, "unvec(v, rows, cols)");
    if (v.m.get_cols() != 1 || v.m.get_rows() != rows * cols) {
        throw TypeError("unvec(v, rows, cols): v must be a single column of rows*cols entries",
                        ctx.pos);
    }
    Flat out(rows, cols, v.m.get_p());
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < cols; ++c) out.set_entry(r, c, v.m.get_entry(r * cols + c, 0));
    return MatrixValue(ctx, std::move(out));
}

// --- the subspace lattice ---------------------------------------------------

Value VerbSumSpaces(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& U = AsMatrix(a[0], ctx.pos, "sum_spaces(U, V)");
    const MatrixVal& V = AsMatrix(a[1], ctx.pos, "sum_spaces(U, V)");
    RequireSameModulus(U.m, V.m, ctx.pos, "sum_spaces(U, V)");
    if (U.m.get_rows() != V.m.get_rows()) {
        throw TypeError("sum_spaces(U, V): the two spaces live in different ambient dimensions",
                        ctx.pos);
    }
    const Flat both = HStackFlat({U.m, V.m}, U.m.get_rows(), U.m.get_p(), ctx.pos, "sum_spaces");
    return MatrixValue(ctx, BasisAsMatrix(both.image(), both.get_rows(), both.get_p()));
}

Value VerbIntersect(CallContext& ctx, const std::vector<Value>& a) {
    // ker[U | -V] gives the coefficient pairs (x, y) with U x = V y; the columns
    // of U x span U /\ V.  One kernel of one stacked system -- plan Sec. 3.5.
    const MatrixVal& U = AsMatrix(a[0], ctx.pos, "intersect(U, V)");
    const MatrixVal& V = AsMatrix(a[1], ctx.pos, "intersect(U, V)");
    RequireSameModulus(U.m, V.m, ctx.pos, "intersect(U, V)");
    if (U.m.get_rows() != V.m.get_rows()) {
        throw TypeError("intersect(U, V): the two spaces live in different ambient dimensions",
                        ctx.pos);
    }
    const uint32_t ell = U.m.get_p();
    const uint32_t d = U.m.get_rows();
    Flat M(d, U.m.get_cols() + V.m.get_cols(), ell);
    for (uint32_t r = 0; r < d; ++r) {
        for (uint32_t c = 0; c < U.m.get_cols(); ++c) M.set_entry(r, c, U.m.get_entry(r, c));
        for (uint32_t c = 0; c < V.m.get_cols(); ++c) {
            const uint32_t e = V.m.get_entry(r, c);
            M.set_entry(r, U.m.get_cols() + c, e == 0 ? 0u : ell - e);
        }
    }
    const std::vector<Flat> K = M.kernel();
    std::vector<Flat> cols;
    for (const Flat& k : K) {
        Flat x(U.m.get_cols(), 1, ell);
        for (uint32_t i = 0; i < U.m.get_cols(); ++i) x.set_entry(i, 0, k.get_entry(i, 0));
        cols.push_back(U.m * x);
    }
    const Flat spanned = BasisAsMatrix(cols, d, ell);
    // Reduce to a basis: the U x may be dependent (the kernel also sees ker U).
    return MatrixValue(ctx, BasisAsMatrix(spanned.image(), d, ell));
}

Value VerbContains(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& U = AsMatrix(a[0], ctx.pos, "contains(U, v)");
    const MatrixVal& v = AsMatrix(a[1], ctx.pos, "contains(U, v)");
    RequireSameModulus(U.m, v.m, ctx.pos, "contains(U, v)");
    if (U.m.get_rows() != v.m.get_rows()) {
        throw TypeError("contains(U, v): different ambient dimensions", ctx.pos);
    }
    const uint32_t r0 = (U.m.get_cols() == 0) ? 0u : U.m.rank();
    const Flat both = HStackFlat({U.m, v.m}, U.m.get_rows(), U.m.get_p(), ctx.pos, "contains");
    return Value::from_bool(both.rank() == r0);
}

Value VerbInSpan(CallContext& ctx, const std::vector<Value>& a) {
    // Membership of a MATRIX in the span of a list of matrices, over `vec`.
    const std::vector<Value>& items = a[0].list_value();
    const MatrixVal& A = AsMatrix(a[1], ctx.pos, "in_span(list, A)");
    hecke::RrefRowSpan span(A.m.get_rows() * A.m.get_cols(), A.m.get_p());
    for (const Value& v : items) {
        const MatrixVal& M = AsMatrix(v, ctx.pos, "in_span(list, A)");
        if (M.m.get_rows() != A.m.get_rows() || M.m.get_cols() != A.m.get_cols()) {
            throw TypeError("in_span(list, A): the shapes differ", ctx.pos);
        }
        span.insert(hecke::VecOfMatrix(ToMatrix(M.m)));
    }
    return Value::from_bool(span.contains(hecke::VecOfMatrix(ToMatrix(A.m))));
}

Value VerbRestrict(CallContext& ctx, const std::vector<Value>& a) {
    // The matrix of A on the invariant subspace spanned by `basis`, certified by
    // the exact solve (SolveColumnsExact throws when A*basis is not in the span,
    // i.e. when the subspace is NOT invariant) -- plan Sec. 3.5.
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "restrict(A, basis)");
    const MatrixVal& B = AsMatrix(a[1], ctx.pos, "restrict(A, basis)");
    RequireSameModulus(A.m, B.m, ctx.pos, "restrict(A, basis)");
    if (A.m.get_rows() != A.m.get_cols() || A.m.get_cols() != B.m.get_rows()) {
        throw TypeError("restrict(A, basis): A must be square and act on the basis' ambient space",
                        ctx.pos);
    }
    if (B.m.get_cols() != 0 && B.m.rank() != B.m.get_cols()) {
        throw RuntimeError("restrict(A, basis): the basis columns are dependent", ctx.pos);
    }
    const Flat AB = A.m * B.m;
    try {
        return MatrixValue(ctx, FromMatrix(hecke::SolveColumnsExact(ToMatrix(B.m), ToMatrix(AB),
                                                                    "console restrict(A, basis)")));
    } catch (const std::exception&) {
        throw RuntimeError("restrict(A, basis): the subspace is not invariant under A", ctx.pos);
    }
}

Value VerbAlgebra(CallContext& ctx, const std::vector<Value>& a) {
    // The unital F_ell-subalgebra generated by pairwise-commuting operators, as
    // the CANONICAL RREF coordinate basis of SubspaceOps' algebra_closure --
    // the same object HeckeSturmBound's <T_q> and HeckeLocalAlgebra's A_S use.
    const std::vector<Value>& items = a[0].list_value();
    if (items.empty()) throw TypeError("algebra(ops): the operator list is empty", ctx.pos);
    std::vector<MatrixT<uint32_t>> gens;
    uint32_t D = 0, ell = 0;
    for (const Value& v : items) {
        const MatrixVal& M = AsMatrix(v, ctx.pos, "algebra(ops)");
        if (M.m.get_rows() != M.m.get_cols()) {
            throw TypeError("algebra(ops): every operator must be square", ctx.pos);
        }
        if (gens.empty()) { D = M.m.get_rows(); ell = M.m.get_p(); }
        else if (M.m.get_rows() != D || M.m.get_p() != ell) {
            throw TypeError("algebra(ops): the operators differ in size or modulus", ctx.pos);
        }
        gens.push_back(ToMatrix(M.m));
    }
    uint32_t dim = 0;
    const MatrixT<uint32_t> coords = hecke::algebra_closure(gens, D, ell, dim);
    std::vector<Value> basis;
    for (uint32_t i = 0; i < coords.rows; ++i) {
        std::vector<uint32_t> row(coords.cols);
        for (uint32_t j = 0; j < coords.cols; ++j) row[j] = coords.data[i][j];
        basis.push_back(MatrixValue(ctx, FromMatrix(hecke::MatrixOfVec(row, D, D, ell))));
    }
    Record out;
    out.set("dim", Value::from_int(static_cast<std::int64_t>(dim)));
    out.set("ambient_dim", Value::from_int(static_cast<std::int64_t>(D)));
    out.set("basis", Value::from_list(std::move(basis)));
    return Value::from_record(std::move(out));
}

Value VerbCommutant(CallContext& ctx, const std::vector<Value>& a) {
    // {X : X A = A X for every A}: one linear system over vec(X), whose kernel
    // is the commutant.  Returned as a basis, one D x D matrix per element.
    const std::vector<Value>& items = a[0].list_value();
    if (items.empty()) throw TypeError("commutant(ops): the operator list is empty", ctx.pos);
    std::vector<Flat> ops;
    uint32_t D = 0, ell = 0;
    for (const Value& v : items) {
        const MatrixVal& M = AsMatrix(v, ctx.pos, "commutant(ops)");
        if (M.m.get_rows() != M.m.get_cols()) {
            throw TypeError("commutant(ops): every operator must be square", ctx.pos);
        }
        if (ops.empty()) { D = M.m.get_rows(); ell = M.m.get_p(); }
        else if (M.m.get_rows() != D || M.m.get_p() != ell) {
            throw TypeError("commutant(ops): the operators differ in size or modulus", ctx.pos);
        }
        ops.push_back(M.m);
    }
    const uint32_t n = D * D;
    Flat S(static_cast<uint32_t>(ops.size()) * n, n, ell);
    // row block t, row (i,j), column (k,l) of  (X A - A X)_{ij}
    //   = sum_k X_{ik} A_{kj} - sum_k A_{ik} X_{kj}
    for (std::size_t t = 0; t < ops.size(); ++t) {
        const Flat& A = ops[t];
        const uint32_t base = static_cast<uint32_t>(t) * n;
        for (uint32_t i = 0; i < D; ++i) {
            for (uint32_t j = 0; j < D; ++j) {
                const uint32_t row = base + i * D + j;
                for (uint32_t k = 0; k < D; ++k) {
                    // + X_{ik} A_{kj}
                    const uint32_t c1 = i * D + k;
                    uint64_t v1 = S.get_entry(row, c1);
                    v1 = (v1 + A.get_entry(k, j)) % ell;
                    S.set_entry(row, c1, static_cast<uint32_t>(v1));
                    // - A_{ik} X_{kj}
                    const uint32_t c2 = k * D + j;
                    uint64_t v2 = S.get_entry(row, c2);
                    v2 = (v2 + (ell - A.get_entry(i, k) % ell)) % ell;
                    S.set_entry(row, c2, static_cast<uint32_t>(v2));
                }
            }
        }
    }
    std::vector<Value> basis;
    for (const Flat& k : S.kernel()) {
        std::vector<uint32_t> row(n);
        for (uint32_t i = 0; i < n; ++i) row[i] = k.get_entry(i, 0);
        basis.push_back(MatrixValue(ctx, FromMatrix(hecke::MatrixOfVec(row, D, D, ell))));
    }
    Record out;
    out.set("dim", Value::from_int(static_cast<std::int64_t>(basis.size())));
    out.set("basis", Value::from_list(std::move(basis)));
    return Value::from_record(std::move(out));
}

// The span of F_ell[ops] . v, as a matrix of columns.
Flat CyclicSpan(const std::vector<Flat>& ops, const Flat& v, uint32_t D, uint32_t ell) {
    hecke::RrefRowSpan span(D, ell);
    std::vector<Flat> frontier;
    std::vector<uint32_t> coords(D);
    for (uint32_t i = 0; i < D; ++i) coords[i] = v.get_entry(i, 0);
    if (span.insert(coords)) frontier.push_back(v);
    while (!frontier.empty()) {
        std::vector<Flat> next;
        for (const Flat& w : frontier) {
            for (const Flat& A : ops) {
                const Flat aw = A * w;
                std::vector<uint32_t> c(D);
                for (uint32_t i = 0; i < D; ++i) c[i] = aw.get_entry(i, 0);
                if (span.insert(c)) next.push_back(aw);
            }
        }
        frontier.swap(next);
    }
    Flat out(D, span.dim(), ell);
    for (uint32_t i = 0; i < span.dim(); ++i)
        for (uint32_t j = 0; j < D; ++j) out.set_entry(j, i, span.rows()[i][j]);
    return out;
}

bool ReadOps(CallContext& ctx, const Value& list, const char* who, std::vector<Flat>& ops,
             uint32_t& D, uint32_t& ell) {
    const std::vector<Value>& items = list.list_value();
    if (items.empty()) throw TypeError(std::string(who) + ": the operator list is empty", ctx.pos);
    for (const Value& v : items) {
        const MatrixVal& M = AsMatrix(v, ctx.pos, who);
        if (M.m.get_rows() != M.m.get_cols()) {
            throw TypeError(std::string(who) + ": every operator must be square", ctx.pos);
        }
        if (ops.empty()) { D = M.m.get_rows(); ell = M.m.get_p(); }
        else if (M.m.get_rows() != D || M.m.get_p() != ell) {
            throw TypeError(std::string(who) + ": the operators differ in size or modulus", ctx.pos);
        }
        ops.push_back(M.m);
    }
    return true;
}

Value VerbIsCyclic(CallContext& ctx, const std::vector<Value>& a) {
    std::vector<Flat> ops;
    uint32_t D = 0, ell = 0;
    ReadOps(ctx, a[0], "is_cyclic(ops, v)", ops, D, ell);
    const MatrixVal& v = AsMatrix(a[1], ctx.pos, "is_cyclic(ops, v)");
    if (v.m.get_cols() != 1 || v.m.get_rows() != D) {
        throw TypeError("is_cyclic(ops, v): v must be a single column of the operators' size",
                        ctx.pos);
    }
    return Value::from_bool(CyclicSpan(ops, v.m, D, ell).get_cols() == D);
}

Value VerbCyclicVector(CallContext& ctx, const std::vector<Value>& a) {
    std::vector<Flat> ops;
    uint32_t D = 0, ell = 0;
    ReadOps(ctx, a[0], "cyclic_vector(ops, seed)", ops, D, ell);
    const uint64_t seed = U64Arg(a[1], ctx.pos, "cyclic_vector(ops, seed)");
    hecke::SplitMix64 rng(seed);
    // A fixed, seed-determined sequence of candidates; the seed is MANDATORY
    // ([I Sec. 5]), so the answer is reproducible.
    for (uint32_t attempt = 0; attempt < 256; ++attempt) {
        Flat v(D, 1, ell);
        for (uint32_t i = 0; i < D; ++i) v.set_entry(i, 0, rng.next_mod(ell));
        if (CyclicSpan(ops, v, D, ell).get_cols() == D) return MatrixValue(ctx, std::move(v));
    }
    throw RuntimeError("cyclic_vector(ops, seed): no cyclic vector found in 256 attempts (the "
                       "module may not be cyclic)",
                       ctx.pos);
}

Value VerbRandomVector(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "random_vector(p, n, seed)");
    const uint32_t n = U32Arg(a[1], ctx.pos, "random_vector(p, n, seed)");
    const uint64_t seed = U64Arg(a[2], ctx.pos, "random_vector(p, n, seed)");
    if (p < 2) throw TypeError("random_vector(p, n, seed): the modulus must be at least 2", ctx.pos);
    hecke::SplitMix64 rng(seed);
    Flat v(n, 1, p);
    for (uint32_t i = 0; i < n; ++i) v.set_entry(i, 0, rng.next_mod(p));
    return MatrixValue(ctx, std::move(v));
}

Value VerbRandomMatrix(CallContext& ctx, const std::vector<Value>& a) {
    const uint32_t p = U32Arg(a[0], ctx.pos, "random_matrix(p, rows, cols, seed)");
    const uint32_t r = U32Arg(a[1], ctx.pos, "random_matrix(p, rows, cols, seed)");
    const uint32_t c = U32Arg(a[2], ctx.pos, "random_matrix(p, rows, cols, seed)");
    const uint64_t seed = U64Arg(a[3], ctx.pos, "random_matrix(p, rows, cols, seed)");
    if (p < 2) throw TypeError("random_matrix: the modulus must be at least 2", ctx.pos);
    hecke::SplitMix64 rng(seed);
    Flat m(r, c, p);
    for (uint32_t i = 0; i < r; ++i)
        for (uint32_t j = 0; j < c; ++j) m.set_entry(i, j, rng.next_mod(p));
    return MatrixValue(ctx, std::move(m));
}

Value VerbJordanType(CallContext& ctx, const std::vector<Value>& a) {
    const MatrixVal& A = AsMatrix(a[0], ctx.pos, "jordan_type(A)");
    return IntList(NilpotentJordanType(A.m, ctx.pos, "jordan_type(A)"));
}

// ===========================================================================
// Types
// ===========================================================================

void RegisterTypes(Registry& registry) {
    {
        TypeInfo t;
        t.name = kTypeFp;
        t.doc = "an element of F_p (plan Sec. 3.4)";
        t.members.push_back({"p", false, "the modulus", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<FpVal>()->p);
                             }});
        t.members.push_back({"value", false, "the canonical residue in [0, p)",
                             [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<FpVal>()->a);
                             }});
        t.print = [](const ObjectPtr& o, std::int64_t) {
            const FpVal* v = o->payload_as<FpVal>();
            return std::to_string(v->a) + " mod " + std::to_string(v->p);
        };
        t.show = [](const ObjectPtr& o) {
            const FpVal* v = o->payload_as<FpVal>();
            return std::to_string(v->a) + " mod " + std::to_string(v->p);
        };
        t.equal = [](const ObjectPtr& x, const ObjectPtr& y) {
            const FpVal* u = x->payload_as<FpVal>();
            const FpVal* v = y->payload_as<FpVal>();
            return u->p == v->p && u->a == v->a;
        };
        t.hash = [](const ObjectPtr& o) { return HashScalarValue(o->payload_as<FpVal>()->a); };
        registry.register_type(std::move(t));
    }
    {
        TypeInfo t;
        t.name = kTypePoly;
        t.doc = "a polynomial over F_ell; prints HIGH -> LOW degree, while coeffs / hash use the "
                "machine order LOW -> HIGH (plan Sec. 2 C5)";
        t.members.push_back({"ell", false, "the modulus", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<PolyVal>()->f.ell);
                             }});
        t.members.push_back({"deg", false, "the degree (-1 for the zero polynomial)",
                             [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<PolyVal>()->f.degree());
                             }});
        t.members.push_back({"monic", false, "whether the leading coefficient is 1",
                             [](const ObjectPtr& o) {
                                 return Value::from_bool(o->payload_as<PolyVal>()->f.is_monic());
                             }});
        t.print = [](const ObjectPtr& o, std::int64_t) {
            return PolyText(o->payload_as<PolyVal>()->f);
        };
        t.show = [](const ObjectPtr& o) { return PolyText(o->payload_as<PolyVal>()->f); };
        t.equal = [](const ObjectPtr& x, const ObjectPtr& y) {
            return x->payload_as<PolyVal>()->f == y->payload_as<PolyVal>()->f;
        };
        t.hash = [](const ObjectPtr& o) { return HashPoly(o->payload_as<PolyVal>()->f); };
        registry.register_type(std::move(t));
    }
    {
        TypeInfo t;
        t.name = kTypeMatrix;
        t.doc = "a dense matrix over F_p (FlatMatrixT<uint32_t>)";
        t.members.push_back({"rows", false, "the row count", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<MatrixVal>()->m.get_rows());
                             }});
        t.members.push_back({"cols", false, "the column count", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<MatrixVal>()->m.get_cols());
                             }});
        t.members.push_back({"p", false, "the modulus", [](const ObjectPtr& o) {
                                 return Value::from_int(o->payload_as<MatrixVal>()->m.get_p());
                             }});
        TypeInfo::Method entry;
        entry.name = "entry";
        entry.params = {"row", "col"};
        entry.required = 2;
        entry.computed = false;
        entry.doc = "the (row, col) entry as a canonical residue";
        entry.call = [](const ObjectPtr& o, const std::vector<Value>& args, const SourcePos& pos) {
            const FlatMatrixT<uint32_t>& m = o->payload_as<MatrixVal>()->m;
            const uint32_t r = U32Arg(args[0], pos, "Matrix.entry(row, col)");
            const uint32_t c = U32Arg(args[1], pos, "Matrix.entry(row, col)");
            if (r >= m.get_rows() || c >= m.get_cols()) {
                throw RuntimeError("Matrix.entry(row, col): out of range", pos);
            }
            return Value::from_int(m.get_entry(r, c));
        };
        t.methods.push_back(std::move(entry));
        t.print = [](const ObjectPtr& o, std::int64_t cap) {
            return MatrixText(o->payload_as<MatrixVal>()->m, cap, false);
        };
        t.show = [](const ObjectPtr& o) {
            return MatrixText(o->payload_as<MatrixVal>()->m, 0, true);
        };
        t.equal = [](const ObjectPtr& x, const ObjectPtr& y) {
            const FlatMatrixT<uint32_t>& a = x->payload_as<MatrixVal>()->m;
            const FlatMatrixT<uint32_t>& b = y->payload_as<MatrixVal>()->m;
            return a.get_p() == b.get_p() && a.get_rows() == b.get_rows() &&
                   a.get_cols() == b.get_cols() && HashMatrix(a) == HashMatrix(b);
        };
        t.hash = [](const ObjectPtr& o) { return HashMatrix(o->payload_as<MatrixVal>()->m); };
        registry.register_type(std::move(t));
    }
}

} // namespace

void RegisterCoreMathBindings(Registry& registry) {
    RegisterTypes(registry);

    const char* g = "arithmetic";
    auto add = [&registry, g](const char* name, Signature sig, VerbImpl impl, const char* doc) {
        registry.register_verb(name, std::move(sig), std::move(impl), doc, g);
    };

    // Fp
    add("Fp", Sig({Param::of("p", ValueKind::Int), Param::of("a", ValueKind::Int)}), &VerbFp,
        "the element a of F_p, as a canonical residue");

    // Poly
    add("poly", Sig({Param::of("p", ValueKind::Int), Param::of("coeffs", ValueKind::List)}),
        &VerbPoly, "the polynomial with the given coefficients LOW -> HIGH over F_p");
    add("X", Sig({Param::of("p", ValueKind::Int)}), &VerbX, "the indeterminate X over F_p");
    add("coeffs", Sig({Param::object("f", kTypePoly)}), &VerbCoeffs,
        "the coefficients LOW -> HIGH (the machine order; the print form is HIGH -> LOW)");
    add("degree", Sig({Param::object("f", kTypePoly)}), &VerbDegree,
        "the degree, -1 for the zero polynomial");
    add("factor", Sig({Param::object("f", kTypePoly)}), &VerbFactor,
        "the certified factorisation of f (always passed through CertifyFactorization)");
    add("gcd", Sig({Param::object("f", kTypePoly), Param::object("g", kTypePoly)}), &VerbGcd,
        "the monic gcd");
    add("powmod",
        Sig({Param::object("g", kTypePoly), Param::of("e", ValueKind::Int),
             Param::object("f", kTypePoly)}),
        &VerbPowmod, "g^e mod f");
    add("eval", Sig({Param::object("f", kTypePoly), Param::of("x", ValueKind::Int)}), &VerbEval,
        "f(x) in F_ell");

    // Matrix construction
    add("matrix", Sig({Param::of("p", ValueKind::Int), Param::of("rows", ValueKind::List)}),
        &VerbMatrix, "a matrix over F_p from a list of equal-length rows");
    add("identity", Sig({Param::of("p", ValueKind::Int), Param::of("n", ValueKind::Int)}),
        &VerbIdentity, "the n x n identity over F_p");
    add("zero",
        Sig({Param::of("p", ValueKind::Int), Param::of("rows", ValueKind::Int),
             Param::of("cols", ValueKind::Int)}),
        &VerbZero, "the zero matrix over F_p");

    // arithmetic (the console's `*` and `+` are Int-only, so matrices and
    // polynomials compose through named verbs)
    add("mul", Sig({Param::any("a"), Param::any("b")}), &VerbMul,
        "the product of two matrices or two polynomials");
    add("add", Sig({Param::any("a"), Param::any("b")}), &VerbAdd,
        "the sum of two matrices or two polynomials");
    add("sub", Sig({Param::any("a"), Param::any("b")}), &VerbSub,
        "the difference of two matrices or two polynomials");
    add("scale", Sig({Param::object("A", kTypeMatrix), Param::of("c", ValueKind::Int)}), &VerbScale,
        "c * A");

    // linear algebra
    add("rank", Sig({Param::object("A", kTypeMatrix)}), &VerbRank, "the rank of A");
    add("nullity", Sig({Param::object("A", kTypeMatrix)}), &VerbNullity, "cols(A) - rank(A)");
    add("rref", Sig({Param::object("A", kTypeMatrix)}), &VerbRref,
        "the reduced row echelon form of A");
    add("kernel", Sig({Param::object("A", kTypeMatrix)}), &VerbKernel,
        "a basis of ker A, one column per basis vector");
    add("image", Sig({Param::object("A", kTypeMatrix)}), &VerbImage,
        "a basis of the column space of A, one column per basis vector");
    add("column_space", Sig({Param::object("A", kTypeMatrix)}), &VerbImage,
        "a basis of the column space of A (the name plan Sec. 3.5 uses)");
    add("inverse", Sig({Param::object("A", kTypeMatrix)}), &VerbInverse, "A^{-1}");
    add("transpose", Sig({Param::object("A", kTypeMatrix)}), &VerbTranspose, "A^T");
    add("pow", Sig({Param::object("A", kTypeMatrix), Param::of("e", ValueKind::Int)}), &VerbPow,
        "A^e for a square A and e >= 0 (A^0 = I)");
    add("solve", Sig({Param::object("A", kTypeMatrix), Param::object("B", kTypeMatrix)}),
        &VerbSolve, "the exact X with A*X = B (A of full column rank; typed error otherwise)");
    add("hstack", Sig({Param::of("parts", ValueKind::List)}), &VerbHStack,
        "the parts side by side");
    add("vstack", Sig({Param::of("parts", ValueKind::List)}), &VerbVStack,
        "the parts stacked");
    add("submatrix",
        Sig({Param::object("A", kTypeMatrix), Param::of("r0", ValueKind::Int),
             Param::of("r1", ValueKind::Int), Param::of("c0", ValueKind::Int),
             Param::of("c1", ValueKind::Int)}),
        &VerbSubmatrix, "the half-open block A[r0:r1, c0:c1]");
    add("columns", Sig({Param::object("A", kTypeMatrix), Param::of("idx", ValueKind::List)}),
        &VerbColumns, "the listed columns of A, in the order given");
    add("vec", Sig({Param::object("A", kTypeMatrix)}), &VerbVec,
        "A as one column, ROW-MAJOR (SubspaceOps' convention)");
    add("unvec",
        Sig({Param::object("v", kTypeMatrix), Param::of("rows", ValueKind::Int),
             Param::of("cols", ValueKind::Int)}),
        &VerbUnvec, "the inverse of vec");

    // the subspace lattice
    add("sum_spaces", Sig({Param::object("U", kTypeMatrix), Param::object("V", kTypeMatrix)}),
        &VerbSumSpaces, "a basis of the column span of [U | V]");
    add("intersect", Sig({Param::object("U", kTypeMatrix), Param::object("V", kTypeMatrix)}),
        &VerbIntersect, "a basis of the intersection of the two column spans");
    add("contains", Sig({Param::object("U", kTypeMatrix), Param::object("v", kTypeMatrix)}),
        &VerbContains, "whether v lies in the column span of U");
    add("in_span", Sig({Param::of("list", ValueKind::List), Param::object("A", kTypeMatrix)}),
        &VerbInSpan, "whether A lies in the F_ell-span of the listed matrices");

    // the module surface
    add("restrict", Sig({Param::object("A", kTypeMatrix), Param::object("basis", kTypeMatrix)}),
        &VerbRestrict, "the matrix of A on the invariant subspace spanned by the basis columns");
    add("algebra", Sig({Param::of("ops", ValueKind::List)}), &VerbAlgebra,
        "the unital F_ell-algebra generated by pairwise-commuting operators: {dim, basis}");
    add("commutant", Sig({Param::of("ops", ValueKind::List)}), &VerbCommutant,
        "{X : X A = A X for every A in ops}: {dim, basis}");
    add("is_cyclic", Sig({Param::of("ops", ValueKind::List), Param::object("v", kTypeMatrix)}),
        &VerbIsCyclic, "whether F_ell[ops] . v is the whole space");
    add("cyclic_vector", Sig({Param::of("ops", ValueKind::List), Param::of("seed", ValueKind::Int)}),
        &VerbCyclicVector, "a cyclic vector for F_ell[ops], from a MANDATORY seed");
    add("random_vector",
        Sig({Param::of("p", ValueKind::Int), Param::of("n", ValueKind::Int),
             Param::of("seed", ValueKind::Int)}),
        &VerbRandomVector, "a SplitMix64 vector; the seed is mandatory, so it is reproducible");
    add("random_matrix",
        Sig({Param::of("p", ValueKind::Int), Param::of("rows", ValueKind::Int),
             Param::of("cols", ValueKind::Int), Param::of("seed", ValueKind::Int)}),
        &VerbRandomMatrix, "a SplitMix64 matrix; the seed is mandatory");
    add("jordan_type", Sig({Param::object("A", kTypeMatrix)}), &VerbJordanType,
        "the Jordan partition of a NILPOTENT A, from the nullities of its powers");

    // polynomials of a matrix
    add("charpoly", Sig({Param::object("A", kTypeMatrix)}), &VerbCharPoly,
        "the characteristic polynomial (Berkowitz; division-free)");
    add("minpoly", Sig({Param::object("A", kTypeMatrix)}), &VerbMinPoly,
        "the minimal polynomial (Krylov)");
    add("trace", Sig({Param::object("A", kTypeMatrix)}), &VerbTrace, "the trace in F_p");
    add("eval_at", Sig({Param::object("f", kTypePoly), Param::object("A", kTypeMatrix)}),
        &VerbEvalAt, "f(A)");
}

} // namespace bind
} // namespace console
