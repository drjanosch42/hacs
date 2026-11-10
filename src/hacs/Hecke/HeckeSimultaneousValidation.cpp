// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S11V -- see HeckeSimultaneousValidation.h for the full independence
// statement and the derivation of every expected value.
//
// This file NEVER reads HeckeSimultaneousSmokeValidation.cpp (S11's own
// smoke test) for expected values, and NEVER calls
// hecke::HeckeModuleDecomposition::certify() as an oracle: every assertion
// below is checked against fields pulled out of the JSON string produced by
// hecke::BuildHeckeReportJson(), using the from-scratch mini-parser defined
// in the first anonymous namespace below (independent of the mini-parser
// embedded in HeckeSimultaneousSmokeValidation.cpp -- different type names,
// different traversal style, written from the JSON grammar directly), plus
// this file's own from-scratch F_ell[X] toolkit (independent of
// hecke::FpPoly / hecke::CharPolyBerkowitz / hecke::CertifyIrreducible).

#include "HeckeSimultaneousValidation.h"

#include "HeckeModuleDecomposition.h"
#include "HeckeOperatorFamily.h"
#include "HeckeReport.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <cctype>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ===========================================================================
// A from-scratch, hand-rolled JSON mini-parser LOCAL to this file (no
// third-party library; independent of the mini-parser embedded in
// HeckeSimultaneousSmokeValidation.cpp -- this one is written directly from
// the JSON grammar, with its own type/traversal conventions). Values in our
// own report JSON are always plain non-negative integers, booleans, null,
// strings, arrays, or objects (no floating point, no negative numbers,
// per HeckeReport.cpp's own emission style) -- this parser is scoped to
// exactly that subset.
// ===========================================================================
namespace s11v_json {

struct Value {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool bool_v = false;
    long long num_v = 0;
    std::string str_v;
    std::vector<Value> arr_v;
    std::vector<std::pair<std::string, Value>> obj_v; // insertion order, linear lookup

    bool has(const std::string& key) const {
        if (kind != Kind::Object) return false;
        for (const auto& kv : obj_v) {
            if (kv.first == key) return true;
        }
        return false;
    }
    const Value& at(const std::string& key) const {
        if (kind != Kind::Object) {
            throw std::runtime_error("s11v_json::Value::at: not an object (key '" + key + "')");
        }
        for (const auto& kv : obj_v) {
            if (kv.first == key) return kv.second;
        }
        throw std::runtime_error("s11v_json::Value::at: missing key '" + key + "'");
    }
    long long as_int() const {
        if (kind != Kind::Number) throw std::runtime_error("s11v_json::Value::as_int: not a number");
        return num_v;
    }
    bool as_bool() const {
        if (kind != Kind::Bool) throw std::runtime_error("s11v_json::Value::as_bool: not a bool");
        return bool_v;
    }
    bool is_null() const { return kind == Kind::Null; }
};

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Value ParseDocument() {
        SkipSpace();
        Value v = ParseAny();
        SkipSpace();
        if (pos_ != text_.size()) {
            throw std::runtime_error("s11v_json::Parser: trailing data after top-level value");
        }
        return v;
    }

private:
    const std::string& text_;
    size_t pos_ = 0;

    char Cur() const {
        if (pos_ >= text_.size()) throw std::runtime_error("s11v_json::Parser: unexpected EOF");
        return text_[pos_];
    }
    void SkipSpace() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    void Consume(char expected) {
        if (Cur() != expected) {
            throw std::runtime_error(std::string("s11v_json::Parser: expected '") + expected + "'");
        }
        ++pos_;
    }
    bool TryLiteral(const char* lit) {
        const size_t n = std::string(lit).size();
        if (text_.compare(pos_, n, lit) == 0) { pos_ += n; return true; }
        return false;
    }

    Value ParseAny() {
        SkipSpace();
        switch (Cur()) {
            case '{': return ParseObj();
            case '[': return ParseArr();
            case '"': return ParseStr();
            case 't':
            case 'f': return ParseBoolLit();
            case 'n': return ParseNullLit();
            default: return ParseNum();
        }
    }

    Value ParseObj() {
        Consume('{');
        Value v;
        v.kind = Value::Kind::Object;
        SkipSpace();
        if (Cur() == '}') { ++pos_; return v; }
        for (;;) {
            SkipSpace();
            Value key = ParseStr();
            SkipSpace();
            Consume(':');
            Value val = ParseAny();
            v.obj_v.emplace_back(key.str_v, std::move(val));
            SkipSpace();
            if (Cur() == ',') { ++pos_; continue; }
            Consume('}');
            break;
        }
        return v;
    }

    Value ParseArr() {
        Consume('[');
        Value v;
        v.kind = Value::Kind::Array;
        SkipSpace();
        if (Cur() == ']') { ++pos_; return v; }
        for (;;) {
            v.arr_v.push_back(ParseAny());
            SkipSpace();
            if (Cur() == ',') { ++pos_; continue; }
            Consume(']');
            break;
        }
        return v;
    }

    Value ParseStr() {
        Consume('"');
        std::string out;
        for (;;) {
            if (pos_ >= text_.size()) throw std::runtime_error("s11v_json::Parser: unterminated string");
            const char c = text_[pos_++];
            if (c == '"') break;
            if (c == '\\') {
                if (pos_ >= text_.size()) throw std::runtime_error("s11v_json::Parser: bad escape at EOF");
                const char e = text_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'u': pos_ += 4; out.push_back('?'); break; // not needed for our data
                    default: throw std::runtime_error("s11v_json::Parser: unknown escape");
                }
            } else {
                out.push_back(c);
            }
        }
        Value v;
        v.kind = Value::Kind::String;
        v.str_v = out;
        return v;
    }

    Value ParseBoolLit() {
        if (TryLiteral("true")) { Value v; v.kind = Value::Kind::Bool; v.bool_v = true; return v; }
        if (TryLiteral("false")) { Value v; v.kind = Value::Kind::Bool; v.bool_v = false; return v; }
        throw std::runtime_error("s11v_json::Parser: expected boolean literal");
    }

    Value ParseNullLit() {
        if (TryLiteral("null")) { Value v; v.kind = Value::Kind::Null; return v; }
        throw std::runtime_error("s11v_json::Parser: expected 'null'");
    }

    Value ParseNum() {
        const size_t start = pos_;
        if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ == start) throw std::runtime_error("s11v_json::Parser: expected a number");
        Value v;
        v.kind = Value::Kind::Number;
        v.num_v = std::stoll(text_.substr(start, pos_ - start));
        return v;
    }
};

Value Parse(const std::string& text) { return Parser(text).ParseDocument(); }

} // namespace s11v_json

// ===========================================================================
// A from-scratch F_ell[X] toolkit LOCAL to this file, deliberately
// independent of hecke::FpPoly (Stage S10/S11's own toolkit): every
// polynomial fact used to cross-check the JSON is re-derived here with
// fresh code, never by calling hecke::CharPolyBerkowitz / FactorPoly /
// CertifyIrreducible. Polynomials are plain vectors of coefficients, LOW ->
// HIGH degree (matching the JSON's own documented convention, A-N4),
// trimmed (top coefficient nonzero; the zero polynomial is {}).
// ===========================================================================
namespace {

using Poly = std::vector<uint32_t>;

uint32_t ModI64(int64_t x, uint32_t ell) {
    int64_t m = x % static_cast<int64_t>(ell);
    if (m < 0) m += static_cast<int64_t>(ell);
    return static_cast<uint32_t>(m);
}

Poly PTrim(Poly p) {
    while (!p.empty() && p.back() == 0u) p.pop_back();
    return p;
}

int64_t PDeg(const Poly& p) { return static_cast<int64_t>(p.size()) - 1; }

Poly PAdd(const Poly& a, const Poly& b, uint32_t ell) {
    Poly r(std::max(a.size(), b.size()), 0u);
    for (size_t i = 0; i < a.size(); ++i) r[i] = ModArithmetic::mod_add(r[i], a[i], ell);
    for (size_t i = 0; i < b.size(); ++i) r[i] = ModArithmetic::mod_add(r[i], b[i], ell);
    return PTrim(r);
}

Poly PScale(const Poly& a, uint32_t c, uint32_t ell) {
    Poly r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = ModArithmetic::mod_mul(a[i], c, ell);
    return PTrim(r);
}

Poly PMul(const Poly& a, const Poly& b, uint32_t ell) {
    if (a.empty() || b.empty()) return {};
    Poly r(a.size() + b.size() - 1, 0u);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0u) continue;
        for (size_t j = 0; j < b.size(); ++j) {
            if (b[j] == 0u) continue;
            r[i + j] = ModArithmetic::mod_add(r[i + j], ModArithmetic::mod_mul(a[i], b[j], ell), ell);
        }
    }
    return PTrim(r);
}

Poly PPow(const Poly& base, uint32_t e, uint32_t ell) {
    Poly result{1u};
    Poly b = PTrim(base);
    uint32_t exp = e;
    while (exp > 0) {
        if (exp & 1u) result = PMul(result, b, ell);
        b = PMul(b, b, ell);
        exp >>= 1;
    }
    return PTrim(result);
}

bool PEq(const Poly& a, const Poly& b) { return PTrim(a) == PTrim(b); }

uint32_t PEval(const Poly& p, uint32_t x, uint32_t ell) {
    uint32_t acc = 0;
    for (size_t i = p.size(); i-- > 0;) {
        acc = ModArithmetic::mod_add(ModArithmetic::mod_mul(acc, x, ell), p[i], ell);
    }
    return acc;
}

std::string PToString(const Poly& p) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < p.size(); ++i) {
        if (i) oss << ", ";
        oss << p[i];
    }
    oss << "] (low->high)";
    return oss.str();
}

// (X - root)^mult mod ell (root may be given as a negative or unreduced
// integer; reduced internally).
Poly PLinearPow(int64_t root, uint32_t mult, uint32_t ell) {
    const Poly lin = PTrim(Poly{ModI64(-root, ell), 1u});
    return PPow(lin, mult, ell);
}

// General field division a = q*b + r, deg r < deg b, b != 0.
void PDivRem(const Poly& a_in, const Poly& b_in, uint32_t ell, Poly& q, Poly& r) {
    Poly a = PTrim(a_in);
    Poly b = PTrim(b_in);
    if (b.empty()) throw std::invalid_argument("PDivRem: division by zero polynomial");
    const int64_t db = PDeg(b);
    const uint32_t lb_inv = ModArithmetic::mod_inv(b.back(), ell);
    r = a;
    q.assign(r.size() > b.size() ? r.size() - b.size() + 1 : 0, 0u);
    while (!r.empty() && PDeg(r) >= db) {
        const int64_t dr = PDeg(r);
        const uint32_t c = ModArithmetic::mod_mul(r.back(), lb_inv, ell);
        const size_t shift = static_cast<size_t>(dr - db);
        if (shift >= q.size()) q.resize(shift + 1, 0u);
        q[shift] = ModArithmetic::mod_add(q[shift], c, ell);
        for (size_t i = 0; i < b.size(); ++i) {
            r[shift + i] = ModArithmetic::mod_sub(r[shift + i], ModArithmetic::mod_mul(c, b[i], ell), ell);
        }
        r = PTrim(r);
    }
    q = PTrim(q);
}

Poly PMod(const Poly& a, const Poly& b, uint32_t ell) {
    Poly q, r;
    PDivRem(a, b, ell, q, r);
    return r;
}

Poly PGCD(Poly a, Poly b, uint32_t ell) {
    a = PTrim(a);
    b = PTrim(b);
    while (!b.empty()) {
        Poly r = PMod(a, b, ell);
        a = b;
        b = r;
    }
    if (a.empty()) return a;
    return PScale(a, ModArithmetic::mod_inv(a.back(), ell), ell);
}

// g^e mod f via square-and-multiply, from scratch.
Poly PPowMod(const Poly& base, uint64_t e, const Poly& modulus, uint32_t ell) {
    Poly result{1u};
    Poly b = PMod(base, modulus, ell);
    uint64_t exp = e;
    while (exp > 0) {
        if (exp & 1ull) result = PMod(PMul(result, b, ell), modulus, ell);
        b = PMod(PMul(b, b, ell), modulus, ell);
        exp >>= 1;
    }
    return result;
}

// X^{ell^k} mod modulus via k successive ell-th-power steps.
Poly FrobeniusIterate(uint32_t k, const Poly& modulus, uint32_t ell) {
    Poly cur = PMod(Poly{0u, 1u}, modulus, ell); // X mod modulus
    for (uint32_t i = 0; i < k; ++i) cur = PPowMod(cur, ell, modulus, ell);
    return cur;
}

// Deterministic, unconditional Rabin irreducibility test (never calls
// hecke::CertifyIrreducible): pi (monic, deg >= 1) is irreducible over F_ell
// iff gcd(pi, X^{ell^i} - X) == 1 for 1 <= i <= floor(deg/2), and
// X^{ell^deg} == X (mod pi).
bool PIrreducible(const Poly& pi_in, uint32_t ell) {
    const Poly pi = PTrim(pi_in);
    const int64_t deg = PDeg(pi);
    if (deg < 1) throw std::invalid_argument("PIrreducible: need deg(pi) >= 1");
    if (deg == 1) return true;
    const int64_t half = deg / 2;
    for (int64_t i = 1; i <= half; ++i) {
        const Poly fi = FrobeniusIterate(static_cast<uint32_t>(i), pi, ell);
        const Poly diff = PAdd(fi, PScale(Poly{0u, 1u}, ell - 1u, ell), ell); // fi - X
        const Poly g = PGCD(diff, pi, ell);
        if (!(g.size() == 1 && g[0] == 1u)) return false;
    }
    const Poly fd = FrobeniusIterate(static_cast<uint32_t>(deg), pi, ell);
    const Poly xmod = PMod(Poly{0u, 1u}, pi, ell);
    return PEq(fd, xmod);
}

// Recursive Leibniz/cofactor-expansion determinant of a matrix of
// polynomials over F_ell (division-free; correct for every prime ell,
// including ell <= matrix dimension) -- from scratch, independent of
// hecke::CharPolyBerkowitz.
Poly PMatDet(const std::vector<std::vector<Poly>>& M, uint32_t ell) {
    const size_t n = M.size();
    if (n == 0) return Poly{1u};
    if (n == 1) return M[0][0];
    Poly result;
    for (size_t j = 0; j < n; ++j) {
        std::vector<std::vector<Poly>> minor(n - 1, std::vector<Poly>(n - 1));
        for (size_t r = 1; r < n; ++r) {
            size_t cc = 0;
            for (size_t c = 0; c < n; ++c) {
                if (c == j) continue;
                minor[r - 1][cc++] = M[r][c];
            }
        }
        Poly term = PMul(M[0][j], PMatDet(minor, ell), ell);
        if (j % 2 == 1) term = PScale(term, ell - 1u, ell); // negate mod ell
        result = PAdd(result, term, ell);
    }
    return result;
}

// charpoly(A) = det(X*I - A), division-free, from scratch (independent of
// hecke::CharPolyBerkowitz). `A` must be square.
Poly CharPolyMine(const Matrix& A, uint32_t ell) {
    const uint32_t n = A.rows;
    std::vector<std::vector<Poly>> M(n, std::vector<Poly>(n));
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t aij = A.data[i][j];
            const uint32_t neg_aij = (aij == 0u) ? 0u : (ell - aij);
            M[i][j] = (i == j) ? PTrim(Poly{neg_aij, 1u}) : PTrim(Poly{neg_aij});
        }
    }
    return PMatDet(M, ell);
}

// Legendre symbol (a|ell) via Euler's criterion, computed AT RUNTIME
// (ModArithmetic::mod_pow), never hard-coded. +1 or -1; throws if a === 0.
int LegendreMine(uint32_t a, uint32_t ell) {
    const uint32_t a_mod = a % ell;
    if (a_mod == 0u) throw std::invalid_argument("LegendreMine: a === 0 (mod ell)");
    const uint32_t r = ModArithmetic::mod_pow(a_mod, (ell - 1) / 2, ell);
    if (r == 1u) return 1;
    if (r == ell - 1u) return -1;
    throw std::runtime_error("LegendreMine: Euler's criterion gave neither 1 nor ell-1");
}

// Eisenstein eigenvalue theta_p = 1 + p^{n+1} mod ell, computed at runtime
// via ModArithmetic::mod_pow (never hard-coded).
uint32_t EisensteinTheta(uint64_t p, uint32_t n, uint32_t ell) {
    const uint32_t p_mod = static_cast<uint32_t>(p % ell);
    const uint32_t p_pow = ModArithmetic::mod_pow(p_mod, n + 1u, ell);
    return ModArithmetic::mod_add(1u, p_pow, ell);
}

// Reconstructs a Matrix from a JSON array-of-rows (the level_b / basis
// entries of the report), residues taken as-is (already canonical in the
// report, but reduced defensively).
Matrix MatrixFromJson(const s11v_json::Value& rows, uint32_t ell) {
    const size_t r = rows.arr_v.size();
    const size_t c = (r > 0) ? rows.arr_v[0].arr_v.size() : 0;
    Matrix M(static_cast<uint32_t>(r), static_cast<uint32_t>(c), ell);
    for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) {
            M.data[i][j] = ModI64(rows.arr_v[i].arr_v[j].as_int(), ell);
        }
    }
    return M;
}

// Reads a coefficient-list (low->high, per A-N4) JSON array into a Poly.
Poly PolyFromJson(const s11v_json::Value& arr) {
    Poly p;
    p.reserve(arr.arr_v.size());
    for (const auto& e : arr.arr_v) p.push_back(static_cast<uint32_t>(e.as_int()));
    return PTrim(p);
}

} // namespace

// ===========================================================================
// Reporting context and the real-pipeline fixture (public API only: the
// pipeline construction itself is unavoidable boilerplate to obtain the REAL
// A^{(1)}(T_2)/A^{(1)}(T_3) matrices -- every ASSERTION below is against the
// re-parsed JSON, not against these C++ objects directly).
// ===========================================================================
namespace {

class Ctx {
public:
    uint32_t failures() const { return failures_; }
    void Fail(const std::string& msg) {
        ++failures_;
        std::cout << "HeckeSimultaneousValidation: FAIL: " << msg << std::endl;
    }
    void Ok(const std::string& msg) {
        std::cout << "HeckeSimultaneousValidation: OK: " << msg << std::endl;
    }

private:
    uint32_t failures_ = 0;
};

// The public pipeline: 7-vertex model -> base local system Sym^n(F_ell^2)
// -> its equivariant cohomology -> a hecke::HeckeOperatorFamily (which
// itself runs HeckeCosetTables(p,1,ell,n) -> HeckeTauTable -> HeckeOperatorOnK
// per requested prime, cached), exactly the pipeline the task specifies,
// packaged through the family class to avoid duplicating that four-stage
// wiring by hand for two primes.
struct PipelineFixture {
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;
    hecke::SL2ZTessellationTables model;
    std::unique_ptr<hecke::HeckeOperatorFamily> family_ptr;

    PipelineFixture(const SimplicialComplex& K, uint32_t ell, uint32_t n_sym)
        : lssc(K, MakeRelations(K, n_sym, ell), n_sym + 1, ell), coh(&lssc), model(K) {
        coh.compute_cohomology();
        family_ptr = std::make_unique<hecke::HeckeOperatorFamily>(&coh, K, model, ell, n_sym);
    }

    hecke::HeckeOperatorFamily& family() { return *family_ptr; }

private:
    static std::vector<LocalSystemSimplicialComplex::SimplexRelation> MakeRelations(
            const SimplicialComplex& K, uint32_t n_sym, uint32_t ell) {
        std::vector<LocalSystemSimplicialComplex::SimplexRelation> rels;
        for (const auto& datum : GenerateSL2ZRelationsWithGroupData(K, n_sym, ell)) {
            rels.push_back(datum.rel);
        }
        return rels;
    }
};

// Builds the family {T_2, T_3} at degree q=1, the HeckeModuleDecomposition,
// and the HeckeReport JSON string, per the task's Definition of Done. Also
// hands back the two level-(b) matrices (needed by the caller only to know
// their shape for a pre-check; every subsequent ASSERTION reads the JSON).
std::string BuildReportJson(uint32_t ell, uint32_t n_sym, Matrix& out_A2, Matrix& out_A3,
                            uint32_t& out_dimH1) {
    const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
    PipelineFixture fx(K, ell, n_sym);

    const auto& dims = fx.coh.get_H_eq_dimensions();
    out_dimH1 = (dims.size() >= 2) ? dims[1] : 0;

    const Matrix A2 = fx.family().T_p(2)[1];
    const Matrix A3 = fx.family().T_p(3)[1];
    out_A2 = A2;
    out_A3 = A3;

    std::vector<std::pair<uint64_t, Matrix>> ops = {{2ull, A2}, {3ull, A3}};
    hecke::HeckeModuleDecomposition decomp(ops, ell);
    decomp.compute();

    hecke::HeckeReportInput input;
    input.ell = ell;
    input.n_sym = n_sym;
    input.degree = 1;
    // (E0b, slot 2 WP-2) The oc1a fields are DERIVED from the complex's own
    // degree count, never written as a literal.
    input.num_degrees = static_cast<uint32_t>(dims.size());
    input.level_b = {{2ull, A2}, {3ull, A3}};
    input.decomp = &decomp;
    return hecke::BuildHeckeReportJson(input);
}

// ---- Small JSON-navigation helpers scoped to the level_c block schema. ----

Poly BlockPoly(const s11v_json::Value& block, const std::string& field, uint64_t p) {
    return PolyFromJson(block.at(field).at(std::to_string(p)));
}

bool BlockHasCertifiedResidueDegree(const s11v_json::Value& block, long long expected_f) {
    return block.has("residue_degree") &&
           block.at("residue_degree").as_int() == expected_f &&
           block.at("residue_degree_certified").as_bool();
}

// Cross-check: the product over ALL reported blocks of char_polys[p] must
// equal the characteristic polynomial (computed HERE, from scratch) of the
// level_b[p] matrix reconstructed from its own JSON array. This is the
// task's "level-(b) matrices ... consistent with the reported blocks" check.
void CheckLevelBConsistency(Ctx& ctx, const s11v_json::Value& root,
                            const s11v_json::Value& blocks, uint64_t p, uint32_t ell,
                            const std::string& tag) {
    const Matrix M = MatrixFromJson(root.at("level_b").at(std::to_string(p)), ell);
    const Poly full = CharPolyMine(M, ell);
    Poly product{1u};
    for (const auto& b : blocks.arr_v) {
        product = PMul(product, BlockPoly(b, "char_polys", p), ell);
    }
    if (!PEq(full, product)) {
        std::ostringstream oss;
        oss << tag << ": level_b[" << p << "]'s independently recomputed charpoly "
            << PToString(full) << " != product of reported level_c char_polys[" << p << "] "
            << PToString(product);
        ctx.Fail(oss.str());
        return;
    }
    ctx.Ok(tag + ": level_b[" + std::to_string(p) + "]'s own charpoly (recomputed by this "
                 "file's from-scratch cofactor-expansion determinant) matches the product of "
                 "the reported level_c blocks' char_polys[" + std::to_string(p) + "].");
}

void CheckSumOfDims(Ctx& ctx, const s11v_json::Value& lc, const std::string& tag) {
    const long long cohdim = lc.at("cohomology_dim").as_int();
    long long sum = 0;
    for (const auto& b : lc.at("blocks").arr_v) sum += b.at("dim").as_int();
    if (sum != cohdim) {
        std::ostringstream oss;
        oss << tag << ": sum of reported block dims (" << sum << ") != cohomology_dim (" << cohdim << ")";
        ctx.Fail(oss.str());
        return;
    }
    ctx.Ok(tag + ": sum of reported block dims == cohomology_dim (" + std::to_string(cohdim) + ").");
}

} // namespace

// ===========================================================================
// Case 1: (ell, n) = (13, 10). Expected (see the .h derivation): dim H^1 = 3;
// blocks = {dim 2 cuspidal (Delta system), dim 1 Eisenstein}; both
// residue_degree = 1 CERTIFIED; Eisenstein eigen_system {2:8, 3:10}.
// ===========================================================================
namespace {

void RunEll13(Ctx& ctx) {
    const uint32_t ell = 13, n = 10;
    const std::string tag = "(13,10)";
    try {
        Matrix A2, A3;
        uint32_t dimH1 = 0;
        const std::string json = BuildReportJson(ell, n, A2, A3, dimH1);
        const s11v_json::Value root = s11v_json::Parse(json);

        if (root.at("ell").as_int() != 13 || root.at("n_sym").as_int() != 10 ||
            root.at("degree").as_int() != 1) {
            ctx.Fail(tag + ": JSON metadata (ell/n_sym/degree) mismatch");
            return;
        }

        const s11v_json::Value& lc = root.at("level_c");
        if (lc.at("cohomology_dim").as_int() != 3) {
            ctx.Fail(tag + ": cohomology_dim != 3, got " + std::to_string(lc.at("cohomology_dim").as_int()));
            return;
        }
        CheckSumOfDims(ctx, lc, tag);

        const s11v_json::Value& blocks = lc.at("blocks");
        if (blocks.arr_v.size() != 2) {
            ctx.Fail(tag + ": expected EXACTLY 2 blocks, got " + std::to_string(blocks.arr_v.size()));
            return;
        }

        const s11v_json::Value* cusp = nullptr;
        const s11v_json::Value* eis = nullptr;
        for (const auto& b : blocks.arr_v) {
            const long long d = b.at("dim").as_int();
            if (d == 2) cusp = &b;
            else if (d == 1) eis = &b;
        }
        if (cusp == nullptr || eis == nullptr) {
            ctx.Fail(tag + ": expected one dim-2 and one dim-1 block");
            return;
        }
        ctx.Ok(tag + ": block dims = {2, 1} as expected (cuspidal Delta system + Eisenstein).");

        // Independent derivation: a_2(Delta) = -24, a_3(Delta) = 252
        // (LMFDB 1.12.a.a). Cuspidal char_polys must be (X-a_2)^2, (X-a_3)^2
        // mod 13.
        const Poly exp_cusp2 = PLinearPow(-24, 2, ell);
        const Poly exp_cusp3 = PLinearPow(252, 2, ell);
        const Poly got_cusp2 = BlockPoly(*cusp, "char_polys", 2);
        const Poly got_cusp3 = BlockPoly(*cusp, "char_polys", 3);
        if (!PEq(got_cusp2, exp_cusp2)) {
            ctx.Fail(tag + ": cuspidal block char_polys[2] != (X+24)^2 mod 13, got " + PToString(got_cusp2));
            return;
        }
        if (!PEq(got_cusp3, exp_cusp3)) {
            ctx.Fail(tag + ": cuspidal block char_polys[3] != (X-252)^2 mod 13, got " + PToString(got_cusp3));
            return;
        }
        ctx.Ok(tag + ": cuspidal block char_polys[2]=(X+24)^2, char_polys[3]=(X-252)^2 mod 13, "
                     "independently re-derived from a_2(Delta)=-24 / a_3(Delta)=252.");
        if (!BlockHasCertifiedResidueDegree(*cusp, 1)) {
            ctx.Fail(tag + ": cuspidal block residue_degree != 1 (certified)");
            return;
        }
        {
            const s11v_json::Value& jp2 = cusp->at("jordan_partitions").at("2");
            const s11v_json::Value& jp3 = cusp->at("jordan_partitions").at("3");
            auto PrintIntArr = [](const s11v_json::Value& arr) {
                std::ostringstream oss;
                oss << "(";
                for (size_t i = 0; i < arr.arr_v.size(); ++i) {
                    if (i) oss << ",";
                    oss << arr.arr_v[i].as_int();
                }
                oss << ")";
                return oss.str();
            };
            std::cout << "HeckeSimultaneousValidation: RECORDED (13,10) cuspidal per-op Jordan "
                         "partitions (isotypic multiplicity 2 -- holo/antiholo share the FULL "
                         "eigensystem, not chased for a further split): T_2 -> " << PrintIntArr(jp2)
                      << ", T_3 -> " << PrintIntArr(jp3) << "; residue_degree=1 (certified)."
                      << std::endl;
        }

        // Independent derivation: Eisenstein theta_p = 1+p^{n+1} mod ell,
        // computed at runtime, never hard-coded.
        const uint32_t exp_theta2 = EisensteinTheta(2, n, ell);
        const uint32_t exp_theta3 = EisensteinTheta(3, n, ell);
        if (exp_theta2 != 8u || exp_theta3 != 10u) {
            ctx.Fail(tag + ": internal derivation error -- expected Eisenstein thetas {2:8,3:10} "
                            "mod 13 not reproduced by EisensteinTheta()");
            return;
        }
        if (!BlockHasCertifiedResidueDegree(*eis, 1)) {
            ctx.Fail(tag + ": Eisenstein block residue_degree != 1 (certified)");
            return;
        }
        if (!eis->has("eigen_system")) {
            ctx.Fail(tag + ": Eisenstein (dim-1, residue_degree-1, certified) block missing "
                            "'eigen_system' in the JSON");
            return;
        }
        const s11v_json::Value& es = eis->at("eigen_system");
        const long long th2 = es.at("2").as_int();
        const long long th3 = es.at("3").as_int();
        if (th2 != static_cast<long long>(exp_theta2) || th3 != static_cast<long long>(exp_theta3)) {
            std::ostringstream oss;
            oss << tag << ": Eisenstein eigen_system {2:" << th2 << ",3:" << th3
                << "} != independently computed {2:" << exp_theta2 << ",3:" << exp_theta3 << "}";
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok(tag + ": Eisenstein eigen_system == {2:8, 3:10} mod 13, matching the "
                     "independently computed 1+2^11 and 1+3^11 mod 13.");

        CheckLevelBConsistency(ctx, root, blocks, 2, ell, tag);
        CheckLevelBConsistency(ctx, root, blocks, 3, ell, tag);

        if (!lc.at("certified").as_bool()) {
            ctx.Fail(tag + ": JSON level_c.certified != true (informational cross-check; the "
                            "mathematical claims above were already independently verified "
                            "without relying on this flag)");
        }

    } catch (const std::exception& e) {
        ctx.Fail(tag + ": EXCEPTION: " + e.what());
    }
}

} // namespace

// ===========================================================================
// Case 2: (ell, n) = (691, 10). Expected: a single dim-3 block; T_2
// char=(X+24)^3, min=(X+24)^2 (partition (2,1)), residue_degree=1 CERTIFIED;
// the Ramanujan congruence re-verified for p=3 as well.
// ===========================================================================
namespace {

void RunEll691(Ctx& ctx) {
    const uint32_t ell = 691, n = 10;
    const std::string tag = "(691,10)";
    try {
        Matrix A2, A3;
        uint32_t dimH1 = 0;
        const std::string json = BuildReportJson(ell, n, A2, A3, dimH1);
        const s11v_json::Value root = s11v_json::Parse(json);

        if (root.at("ell").as_int() != 691 || root.at("n_sym").as_int() != 10) {
            ctx.Fail(tag + ": JSON metadata mismatch");
            return;
        }

        const s11v_json::Value& lc = root.at("level_c");
        if (lc.at("cohomology_dim").as_int() != 3) {
            ctx.Fail(tag + ": cohomology_dim != 3");
            return;
        }
        CheckSumOfDims(ctx, lc, tag);

        const s11v_json::Value& blocks = lc.at("blocks");
        if (blocks.arr_v.size() != 1) {
            ctx.Fail(tag + ": expected EXACTLY ONE block under {T_2,T_3}, got " +
                     std::to_string(blocks.arr_v.size()));
            return;
        }
        const s11v_json::Value& blk = blocks.arr_v[0];
        if (blk.at("dim").as_int() != 3) {
            ctx.Fail(tag + ": block dim != 3");
            return;
        }
        ctx.Ok(tag + ": simultaneous decomposition under {T_2,T_3} = a single dim-3 block.");

        // Independent re-derivation of the classical Ramanujan congruence
        // tau(p) === 1+p^11 (mod 691), checked for BOTH p=2 and p=3 by
        // direct integer / mod_pow arithmetic (not merely quoted).
        if ((2049 - (-24)) % 691 != 0) {
            ctx.Fail(tag + ": internal derivation error -- 2049-(-24) is not a multiple of 691");
            return;
        }
        const uint32_t e3_theta = EisensteinTheta(3, n, ell); // 1+3^11 mod 691
        if (e3_theta != 252u) {
            ctx.Fail(tag + ": internal derivation error -- 1+3^11 mod 691 != 252, got " +
                     std::to_string(e3_theta));
            return;
        }
        ctx.Ok(tag + ": Ramanujan congruence re-verified: 1+3^11 === 252 === a_3(Delta) mod 691 "
                     "(so T_3 also collapses onto the single block, consistent with 2049 === -24 "
                     "mod 691 for T_2).");

        const Poly exp_cp2 = PLinearPow(-24, 3, ell);
        const Poly exp_mp2 = PLinearPow(-24, 2, ell);
        const Poly got_cp2 = BlockPoly(blk, "char_polys", 2);
        const Poly got_mp2 = BlockPoly(blk, "min_polys", 2);
        if (!PEq(got_cp2, exp_cp2)) {
            ctx.Fail(tag + ": block char_polys[2] != (X+24)^3 mod 691, got " + PToString(got_cp2));
            return;
        }
        if (!PEq(got_mp2, exp_mp2)) {
            ctx.Fail(tag + ": block min_polys[2] != (X+24)^2 mod 691, got " + PToString(got_mp2));
            return;
        }
        ctx.Ok(tag + ": T_2 char_poly=(X+24)^3, min_poly=(X+24)^2 mod 691 confirmed from the JSON.");

        const s11v_json::Value& jp2 = blk.at("jordan_partitions").at("2");
        if (jp2.arr_v.size() != 2 || jp2.arr_v[0].as_int() != 2 || jp2.arr_v[1].as_int() != 1) {
            std::ostringstream oss;
            oss << tag << ": T_2 Jordan partition != (2,1), got (";
            for (size_t i = 0; i < jp2.arr_v.size(); ++i) {
                if (i) oss << ",";
                oss << jp2.arr_v[i].as_int();
            }
            oss << ")";
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok(tag + ": T_2 Jordan partition == (2,1) confirmed from the JSON.");

        if (!BlockHasCertifiedResidueDegree(blk, 1)) {
            ctx.Fail(tag + ": block residue_degree != 1 (certified)");
            return;
        }
        ctx.Ok(tag + ": residue_degree == 1 (certified).");

        CheckLevelBConsistency(ctx, root, blocks, 2, ell, tag);
        CheckLevelBConsistency(ctx, root, blocks, 3, ell, tag);

    } catch (const std::exception& e) {
        ctx.Fail(tag + ": EXCEPTION: " + e.what());
    }
}

} // namespace

// ===========================================================================
// Cases 3 & 4: weight 24 (n=22), ell in {10007, 101}. The Legendre symbol
// (144169|ell) is computed HERE at runtime (Euler's criterion via
// ModArithmetic::mod_pow) and used both to (a) confirm the expected branch
// (+1 for 10007, -1 for 101) and (b) select which block structure to assert.
// ===========================================================================
namespace {

void RunWeight24(Ctx& ctx, uint32_t ell, int expected_legendre) {
    const uint32_t n = 22;
    std::ostringstream tag_oss;
    tag_oss << "(" << ell << ",22)";
    const std::string tag = tag_oss.str();
    try {
        Matrix A2, A3;
        uint32_t dimH1 = 0;
        const std::string json = BuildReportJson(ell, n, A2, A3, dimH1);
        const s11v_json::Value root = s11v_json::Parse(json);

        if (root.at("ell").as_int() != static_cast<long long>(ell) || root.at("n_sym").as_int() != 22) {
            ctx.Fail(tag + ": JSON metadata mismatch");
            return;
        }

        const s11v_json::Value& lc = root.at("level_c");
        if (lc.at("cohomology_dim").as_int() != 5) {
            ctx.Fail(tag + ": cohomology_dim != 5, got " + std::to_string(lc.at("cohomology_dim").as_int()));
            return;
        }
        CheckSumOfDims(ctx, lc, tag);

        // Independent derivation checks (re-verified by direct arithmetic,
        // not merely quoted): the discriminant identity behind the
        // Legendre-symbol criterion, and 144169 not vanishing mod ell.
        const int64_t disc = 1080LL * 1080LL + 4LL * 20468736LL;
        if (disc != 576LL * 144169LL) {
            ctx.Fail(tag + ": internal derivation error -- 1080^2+4*20468736 != 24^2*144169");
            return;
        }
        if (144169u % ell == 0u) {
            ctx.Fail(tag + ": internal derivation error -- 144169 === 0 mod ell, Legendre symbol undefined");
            return;
        }

        // The Legendre symbol, computed AT RUNTIME here (never hard-coded),
        // both confirmed against the expected branch AND used to select it.
        const int legendre = LegendreMine(144169u, ell);
        if (legendre != expected_legendre) {
            std::ostringstream oss;
            oss << tag << ": Legendre(144169|" << ell << ") computed as " << legendre
                << ", expected " << expected_legendre << " -- BRANCH MISMATCH";
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok(tag + ": Legendre(144169|" + std::to_string(ell) + ") = " + std::to_string(legendre) +
                     " confirmed at runtime via Euler's criterion (ModArithmetic::mod_pow).");

        const Poly quad = PTrim(Poly{ModI64(-20468736LL, ell), ModI64(-1080LL, ell), 1u});
        const Poly quad_sq = PMul(quad, quad, ell);
        const bool quad_irreducible = PIrreducible(quad, ell);
        if ((legendre == -1) != quad_irreducible) {
            ctx.Fail(tag + ": Legendre-symbol prediction and this file's OWN gcd/Frobenius "
                            "irreducibility test on X^2-1080X-20468736 disagree");
            return;
        }
        ctx.Ok(tag + ": OWN gcd/Frobenius irreducibility test on X^2-1080X-20468736 agrees with "
                     "the Legendre-symbol prediction (irreducible == " +
               std::string(quad_irreducible ? "true" : "false") + ").");

        const s11v_json::Value& blocks = lc.at("blocks");
        const s11v_json::Value* eis = nullptr;
        std::vector<const s11v_json::Value*> others;
        for (const auto& b : blocks.arr_v) {
            if (b.at("dim").as_int() == 1) eis = &b; else others.push_back(&b);
        }
        if (eis == nullptr) {
            ctx.Fail(tag + ": no dim-1 (Eisenstein) block found");
            return;
        }
        if (!BlockHasCertifiedResidueDegree(*eis, 1)) {
            ctx.Fail(tag + ": Eisenstein block residue_degree != 1 (certified)");
            return;
        }
        const Poly eis_cp2 = BlockPoly(*eis, "char_polys", 2);
        const uint32_t exp_theta2 = EisensteinTheta(2, n, ell);
        const uint32_t exp_theta3 = EisensteinTheta(3, n, ell);
        if (!PEq(eis_cp2, PLinearPow(static_cast<int64_t>(exp_theta2), 1, ell))) {
            ctx.Fail(tag + ": Eisenstein char_polys[2] != X - (1+2^23 mod ell), got " + PToString(eis_cp2));
            return;
        }
        if (!eis->has("eigen_system")) {
            ctx.Fail(tag + ": Eisenstein block missing 'eigen_system'");
            return;
        }
        const s11v_json::Value& es = eis->at("eigen_system");
        if (es.at("2").as_int() != static_cast<long long>(exp_theta2) ||
            es.at("3").as_int() != static_cast<long long>(exp_theta3)) {
            std::ostringstream oss;
            oss << tag << ": Eisenstein eigen_system {2:" << es.at("2").as_int() << ",3:"
                << es.at("3").as_int() << "} != independently computed {2:" << exp_theta2
                << ",3:" << exp_theta3 << "}";
            ctx.Fail(oss.str());
            return;
        }
        ctx.Ok(tag + ": Eisenstein block (dim 1, residue_degree 1 certified) eigen_system == "
                     "independently computed {2:1+2^23, 3:1+3^23} mod " + std::to_string(ell) + ".");

        if (legendre == -1) {
            // ---- (144169|ell) = -1: ONE dim-4 block, residue_degree 2 CERTIFIED. ----
            if (others.size() != 1) {
                ctx.Fail(tag + ": Legendre=-1 branch: expected exactly one non-Eisenstein block, got " +
                         std::to_string(others.size()));
                return;
            }
            const s11v_json::Value& cusp = *others[0];
            if (cusp.at("dim").as_int() != 4) {
                ctx.Fail(tag + ": Legendre=-1 branch: non-Eisenstein block dim != 4");
                return;
            }
            if (!BlockHasCertifiedResidueDegree(cusp, 2)) {
                ctx.Fail(tag + ": Legendre=-1 branch: dim-4 block residue_degree != 2 (certified)");
                return;
            }
            if (!cusp.has("residue_witness") || cusp.at("residue_witness").arr_v.empty()) {
                ctx.Fail(tag + ": Legendre=-1 branch: certified dim-4 block has no stored residue_witness");
                return;
            }
            ctx.Ok(tag + ": dim-4 block: residue_degree = 2, residue_degree_certified = true, "
                         "with a stored, non-empty residue_witness (" +
                   std::to_string(cusp.at("residue_witness").arr_v.size()) + " term(s)).");

            const Poly cusp_cp2 = BlockPoly(cusp, "char_polys", 2);
            if (!PEq(cusp_cp2, quad_sq)) {
                ctx.Fail(tag + ": Legendre=-1 branch: dim-4 block char_polys[2] != "
                                "(X^2-1080X-20468736)^2, got " + PToString(cusp_cp2));
                return;
            }
            ctx.Ok(tag + ": dim-4 block char_polys[2] == (X^2-1080X-20468736)^2, the irreducible "
                         "weight-24 Galois pair fused over F_" + std::to_string(ell) + ".");

            ctx.Ok(tag + ": block dims = {4, 1} confirmed for the Legendre=-1 branch.");
        } else {
            // ---- (144169|ell) = +1: TWO dim-2 blocks, residue_degree 1 each. ----
            if (others.size() != 2) {
                ctx.Fail(tag + ": Legendre=+1 branch: expected exactly two non-Eisenstein blocks, got " +
                         std::to_string(others.size()));
                return;
            }
            std::vector<uint32_t> roots;
            for (const s11v_json::Value* bp : others) {
                const s11v_json::Value& b = *bp;
                if (b.at("dim").as_int() != 2) {
                    ctx.Fail(tag + ": Legendre=+1 branch: non-Eisenstein block dim != 2");
                    return;
                }
                if (!BlockHasCertifiedResidueDegree(b, 1)) {
                    ctx.Fail(tag + ": Legendre=+1 branch: non-Eisenstein block residue_degree != 1 (certified)");
                    return;
                }
                const Poly cp = BlockPoly(b, "char_polys", 2);
                if (cp.size() != 3 || cp.back() != 1u) {
                    ctx.Fail(tag + ": Legendre=+1 branch: dim-2 block char_polys[2] is not a monic "
                                    "quadratic: " + PToString(cp));
                    return;
                }
                const uint32_t two_inv = ModArithmetic::mod_inv(2u, ell);
                const uint32_t r = ModArithmetic::mod_mul(ModArithmetic::mod_sub(0u, cp[1], ell), two_inv, ell);
                const uint32_t r_sq = ModArithmetic::mod_mul(r, r, ell);
                if (cp[0] != r_sq) {
                    ctx.Fail(tag + ": Legendre=+1 branch: dim-2 block char_polys[2] is not a perfect "
                                    "square (X-r)^2 for the r extracted from its linear coefficient");
                    return;
                }
                if (PEval(quad, r, ell) != 0u) {
                    ctx.Fail(tag + ": Legendre=+1 branch: extracted root r=" + std::to_string(r) +
                                    " is not a root of the theoretical quadratic mod " + std::to_string(ell));
                    return;
                }
                roots.push_back(r);
            }
            if (roots.size() != 2 || roots[0] == roots[1]) {
                ctx.Fail(tag + ": Legendre=+1 branch: the two dim-2 blocks' extracted roots are not distinct");
                return;
            }
            const Poly cp0 = BlockPoly(*others[0], "char_polys", 2);
            const Poly cp1 = BlockPoly(*others[1], "char_polys", 2);
            const Poly product = PMul(cp0, cp1, ell);
            if (!PEq(product, quad_sq)) {
                ctx.Fail(tag + ": Legendre=+1 branch: product of the two dim-2 blocks' char_polys[2] "
                                "!= (X^2-1080X-20468736)^2");
                return;
            }
            ctx.Ok(tag + ": block dims = {2, 2, 1} confirmed for the Legendre=+1 branch; the two "
                         "dim-2 blocks' extracted roots are distinct F_" + std::to_string(ell) +
                         "-rationals, each a genuine root of the theoretical quadratic, product == "
                         "(quadratic)^2.");
        }

        CheckLevelBConsistency(ctx, root, blocks, 2, ell, tag);
        CheckLevelBConsistency(ctx, root, blocks, 3, ell, tag);

    } catch (const std::exception& e) {
        ctx.Fail(tag + ": EXCEPTION: " + e.what());
    }
}

} // namespace

uint32_t HeckeSimultaneousValidation() {
    try {
        Ctx ctx;
        RunEll13(ctx);
        RunEll691(ctx);
        RunWeight24(ctx, 10007u, 1);
        RunWeight24(ctx, 101u, -1);
        std::cout << "HeckeSimultaneousValidation: " << ctx.failures() << " failing case(s)." << std::endl;
        return ctx.failures();
    } catch (const std::exception& e) {
        std::cout << "HeckeSimultaneousValidation: EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
