// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// Stage S11 -- see HeckeSimultaneousSmokeValidation.h for the full
// Definition-of-Done statement.

#include "HeckeSimultaneousSmokeValidation.h"

#include "FpPoly.h"
#include "HeckeModuleDecomposition.h"
#include "HeckeOperatorFamily.h"
#include "HeckeReport.h"
#include "SL2ZTessellationTables.h"
#include "../EquivariantCohomologyOnK.h"
#include "../LocalSystemSL2Z.h"
#include "../LocalSystemSimplicialComplex.h"
#include "../SimplicialCohomology.h"
#include "../../ModArithmetic/Matrix.h"

#include <cctype>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ===========================================================================
// A minimal, from-scratch, hand-rolled JSON parser LOCAL to this file (no
// third-party library, per the plan's binding requirement): just enough to
// round-trip hecke::BuildHeckeReportJson()'s output and navigate a handful
// of fields for the S11 Definition-of-Done check "the JSON round-trips
// through an embedded mini-parser". Numbers in our own JSON output are
// always plain (possibly negative, though none of ours are) integers --
// this parser stores every number as a plain long long.
// ===========================================================================
namespace {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool bool_val = false;
    long long num_val = 0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val; // insertion order; linear lookup

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }

    bool has(const std::string& key) const {
        if (type != Type::Object) return false;
        for (const auto& kv : obj_val) if (kv.first == key) return true;
        return false;
    }
    const JsonValue& at(const std::string& key) const {
        if (type != Type::Object) {
            throw std::runtime_error("JsonValue::at: not an object (looking up '" + key + "')");
        }
        for (const auto& kv : obj_val) if (kv.first == key) return kv.second;
        throw std::runtime_error("JsonValue::at: missing key '" + key + "'");
    }
    const std::vector<JsonValue>& items() const {
        if (type != Type::Array) throw std::runtime_error("JsonValue::items: not an array");
        return arr_val;
    }
    long long as_int() const {
        if (type != Type::Number) throw std::runtime_error("JsonValue::as_int: not a number");
        return num_val;
    }
    bool as_bool() const {
        if (type != Type::Bool) throw std::runtime_error("JsonValue::as_bool: not a bool");
        return bool_val;
    }
    bool is_null() const { return type == Type::Null; }
};

class MiniJsonParser {
public:
    explicit MiniJsonParser(const std::string& text) : s_(text) {}

    JsonValue Parse() {
        SkipWs();
        JsonValue v = ParseValue();
        SkipWs();
        if (i_ != s_.size()) {
            throw std::runtime_error("MiniJsonParser: trailing characters after top-level value");
        }
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    void SkipWs() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    char Peek() const {
        if (i_ >= s_.size()) throw std::runtime_error("MiniJsonParser: unexpected end of input");
        return s_[i_];
    }
    void Expect(char c) {
        if (Peek() != c) {
            throw std::runtime_error(std::string("MiniJsonParser: expected '") + c + "' at offset " +
                                     std::to_string(i_));
        }
        ++i_;
    }
    bool MatchLiteral(const char* lit) {
        const size_t len = std::string(lit).size();
        if (s_.compare(i_, len, lit) == 0) { i_ += len; return true; }
        return false;
    }

    JsonValue ParseValue() {
        SkipWs();
        const char c = Peek();
        if (c == '{') return ParseObject();
        if (c == '[') return ParseArray();
        if (c == '"') return ParseString();
        if (c == 't' || c == 'f') return ParseBool();
        if (c == 'n') return ParseNull();
        return ParseNumber();
    }

    JsonValue ParseObject() {
        Expect('{');
        JsonValue v;
        v.type = JsonValue::Type::Object;
        SkipWs();
        if (Peek() == '}') { ++i_; return v; }
        while (true) {
            SkipWs();
            JsonValue key = ParseString();
            SkipWs();
            Expect(':');
            JsonValue val = ParseValue();
            v.obj_val.emplace_back(key.str_val, std::move(val));
            SkipWs();
            const char c = Peek();
            if (c == ',') { ++i_; continue; }
            if (c == '}') { ++i_; break; }
            throw std::runtime_error("MiniJsonParser: expected ',' or '}' in object");
        }
        return v;
    }

    JsonValue ParseArray() {
        Expect('[');
        JsonValue v;
        v.type = JsonValue::Type::Array;
        SkipWs();
        if (Peek() == ']') { ++i_; return v; }
        while (true) {
            v.arr_val.push_back(ParseValue());
            SkipWs();
            const char c = Peek();
            if (c == ',') { ++i_; continue; }
            if (c == ']') { ++i_; break; }
            throw std::runtime_error("MiniJsonParser: expected ',' or ']' in array");
        }
        return v;
    }

    JsonValue ParseString() {
        Expect('"');
        std::string out;
        while (true) {
            if (i_ >= s_.size()) throw std::runtime_error("MiniJsonParser: unterminated string");
            const char c = s_[i_++];
            if (c == '"') break;
            if (c == '\\') {
                if (i_ >= s_.size()) throw std::runtime_error("MiniJsonParser: bad escape at EOF");
                const char e = s_[i_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'u': i_ += 4; out.push_back('?'); break; // not needed for our own data
                    default: throw std::runtime_error("MiniJsonParser: unknown escape");
                }
            } else {
                out.push_back(c);
            }
        }
        JsonValue v;
        v.type = JsonValue::Type::String;
        v.str_val = out;
        return v;
    }

    JsonValue ParseBool() {
        if (MatchLiteral("true")) { JsonValue v; v.type = JsonValue::Type::Bool; v.bool_val = true; return v; }
        if (MatchLiteral("false")) { JsonValue v; v.type = JsonValue::Type::Bool; v.bool_val = false; return v; }
        throw std::runtime_error("MiniJsonParser: expected 'true' or 'false'");
    }

    JsonValue ParseNull() {
        if (MatchLiteral("null")) { JsonValue v; v.type = JsonValue::Type::Null; return v; }
        throw std::runtime_error("MiniJsonParser: expected 'null'");
    }

    JsonValue ParseNumber() {
        const size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') ++i_;
        while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        }
        if (i_ == start) throw std::runtime_error("MiniJsonParser: expected a number");
        const std::string tok = s_.substr(start, i_ - start);
        JsonValue v;
        v.type = JsonValue::Type::Number;
        v.num_val = std::stoll(tok); // our own output never emits fractional/exponential numbers
        return v;
    }
};

} // namespace

// ===========================================================================
// Pipeline fixture: the REAL pipeline (7-vertex SL(2,Z) model, base local
// system Sym^n(F_ell^2), its cohomology, and a HeckeOperatorFamily over it),
// exactly as HeckeOperatorFamilySmokeValidation.cpp / HeckeBlockFormValidation.cpp
// build it.
// ===========================================================================
namespace {

struct PipelineFixture {
    LocalSystemSimplicialComplex lssc;
    EquivariantCohomologyOnK coh;
    hecke::SL2ZTessellationTables model;
    std::unique_ptr<hecke::HeckeOperatorFamily> family_ptr;

    PipelineFixture(const SimplicialComplex& K, uint32_t ell, uint32_t n_sym)
        : lssc(K, MakeRelations(K, n_sym, ell), n_sym + 1, ell),
          coh(&lssc),
          model(K) {
        // compute_cohomology() must run before HeckeOperatorFamily is
        // constructed (its ctor checks cohomL's dims); do this in the
        // constructor BODY (members are already initialized in declaration
        // order by this point) rather than via initializer-list trickery.
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

} // namespace

// ===========================================================================
// Shared reporting context (mirrors the Fail()/Ok() pattern used elsewhere
// under Hecke/, e.g. HeckeBlockFormValidation.cpp).
// ===========================================================================
namespace {

class SmokeContext {
public:
    uint32_t failures() const { return failures_; }
    void Fail(const std::string& msg) {
        ++failures_;
        std::cout << "HeckeSimultaneousSmokeValidation: FAIL: " << msg << std::endl;
    }
    void Ok(const std::string& msg) {
        std::cout << "HeckeSimultaneousSmokeValidation: OK: " << msg << std::endl;
    }

private:
    uint32_t failures_ = 0;
};

std::string JordanPartitionToString(const std::vector<uint32_t>& partition) {
    std::ostringstream oss;
    oss << "(";
    for (size_t t = 0; t < partition.size(); ++t) {
        if (t > 0) oss << ",";
        oss << partition[t];
    }
    oss << ")";
    return oss.str();
}

} // namespace

// ===========================================================================
// Scenario A: (ell, n_sym) = (13, 10), family {T_2, T_3}, degree q = 1.
// ===========================================================================
namespace {

void RunEll13(SmokeContext& ctx, std::string& out_json) {
    const uint32_t ell = 13, n_sym = 10;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        PipelineFixture fx(K, ell, n_sym);

        const auto& dims = fx.coh.get_H_eq_dimensions();
        if (dims.size() < 2 || dims[1] != 3) {
            ctx.Fail("(13,10): dim H^1 != 3 (got " +
                     (dims.size() >= 2 ? std::to_string(dims[1]) : std::string("<missing>")) + ")");
            return;
        }
        ctx.Ok("(13,10): dim H^1 = 3");

        const Matrix A2 = fx.family().T_p(2)[1];
        const Matrix A3 = fx.family().T_p(3)[1];
        if (A2.rows != 3 || A2.cols != 3 || A3.rows != 3 || A3.cols != 3) {
            ctx.Fail("(13,10): A^(1)(T_2) or A^(1)(T_3) has unexpected shape");
            return;
        }

        // Cross-check charpoly identities against the embedded (already
        // ell-reduced, per the task) eigenvalue data: T_2 -> {2,2,8},
        // T_3 -> {5,5,10} mod 13.
        using hecke::FpPoly;
        auto linear_pow = [&](uint32_t root, uint32_t mult) {
            const uint32_t c0 = (root == 0) ? 0u : (ell - root); // X - root
            return FpPoly::pow(FpPoly(ell, {c0, 1u}), mult);
        };
        const FpPoly expected_cp2 = linear_pow(2, 2) * linear_pow(8, 1);
        const FpPoly expected_cp3 = linear_pow(5, 2) * linear_pow(10, 1);
        const FpPoly cp2 = hecke::CharPolyBerkowitz(A2);
        const FpPoly cp3 = hecke::CharPolyBerkowitz(A3);
        if (cp2 != expected_cp2) {
            ctx.Fail("(13,10): charpoly(A^(1)(T_2)) != (X-2)^2(X-8) mod 13; got " + cp2.to_string());
            return;
        }
        if (cp3 != expected_cp3) {
            ctx.Fail("(13,10): charpoly(A^(1)(T_3)) != (X-5)^2(X-10) mod 13; got " + cp3.to_string());
            return;
        }
        ctx.Ok("(13,10): charpoly(A^(1)(T_2)) == (X-2)^2(X-8), charpoly(A^(1)(T_3)) == "
               "(X-5)^2(X-10) mod 13 (T_2 eigenvalues {-24=2,2,2049=8}, T_3 eigenvalues "
               "{252=5,5,177148=10} mod 13, as expected).");

        // The simultaneous decomposition under the family {T_2, T_3}.
        std::vector<std::pair<uint64_t, Matrix>> ops = { {2ull, A2}, {3ull, A3} };
        hecke::HeckeModuleDecomposition decomp(ops, ell);
        decomp.compute();
        if (decomp.certify() != 0) {
            ctx.Fail("(13,10): certify() != 0 (code " + std::to_string(decomp.certify()) + ")");
            return;
        }
        const auto& blocks = decomp.blocks();
        if (blocks.size() != 2) {
            ctx.Fail("(13,10): expected EXACTLY 2 blocks, got " + std::to_string(blocks.size()));
            return;
        }

        const hecke::PrimaryBlock* cusp = nullptr;
        const hecke::PrimaryBlock* eis = nullptr;
        for (const auto& b : blocks) {
            if (b.dim == 2) cusp = &b;
            else if (b.dim == 1) eis = &b;
        }
        if (cusp == nullptr || eis == nullptr) {
            ctx.Fail("(13,10): expected one dim-2 and one dim-1 block; got dims: " +
                     [&]{ std::string s; for (auto& b : blocks) s += std::to_string(b.dim) + " ";
                          return s; }());
            return;
        }
        ctx.Ok("(13,10): simultaneous decomposition = {dim 2, dim 1} as expected "
               "(cuspidal Delta system + Eisenstein).");

        // Eisenstein (dim-1) block: char_poly[2] = X-8, char_poly[3] = X-10,
        // residue degree 1, certified, eigen-system values asserted.
        if (eis->residue_degree != 1 || !eis->residue_degree_certified) {
            ctx.Fail("(13,10): Eisenstein block residue_degree != 1 (certified), got " +
                     std::to_string(eis->residue_degree) + " certified=" +
                     std::string(eis->residue_degree_certified ? "true" : "false"));
            return;
        }
        const FpPoly exp_eis2 = linear_pow(8, 1);
        const FpPoly exp_eis3 = linear_pow(10, 1);
        if (eis->char_poly.at(2) != exp_eis2 || eis->char_poly.at(3) != exp_eis3) {
            ctx.Fail("(13,10): Eisenstein block char_polys != {X-8, X-10}");
            return;
        }
        ctx.Ok("(13,10): Eisenstein block (dim 1): char_poly[2]=X-8, char_poly[3]=X-10, "
               "residue_degree=1 (certified) -- eigen-system T_2 |-> 8, T_3 |-> 10 confirmed.");

        // Cuspidal (dim-2) block: char_poly[2] = (X-2)^2, char_poly[3] = (X-5)^2;
        // Jordan partitions RECORDED (not asserted -- no operator in this
        // family separates the holomorphic/antiholomorphic pair; the
        // partition being (1,1) [semisimple, Delta !== Eis mod 13] or (2)
        // is not asserted here, matching the task's own risk note).
        if (cusp->char_poly.at(2) != linear_pow(2, 2) || cusp->char_poly.at(3) != linear_pow(5, 2)) {
            ctx.Fail("(13,10): cuspidal block char_polys != {(X-2)^2, (X-5)^2}");
            return;
        }
        std::cout << "HeckeSimultaneousSmokeValidation: RECORDED (13,10) cuspidal block "
                     "(dim 2, isotypic multiplicity 2 over the computed family {T_2, T_3} -- "
                     "holo/antiholo share the full eigensystem, so no T_p separates them): "
                  << "T_2 Jordan partition = " << JordanPartitionToString(cusp->jordan_partition.at(2))
                  << ", T_3 Jordan partition = " << JordanPartitionToString(cusp->jordan_partition.at(3))
                  << ", residue_degree = " << cusp->residue_degree
                  << " (certified=" << (cusp->residue_degree_certified ? "true" : "false") << ")"
                  << std::endl;
        ctx.Ok("(13,10): cuspidal block (dim 2): char_poly[2]=(X-2)^2, char_poly[3]=(X-5)^2 as "
               "expected; multiplicity-2 isotypic block over the computed family documented, "
               "not chased for a phantom further split.");

        // Build the HeckeReport JSON (level (b) + level (c); level (a) omitted).
        hecke::HeckeReportInput input;
        input.ell = ell;
        input.n_sym = n_sym;
        input.degree = 1;
        // (E0b, slot 2 WP-2) The oc1a fields are DERIVED from the complex's
        // own degree count, never written as a literal.
        input.num_degrees = static_cast<uint32_t>(fx.coh.get_H_eq_dimensions().size());
        input.level_b = { {2ull, A2}, {3ull, A3} };
        input.decomp = &decomp;
        out_json = hecke::BuildHeckeReportJson(input);
        std::cout << "HeckeSimultaneousSmokeValidation: sample HeckeReport JSON (13,10), "
                     "first 600 chars:\n"
                  << out_json.substr(0, 600) << (out_json.size() > 600 ? " ...\n" : "\n");

    } catch (const std::exception& e) {
        ctx.Fail(std::string("(13,10): EXCEPTION: ") + e.what());
    }
}

} // namespace

// ===========================================================================
// Scenario B: (ell, n_sym) = (691, 10), family {T_2, T_3}: a single dim-3
// block (the O6 result of Stage S10V), now re-derived under the two-operator
// family.
// ===========================================================================
namespace {

void RunEll691(SmokeContext& ctx) {
    const uint32_t ell = 691, n_sym = 10;
    try {
        const SimplicialComplex K = CreateSL2ZFundamentalDomainTriangulation();
        PipelineFixture fx(K, ell, n_sym);

        const auto& dims = fx.coh.get_H_eq_dimensions();
        if (dims.size() < 2 || dims[1] != 3) {
            ctx.Fail("(691,10): dim H^1 != 3 (got " +
                     (dims.size() >= 2 ? std::to_string(dims[1]) : std::string("<missing>")) + ")");
            return;
        }

        const Matrix A2 = fx.family().T_p(2)[1];
        const Matrix A3 = fx.family().T_p(3)[1];
        if (A2.rows != 3 || A2.cols != 3 || A3.rows != 3 || A3.cols != 3) {
            ctx.Fail("(691,10): A^(1)(T_2) or A^(1)(T_3) has unexpected shape");
            return;
        }

        std::vector<std::pair<uint64_t, Matrix>> ops = { {2ull, A2}, {3ull, A3} };
        hecke::HeckeModuleDecomposition decomp(ops, ell);
        decomp.compute();
        if (decomp.certify() != 0) {
            ctx.Fail("(691,10): certify() != 0 (code " + std::to_string(decomp.certify()) + ")");
            return;
        }
        const auto& blocks = decomp.blocks();
        if (blocks.size() != 1) {
            ctx.Fail("(691,10): expected EXACTLY ONE block under the family {T_2,T_3}, got " +
                     std::to_string(blocks.size()));
            return;
        }
        const hecke::PrimaryBlock& blk = blocks[0];
        if (blk.dim != 3) {
            ctx.Fail("(691,10): block dim != 3, got " + std::to_string(blk.dim));
            return;
        }
        ctx.Ok("(691,10): simultaneous decomposition under {T_2,T_3} = a single dim-3 block "
               "(the O6 Ramanujan-congruence result, now re-derived jointly with T_3).");

        // Cross-check the T_2 data against the already-validated S10V oracle
        // O6: char_poly = (X+24)^3, min_poly = (X+24)^2 (partition (2,1)).
        using hecke::FpPoly;
        // (X + 24): coeffs low->high = [24, 1] (constant term 24, since
        // 24 + X = X + 24; NOT ell-24, which would instead give X - 24).
        const uint32_t twenty_four = 24u % ell;
        const FpPoly expected_cp2 = FpPoly::pow(FpPoly(ell, {twenty_four, 1u}), 3);
        const FpPoly expected_mp2 = FpPoly::pow(FpPoly(ell, {twenty_four, 1u}), 2);
        if (blk.char_poly.at(2) != expected_cp2) {
            ctx.Fail("(691,10): block char_poly[2] != (X+24)^3");
            return;
        }
        if (blk.min_poly.at(2) != expected_mp2) {
            ctx.Fail("(691,10): block min_poly[2] != (X+24)^2 (partition should still be (2,1))");
            return;
        }
        ctx.Ok("(691,10): T_2 data unchanged from the S10V single-operator result: "
               "char_poly=(X+24)^3, min_poly=(X+24)^2 (Jordan partition (2,1)).");

        std::cout << "HeckeSimultaneousSmokeValidation: RECORDED (691,10) single block "
                     "T_3 data: char_poly = " << blk.char_poly.at(3).to_string()
                  << ", min_poly = " << blk.min_poly.at(3).to_string()
                  << ", Jordan partition = " << JordanPartitionToString(blk.jordan_partition.at(3))
                  << ", residue_degree = " << blk.residue_degree
                  << " (certified=" << (blk.residue_degree_certified ? "true" : "false") << ")"
                  << std::endl;

    } catch (const std::exception& e) {
        ctx.Fail(std::string("(691,10): EXCEPTION: ") + e.what());
    }
}

} // namespace

// ===========================================================================
// JSON round-trip check (embedded mini-parser; a representative sample of
// fields, not an exhaustive schema walk).
// ===========================================================================
namespace {

void CheckJsonRoundTrip(SmokeContext& ctx, const std::string& json) {
    if (json.empty()) {
        ctx.Fail("JSON round-trip: no JSON was produced (an earlier step must have failed)");
        return;
    }
    try {
        MiniJsonParser parser(json);
        const JsonValue root = parser.Parse();

        if (!root.is_object()) { ctx.Fail("JSON round-trip: root is not an object"); return; }
        if (root.at("ell").as_int() != 13) { ctx.Fail("JSON round-trip: ell != 13"); return; }
        if (root.at("n_sym").as_int() != 10) { ctx.Fail("JSON round-trip: n_sym != 10"); return; }
        if (root.at("degree").as_int() != 1) { ctx.Fail("JSON round-trip: degree != 1"); return; }

        const JsonValue& level_c = root.at("level_c");
        if (level_c.at("cohomology_dim").as_int() != 3) {
            ctx.Fail("JSON round-trip: level_c.cohomology_dim != 3");
            return;
        }
        if (!level_c.at("certified").as_bool()) {
            ctx.Fail("JSON round-trip: level_c.certified != true");
            return;
        }
        const std::vector<JsonValue>& blk_arr = level_c.at("blocks").items();
        if (blk_arr.size() != 2) {
            ctx.Fail("JSON round-trip: level_c.blocks does not have exactly 2 entries");
            return;
        }

        bool saw_dim2 = false, saw_dim1_eigen = false;
        for (const JsonValue& b : blk_arr) {
            const long long dim = b.at("dim").as_int();
            if (dim == 2) {
                saw_dim2 = true;
                if (!b.has("residue_degree")) {
                    ctx.Fail("JSON round-trip: dim-2 block missing 'residue_degree' "
                             "(certified value expected)");
                    return;
                }
            } else if (dim == 1) {
                if (!b.has("eigen_system")) {
                    ctx.Fail("JSON round-trip: dim-1 block missing 'eigen_system'");
                    return;
                }
                const JsonValue& eig = b.at("eigen_system");
                if (eig.at("2").as_int() != 8 || eig.at("3").as_int() != 10) {
                    ctx.Fail("JSON round-trip: dim-1 block eigen_system != {2:8, 3:10}");
                    return;
                }
                saw_dim1_eigen = true;
            } else {
                ctx.Fail("JSON round-trip: unexpected block dim " + std::to_string(dim));
                return;
            }
        }
        if (!saw_dim2 || !saw_dim1_eigen) {
            ctx.Fail("JSON round-trip: did not find both the dim-2 and dim-1(+eigen_system) blocks");
            return;
        }

        const JsonValue& level_b = root.at("level_b");
        if (!level_b.has("2") || !level_b.has("3")) {
            ctx.Fail("JSON round-trip: level_b missing an entry for p=2 or p=3");
            return;
        }

        ctx.Ok("JSON round-trip through the embedded mini-parser: ell/n_sym/degree, "
               "level_c.cohomology_dim/certified, both block dims, the dim-1 block's "
               "eigen_system {2:8,3:10}, and level_b's operator keys all confirmed.");
    } catch (const std::exception& e) {
        ctx.Fail(std::string("JSON round-trip: EXCEPTION: ") + e.what());
    }
}

} // namespace

uint32_t HeckeSimultaneousSmokeValidation() {
    SmokeContext ctx;
    std::string json_13_10;
    RunEll13(ctx, json_13_10);
    CheckJsonRoundTrip(ctx, json_13_10);
    RunEll691(ctx);

    std::cout << "HeckeSimultaneousSmokeValidation: " << ctx.failures() << " failing case(s)."
              << std::endl;
    return ctx.failures();
}
