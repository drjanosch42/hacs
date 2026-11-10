// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// The sign split of the Hecke module -- phase E3, hecke-extensions slot 4 WP-3.
// See HeckeSignSplit.h for what this discharges and why it takes matrices
// rather than a pipeline object.

#include "HeckeSignSplit.h"

#include "../Hecke/SubspaceOps.h"
#include "../../ModArithmetic/EngineCheck.h"
#include "../../ModArithmetic/Fnv.h"
#include "../../ModArithmetic/ModArithmetic.h"

#include <algorithm>
#include <sstream>

namespace hecke_ext {

namespace {

uint64_t hash_matrix(const ExtMatrix& M) {
    uint64_t h = engine::kFnvOffset64;
    h = engine::FnvU64(h, M.rows);
    h = engine::FnvU64(h, M.cols);
    for (uint32_t r = 0; r < M.rows; ++r)
        for (uint32_t c = 0; c < M.cols; ++c) h = engine::FnvU64(h, static_cast<uint64_t>(M.data[r][c]));
    return h;
}

uint64_t hash_scalar(uint64_t v) { return engine::FnvU64(engine::kFnvOffset64, v); }

uint64_t hash_poly(const hecke::FpPoly& P) {
    uint64_t h = engine::FnvU64(engine::kFnvOffset64, P.coeffs.size());
    for (uint32_t c : P.coeffs) h = engine::FnvU64(h, static_cast<uint64_t>(c));
    return h;
}

ExtMatrix hstack_cols(const std::vector<ExtMatrix>& parts, uint32_t rows, uint32_t ell) {
    uint32_t total = 0;
    for (const auto& p : parts) total += p.cols;
    ExtMatrix out(rows, total, ell);
    uint32_t off = 0;
    for (const auto& p : parts) {
        ENGINE_CHECK(p.rows == rows);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < p.cols; ++c) out.data[r][off + c] = p.data[r][c];
        off += p.cols;
    }
    return out;
}

ExtMatrix vstack_rows(const std::vector<ExtMatrix>& parts, uint32_t cols, uint32_t ell) {
    uint32_t total = 0;
    for (const auto& p : parts) total += p.rows;
    ExtMatrix out(total, cols, ell);
    uint32_t off = 0;
    for (const auto& p : parts) {
        ENGINE_CHECK(p.cols == cols);
        for (uint32_t r = 0; r < p.rows; ++r)
            for (uint32_t c = 0; c < cols; ++c) out.data[off + r][c] = p.data[r][c];
        off += p.rows;
    }
    return out;
}

// span(A) == span(B) for two matrices with the same number of rows.
bool same_span(const ExtMatrix& A, const ExtMatrix& B, uint32_t rows, uint32_t ell) {
    const uint32_t ra = (A.cols == 0u) ? 0u : hecke::RankOf(A);
    const uint32_t rb = (B.cols == 0u) ? 0u : hecke::RankOf(B);
    if (ra != rb) return false;
    if (ra == 0u) return true;
    const ExtMatrix both = hstack_cols({ A, B }, rows, ell);
    return hecke::RankOf(both) == ra;
}

} // namespace

const char* ToString(SignSide s) { return (s == SignSide::Plus) ? "+" : "-"; }

const char* ToString(SignSplitStatus s) {
    switch (s) {
        case SignSplitStatus::Split: return "split";
        case SignSplitStatus::NotApplicable: return "not_applicable";
    }
    return "?";
}

// ---------------------------------------------------------------------------

HeckeSignSplit::HeckeSignSplit(uint32_t ell, uint32_t n_sym, uint32_t degree,
                               ExtMatrix iota, ExtMatrix plus_basis, ExtMatrix minus_basis,
                               std::vector<std::pair<uint64_t, ExtMatrix>> ops)
    : ell_(ell), n_sym_(n_sym), degree_(degree),
      iota_(std::move(iota)), plus_(std::move(plus_basis)), minus_(std::move(minus_basis)),
      input_ops_(std::move(ops)) {
    ENGINE_CHECK_MSG(ell_ >= 2u, "HeckeSignSplit: ell must be a prime.");
    ENGINE_CHECK_MSG(iota_.rows == iota_.cols, "HeckeSignSplit: iota^* must be square.");
    ENGINE_CHECK_MSG(iota_.rows == 0u || iota_.p == ell_, "HeckeSignSplit: iota^* has the wrong modulus.");
    dim_H_ = iota_.rows;
    for (const auto& kv : input_ops_) {
        ENGINE_CHECK_MSG(kv.second.rows == dim_H_ && kv.second.cols == dim_H_,
                         "HeckeSignSplit: an operator has the wrong shape.");
        ENGINE_CHECK_MSG(dim_H_ == 0u || kv.second.p == ell_,
                         "HeckeSignSplit: an operator has the wrong modulus.");
    }
    status_ = (ell_ == 2u) ? SignSplitStatus::NotApplicable : SignSplitStatus::Split;
    if (status_ == SignSplitStatus::Split && dim_H_ != 0u) {
        ENGINE_CHECK_MSG(plus_.rows == dim_H_ && minus_.rows == dim_H_,
                         "HeckeSignSplit: an eigenbasis has the wrong number of rows.");
        ENGINE_CHECK_MSG(plus_.cols + minus_.cols == dim_H_,
                         "HeckeSignSplit: the two eigenbases do not exhaust H^q.");
    }
    std::stable_sort(input_ops_.begin(), input_ops_.end(),
                     [](const std::pair<uint64_t, ExtMatrix>& a, const std::pair<uint64_t, ExtMatrix>& b) {
                         return a.first < b.first;
                     });
}

HeckeSignSplit::~HeckeSignSplit() = default;

uint32_t HeckeSignSplit::dim_plus() const {
    return (status_ == SignSplitStatus::Split) ? plus_.cols : 0u;
}
uint32_t HeckeSignSplit::dim_minus() const {
    return (status_ == SignSplitStatus::Split) ? minus_.cols : 0u;
}

void HeckeSignSplit::restrict_operators() {
    ops_.clear();
    ops_.reserve(input_ops_.size());
    const bool assert_commutation = (ell_ >= 5u);
    cert_.commutation_asserted = assert_commutation;
    for (const auto& kv : input_ops_) {
        SignSplitOperator o;
        o.p = kv.first;
        o.nu = 1u;
        o.A = kv.second;
        if (dim_H_ == 0u) { o.commutes = true; ops_.push_back(std::move(o)); continue; }
        // [C, obl:iota](v): [iota^*, A^{(q)}] = 0, exactly.
        o.commutes = ((iota_ * o.A) == (o.A * iota_));
        if (assert_commutation)
            ENGINE_CHECK_MSG(o.commutes,
                             "[C, obl:iota](v): iota^* does not commute with a computed Hecke operator at "
                             "ell >= 5, where [C, prop:iotacommutes](3c) proves that it must.");
        if (o.commutes) ++cert_.num_commuting;
        // The restriction is formed ONLY where the commutation holds exactly:
        // at ell = 3 the commutation is recorded, not proved, and a
        // non-commuting operator has no restriction to the eigenspaces.
        if (o.commutes && status_ == SignSplitStatus::Split) {
            if (plus_.cols != 0u)
                o.A_plus = hecke::SolveColumnsExact(plus_, o.A * plus_, "HeckeSignSplit: A|_{H^q_+}");
            else
                o.A_plus = ExtMatrix(0, 0, ell_);
            if (minus_.cols != 0u)
                o.A_minus = hecke::SolveColumnsExact(minus_, o.A * minus_, "HeckeSignSplit: A|_{H^q_-}");
            else
                o.A_minus = ExtMatrix(0, 0, ell_);
            o.restricted = true;
        }
        ops_.push_back(std::move(o));
    }
}

void HeckeSignSplit::decompose() {
    auto build = [&](const std::vector<std::pair<uint64_t, ExtMatrix>>& fam, uint32_t d)
        -> std::unique_ptr<hecke::HeckeModuleDecomposition> {
        if (d == 0u || fam.empty()) return nullptr;
        auto dec = std::make_unique<hecke::HeckeModuleDecomposition>(fam, ell_);
        dec->compute();
        return dec;
    };

    std::vector<std::pair<uint64_t, ExtMatrix>> fam_unsplit, fam_plus, fam_minus;
    for (const auto& o : ops_) {
        fam_unsplit.emplace_back(o.p, o.A);
        if (o.restricted) {
            fam_plus.emplace_back(o.p, o.A_plus);
            fam_minus.emplace_back(o.p, o.A_minus);
        }
    }
    dec_unsplit_ = build(fam_unsplit, dim_H_);
    if (status_ == SignSplitStatus::Split) {
        dec_plus_ = build(fam_plus, dim_plus());
        dec_minus_ = build(fam_minus, dim_minus());
    }

    auto dims_of = [](const hecke::HeckeModuleDecomposition* d) {
        std::vector<uint32_t> out;
        if (d) for (const auto& b : d->blocks()) out.push_back(b.dim);
        return out;
    };
    cert_.unsplit_block_dims = dims_of(dec_unsplit_.get());
    cert_.plus_block_dims = dims_of(dec_plus_.get());
    cert_.minus_block_dims = dims_of(dec_minus_.get());

    // Cyclicity per sign, through slot-2 WP-4's generation certificate (a).
    auto cyclic_flags = [&](const hecke::HeckeModuleDecomposition* dec,
                            const std::vector<std::pair<uint64_t, ExtMatrix>>& fam) {
        std::vector<uint8_t> out;
        if (!dec) return out;
        for (const auto& blk : dec->blocks()) {
            const LocalAlgebra L = ComputeLocalAlgebra(blk, fam, ell_);
            out.push_back(static_cast<uint8_t>(L.ok && L.cyclicity.cyclic ? 1 : 0));
        }
        return out;
    };
    cert_.plus_block_cyclic = cyclic_flags(dec_plus_.get(), fam_plus);
    cert_.minus_block_cyclic = cyclic_flags(dec_minus_.get(), fam_minus);
}

void HeckeSignSplit::find_eisenstein_line() {
    // [E] D6 / [X] Sec. 6.2 E3(b): the Eisenstein line at m is
    // soc(X^+_{4,m}) = m_S X^+_{4,m}, DEFINED when that space has
    // F-dimension 1 (F = T_m/m) and "undefined" otherwise.
    eis_ = EisensteinLine();
    eis_.status = "not_applicable";
    if (status_ != SignSplitStatus::Split || dec_plus_ == nullptr || dim_plus() == 0u) return;

    std::vector<std::pair<uint64_t, ExtMatrix>> fam_plus;
    for (const auto& o : ops_) if (o.restricted) fam_plus.emplace_back(o.p, o.A_plus);
    if (fam_plus.empty()) return;

    eis_.applicable = true;
    // The Eisenstein block is the one whose socle is smallest over F; ties are
    // broken by the block's position, which HeckeModuleDecomposition fixes
    // canonically (CanonicalLess order).  On every cell this document computes
    // the plus side has one block carrying the Eisenstein class, and the choice
    // is recorded either way.
    bool have = false;
    for (uint32_t bi = 0; bi < dec_plus_->blocks().size(); ++bi) {
        const hecke::PrimaryBlock& blk = dec_plus_->blocks()[bi];
        const LocalAlgebra L = ComputeLocalAlgebra(blk, fam_plus, ell_);
        if (!L.ok || L.n.empty()) continue;
        const uint32_t D = L.D;
        // soc(V) = Ann_V(m_S) = intersect_p ker n_p  =  ker of the stacked n_p.
        const ExtMatrix stacked = vstack_rows(L.n, D, ell_);
        const ExtMatrix soc = hecke::KernelAsMatrix(stacked);      // D x dim soc
        // m_S V = sum_p im n_p: the column span of the n_p side by side.
        const ExtMatrix mv = hstack_cols(L.n, D, ell_);
        const bool eq = same_span(soc, mv, D, ell_);
        const uint32_t dim_soc = soc.cols;
        const uint32_t f = (L.f == 0u) ? 1u : L.f;
        if (!have || dim_soc / f < eis_.dim_socle_over_F) {
            have = true;
            eis_.block_index = bi;
            eis_.f = f;
            eis_.dim_socle = dim_soc;
            eis_.dim_socle_over_F = dim_soc / f;
            eis_.socle_equals_mV = eq;
            eis_.socle_basis = soc;
            // Back into the stored H^q basis: H^q_+ basis columns times the
            // block's own basis columns times the socle basis.
            const ExtMatrix block_in_plus = blk.basis_columns;     // dim_+ x dim blk
            eis_.line_in_H = plus_ * (block_in_plus * soc);
        }
    }
    if (!have) { eis_.applicable = false; return; }
    eis_.defined = (eis_.dim_socle_over_F == 1u);
    eis_.status = eis_.defined ? "defined" : "undefined";
}

void HeckeSignSplit::compute() {
    if (computed_) return;
    cert_ = SignSplitCertificate();
    cert_.ell = ell_;
    cert_.n_sym = n_sym_;
    cert_.degree = degree_;
    cert_.dim_H = dim_H_;
    cert_.status = status_;
    cert_.iota = iota_;
    cert_.plus_basis = (status_ == SignSplitStatus::Split) ? plus_ : ExtMatrix(0, 0, ell_);
    cert_.minus_basis = (status_ == SignSplitStatus::Split) ? minus_ : ExtMatrix(0, 0, ell_);
    cert_.dim_plus = dim_plus();
    cert_.dim_minus = dim_minus();

    restrict_operators();
    decompose();
    find_eisenstein_line();

    cert_.ops = ops_;
    cert_.eisenstein = eis_;

    std::vector<std::string> failures;
    const uint32_t code = cert_.verify(failures);
    if (code != 0u) {
        std::ostringstream os;
        os << "HeckeSignSplit: the certificate failed with code " << code << " ("
           << SignSplitCertificate::verify_message(code) << ")";
        for (const auto& f : failures) os << "\n  " << f;
        ENGINE_CHECK_MSG(false, os.str().c_str());
    }
    computed_ = true;
}

size_t HeckeSignSplit::num_operators() const { return ops_.size(); }

const SignSplitOperator& HeckeSignSplit::op(size_t i) const {
    ENGINE_CHECK_MSG(i < ops_.size(), "HeckeSignSplit::op: index out of range.");
    return ops_[i];
}

const SignSplitOperator& HeckeSignSplit::op(uint64_t p, uint32_t nu) const {
    for (const auto& o : ops_) if (o.p == p && o.nu == nu) return o;
    ENGINE_CHECK_MSG(false, "HeckeSignSplit::op: no such operator.");
    return ops_[0];
}

const hecke::HeckeModuleDecomposition* HeckeSignSplit::decomposition(SignSide s) const {
    return (s == SignSide::Plus) ? dec_plus_.get() : dec_minus_.get();
}
const hecke::HeckeModuleDecomposition* HeckeSignSplit::decomposition_unsplit() const {
    return dec_unsplit_.get();
}

const EisensteinLine& HeckeSignSplit::eisenstein_line() const {
    ENGINE_CHECK_MSG(computed_, "HeckeSignSplit::eisenstein_line: compute() has not run.");
    return eis_;
}

const SignSplitCertificate& HeckeSignSplit::certificate() const {
    ENGINE_CHECK_MSG(computed_, "HeckeSignSplit::certificate: compute() has not run.");
    return cert_;
}

std::vector<GoldenObject> HeckeSignSplit::golden_objects() const {
    // DEFINED, never emitted.  See the header.
    ENGINE_CHECK_MSG(computed_, "HeckeSignSplit::golden_objects: compute() has not run.");
    std::vector<GoldenObject> out;
    const std::string q = "^" + std::to_string(degree_);
    out.push_back(GoldenObject{ "rank_iota" + q, 'A', dim_H_, 0,
                                hash_scalar(dim_H_ == 0u ? 0u : hecke::RankOf(iota_)) });
    out.push_back(GoldenObject{ "charpoly_iota" + q, 'A', 0, 0,
                                dim_H_ == 0u ? hash_scalar(0)
                                             : hash_poly(hecke::CharPolyBerkowitz(iota_)) });
    out.push_back(GoldenObject{ "dim_Hplus" + q, 'A', dim_plus(), 0, hash_scalar(dim_plus()) });
    out.push_back(GoldenObject{ "dim_Hminus" + q, 'A', dim_minus(), 0, hash_scalar(dim_minus()) });
    out.push_back(GoldenObject{ "eisenstein_line_defined", 'A', eis_.defined ? 1u : 0u, 0,
                                hash_scalar(eis_.defined ? 1u : 0u) });
    out.push_back(GoldenObject{ "iota" + q, 'B', iota_.rows, iota_.cols, hash_matrix(iota_) });
    return out;
}

uint64_t HeckeSignSplit::golden_digest() const {
    uint64_t h = engine::kFnvOffset64;
    for (const auto& g : golden_objects()) {
        h = engine::FnvBytes(h, g.object);
        h = engine::FnvU64(h, static_cast<uint64_t>(g.tier));
        h = engine::FnvU64(h, g.rows);
        h = engine::FnvU64(h, g.cols);
        h = engine::FnvU64(h, g.hash);
    }
    return h;
}

std::string HeckeSignSplit::to_string() const { return to_string_offset(0); }

std::string HeckeSignSplit::to_string_offset(uint32_t offset) const {
    const std::string pad(offset, ' ');
    std::ostringstream os;
    os << pad << "HeckeSignSplit (E3): ell = " << ell_ << ", n = " << n_sym_ << ", q = " << degree_
       << ", dim H^q = " << dim_H_ << ", split = " << ToString(status_) << "\n";
    if (!computed_) { os << pad << "  (not computed)\n"; return os.str(); }
    os << pad << "  (dim H^q_+, dim H^q_-) = (" << dim_plus() << ", " << dim_minus() << ")\n";
    for (const auto& o : ops_)
        os << pad << "  T_" << o.p << ": [iota, A] = 0 -> " << (o.commutes ? "yes" : "NO")
           << ", restricted -> " << (o.restricted ? "yes" : "no") << "\n";
    os << pad << "  Eisenstein line: " << eis_.status;
    if (eis_.applicable)
        os << " (block " << eis_.block_index << ", f = " << eis_.f << ", dim_F soc = "
           << eis_.dim_socle_over_F << ", soc == m V: " << (eis_.socle_equals_mV ? "yes" : "no") << ")";
    os << "\n";
    return os.str();
}

std::string HeckeSignSplit::to_json() const {
    std::ostringstream os;
    os << "{\"schema\": \"hecke_sign_split/v1\", \"ell\": " << ell_ << ", \"n_sym\": " << n_sym_
       << ", \"degree\": " << degree_ << ", \"dim_H\": " << dim_H_
       << ", \"sign_split\": \"" << ToString(status_) << "\""
       << ", \"iota_sign\": \"eisenstein_in_H1_plus\""
       << ", \"dim_H_plus\": " << dim_plus() << ", \"dim_H_minus\": " << dim_minus();
    os << ", \"operators\": [";
    for (size_t i = 0; i < ops_.size(); ++i)
        os << (i ? ", " : "") << "{\"p\": " << ops_[i].p << ", \"nu\": " << ops_[i].nu
           << ", \"commutes\": " << (ops_[i].commutes ? "true" : "false")
           << ", \"restricted\": " << (ops_[i].restricted ? "true" : "false") << "}";
    os << "], \"eisenstein\": {\"status\": \"" << eis_.status << "\", \"defined\": "
       << (eis_.defined ? "true" : "false") << ", \"dim_socle_over_F\": " << eis_.dim_socle_over_F
       << ", \"socle_equals_mV\": " << (eis_.socle_equals_mV ? "true" : "false") << "}";
    auto dims = [&](const char* key, const std::vector<uint32_t>& v) {
        os << ", \"" << key << "\": [";
        for (size_t i = 0; i < v.size(); ++i) os << (i ? ", " : "") << v[i];
        os << "]";
    };
    dims("block_dims_unsplit", cert_.unsplit_block_dims);
    dims("block_dims_plus", cert_.plus_block_dims);
    dims("block_dims_minus", cert_.minus_block_dims);
    os << "}";
    return os.str();
}

// ---------------------------------------------------------------------------
// SignSplitCertificate::verify -- re-derives, never reads.
// ---------------------------------------------------------------------------

const char* SignSplitCertificate::verify_message(uint32_t code) {
    switch (code) {
        case 0: return "ok";
        case 1: return "shape: ell / dimensions inconsistent";
        case 2: return "the split status disagrees with 2 in R^x";
        case 3: return "[C, prop:iotasplit](1): the eigenbases are not the +-1 eigenspaces, or do not exhaust H^q";
        case 4: return "[C, obl:iota](v): a recorded commutation verdict is wrong";
        case 5: return "[C, obl:iota](v): an operator does not commute with iota^* at ell >= 5";
        case 6: return "the restriction A_+- does not satisfy A U = U A_+-";
        case 7: return "the per-sign block dimensions do not sum to dim H^q_+-";
        case 8: return "the Eisenstein line's recorded dimension or status is wrong";
        default: return "unknown code";
    }
}

std::vector<uint32_t> SignSplitCertificate::failing_items() const {
    std::vector<std::string> ignored;
    std::vector<uint32_t> out;
    const uint32_t first = verify(ignored);
    if (first != 0u) out.push_back(first);
    return out;
}

uint32_t SignSplitCertificate::verify() const {
    std::vector<std::string> ignored;
    return verify(ignored);
}

uint32_t SignSplitCertificate::verify(std::vector<std::string>& failures) const {
    failures.clear();
    auto fail = [&](uint32_t code, const std::string& what) {
        failures.push_back(std::to_string(code) + ": " + what);
        return code;
    };

    if (ell < 2u) return fail(1, "ell");
    if (iota.rows != iota.cols || iota.rows != dim_H) return fail(1, "iota^* shape");
    if (status != ((ell == 2u) ? SignSplitStatus::NotApplicable : SignSplitStatus::Split))
        return fail(2, "status vs 2 in R^x");

    if (status == SignSplitStatus::Split && dim_H != 0u) {
        if (plus_basis.rows != dim_H || minus_basis.rows != dim_H) return fail(3, "eigenbasis rows");
        if (plus_basis.cols != dim_plus || minus_basis.cols != dim_minus) return fail(3, "eigenbasis cols");
        if (dim_plus + dim_minus != dim_H) return fail(3, "dim H_+ + dim H_- != dim H");
        // RE-DERIVED: the eigenvalue equations and independence.
        if (dim_plus != 0u && !((iota * plus_basis) == plus_basis))
            return fail(3, "H^q_+ is not the (+1)-eigenspace");
        if (dim_minus != 0u && !((iota * minus_basis) == (minus_basis * (ell - 1u))))
            return fail(3, "H^q_- is not the (-1)-eigenspace");
        const ExtMatrix both = hstack_cols({ plus_basis, minus_basis }, dim_H, ell);
        if (hecke::RankOf(both) != dim_H) return fail(3, "the eigenbases are not independent");
    } else if (status == SignSplitStatus::NotApplicable) {
        if (plus_basis.cols != 0u || minus_basis.cols != 0u)
            return fail(2, "an eigenbasis was formed at ell = 2");
        if (dim_plus != 0u || dim_minus != 0u) return fail(2, "a nonzero +- dimension at ell = 2");
    }

    uint32_t commuting = 0;
    for (const auto& o : ops) {
        if (o.A.rows != dim_H || o.A.cols != dim_H) return fail(1, "operator shape");
        const bool c = (dim_H == 0u) ? true : ((iota * o.A) == (o.A * iota));
        if (c != o.commutes) return fail(4, "recorded commutation verdict");
        if (c) ++commuting;
        if (commutation_asserted && !c) return fail(5, "non-commuting operator at ell >= 5");
        if (o.restricted) {
            if (status != SignSplitStatus::Split) return fail(6, "restricted at ell = 2");
            if (!c) return fail(6, "restricted a non-commuting operator");
            if (dim_plus != 0u && !((o.A * plus_basis) == (plus_basis * o.A_plus)))
                return fail(6, "A U_+ != U_+ A_+");
            if (dim_minus != 0u && !((o.A * minus_basis) == (minus_basis * o.A_minus)))
                return fail(6, "A U_- != U_- A_-");
        }
    }
    if (commuting != num_commuting) return fail(4, "num_commuting");

    auto sum = [](const std::vector<uint32_t>& v) {
        uint32_t s = 0;
        for (uint32_t x : v) s += x;
        return s;
    };
    if (!plus_block_dims.empty() && sum(plus_block_dims) != dim_plus) return fail(7, "plus blocks");
    if (!minus_block_dims.empty() && sum(minus_block_dims) != dim_minus) return fail(7, "minus blocks");
    if (!unsplit_block_dims.empty() && sum(unsplit_block_dims) != dim_H) return fail(7, "unsplit blocks");
    if (plus_block_cyclic.size() != plus_block_dims.size()) return fail(7, "plus cyclic flags");
    if (minus_block_cyclic.size() != minus_block_dims.size()) return fail(7, "minus cyclic flags");

    if (eisenstein.applicable) {
        if (eisenstein.f == 0u) return fail(8, "residue degree");
        if (eisenstein.dim_socle_over_F * eisenstein.f != eisenstein.dim_socle)
            return fail(8, "dim_F soc * f != dim soc");
        if (eisenstein.defined != (eisenstein.dim_socle_over_F == 1u)) return fail(8, "defined verdict");
        const std::string want = eisenstein.defined ? "defined" : "undefined";
        if (eisenstein.status != want) return fail(8, "status string");
        if (eisenstein.socle_basis.cols != eisenstein.dim_socle) return fail(8, "socle basis width");
        if (eisenstein.line_in_H.rows != dim_H || eisenstein.line_in_H.cols != eisenstein.dim_socle)
            return fail(8, "line_in_H shape");
    } else if (eisenstein.status != "not_applicable") {
        return fail(8, "status string on a non-applicable cell");
    }
    return 0u;
}

} // namespace hecke_ext
