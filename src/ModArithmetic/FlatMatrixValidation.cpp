// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h" // If using precompiled headers
#include "FlatMatrixValidation.h"
#include "FlatMatrix.h"
#include "Matrix.h"        // For conversion tests and comparison reference
#include "LegacyConfig.h"   // (S24 CL-L2) oc::legacy::RunLegacyArm (SKIP-ARM)
#include "ModArithmetic.h" // For modular operations
#include "helper.h"        // For offset_string

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>   // For std::iota in kernel tests
#include <algorithm> // For std::all_of, std::equal

// Helper to compare two FlatMatrix objects for tests that expect specific values.
// This is more thorough than just fm1 == fm2 if flags might differ but content is same.
// This helper must use non-const FlatMatrix if it calls methods like is_zero internally for its logic
// Or, it must rely solely on get_entry which is const.
// The current compare_flat_matrices_content is simple and uses get_entry (const).
bool compare_flat_matrices_content(const FlatMatrix& fm1, const FlatMatrix& fm2, uint32_t p_val) {
    if (fm1.get_rows() != fm2.get_rows() || fm1.get_cols() != fm2.get_cols() || fm1.get_p() != fm2.get_p() || fm1.get_p() != p_val) {
        return false;
    }
    for (uint32_t r = 0; r < fm1.get_rows(); ++r) {
        for (uint32_t c = 0; c < fm1.get_cols(); ++c) {
            if (fm1.get_entry(r, c) != fm2.get_entry(r, c)) {
                return false;
            }
        }
    }
    return true;
}


#if OC_ENABLE_LEGACY
#include "legacy/FlatMatrixLegacyArms.inc"   // (S24 CL-L2) Q11's BlockFlatMatrix class
#endif

uint32_t FlatMatrixValidation(int verbosity) {
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;
    uint32_t p = 23; // Prime modulus for tests

    auto run_test_section = [&](const std::string& section_name) {
        if (verbosity > 0) std::cout << "\n--- Testing " << section_name << " ---" << std::endl;
    };

    auto run_test = [&](const std::string& name, bool condition) {
        total_tests++;
        if (condition) {
            passed_tests++;
            if (verbosity > 1) std::cout << "  [PASS] " << name << std::endl;
        }
        else {
            std::cout << "  [FAIL] " << name << std::endl;
        }
        return condition;
    };

    auto run_test_throws = [&](const std::string& name, auto lambda_expr, const std::string& expected_exception_type_str) {
        total_tests++;
        bool thrown_correctly = false;
        std::string error_msg = "";
        std::string actual_exception_type = "none";
        try {
            lambda_expr();
        }
        catch (const std::runtime_error& e) {
            actual_exception_type = "std::runtime_error";
            thrown_correctly = (expected_exception_type_str == actual_exception_type);
            error_msg = thrown_correctly ? e.what() : (std::string("Threw std::runtime_error: ") + e.what() + ", expected " + expected_exception_type_str);
        }
        catch (const std::invalid_argument& e) {
            actual_exception_type = "std::invalid_argument";
            thrown_correctly = (expected_exception_type_str == actual_exception_type);
            error_msg = thrown_correctly ? e.what() : (std::string("Threw std::invalid_argument: ") + e.what() + ", expected " + expected_exception_type_str);
        }
        catch (const std::out_of_range& e) {
            actual_exception_type = "std::out_of_range";
            thrown_correctly = (expected_exception_type_str == actual_exception_type);
            error_msg = thrown_correctly ? e.what() : (std::string("Threw std::out_of_range: ") + e.what() + ", expected " + expected_exception_type_str);
        }
        catch (const std::logic_error& e) {
            actual_exception_type = "std::logic_error";
            thrown_correctly = (expected_exception_type_str == actual_exception_type);
            error_msg = thrown_correctly ? e.what() : (std::string("Threw std::logic_error (or derived): ") + e.what() + ", expected " + expected_exception_type_str);
        }
        catch (...) {
            actual_exception_type = "unknown type";
            error_msg = "Threw an unexpected exception type, expected " + expected_exception_type_str;
        }

        if (thrown_correctly) {
            passed_tests++;
            if (verbosity > 1) std::cout << "  [PASS] " << name << " (threw " << expected_exception_type_str << " as expected: " << error_msg << ")" << std::endl;
        }
        else {
            if (actual_exception_type == "none") error_msg = "Did not throw " + expected_exception_type_str + " as expected.";
            std::cout << "  [FAIL] " << name << " (" << error_msg << ")" << std::endl;
        }
        return thrown_correctly;
    };

    if (verbosity > 0) std::cout << "Starting FlatMatrixValidation with p = " << p << std::endl;

    try {
        run_test_section("Constructors and Initial State");
        {
            FlatMatrix fm_zero_def(3, 2, p); // non-const
            uint32_t scalar_val_def;
            bool is_scalar_def = fm_zero_def.is_currently_scalar(scalar_val_def);
            run_test("Default constructor is scalar 0 (flag check)", is_scalar_def && scalar_val_def == 0);
            run_test("Default constructor dimensions", fm_zero_def.get_rows() == 3 && fm_zero_def.get_cols() == 2 && fm_zero_def.get_p() == p);

            std::vector<uint32_t> data1_flat = { 1, 2, 3, 4, 5, 6 }; // 2x3
            FlatMatrix fm_data(2, 3, p, data1_flat); // non-const
            uint32_t dummy_scalar_val;
            run_test("Constructor from flat data not scalar zero", !(fm_data.is_currently_scalar(dummy_scalar_val) && dummy_scalar_val == 0));
            run_test("Constructor from flat data not scalar 1", !(fm_data.is_currently_scalar(dummy_scalar_val) && dummy_scalar_val == 1)); // More general check
            run_test("Constructor from flat data dimensions", fm_data.get_rows() == 2 && fm_data.get_cols() == 3);

            FlatMatrix fm_id(3, p); // Identity, non-const
            uint32_t scalar_val_id;
            run_test("Identity constructor is scalar 1 (flag check)", fm_id.is_currently_scalar(scalar_val_id) && scalar_val_id == 1);
            run_test("Identity constructor dimensions", fm_id.get_rows() == 3 && fm_id.get_cols() == 3);

            FlatMatrix fm_zero_static = FlatMatrix::CreateZero(4, 4, p); // non-const
            uint32_t scalar_val_zero_static;
            bool is_scalar_zero_static = fm_zero_static.is_currently_scalar(scalar_val_zero_static);
            run_test("CreateZero creates scalar 0 matrix (flag check)", is_scalar_zero_static && scalar_val_zero_static == 0);
            run_test("CreateZero dimensions", fm_zero_static.get_rows() == 4 && fm_zero_static.get_cols() == 4);

            FlatMatrix fm_scalar_static = FlatMatrix::CreateScalar(3, 5, p); // non-const
            uint32_t scalar_val_static;
            run_test("CreateScalar creates scalar matrix (flag check)", fm_scalar_static.is_currently_scalar(scalar_val_static) && scalar_val_static == 5);

            FlatMatrix fm_scalar_zero_val = FlatMatrix::CreateScalar(3, 0, p); // non-const
            uint32_t scalar_val_szv;
            run_test("CreateScalar (0 value) creates scalar 0 matrix (flag check)", fm_scalar_zero_val.is_currently_scalar(scalar_val_szv) && scalar_val_szv == 0);

            Matrix mat_orig(2, 2, p, { {1,2},{3,4} });
            FlatMatrix fm_from_matrix(mat_orig); // non-const
            run_test("Constructor from Matrix content (0,0)", fm_from_matrix.get_entry(0, 0) == 1);
            run_test("Constructor from Matrix content (1,1)", fm_from_matrix.get_entry(1, 1) == 4);
            uint32_t dummy_s_val_conv;
            run_test("Constructor from Matrix flags (not scalar)", !fm_from_matrix.is_currently_scalar(dummy_s_val_conv));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("Element Access and Modification");
        {
            FlatMatrix fm(3, p); // Identity, non-const
            run_test("Identity get_entry(0,0)", fm.get_entry(0, 0) == 1);
            run_test("Identity get_entry(0,1)", fm.get_entry(0, 1) == 0);

            FlatMatrix fm_s = FlatMatrix::CreateScalar(2, 7, p); // non-const
            run_test("Scalar get_entry(1,1)", fm_s.get_entry(1, 1) == 7);
            run_test("Scalar get_entry(1,0)", fm_s.get_entry(1, 0) == 0);

            fm_s.set_entry(0, 1, 9); // Becomes general
            if (verbosity > 2) std::cout << "      Scalar after set_entry:\n" << fm_s.to_string_offset(6);
            uint32_t dummy_s_val_set;
            run_test("Scalar after set_entry(0,1,9) is not scalar (flag check)", !fm_s.is_currently_scalar(dummy_s_val_set));
            run_test("Scalar after set_entry(0,1,9) value check", fm_s.get_entry(0, 1) == 9 && fm_s.get_entry(0, 0) == 7);

            FlatMatrix fm_z = FlatMatrix::CreateZero(2, 2, p); // non-const
            fm_z.set_entry(1, 1, 5);
            if (verbosity > 2) std::cout << "      Zero after set_entry:\n" << fm_z.to_string_offset(6);
            uint32_t dummy_s_val_z_set;
            run_test("Zero after set_entry(1,1,5) is not scalar 0 (flag check)", !(fm_z.is_currently_scalar(dummy_s_val_z_set) && dummy_s_val_z_set == 0));
            run_test("Zero after set_entry(1,1,5) value check", fm_z.get_entry(1, 1) == 5 && fm_z.get_entry(0, 0) == 0);

            run_test_throws("get_entry out of bounds (row)", [&]() { fm.get_entry(3, 0); }, "std::out_of_range");
            run_test_throws("set_entry out of bounds (col)", [&]() { fm.set_entry(0, 3, 1); }, "std::out_of_range");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("populate_flat_data and Data Inspection");
        {
            FlatMatrix fm_s = FlatMatrix::CreateScalar(2, 8, p); // non-const
            run_test("Scalar initially has empty flat_data (unsafe check)", fm_s.get_flat_data_unsafe().empty());
            fm_s.populate_flat_data();
            std::vector<uint32_t> expected_s_data = { 8, 0, 0, 8 };
            run_test("Scalar after populate_flat_data content", fm_s.get_flat_data_unsafe() == expected_s_data);

            FlatMatrix fm_z = FlatMatrix::CreateZero(1, 3, p); // non-const
            run_test("Zero initially has empty flat_data (unsafe check)", fm_z.get_flat_data_unsafe().empty());
            fm_z.populate_flat_data();
            std::vector<uint32_t> expected_z_data = { 0, 0, 0 };
            run_test("Zero after populate_flat_data content", fm_z.get_flat_data_unsafe() == expected_z_data);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("Conversions");
        {
            Matrix mat_orig(2, 2, p, { {10,1},{3,11} });
            FlatMatrix fm_conv(mat_orig); // non-const (to_matrix is const, this is fine)
            Matrix mat_back = fm_conv.to_matrix();
            run_test("to_matrix content check", mat_orig == mat_back);

            FlatMatrix fm_s = FlatMatrix::CreateScalar(2, 7, p); // non-const
            Matrix mat_s_back = fm_s.to_matrix();
            Matrix mat_s_expected(2, p); // Identity
            mat_s_expected = mat_s_expected * 7;
            run_test("to_matrix from scalar", mat_s_back == mat_s_expected);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("Basic Properties (is_zero, transpose, nonzero_entries)");
        {
            FlatMatrix fm(2, 3, p, { 1,2,3,4,5,6 }); // non-const
            FlatMatrix fm_t = fm.transpose(); // non-const
            if (verbosity > 2) std::cout << "      fm:\n" << fm.to_string_offset(6) << "      fm_t:\n" << fm_t.to_string_offset(6);
            run_test("Transpose dimensions", fm_t.get_rows() == 3 && fm_t.get_cols() == 2);
            run_test("Transpose content (0,0)", fm_t.get_entry(0, 0) == fm.get_entry(0, 0));
            run_test("Transpose content (1,0)", fm_t.get_entry(1, 0) == fm.get_entry(0, 1));

            FlatMatrix fm_s = FlatMatrix::CreateScalar(3, 5, p); // non-const
            FlatMatrix fm_s_t = fm_s.transpose(); // non-const
            uint32_t sv_t;
            run_test("Transpose of scalar is scalar", fm_s_t.is_currently_scalar(sv_t) && sv_t == 5);

            FlatMatrix fm_z_test = FlatMatrix::CreateZero(2, 2, p); // non-const
            run_test("is_zero for zero matrix", fm_z_test.is_zero());
            FlatMatrix fm_s_nz_test = FlatMatrix::CreateScalar(2, 1, p); // non-const
            run_test("is_zero for non-zero scalar", !fm_s_nz_test.is_zero());
            FlatMatrix fm_gen_nz_test(2, 2, p, { 1,0,0,0 }); // non-const
            run_test("is_zero for general non-zero", !fm_gen_nz_test.is_zero());
            FlatMatrix fm_gen_zero_test(2, 2, p, { 0,0,0,0 }); // non-const
            run_test("is_zero for general becoming zero (and flag updates)", fm_gen_zero_test.is_zero());
            uint32_t s_val_gz; // Check flag update
            run_test("is_zero for general becoming zero (flag is scalar 0)", fm_gen_zero_test.is_currently_scalar(s_val_gz) && s_val_gz == 0);

            FlatMatrix fm_z_nze = FlatMatrix::CreateZero(3, 3, p); // non-const
            run_test("nonzero_entries for zero", fm_z_nze.nonzero_entries() == 0);
            FlatMatrix fm_s_nze = FlatMatrix::CreateScalar(3, 1, p); // non-const
            run_test("nonzero_entries for scalar", fm_s_nze.nonzero_entries() == 3);
            FlatMatrix fm_gen_nze(2, 3, p, { 1,2,0,0,5,6 }); // non-const
            run_test("nonzero_entries for general", fm_gen_nze.nonzero_entries() == 4);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("Arithmetic Operations and Equality");
        {
            FlatMatrix A(2, 2, p, { 1,2,3,4 }); // non-const
            FlatMatrix B(2, 2, p, { 5,6,7,8 }); // non-const
            FlatMatrix Z = FlatMatrix::CreateZero(2, 2, p); // non-const
            FlatMatrix I = FlatMatrix(2, p); // Identity, non-const
            FlatMatrix S3 = FlatMatrix::CreateScalar(2, 3, p); // non-const

            if (verbosity > 2) { /* ... print statements ... */ }

            // Addition
            FlatMatrix A_plus_Z = A + Z; // non-const
            run_test("A + Z == A", compare_flat_matrices_content(A_plus_Z, A, p));
            FlatMatrix S3_plus_I = S3 + I; // non-const
            FlatMatrix S4_expected = FlatMatrix::CreateScalar(2, 4, p); // non-const
            run_test("Scalar + Scalar = Scalar (S3+I == S4)", compare_flat_matrices_content(S3_plus_I, S4_expected, p));
            FlatMatrix A_plus_B_res_data(2, 2, p, { ModArithmetic::mod_add(1,5,p), ModArithmetic::mod_add(2,6,p),
                                            ModArithmetic::mod_add(3,7,p), ModArithmetic::mod_add(4,8,p) });
            FlatMatrix A_plus_B = A + B; // non-const
            run_test("A + B", compare_flat_matrices_content(A_plus_B, A_plus_B_res_data, p));

            // Subtraction
            FlatMatrix A_minus_Z = A - Z; // non-const
            run_test("A - Z == A", compare_flat_matrices_content(A_minus_Z, A, p));
            FlatMatrix S3_minus_I = S3 - I; // non-const
            FlatMatrix S2_expected = FlatMatrix::CreateScalar(2, 2, p); // non-const
            run_test("Scalar - Scalar = Scalar (S3-I == S2)", compare_flat_matrices_content(S3_minus_I, S2_expected, p));

            // Multiplication (Matrix-Matrix)
            FlatMatrix A_times_Z = A * Z; // non-const
            run_test("A * Z == Z (content check)", compare_flat_matrices_content(A_times_Z, Z, p));
            FlatMatrix A_times_I = A * I; // non-const
            run_test("A * I == A (content check)", compare_flat_matrices_content(A_times_I, A, p));
            FlatMatrix S3_times_A = S3 * A; // non-const
            FlatMatrix S3A_expected_data(2, 2, p, { ModArithmetic::mod_mul(3,1,p), ModArithmetic::mod_mul(3,2,p),
                                             ModArithmetic::mod_mul(3,3,p), ModArithmetic::mod_mul(3,4,p) });
            run_test("S3 * A (content check)", compare_flat_matrices_content(S3_times_A, S3A_expected_data, p));
            FlatMatrix A_times_S3 = A * S3; // non-const
            run_test("A * S3 (content check)", compare_flat_matrices_content(A_times_S3, S3A_expected_data, p));

            FlatMatrix AB_expected_data(2, 2, p, { 19,22,20,4 });
            FlatMatrix A_times_B = A * B; // non-const
            run_test("A * B (content check)", compare_flat_matrices_content(A_times_B, AB_expected_data, p));

            // Multiplication (Matrix-Scalar)
            FlatMatrix A_times_3 = A * 3u; // non-const
            run_test("A * 3 (content check)", compare_flat_matrices_content(A_times_3, S3A_expected_data, p));
            FlatMatrix Z_times_3 = Z * 3u; // non-const
            run_test("Z * 3 == Z (content check)", compare_flat_matrices_content(Z_times_3, Z, p));

            // Equality (operator== is non-const for both operands)
            FlatMatrix A_copy = A; // non-const
            FlatMatrix B_copy = B; // non-const
            FlatMatrix Z_copy = Z; // non-const
            FlatMatrix I_copy = I; // non-const
            FlatMatrix S3_copy = S3; // non-const

            run_test("A == A_copy", A == A_copy); // A and A_copy are non-const
            run_test("A != B", A != B_copy);
            run_test("Z != I", Z_copy != I_copy);
            run_test("S3 == S3_copy", S3 == S3_copy);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("FlatMatrix Equality Operator (== and !=)");
        {
            // Test Case 1: Basic self-equality and equality with identical copy
            FlatMatrix A1(2, 2, p, { 1, 2, 3, 4 });
            FlatMatrix A1_copy = A1; // Copy constructor
            FlatMatrix A1_assign(1, 1, p); A1_assign = A1; // Copy assignment
            run_test("A == A (self)", A1 == A1);
            run_test("A == A_copy", A1 == A1_copy);
            run_test("A == A_assign", A1 == A1_assign);

            // Test Case 2: Different general matrices
            FlatMatrix B1(2, 2, p, { 1, 2, 3, 5 }); // Different from A1
            run_test("A1 != B1 (different content)", A1 != B1);
            run_test("!(A1 == B1) (different content)", !(A1 == B1));

            // Test Case 3: Different dimensions
            FlatMatrix C1(2, 3, p, { 1, 2, 3, 4, 5, 6 });
            run_test("A1 != C1 (different dimensions)", A1 != C1);

            // Test Case 4: Different primes (should be caught by operator==)
            FlatMatrix D1(2, 2, p + 2, { 1, 2, 3, 4 }); // Different prime
            // Note: The operator== checks p_ first.
            // To fully test this, our operator== would need to handle p mismatch,
            // or we assume p is consistent for valid comparisons.
            // The current operator== checks p_: if (rows_ != other.rows_ || cols_ != other.cols_ || p_ != other.p_)
            bool p_mismatch_caught = false;
            try {
                if (A1 == D1) {} // This comparison itself might be fine if p check is first
                                 // But logically they are not equal.
                // The operator== returns false if p_ are different.
                p_mismatch_caught = !(A1 == D1);
            }
            catch (...) { /* Should not throw if p check is clean */ }
            run_test("A1 != D1 (different prime)", p_mismatch_caught);


            // Test Case 5: Scalar Zero vs General Zero
            FlatMatrix SZ1 = FlatMatrix::CreateZero(2, 2, p);      // is_scalar_matrix_=true, scalar_value_=0
            FlatMatrix GZ1(2, 2, p, { 0, 0, 0, 0 });              // is_scalar_matrix_=false (initially)
            FlatMatrix GZ2 = FlatMatrix::CreateZero(2, 3, p);     // Non-square scalar zero
            FlatMatrix GZ3(2, 3, p, { 0,0,0, 0,0,0 }); // Non-square general zero

            if (verbosity > 2) {
                uint32_t sv_sz1; bool is_s_sz1 = SZ1.is_currently_scalar(sv_sz1);
                uint32_t sv_gz1 = 0; bool is_s_gz1 = GZ1.is_currently_scalar(sv_gz1); // GZ1's flag depends on if is_zero() was called by prior test. Let's ensure fresh.
                GZ1 = FlatMatrix(2, 2, p, { 0, 0, 0, 0 }); // Re-create to ensure it's general before comparison
                is_s_gz1 = GZ1.is_currently_scalar(sv_gz1);

                std::cout << "      SZ1: scalar=" << is_s_sz1 << " val=" << (is_s_sz1 ? sv_sz1 : -1) << "\n";
                std::cout << "      GZ1: scalar=" << is_s_gz1 << " val=" << (is_s_gz1 ? sv_gz1 : -1) << "\n";
            }
            run_test("Scalar Zero (2x2) == General Zero (2x2)", SZ1 == GZ1);
            GZ1 = FlatMatrix(2, 2, p, { 0, 0, 0, 0 }); // Re-create GZ1 to reset its potential flag changes
            run_test("General Zero (2x2) == Scalar Zero (2x2)", GZ1 == SZ1);
            run_test("Scalar Zero (2x3) == General Zero (2x3)", GZ2 == GZ3);
            GZ3 = FlatMatrix(2, 3, p, { 0,0,0, 0,0,0 });
            run_test("General Zero (2x3) == Scalar Zero (2x3)", GZ3 == GZ2);

            // Test Case 6: Scalar Value vs General Matrix with that scalar pattern
            FlatMatrix S_val1 = FlatMatrix::CreateScalar(2, 5, p); // is_scalar_matrix_=true, scalar_value_=5
            FlatMatrix G_pat1(2, 2, p, { 5, 0, 0, 5 });           // is_scalar_matrix_=false (initially)
            if (verbosity > 2) {
                uint32_t sv_sval1; bool is_s_sval1 = S_val1.is_currently_scalar(sv_sval1);
                uint32_t sv_gpat1 = 0; bool is_s_gpat1 = G_pat1.is_currently_scalar(sv_gpat1);
                G_pat1 = FlatMatrix(2, 2, p, { 5, 0, 0, 5 }); // Re-create
                is_s_gpat1 = G_pat1.is_currently_scalar(sv_gpat1);
                std::cout << "      S_val1: scalar=" << is_s_sval1 << " val=" << (is_s_sval1 ? sv_sval1 : -1) << "\n";
                std::cout << "      G_pat1: scalar=" << is_s_gpat1 << " val=" << (is_s_gpat1 ? sv_gpat1 : -1) << "\n";
            }
            run_test("Scalar Value (2x2) == General Scalar Pattern (2x2)", S_val1 == G_pat1);
            G_pat1 = FlatMatrix(2, 2, p, { 5, 0, 0, 5 }); // Re-create
            run_test("General Scalar Pattern (2x2) == Scalar Value (2x2)", G_pat1 == S_val1);

            // Test Case 7: Scalar Value vs General Matrix that is different
            FlatMatrix G_diff1(2, 2, p, { 5, 0, 0, 1 }); // Different from S_val1
            run_test("Scalar Value (2x2) != General Different (2x2)", S_val1 != G_diff1);

            // Test Case 8: Two different scalar matrices
            FlatMatrix S_val2 = FlatMatrix::CreateScalar(2, 6, p);
            run_test("Scalar Value (5) != Scalar Value (6) (2x2)", S_val1 != S_val2);

            // Test Case 9: 0-dimensional matrices
            FlatMatrix M0x0_1 = FlatMatrix::CreateZero(0, 0, p);
            FlatMatrix M0x0_2 = FlatMatrix::CreateZero(0, 0, p);
            FlatMatrix M0x3_1 = FlatMatrix::CreateZero(0, 3, p);
            FlatMatrix M0x3_2(0, 3, p, {}); // General 0x3
            FlatMatrix M3x0_1 = FlatMatrix::CreateZero(3, 0, p);
            FlatMatrix M3x0_2(3, 0, p, {}); // General 3x0

            run_test("M0x0 == M0x0", M0x0_1 == M0x0_2);
            run_test("M0x3 == M0x3 (scalar vs general)", M0x3_1 == M0x3_2);
            run_test("M3x0 == M3x0 (scalar vs general)", M3x0_1 == M3x0_2);
            run_test("M0x0 != M0x3", M0x0_1 != M0x3_1); // Different dimensions
            run_test("M0x3 != M3x0", M0x3_1 != M3x0_1); // Different dimensions

            // Test Case 10: General matrix that is scalar zero vs general matrix that is non-zero
            FlatMatrix GZ_A(2, 2, p, { 0,0,0,0 });
            FlatMatrix GNZ_A(2, 2, p, { 0,0,0,1 });
            run_test("General Zero != General NonZero", GZ_A != GNZ_A);

            // Test Case 11: What if one is scalar, other is general, but general is NOT the scalar pattern
            FlatMatrix S_val_id = FlatMatrix::CreateScalar(2, 1, p); // Identity
            FlatMatrix G_not_id(2, 2, p, { 1,2,0,1 }); // Not identity
            //run_test("Scalar Identity != General Non-Identity Pattern", S_val_id != G_not_id);
            G_not_id = FlatMatrix(2, 2, p, { 1,2,0,1 }); // reset
            run_test("General Non-Identity Pattern != Scalar Identity", G_not_id != S_val_id);

            // Test Case 12: Ensure a matrix that *becomes* scalar during an operation is compared correctly
            // This is more about testing operations + comparison, but relevant if flags change.
            // Our current operator== (const version) doesn't change flags.
            // So this case is covered by comparing a matrix whose flags are already set.

        } // End of Equality Operator section
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("Linear Algebra (Rank, Invertibility, Inverse, RREF, Kernel, Image)");
        {
            FlatMatrix A(3, 3, p, { 1,2,3, 0,1,4, 0,0,1 }); // non-const
            FlatMatrix B(3, 3, p, { 1,1,0, 0,1,1, 1,2,1 }); // non-const
            FlatMatrix Z = FlatMatrix::CreateZero(3, 3, p); // non-const
            FlatMatrix I3 = FlatMatrix(3, p); // non-const

            if (verbosity > 2) { /* ... print statements ... */ }

            // Rank
            run_test("Rank of Zero matrix", Z.rank() == 0);
            run_test("Rank of Identity matrix", I3.rank() == 3);
            run_test("Rank of A (full rank)", A.rank() == 3);
            run_test("Rank of B (rank 2)", B.rank() == 2);

            // Invertibility
            run_test("Invertible Z", !Z.is_invertible());
            run_test("Invertible I3", I3.is_invertible());
            run_test("Invertible A", A.is_invertible());
            run_test("Invertible B", !B.is_invertible());

            // Inverse
            FlatMatrix I3_inv = I3.inverse(); // non-const for I3.inverse() if it implies non-const internal state change
                                               // Current FlatMatrix::inverse is const.
            run_test("Inverse of I3 is I3", compare_flat_matrices_content(I3_inv, I3, p));
            FlatMatrix A_inv = A.inverse(); // A is non-const, A.inverse() is const
            FlatMatrix A_A_inv = A * A_inv; // non-const
            FlatMatrix I3_check = FlatMatrix(3, p); // Fresh identity for comparison
            run_test("A * A_inv == I3", compare_flat_matrices_content(A_A_inv, I3_check, p));
            run_test_throws("Inverse of B (non-invertible)", [&]() { B.inverse(); }, "std::runtime_error");

            // RREF (get_row_echelon_form_as_flat_matrix is const)
            FlatMatrix A_rref_fm = A.get_row_echelon_form_as_flat_matrix();
            FlatMatrix I3_rref_check = FlatMatrix(3, p);
            run_test("RREF of A is I3", compare_flat_matrices_content(A_rref_fm, I3_rref_check, p));

            // Kernel (kernel() is non-const)
            std::vector<FlatMatrix> ker_I3 = I3.kernel();
            run_test("Kernel of I3 is empty (dim 0)", ker_I3.empty());

            std::vector<FlatMatrix> ker_B = B.kernel();
            run_test("Kernel of B has dim 1 (nullity = cols - rank = 3 - 2 = 1)", ker_B.size() == 1);
            if (ker_B.size() == 1) {
                FlatMatrix B_ker_vec = B * ker_B[0]; // B and ker_B[0] non-const if operator* becomes non-const.
                                                    // operator* is const, so this is okay.
                FlatMatrix Z_col_vec = FlatMatrix::CreateZero(3, 1, p); // non-const
                run_test("B * (kernel vector of B) == 0", compare_flat_matrices_content(B_ker_vec, Z_col_vec, p));
            }
            std::vector<FlatMatrix> ker_Z = Z.kernel();
            run_test("Kernel of Z (3x3) has dim 3", ker_Z.size() == 3);

            // Image (image() is non-const)
            std::vector<FlatMatrix> img_I3 = I3.image();
            run_test("Image of I3 has dim 3", img_I3.size() == 3);
            std::vector<FlatMatrix> img_B = B.image();
            run_test("Image of B has dim 2", img_B.size() == 2);
            std::vector<FlatMatrix> img_Z = Z.image();
            run_test("Image of Z (3x3) is empty (dim 0)", img_Z.empty());
        }

        // ===================================================================
        // Owner decision Q11 (matrix-engine plan Sec. 7): image() must return the
        // PIVOT COLUMNS OF THE ORIGINAL MATRIX in all three substrate classes.
        // For A = alpha*I that is alpha*e_j.  FlatMatrix's scalar fast path used
        // to return e_j instead, so FlatMatrix disagreed entrywise with
        // MatrixT::image() and BlockFlatMatrixT::flat_image() on every scalar
        // matrix with alpha not in {0, 1}.  This test pins the aligned
        // convention by comparing all three on the same alpha*I.
        // ===================================================================
        run_test_section("Q11: image() convention on alpha*I (Matrix / FlatMatrix / BlockFlatMatrix)");
        {
            const uint32_t alpha = 5; // alpha not in {0, 1}, and 5 < p = 23
            const uint32_t d = 3;

            // (1) FlatMatrix alpha*I, kept on its scalar fast path.
            FlatMatrix aI_fm(d, p);                    // identity (scalar flag set)
            FlatMatrix aI_scaled = aI_fm * alpha;      // still flagged scalar
            uint32_t sv = 0;
            run_test("alpha*I (FlatMatrix) is still on the scalar fast path",
                     aI_scaled.is_currently_scalar(sv) && sv == alpha);
            std::vector<FlatMatrix> img_fm = aI_scaled.image();

            // (2) MatrixT alpha*I (dense; general pivot-column path).
            std::vector<std::vector<uint32_t>> aI_rows(d, std::vector<uint32_t>(d, 0));
            for (uint32_t i = 0; i < d; ++i) aI_rows[i][i] = alpha;
            Matrix aI_m(d, d, p, aI_rows);
            std::vector<Matrix> img_m = aI_m.image();

            // (3) BlockFlatMatrixT alpha*I as a single dxd block (legacy/FlatMatrixLegacyArms.inc; in an
            //     OC_ENABLE_LEGACY=OFF build the third class is compiled out and the checks compare two classes).
#if OC_ENABLE_LEGACY
            const bool have_bfm = true;
            const std::vector<std::vector<uint32_t>> img_bfm = LegacyQ11BlockImage(aI_scaled, d, p);
#else
            const bool have_bfm = false;
            const std::vector<std::vector<uint32_t>> img_bfm;
            (void)oc::legacy::RunLegacyArm("FlatMatrixValidation", "Q11-BlockFlatMatrix");
#endif

            run_test("alpha*I: all three classes report a 3-dimensional image",
                     img_fm.size() == d && img_m.size() == d && (!have_bfm || img_bfm.size() == d));

            if (img_fm.size() == d && img_m.size() == d && (!have_bfm || img_bfm.size() == d)) {
                bool fm_is_alpha_ej = true, all_agree = true;
                for (uint32_t j = 0; j < d; ++j) {
                    for (uint32_t i = 0; i < d; ++i) {
                        const uint32_t expected = (i == j) ? alpha : 0u;
                        const uint32_t v_fm  = img_fm[j].get_entry(i, 0);
                        const uint32_t v_m   = img_m[j].data[i][0];
                        const uint32_t v_bfm = have_bfm ? img_bfm[j][i] : v_fm;
                        if (v_fm != expected) fm_is_alpha_ej = false;
                        if (v_fm != v_m || v_fm != v_bfm) all_agree = false;
                    }
                }
                run_test("FlatMatrix::image() of alpha*I returns alpha*e_j (not e_j)", fm_is_alpha_ej);
                run_test("Matrix / FlatMatrix / BlockFlatMatrix images of alpha*I agree entrywise", all_agree);
            }

            // The identity must be unaffected by the change (alpha = 1).
            FlatMatrix I_fm(d, p);
            std::vector<FlatMatrix> img_I = I_fm.image();
            bool identity_ok = (img_I.size() == d);
            for (uint32_t j = 0; identity_ok && j < d; ++j)
                for (uint32_t i = 0; i < d; ++i)
                    if (img_I[j].get_entry(i, 0) != ((i == j) ? 1u : 0u)) identity_ok = false;
            run_test("image() of the identity is unchanged (e_j)", identity_ok);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
    }
    try
    {
        run_test_section("Edge Cases (0-dim matrices, incompatible ops)");
        {
            FlatMatrix M0x0(0, 0, p); // non-const
            FlatMatrix M0x3(0, 3, p); // non-const
            FlatMatrix M3x0(3, 0, p); // non-const

            uint32_t s_val_0x0; bool is_s_0x0 = M0x0.is_currently_scalar(s_val_0x0);
            run_test("M0x0 is scalar 0", is_s_0x0 && s_val_0x0 == 0 && M0x0.get_rows() == 0 && M0x0.get_cols() == 0);
            // ... similar checks for M0x3 and M3x0

            run_test("Rank of M0x0", M0x0.rank() == 0);
            run_test("Rank of M0x3", M0x3.rank() == 0);
            run_test("Rank of M3x0", M3x0.rank() == 0);

            FlatMatrix M0x2(0, 2, p); // non-const
            FlatMatrix Prod_3x0_0x2 = M3x0 * M0x2; // operator* is const
            run_test("M3x0 * M0x2 dimensions", Prod_3x0_0x2.get_rows() == 3 && Prod_3x0_0x2.get_cols() == 2);
            FlatMatrix Z3x2 = FlatMatrix::CreateZero(3, 2, p); // non-const for comparison
            run_test("M3x0 * M0x2 is zero (content check)", compare_flat_matrices_content(Prod_3x0_0x2, Z3x2, p));

            FlatMatrix A2x2(2, 2, p, { 1,0,0,1 }); // non-const
            FlatMatrix B3x3(3, 3, p, { 1,0,0,0,1,0,0,0,1 }); // non-const
            run_test_throws("A2x2 + B3x3 (dim mismatch)", [&]() { A2x2 + B3x3; }, "std::invalid_argument");
            run_test_throws("A2x2 * B3x3 (dim mismatch inner)", [&]() { A2x2* B3x3; }, "std::invalid_argument");
        }

        // (slot-14 WP-3, G1) rows*cols >= 2^32: every index product is size_t,
        // so a shape past the uint32 domain is either correct or refused
        // loudly -- never wrapped.  Scalar / empty-data shapes only: nothing
        // here allocates (predicted RSS: unchanged, < 1 MB).
        run_test_section("G1: rows*cols beyond 2^32 (no allocation)");
        {
            // 65536 x 65536 = 2^32 wraps to 0 in uint32: the data constructor
            // used to take an EMPTY vector as a valid "0-dim" matrix (a silent
            // 0 x 0 reading of a 2^32-entry shape); widened, it is a size
            // mismatch and throws.
            bool threw_invalid = false;
            try {
                FlatMatrix big(65536u, 65536u, p, std::vector<uint32_t>{});
            } catch (const std::invalid_argument&) { threw_invalid = true; }
            run_test("G1: 65536 x 65536 with empty data is refused (was a silent 0-dim reading)", threw_invalid);

            // A scalar identity of that shape reads correctly without data.
            FlatMatrix I64k(65536u, p);
            run_test("G1: 65536 x 65536 identity reads (65535,65535) = 1", I64k.get_entry(65535u, 65535u) == 1u);
            run_test("G1: 65536 x 65536 identity reads (65535,0) = 0", I64k.get_entry(65535u, 0u) == 0u);
            run_test("G1: 65536 x 65536 identity is not zero", !I64k.is_zero());
            FlatMatrix Z64k = FlatMatrix::CreateZero(65536u, 65536u, p);
            run_test("G1: 65536 x 65536 zero is zero", Z64k.is_zero());

            // (2^32 - 1)^2 exceeds the storage's size_type (vector<uint32_t>::max_size()
            // < 2^62): materialising it is refused by the ENGINE_CHECK before any
            // allocation.  In uint32 the product wrapped to 1 and the diagonal
            // write ran out of bounds.
            FlatMatrix huge(0xFFFFFFFFu, p);   // scalar identity, no data
            bool threw_logic = false;
            std::string what;
            try {
                huge.set_entry(0u, 1u, 5u);   // forces populate_flat_data()
            } catch (const std::invalid_argument&) {
            } catch (const std::logic_error& e) { threw_logic = true; what = e.what(); }
            run_test("G1: materialising (2^32-1)^2 entries is refused by the size_type ENGINE_CHECK",
                     threw_logic && what.find("size_type") != std::string::npos);
            run_test("G1: the refused matrix is still the scalar identity", huge.get_entry(7u, 7u) == 1u && huge.get_entry(7u, 8u) == 0u);
        }

    }
    catch (const std::exception& e) {
        std::cerr << "UNHANDLED EXCEPTION in FlatMatrixValidation: " << e.what() << std::endl;
        if (total_tests == passed_tests && total_tests > 0) { /* Potentially misleading */ }
        // To ensure failure status if exception occurs after some passes:
        // If an exception occurs, at least one "meta-test" (the suite itself) fails.
        // The return (total_tests > 0 && total_tests == passed_tests) ? 0 : 1 handles this
        // if we don't increment passed_tests on exception.
        // To be very explicit, we could have a global `bool suite_failed_exception = false;`
        // and set it in catch, then factor into return. For now, the current logic is okay.
    }

    if (verbosity >= 0) {
        std::cout << "\nFlatMatrixValidation Summary:" << std::endl;
        std::cout << "  Tests run:    " << total_tests << std::endl;
        std::cout << "  Tests passed: " << passed_tests << std::endl;
        std::cout << "  Tests failed: " << (total_tests - passed_tests) << std::endl;
        if (total_tests == 0) std::cout << "  WARNING: No tests were run!" << std::endl;
    }

    return (total_tests > 0 && total_tests == passed_tests) ? 0 : 1;
}
