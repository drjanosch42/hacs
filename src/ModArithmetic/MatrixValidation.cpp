// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#include "pch.h"
#include "MatrixValidation.h"

uint32_t MatrixValidation(void) {
    uint32_t failures = 0;
    try {
        // Example usage:
        uint32_t p = 23; // Example prime
        std::cout << "We are computing modulo p = " << p << std::endl;

        // Create a 3x3 matrix
        std::vector<std::vector<uint32_t>> data = { {1, 2, 3}, {0, 4, 5}, {1, 0, 6} };
        Matrix A(3, 3, p, data);
        std::cout << "Matrix A:" << std::endl;
        std::cout << A.to_string_offset(2) << std::endl;

        // Create another 3x3 matrix
        std::vector<std::vector<uint32_t>> data2 = { {7, 8, 9}, {10, 11, 12}, {13, 14, 15} };
        Matrix B(3, 3, p, data2);
        std::cout << "Matrix B:" << std::endl;
        std::cout << B.to_string_offset(2) << std::endl;

        // Matrix addition
        Matrix C = A + B;
        std::cout << "Matrix C = (A + B):" << std::endl;
        std::cout << C.to_string_offset(2) << std::endl;

        // Matrix multiplication
        Matrix D = A * B;
        std::cout << "\nMatrix D = (A * B):" << std::endl;
        std::cout << D.to_string_offset(2) << std::endl;

        // Matrix inverse
        Matrix A_inv = A.inverse();
        std::cout << "\nMatrix A_inv (A inverse):" << std::endl;
        std::cout << A_inv.to_string_offset(2) << std::endl;

        Matrix I = A * A_inv;
        std::cout << "\nMatrix I = (A * A_inv):" << std::endl;
        std::cout << I.to_string_offset(2) << std::endl;


        // Compute kernel of a matrix
        std::vector<std::vector<uint32_t>> kernel_data = { {2, 5, 7}, {6, 3, 4}, {8, 8, 11} };
        Matrix E(3, 3, p, kernel_data);
        std::cout << "Matrix E:" << std::endl;
        std::cout << E.to_string_offset(2) << std::endl;
        std::vector<Matrix> kernel_basis = E.kernel();

        std::cout << "\nKernel basis of matrix E:" << std::endl;
        for (const auto& vec : kernel_basis) {
            std::cout << vec.to_string_offset(2) << std::endl;
        }

        // Compute image of a matrix
        std::vector<Matrix> image_basis = E.image();
        std::cout << "\nImage basis of matrix E:" << std::endl;
        for (const auto& vec : image_basis) {
            std::cout << vec.to_string_offset(2) << std::endl;
        }

        // (slot-24 CL-M) The change-of-basis / factor-basis / basis_to_string
        // demo that stood here exercised the deleted deprecated free functions
        // (CODEAUDIT C-02); it printed only and carried no check.

        // ===================================================================
        // Defect #9 regression (matrix-engine Phase 0): MatrixT caches its RREF
        // lazily, but `data` is a public member. Before the fix, the sequence
        //     rank();  data[i][j] = ...;  rank();
        // returned the rank of the ORIGINAL matrix, silently and forever.
        // ===================================================================
        std::cout << "\n--- Defect #9: RREF cache invalidation on direct `data` writes ---" << std::endl;
        {
            auto check = [&failures](const std::string& name, bool ok) {
                std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << name << std::endl;
                if (!ok) failures++;
            };

            // (a) rank: rank-1 matrix -> mutate -> rank 2.
            Matrix M(2, 2, p, { {1, 1}, {1, 1} });
            const uint32_t r_before = M.rank();
            check("rank of [[1,1],[1,1]] is 1", r_before == 1);
            M.data[1][1] = 2;                 // direct write to the public member
            const uint32_t r_after = M.rank();
            check("rank after data[1][1]=2 is 2 (cache invalidated)", r_after == 2);

            // (b) the same matrix object must also report the NEW pivots/RREF.
            const std::vector<uint32_t>& piv = M.get_pivot_columns();
            check("pivot columns after mutation are {0,1}",
                  piv.size() == 2 && piv[0] == 0 && piv[1] == 1);
            const std::vector<std::vector<uint32_t>>& rr = M.get_row_echelon_form();
            check("RREF after mutation is the identity",
                  rr[0][0] == 1 && rr[0][1] == 0 && rr[1][0] == 0 && rr[1][1] == 1);

            // (c) mutating back must restore the original answer.
            M.data[1][1] = 1;
            check("rank restored to 1 after undoing the write", M.rank() == 1);

            // (d) kernel/image/is_invertible must follow the cache too.
            Matrix N(2, 2, p, { {1, 0}, {0, 0} });
            check("N=[[1,0],[0,0]] is not invertible", !N.is_invertible());
            check("N has a 1-dimensional kernel", N.kernel().size() == 1);
            check("N has a 1-dimensional image", N.image().size() == 1);
            N.data[1][1] = 1;                 // N becomes the identity
            check("N is invertible after data[1][1]=1", N.is_invertible());
            check("N has a trivial kernel after mutation", N.kernel().empty());
            check("N has a 2-dimensional image after mutation", N.image().size() == 2);

            // (e) an unmutated matrix must NOT be recomputed into a different
            //     answer (guards against over-eager invalidation).
            Matrix S(3, 3, p, { {1, 2, 3}, {0, 4, 5}, {1, 0, 6} });
            const uint32_t s1 = S.rank();
            const uint32_t s2 = S.rank();
            check("repeated rank() on an untouched matrix agrees", s1 == s2 && s1 == 3);
        }
    }

        catch (const std::runtime_error& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    if (failures) {
        std::cerr << "MatrixValidation: " << failures << " check(s) FAILED." << std::endl;
        return 1;
    }
    return 0;
}
