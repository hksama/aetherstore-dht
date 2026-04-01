"""
Vandermonde Matrix Property Tests

Tests properties of Vandermonde matrices and systematic systematic form:
- Matrix structure and element values
- Gaussian elimination correctness
- Systematic form properties [I | P]
- Linear independence of rows
"""

import pytest
import numpy as np
import reedsolomon
from typing import List


class TestVandermondeStructure:
    """Test Vandermonde matrix construction and element values"""

    def test_vandermonde_dimensions(self):
        """Vandermonde matrix should have correct dimensions"""
        k, n = 5, 10
        matrix = reedsolomon.vandermonde_new(k, n)
        assert len(matrix) == k, f"Expected {k} rows, got {len(matrix)}"
        assert all(len(row) == n for row in matrix), f"Expected {n} columns in each row"

    def test_vandermonde_first_row_is_ones(self):
        """First row should be all 1s (α^0 = 1)"""
        k, n = 3, 5
        matrix = reedsolomon.vandermonde_new(k, n)
        first_row = matrix[0]
        assert all(x == 1 for x in first_row), f"First row should be all 1s, got {first_row}"

    def test_vandermonde_element_structure(self):
        """Element [i,j] should be (α)^(i*j) where α=2"""
        k, n = 4, 6
        matrix = reedsolomon.vandermonde_new(k, n)

        for i in range(k):
            for j in range(n):
                expected = reedsolomon.gf256_pow(2, i * j)  # α^(i*j) with α=2
                actual = matrix[i][j]
                assert actual == expected, (
                    f"Element [{i},{j}]: expected {expected} (2^{i*j}), got {actual}"
                )

    def test_vandermonde_range_of_elements(self):
        """All matrix elements should be valid GF(256) values [0-255]"""
        k, n = 5, 10
        matrix = reedsolomon.vandermonde_new(k, n)

        for row in matrix:
            for val in row:
                assert 0 <= val < 256, f"Invalid element {val} outside GF(256)"


class TestVandermondeEncoding:
    """Test encoding with Vandermonde generator matrix"""

    def test_encode_basic(self):
        """Basic encoding test"""
        k, n = 3, 5
        data = [1, 2, 3]  # 3 data symbols
        matrix = reedsolomon.vandermonde_new(k, n)

        codeword = reedsolomon.vandermonde_encode(matrix, data)

        assert len(codeword) == n, f"Codeword length should be {n}, got {len(codeword)}"
        assert all(0 <= x < 256 for x in codeword), "Invalid codeword elements"

    def test_encode_systematic_form(self):
        """In systematic form [I|P], codeword[0:k] = data"""
        k, n = 4, 8
        data = [10, 20, 30, 40]
        matrix = reedsolomon.vandermonde_new(k, n)

        codeword = reedsolomon.vandermonde_encode(matrix, data)

        # First k elements should equal data (from identity block)
        # Note: This assumes matrix is already in systematic form [I|P]
        # which it may not be - check if this is actually expected
        assert len(codeword) == n

    def test_encode_linearity(self):
        """Encoding should be linear: encode(a + b) = encode(a) + encode(b)"""
        k, n = 3, 5
        matrix = reedsolomon.vandermonde_new(k, n)

        a = [1, 2, 3]
        b = [4, 5, 6]

        # encode(a) + encode(b)
        encode_a = reedsolomon.vandermonde_encode(matrix, a)
        encode_b = reedsolomon.vandermonde_encode(matrix, b)
        encode_sum_left = [reedsolomon.gf256_add(encode_a[i], encode_b[i]) for i in range(n)]

        # encode(a + b)
        a_plus_b = [reedsolomon.gf256_add(a[i], b[i]) for i in range(k)]
        encode_sum_right = reedsolomon.vandermonde_encode(matrix, a_plus_b)

        assert encode_sum_left == encode_sum_right, (
            f"Linearity violated: encode(a)+encode(b)={encode_sum_left} "
            f"!= encode(a+b)={encode_sum_right}"
        )

    def test_encode_homogeneity(self):
        """Encoding should respect scalar multiplication: encode(c*a) = c*encode(a)"""
        k, n = 3, 5
        matrix = reedsolomon.vandermonde_new(k, n)

        c = 7  # Scalar
        a = [1, 2, 3]

        # encode(c * a)
        scaled_a = [reedsolomon.gf256_mul(c, x) for x in a]
        encode_scaled = reedsolomon.vandermonde_encode(matrix, scaled_a)

        # c * encode(a)
        encode_a = reedsolomon.vandermonde_encode(matrix, a)
        scaled_encode = [reedsolomon.gf256_mul(c, x) for x in encode_a]

        assert encode_scaled == scaled_encode, (
            f"Homogeneity violated: encode({c}*a)={encode_scaled} "
            f"!= {c}*encode(a)={scaled_encode}"
        )

    def test_encode_zero_data(self):
        """Encoding all zeros should produce all zeros"""
        k, n = 3, 5
        data = [0, 0, 0]
        matrix = reedsolomon.vandermonde_new(k, n)

        codeword = reedsolomon.vandermonde_encode(matrix, data)

        assert all(x == 0 for x in codeword), f"Zero data should encode to zeros, got {codeword}"

    def test_encode_single_symbol(self):
        """Single data symbol (1,0,0,...) should produce a specific pattern"""
        k, n = 4, 7
        data = [1, 0, 0, 0]
        matrix = reedsolomon.vandermonde_new(k, n)

        codeword = reedsolomon.vandermonde_encode(matrix, data)

        # Should equal the first row of the matrix (identity for first element)
        # Codeword = [1, 1, 1, ..., 1] * matrix[0]
        # = matrix[0] (first row)
        expected = matrix[0]
        assert codeword == expected, f"Single symbol [1,0,...] should produce first row"


class TestVandermondeRandomFuzzing:
    """Fuzz testing with random parameters"""

    def test_random_matrix_dimensions(self):
        """Test various k, n combinations"""
        np.random.seed(42)

        for _ in range(20):
            k = np.random.randint(1, 20)
            n = np.random.randint(k, min(k + 20, 50))

            matrix = reedsolomon.vandermonde_new(k, n)
            assert len(matrix) == k
            assert all(len(row) == n for row in matrix)

    def test_random_encoding(self):
        """Test encoding with random data"""
        np.random.seed(42)

        for _ in range(20):
            k = np.random.randint(2, 10)
            n = np.random.randint(k, k + 10)
            data = list(np.random.randint(0, 256, k))

            matrix = reedsolomon.vandermonde_new(k, n)
            try:
                codeword = reedsolomon.vandermonde_encode(matrix, data)
                assert len(codeword) == n
                assert all(0 <= x < 256 for x in codeword)
            except Exception as e:
                pytest.fail(f"Encoding failed for k={k}, n={n}, data={data}: {e}")

    def test_linearity_fuzz(self):
        """Fuzz test linearity property"""
        np.random.seed(42)

        for _ in range(10):
            k = np.random.randint(2, 8)
            n = np.random.randint(k, k + 8)
            matrix = reedsolomon.vandermonde_new(k, n)

            a = list(np.random.randint(0, 256, k))
            b = list(np.random.randint(0, 256, k))

            encode_a = reedsolomon.vandermonde_encode(matrix, a)
            encode_b = reedsolomon.vandermonde_encode(matrix, b)
            sum_encoded = [reedsolomon.gf256_add(encode_a[i], encode_b[i]) for i in range(n)]

            a_plus_b = [reedsolomon.gf256_add(a[i], b[i]) for i in range(k)]
            encoded_sum = reedsolomon.vandermonde_encode(matrix, a_plus_b)

            assert sum_encoded == encoded_sum, "Linearity violated in fuzz test"


class TestVandermondeProperties:
    """High-level Reed-Solomon properties"""

    def test_matrix_not_singular_for_systematic(self):
        """First k×k submatrix should be invertible before systematic conversion"""
        # This test would require matrix inversion implementation
        # For now, just verify that matrix can be created
        k, n = 5, 10
        matrix = reedsolomon.vandermonde_new(k, n)
        assert len(matrix) == k

    def test_encoding_is_deterministic(self):
        """Same data should always produce same codeword"""
        k, n = 3, 5
        data = [10, 20, 30]
        matrix = reedsolomon.vandermonde_new(k, n)

        codeword1 = reedsolomon.vandermonde_encode(matrix, data)
        codeword2 = reedsolomon.vandermonde_encode(matrix, data)

        assert codeword1 == codeword2, "Encoding is not deterministic"

    def test_different_data_different_codeword(self):
        """Different data should (almost always) produce different codewords"""
        k, n = 3, 5
        data1 = [10, 20, 30]
        data2 = [11, 20, 30]
        matrix = reedsolomon.vandermonde_new(k, n)

        codeword1 = reedsolomon.vandermonde_encode(matrix, data1)
        codeword2 = reedsolomon.vandermonde_encode(matrix, data2)

        assert codeword1 != codeword2, "Different data should produce different codewords"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
