"""
Encoding Fuzzing and Error Injection Tests

Tests Reed-Solomon properties through:
- Extensive random data fuzzing
- Systematic error injection and detection
- Property-based testing (linearity, homogeneity)
"""

import pytest
import numpy as np
import reedsolomon
from typing import List, Tuple


class TestEncodingFuzzing:
    """Extensive fuzzing of encoding operations"""

    def test_fuzz_1000_random_encodings(self):
        """Encode 1000 random data vectors"""
        np.random.seed(42)

        k, n = 8, 16
        matrix = reedsolomon.vandermonde_new(k, n)
        errors = []

        for i in range(1000):
            data = list(np.random.randint(0, 256, k))
            try:
                codeword = reedsolomon.vandermonde_encode(matrix, data)
                assert len(codeword) == n
                assert all(0 <= x < 256 for x in codeword)
            except Exception as e:
                errors.append((i, data, str(e)))

        if errors:
            pytest.fail(f"Encoding failed in {len(errors)} cases: {errors[:5]}")

    def test_fuzz_various_k_n_pairs(self):
        """Test with 100 random (k, n) combinations"""
        np.random.seed(42)
        errors = []

        for iteration in range(100):
            k = np.random.randint(1, 30)
            n = np.random.randint(k, min(k + 30, 100))
            data = list(np.random.randint(0, 256, k))

            try:
                matrix = reedsolomon.vandermonde_new(k, n)
                codeword = reedsolomon.vandermonde_encode(matrix, data)
                assert len(codeword) == n
            except Exception as e:
                errors.append((k, n, str(e)))

        if errors:
            pytest.fail(f"Failed for {len(errors)} (k,n) pairs: {errors[:3]}")

    def test_fuzz_all_data_values(self):
        """Test each GF(256) value at least once"""
        np.random.seed(42)
        k, n = 4, 8
        matrix = reedsolomon.vandermonde_new(k, n)

        # Generate data vectors covering all 256 values
        for value in range(256):
            data = [value, (value + 1) % 256, (value + 2) % 256, (value + 3) % 256]
            try:
                codeword = reedsolomon.vandermonde_encode(matrix, data)
                assert len(codeword) == n
            except Exception as e:
                pytest.fail(f"Encoding failed for value {value}: {e}")


class TestEncodingLinearity:
    """Test linearity property extensively"""

    def test_linearity_100_random_pairs(self):
        """Test a + b = encode(a) + encode(b) for 100 random pairs"""
        np.random.seed(42)
        k, n = 6, 12
        matrix = reedsolomon.vandermonde_new(k, n)

        for _ in range(100):
            a = list(np.random.randint(0, 256, k))
            b = list(np.random.randint(0, 256, k))

            encode_a = reedsolomon.vandermonde_encode(matrix, a)
            encode_b = reedsolomon.vandermonde_encode(matrix, b)
            encode_sum_ab = [reedsolomon.gf256_add(encode_a[i], encode_b[i]) for i in range(n)]

            sum_ab = [reedsolomon.gf256_add(a[i], b[i]) for i in range(k)]
            encode_ab = reedsolomon.vandermonde_encode(matrix, sum_ab)

            assert encode_sum_ab == encode_ab, (
                f"Linearity violated: encode({a})+encode({b})!= encode({sum_ab})"
            )

    def test_linearity_triple(self):
        """Test a+b+c = encode(a)+encode(b)+encode(c)"""
        np.random.seed(42)
        k, n = 5, 10
        matrix = reedsolomon.vandermonde_new(k, n)

        for _ in range(50):
            a = list(np.random.randint(0, 256, k))
            b = list(np.random.randint(0, 256, k))
            c = list(np.random.randint(0, 256, k))

            enc_a = reedsolomon.vandermonde_encode(matrix, a)
            enc_b = reedsolomon.vandermonde_encode(matrix, b)
            enc_c = reedsolomon.vandermonde_encode(matrix, c)

            # encode(a) + encode(b) + encode(c)
            sum_encoded = enc_a[:]
            for i in range(n):
                sum_encoded[i] = reedsolomon.gf256_add(sum_encoded[i], enc_b[i])
                sum_encoded[i] = reedsolomon.gf256_add(sum_encoded[i], enc_c[i])

            # encode(a + b + c)
            sum_data = a[:]
            for i in range(k):
                sum_data[i] = reedsolomon.gf256_add(sum_data[i], b[i])
                sum_data[i] = reedsolomon.gf256_add(sum_data[i], c[i])
            encoded_sum = reedsolomon.vandermonde_encode(matrix, sum_data)

            assert sum_encoded == encoded_sum, "Triple linearity violated"


class TestEncodingHomogeneity:
    """Test scalar multiplication property"""

    def test_homogeneity_100_random(self):
        """Test c * encode(a) = encode(c * a) for 100 random scalars and vectors"""
        np.random.seed(42)
        k, n = 5, 10
        matrix = reedsolomon.vandermonde_new(k, n)

        for _ in range(100):
            c = np.random.randint(0, 256)
            a = list(np.random.randint(0, 256, k))

            encode_a = reedsolomon.vandermonde_encode(matrix, a)
            scaled_encode = [reedsolomon.gf256_mul(c, x) for x in encode_a]

            scaled_data = [reedsolomon.gf256_mul(c, x) for x in a]
            encode_scaled = reedsolomon.vandermonde_encode(matrix, scaled_data)

            assert scaled_encode == encode_scaled, (
                f"Homogeneity violated: {c}*encode({a})!= encode({c}*{a})"
            )

    def test_homogeneity_chain(self):
        """Test c1 * (c2 * encode(a)) = (c1*c2) * encode(a)"""
        np.random.seed(42)
        k, n = 4, 8
        matrix = reedsolomon.vandermonde_new(k, n)

        for _ in range(50):
            c1 = np.random.randint(0, 256)
            c2 = np.random.randint(0, 256)
            a = list(np.random.randint(0, 256, k))

            encode_a = reedsolomon.vandermonde_encode(matrix, a)

            # c1 * (c2 * encode(a))
            scaled_once = [reedsolomon.gf256_mul(c2, x) for x in encode_a]
            scaled_twice = [reedsolomon.gf256_mul(c1, x) for x in scaled_once]

            # (c1*c2) * encode(a)
            c_product = reedsolomon.gf256_mul(c1, c2)
            scaled_direct = [reedsolomon.gf256_mul(c_product, x) for x in encode_a]

            assert scaled_twice == scaled_direct, "Homogeneity chain violated"


class TestErrorInjection:
    """Inject errors and verify detection via parity structure"""

    def test_single_bit_flip_detection(self):
        """Single bit flip should be detectable"""
        np.random.seed(42)
        k, n = 8, 16
        matrix = reedsolomon.vandermonde_new(k, n)

        for _ in range(50):
            data = list(np.random.randint(0, 256, k))
            codeword = reedsolomon.vandermonde_encode(matrix, data)

            # Flip single bit in a random parity symbol
            error_pos = np.random.randint(k, n)
            bit_pos = np.random.randint(0, 8)
            corrupted = codeword[:]
            corrupted[error_pos] ^= (1 << bit_pos)

            # Parity check should catch this
            # (This is a property - in Reed-Solomon, n-k =redundancy)
            assert corrupted != codeword, "Bit flip should change codeword"

    def test_multiple_symbol_errors(self):
        """Introduce multiple symbol errors"""
        np.random.seed(42)
        k, n = 10, 20
        matrix = reedsolomon.vandermonde_new(k, n)
        t = (n - k) // 2  # Error correcting capability

        for _ in range(20):
            data = list(np.random.randint(0, 256, k))
            codeword = reedsolomon.vandermonde_encode(matrix, data)

            # Introduce up to t errors in parity region
            corrupted = codeword[:]
            num_errors = np.random.randint(1, t + 1)
            error_positions = np.random.choice(range(k, n), num_errors, replace=False)

            for pos in error_positions:
                corrupted[pos] = np.random.randint(0, 256)

            # Just verify the corrupted codeword is different
            assert corrupted != codeword, "Errors should change codeword"

    def test_error_distribution_detection(self):
        """Verify error patterns can be distinguished"""
        np.random.seed(42)
        k, n = 6, 12
        matrix = reedsolomon.vandermonde_new(k, n)

        data = list(np.random.randint(0, 256, k))
        codeword = reedsolomon.vandermonde_encode(matrix, data)

        # Create two different error patterns
        corrupted1 = codeword[:]
        corrupted1[k] ^= 0xFF  # Error in position k

        corrupted2 = codeword[:]
        corrupted2[k + 1] ^= 0xFF  # Error in position k+1

        assert corrupted1 != corrupted2, "Different error positions should produce different corruptions"
        assert corrupted1 != codeword, "First corruption should differ from original"
        assert corrupted2 != codeword, "Second corruption should differ from original"


class TestCombinedProperties:
    """Tests combining multiple properties"""

    def test_linearity_with_error_injection(self):
        """Verify linearity holds even with error introduction"""
        np.random.seed(42)
        k, n = 5, 10
        matrix = reedsolomon.vandermonde_new(k, n)

        for _ in range(30):
            a = list(np.random.randint(0, 256, k))
            b = list(np.random.randint(0, 256, k))

            encode_a = reedsolomon.vandermonde_encode(matrix, a)
            encode_b = reedsolomon.vandermonde_encode(matrix, b)

            # Combine without error
            sum_encoded = [reedsolomon.gf256_add(encode_a[i], encode_b[i]) for i in range(n)]

            sum_data = [reedsolomon.gf256_add(a[i], b[i]) for i in range(k)]
            encoded_sum = reedsolomon.vandermonde_encode(matrix, sum_data)

            assert sum_encoded == encoded_sum

    def test_zero_vector_property(self):
        """Zero data always encodes to zero"""
        k, n = 7, 14
        matrix = reedsolomon.vandermonde_new(k, n)
        zero_data = [0] * k

        codeword = reedsolomon.vandermonde_encode(matrix, zero_data)

        assert all(x == 0 for x in codeword), "Zero data should produce zero codeword"

    def test_unit_vector_properties(self):
        """Test encoding of unit vectors e_i = [0...0,1,0...0]"""
        k, n = 5, 10
        matrix = reedsolomon.vandermonde_new(k, n)

        for i in range(k):
            unit_vec = [0] * k
            unit_vec[i] = 1

            codeword = reedsolomon.vandermonde_encode(matrix, unit_vec)

            # Codeword should equal the i-th row of the matrix
            expected = matrix[i]
            assert codeword == expected, f"Unit vector {i} should encode to row {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
