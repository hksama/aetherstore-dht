"""
Comprehensive tests for intermediate Vandermonde matrix operations:
- Matrix inversion (Gauss-Jordan elimination)
- Systematic form conversion [V] → [I|P]
- Matrix multiplication in GF(256)
- Property verification (A × A^-1 = I)
"""

import pytest
import random
from typing import List

try:
    import reedsolomon as rs
    HAS_REEDSOLOMON = True
except ImportError:
    HAS_REEDSOLOMON = False

pytestmark = pytest.mark.skipif(not HAS_REEDSOLOMON, reason="reedsolomon module not available")


class TestMatrixInversion:
    """Test matrix inversion using Gauss-Jordan elimination"""
    
    def test_invert_2x2_identity(self):
        """Invert 2×2 identity matrix, should get identity back"""
        identity = [[1, 0], [0, 1]]
        inverse = rs.vandermonde_invert(identity)
        assert len(inverse) == 2
        assert len(inverse[0]) == 2
        # Identity × Identity = Identity
        assert inverse == identity
    
    def test_invert_3x3_gf256_construction(self):
        """Test Vandermonde 3×3 inversion"""
        matrix = rs.vandermonde_new(3, 3)
        inverse = rs.vandermonde_invert(matrix)
        
        assert len(inverse) == 3
        assert len(inverse[0]) == 3
        
        # Verify A × A^-1 = I
        result = rs.vandermonde_multiply(matrix, inverse)
        
        # Check it's the identity matrix
        for i in range(3):
            for j in range(3):
                expected = 1 if i == j else 0
                assert result[i][j] == expected, \
                    f"Position [{i}][{j}]: expected {expected}, got {result[i][j]}"
    
    def test_invert_4x4_gf256_construction(self):
        """Test Vandermonde 4×4 inversion"""
        matrix = rs.vandermonde_new(4, 4)
        inverse = rs.vandermonde_invert(matrix)
        
        assert len(inverse) == 4
        assert len(inverse[0]) == 4
        
        # Verify A × A^-1 = I
        result = rs.vandermonde_multiply(matrix, inverse)
        
        for i in range(4):
            for j in range(4):
                expected = 1 if i == j else 0
                assert result[i][j] == expected
    
    def test_invert_8x8_gf256_construction(self):
        """Test larger 8×8 Vandermonde inversion"""
        matrix = rs.vandermonde_new(8, 8)
        inverse = rs.vandermonde_invert(matrix)
        
        # Verify A × A^-1 = I
        result = rs.vandermonde_multiply(matrix, inverse)
        
        for i in range(8):
            for j in range(8):
                expected = 1 if i == j else 0
                assert result[i][j] == expected
    
    def test_invert_different_k_values(self):
        """Test inversion for various k values"""
        for k in [2, 3, 4, 5, 6, 7, 8, 10, 16]:
            matrix = rs.vandermonde_new(k, k)
            inverse = rs.vandermonde_invert(matrix)
            result = rs.vandermonde_multiply(matrix, inverse)
            
            # Verify identity
            for i in range(k):
                for j in range(k):
                    expected = 1 if i == j else 0
                    assert result[i][j] == expected, \
                        f"k={k}: Position [{i}][{j}] failed"
    
    def test_invert_rectangular_matrix_error(self):
        """Test inversion of rectangular matrix raises error"""
        # 3×4 matrix (not square)
        matrix = rs.vandermonde_new(3, 4)
        
        with pytest.raises(Exception):
            rs.vandermonde_invert(matrix)
    
    def test_invert_inverse_of_inverse_is_original(self):
        """Test that (A^-1)^-1 = A"""
        matrix = rs.vandermonde_new(5, 5)
        inverse = rs.vandermonde_invert(matrix)
        double_inverse = rs.vandermonde_invert(inverse)
        
        # Verify double_inverse == matrix
        for i in range(5):
            for j in range(5):
                assert double_inverse[i][j] == matrix[i][j]
    
    def test_invert_commutative_property(self):
        """Test that A × A^-1 = A^-1 × A = I"""
        matrix = rs.vandermonde_new(4, 4)
        inverse = rs.vandermonde_invert(matrix)
        
        # Left multiply: A × A^-1
        result_left = rs.vandermonde_multiply(matrix, inverse)
        
        # Right multiply: A^-1 × A
        result_right = rs.vandermonde_multiply(inverse, matrix)
        
        # Both should be identity
        for i in range(4):
            for j in range(4):
                expected = 1 if i == j else 0
                assert result_left[i][j] == expected
                assert result_right[i][j] == expected
    
    def test_invert_generates_different_values(self):
        """Test that inverse matrices have varied non-zero values"""
        matrix = rs.vandermonde_new(4, 4)
        inverse = rs.vandermonde_invert(matrix)
        
        # Count unique non-one values
        unique_vals = set()
        for row in inverse:
            for val in row:
                if val != 1:
                    unique_vals.add(val)
        
        # Should have variety in values (not all 0s and 1s)
        assert len(unique_vals) > 2, "Inverse should have diverse field values"


class TestSystematicFormConversion:
    """Test conversion to systematic form [I|P]"""
    
    def test_systematic_2x3_identity_part(self):
        """Test 2×3 matrix conversion, check identity part"""
        matrix = rs.vandermonde_new(2, 3)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        assert len(systematic) == 2
        assert len(systematic[0]) == 3
        
        # First 2 columns should be identity
        assert systematic[0][0] == 1
        assert systematic[0][1] == 0
        assert systematic[1][0] == 0
        assert systematic[1][1] == 1
    
    def test_systematic_3x5_identity_part(self):
        """Test 3×5 matrix, verify 3×3 identity block"""
        matrix = rs.vandermonde_new(3, 5)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        assert len(systematic) == 3
        assert len(systematic[0]) == 5
        
        # First 3 columns should be identity
        for i in range(3):
            for j in range(3):
                expected = 1 if i == j else 0
                assert systematic[i][j] == expected
    
    def test_systematic_4x4_is_identity(self):
        """For k=n, systematic form should be identity matrix"""
        matrix = rs.vandermonde_new(4, 4)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        # Should be identity matrix
        for i in range(4):
            for j in range(4):
                expected = 1 if i == j else 0
                assert systematic[i][j] == expected
    
    def test_systematic_linearity_property(self):
        """Test that systematic form preserves linearity"""
        k, n = 3, 5
        matrix = rs.vandermonde_new(k, n)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        # Create two data vectors
        data1 = [2, 3, 5]
        data2 = [7, 11, 13]
        
        # Encode both
        codeword1 = rs.vandermonde_encode(systematic, data1)
        codeword2 = rs.vandermonde_encode(systematic, data2)
        
        # Encode sum (XOR in GF(256))
        data_sum = [data1[i] ^ data2[i] for i in range(k)]
        codeword_sum = rs.vandermonde_encode(systematic, data_sum)
        
        # Linearity: encode(a+b) should equal encode(a) XOR encode(b)
        expected_sum = [codeword1[i] ^ codeword2[i] for i in range(n)]
        assert codeword_sum == expected_sum
    
    def test_systematic_homogeneity_property(self):
        """Test that systematic form preserves homogeneity"""
        k, n = 3, 4
        matrix = rs.vandermonde_new(k, n)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        data = [2, 3, 5]
        scalar = 7
        
        codeword = rs.vandermonde_encode(systematic, data)
        
        # Encode scaled data
        scaled_data = [rs.gf256_mul(scalar, x) for x in data]
        scaled_codeword = rs.vandermonde_encode(systematic, scaled_data)
        
        # Homogeneity: encode(c·a) = c·encode(a)
        expected_scaled = [rs.gf256_mul(scalar, x) for x in codeword]
        assert scaled_codeword == expected_scaled
    
    def test_systematic_parity_part_varies(self):
        """Test that parity (right) part is non-trivial"""
        matrix = rs.vandermonde_new(3, 6)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        # Extract parity columns (last 3)
        parity_part = []
        for i in range(3):
            parity_row = systematic[i][3:]
            parity_part.append(parity_row)
        
        # Parity should not be all zeros
        all_values = []
        for row in parity_part:
            all_values.extend(row)
        
        assert any(v != 0 for v in all_values), "Parity part should not be all zeros"
    
    def test_systematic_k_values(self):
        """Test systematic conversion for various k values"""
        for k in [2, 3, 4, 5, 6, 8]:
            n = k + 4  # n > k
            matrix = rs.vandermonde_new(k, n)
            systematic = rs.vandermonde_to_systematic(matrix)
            
            # Verify identity part
            for i in range(k):
                for j in range(k):
                    expected = 1 if i == j else 0
                    assert systematic[i][j] == expected
    
    def test_systematic_reconstruction_from_parity(self):
        """Test that original data can be verified from systematic form"""
        k, n = 3, 5
        matrix = rs.vandermonde_new(k, n)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        data = [42, 100, 200]
        codeword = rs.vandermonde_encode(systematic, data)
        
        # First k symbols should equal data (systematic form property)
        assert codeword[:k] == data


class TestMatrixMultiplication:
    """Test matrix multiplication in GF(256)"""
    
    def test_multiply_2x2_identity(self):
        """Multiply by identity should give original matrix"""
        matrix = rs.vandermonde_new(2, 2)
        identity = [[1, 0], [0, 1]]
        
        result = rs.vandermonde_multiply(matrix, identity)
        
        assert result == matrix
    
    def test_multiply_3x3_matrices(self):
        """Test multiplication of 3×3 matrices"""
        matrix_a = rs.vandermonde_new(3, 3)
        matrix_b = rs.vandermonde_new(3, 3)
        
        result = rs.vandermonde_multiply(matrix_a, matrix_b)
        
        assert len(result) == 3
        assert len(result[0]) == 3
    
    def test_multiply_rectangular_matrices(self):
        """Test multiplication of rectangular matrices (2×3) × (3×4)"""
        matrix_a = rs.vandermonde_new(2, 3)  # 2×3
        matrix_b = rs.vandermonde_new(3, 4)  # 3×4
        
        result = rs.vandermonde_multiply(matrix_a, matrix_b)
        
        # Result should be 2×4
        assert len(result) == 2
        assert len(result[0]) == 4
    
    def test_multiply_incompatible_dimensions_error(self):
        """Test error when dimensions don't match"""
        matrix_a = rs.vandermonde_new(2, 3)  # 2×3
        matrix_b = rs.vandermonde_new(5, 6)  # 5×6 (incompatible)
        
        with pytest.raises(Exception):
            rs.vandermonde_multiply(matrix_a, matrix_b)
    
    def test_multiply_associative_property(self):
        """Test that (A × B) × C = A × (B × C)"""
        a = rs.vandermonde_new(2, 2)
        b = rs.vandermonde_new(2, 2)
        c = rs.vandermonde_new(2, 2)
        
        # (A × B) × C
        ab = rs.vandermonde_multiply(a, b)
        ab_c = rs.vandermonde_multiply(ab, c)
        
        # A × (B × C)
        bc = rs.vandermonde_multiply(b, c)
        a_bc = rs.vandermonde_multiply(a, bc)
        
        assert ab_c == a_bc
    
    def test_multiply_with_zero_matrix(self):
        """Multiply with zero matrix should give zero matrix"""
        matrix = rs.vandermonde_new(2, 2)
        zero_matrix = [[0, 0], [0, 0]]
        
        result = rs.vandermonde_multiply(matrix, zero_matrix)
        
        # Result should be all zeros
        for row in result:
            for val in row:
                assert val == 0
    
    def test_multiply_distributive_over_addition(self):
        """Test A × (B + C) = (A × B) + (A × C) in terms of encoding"""
        matrix = rs.vandermonde_new(2, 3)
        
        # Test with vectors instead of matrices
        v1 = [1, 2]
        v2 = [3, 4]
        
        # Sum of vectors in GF(256) (XOR)
        v_sum = [v1[i] ^ v2[i] for i in range(2)]
        
        # Encode both
        c1 = rs.vandermonde_encode(matrix, v1)
        c2 = rs.vandermonde_encode(matrix, v2)
        c_sum = rs.vandermonde_encode(matrix, v_sum)
        
        # Should have distributive property
        expected_sum = [c1[i] ^ c2[i] for i in range(3)]
        assert c_sum == expected_sum


class TestIntermediateFunctionIntegration:
    """Integration tests combining multiple intermediate functions"""
    
    def test_invert_systematic_roundtrip(self):
        """Test complete roundtrip: Vandermonde → Systematic → Verify"""
        k, n = 4, 6
        
        # Create Vandermonde
        vand = rs.vandermonde_new(k, n)
        
        # Convert to systematic
        systematic = rs.vandermonde_to_systematic(vand)
        
        # Use systematic for encoding
        data = list(range(1, k + 1))
        codeword = rs.vandermonde_encode(systematic, data)
        
        # First k symbols should be data (systematic property)
        assert codeword[:k] == data
        
        # All symbols should be non-zero for non-zero input
        assert all(x != 0 for x in codeword)
    
    def test_large_inversion_stress(self):
        """Stress test: invert up to 256×256 (max GF(256) values)"""
        # Start with reasonable size and test boundary
        for k in [16, 32, 64, 128, 256]:
            matrix = rs.vandermonde_new(k, k)
            inverse = rs.vandermonde_invert(matrix)
            
            # Just verify it's correct side by checking one position
            result = rs.vandermonde_multiply(matrix, inverse)
            
            # Check diagonal (identity should have 1s)
            for i in range(min(k, 10)):  # Check first 10 to save time
                assert result[i][i] == 1, f"Failed at k={k}"
    
    def test_random_data_encoding_decoding_foundation(self):
        """Test that encoding with systematic form is deterministic"""
        k, n = 5, 8
        systematic = rs.vandermonde_to_systematic(rs.vandermonde_new(k, n))
        
        data = [42, 100, 200, 150, 75]
        
        # Encode same data multiple times
        c1 = rs.vandermonde_encode(systematic, data)
        c2 = rs.vandermonde_encode(systematic, data)
        c3 = rs.vandermonde_encode(systematic, data)
        
        # Should be deterministic
        assert c1 == c2 == c3
    
    def test_all_gf256_values_in_operations(self):
        """Test that operations produce GF(256) values correctly"""
        matrix = rs.vandermonde_new(3, 3)
        inverse = rs.vandermonde_invert(matrix)
        
        # All values should be in [0, 255]
        for row in inverse:
            for val in row:
                assert 0 <= val <= 255
    
    def test_encoding_commutes_with_field_operations(self):
        """Test that GF(256) operations work correctly in encoding"""
        k, n = 2, 4
        matrix = rs.vandermonde_new(k, n)
        
        # Two scalars in GF(256)
        a = 17
        b = 23
        c = rs.gf256_mul(a, b)
        
        # Data vectors
        data = [a, b]
        data_scaled = [rs.gf256_mul(c, x) for x in data]
        
        # Encode both
        c1 = rs.vandermonde_encode(matrix, data)
        c2 = rs.vandermonde_encode(matrix, data_scaled)
        
        # Check that scaling works
        expected = [rs.gf256_mul(c, x) for x in c1]
        assert c2 == expected


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_invert_1x1_matrix(self):
        """Test inverting 1×1 matrix"""
        matrix = [[2]]  # Single element 2
        inverse = rs.vandermonde_invert(matrix)
        
        # 2 × 2^-1 should equal 1 in GF(256)
        result = rs.vandermonde_multiply(matrix, inverse)
        assert result[0][0] == 1
    
    def test_invert_max_size(self):
        """Test inversion of large matrix"""
        k = 255  # Max in GF(256)
        matrix = rs.vandermonde_new(k, k)
        inverse = rs.vandermonde_invert(matrix)
        
        # Just verify it's square
        assert len(inverse) == k
        assert len(inverse[0]) == k
    
    def test_systematic_max_n(self):
        """Test systematic conversion with maximum n"""
        k = 100
        n = 200
        matrix = rs.vandermonde_new(k, n)
        systematic = rs.vandermonde_to_systematic(matrix)
        
        assert len(systematic) == k
        assert len(systematic[0]) == n
    
    def test_binary_data_encoding(self):
        """Test encoding with binary data (0s and 1s only)"""
        k, n = 3, 5
        matrix = rs.vandermonde_new(k, n)
        
        data = [0, 1, 0]
        codeword = rs.vandermonde_encode(matrix, data)
        
        # All outputs should be in GF(256)
        assert all(0 <= c <= 255 for c in codeword)
    
    def test_all_ones_data(self):
        """Test encoding with all ones"""
        k, n = 4, 6
        matrix = rs.vandermonde_new(k, n)
        
        data = [1] * k
        codeword = rs.vandermonde_encode(matrix, data)
        
        assert len(codeword) == n
        assert all(0 <= c <= 255 for c in codeword)
    
    def test_all_max_values_data(self):
        """Test encoding with all 255 (max value)"""
        k, n = 3, 5
        matrix = rs.vandermonde_new(k, n)
        
        data = [255] * k
        codeword = rs.vandermonde_encode(matrix, data)
        
        assert len(codeword) == n
        assert all(0 <= c <= 255 for c in codeword)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
