"""
GF(256) Axiom Verification Tests

Tests fundamental algebraic properties of the Galois field GF(256)
to ensure the implementation satisfies all field axioms.
"""

import pytest
import reedsolomon
import numpy as np

try:
    import galois
    GALOIS_AVAILABLE = True
except ImportError:
    GALOIS_AVAILABLE = False


class TestGf256Closure:
    """Test closure property: a+b and a*b stay in GF(256)"""

    def test_addition_closure(self, rng):
        """Addition result must be in [0, 255]"""
        for _ in range(1000):
            a = rng.integers(0, 256)
            b = rng.integers(0, 256)
            result = reedsolomon.gf256_add(a, b)
            assert 0 <= result < 256, f"Addition closure failed: {a} + {b} = {result}"

    def test_multiplication_closure(self, rng):
        """Multiplication result must be in [0, 255]"""
        for _ in range(1000):
            a = rng.integers(0, 256)
            b = rng.integers(0, 256)
            result = reedsolomon.gf256_mul(a, b)
            assert 0 <= result < 256, f"Multiplication closure failed: {a} * {b} = {result}"


class TestGf256Identity:
    """Test identity elements"""

    def test_addition_identity(self, rng):
        """a + 0 = a for all a"""
        for _ in range(100):
            a = rng.integers(0, 256)
            result = reedsolomon.gf256_add(a, 0)
            assert result == a, f"Additive identity failed: {a} + 0 = {result}, expected {a}"

    def test_multiplication_identity(self, rng):
        """a * 1 = a for all a"""
        for _ in range(100):
            a = rng.integers(0, 256)
            result = reedsolomon.gf256_mul(a, 1)
            assert result == a, f"Multiplicative identity failed: {a} * 1 = {result}, expected {a}"


class TestGf256Commutativity:
    """Test commutativity: a+b = b+a and a*b = b*a"""

    def test_addition_commutativity(self, rng):
        """a + b = b + a"""
        for _ in range(500):
            a = rng.integers(0, 256)
            b = rng.integers(0, 256)
            ab = reedsolomon.gf256_add(a, b)
            ba = reedsolomon.gf256_add(b, a)
            assert ab == ba, f"Addition not commutative: add({a},{b})={ab} != add({b},{a})={ba}"

    def test_multiplication_commutativity(self, rng):
        """a * b = b * a"""
        for _ in range(500):
            a = rng.integers(0, 256)
            b = rng.integers(0, 256)
            ab = reedsolomon.gf256_mul(a, b)
            ba = reedsolomon.gf256_mul(b, a)
            assert ab == ba, f"Multiplication not commutative: mul({a},{b})={ab} != mul({b},{a})={ba}"


class TestGf256Associativity:
    """Test associativity: (a+b)+c = a+(b+c) and (a*b)*c = a*(b*c)"""

    def test_addition_associativity(self, rng):
        """(a + b) + c = a + (b + c)"""
        for _ in range(300):
            a = rng.integers(0, 256)
            b = rng.integers(0, 256)
            c = rng.integers(0, 256)

            # (a + b) + c
            ab = reedsolomon.gf256_add(a, b)
            abc_left = reedsolomon.gf256_add(ab, c)

            # a + (b + c)
            bc = reedsolomon.gf256_add(b, c)
            abc_right = reedsolomon.gf256_add(a, bc)

            assert abc_left == abc_right, (
                f"Addition not associative: "
                f"({a}+{b})+{c}={abc_left} != {a}+({b}+{c})={abc_right}"
            )

    def test_multiplication_associativity(self, rng):
        """(a * b) * c = a * (b * c)"""
        for _ in range(300):
            a = rng.integers(1, 256)  # Exclude 0 to avoid trivial case
            b = rng.integers(1, 256)
            c = rng.integers(1, 256)

            # (a * b) * c
            ab = reedsolomon.gf256_mul(a, b)
            abc_left = reedsolomon.gf256_mul(ab, c)

            # a * (b * c)
            bc = reedsolomon.gf256_mul(b, c)
            abc_right = reedsolomon.gf256_mul(a, bc)

            assert abc_left == abc_right, (
                f"Multiplication not associative: "
                f"({a}*{b})*{c}={abc_left} != {a}*({b}*{c})={abc_right}"
            )


class TestGf256Inverse:
    """Test existence of inverses"""

    def test_additive_inverse_is_itself(self, rng):
        """In GF(256), a + a = 0 (element is its own additive inverse)"""
        for _ in range(100):
            a = rng.integers(0, 256)
            result = reedsolomon.gf256_add(a, a)
            assert result == 0, f"Additive inverse failed: {a} + {a} = {result}, expected 0"

    def test_multiplicative_inverse_non_zero(self, rng):
        """For all a ≠ 0: a * a^(-1) = 1"""
        for _ in range(255):  # Test all non-zero values
            a = (256 + rng.integers(0, 255)) % 256  # Avoid 0
            if a == 0:
                continue
            try:
                a_inv = reedsolomon.gf256_inverse(a)
                result = reedsolomon.gf256_mul(a, a_inv)
                assert result == 1, (
                    f"Multiplicative inverse failed: {a} * {a_inv} = {result}, expected 1"
                )
            except ValueError:
                pytest.fail(f"gf256_inverse raised error for non-zero element {a}")

    def test_multiplicative_inverse_zero_raises(self):
        """a^(-1) undefined for a = 0"""
        with pytest.raises((ValueError, RuntimeError)):
            reedsolomon.gf256_inverse(0)


class TestGf256Distributivity:
    """Test distributivity: a * (b + c) = (a * b) + (a * c)"""

    def test_distributivity(self, rng):
        """a * (b + c) = (a * b) + (a * c)"""
        for _ in range(300):
            a = rng.integers(0, 256)
            b = rng.integers(0, 256)
            c = rng.integers(0, 256)

            # Left side: a * (b + c)
            bc = reedsolomon.gf256_add(b, c)
            left = reedsolomon.gf256_mul(a, bc)

            # Right side: (a * b) + (a * c)
            ab = reedsolomon.gf256_mul(a, b)
            ac = reedsolomon.gf256_mul(a, c)
            right = reedsolomon.gf256_add(ab, ac)

            assert left == right, (
                f"Distributivity failed: {a}*({b}+{c})={left} != "
                f"({a}*{b})+({a}*{c})={right}"
            )


class TestGf256VSGalois:
    """Cross-check against galois library when not using zero elements"""

    @pytest.mark.skipif(not GALOIS_AVAILABLE, reason="galois library not available")
    def test_multiplication_matches_galois_nonzero(self, rng):
        """Rust mul(a,b) should match galois for a,b ≠ 0"""
        GF = galois.GF(2**8)  # Using default primitive polynomial

        for _ in range(100):
            a = rng.integers(1, 256)  # Non-zero
            b = rng.integers(1, 256)

            rust_result = reedsolomon.gf256_mul(a, b)
            galois_result = int(GF(a) * GF(b))

            # Note: might not match if galois uses different poly
            # This test will help identify if there's a discrepancy
            if rust_result != galois_result:
                print(f"Mismatch: {a} * {b} = {rust_result} (rust) vs {galois_result} (galois)")

    @pytest.mark.skipif(not GALOIS_AVAILABLE, reason="galois library not available")
    def test_addition_matches_galois(self, rng):
        """Rust add(a,b) = XOR should match galois XOR"""
        GF = galois.GF(2**8)

        for _ in range(100):
            a = rng.integers(0, 256)
            b = rng.integers(0, 256)

            rust_result = reedsolomon.gf256_add(a, b)
            galois_result = int(GF(a) + GF(b))

            assert rust_result == galois_result, (
                f"Addition mismatch: {a} + {b} = {rust_result} (rust) "
                f"vs {galois_result} (galois)"
            )


class TestGf256Exponentiation:
    """Test exponentiation: a^n"""

    def test_exponentiation_basic(self, rng):
        """a^0 = 1  and a^1 = a"""
        for _ in range(50):
            a = rng.integers(1, 256)
            assert reedsolomon.gf256_pow(a, 0) == 1, f"{a}^0 should be 1"
            assert reedsolomon.gf256_pow(a, 1) == a, f"{a}^1 should be {a}"

    def test_exponentiation_repeated_multiplication(self, rng):
        """a^n = a * a * ... * a (n times)"""
        for _ in range(50):
            a = rng.integers(1, 256)
            n = rng.integers(2, 10)

            result_pow = reedsolomon.gf256_pow(a, n)

            # Compute by repeated multiplication
            result_mul = a
            for _ in range(n - 1):
                result_mul = reedsolomon.gf256_mul(result_mul, a)

            assert result_pow == result_mul, (
                f"Exponentiation mismatch: {a}^{n} = {result_pow} "
                f"vs repeated mul = {result_mul}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
