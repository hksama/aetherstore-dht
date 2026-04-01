"""
Pytest configuration and fixtures for Reed-Solomon testing

Provides:
- galois_gf256: Reference GF(256) implementation for cross-checking (optional)
- rust_reedsolomon: Imported Rust extension module
- rng: Numpy random generator with fixed seed for reproducibility
"""

import pytest
import numpy as np
import reedsolomon

try:
    import galois
    GALOIS_AVAILABLE = True
except ImportError:
    GALOIS_AVAILABLE = False


@pytest.fixture(scope="session")
def galois_gf256():
    """Initialize galois GF(256) field for reference testing"""
    if not GALOIS_AVAILABLE:
        pytest.skip("galois library not available")
    return galois.GF(2**8)


@pytest.fixture(scope="session")
def rust_reedsolomon():
    """Return the compiled Rust reedsolomon module"""
    return reedsolomon


@pytest.fixture(scope="function")
def rng():
    """Random number generator with fixed seed for reproducibility"""
    np.random.seed(42)
    return np.random.default_rng(42)


@pytest.fixture(scope="function")
def gf256_test_vectors():
    """Common GF(256) test vectors"""
    return {
        'multiplication': [
            (0x57, 0x83, 0xC1),  # AES test vector
            (5, 150, 217),        # From original test
            (0, 5, 0),            # Zero multiplication
            (1, 100, 100),        # Identity
            (255, 255, None),     # Will compute
        ],
        'addition': [
            (5, 3, 6),            # XOR
            (0xFF, 0xFF, 0),      # XOR self = 0
            (1, 1, 0),            # Identity
            (0, 5, 5),            # Additive identity
        ],
        'inverse': [
            (1, 1),    # Inverse of 1 is 1
            (2, None), # Will compute
            (255, None),# Will compute
        ]
    }
