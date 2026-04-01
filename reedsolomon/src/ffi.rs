/// PyO3 FFI Layer - All Python/Rust interop in one place
/// 
/// This module handles:
/// - Converting Python types to Rust types
/// - Calling internal Rust functions
/// - Converting Rust results back to Python
/// - All PyO3 decorators and type conversions

use pyo3::prelude::*;
use crate::Gf256;
use crate::vandermonde::VandermondeMatrix;

// ============================================
// GF(256) Field Operations
// ============================================

/// Multiply two GF(256) elements
#[pyfunction]
pub fn gf256_mul(a: u8, b: u8) -> u8 {
    Gf256(a).mul(Gf256(b)).0
}

/// Add two GF(256) elements (XOR)
#[pyfunction]
pub fn gf256_add(a: u8, b: u8) -> u8 {
    (Gf256(a) + Gf256(b)).0
}

/// Compute multiplicative inverse in GF(256)
#[pyfunction]
pub fn gf256_inverse(a: u8) -> PyResult<u8> {
    if a == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Zero has no inverse",
        ));
    }
    Ok(Gf256(a).inverse().0)
}

/// Compute exponentiation in GF(256)
#[pyfunction]
pub fn gf256_pow(a: u8, exp: u32) -> u8 {
    Gf256(a).pow(exp).0
}

// ============================================
// Vandermonde Matrix Operations
// ============================================

/// Create a k×n Vandermonde matrix and return as nested list
#[pyfunction]
pub fn vandermonde_new(k: usize, n: usize) -> PyResult<Vec<Vec<u8>>> {
    if k == 0 || n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Matrix dimensions must be non-zero",
        ));
    }
    let matrix = VandermondeMatrix::new(k, n);
    Ok(matrix.to_vec())
}

/// Encode data using a Vandermonde generator matrix
///
/// Performs matrix-vector multiplication: codeword = data · generator_matrix
#[pyfunction]
pub fn vandermonde_encode(matrix_list: Vec<Vec<u8>>, data: Vec<u8>) -> PyResult<Vec<u8>> {
    let matrix = VandermondeMatrix::from_vec(matrix_list)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    matrix.encode_vec(data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

/// Invert a k×k Vandermonde matrix using Gauss-Jordan elimination
#[pyfunction]
pub fn vandermonde_invert(matrix_list: Vec<Vec<u8>>) -> PyResult<Vec<Vec<u8>>> {
    let matrix = VandermondeMatrix::from_vec(matrix_list)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    let inverse = matrix.invert_kxk()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    Ok(inverse.to_vec())
}

/// Convert a k×n Vandermonde matrix to systematic form [I | P]
#[pyfunction]
pub fn vandermonde_to_systematic(matrix_list: Vec<Vec<u8>>) -> PyResult<Vec<Vec<u8>>> {
    let matrix = VandermondeMatrix::from_vec(matrix_list)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    matrix.to_systematic_full()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

/// Multiply two matrices in GF(256): result = A × B
#[pyfunction]
pub fn vandermonde_multiply(matrix_a: Vec<Vec<u8>>, matrix_b: Vec<Vec<u8>>) -> PyResult<Vec<Vec<u8>>> {
    let a = VandermondeMatrix::from_vec(matrix_a)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    let b = VandermondeMatrix::from_vec(matrix_b)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    if a.num_cols() != b.num_rows() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Cannot multiply matrices: first has {} cols, second has {} rows", a.num_cols(), b.num_rows())
        ));
    }
    
    a.multiply(&b)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

// ============================================
// PyO3 Module Registration
// ============================================

/// Register all PyO3 functions with the Python module
pub fn create_module(_py: Python, m: &PyModule) -> PyResult<()> {
    // GF(256) operations
    m.add_function(wrap_pyfunction!(gf256_mul, m)?)?;
    m.add_function(wrap_pyfunction!(gf256_add, m)?)?;
    m.add_function(wrap_pyfunction!(gf256_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(gf256_pow, m)?)?;
    
    // Vandermonde matrix operations
    m.add_function(wrap_pyfunction!(vandermonde_new, m)?)?;
    m.add_function(wrap_pyfunction!(vandermonde_encode, m)?)?;
    m.add_function(wrap_pyfunction!(vandermonde_invert, m)?)?;
    m.add_function(wrap_pyfunction!(vandermonde_to_systematic, m)?)?;
    m.add_function(wrap_pyfunction!(vandermonde_multiply, m)?)?;
    
    Ok(())
}
