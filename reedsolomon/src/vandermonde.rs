use crate::Gf256;

pub(crate) struct VandermondeMatrix {
    data: Vec<Gf256>,
    rows: usize,
    cols: usize,
}
const ALPHA: Gf256 = Gf256(2);

impl VandermondeMatrix {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        let mut data = vec![Gf256::ZERO; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                // Vandermonde entry: α^(i*j)
                data[i * cols + j] = ALPHA.pow((i * j) as u32);
            }
        }
        VandermondeMatrix { data, rows, cols }
    }

    fn get(&self, row: usize, col: usize) -> Gf256 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, val: Gf256) {
        self.data[row * self.cols + col] = val;
    }
    
    /// Get number of rows
    pub(crate) fn num_rows(&self) -> usize {
        self.rows
    }
    
    /// Get number of columns
    pub(crate) fn num_cols(&self) -> usize {
        self.cols
    }
    
    /// Multiply two matrices: result = self × other
    pub(crate) fn multiply(&self, other: &VandermondeMatrix) -> Result<Vec<Vec<u8>>, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Cannot multiply matrices: first has {} cols, second has {} rows",
                self.cols, other.rows
            ));
        }
        
        let mut result_data = vec![Gf256::ZERO; self.rows * other.cols];
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Gf256::ZERO;
                for k in 0..self.cols {
                    sum = sum + (self.get(i, k) * other.get(k, j));
                }
                result_data[i * other.cols + j] = sum;
            }
        }
        
        // Convert to Vec<Vec<u8>>
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(other.cols);
            for j in 0..other.cols {
                row.push(result_data[i * other.cols + j].0);
            }
            result.push(row);
        }
        
        Ok(result)
    }
    
    /// Export matrix as Vec of Vec for FFI
    pub(crate) fn to_vec(&self) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for j in 0..self.cols {
                row.push(self.get(i, j).0);
            }
            result.push(row);
        }
        result
    }
    
    /// Create from Vec for FFI
    pub(crate) fn from_vec(matrix: Vec<Vec<u8>>) -> Result<Self, String> {
        if matrix.is_empty() {
            return Err("Matrix is empty".to_string());
        }
        let rows = matrix.len();
        let cols = matrix[0].len();
        
        let mut data = Vec::with_capacity(rows * cols);
        for row in matrix {
            if row.len() != cols {
                return Err("Inconsistent row size".to_string());
            }
            for val in row {
                data.push(Gf256(val));
            }
        }
        
        Ok(VandermondeMatrix { data, rows, cols })
    }
    
    /// Encode data and return as Vec
    pub(crate) fn encode_vec(&self, data: Vec<u8>) -> Result<Vec<u8>, String> {
        if data.len() != self.rows {
            return Err("Data length must match number of rows".to_string());
        }
        
        let gf_data_vec: Vec<Gf256> = data.into_iter().map(Gf256).collect();
        let mut codeword = vec![Gf256::ZERO; self.cols];
        
        for j in 0..self.cols {
            let mut sum = Gf256::ZERO;
            for i in 0..gf_data_vec.len() {
                sum = sum + (gf_data_vec[i] * self.get(i, j));
            }
            codeword[j] = sum;
        }
        
        Ok(codeword.iter().map(|g| g.0).collect())
    }

    /// Invert a k×k matrix using Gauss-Jordan elimination
    /// Returns Ok(inverse) if invertible, Err if singular/non-square
    pub(crate)fn invert_kxk(&self) -> Result<VandermondeMatrix, String> {
        if self.rows != self.cols {
            return Err(format!(
                "Cannot invert non-square matrix {}x{}",
                self.rows, self.cols
            ));
        }
        
        let k = self.rows;
        
        if k == 0 {
            return Err("Cannot invert empty matrix".to_string());
        }
        
        // Create augmented matrix [A | I]
        let mut augmented = vec![Gf256::ZERO; k * 2 * k];
        
        // Copy A into left half
        for i in 0..k {
            for j in 0..k {
                augmented[i * (2 * k) + j] = self.get(i, j);
            }
        }
        
        // Create identity matrix in right half
        for i in 0..k {
            for j in 0..k {
                augmented[i * (2 * k) + (k + j)] = if i == j {
                    Gf256::ONE
                } else {
                    Gf256::ZERO
                };
            }
        }
        
        // Forward elimination with row operations
        for col in 0..k {
            // Find pivot
            let mut pivot_row = None;
            for i in col..k {
                let idx = i * (2 * k) + col;
                if augmented[idx].0 != 0 {
                    pivot_row = Some(i);
                    break;
                }
            }
            
            let pivot_row = match pivot_row {
                Some(r) => r,
                None => return Err("Matrix is singular (not invertible)".to_string()),
            };
            
            // Swap rows if needed
            if pivot_row != col {
                for j in 0..(2 * k) {
                    let tmp = augmented[col * (2 * k) + j];
                    augmented[col * (2 * k) + j] = augmented[pivot_row * (2 * k) + j];
                    augmented[pivot_row * (2 * k) + j] = tmp;
                }
            }
            
            // Scale pivot row to make pivot element 1
            let pivot = augmented[col * (2 * k) + col];
            let pivot_inv = pivot.inverse();
            
            for j in 0..(2 * k) {
                augmented[col * (2 * k) + j] = augmented[col * (2 * k) + j] * pivot_inv;
            }
            
            // Eliminate column in all other rows
            for i in 0..k {
                if i != col {
                    let factor = augmented[i * (2 * k) + col];
                    for j in 0..(2 * k) {
                        let val = augmented[col * (2 * k) + j] * factor;
                        augmented[i * (2 * k) + j] = augmented[i * (2 * k) + j] + val;
                    }
                }
            }
        }
        
        // Extract inverse matrix from right half of augmented matrix
        let mut inverse_data = vec![Gf256::ZERO; k * k];
        for i in 0..k {
            for j in 0..k {
                inverse_data[i * k + j] = augmented[i * (2 * k) + (k + j)];
            }
        }
        
        Ok(VandermondeMatrix {
            data: inverse_data,
            rows: k,
            cols: k,
        })
    }

    /// Convert k×n Vandermonde matrix to systematic form [I | P]
    /// Returns (identity_part, parity_part)
    pub(crate) fn to_systematic_form(&self) -> Result<(Vec<Vec<u8>>, Vec<Vec<u8>>), String> {
        if self.cols < self.rows {
            return Err(format!(
                "Cannot form systematic code: n({}) must be >= k({})",
                self.cols, self.rows
            ));
        }
        
        let k = self.rows;
        let n = self.cols;
        
        // Extract first k×k submatrix (generator part)
        let mut gen_matrix_data = vec![Gf256::ZERO; k * k];
        for i in 0..k {
            for j in 0..k {
                gen_matrix_data[i * k + j] = self.get(i, j);
            }
        }
        
        let gen_matrix = VandermondeMatrix {
            data: gen_matrix_data,
            rows: k,
            cols: k,
        };
        
        // Invert the first k×k submatrix
        let gen_inv = gen_matrix.invert_kxk()?;
        
        // Multiply entire Vandermonde by the inverse: result = A · A_k^{-1}
        let mut systematic_data = vec![Gf256::ZERO; k * n];
        
        for i in 0..k {
            for j in 0..n {
                let mut sum = Gf256::ZERO;
                for l in 0..k {
                    sum = sum + (self.get(i, l) * gen_inv.get(l, j));
                }
                systematic_data[i * n + j] = sum;
            }
        }
        
        let systematic_matrix = VandermondeMatrix {
            data: systematic_data,
            rows: k,
            cols: n,
        };
        
        // Extract identity and parity parts
        let mut identity_part = Vec::new();
        let mut parity_part = Vec::new();
        
        for i in 0..k {
            let mut id_row = Vec::new();
            let mut par_row = Vec::new();
            
            for j in 0..k {
                id_row.push(systematic_matrix.get(i, j).0);
            }
            
            for j in k..n {
                par_row.push(systematic_matrix.get(i, j).0);
            }
            
            identity_part.push(id_row);
            parity_part.push(par_row);
        }
        
        Ok((identity_part, parity_part))
    }

    /// Convert matrix to the form [I | P] where I is the identity matrix and P is the parity part
    /// Returns the full systematic matrix [I | P]
    pub(crate) fn to_systematic_full(&self) -> Result<Vec<Vec<u8>>, String> {
        let (mut identity_part, parity_part) = self.to_systematic_form()?;
        
        for (i, par_row) in parity_part.into_iter().enumerate() {
            identity_part[i].extend(par_row);
        }
        
        Ok(identity_part)
    }

    /// Verify that two matrices are inverses: A × B = I
    fn are_inverses(&self, other: &VandermondeMatrix) -> bool {
        if self.rows != other.cols || self.cols != other.rows {
            return false;
        }
        
        let identity_size = self.rows;
        
        for i in 0..identity_size {
            for j in 0..identity_size {
                let mut sum = Gf256::ZERO;
                for k in 0..self.cols {
                    sum = sum + (self.get(i, k) * other.get(k, j));
                }
                
                let expected = if i == j {
                    Gf256::ONE
                } else {
                    Gf256::ZERO
                };
                
                if sum.0 != expected.0 {
                    return false;
                }
            }
        }
        
        true
    }

    /// Compute the inverse of the Vandermonde matrix using Gaussian elimination
    fn compute_inverse(&self) -> Result<VandermondeMatrix, String> {
        self.invert_kxk()
    }

    /// Compute the generator matrix for Reed-Solomon encoding using the Vandermonde matrix
    fn compute_generator_matrix(&self) -> Result<Vec<Vec<u8>>, String> {
        self.to_systematic_full()
    }

    /// Encode the input data using the generator matrix to produce the codeword
    pub fn encode(&mut self, data: &[Gf256]) {
        let mut codeword = vec![Gf256(0); self.cols];
    }
}
