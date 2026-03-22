use crate::Gf256;

struct VandermondeMatrix {
    data: Vec<Gf256>,
    rows: usize,
    cols: usize,
}
const ALPHA: Gf256 = Gf256(2);

impl VandermondeMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                if i == 0 && j == 0 {
                    data.push(Gf256(1));
                } else if j == 0 {
                    data.push(Gf256(1));
                }
                data.push(Gf256(2).pow(j as u32));
            }
        }
        VandermondeMatrix { data, rows, cols }
    }

    /// Convert matrix to the form [I | P] where I is the identity matrix and P is the parity part
    /// Is the first k rows and P is the last n-k rows
    fn convert_to_systematic_form(&mut self) {}

    /// Compute the inverse of the Vandermonde matrix using Gaussian elimination
    fn compute_inverse(&mut self) -> VandermondeMatrix {
        VandermondeMatrix::new(self.rows, self.cols)
    }

    /// Compute the generator matrix for Reed-Solomon encoding using the Vandermonde matrix
    fn compute_generator_matrix(&mut self) -> VandermondeMatrix {
        self.convert_to_systematic_form();
        self.compute_inverse();

        VandermondeMatrix::new(self.rows, self.cols)
    }

    /// Encode the input data using the generator matrix to produce the codeword
    pub fn encode(&mut self, data: &[Gf256]) {
        let mut codeword = vec![Gf256(0); self.cols];
    }
}
