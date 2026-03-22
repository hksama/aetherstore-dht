use core::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Gf256(u8);
mod vandermonde;

impl Gf256 {
    pub const ZERO: Gf256 = Gf256(0);
    pub const ONE: Gf256 = Gf256(1);
    /// Create from raw byte
    pub const fn new(val: u8) -> Self {
        Gf256(val)
    }

    /// Get raw byte
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Create log tables for GF(256) at compile time
    const fn init_tables() -> ([u8; 256], [u8; 256]) {
        // Use 2 as generator (primitive element for GF(256) with polynomial 0x11D)
        // Note: 3 has order 204, NOT 255, so it cannot generate all non-zero elements
        const GENERATOR: u8 = 0x02;
        let mut exp = [0u8; 256];
        let mut log = [0u8; 256];
        log[0] = 0; // log(0) is undefined, set to sentinel
        let mut x = 1u8;
        let mut i = 0;
        while i < 255 {
            exp[i] = x;
            log[x as usize] = i as u8;
            // Multiply x by generator (0x02) in GF(256)
            x = Gf256::mul_naive(x, GENERATOR);
            i += 1;
        }
        exp[255] = exp[0]; // Wrap around for easy modulo arithmetic
        (exp, log)
    }

    /// Precomputed log and antilog tables for GF(256)
    pub const TABLES: ([u8; 256], [u8; 256]) = Self::init_tables();

    /// used for generating log tables, not for general multiplication
    const fn mul_naive(mut a: u8, mut b: u8) -> u8 {
        const POLY: u8 = 0x1D; // lower 8 bits of 0x11D
        let mut res = 0u8;

        let mut i = 0;
        while i < 8 {
            if (b & 1) != 0 {
                res ^= a;
            }

            let carry = a & 0x80; // highest bit
            a <<= 1;

            if carry != 0 {
                a ^= POLY;
            }

            b >>= 1;
            i += 1;
        }

        res
    }

    /// Fn to check if TABLES are correctly initialized at compile time
    const CHECK_TABLES_PRECOMPUTED: () = {
        const TABLES: ([u8; 256], [u8; 256]) = Gf256::init_tables();

        // If this fails, compile-time error
        assert!(TABLES.0[0] == 1);
    };

    /// Fast multiplication using Log/Antilog tables
    pub fn mul(self, other: Gf256) -> Gf256 {
        if self.0 == 0 || other.0 == 0 {
            return Gf256::ZERO;
        }

        let (exp, log) = Self::TABLES;
        // log(a) + log(b) mod 255
        let log_sum = (log[self.0 as usize] as u16 + log[other.0 as usize] as u16) % 255;
        Gf256(exp[log_sum as usize])
    }

    /// Multiplicative Inverse using Log/Antilog tables
    pub fn inverse(self) -> Gf256 {
        if self.0 == 0 {
            panic!("Zero has no inverse");
        }
        let (exp, log) = Self::TABLES;
        // a^-1 = g^(255 - log(a))
        let inv_log = (255 - log[self.0 as usize]) % 255;
        Gf256(exp[inv_log as usize])
    }

    /// Exponentiation using Log/Antilog tables
    pub fn pow(self, exp: u32) -> Gf256 {
        if self.0 == 0 {
            return Gf256::ZERO;
        }
        let (exp_table, log_table) = Self::TABLES;
        let log_val = log_table[self.0 as usize] as u32;
        let new_log = (log_val * exp) % 255;
        Gf256(exp_table[new_log as usize])
    }
}

// Trait Implementations

impl Add for Gf256 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Gf256(self.0 ^ other.0)
    }
}

impl Sub for Gf256 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        // Subtraction is identical to addition in GF(2^n)
        Gf256(self.0 ^ other.0)
    }
}

impl Mul for Gf256 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
}

impl Div for Gf256 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self * other.inverse()
    }
}

#[cfg(test)]
mod gf256_tests {
    use super::*;

    #[test]
    fn test_gf256_log_tables() {
        let a = Gf256(5);
        let b = Gf256(150);
        let c = Gf256(0);
        println!("a: {}, b: {}, c: {}", a.get(), b.get(), c.get());
        for i in 0..256 {
            // println!("{}: log={}, exp={}", i, Gf256::TABLES.1[i], Gf256::TABLES.0[i]);
        }
        // println!("{:?} {:?}",a.pow(1390218),a.mul(Gf256(217)));
        assert_eq!(a.mul(b), Gf256(217));
    }

    #[test]
    fn test_tables_correctness() {
        let (exp, log) = Gf256::TABLES;

        // Check generator cycle
        let mut seen = [false; 256];

        for i in 0..255 {
            let val = exp[i];
            assert_ne!(val, 0);
            assert!(!seen[val as usize]);
            seen[val as usize] = true;
        }

        // All non-zero values must appear
        for i in 1..256 {
            assert!(seen[i], "Value {} missing from exp table", i);
        }
    }
}
