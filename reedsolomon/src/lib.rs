

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Gf256(u8);

impl Gf256{
        /// Create from raw byte
    pub const fn new(val: u8) -> Self {
        Gf256(val)
    }

    /// Get raw byte
    pub const fn get(self) -> u8 {
        self.0
    }
}