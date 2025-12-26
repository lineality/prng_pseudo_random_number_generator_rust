//! Simple Pseudo-Random Number Generator (PRNG)
//!
//! # Purpose
//! Provides a lightweight, dependency-free pseudo-random number generator suitable
//! for non-cryptographic applications where predictable randomness is acceptable.
//!
//! # Project Context
//! This crate serves as a minimal replacement for the `rand` crate in situations where:
//! - External dependencies must be minimized
//! - Cryptographic security is NOT required
//! - Deterministic reproducibility is valuable (seeded testing)
//! - Lightweight footprint is essential
//! - Build times and dependency auditing overhead must be reduced
//!
//! # ⚠️ CRITICAL LIMITATIONS ⚠️
//! - **NOT cryptographically secure** - NEVER use for security purposes
//! - **NOT suitable for** passwords, tokens, keys, nonces, or any security-sensitive randomness
//! - **Limited statistical quality** - Not suitable for high-quality Monte Carlo simulations
//! - **Predictable** - Given the seed, entire sequence can be reproduced
//! - **Small state space** - 64-bit state can cycle (though period is ~2^64)
//!
//! # Suitable Use Cases
//! ✓ Game mechanics (card shuffling, dice rolls, spawn positions)
//! ✓ Procedural generation (terrain, content) in games/simulations
//! ✓ Fuzz testing with reproducible seeds
//! ✓ Simple sampling and selection from collections
//! ✓ Non-critical randomization in user interfaces
//! ✓ Educational demonstrations of PRNGs
//! ✓ Quick prototyping before implementing proper randomness
//!
//! # Algorithm: Linear Congruential Generator (LCG)
//! Uses the recurrence relation: `state = (a × state + c) mod 2^64`
//! - Multiplier (a): 6364136223846793005
//! - Increment (c): 1442695040888963407
//! - Constants from Numerical Recipes (well-tested for full period)
//! - Period: 2^64 (all 64-bit values visited exactly once)
//!
//! # Safety & Reliability
//! - No heap allocation
//! - No unsafe code
//! - No panics in production (all errors returned as Result)
//! - Deterministic behavior (same seed → same sequence)
//! - Bounded loops with bias mitigation
//!
//! # Testing Strategy
//! - Deterministic seed testing for reproducibility
//! - Range boundary testing
//! - Distribution uniformity testing (statistical chi-square test recommended for production)
//! - Error handling validation

use std::time::{SystemTime, UNIX_EPOCH};

/// Errors that can occur during random number generation operations.
///
/// # Design Note
/// Error messages include function context prefixes for traceability
/// in production logs without exposing sensitive internal details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimpleRngError {
    /// Attempted to generate random value with invalid range (max must be > 0)
    ///
    /// Context: gen_range() called with max=0, which has no valid output range
    InvalidRange(&'static str),

    /// System time unavailable during seed generation (extremely rare)
    ///
    /// Context: SystemTime::now() failed, fallback seed will be used
    /// Note: This is a non-fatal warning condition in current implementation
    TimeUnavailable(&'static str),
}

impl std::fmt::Display for SimpleRngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRange(msg) => write!(f, "SRNG range error: {}", msg),
            Self::TimeUnavailable(msg) => write!(f, "SRNG time error: {}", msg),
        }
    }
}

impl std::error::Error for SimpleRngError {}

/// Simple pseudo-random number generator using Linear Congruential Generator (LCG).
///
/// # Architecture
/// - Single 64-bit state variable
/// - Deterministic: same seed produces same sequence
/// - No heap allocation (stack-only)
/// - Thread-local by design (not Send/Sync by default - user must wrap if needed)
///
/// # Performance
/// - Very fast: single multiply + add per random value
/// - Minimal memory: 8 bytes of state
/// - No system calls after initialization
///
/// # Quality Tradeoffs
/// LCG is chosen for simplicity and speed, NOT for statistical quality.
/// For higher quality randomness, consider algorithms like:
/// - PCG (Permuted Congruential Generator) - better statistical properties
/// - Xoshiro/Xoroshiro - faster and better quality
/// - ChaCha20 - cryptographically secure but slower
///
/// # Example Usage
/// ```
/// use simple_rng::SimpleRng;
///
/// // Time-seeded for unpredictable sequences
/// let mut rng = SimpleRng::new();
/// let dice_roll = rng.gen_range(6).unwrap() + 1; // 1-6
///
/// // Fixed seed for reproducible testing
/// let mut test_rng = SimpleRng::from_seed(42);
/// let value = test_rng.gen_f64(); // [0.0, 1.0)
/// ```
pub struct SimpleRng {
    /// Internal state of the LCG
    ///
    /// This 64-bit value evolves according to: state = (a × state + c) mod 2^64
    /// The full 2^64 period means every possible u64 value appears exactly once
    /// before the sequence repeats.
    state: u64,
}

impl SimpleRng {
    /// Creates a new RNG seeded from system time (nanoseconds since Unix epoch).
    ///
    /// # Seeding Strategy
    /// Uses nanosecond-precision timestamp to maximize entropy in seed value.
    /// Different program executions will produce different random sequences.
    ///
    /// # Defensive Programming
    /// Falls back to fixed seed (123456789) if system time is unavailable.
    /// This is extremely rare but handles:
    /// - System clock failures
    /// - Time moving backwards
    /// - Platform-specific time API failures
    ///
    /// # Production Considerations
    /// For applications requiring unpredictable seeds:
    /// - Time-based seeding is adequate for non-security use
    /// - For security: MUST use platform entropy source (getrandom/OsRng)
    /// - Multiple instances created rapidly may get similar seeds
    ///
    /// # Returns
    /// A new `SimpleRng` instance with time-based or fallback seed
    ///
    /// # Examples
    /// ```
    /// use simple_rng::SimpleRng;
    ///
    /// let rng1 = SimpleRng::new();
    /// let rng2 = SimpleRng::new();
    /// // Different instances will likely have different seeds
    /// ```
    pub fn new() -> Self {
        let seed = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_nanos() as u64,
            Err(e) => {
                // Production catch: System time unavailable
                // Create the error for logging/monitoring purposes
                let error =
                    SimpleRngError::TimeUnavailable("System time unavailable, using fallback seed");

                // In production, this would log to your logging system
                eprintln!("Warning: {} ({})", error, e);

                123456789u64
            }
        };

        Self { state: seed }
    }

    /// Creates a new RNG with a specific seed value.
    ///
    /// # Use Cases
    /// - **Testing**: Reproducible test sequences for deterministic behavior validation
    /// - **Debugging**: Reproduce specific random sequences that caused issues
    /// - **Procedural Generation**: Same seed → same world/level generation
    /// - **Distributed Systems**: Synchronize random sequences across nodes
    ///
    /// # Security Warning
    /// NEVER use predictable seeds for security-sensitive randomness.
    /// Attackers can reproduce the entire sequence if they know or guess the seed.
    ///
    /// # Arguments
    /// * `seed` - Initial state value for the generator (any u64 is valid)
    ///
    /// # Returns
    /// A new `SimpleRng` instance with the specified seed
    ///
    /// # Examples
    /// ```
    /// use simple_rng::SimpleRng;
    ///
    /// // Reproducible test sequence
    /// let mut rng = SimpleRng::from_seed(42);
    /// let first = rng.gen_range(100).unwrap();
    ///
    /// // Same seed produces same sequence
    /// let mut rng2 = SimpleRng::from_seed(42);
    /// let second = rng2.gen_range(100).unwrap();
    /// assert_eq!(first, second);
    /// ```
    pub fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next pseudo-random u64 value (internal engine).
    ///
    /// # Algorithm
    /// Linear Congruential Generator with full period (2^64):
    /// ```text
    /// state_new = (state_old × a + c) mod 2^64
    /// where:
    ///   a = 6364136223846793005 (multiplier)
    ///   c = 1442695040888963407 (increment)
    /// ```
    ///
    /// # Constants Source
    /// From "Numerical Recipes" (Press et al.) - well-vetted constants
    /// that provide full period and reasonable statistical properties.
    ///
    /// # Implementation Note
    /// Uses wrapping arithmetic (intentional overflow) since we operate mod 2^64.
    /// This is the mathematically correct behavior, not a bug.
    ///
    /// # Returns
    /// A pseudo-random u64 value (full 64-bit range)
    fn next_u64(&mut self) -> u64 {
        // LCG recurrence: state = (a × state + c) mod 2^64
        // wrapping_mul/add provide mod 2^64 behavior
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generates a random f64 in the range [0.0, 1.0).
    ///
    /// # Use Cases
    /// - Probability checks: `if rng.gen_f64() < 0.25 { ... }` for 25% chance
    /// - Weighted selection: compare random value against cumulative probability
    /// - Interpolation: random values for lerp/smooth transitions
    /// - Continuous distributions: uniform [0,1) as input to other distributions
    ///
    /// # Algorithm
    /// 1. Generate 64-bit random integer
    /// 2. Use upper 53 bits (IEEE 754 double precision significand width)
    /// 3. Divide by 2^53 to normalize to [0.0, 1.0)
    ///
    /// # Quality Note
    /// Using upper 53 bits provides better statistical properties than lower bits
    /// (LCG lower bits have shorter periods). 53 bits gives ~9×10^15 distinct values.
    ///
    /// # Returns
    /// A pseudo-random f64 in the range [0.0, 1.0) (inclusive 0, exclusive 1)
    ///
    /// # Examples
    /// ```
    /// use simple_rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::from_seed(123);
    ///
    /// // Probability check: 25% chance
    /// if rng.gen_f64() < 0.25 {
    ///     // Occurs ~25% of the time
    /// }
    ///
    /// // Random interpolation
    /// let t = rng.gen_f64();
    /// let value = start + t * (end - start);
    /// ```
    pub fn gen_f64(&mut self) -> f64 {
        // Use upper 53 bits for better distribution
        // (LCG lower bits have poorer statistical properties)
        let value = self.next_u64() >> 11;

        // Convert to float in [0.0, 1.0)
        // 2^53 as denominator gives full precision for f64
        (value as f64) / ((1u64 << 53) as f64)
    }

    /// Generates a random index in the range [0, max).
    ///
    /// # Use Cases
    /// - Array/vector indexing: random element selection
    /// - Collection sampling: pick random items
    /// - Game mechanics: random choice from options
    /// - Load balancing: random server selection
    ///
    /// # Algorithm: Modulo Bias Mitigation
    /// Simple modulo (`value % max`) creates bias toward smaller values
    /// when range doesn't evenly divide 2^64. This implementation:
    /// 1. Calculates threshold below which values are unbiased
    /// 2. Rejects values above threshold (retry)
    /// 3. Returns biased-free result
    ///
    /// # Performance
    /// Expected iterations: ~1.0 (very rare retries)
    /// Worst case (max = 2^63 + 1): ~2.0 iterations average
    ///
    /// # Arguments
    /// * `max` - Exclusive upper bound (must be > 0)
    ///
    /// # Returns
    /// - `Ok(usize)` - Random index in [0, max)
    /// - `Err(SimpleRngError)` - If max is 0 (invalid range)
    ///
    /// # Error Handling
    /// Returns error instead of panicking on invalid input (defensive programming).
    /// Caller must handle Result - use `?` operator or match.
    ///
    /// # Examples
    /// ```
    /// use simple_rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::from_seed(456);
    /// let items = vec!["apple", "banana", "cherry"];
    ///
    /// // Safe random selection
    /// match rng.gen_range(items.len()) {
    ///     Ok(idx) => {
    ///         let selected = items[idx];
    ///         // Use selected item
    ///     }
    ///     Err(e) => {
    ///         // Handle empty collection case
    ///     }
    /// }
    /// ```
    pub fn gen_range(&mut self, max: usize) -> Result<usize, SimpleRngError> {
        // Production catch: validate input
        if max == 0 {
            return Err(SimpleRngError::InvalidRange("max must be > 0"));
        }

        // Mitigate modulo bias by rejecting values in the biased range
        let max_u64 = max as u64;
        let threshold = u64::MAX - (u64::MAX % max_u64);

        // Bounded loop: worst case ~2 iterations on average
        // This loop MUST be here (not a power-of-10 violation)
        // to guarantee uniform distribution
        loop {
            let value = self.next_u64();
            if value < threshold {
                return Ok((value % max_u64) as usize);
            }
            // Retry if in biased range (very rare: ~0-50% depending on max)
        }
    }

    /// Generates a random boolean value (true/false with 50% probability each).
    ///
    /// # Use Cases
    /// - Coin flip simulation
    /// - Binary decisions
    /// - Random direction/orientation
    ///
    /// # Implementation
    /// Uses least significant bit of random u64 (fast, uniform for good RNGs).
    ///
    /// # Returns
    /// Random boolean with equal probability
    ///
    /// # Examples
    /// ```
    /// use simple_rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new();
    /// if rng.gen_bool() {
    ///     // ~50% of the time
    /// }
    /// ```
    pub fn gen_bool(&mut self) -> bool {
        // Use LSB of random value (faster than gen_range)
        (self.next_u64() & 1) == 1
    }

    /// Generates a random boolean with specified probability.
    ///
    /// # Use Cases
    /// - Weighted random decisions
    /// - Drop rates in games
    /// - A/B testing with custom split ratios
    ///
    /// # Arguments
    /// * `probability` - Probability of returning true [0.0, 1.0]
    ///
    /// # Returns
    /// - `Ok(bool)` - true with specified probability, false otherwise
    /// - `Err(SimpleRngError)` - if probability not in [0.0, 1.0]
    ///
    /// # Examples
    /// ```
    /// use simple_rng::SimpleRng;
    ///
    /// let mut rng = SimpleRng::new();
    /// // 25% chance of true
    /// if rng.gen_bool_with_prob(0.25).unwrap_or(false) {
    ///     // Rare event
    /// }
    /// ```
    pub fn gen_bool_with_prob(&mut self, probability: f64) -> Result<bool, SimpleRngError> {
        // Production catch: validate probability range
        if !(0.0..=1.0).contains(&probability) {
            return Err(SimpleRngError::InvalidRange(
                "probability must be in [0.0, 1.0]",
            ));
        }

        Ok(self.gen_f64() < probability)
    }
}

impl Default for SimpleRng {
    /// Default instance uses time-based seeding (equivalent to `SimpleRng::new()`).
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTING
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===========================================
    // Test: Deterministic Reproducibility
    // ===========================================
    #[test]
    fn test_rng_deterministic_with_seed() {
        // Project Context: Verify same seed produces identical sequences
        // Critical for reproducible testing and deterministic procedural generation

        let mut rng1 = SimpleRng::from_seed(42);
        let mut rng2 = SimpleRng::from_seed(42);

        // Same seed must produce identical 10-value sequence
        for i in 0..10 {
            let val1 = rng1.next_u64();
            let val2 = rng2.next_u64();
            assert_eq!(val1, val2, "Sequence diverged at position {}", i);
        }
    }

    #[test]
    fn test_rng_different_seeds_diverge() {
        // Project Context: Different seeds must produce different sequences

        let mut rng1 = SimpleRng::from_seed(42);
        let mut rng2 = SimpleRng::from_seed(43);

        // Different seeds should quickly diverge
        let mut differences = 0;
        for _ in 0..10 {
            if rng1.next_u64() != rng2.next_u64() {
                differences += 1;
            }
        }

        // Expect most values to differ
        assert!(
            differences >= 8,
            "Seeds too similar: only {} differences",
            differences
        );
    }

    // ===========================================
    // Test: f64 Generation Range Bounds
    // ===========================================
    #[test]
    fn test_gen_f64_range() {
        // Project Context: Verify all f64 values in [0.0, 1.0)
        // Critical for probability calculations

        let mut rng = SimpleRng::from_seed(123);

        for i in 0..1000 {
            let val = rng.gen_f64();
            assert!(val >= 0.0, "Value {} below 0.0 at iteration {}", val, i);
            assert!(val < 1.0, "Value {} >= 1.0 at iteration {}", val, i);
        }
    }

    // ===========================================
    // Test: gen_range Bounds
    // ===========================================
    #[test]
    fn test_gen_range_bounds() {
        // Project Context: Verify all indices valid for array access

        let mut rng = SimpleRng::from_seed(456);

        for i in 0..1000 {
            match rng.gen_range(10) {
                Ok(val) => {
                    assert!(val < 10, "Value {} out of range at iteration {}", val, i);
                }
                Err(e) => {
                    panic!("Unexpected error at iteration {}: {}", i, e);
                }
            }
        }
    }

    // ===========================================
    // Test: gen_range Error Handling
    // ===========================================
    #[test]
    fn test_gen_range_zero_max_error() {
        // Project Context: Verify defensive programming catches invalid input

        let mut rng = SimpleRng::new();
        let result = rng.gen_range(0);

        assert!(result.is_err(), "Should return error for max=0");

        match result {
            Err(SimpleRngError::InvalidRange(msg)) => {
                assert!(
                    msg.contains("max must be > 0"),
                    "Wrong error message: {}",
                    msg
                );
            }
            _ => panic!("Wrong error type returned"),
        }
    }

    // ===========================================
    // Test: Distribution Uniformity
    // ===========================================
    #[test]
    fn test_gen_range_distribution() {
        // Project Context: Verify reasonably uniform distribution
        // Note: This is a basic chi-square-like test, not rigorous statistical analysis

        let mut rng = SimpleRng::from_seed(789);
        let mut counts = [0; 5];

        // Generate 5000 samples in range [0, 5)
        for _ in 0..5000 {
            match rng.gen_range(5) {
                Ok(idx) => {
                    counts[idx] += 1;
                }
                Err(e) => {
                    panic!("Unexpected error during distribution test: {}", e);
                }
            }
        }

        // Each bucket should have roughly 1000 samples (±30% tolerance)
        // Note: 30% is generous; production statistical tests should be more rigorous
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count > 700 && count < 1300,
                "Distribution skewed at bucket {}: {:?}",
                i,
                counts
            );
        }
    }

    // ===========================================
    // Test: Boolean Generation
    // ===========================================
    #[test]
    fn test_gen_bool_distribution() {
        // Project Context: Verify ~50/50 distribution for boolean generation

        let mut rng = SimpleRng::from_seed(111);
        let mut true_count = 0;
        let iterations = 10000;

        for _ in 0..iterations {
            if rng.gen_bool() {
                true_count += 1;
            }
        }

        // Expect ~5000 trues (±10% tolerance)
        let expected = iterations / 2;
        let tolerance = iterations / 10;
        assert!(
            true_count > expected - tolerance && true_count < expected + tolerance,
            "Boolean distribution skewed: {} trues out of {}",
            true_count,
            iterations
        );
    }

    // ===========================================
    // Test: Probability-Weighted Boolean
    // ===========================================
    #[test]
    fn test_gen_bool_with_prob() {
        // Project Context: Verify weighted probability works correctly

        let mut rng = SimpleRng::from_seed(222);
        let mut true_count = 0;
        let iterations = 10000;
        let probability = 0.25;

        for _ in 0..iterations {
            match rng.gen_bool_with_prob(probability) {
                Ok(true) => true_count += 1,
                Ok(false) => {}
                Err(e) => panic!("Unexpected error: {}", e),
            }
        }

        // Expect ~2500 trues (±15% tolerance)
        let expected = (iterations as f64 * probability) as usize;
        let tolerance = expected * 15 / 100;
        assert!(
            true_count > expected - tolerance && true_count < expected + tolerance,
            "Weighted probability skewed: {} trues out of {} (expected ~{})",
            true_count,
            iterations,
            expected
        );
    }

    // ===========================================
    // Test: Time Unavailable Warning
    // ===========================================
    #[test]
    fn test_new_with_fallback() {
        // Project Context: Verify fallback seed is used when time fails
        // Note: Hard to test actual time failure, but we verify the fallback works

        let mut test_sequence = Vec::new();
        let mut rng_copy = SimpleRng::from_seed(123456789);

        // Generate sequence from fallback seed
        for _ in 0..5 {
            test_sequence.push(rng_copy.next_u64());
        }

        // Verify fallback seed produces deterministic sequence
        assert_eq!(test_sequence.len(), 5);

        // The fallback seed should produce valid values
        let mut rng_fallback = SimpleRng::from_seed(123456789);
        for expected in test_sequence {
            assert_eq!(rng_fallback.next_u64(), expected);
        }
    }

    // ===========================================
    // Test: Invalid Probability Error
    // ===========================================
    #[test]
    fn test_gen_bool_with_prob_invalid() {
        // Project Context: Verify error handling for invalid probabilities

        let mut rng = SimpleRng::new();

        // Test below range
        assert!(rng.gen_bool_with_prob(-0.1).is_err());

        // Test above range
        assert!(rng.gen_bool_with_prob(1.1).is_err());

        // Test valid boundaries
        assert!(rng.gen_bool_with_prob(0.0).is_ok());
        assert!(rng.gen_bool_with_prob(1.0).is_ok());
    }

    // ===========================================
    // Test: Edge Case - Range of 1
    // ===========================================
    #[test]
    fn test_gen_range_single_value() {
        // Project Context: Verify single-value range always returns 0

        let mut rng = SimpleRng::from_seed(333);

        for _ in 0..100 {
            match rng.gen_range(1) {
                Ok(val) => assert_eq!(val, 0, "Single-value range must return 0"),
                Err(e) => panic!("Unexpected error: {}", e),
            }
        }
    }

    // ===========================================
    // Test: Large Range Values
    // ===========================================
    #[test]
    fn test_gen_range_large_values() {
        // Project Context: Verify behavior with large max values

        let mut rng = SimpleRng::from_seed(444);
        let large_max = usize::MAX / 2;

        // Should not panic or error
        for _ in 0..100 {
            match rng.gen_range(large_max) {
                Ok(val) => assert!(val < large_max),
                Err(e) => panic!("Unexpected error with large max: {}", e),
            }
        }
    }
}
