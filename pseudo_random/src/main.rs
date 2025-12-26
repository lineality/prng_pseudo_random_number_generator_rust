mod pseudo_random_hardware_xoshiro_module;
mod simple_lcg_pseudo_random;

/// Generates a pseudo-random number in range [0, max) using system time.
///
/// # Project Context
/// Provides minimal randomness. Uses nanosecond-precision
/// system time as entropy source. Quality is sufficient for casual use where
/// cryptographic security is not required. User interaction delays between calls
/// provide adequate entropy distribution.
///
/// # Parameters
/// - `max`: Upper bound (exclusive) for the random number range
///
/// # Returns
/// - `usize` in range [0, max), or 0 if an error occurs
///
/// # Error Handling
/// If system time cannot be read (clock before Unix Epoch, time moved backward),
/// returns 0 as a safe fallback. This allows to continue functioning
/// even with degraded randomness.
///
/// # Edge Cases
/// - `max == 0`: Returns 0 (avoids division by zero panic from modulo)
/// - `max == 1`: Always returns 0 (only one possible value)
/// - System time before 1970: Returns 0
/// - System clock moved backward: Returns 0
fn random_usize(max: usize) -> usize {
    use std::time::{SystemTime, UNIX_EPOCH};
    // Production catch: Handle zero max case to prevent modulo by zero
    if max == 0 {
        return 0;
    }

    // Attempt to get system time duration since Unix Epoch
    let duration_result = SystemTime::now().duration_since(UNIX_EPOCH);

    // Production catch: Handle time errors gracefully
    let nanos = match duration_result {
        Ok(duration) => duration.as_nanos(),
        Err(_) => {
            // System time is before Unix Epoch or moved backward
            // Fallback: Return 0 to keep running
            // Alternative: Could use a counter, hash of process ID, or other fallback
            return 0;
        }
    };

    // Cast to usize and apply modulo to constrain to [0, max)
    // Note: On 32-bit systems, casting u128 to usize truncates high bits
    // This still provides adequate randomness for this use case
    (nanos as usize) % max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_usize_bounded() {
        // Test that output is always less than max
        for _ in 0..1000 {
            let result = random_usize(10);
            assert!(result < 10, "random_usize must return value < max");
        }
    }

    #[test]
    fn test_random_usize_zero_max() {
        // Edge case: max = 0 should return 0
        let result = random_usize(0);
        assert_eq!(result, 0, "random_usize(0) should return 0");
    }

    #[test]
    fn test_random_usize_one_max() {
        // Edge case: max = 1 can only return 0
        let result = random_usize(1);
        assert_eq!(result, 0, "random_usize(1) can only return 0");
    }

    #[test]
    fn test_random_usize_generates_variation() {
        // Statistical test: over many calls with delays, should see multiple values
        use std::collections::HashSet;
        use std::thread;
        use std::time::Duration;

        let mut seen = HashSet::new();
        for _ in 0..20 {
            seen.insert(random_usize(10));
            thread::sleep(Duration::from_micros(100));
        }

        // Should see at least 5 different values out of 20 attempts
        assert!(seen.len() >= 5, "Should generate varied outputs over time");
    }
}

fn main() {
    let result1 = random_usize(100);
    let result2 = random_usize(100);
    let result3 = random_usize(100);
    println!(
        "Hello, random world!\nuse 'cargo test' to run tests.\n1-100: {}, {}, {}",
        result1, result2, result3,
    );
}
