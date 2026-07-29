//! WASM bindings for vectro_lib.
//!
//! Exposes INT8 and NF4 single-vector encoding to JavaScript via wasm-bindgen.
//! Build with:
//!   wasm-pack build rust/vectro_lib --target web --release
//!
//! The resulting `pkg/` directory contains the `.wasm` binary, a JS wrapper,
//! and TypeScript type declarations.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use crate::quant::int8::Int8Vector;
use crate::quant::nf4::Nf4Vector;

// --------------------------------------------------------------------------- //
// INT8 bindings
// --------------------------------------------------------------------------- //

/// Encode a f32 slice to INT8 quantised codes.
///
/// Returns a `Int8Array`-compatible `Vec<i32>` (wasm-bindgen does not support
/// `Vec<i8>` directly — values are sign-extended to i32 for portability).
///
/// Pair with `encode_int8_scale` to obtain the per-vector scale factor needed
/// for dequantisation: `decoded[i] = codes[i] / 127.0 * scale`.
#[wasm_bindgen]
pub fn encode_int8(vec: &[f32]) -> Vec<i32> {
    Int8Vector::encode_fast(vec)
        .codes
        .into_iter()
        .map(|c| c as i32)
        .collect()
}

/// Return the per-vector abs-max scale for an INT8-encoded vector.
///
/// Call this alongside `encode_int8` using the **same** input slice.
/// Because both functions call `encode_fast` independently, this incurs a
/// second pass over the data. Consumers that need both fields should call
/// `encode_int8_full` instead.
#[wasm_bindgen]
pub fn encode_int8_scale(vec: &[f32]) -> f32 {
    Int8Vector::encode_fast(vec).scale
}

/// Encode a f32 slice to INT8 and return `[codes..., scale]` as a single
/// `Float32Array`-friendly `Vec<f32>`.
///
/// Layout: `result[0..dim]` = codes cast to f32, `result[dim]` = scale.
/// This single-pass variant is preferred when both codes and scale are needed.
#[wasm_bindgen]
pub fn encode_int8_full(vec: &[f32]) -> Vec<f32> {
    let q = Int8Vector::encode_fast(vec);
    let mut out: Vec<f32> = q.codes.into_iter().map(|c| c as f32).collect();
    out.push(q.scale);
    out
}

// --------------------------------------------------------------------------- //
// NF4 bindings
// --------------------------------------------------------------------------- //

/// Encode a f32 slice to packed NF4 nibbles (QLoRA-style).
///
/// Returns a `Uint8Array`-compatible `Vec<u8>` of `ceil(dim / 2)` bytes.
/// Each byte packs two consecutive 4-bit NF4 codes (low nibble first).
///
/// Pair with `encode_nf4_scale` and `encode_nf4_dim` to fully reconstruct.
#[wasm_bindgen]
pub fn encode_nf4(vec: &[f32]) -> Vec<u8> {
    Nf4Vector::encode_fast(vec).packed
}

/// Return the per-vector abs-max scale for an NF4-encoded vector.
#[wasm_bindgen]
pub fn encode_nf4_scale(vec: &[f32]) -> f32 {
    Nf4Vector::encode_fast(vec).scale
}

/// Return the original embedding dimension encoded in the NF4 struct.
///
/// Needed to correctly unpack the last byte when `dim` is odd.
#[wasm_bindgen]
pub fn encode_nf4_dim(vec: &[f32]) -> usize {
    vec.len()
}

// --------------------------------------------------------------------------- //
// Browser tests (wasm-pack test --headless --chrome)
//
// ADR-002 Decision 2 gate: all tests must pass on `wasm32-unknown-unknown`
// target in a headless Chrome instance via wasm-pack.
//
// Run locally:
//   wasm-pack test --headless --chrome rust/vectro_lib
// --------------------------------------------------------------------------- //

#[cfg(all(target_arch = "wasm32", test))]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    // ── INT8 tests ──────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_encode_int8_output_length() {
        let vec: Vec<f32> = (0..768).map(|i| (i as f32 - 383.5) / 100.0).collect();
        let codes = encode_int8(&vec);
        assert_eq!(
            codes.len(),
            768,
            "INT8 codes length must equal input dimension"
        );
    }

    #[wasm_bindgen_test]
    fn test_encode_int8_codes_in_range() {
        let vec: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 10.0).collect();
        let codes = encode_int8(&vec);
        for &c in &codes {
            assert!(c >= -128 && c <= 127, "INT8 code {} out of i8 range", c);
        }
    }

    #[wasm_bindgen_test]
    fn test_encode_int8_scale_positive() {
        let vec: Vec<f32> = vec![1.0_f32; 64];
        let scale = encode_int8_scale(&vec);
        assert!(scale > 0.0, "scale must be positive for non-zero input");
    }

    #[wasm_bindgen_test]
    fn test_encode_int8_zero_vector_scale() {
        // Zero vector: abs-max is 0, scale should default to 1.0 (no NaN/panic)
        let vec: Vec<f32> = vec![0.0_f32; 64];
        let scale = encode_int8_scale(&vec);
        assert!(scale.is_finite(), "scale must be finite for zero vector");
    }

    #[wasm_bindgen_test]
    fn test_encode_int8_full_output_length() {
        // encode_int8_full returns [codes..., scale] — length = dim + 1
        let dim: usize = 32;
        let vec: Vec<f32> = vec![0.5_f32; dim];
        let full = encode_int8_full(&vec);
        assert_eq!(
            full.len(),
            dim + 1,
            "encode_int8_full must return dim+1 values"
        );
    }

    #[wasm_bindgen_test]
    fn test_encode_int8_full_last_element_is_scale() {
        let vec: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let full = encode_int8_full(&vec);
        let scale_from_full = *full.last().expect("output non-empty");
        let scale_direct = encode_int8_scale(&vec);
        assert!(
            (scale_from_full - scale_direct).abs() < 1e-6,
            "scale in encode_int8_full must match encode_int8_scale"
        );
    }

    // ── NF4 tests ───────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_encode_nf4_packed_length_even_dim() {
        let dim: usize = 768; // even
        let vec: Vec<f32> = (0..dim).map(|i| (i as f32 - 383.5) / 100.0).collect();
        let packed = encode_nf4(&vec);
        assert_eq!(
            packed.len(),
            dim / 2,
            "NF4 packed length must be dim/2 for even dimension"
        );
    }

    #[wasm_bindgen_test]
    fn test_encode_nf4_packed_length_odd_dim() {
        let dim: usize = 7; // odd
        let vec: Vec<f32> = vec![0.1_f32; dim];
        let packed = encode_nf4(&vec);
        assert_eq!(
            packed.len(),
            (dim + 1) / 2,
            "NF4 packed length must be ceil(dim/2) for odd dimension"
        );
    }

    #[wasm_bindgen_test]
    fn test_encode_nf4_scale_positive() {
        let vec: Vec<f32> = vec![-1.0_f32, 0.5, 1.0, -0.5];
        let scale = encode_nf4_scale(&vec);
        assert!(scale > 0.0, "NF4 scale must be positive for non-zero input");
    }

    #[wasm_bindgen_test]
    fn test_encode_nf4_dim_passthrough() {
        let vec: Vec<f32> = vec![0.1_f32; 42];
        assert_eq!(
            encode_nf4_dim(&vec),
            42,
            "encode_nf4_dim must return input length"
        );
    }

    // ── Cross-method coherence ───────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_int8_and_nf4_accept_same_input() {
        // Both methods must accept any f32 vector without panic
        let vec: Vec<f32> = vec![f32::MAX / 2.0, f32::MIN / 2.0, 0.0, 1.0, -1.0, 0.5];
        let _ = encode_int8(&vec);
        let _ = encode_nf4(&vec);
    }
}
