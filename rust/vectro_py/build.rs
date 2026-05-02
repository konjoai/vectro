// Wave 3d — link Apple's Accelerate framework when the `vectro_lib_accelerate`
// feature is on.  The vDSP_vsmsa symbol used by the AMX-routed encode lives
// inside Accelerate.framework on macOS.
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let accelerate = std::env::var("CARGO_FEATURE_VECTRO_LIB_ACCELERATE").is_ok();
    if target_os == "macos" && accelerate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
