#!/usr/bin/env python3
"""
Platform and hardware detection for cross-platform benchmarking.

Detects:
- OS (macOS, Linux, Windows)
- CPU model and generation
- Available SIMD capabilities (SSE, AVX, AVX2, AVX-512, NEON)
- Memory configuration
- Thermal state (macOS only)

Usage:
    from benchmarks.platform_detection import detect_platform, get_simd_capabilities

    platform = detect_platform()
    print(platform)

    simd = get_simd_capabilities()
    print(f"Available SIMD: {simd}")
"""

import platform
import subprocess
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class PlatformInfo:
    """Complete platform and hardware information."""

    os_type: str  # 'macOS', 'Linux', 'Windows'
    os_version: str
    architecture: str  # 'arm64', 'x86_64', 'i386'
    cpu_model: str
    cpu_cores: int
    cpu_frequency_ghz: Optional[float]
    cpu_generation: Optional[str]  # '10th Gen Intel Core i9', 'Apple M3 Pro', etc.
    memory_gb: Optional[float]
    simd_capabilities: List[str]
    mojo_available: bool
    rust_available: bool
    faiss_available: bool
    numpy_version: str
    python_version: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def detect_platform() -> PlatformInfo:
    """Detect complete platform information."""
    from datetime import datetime as dt

    os_type = platform.system()  # 'Darwin' for macOS, 'Linux', 'Windows'
    if os_type == "Darwin":
        os_type = "macOS"

    os_version = platform.release()
    architecture = platform.machine()

    # Detect CPU model and generation
    cpu_model, cpu_generation = _detect_cpu()
    cpu_cores = _detect_cpu_cores()
    cpu_frequency_ghz = _detect_cpu_frequency()
    memory_gb = _detect_memory()

    # Detect SIMD capabilities
    simd_capabilities = get_simd_capabilities()

    # Check availability of key dependencies
    mojo_available = _check_mojo()
    rust_available = _check_rust()
    faiss_available = _check_faiss()

    numpy_version = _get_numpy_version()
    python_version = platform.python_version()

    timestamp = dt.utcnow().isoformat() + "Z"

    return PlatformInfo(
        os_type=os_type,
        os_version=os_version,
        architecture=architecture,
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        cpu_frequency_ghz=cpu_frequency_ghz,
        cpu_generation=cpu_generation,
        memory_gb=memory_gb,
        simd_capabilities=simd_capabilities,
        mojo_available=mojo_available,
        rust_available=rust_available,
        faiss_available=faiss_available,
        numpy_version=numpy_version,
        python_version=python_version,
        timestamp=timestamp,
    )


def _detect_cpu() -> tuple[str, Optional[str]]:
    """Detect CPU model and generation."""
    import platform

    os_type = platform.system()
    cpu_model = "Unknown"
    cpu_generation = None

    if os_type == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                cpu_model = result.stdout.strip()
                # Parse Apple Silicon generation
                if "M3" in cpu_model:
                    cpu_generation = "Apple M3"
                elif "M2" in cpu_model:
                    cpu_generation = "Apple M2"
                elif "M1" in cpu_model:
                    cpu_generation = "Apple M1"
                elif "Intel" in cpu_model:
                    # Try to extract Intel generation
                    if "10th" in cpu_model or "i7-10" in cpu_model or "i9-10" in cpu_model:
                        cpu_generation = "10th Gen Intel"
                    elif "9th" in cpu_model or "i7-9" in cpu_model or "i9-9" in cpu_model:
                        cpu_generation = "9th Gen Intel"
                    elif "Core i9" in cpu_model:
                        cpu_generation = "Intel Core i9 (macOS)"
                    elif "Core i7" in cpu_model:
                        cpu_generation = "Intel Core i7 (macOS)"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    elif os_type == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
                    elif line.startswith("Hardware"):
                        cpu_model = line.split(":", 1)[1].strip()

            # Parse CPU generation from model name
            if "Intel" in cpu_model:
                if "Xeon" in cpu_model:
                    if "Gold" in cpu_model:
                        cpu_generation = "Intel Xeon Gold (Skylake/Cascade Lake)"
                    elif "Platinum" in cpu_model:
                        cpu_generation = "Intel Xeon Platinum"
                else:
                    if "Core i9" in cpu_model:
                        cpu_generation = "Intel Core i9"
                    elif "Core i7" in cpu_model:
                        cpu_generation = "Intel Core i7"
            elif "ARM" in cpu_model or "aarch64" in cpu_model:
                cpu_generation = "ARM64"
        except FileNotFoundError:
            pass

    return cpu_model, cpu_generation


def _detect_cpu_cores() -> int:
    """Detect number of logical CPU cores."""
    import os

    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _detect_cpu_frequency() -> Optional[float]:
    """Detect CPU frequency in GHz."""
    os_type = platform.system()

    if os_type == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.cpufrequency"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                freq_hz = int(result.stdout.strip())
                return freq_hz / 1e9
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

    elif os_type == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "cpu MHz" in line:
                        freq_mhz = float(line.split(":", 1)[1].strip())
                        return freq_mhz / 1000.0
        except (FileNotFoundError, ValueError):
            pass

    return None


def _detect_memory() -> Optional[float]:
    """Detect total system memory in GB."""
    os_type = platform.system()

    if os_type == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                bytes_val = int(result.stdout.strip())
                return bytes_val / (1024**3)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

    elif os_type == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split(":", 1)[1].strip().split()[0])
                        return kb / (1024**2)
        except (FileNotFoundError, ValueError):
            pass

    return None


def get_simd_capabilities() -> List[str]:
    """Detect available SIMD instruction sets."""
    capabilities = []

    # Check CPU flags (works on Linux x86)
    import platform

    os_type = platform.system()

    if os_type == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("flags"):
                        flags = line.split(":", 1)[1].strip().split()
                        if "avx512f" in flags:
                            capabilities.append("AVX-512F")
                        if "avx512cd" in flags:
                            capabilities.append("AVX-512CD")
                        if "avx2" in flags:
                            capabilities.append("AVX2")
                        if "avx" in flags:
                            capabilities.append("AVX")
                        if "sse4_2" in flags:
                            capabilities.append("SSE4.2")
                        break
        except FileNotFoundError:
            pass

    elif os_type == "Darwin":  # macOS
        arch = platform.machine()
        if "arm64" in arch:
            capabilities.append("NEON")
            capabilities.append("SVE (partial)")
        elif "x86_64" in arch:
            # Check for AVX-512 on Intel Mac
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.avx512f"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip() == "1":
                    capabilities.append("AVX-512F")
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.optional.avx512cd"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip() == "1":
                        capabilities.append("AVX-512CD")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Check for AVX2
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.avx2_0"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip() == "1":
                    capabilities.append("AVX2")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Check for AVX
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.avx1_0"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip() == "1":
                    capabilities.append("AVX")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    # Remove duplicates and sort
    capabilities = sorted(list(set(capabilities)))
    if not capabilities:
        capabilities = ["Generic (SIMD capabilities not detected)"]

    return capabilities


def _check_mojo() -> bool:
    """Check if Mojo binary is available."""
    try:
        binary_path = Path(__file__).parent.parent / "vectro_quantizer"
        return binary_path.exists() and binary_path.is_file()
    except OSError:
        return False


def _check_rust() -> bool:
    """Check if Rust extension is available."""
    try:
        import vectro_lib

        return True
    except ImportError:
        return False


def _check_faiss() -> bool:
    """Check if FAISS is available."""
    try:
        import faiss

        return True
    except ImportError:
        return False


def _get_numpy_version() -> str:
    """Get NumPy version."""
    try:
        import numpy

        return numpy.__version__
    except ImportError:
        return "Unknown"


if __name__ == "__main__":
    platform_info = detect_platform()
    print(platform_info.to_json())
