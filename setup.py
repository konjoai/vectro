from setuptools import setup, Extension
from setuptools.command.build_py import build_py
import numpy as np
import os
import subprocess
from pathlib import Path

try:
    from Cython.Build import cythonize as _cythonize
except ImportError:
    _cythonize = None


class BuildPyWithMojo(build_py):
    """Custom build command that compiles Mojo code before building Python package."""

    def run(self):
        # Try to compile Mojo binary if Mojo is available
        mojo_source = Path("experimental/mojo/vectro_standalone.mojo")
        mojo_binary = Path("vectro_quantizer")

        if mojo_source.exists():
            print("=" * 70)
            print("Attempting to compile Mojo quantizer...")
            print("=" * 70)

            try:
                # Check if mojo command is available
                result = subprocess.run(
                    ["mojo", "--version"], capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    print(f"Found Mojo: {result.stdout.strip()}")

                    # Compile Mojo to binary
                    compile_result = subprocess.run(
                        ["mojo", "build", str(mojo_source), "-o", str(mojo_binary)],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )

                    if compile_result.returncode == 0:
                        print(f"✓ Successfully compiled {mojo_source} -> {mojo_binary}")
                        print(f"  Binary size: {mojo_binary.stat().st_size / 1024:.1f} KB")
                    else:
                        print("✗ Mojo compilation failed:")
                        print(compile_result.stderr)
                        print("\nContinuing without Mojo backend...")
                else:
                    print("✗ Mojo command failed")
                    print("Continuing without Mojo backend...")

            except FileNotFoundError:
                print("ℹ Mojo compiler not found in PATH")
                print("  To enable Mojo backend, install Mojo from: https://www.modular.com/mojo")
                print("  Continuing without Mojo backend...")
            except subprocess.TimeoutExpired:
                print("✗ Mojo compilation timed out")
                print("Continuing without Mojo backend...")
            except Exception as e:
                print(f"✗ Error during Mojo compilation: {e}")
                print("Continuing without Mojo backend...")

        # Continue with normal build
        super().run()


# Cython extension — only built when both Cython and the .pyx source are present
_pyx_file = "src/quantizer_cython.pyx"
if _cythonize is not None and os.path.exists(_pyx_file):
    extensions = [
        Extension(
            "python.quantizer_cython",
            [_pyx_file],
            include_dirs=[np.get_include()],
        )
    ]
    ext_modules = _cythonize(extensions, language_level=3)
else:
    ext_modules = []

setup(
    ext_modules=ext_modules,
    cmdclass={
        "build_py": BuildPyWithMojo,
    },
)
