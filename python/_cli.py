"""
vectro._cli — CLI entry-point shim for the maturin wheel.

When the wheel is installed, `pip` places the Rust `vectro_cli` binary in the
package's `bin/` directory (via maturin's `scripts` mechanism) *and* registers
this shim as a console-script entry-point named `vectro`.

The shim locates the bundled binary and exec()s it, forwarding all arguments.
Falls back to the Python CLI (.cli:main) when the Rust binary is not
present so the package remains usable in dev/editable installs that have not
yet run `cargo build`.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


def _find_rust_binary() -> pathlib.Path | None:
    """Locate the compiled vectro_cli binary relative to this module."""
    # When installed via maturin the binary lands in
    # <site-packages>/vectro_py/bin/vectro  (or vectro.exe on Windows).
    here = pathlib.Path(__file__).parent
    for candidate in [
        here / "bin" / "vectro",
        here / "bin" / "vectro.exe",
        # Development: cargo build --release output
        pathlib.Path(__file__).parents[2] / "rust" / "target" / "release" / "vectro",
        pathlib.Path(__file__).parents[2] / "rust" / "target" / "debug" / "vectro",
    ]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def main() -> None:
    binary = _find_rust_binary()
    if binary is not None:
        # exec the Rust binary, replacing the current process (POSIX) or
        # spawning a subprocess (Windows / fallback).
        try:
            os.execv(str(binary), [str(binary)] + sys.argv[1:])
        except OSError:
            # execv failed (e.g. on Windows); fall through to subprocess
            result = subprocess.run([str(binary)] + sys.argv[1:])
            sys.exit(result.returncode)

    # No Rust binary found — fall back to the Python CLI for dev convenience.
    try:
        from .cli import main as py_main  # relative import within the package

        py_main()
    except ImportError:
        print(
            "error: vectro_cli binary not found and Python CLI fallback is "
            "unavailable.\n"
            "Build the Rust binary with:  cargo build --release -p vectro_cli",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
