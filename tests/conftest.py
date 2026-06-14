"""
Pytest configuration and shared fixtures for the Vectro test suite.

Registers custom markers used across the test suite, so pytest does not emit
PytestUnknownMarkWarning when test files use them without importing conftest.
"""

import tests._path_setup as _path_setup  # noqa: F401

_path_setup.ensure_repo_root_on_path()


def pytest_configure(config):
    config.addinivalue_line("markers", "intel: test targets Intel x86 platforms")
    config.addinivalue_line("markers", "m3: test targets Apple M3/ARM64 platforms")
    config.addinivalue_line("markers", "linux: test targets Linux platforms")
    config.addinivalue_line("markers", "throughput: benchmark-style throughput test (slower)")
    config.addinivalue_line("markers", "quality: quantization accuracy assessment")
    config.addinivalue_line("markers", "latency: single-vector latency measurement")
