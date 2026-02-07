"""
Shared pytest fixtures for the PROVE test suite.

Provides reusable fixtures for creating test vector clocks, events,
and temporary file paths used across unit and integration tests.
"""

from pathlib import Path

import pytest


@pytest.fixture
def three_processes() -> frozenset[str]:
    """A standard set of three process identifiers."""
    return frozenset({"P1", "P2", "P3"})


@pytest.fixture
def two_processes() -> frozenset[str]:
    """A standard set of two process identifiers."""
    return frozenset({"P1", "P2"})


@pytest.fixture
def tmp_trace_file(tmp_path: Path) -> Path:
    """Path for a temporary trace CSV file."""
    return tmp_path / "trace.csv"


@pytest.fixture
def tmp_property_file(tmp_path: Path) -> Path:
    """Path for a temporary property file."""
    return tmp_path / "formula.prop"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def traces_dir(fixtures_dir: Path) -> Path:
    """Path to the test trace fixtures directory."""
    return fixtures_dir / "traces"


@pytest.fixture
def properties_dir(fixtures_dir: Path) -> Path:
    """Path to the test property fixtures directory."""
    return fixtures_dir / "properties"
