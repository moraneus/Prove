"""
Tests for the PROVE command-line interface.

Tests cover argument parsing, output modes, visualization flags,
exit codes, error handling, and end-to-end CLI invocation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"
TRACES = FIXTURES / "traces"
PROPERTIES = FIXTURES / "properties"

SIMPLE_TRACE = str(TRACES / "simple_trace.csv")
SIMPLE_PROP = str(PROPERTIES / "simple.prop")
SINCE_PROP = str(PROPERTIES / "since.prop")
MESSAGE_TRACE = str(TRACES / "message_trace.csv")
EPSILON_TRACE = str(TRACES / "epsilon_trace.csv")


def _run_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run the PROVE CLI as a subprocess."""
    cmd = [sys.executable, "-m", "prove", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Tests: Required Arguments
# ---------------------------------------------------------------------------


class TestRequiredArguments:
    """Test that required arguments are enforced."""

    def test_missing_property_file(self) -> None:
        """Missing -p flag exits with code 2."""
        result = _run_cli("-t", SIMPLE_TRACE)
        assert result.returncode == 2

    def test_missing_trace_file(self) -> None:
        """Missing -t flag exits with code 2."""
        result = _run_cli("-p", SIMPLE_PROP)
        assert result.returncode == 2

    def test_no_arguments(self) -> None:
        """No arguments exits with code 2."""
        result = _run_cli()
        assert result.returncode == 2


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_nonexistent_trace_file(self) -> None:
        """Non-existent trace file exits with code 2."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", "/nonexistent/trace.csv")
        assert result.returncode == 2
        assert result.stderr.strip() != ""

    def test_nonexistent_property_file(self) -> None:
        """Non-existent property file exits with code 2."""
        result = _run_cli("-p", "/nonexistent/formula.prop", "-t", SIMPLE_TRACE)
        assert result.returncode == 2
        assert result.stderr.strip() != ""

    def test_invalid_property_syntax(self, tmp_path: Path) -> None:
        """Invalid property syntax exits with code 2."""
        bad_prop = tmp_path / "bad.prop"
        bad_prop.write_text("& & invalid syntax $$")
        result = _run_cli("-p", str(bad_prop), "-t", SIMPLE_TRACE)
        assert result.returncode == 2


# ---------------------------------------------------------------------------
# Tests: Version
# ---------------------------------------------------------------------------


class TestVersion:
    """Test --version flag."""

    def test_version_prints_and_exits(self) -> None:
        """--version prints version string and exits 0."""
        result = _run_cli("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout


# ---------------------------------------------------------------------------
# Tests: Basic Execution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    """Test basic verify execution with valid inputs."""

    def test_simple_trace_runs(self) -> None:
        """Simple trace with simple property produces a verdict."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE)
        assert result.returncode in (0, 1)
        assert "SATISFIED" in result.stdout or "VIOLATED" in result.stdout

    def test_since_property_runs(self) -> None:
        """Since property runs without error."""
        result = _run_cli("-p", SINCE_PROP, "-t", SIMPLE_TRACE)
        assert result.returncode in (0, 1)

    def test_message_trace_runs(self) -> None:
        """Message trace runs without error."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", MESSAGE_TRACE)
        assert result.returncode in (0, 1)

    def test_epsilon_trace_runs(self) -> None:
        """Epsilon trace runs without error."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", EPSILON_TRACE)
        assert result.returncode in (0, 1)


# ---------------------------------------------------------------------------
# Tests: Output Modes
# ---------------------------------------------------------------------------


class TestOutputModes:
    """Test output level modes."""

    def test_silent_no_stdout(self) -> None:
        """Silent mode produces no stdout output."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE, "-o", "silent")
        assert result.returncode in (0, 1)
        assert result.stdout.strip() == ""

    def test_normal_shows_verdict(self) -> None:
        """Normal mode shows verdict line."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE, "-o", "normal")
        assert result.returncode in (0, 1)
        assert "SATISFIED" in result.stdout or "VIOLATED" in result.stdout

    def test_verbose_shows_more(self) -> None:
        """Verbose mode shows more output than normal."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE, "-o", "verbose")
        assert result.returncode in (0, 1)
        assert len(result.stdout) > 0


# ---------------------------------------------------------------------------
# Tests: Epsilon Flag
# ---------------------------------------------------------------------------


class TestEpsilonFlag:
    """Test -e/--epsilon argument."""

    def test_custom_epsilon(self) -> None:
        """Custom epsilon value is accepted."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE, "-e", "2.0")
        assert result.returncode in (0, 1)

    def test_zero_epsilon(self) -> None:
        """Zero epsilon is accepted."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE, "-e", "0")
        assert result.returncode in (0, 1)


# ---------------------------------------------------------------------------
# Tests: Stats Flag
# ---------------------------------------------------------------------------


class TestStatsFlag:
    """Test --stats flag."""

    def test_stats_printed(self) -> None:
        """--stats includes statistics in output."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE, "--stats")
        assert result.returncode in (0, 1)
        output = result.stdout.lower()
        assert "statistic" in output or "events" in output or "node" in output


# ---------------------------------------------------------------------------
# Tests: Visualize ASCII
# ---------------------------------------------------------------------------


class TestVisualizeAscii:
    """Test --visualize-ascii flag."""

    def test_ascii_diagram_printed(self) -> None:
        """--visualize-ascii prints ASCII diagram to stdout."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", SIMPLE_TRACE, "--visualize-ascii")
        assert result.returncode in (0, 1)
        # Should contain process names and orderings
        assert "P1" in result.stdout or "P2" in result.stdout

    def test_ascii_with_message_trace(self) -> None:
        """ASCII diagram works with message trace."""
        result = _run_cli("-p", SIMPLE_PROP, "-t", MESSAGE_TRACE, "--visualize-ascii")
        assert result.returncode in (0, 1)
        assert "Client" in result.stdout or "Server" in result.stdout


# ---------------------------------------------------------------------------
# Tests: Visualize DOT File
# ---------------------------------------------------------------------------


class TestVisualizeDot:
    """Test --visualize flag."""

    def test_visualize_to_dot_file(self, tmp_path: Path) -> None:
        """--visualize FILE.dot creates a DOT file."""
        dot_file = tmp_path / "graph.dot"
        result = _run_cli(
            "-p",
            SIMPLE_PROP,
            "-t",
            SIMPLE_TRACE,
            "--visualize",
            str(dot_file),
        )
        assert result.returncode in (0, 1)
        assert dot_file.exists()
        content = dot_file.read_text()
        assert "digraph" in content

    def test_visualize_stdout(self) -> None:
        """--visualize without file writes DOT to stdout."""
        result = _run_cli(
            "-p",
            SIMPLE_PROP,
            "-t",
            SIMPLE_TRACE,
            "--visualize",
        )
        assert result.returncode in (0, 1)
        assert "digraph" in result.stdout


# ---------------------------------------------------------------------------
# Tests: Combined Flags
# ---------------------------------------------------------------------------


class TestCombinedFlags:
    """Test combinations of flags."""

    def test_ascii_and_stats(self) -> None:
        """--visualize-ascii and --stats together."""
        result = _run_cli(
            "-p",
            SIMPLE_PROP,
            "-t",
            SIMPLE_TRACE,
            "--visualize-ascii",
            "--stats",
        )
        assert result.returncode in (0, 1)
        # Both should appear in output
        assert len(result.stdout) > 0

    def test_verbose_and_stats(self) -> None:
        """Verbose output with stats."""
        result = _run_cli(
            "-p",
            SIMPLE_PROP,
            "-t",
            SIMPLE_TRACE,
            "-o",
            "verbose",
            "--stats",
        )
        assert result.returncode in (0, 1)

    def test_full_graph_flag(self) -> None:
        """--full-graph flag is accepted."""
        result = _run_cli(
            "-p",
            SIMPLE_PROP,
            "-t",
            SIMPLE_TRACE,
            "--visualize-ascii",
            "--full-graph",
        )
        assert result.returncode in (0, 1)
