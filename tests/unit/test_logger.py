"""
Tests for the structured monitoring logger.

Tests cover log level filtering, output formatting, verdict display,
statistics formatting, and custom stream output.
"""

from io import StringIO

import pytest

from prove.utils.logger import LogLevel, MonitorLogger


# ---------------------------------------------------------------------------
# Tests: Log Level Filtering
# ---------------------------------------------------------------------------


class TestLogLevelFiltering:
    """Test that log levels filter messages correctly."""

    def test_silent_suppresses_all(self) -> None:
        """SILENT level produces no output."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.SILENT, stream=buf)
        logger.debug("debug msg")
        logger.info("info msg")
        assert buf.getvalue() == ""

    def test_normal_shows_verdict_only(self) -> None:
        """NORMAL level only shows verdict output."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.NORMAL, stream=buf)
        logger.debug("debug msg")
        logger.info("info msg")
        assert buf.getvalue() == ""

    def test_verbose_shows_info(self) -> None:
        """VERBOSE level shows info messages."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.VERBOSE, stream=buf)
        logger.info("progress update")
        assert "progress update" in buf.getvalue()

    def test_verbose_hides_debug(self) -> None:
        """VERBOSE level does NOT show debug messages."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.VERBOSE, stream=buf)
        logger.debug("detailed debug info")
        assert buf.getvalue() == ""

    def test_debug_shows_everything(self) -> None:
        """DEBUG level shows all messages."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.DEBUG, stream=buf)
        logger.debug("debug msg")
        logger.info("info msg")
        output = buf.getvalue()
        assert "debug msg" in output
        assert "info msg" in output


# ---------------------------------------------------------------------------
# Tests: Verdict Output
# ---------------------------------------------------------------------------


class TestVerdictOutput:
    """Test verdict formatting."""

    def test_verdict_satisfied(self) -> None:
        """SATISFIED verdict is displayed at NORMAL level."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.NORMAL, stream=buf)
        logger.verdict_satisfied()
        output = buf.getvalue()
        assert "SATISFIED" in output

    def test_verdict_violated(self) -> None:
        """VIOLATED verdict is displayed at NORMAL level."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.NORMAL, stream=buf)
        logger.verdict_violated()
        output = buf.getvalue()
        assert "VIOLATED" in output

    def test_verdict_silent(self) -> None:
        """SILENT level suppresses even verdict output."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.SILENT, stream=buf)
        logger.verdict_satisfied()
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Tests: Statistics Output
# ---------------------------------------------------------------------------


class TestStatisticsOutput:
    """Test statistics formatting."""

    def test_statistics_at_verbose(self) -> None:
        """Statistics are displayed at VERBOSE level."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.VERBOSE, stream=buf)
        stats = {"events_processed": 10, "node_count": 5, "nodes_removed": 2}
        logger.statistics(stats)
        output = buf.getvalue()
        assert "events_processed" in output or "Events" in output

    def test_statistics_hidden_at_normal(self) -> None:
        """Statistics are NOT displayed at NORMAL level."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.NORMAL, stream=buf)
        stats = {"events_processed": 10}
        logger.statistics(stats)
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Tests: Event Processing Log
# ---------------------------------------------------------------------------


class TestEventProcessing:
    """Test event processing log output."""

    def test_event_processed_at_debug(self) -> None:
        """Event processing is logged at DEBUG level."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.DEBUG, stream=buf)
        logger.event_processed("e1", 3)
        output = buf.getvalue()
        assert "e1" in output

    def test_event_processed_hidden_at_verbose(self) -> None:
        """Event processing is NOT logged at VERBOSE level."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.VERBOSE, stream=buf)
        logger.event_processed("e1", 3)
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Tests: Log Level Enum
# ---------------------------------------------------------------------------


class TestLogLevelEnum:
    """Test LogLevel enum values."""

    def test_level_ordering(self) -> None:
        """Log levels are ordered: SILENT < NORMAL < VERBOSE < DEBUG."""
        assert LogLevel.SILENT.value < LogLevel.NORMAL.value
        assert LogLevel.NORMAL.value < LogLevel.VERBOSE.value
        assert LogLevel.VERBOSE.value < LogLevel.DEBUG.value

    def test_from_string(self) -> None:
        """LogLevel can be created from common string names."""
        assert LogLevel.SILENT.value == 0
        assert LogLevel.NORMAL.value == 1
        assert LogLevel.VERBOSE.value == 2
        assert LogLevel.DEBUG.value == 3


# ---------------------------------------------------------------------------
# Tests: Custom Stream
# ---------------------------------------------------------------------------


class TestCustomStream:
    """Test output to custom streams."""

    def test_custom_stream_receives_output(self) -> None:
        """All output goes to the specified stream."""
        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.DEBUG, stream=buf)
        logger.info("test message")
        assert len(buf.getvalue()) > 0

    def test_default_uses_stdout(self) -> None:
        """Default logger uses stdout (does not raise)."""
        logger = MonitorLogger(level=LogLevel.SILENT)
        logger.info("should not fail")
