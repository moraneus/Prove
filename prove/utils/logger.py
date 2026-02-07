"""
Structured logging for the EPLTL monitor.

Provides configurable log levels (silent, normal, verbose, debug)
with consistent formatting for progress updates, verdicts,
and monitoring statistics.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, Dict, TextIO


class LogLevel(Enum):
    """
    Logging levels for the monitor.

    SILENT:  No output at all.
    NORMAL:  Final verdict only.
    VERBOSE: Progress information and statistics.
    DEBUG:   Detailed per-event processing output.
    """

    SILENT = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


class MonitorLogger:
    """
    Structured logger for the EPLTL monitor.

    Provides consistent formatting for progress updates, debug
    information, verdicts, and statistics. Output is filtered
    by the configured log level.

    Attributes:
        level: The minimum log level to display.
        stream: The output stream (defaults to stdout).
    """

    def __init__(
        self,
        level: LogLevel = LogLevel.NORMAL,
        stream: TextIO = sys.stdout,
    ) -> None:
        """
        Initialize logger with level and output stream.

        Args:
            level: Minimum log level to display.
            stream: Output stream (default: sys.stdout).
        """
        self.level: LogLevel = level
        self.stream: TextIO = stream

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log a debug message (only shown at DEBUG level).

        Args:
            message: The message to log.
            **kwargs: Additional key-value pairs to include.
        """
        if self.level.value >= LogLevel.DEBUG.value:
            self._write(f"[DEBUG] {message}")
            for k, v in kwargs.items():
                self._write(f"  {k}: {v}")

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log an info message (shown at VERBOSE and DEBUG levels).

        Args:
            message: The message to log.
            **kwargs: Additional key-value pairs to include.
        """
        if self.level.value >= LogLevel.VERBOSE.value:
            self._write(message)
            for k, v in kwargs.items():
                self._write(f"  {k}: {v}")

    def verdict_satisfied(self) -> None:
        """Log a SATISFIED verdict (shown at NORMAL level and above)."""
        if self.level.value >= LogLevel.NORMAL.value:
            self._write(
                "SATISFIED: Property holds for at least one linearization"
            )

    def verdict_violated(self) -> None:
        """Log a VIOLATED verdict (shown at NORMAL level and above)."""
        if self.level.value >= LogLevel.NORMAL.value:
            self._write(
                "VIOLATED: Property does not hold for any linearization"
            )

    def statistics(self, stats: Dict[str, Any]) -> None:
        """
        Log monitoring statistics (shown at VERBOSE level and above).

        Args:
            stats: Dictionary of statistic names to values.
        """
        if self.level.value >= LogLevel.VERBOSE.value:
            self._write("=== Statistics ===")
            for key, value in stats.items():
                label = key.replace("_", " ").title()
                self._write(f"  {label}: {value}")

    def event_processed(self, event_id: str, node_count: int) -> None:
        """
        Log event processing (shown at DEBUG level).

        Args:
            event_id: The ID of the processed event.
            node_count: Current number of nodes in the graph.
        """
        if self.level.value >= LogLevel.DEBUG.value:
            self._write(f"[DEBUG] Processed {event_id} (nodes: {node_count})")

    def _write(self, message: str) -> None:
        """Write a line to the output stream."""
        self.stream.write(message + "\n")
