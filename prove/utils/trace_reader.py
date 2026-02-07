"""
CSV trace file parser for partial order executions.

Reads event traces in CSV format, constructing Event objects with
vector clocks and timestamps, and optionally computing the complete
partial order including epsilon-based ordering.

The algorithm operates offline, so all events are loaded upfront
before the partial order is computed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, List, Optional

from prove.core.event import Event
from prove.core.partial_order import PartialOrder
from prove.core.vector_clock import VectorClock


@dataclass
class TraceMetadata:
    """
    Metadata extracted from a trace file.

    Attributes:
        processes: Set of all process IDs.
        event_count: Total number of events.
        epsilon: Maximum clock skew (if specified in file or overridden).
    """

    processes: FrozenSet[str]
    event_count: int
    epsilon: Optional[float] = None


@dataclass
class TraceData:
    """
    Complete trace data loaded from a file.

    Since the algorithm operates offline, all events are loaded
    and the complete partial order is computed upfront.

    Attributes:
        events: List of all events in file order.
        metadata: Trace metadata.
        partial_order: Complete partial order relation.
    """

    events: List[Event]
    metadata: TraceMetadata
    partial_order: PartialOrder


_REQUIRED_HEADERS = {"eid", "processes", "vc", "timestamp"}


class TraceReader:
    """
    Parses CSV trace files into Event objects.

    Loads all events (offline processing) and can compute the complete
    partial order including epsilon-based ordering.

    Expected CSV format::

        # Optional: system_processes directive
        # system_processes: P1|P2|P3

        # Optional: epsilon directive
        # epsilon: 2.0

        # Required headers
        eid,processes,vc,timestamp,props,event_type,msg_partner

    Attributes:
        filepath: Path to the trace CSV file.
    """

    def __init__(self, filepath: Path) -> None:
        """
        Initialize reader with file path.

        Args:
            filepath: Path to the CSV trace file.
        """
        self.filepath: Path = Path(filepath)

    def read_all(self, epsilon: Optional[float] = None) -> TraceData:
        """
        Read all events and compute complete partial order.

        Args:
            epsilon: Override epsilon from file. If None, uses the value
                     from the file directive (or infinity if absent).

        Returns:
            TraceData with all events and computed partial order.
        """
        directives = self._parse_directives()
        events = self.read_events()

        # Determine epsilon
        if epsilon is not None:
            eps = epsilon
        elif directives.get("epsilon") is not None:
            eps = directives["epsilon"]
        else:
            eps = float("inf")

        # Determine processes
        if directives.get("processes") is not None:
            processes = directives["processes"]
        else:
            processes = frozenset(e.process for e in events)

        po = PartialOrder(events, eps)

        metadata = TraceMetadata(
            processes=processes,
            event_count=len(events),
            epsilon=epsilon if epsilon is not None else directives.get("epsilon"),
        )

        return TraceData(events=events, metadata=metadata, partial_order=po)

    def read_metadata(self) -> TraceMetadata:
        """
        Read only metadata without parsing all events.

        Reads directives and counts events from the file.

        Returns:
            TraceMetadata with processes, event count, and epsilon.
        """
        directives = self._parse_directives()

        # If no process directive, scan events for process IDs
        if directives.get("processes") is not None:
            processes = directives["processes"]
        else:
            processes = self._infer_processes()

        event_count = self._count_events()

        return TraceMetadata(
            processes=processes,
            event_count=event_count,
            epsilon=directives.get("epsilon"),
        )

    def read_events(self) -> List[Event]:
        """
        Read all events from the file.

        Returns:
            List of Event objects in file order.

        Raises:
            FileNotFoundError: If the trace file does not exist.
            ValueError: If the file has missing required headers.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Trace file not found: {self.filepath}")

        directives = self._parse_directives()
        lines = self._read_data_lines()

        if not lines:
            return []

        reader = csv.DictReader(lines)
        headers = set(reader.fieldnames or [])

        # Validate required headers
        missing = _REQUIRED_HEADERS - headers
        if missing:
            raise ValueError(f"Missing required headers: {sorted(missing)}")

        # Determine processes (for VC parsing)
        if directives.get("processes") is not None:
            all_processes = directives["processes"]
        else:
            # First pass: infer processes from VC fields
            all_processes = self._infer_processes()

        events: List[Event] = []
        for row in reader:
            event = self._parse_event_row(row, all_processes)
            events.append(event)

        return events

    def validate(self) -> List[str]:
        """
        Validate the trace file and return a list of error strings.

        Validates:
        - Required headers are present (including timestamp)
        - Vector clocks are parseable
        - Basic structural integrity

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []

        if not self.filepath.exists():
            errors.append(f"File not found: {self.filepath}")
            return errors

        lines = self._read_data_lines()
        if not lines:
            errors.append("No data rows found in file")
            return errors

        # Check headers
        reader = csv.DictReader(lines)
        headers = set(reader.fieldnames or [])

        if not headers:
            errors.append("No headers found in file")
            return errors

        missing = _REQUIRED_HEADERS - headers
        if missing:
            errors.append(f"Missing required headers: {sorted(missing)}")

        return errors

    # ------------------------------------------------------------------ #
    # Static helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_vector_clock(s: str, processes: FrozenSet[str]) -> VectorClock:
        """
        Parse vector clock from string format ``P1:2;P2:1;P3:0``.

        Args:
            s: The vector clock string.
            processes: Set of all process IDs.

        Returns:
            A VectorClock instance.
        """
        s = s.strip()
        values: dict[str, int] = {}
        for part in s.split(";"):
            proc, count = part.strip().split(":")
            values[proc.strip()] = int(count.strip())
        return VectorClock(processes, initial_values=values)

    @staticmethod
    def parse_timestamp(s: str) -> float:
        """
        Parse timestamp string into float.

        Args:
            s: The timestamp string.

        Returns:
            The timestamp as a float.
        """
        return float(s.strip())

    @staticmethod
    def parse_propositions(s: str) -> FrozenSet[str]:
        """
        Parse pipe-separated propositions into a frozenset.

        Args:
            s: The propositions string (e.g. ``ready|done``).

        Returns:
            A frozenset of proposition names (empty if input is empty).
        """
        s = s.strip()
        if not s:
            return frozenset()
        return frozenset(p.strip() for p in s.split("|") if p.strip())

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _parse_directives(self) -> dict:
        """Extract directives from comment lines at the top of the file."""
        directives: dict = {}
        if not self.filepath.exists():
            return directives

        with open(self.filepath) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    continue
                content = line.lstrip("#").strip()
                if content.startswith("system_processes:"):
                    val = content.split(":", 1)[1].strip()
                    directives["processes"] = frozenset(
                        p.strip() for p in val.split("|") if p.strip()
                    )
                elif content.startswith("epsilon:"):
                    val = content.split(":", 1)[1].strip()
                    directives["epsilon"] = float(val)

        return directives

    def _read_data_lines(self) -> List[str]:
        """Read non-comment, non-empty lines from the file."""
        if not self.filepath.exists():
            return []

        lines: List[str] = []
        with open(self.filepath) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    lines.append(stripped)
        return lines

    def _infer_processes(self) -> FrozenSet[str]:
        """Infer process set from event data."""
        lines = self._read_data_lines()
        if not lines:
            return frozenset()

        reader = csv.DictReader(lines)
        processes: set[str] = set()
        for row in reader:
            proc = row.get("processes", "").strip()
            if proc:
                processes.add(proc)
        return frozenset(processes)

    def _count_events(self) -> int:
        """Count data rows (excluding header)."""
        lines = self._read_data_lines()
        if not lines:
            return 0
        return len(lines) - 1  # Subtract header row

    def _parse_event_row(self, row: dict, all_processes: FrozenSet[str]) -> Event:
        """Parse a single CSV row into an Event."""
        eid = row["eid"].strip()
        process = row["processes"].strip()
        vc = self.parse_vector_clock(row["vc"], all_processes)
        timestamp = self.parse_timestamp(row["timestamp"])
        props = self.parse_propositions(row.get("props", ""))

        event_type = row.get("event_type", "").strip() or "local"
        msg_partner = row.get("msg_partner", "").strip() or None

        source_process = None
        target_process = None
        if event_type == "send" and msg_partner:
            target_process = msg_partner
        elif event_type == "receive" and msg_partner:
            source_process = msg_partner

        return Event(
            eid=eid,
            process=process,
            vector_clock=vc,
            timestamp=timestamp,
            propositions=props,
            event_type=event_type,
            source_process=source_process,
            target_process=target_process,
        )
