"""
Comprehensive trace file validator for PROVE.

Validates CSV trace files against all structural and semantic rules
required by the PROVE runtime verification engine. Reports errors
and warnings with row-level detail.

Run standalone::

    python -m prove.utils.trace_validator trace.csv
    python -m prove.utils.trace_validator trace.csv --epsilon 2.0
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


class Severity(Enum):
    """Severity level for validation messages."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass
class ValidationMessage:
    """A single validation finding."""

    severity: Severity
    message: str
    row: Optional[int] = None
    eid: Optional[str] = None

    def __str__(self) -> str:
        parts: list[str] = []
        if self.row is not None:
            if self.eid:
                parts.append(f"Row {self.row} (eid={self.eid})")
            else:
                parts.append(f"Row {self.row}")
        return f"{'  '.join(parts)}: {self.message}" if parts else f"  {self.message}"


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    filepath: Path
    messages: List[ValidationMessage] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationMessage]:
        return [m for m in self.messages if m.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def format(self) -> str:
        lines = [f"Validating: {self.filepath}", ""]

        errs = self.errors
        warns = self.warnings

        if errs:
            lines.append("[ERRORS]")
            for m in errs:
                lines.append(f"  {m}")
            lines.append("")

        if warns:
            lines.append("[WARNINGS]")
            for m in warns:
                lines.append(f"  {m}")
            lines.append("")

        if not errs and not warns:
            lines.append("No issues found. Trace is valid.")
            lines.append("")

        lines.append(f"Summary: {len(errs)} error(s), {len(warns)} warning(s)")
        return "\n".join(lines)


_REQUIRED_HEADERS = {"eid", "processes", "vc", "timestamp"}
_VALID_EVENT_TYPES = {"local", "send", "receive"}
_PROP_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_'.]*$")


class TraceValidator:
    """
    Validates a CSV trace file against all PROVE requirements.

    Checks structural integrity, field parsing, event ID uniqueness,
    process consistency, initial events, intra-process ordering,
    vector clock consistency, message events, FIFO ordering, and
    proposition naming.
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath = Path(filepath)

    def validate(self) -> ValidationReport:
        """Run all validation checks and return the report."""
        report = ValidationReport(filepath=self.filepath)

        # Phase 1: structural checks (must pass to continue)
        if not self._check_file_readable(report):
            return report

        directives = self._parse_directives()
        raw_rows, headers = self._read_raw_rows(report)
        if raw_rows is None:
            return report

        if not self._check_headers(headers, report):
            return report

        # Phase 2: parse rows into validated dicts
        parsed_rows = self._parse_rows(raw_rows, headers, directives, report)
        if not parsed_rows:
            return report

        # Determine process set
        declared_processes = directives.get("processes")
        observed_processes = frozenset(r["process"] for r in parsed_rows)
        all_processes = declared_processes if declared_processes else observed_processes

        # Phase 3: semantic checks
        self._check_duplicate_eids(parsed_rows, report)
        self._check_process_consistency(
            parsed_rows, declared_processes, observed_processes, report
        )
        self._check_vc_keys(parsed_rows, all_processes, report)
        self._check_initial_events(parsed_rows, all_processes, report)
        self._check_intra_process_ordering(parsed_rows, report)
        self._check_vc_correctness(parsed_rows, report)
        self._check_vc_timestamp_consistency(parsed_rows, report)
        self._check_message_events(parsed_rows, all_processes, report)
        self._check_message_pairing_and_fifo(parsed_rows, report)
        self._check_proposition_names(parsed_rows, report)

        return report

    # ------------------------------------------------------------------ #
    # Phase 1: Structural
    # ------------------------------------------------------------------ #

    def _check_file_readable(self, report: ValidationReport) -> bool:
        if not self.filepath.exists():
            report.messages.append(
                ValidationMessage(Severity.ERROR, f"File not found: {self.filepath}")
            )
            return False
        try:
            self.filepath.read_text()
        except OSError as exc:
            report.messages.append(
                ValidationMessage(Severity.ERROR, f"Cannot read file: {exc}")
            )
            return False
        return True

    def _parse_directives(self) -> dict:
        directives: dict = {}
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
                    try:
                        directives["epsilon"] = float(val)
                    except ValueError:
                        pass
        return directives

    def _read_raw_rows(
        self, report: ValidationReport
    ) -> tuple[Optional[list[dict]], Optional[set[str]]]:
        lines: list[str] = []
        with open(self.filepath) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    lines.append(stripped)

        if not lines:
            report.messages.append(
                ValidationMessage(Severity.ERROR, "No data rows found in file")
            )
            return None, None

        reader = csv.DictReader(lines)
        headers = set(reader.fieldnames or [])
        if not headers:
            report.messages.append(
                ValidationMessage(Severity.ERROR, "No headers found in file")
            )
            return None, None

        rows = list(reader)
        if not rows:
            report.messages.append(
                ValidationMessage(Severity.ERROR, "No event rows found after header")
            )
            return None, None

        return rows, headers

    def _check_headers(self, headers: Optional[set[str]], report: ValidationReport) -> bool:
        if headers is None:
            return False
        missing = _REQUIRED_HEADERS - headers
        if missing:
            report.messages.append(
                ValidationMessage(
                    Severity.ERROR, f"Missing required headers: {sorted(missing)}"
                )
            )
            return False
        return True

    # ------------------------------------------------------------------ #
    # Phase 2: Field-level parsing
    # ------------------------------------------------------------------ #

    def _parse_rows(
        self,
        raw_rows: list[dict],
        headers: set[str],
        directives: dict,
        report: ValidationReport,
    ) -> list[dict]:
        """Parse raw CSV rows into validated dicts with typed fields."""
        parsed = []
        has_event_type = "event_type" in headers
        has_msg_partner = "msg_partner" in headers
        has_props = "props" in headers

        for i, row in enumerate(raw_rows):
            row_num = i + 2  # 1-indexed, +1 for header row
            eid = (row.get("eid") or "").strip()
            process = (row.get("processes") or "").strip()
            vc_str = (row.get("vc") or "").strip()
            ts_str = (row.get("timestamp") or "").strip()
            event_type = (row.get("event_type") or "").strip() if has_event_type else ""
            msg_partner = (row.get("msg_partner") or "").strip() if has_msg_partner else ""
            props_str = (row.get("props") or "").strip() if has_props else ""

            if not eid:
                report.messages.append(
                    ValidationMessage(Severity.ERROR, "Empty event ID", row=row_num)
                )
                continue

            if not process:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR, "Empty process field", row=row_num, eid=eid
                    )
                )
                continue

            # Timestamp
            timestamp: Optional[float] = None
            if not ts_str:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR, "Empty timestamp", row=row_num, eid=eid
                    )
                )
                continue
            try:
                timestamp = float(ts_str)
            except ValueError:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Invalid timestamp '{ts_str}' (must be a number)",
                        row=row_num,
                        eid=eid,
                    )
                )
                continue

            if timestamp < 0:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Negative timestamp {timestamp}",
                        row=row_num,
                        eid=eid,
                    )
                )

            # Vector clock
            vc_dict: Optional[dict[str, int]] = None
            if not vc_str:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR, "Empty vector clock", row=row_num, eid=eid
                    )
                )
                continue
            try:
                vc_dict = {}
                for part in vc_str.split(";"):
                    proc, count = part.strip().split(":")
                    val = int(count.strip())
                    if val < 0:
                        report.messages.append(
                            ValidationMessage(
                                Severity.ERROR,
                                f"Negative VC component {proc.strip()}:{val}",
                                row=row_num,
                                eid=eid,
                            )
                        )
                    vc_dict[proc.strip()] = val
            except (ValueError, TypeError):
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Malformed vector clock '{vc_str}'",
                        row=row_num,
                        eid=eid,
                    )
                )
                continue

            # Event type
            if not event_type:
                event_type = "local"
            if event_type not in _VALID_EVENT_TYPES:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Invalid event_type '{event_type}' "
                        f"(must be one of {sorted(_VALID_EVENT_TYPES)})",
                        row=row_num,
                        eid=eid,
                    )
                )

            # Message partner validation
            if event_type == "send" and not msg_partner:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        "Send event missing msg_partner (target process)",
                        row=row_num,
                        eid=eid,
                    )
                )
            if event_type == "receive" and not msg_partner:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        "Receive event missing msg_partner (source process)",
                        row=row_num,
                        eid=eid,
                    )
                )

            # Propositions
            props: frozenset[str] = frozenset()
            if props_str:
                props = frozenset(p.strip() for p in props_str.split("|") if p.strip())

            parsed.append(
                {
                    "row": row_num,
                    "eid": eid,
                    "process": process,
                    "vc": vc_dict,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "msg_partner": msg_partner or None,
                    "props": props,
                }
            )

        return parsed

    # ------------------------------------------------------------------ #
    # Phase 3: Semantic checks
    # ------------------------------------------------------------------ #

    def _check_duplicate_eids(
        self, rows: list[dict], report: ValidationReport
    ) -> None:
        seen: dict[str, int] = {}
        for r in rows:
            eid = r["eid"]
            if eid in seen:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Duplicate event ID '{eid}' (first seen at row {seen[eid]})",
                        row=r["row"],
                        eid=eid,
                    )
                )
            else:
                seen[eid] = r["row"]

    def _check_process_consistency(
        self,
        rows: list[dict],
        declared: Optional[frozenset[str]],
        observed: frozenset[str],
        report: ValidationReport,
    ) -> None:
        if declared is None:
            return

        for r in rows:
            if r["process"] not in declared:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Process '{r['process']}' not in declared "
                        f"system_processes {sorted(declared)}",
                        row=r["row"],
                        eid=r["eid"],
                    )
                )

        for p in sorted(declared):
            if p not in observed:
                report.messages.append(
                    ValidationMessage(
                        Severity.WARNING,
                        f"Declared process '{p}' has no events in the trace",
                    )
                )

    def _check_vc_keys(
        self,
        rows: list[dict],
        all_processes: frozenset[str],
        report: ValidationReport,
    ) -> None:
        for r in rows:
            vc_keys = frozenset(r["vc"].keys())
            if vc_keys != all_processes:
                missing = all_processes - vc_keys
                extra = vc_keys - all_processes
                parts = []
                if missing:
                    parts.append(f"missing {sorted(missing)}")
                if extra:
                    parts.append(f"extra {sorted(extra)}")
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"VC processes mismatch ({', '.join(parts)})",
                        row=r["row"],
                        eid=r["eid"],
                    )
                )

    def _check_initial_events(
        self,
        rows: list[dict],
        all_processes: frozenset[str],
        report: ValidationReport,
    ) -> None:
        candidates: dict[str, list[dict]] = {p: [] for p in all_processes}

        for r in rows:
            p = r["process"]
            if p not in all_processes:
                continue
            vc = r["vc"]
            if vc.get(p, 0) == 1 and all(
                vc.get(q, 0) == 0 for q in all_processes if q != p
            ):
                candidates[p].append(r)

        for p in sorted(all_processes):
            matches = candidates[p]
            if len(matches) == 0:
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Process {p}: no initial event "
                        f"(expected VC[{p}]=1, VC[q]=0 for all q!={p})",
                    )
                )
            elif len(matches) > 1:
                eids = ", ".join(m["eid"] for m in matches)
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"Process {p}: multiple initial event candidates: {eids}",
                    )
                )

    def _check_intra_process_ordering(
        self, rows: list[dict], report: ValidationReport
    ) -> None:
        by_process: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_process[r["process"]].append(r)

        for p, events in by_process.items():
            # Sort by own VC component to establish process-local order
            events_sorted = sorted(events, key=lambda e: e["vc"].get(p, 0))

            for i in range(1, len(events_sorted)):
                prev = events_sorted[i - 1]
                curr = events_sorted[i]
                prev_vc_p = prev["vc"].get(p, 0)
                curr_vc_p = curr["vc"].get(p, 0)

                # VC[p] should increment by 1 for consecutive events
                if curr_vc_p != prev_vc_p + 1:
                    report.messages.append(
                        ValidationMessage(
                            Severity.ERROR,
                            f"VC[{p}] gap: {prev['eid']} has VC[{p}]={prev_vc_p}, "
                            f"next event has VC[{p}]={curr_vc_p} (expected {prev_vc_p + 1})",
                            row=curr["row"],
                            eid=curr["eid"],
                        )
                    )

                # Timestamps must be non-decreasing within a process
                if curr["timestamp"] < prev["timestamp"]:
                    report.messages.append(
                        ValidationMessage(
                            Severity.ERROR,
                            f"Timestamp decreased on {p}: "
                            f"{prev['eid']} t={prev['timestamp']} > "
                            f"{curr['eid']} t={curr['timestamp']}",
                            row=curr["row"],
                            eid=curr["eid"],
                        )
                    )

    def _check_vc_correctness(
        self, rows: list[dict], report: ValidationReport
    ) -> None:
        """Check vector clock update rules for local, send, and receive events.

        - Local/send on process p: VC[k] for k != p must equal the
          previous event's VC[k] on the same process (only own component
          should change).
        - Receive on process q: VC[k] for k != q must be >= the previous
          event's VC[k] on the same process (merge must not decrease
          any component).
        """
        by_process: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_process[r["process"]].append(r)

        for p, events in by_process.items():
            sorted_events = sorted(events, key=lambda e: e["vc"].get(p, 0))

            for i in range(1, len(sorted_events)):
                prev = sorted_events[i - 1]
                curr = sorted_events[i]
                prev_vc = prev["vc"]
                curr_vc = curr["vc"]
                et = curr["event_type"]

                if et in ("local", "send"):
                    # For local/send: other components must not change
                    for k in curr_vc:
                        if k == p:
                            continue
                        if curr_vc.get(k, 0) != prev_vc.get(k, 0):
                            report.messages.append(
                                ValidationMessage(
                                    Severity.ERROR,
                                    f"{et.capitalize()} event on {p}: "
                                    f"VC[{k}] changed from {prev_vc.get(k, 0)} to "
                                    f"{curr_vc.get(k, 0)} (should stay the same)",
                                    row=curr["row"],
                                    eid=curr["eid"],
                                )
                            )
                elif et == "receive":
                    # For receive: other components must not decrease (merge)
                    for k in curr_vc:
                        if k == p:
                            continue
                        if curr_vc.get(k, 0) < prev_vc.get(k, 0):
                            report.messages.append(
                                ValidationMessage(
                                    Severity.ERROR,
                                    f"Receive event on {p}: VC[{k}] decreased "
                                    f"from {prev_vc.get(k, 0)} to {curr_vc.get(k, 0)} "
                                    f"(merge must not decrease components)",
                                    row=curr["row"],
                                    eid=curr["eid"],
                                )
                            )

    def _check_vc_timestamp_consistency(
        self, rows: list[dict], report: ValidationReport
    ) -> None:
        """If e ≺ f by VC (strict componentwise), then t(e) < t(f)."""
        # Only check for cross-process pairs to avoid quadratic blowup
        # (intra-process is already checked in _check_intra_process_ordering)
        by_process: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_process[r["process"]].append(r)

        processes = list(by_process.keys())
        for i, p1 in enumerate(processes):
            for p2 in processes[i + 1 :]:
                for e1 in by_process[p1]:
                    for e2 in by_process[p2]:
                        vc1 = e1["vc"]
                        vc2 = e2["vc"]
                        common = set(vc1.keys()) & set(vc2.keys())
                        if not common:
                            continue
                        # e1 < e2 by VC?
                        if all(vc1.get(k, 0) <= vc2.get(k, 0) for k in common) and any(
                            vc1.get(k, 0) < vc2.get(k, 0) for k in common
                        ):
                            if e1["timestamp"] >= e2["timestamp"]:
                                report.messages.append(
                                    ValidationMessage(
                                        Severity.ERROR,
                                        f"VC ordering {e1['eid']} ≺ {e2['eid']} "
                                        f"but timestamp {e1['eid']} "
                                        f"(t={e1['timestamp']}) >= "
                                        f"{e2['eid']} (t={e2['timestamp']})",
                                        row=e2["row"],
                                        eid=e2["eid"],
                                    )
                                )
                        # e2 < e1 by VC?
                        elif all(vc2.get(k, 0) <= vc1.get(k, 0) for k in common) and any(
                            vc2.get(k, 0) < vc1.get(k, 0) for k in common
                        ):
                            if e2["timestamp"] >= e1["timestamp"]:
                                report.messages.append(
                                    ValidationMessage(
                                        Severity.ERROR,
                                        f"VC ordering {e2['eid']} ≺ {e1['eid']} "
                                        f"but timestamp {e2['eid']} "
                                        f"(t={e2['timestamp']}) >= "
                                        f"{e1['eid']} (t={e1['timestamp']})",
                                        row=e1["row"],
                                        eid=e1["eid"],
                                    )
                                )

    def _check_message_events(
        self,
        rows: list[dict],
        all_processes: frozenset[str],
        report: ValidationReport,
    ) -> None:
        for r in rows:
            et = r["event_type"]
            partner = r["msg_partner"]

            if et == "send" and partner:
                if partner not in all_processes:
                    report.messages.append(
                        ValidationMessage(
                            Severity.ERROR,
                            f"Send target '{partner}' is not a known process",
                            row=r["row"],
                            eid=r["eid"],
                        )
                    )
                if partner == r["process"]:
                    report.messages.append(
                        ValidationMessage(
                            Severity.ERROR,
                            f"Send event targets own process '{partner}'",
                            row=r["row"],
                            eid=r["eid"],
                        )
                    )

            if et == "receive" and partner:
                if partner not in all_processes:
                    report.messages.append(
                        ValidationMessage(
                            Severity.ERROR,
                            f"Receive source '{partner}' is not a known process",
                            row=r["row"],
                            eid=r["eid"],
                        )
                    )
                if partner == r["process"]:
                    report.messages.append(
                        ValidationMessage(
                            Severity.ERROR,
                            f"Receive event has own process '{partner}' as source",
                            row=r["row"],
                            eid=r["eid"],
                        )
                    )

    def _check_message_pairing_and_fifo(
        self, rows: list[dict], report: ValidationReport
    ) -> None:
        # Collect sends and receives per (sender, receiver) pair
        sends: dict[tuple[str, str], list[dict]] = defaultdict(list)
        receives: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for r in rows:
            if r["event_type"] == "send" and r["msg_partner"]:
                pair = (r["process"], r["msg_partner"])
                sends[pair].append(r)
            elif r["event_type"] == "receive" and r["msg_partner"]:
                pair = (r["msg_partner"], r["process"])
                receives[pair].append(r)

        # Check pairing: sends without receives
        for pair, send_list in sends.items():
            recv_list = receives.get(pair, [])
            if len(send_list) > len(recv_list):
                diff = len(send_list) - len(recv_list)
                report.messages.append(
                    ValidationMessage(
                        Severity.WARNING,
                        f"{diff} send(s) from {pair[0]} to {pair[1]} "
                        f"without matching receive(s)",
                    )
                )

        # Check pairing: receives without sends
        for pair, recv_list in receives.items():
            send_list = sends.get(pair, [])
            if len(recv_list) > len(send_list):
                diff = len(recv_list) - len(send_list)
                report.messages.append(
                    ValidationMessage(
                        Severity.ERROR,
                        f"{diff} receive(s) on {pair[1]} from {pair[0]} "
                        f"without matching send(s)",
                    )
                )

        # FIFO check: for each (sender, receiver), sends and receives
        # must be in consistent order by their process-local VC component
        for pair in sends:
            send_list = sends[pair]
            recv_list = receives.get(pair, [])
            if len(recv_list) < 2:
                continue

            # Sort sends by sender's VC component
            sender_proc = pair[0]
            recv_proc = pair[1]
            sorted_sends = sorted(
                send_list, key=lambda e: e["vc"].get(sender_proc, 0)
            )
            sorted_recvs = sorted(
                recv_list, key=lambda e: e["vc"].get(recv_proc, 0)
            )

            # The i-th receive should correspond to the i-th send (FIFO)
            # Check that receive timestamps are in the same order as send timestamps
            for i in range(1, min(len(sorted_sends), len(sorted_recvs))):
                prev_recv = sorted_recvs[i - 1]
                curr_recv = sorted_recvs[i]
                prev_send = sorted_sends[i - 1]
                curr_send = sorted_sends[i]

                # If send_1 is before send_2 (by sender VC),
                # then recv_1 must be before recv_2 (by receiver VC)
                s1_vc = prev_send["vc"].get(sender_proc, 0)
                s2_vc = curr_send["vc"].get(sender_proc, 0)
                r1_vc = prev_recv["vc"].get(recv_proc, 0)
                r2_vc = curr_recv["vc"].get(recv_proc, 0)

                if s1_vc < s2_vc and r1_vc >= r2_vc:
                    report.messages.append(
                        ValidationMessage(
                            Severity.ERROR,
                            f"FIFO violation {pair[0]}->{pair[1]}: "
                            f"send order {prev_send['eid']} before {curr_send['eid']}, "
                            f"but receive order {prev_recv['eid']} "
                            f"(VC[{recv_proc}]={r1_vc}) not before "
                            f"{curr_recv['eid']} (VC[{recv_proc}]={r2_vc})",
                        )
                    )

    def _check_proposition_names(
        self, rows: list[dict], report: ValidationReport
    ) -> None:
        warned: set[str] = set()
        for r in rows:
            for prop in r["props"]:
                if prop not in warned and not _PROP_PATTERN.match(prop):
                    report.messages.append(
                        ValidationMessage(
                            Severity.WARNING,
                            f"Proposition '{prop}' does not match naming "
                            f"convention [a-zA-Z_][a-zA-Z0-9_'.]*",
                            row=r["row"],
                            eid=r["eid"],
                        )
                    )
                    warned.add(prop)


def main() -> None:
    """CLI entry point for the trace validator."""
    parser = argparse.ArgumentParser(
        description="Validate a PROVE trace CSV file",
        prog="python -m prove.utils.trace_validator",
    )
    parser.add_argument("trace", type=Path, help="Path to trace CSV file")
    args = parser.parse_args()

    validator = TraceValidator(args.trace)
    report = validator.validate()
    print(report.format())

    if not report.is_valid:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
