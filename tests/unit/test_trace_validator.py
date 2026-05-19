"""
Tests for the trace file validator.

Covers structural checks, field parsing, duplicate IDs, process consistency,
initial events, intra-process ordering, VC correctness, VC-timestamp
consistency, message events, FIFO ordering, and proposition naming.
"""

from pathlib import Path

import pytest

from prove.utils.trace_validator import (
    Severity,
    TraceValidator,
    ValidationMessage,
    ValidationReport,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
TRACES = FIXTURES / "traces"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_trace(tmp_path: Path, content: str) -> Path:
    """Write trace content to a temp file and return its path."""
    p = tmp_path / "trace.csv"
    p.write_text(content)
    return p


def _error_messages(report: ValidationReport) -> list[str]:
    return [m.message for m in report.errors]


def _warning_messages(report: ValidationReport) -> list[str]:
    return [m.message for m in report.warnings]


# ---------------------------------------------------------------------------
# Tests: File-Level Structural Checks
# ---------------------------------------------------------------------------


class TestStructuralChecks:
    """Test file existence, readability, and header validation."""

    def test_missing_file(self, tmp_path: Path) -> None:
        v = TraceValidator(tmp_path / "nonexistent.csv")
        report = v.validate()
        assert not report.is_valid
        assert any("not found" in m.message for m in report.errors)

    def test_empty_file(self, tmp_path: Path) -> None:
        p = _write_trace(tmp_path, "")
        report = TraceValidator(p).validate()
        assert not report.is_valid
        assert any("No data rows" in m.message for m in report.errors)

    def test_header_only(self, tmp_path: Path) -> None:
        p = _write_trace(tmp_path, "eid,processes,vc,timestamp,props\n")
        report = TraceValidator(p).validate()
        assert not report.is_valid
        assert any("No event rows" in m.message for m in report.errors)

    def test_missing_required_header(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc\ne1,P1,P1:1\n",
        )
        report = TraceValidator(p).validate()
        assert not report.is_valid
        assert any("timestamp" in m.message.lower() for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Valid Traces
# ---------------------------------------------------------------------------


class TestValidTraces:
    """Known-good traces should pass with no errors."""

    def test_simple_trace(self) -> None:
        report = TraceValidator(TRACES / "simple_trace.csv").validate()
        assert report.is_valid, report.format()

    def test_message_trace(self) -> None:
        report = TraceValidator(TRACES / "message_trace.csv").validate()
        assert report.is_valid, report.format()

    def test_epsilon_trace(self) -> None:
        report = TraceValidator(TRACES / "epsilon_trace.csv").validate()
        assert report.is_valid, report.format()


# ---------------------------------------------------------------------------
# Tests: Duplicate Event IDs
# ---------------------------------------------------------------------------


class TestDuplicateEids:
    def test_duplicate_eid_detected(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
            "e1,P1,P1:2,1.0,,local,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Duplicate event ID" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Process Consistency
# ---------------------------------------------------------------------------


class TestProcessConsistency:
    def test_event_process_not_declared(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
            "e2,P2,P2:1,1.0,,local,\n",
        )
        report = TraceValidator(p).validate()
        assert any("not in declared" in m.message for m in report.errors)

    def test_declared_process_no_events(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1;P2:0,0.0,,local,\n",
        )
        report = TraceValidator(p).validate()
        assert any("has no events" in m.message for m in report.warnings)


# ---------------------------------------------------------------------------
# Tests: VC Key Consistency
# ---------------------------------------------------------------------------


class TestVCKeys:
    def test_vc_missing_process(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
            "e2,P2,P1:0;P2:1,0.0,,local,\n",
        )
        report = TraceValidator(p).validate()
        assert any("VC processes mismatch" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Initial Events
# ---------------------------------------------------------------------------


class TestInitialEvents:
    def test_missing_initial_event(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1;P2:0,0.0,,local,\n"
            "e2,P2,P1:0;P2:2,1.0,,local,\n",
        )
        report = TraceValidator(p).validate()
        assert any("no initial event" in m.message for m in report.errors)

    def test_multiple_initial_events(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
            "e2,P1,P1:1,0.5,,local,\n",
        )
        report = TraceValidator(p).validate()
        assert any("multiple initial" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Intra-Process Ordering
# ---------------------------------------------------------------------------


class TestIntraProcessOrdering:
    def test_vc_gap_detected(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
            "e2,P1,P1:3,1.0,,local,\n",  # gap: 1 -> 3
        )
        report = TraceValidator(p).validate()
        assert any("VC[P1] gap" in m.message for m in report.errors)

    def test_timestamp_decreased(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,1.0,,local,\n"
            "e2,P1,P1:2,0.5,,local,\n",  # timestamp decreased
        )
        report = TraceValidator(p).validate()
        assert any("Timestamp decreased" in m.message for m in report.errors)

    def test_same_timestamp_ok(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
            "e2,P1,P1:2,0.0,,local,\n",  # same timestamp is ok
        )
        report = TraceValidator(p).validate()
        assert not any("Timestamp" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: VC Correctness (local/send/receive rules)
# ---------------------------------------------------------------------------


class TestVCCorrectness:
    def test_local_event_changes_other_component(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "e1,P1,P1:2;P2:1,1.0,,local,\n",  # VC[P2] changed in local event
        )
        report = TraceValidator(p).validate()
        assert any("should stay the same" in m.message for m in report.errors)

    def test_send_event_changes_other_component(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "e1,P1,P1:2;P2:1,1.0,,send,P2\n",  # VC[P2] changed in send
        )
        report = TraceValidator(p).validate()
        assert any("should stay the same" in m.message for m in report.errors)

    def test_receive_decreases_component(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "s1,P1,P1:2;P2:0,1.0,,send,P2\n"
            # receive should merge, VC[P1] should be >= 0, but here it went to -1
            # Actually, let's make a scenario where a component decreases
            # P2 had VC[P1]=0 from i2, then after receiving from P1, VC[P1]
            # should be >= 2 (from sender). If it's 0, that's a decrease vs. expected merge.
            # But our check compares with previous event on same process.
            # Let's create: P2 has two events, second one has lower VC for P1
            "r1,P2,P1:2;P2:2,2.0,,receive,P1\n"
            "e2,P2,P1:1;P2:3,3.0,,receive,P1\n",  # VC[P1] went from 2 to 1 on receive
        )
        report = TraceValidator(p).validate()
        assert any("decreased" in m.message for m in report.errors)

    def test_valid_receive_merge(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "s1,P1,P1:2;P2:0,1.0,,send,P2\n"
            "r1,P2,P1:2;P2:2,2.0,,receive,P1\n",
        )
        report = TraceValidator(p).validate()
        assert report.is_valid, report.format()


# ---------------------------------------------------------------------------
# Tests: VC-Timestamp Consistency
# ---------------------------------------------------------------------------


class TestVCTimestampConsistency:
    def test_vc_before_but_timestamp_not(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "s1,P1,P1:2;P2:0,5.0,,send,P2\n"
            # r1 has VC > s1 (P1:2;P2:2 > P1:2;P2:0) but timestamp < s1
            "r1,P2,P1:2;P2:2,3.0,,receive,P1\n",
        )
        report = TraceValidator(p).validate()
        assert any("VC ordering" in m.message and "timestamp" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Message Events
# ---------------------------------------------------------------------------


class TestMessageEvents:
    def test_send_to_unknown_process(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "s1,P1,P1:2;P2:0,1.0,,send,P3\n",
        )
        report = TraceValidator(p).validate()
        assert any("not a known process" in m.message for m in report.errors)

    def test_send_to_self(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "s1,P1,P1:2;P2:0,1.0,,send,P1\n",
        )
        report = TraceValidator(p).validate()
        assert any("targets own process" in m.message for m in report.errors)

    def test_receive_from_self(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "r1,P1,P1:2;P2:0,1.0,,receive,P1\n",
        )
        report = TraceValidator(p).validate()
        assert any("own process" in m.message for m in report.errors)

    def test_send_missing_partner(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
            "e2,P1,P1:2,1.0,,send,\n",
        )
        report = TraceValidator(p).validate()
        assert any("missing msg_partner" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Message Pairing and FIFO
# ---------------------------------------------------------------------------


class TestMessagePairingAndFIFO:
    def test_unmatched_send_warning(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "s1,P1,P1:2;P2:0,1.0,,send,P2\n",
        )
        report = TraceValidator(p).validate()
        assert any("without matching receive" in m.message for m in report.warnings)

    def test_unmatched_receive_error(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1|P2\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "i1,P1,P1:1;P2:0,0.0,,local,\n"
            "i2,P2,P1:0;P2:1,0.0,,local,\n"
            "r1,P2,P1:1;P2:2,1.0,,receive,P1\n",
        )
        report = TraceValidator(p).validate()
        assert any("without matching send" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Proposition Naming
# ---------------------------------------------------------------------------


class TestPropositionNaming:
    def test_invalid_proposition_name_warned(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,123bad,local,\n",
        )
        report = TraceValidator(p).validate()
        assert any("naming convention" in m.message for m in report.warnings)

    def test_valid_proposition_names(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "# system_processes: P1\n"
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,ready|_done|state.x|p'|P_1,local,\n",
        )
        report = TraceValidator(p).validate()
        assert not any("naming convention" in m.message for m in report.warnings)


# ---------------------------------------------------------------------------
# Tests: Field-Level Parsing Errors
# ---------------------------------------------------------------------------


class TestFieldParsing:
    def test_invalid_timestamp(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc,timestamp,props\n" "e1,P1,P1:1,abc,,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Invalid timestamp" in m.message for m in report.errors)

    def test_negative_timestamp(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc,timestamp,props\n" "e1,P1,P1:1,-1.0,,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Negative timestamp" in m.message for m in report.errors)

    def test_malformed_vc(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc,timestamp,props\n" "e1,P1,bad_vc,0.0,,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Malformed vector clock" in m.message for m in report.errors)

    def test_negative_vc_component(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc,timestamp,props\n" "e1,P1,P1:-1,0.0,,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Negative VC component" in m.message for m in report.errors)

    def test_invalid_event_type(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,badtype,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Invalid event_type" in m.message for m in report.errors)

    def test_empty_eid(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc,timestamp,props\n" ",P1,P1:1,0.0,,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Empty event ID" in m.message for m in report.errors)

    def test_empty_process(self, tmp_path: Path) -> None:
        p = _write_trace(
            tmp_path,
            "eid,processes,vc,timestamp,props\n" "e1,,P1:1,0.0,,\n",
        )
        report = TraceValidator(p).validate()
        assert any("Empty process" in m.message for m in report.errors)


# ---------------------------------------------------------------------------
# Tests: Report Formatting
# ---------------------------------------------------------------------------


class TestReportFormatting:
    def test_valid_report_format(self) -> None:
        report = ValidationReport(filepath=Path("test.csv"))
        text = report.format()
        assert "No issues found" in text
        assert "0 error(s)" in text

    def test_error_report_format(self) -> None:
        report = ValidationReport(filepath=Path("test.csv"))
        report.messages.append(ValidationMessage(Severity.ERROR, "test error", row=2, eid="e1"))
        text = report.format()
        assert "[ERRORS]" in text
        assert "test error" in text
        assert "1 error(s)" in text

    def test_warning_report_format(self) -> None:
        report = ValidationReport(filepath=Path("test.csv"))
        report.messages.append(ValidationMessage(Severity.WARNING, "test warning"))
        text = report.format()
        assert "[WARNINGS]" in text
        assert "1 warning(s)" in text
