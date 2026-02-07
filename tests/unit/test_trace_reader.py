"""
Tests for the CSV trace file parser.

Tests cover parsing simple traces, directive extraction, field parsing,
validation, error handling, message events, and complete TraceData loading.
"""

from pathlib import Path

import pytest

from prove.core.event import Event
from prove.core.vector_clock import VectorClock
from prove.utils.trace_reader import TraceData, TraceMetadata, TraceReader


FIXTURES = Path(__file__).parent.parent / "fixtures"
TRACES = FIXTURES / "traces"


# ---------------------------------------------------------------------------
# Tests: Static Helpers
# ---------------------------------------------------------------------------


class TestParseHelpers:
    """Test static parsing helper methods."""

    def test_parse_vector_clock(self) -> None:
        """Parse semicolon-separated vector clock string."""
        procs = frozenset({"P1", "P2", "P3"})
        vc = TraceReader.parse_vector_clock("P1:2;P2:1;P3:0", procs)
        assert vc.clock["P1"] == 2
        assert vc.clock["P2"] == 1
        assert vc.clock["P3"] == 0

    def test_parse_vector_clock_single_process(self) -> None:
        """Parse VC with only one process."""
        procs = frozenset({"P1"})
        vc = TraceReader.parse_vector_clock("P1:5", procs)
        assert vc.clock["P1"] == 5

    def test_parse_propositions(self) -> None:
        """Parse pipe-separated propositions."""
        props = TraceReader.parse_propositions("ready|done|active")
        assert props == frozenset({"ready", "done", "active"})

    def test_parse_propositions_empty(self) -> None:
        """Empty proposition string returns empty frozenset."""
        props = TraceReader.parse_propositions("")
        assert props == frozenset()

    def test_parse_propositions_single(self) -> None:
        """Single proposition without pipe."""
        props = TraceReader.parse_propositions("ready")
        assert props == frozenset({"ready"})

    def test_parse_timestamp(self) -> None:
        """Parse timestamp string to float."""
        assert TraceReader.parse_timestamp("1.5") == 1.5
        assert TraceReader.parse_timestamp("0.0") == 0.0
        assert TraceReader.parse_timestamp("100") == 100.0


# ---------------------------------------------------------------------------
# Tests: Reading Simple Traces
# ---------------------------------------------------------------------------


class TestReadSimpleTrace:
    """Test reading a basic trace file."""

    def test_read_events_count(self) -> None:
        """Simple trace has the expected number of events."""
        reader = TraceReader(TRACES / "simple_trace.csv")
        events = reader.read_events()
        assert len(events) == 5

    def test_read_events_types(self) -> None:
        """All events in simple trace are Event objects."""
        reader = TraceReader(TRACES / "simple_trace.csv")
        events = reader.read_events()
        for e in events:
            assert isinstance(e, Event)

    def test_read_event_fields(self) -> None:
        """Events have correct field values."""
        reader = TraceReader(TRACES / "simple_trace.csv")
        events = reader.read_events()
        iota = events[0]
        assert iota.eid == "iota_P1"
        assert iota.process == "P1"
        assert iota.timestamp == 0.0
        assert "init_P1" in iota.propositions

    def test_read_metadata(self) -> None:
        """Metadata extracted from directives."""
        reader = TraceReader(TRACES / "simple_trace.csv")
        meta = reader.read_metadata()
        assert meta.processes == frozenset({"P1", "P2"})
        assert meta.epsilon == 2.0

    def test_read_all_returns_trace_data(self) -> None:
        """read_all returns a TraceData with events, metadata, and partial order."""
        reader = TraceReader(TRACES / "simple_trace.csv")
        data = reader.read_all()
        assert isinstance(data, TraceData)
        assert len(data.events) == 5
        assert data.metadata.processes == frozenset({"P1", "P2"})
        assert data.partial_order is not None

    def test_read_all_with_epsilon_override(self) -> None:
        """Epsilon parameter overrides file directive."""
        reader = TraceReader(TRACES / "simple_trace.csv")
        data = reader.read_all(epsilon=5.0)
        assert data.metadata.epsilon == 5.0


# ---------------------------------------------------------------------------
# Tests: Message Trace
# ---------------------------------------------------------------------------


class TestMessageTrace:
    """Test reading traces with send/receive events."""

    def test_message_events_parsed(self) -> None:
        """Send and receive events have correct event_type."""
        reader = TraceReader(TRACES / "message_trace.csv")
        events = reader.read_events()
        by_eid = {e.eid: e for e in events}
        assert by_eid["c_send"].event_type == "send"
        assert by_eid["c_send"].target_process == "Server"
        assert by_eid["s_recv"].event_type == "receive"
        assert by_eid["s_recv"].source_process == "Client"

    def test_message_trace_event_count(self) -> None:
        """Message trace has expected event count."""
        reader = TraceReader(TRACES / "message_trace.csv")
        events = reader.read_events()
        assert len(events) == 7

    def test_local_events_default_type(self) -> None:
        """Events without event_type default to local."""
        reader = TraceReader(TRACES / "message_trace.csv")
        events = reader.read_events()
        by_eid = {e.eid: e for e in events}
        assert by_eid["s_process"].event_type == "local"


# ---------------------------------------------------------------------------
# Tests: Epsilon Trace
# ---------------------------------------------------------------------------


class TestEpsilonTrace:
    """Test reading traces with epsilon directive."""

    def test_epsilon_directive_parsed(self) -> None:
        """Epsilon value is extracted from directive."""
        reader = TraceReader(TRACES / "epsilon_trace.csv")
        meta = reader.read_metadata()
        assert meta.epsilon == 2.0

    def test_partial_order_includes_epsilon(self) -> None:
        """Partial order computed with epsilon from directive."""
        reader = TraceReader(TRACES / "epsilon_trace.csv")
        data = reader.read_all()
        # With epsilon=2: t(beta2) - t(alpha1) = 5.5 - 2.5 = 3.0 > 2
        # So alpha1 ≺ beta2
        by_eid = {e.eid: e for e in data.events}
        assert data.partial_order.is_before(
            by_eid["alpha1"], by_eid["beta2"]
        )


# ---------------------------------------------------------------------------
# Tests: Inferred Metadata
# ---------------------------------------------------------------------------


class TestInferredMetadata:
    """Test metadata inference when directives are absent."""

    def test_infer_processes_from_events(self, tmp_path: Path) -> None:
        """Processes inferred from event data when no directive."""
        trace = tmp_path / "trace.csv"
        trace.write_text(
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1;P2:0,0.0,ready,local,\n"
            "e2,P2,P1:0;P2:1,0.0,,local,\n"
        )
        reader = TraceReader(trace)
        meta = reader.read_metadata()
        assert meta.processes == frozenset({"P1", "P2"})

    def test_no_epsilon_directive_defaults_none(self, tmp_path: Path) -> None:
        """Epsilon is None when no directive provided."""
        trace = tmp_path / "trace.csv"
        trace.write_text(
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,ready,local,\n"
        )
        reader = TraceReader(trace)
        meta = reader.read_metadata()
        assert meta.epsilon is None


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Test trace file validation."""

    def test_valid_trace_no_errors(self) -> None:
        """Valid trace produces no errors."""
        reader = TraceReader(TRACES / "simple_trace.csv")
        errors = reader.validate()
        assert errors == []

    def test_missing_timestamp_header(self) -> None:
        """Missing timestamp column is detected."""
        reader = TraceReader(TRACES / "invalid_trace.csv")
        errors = reader.validate()
        assert any("timestamp" in e.lower() for e in errors)

    def test_malformed_csv(self, tmp_path: Path) -> None:
        """Malformed CSV content is detected."""
        trace = tmp_path / "bad.csv"
        trace.write_text("this is not csv\n")
        reader = TraceReader(trace)
        errors = reader.validate()
        assert len(errors) > 0

    def test_missing_file(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        reader = TraceReader(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            reader.read_events()


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_propositions_field(self, tmp_path: Path) -> None:
        """Empty props field results in empty frozenset."""
        trace = tmp_path / "trace.csv"
        trace.write_text(
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            "e1,P1,P1:1,0.0,,local,\n"
        )
        reader = TraceReader(trace)
        events = reader.read_events()
        assert events[0].propositions == frozenset()

    def test_whitespace_in_fields(self, tmp_path: Path) -> None:
        """Whitespace around fields is stripped."""
        trace = tmp_path / "trace.csv"
        trace.write_text(
            "eid,processes,vc,timestamp,props,event_type,msg_partner\n"
            " e1 , P1 , P1:1 , 0.0 , ready , local ,\n"
        )
        reader = TraceReader(trace)
        events = reader.read_events()
        assert events[0].eid == "e1"
        assert events[0].process == "P1"
        assert "ready" in events[0].propositions
