"""
Tests for the EPLTL monitor orchestration.

Tests cover initialization, sequential processing, concurrent events,
epsilon ordering, verdict computation, reset, and file-based creation.
"""

from pathlib import Path

import pytest

from prove.core.event import Event
from prove.core.monitor import EPLTLMonitor, MonitorResult
from prove.core.vector_clock import VectorClock
from prove.parser.formula import parse_formula
from prove.utils.logger import LogLevel, MonitorLogger

FIXTURES = Path(__file__).parent.parent / "fixtures"
TRACES = FIXTURES / "traces"
PROPERTIES = FIXTURES / "properties"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    eid: str,
    process: str,
    vc_vals: dict[str, int],
    timestamp: float,
    props: frozenset[str] = frozenset(),
    event_type: str = "local",
    source_process: str | None = None,
    target_process: str | None = None,
) -> Event:
    """Create an event with full VC over two processes P1, P2."""
    procs = frozenset(vc_vals.keys())
    vc = VectorClock(procs, initial_values=vc_vals)
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


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestMonitorInitialization:
    """Test EPLTLMonitor creation and setup."""

    def test_create_monitor(self) -> None:
        """Monitor can be created with formula and processes."""
        formula = parse_formula("ready")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=["P1", "P2"],
            epsilon=float("inf"),
        )
        assert monitor is not None

    def test_create_with_logger(self) -> None:
        """Monitor accepts an optional logger."""
        from io import StringIO

        buf = StringIO()
        logger = MonitorLogger(level=LogLevel.DEBUG, stream=buf)
        formula = parse_formula("ready")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=["P1", "P2"],
            logger=logger,
        )
        assert monitor is not None

    def test_from_files(self) -> None:
        """Monitor can be created from property and trace files."""
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "simple.prop",
            trace_file=TRACES / "simple_trace.csv",
        )
        assert monitor is not None


# ---------------------------------------------------------------------------
# Tests: Sequential Processing
# ---------------------------------------------------------------------------


class TestSequentialProcessing:
    """Test monitoring a simple sequential trace."""

    def test_simple_satisfied(self) -> None:
        """Simple proposition is satisfied when it holds at the end."""
        formula = parse_formula("ready")
        events = [
            _make_event("i1", "P1", {"P1": 1}, 0.0, frozenset({"ready"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is True

    def test_simple_violated(self) -> None:
        """Simple proposition is violated when it does not hold."""
        formula = parse_formula("ready")
        events = [
            _make_event("i1", "P1", {"P1": 1}, 0.0, frozenset({"other"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is False

    def test_sequence_of_events(self) -> None:
        """Monitor processes a sequence of events on one process."""
        formula = parse_formula("done")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"init"})),
            _make_event("e2", "P1", {"P1": 2}, 1.0, frozenset({"working"})),
            _make_event("e3", "P1", {"P1": 3}, 2.0, frozenset({"done"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is True


# ---------------------------------------------------------------------------
# Tests: Two-Process Concurrent Events
# ---------------------------------------------------------------------------


class TestConcurrentEvents:
    """Test monitoring with concurrent events from multiple processes."""

    def test_two_process_both_initial(self) -> None:
        """Two-process trace with concurrent initial events."""
        formula = parse_formula("a & b")
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"a"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"b"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1", "P2"], epsilon=float("inf"))
        result = monitor.run(events)
        # After processing both initial events, the frontier has both
        # propositions visible in the global state.
        assert result.satisfied is True

    def test_concurrent_events_branching(self) -> None:
        """Concurrent events create branching in the graph."""
        formula = parse_formula("a")
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"a"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"b"})),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"a"})),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 1.0, frozenset({"b"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1", "P2"], epsilon=float("inf"))
        result = monitor.run(events)
        assert isinstance(result, MonitorResult)


# ---------------------------------------------------------------------------
# Tests: Since Operator
# ---------------------------------------------------------------------------


class TestSinceOperator:
    """Test the Since temporal operator in monitoring."""

    def test_since_satisfied(self) -> None:
        """Since operator satisfied: q was true, p held since then."""
        formula = parse_formula("p S q")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"q"})),
            _make_event("e2", "P1", {"P1": 2}, 1.0, frozenset({"p"})),
            _make_event("e3", "P1", {"P1": 3}, 2.0, frozenset({"p"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is True

    def test_since_violated(self) -> None:
        """Since violated: q never held."""
        formula = parse_formula("p S q")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"p"})),
            _make_event("e2", "P1", {"P1": 2}, 1.0, frozenset({"p"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is False


# ---------------------------------------------------------------------------
# Tests: Yesterday Operator
# ---------------------------------------------------------------------------


class TestYesterdayOperator:
    """Test the Yesterday temporal operator."""

    def test_yesterday_true(self) -> None:
        """Yesterday is true when proposition held in previous state."""
        formula = parse_formula("@ready")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"ready"})),
            _make_event("e2", "P1", {"P1": 2}, 1.0, frozenset()),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is True

    def test_yesterday_false_first_state(self) -> None:
        """Yesterday is false at the first state (no predecessor)."""
        formula = parse_formula("@ready")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"ready"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is False


# ---------------------------------------------------------------------------
# Tests: Epsilon Ordering
# ---------------------------------------------------------------------------


class TestEpsilonOrdering:
    """Test monitoring with epsilon-based ordering."""

    def test_epsilon_orders_events(self) -> None:
        """Events ordered by epsilon create fewer linearizations."""
        formula = parse_formula("a")
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"a"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"b"})),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"a"})),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 5.0, frozenset({"b"})),
        ]
        # With epsilon=2: t(e2)-t(e1) = 4.0 > 2, so e1 ≺ e2
        monitor = EPLTLMonitor(formula=formula, processes=["P1", "P2"], epsilon=2.0)
        result = monitor.run(events)
        assert isinstance(result, MonitorResult)


# ---------------------------------------------------------------------------
# Tests: MonitorResult
# ---------------------------------------------------------------------------


class TestMonitorResult:
    """Test MonitorResult data structure."""

    def test_result_has_fields(self) -> None:
        """MonitorResult has required fields."""
        result = MonitorResult(
            satisfied=True,
            verdict="SATISFIED",
            witness=None,
            statistics={"events_processed": 5},
        )
        assert result.satisfied is True
        assert result.verdict == "SATISFIED"
        assert result.witness is None
        assert result.statistics["events_processed"] == 5


# ---------------------------------------------------------------------------
# Tests: Run from Files
# ---------------------------------------------------------------------------


class TestRunFromFiles:
    """Test end-to-end monitoring from fixture files."""

    def test_simple_trace_with_simple_property(self) -> None:
        """Run monitor on simple trace with simple property."""
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "simple.prop",
            trace_file=TRACES / "simple_trace.csv",
        )
        result = monitor.run_from_trace()
        assert isinstance(result, MonitorResult)
        assert isinstance(result.satisfied, bool)

    def test_simple_trace_with_since_property(self) -> None:
        """Run monitor on simple trace with since property."""
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "since.prop",
            trace_file=TRACES / "simple_trace.csv",
        )
        result = monitor.run_from_trace()
        assert isinstance(result, MonitorResult)

    def test_message_trace(self) -> None:
        """Run monitor on message trace with send/receive."""
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "simple.prop",
            trace_file=TRACES / "message_trace.csv",
        )
        result = monitor.run_from_trace()
        assert isinstance(result, MonitorResult)

    def test_epsilon_trace(self) -> None:
        """Run monitor on epsilon trace with timing constraints."""
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "simple.prop",
            trace_file=TRACES / "epsilon_trace.csv",
        )
        result = monitor.run_from_trace()
        assert isinstance(result, MonitorResult)


# ---------------------------------------------------------------------------
# Tests: Finalize and Statistics
# ---------------------------------------------------------------------------


class TestFinalizeAndStatistics:
    """Test finalization and statistics reporting."""

    def test_statistics_in_result(self) -> None:
        """Result contains monitoring statistics."""
        formula = parse_formula("ready")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"ready"})),
            _make_event("e2", "P1", {"P1": 2}, 1.0, frozenset({"done"})),
            _make_event("e3", "P1", {"P1": 3}, 2.0, frozenset({"ready"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert "events_processed" in result.statistics
        # Initial event is loaded into graph directly, not counted by process_event
        assert result.statistics["events_processed"] == 2

    def test_reset(self) -> None:
        """Monitor can be reset and reused."""
        formula = parse_formula("ready")
        events1 = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"ready"})),
        ]
        events2 = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"other"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))

        r1 = monitor.run(events1)
        assert r1.satisfied is True

        monitor.reset()

        r2 = monitor.run(events2)
        assert r2.satisfied is False


# ---------------------------------------------------------------------------
# Tests: Complex Formulas
# ---------------------------------------------------------------------------


class TestComplexFormulas:
    """Test monitoring with complex formulas."""

    def test_implication_satisfied(self) -> None:
        """Implication satisfied: antecedent false makes it trivially true."""
        formula = parse_formula("error -> @warning")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"ok"})),
            _make_event("e2", "P1", {"P1": 2}, 1.0, frozenset({"ok"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is True

    def test_negation(self) -> None:
        """Negated proposition works correctly."""
        formula = parse_formula("!error")
        events = [
            _make_event("e1", "P1", {"P1": 1}, 0.0, frozenset({"ok"})),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"], epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is True


# ---------------------------------------------------------------------------
# Tests: Strict Initial Event Identification
# ---------------------------------------------------------------------------


class TestIdentifyInitialEvents:
    """Test _identify_initial_events strict VC-based identification."""

    def test_identifies_correct_initial_events(self) -> None:
        """Correct initial events are identified by VC structure."""
        formula = parse_formula("a")
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"a"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"b"})),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1", "P2"])
        initial = monitor._identify_initial_events(events)
        assert initial["P1"].eid == "i1"
        assert initial["P2"].eid == "i2"

    def test_raises_on_missing_initial(self) -> None:
        """Raises ValueError when a process has no initial event."""
        formula = parse_formula("a")
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0),
            # P2 has VC[P2]=2, not a valid initial event
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 1.0),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1", "P2"])
        with pytest.raises(ValueError, match="No initial event"):
            monitor._identify_initial_events(events)

    def test_raises_on_duplicate_initial(self) -> None:
        """Raises ValueError when a process has multiple initial candidates."""
        formula = parse_formula("a")
        events = [
            _make_event("i1a", "P1", {"P1": 1, "P2": 0}, 0.0),
            _make_event("i1b", "P1", {"P1": 1, "P2": 0}, 0.1),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1", "P2"])
        with pytest.raises(ValueError, match="multiple initial"):
            monitor._identify_initial_events(events)

    def test_no_fallback_to_first_event(self) -> None:
        """The lenient fallback is gone — invalid VC causes an error."""
        formula = parse_formula("a")
        # Both events have VC[P1]=2, neither is a valid initial
        events = [
            _make_event("e1", "P1", {"P1": 2}, 0.0),
            _make_event("e2", "P1", {"P1": 3}, 1.0),
        ]
        monitor = EPLTLMonitor(formula=formula, processes=["P1"])
        with pytest.raises(ValueError, match="No initial event"):
            monitor._identify_initial_events(events)
