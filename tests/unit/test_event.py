"""
Tests for the Event dataclass.

Tests cover event creation, causal ordering via vector clocks, independence
checks (different processes), time-based ordering with epsilon, combined
ordering, and send/receive event validation.
"""

import pytest

from prove.core.event import Event
from prove.core.vector_clock import VectorClock


PROCS = frozenset({"P1", "P2", "P3"})


def _vc(p1: int = 0, p2: int = 0, p3: int = 0) -> VectorClock:
    """Shorthand for creating a VectorClock with P1/P2/P3 values."""
    return VectorClock(PROCS, initial_values={"P1": p1, "P2": p2, "P3": p3})


class TestEventCreation:
    """Test Event construction and attribute access."""

    def test_create_local_event(self) -> None:
        """Create a basic local event with all required fields."""
        e = Event(
            eid="e1",
            process="P1",
            vector_clock=_vc(1, 0, 0),
            timestamp=1.0,
            propositions=frozenset({"ready"}),
        )
        assert e.eid == "e1"
        assert e.process == "P1"
        assert e.timestamp == 1.0
        assert e.propositions == frozenset({"ready"})
        assert e.event_type == "local"

    def test_create_event_empty_propositions(self) -> None:
        """Event with no propositions defaults to empty frozenset."""
        e = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=0.0)
        assert e.propositions == frozenset()

    def test_event_is_frozen(self) -> None:
        """Event is immutable (frozen dataclass)."""
        e = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=0.0)
        with pytest.raises(AttributeError):
            e.eid = "e2"  # type: ignore[misc]

    def test_create_send_event(self) -> None:
        """Create a send event with target_process."""
        e = Event(
            eid="s1",
            process="P1",
            vector_clock=_vc(2, 0, 0),
            timestamp=1.0,
            event_type="send",
            target_process="P2",
        )
        assert e.is_send()
        assert e.target_process == "P2"

    def test_create_receive_event(self) -> None:
        """Create a receive event with source_process."""
        e = Event(
            eid="r1",
            process="P2",
            vector_clock=_vc(2, 2, 0),
            timestamp=2.0,
            event_type="receive",
            source_process="P1",
        )
        assert e.is_receive()
        assert e.source_process == "P1"


class TestEventValidation:
    """Test Event __post_init__ validation."""

    def test_send_requires_target(self) -> None:
        """Send event without target_process raises ValueError."""
        with pytest.raises(ValueError, match="target_process"):
            Event(
                eid="s1",
                process="P1",
                vector_clock=_vc(1, 0, 0),
                timestamp=1.0,
                event_type="send",
            )

    def test_receive_requires_source(self) -> None:
        """Receive event without source_process raises ValueError."""
        with pytest.raises(ValueError, match="source_process"):
            Event(
                eid="r1",
                process="P2",
                vector_clock=_vc(1, 1, 0),
                timestamp=1.0,
                event_type="receive",
            )

    def test_invalid_event_type_raises(self) -> None:
        """Invalid event_type raises ValueError."""
        with pytest.raises(ValueError, match="event_type"):
            Event(
                eid="x",
                process="P1",
                vector_clock=_vc(1, 0, 0),
                timestamp=1.0,
                event_type="invalid",
            )


class TestEventTypeChecks:
    """Test is_send, is_receive, is_local helper methods."""

    def test_is_local_default(self) -> None:
        """Default event type is local."""
        e = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=0.0)
        assert e.is_local()
        assert not e.is_send()
        assert not e.is_receive()

    def test_is_send(self) -> None:
        e = Event(
            eid="s1", process="P1", vector_clock=_vc(1, 0, 0),
            timestamp=0.0, event_type="send", target_process="P2",
        )
        assert e.is_send()
        assert not e.is_local()

    def test_is_receive(self) -> None:
        e = Event(
            eid="r1", process="P2", vector_clock=_vc(1, 1, 0),
            timestamp=0.0, event_type="receive", source_process="P1",
        )
        assert e.is_receive()
        assert not e.is_local()


class TestCausalOrdering:
    """Test causal ordering via vector clocks."""

    def test_causally_before(self) -> None:
        """e1 is causally before e2 when VC(e1) < VC(e2)."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P1", vector_clock=_vc(2, 0, 0), timestamp=2.0)
        assert e1.causally_before(e2)
        assert not e2.causally_before(e1)

    def test_not_causally_before_concurrent(self) -> None:
        """Concurrent events are not causally ordered."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=1.0)
        assert not e1.causally_before(e2)
        assert not e2.causally_before(e1)

    def test_causally_before_cross_process(self) -> None:
        """Causal ordering across processes (message causality)."""
        e_send = Event(eid="s1", process="P1", vector_clock=_vc(2, 0, 0), timestamp=1.0)
        e_recv = Event(eid="r1", process="P2", vector_clock=_vc(2, 1, 0), timestamp=2.0)
        assert e_send.causally_before(e_recv)

    def test_not_causally_before_self(self) -> None:
        """An event is not causally before itself (strict ordering)."""
        e = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        assert not e.causally_before(e)


class TestConcurrency:
    """Test concurrency detection via vector clocks."""

    def test_concurrent_events(self) -> None:
        """Events on different processes with incomparable VCs are concurrent."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(2, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 2, 0), timestamp=1.5)
        assert e1.is_concurrent_with(e2)
        assert e2.is_concurrent_with(e1)

    def test_not_concurrent_when_ordered(self) -> None:
        """Causally ordered events are not concurrent."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P1", vector_clock=_vc(2, 0, 0), timestamp=2.0)
        assert not e1.is_concurrent_with(e2)


class TestIndependence:
    """Test independence detection (different processes)."""

    def test_independent_different_processes(self) -> None:
        """Events on different processes are independent."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=1.0)
        assert e1.is_independent_of(e2)

    def test_not_independent_same_process(self) -> None:
        """Events on the same process are not independent."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P1", vector_clock=_vc(2, 0, 0), timestamp=2.0)
        assert not e1.is_independent_of(e2)


class TestTimeBasedOrdering:
    """Test epsilon-based time ordering."""

    def test_time_before_with_epsilon(self) -> None:
        """e1 is definitely before e2 when timestamp diff > epsilon."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=4.0)
        assert e1.time_before(e2, epsilon=2.0)

    def test_not_time_before_within_epsilon(self) -> None:
        """Events within epsilon are NOT definitely time-ordered."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=2.5)
        assert not e1.time_before(e2, epsilon=2.0)

    def test_time_before_exactly_epsilon(self) -> None:
        """At exactly epsilon boundary, events are NOT time-ordered (must be strictly >)."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=3.0)
        assert not e1.time_before(e2, epsilon=2.0)

    def test_time_before_zero_epsilon(self) -> None:
        """With epsilon=0, any positive time diff orders events."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=1.001)
        assert e1.time_before(e2, epsilon=0.0)

    def test_time_before_same_timestamp(self) -> None:
        """Events with the same timestamp are not time-ordered."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=1.0)
        assert not e1.time_before(e2, epsilon=0.0)


class TestCombinedOrdering:
    """Test combined ordering (vector clock OR time-based)."""

    def test_ordered_by_vc_only(self) -> None:
        """Events ordered by VC are ordered even with large epsilon."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P1", vector_clock=_vc(2, 0, 0), timestamp=2.0)
        assert e1.ordered_before(e2, epsilon=float("inf"))

    def test_ordered_by_time_only(self) -> None:
        """VC-concurrent events ordered by time with small epsilon."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=5.0)
        assert e1.ordered_before(e2, epsilon=2.0)

    def test_not_ordered_concurrent_within_epsilon(self) -> None:
        """VC-concurrent events within epsilon are not ordered."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P2", vector_clock=_vc(0, 1, 0), timestamp=2.0)
        assert not e1.ordered_before(e2, epsilon=5.0)
        assert not e2.ordered_before(e1, epsilon=5.0)


class TestEventEquality:
    """Test Event equality and hashing."""

    def test_equal_events(self) -> None:
        """Events with identical attributes are equal."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        assert e1 == e2

    def test_different_eid_not_equal(self) -> None:
        """Events with different eids are not equal."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e2", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        assert e1 != e2

    def test_hash_equal_events(self) -> None:
        """Equal events have equal hashes."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        assert hash(e1) == hash(e2)

    def test_usable_in_set(self) -> None:
        """Events can be stored in sets."""
        e1 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        e2 = Event(eid="e1", process="P1", vector_clock=_vc(1, 0, 0), timestamp=1.0)
        assert len({e1, e2}) == 1
