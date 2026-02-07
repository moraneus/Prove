"""
Tests for the Frontier class.

Tests cover frontier creation, global state computation, successor
computation, event enablement checking, equality, and hashing.
"""

import pytest

from prove.core.event import Event
from prove.core.frontier import Frontier
from prove.core.partial_order import PartialOrder
from prove.core.vector_clock import VectorClock

PROCS2 = frozenset({"P1", "P2"})
PROCS3 = frozenset({"P1", "P2", "P3"})


def _vc2(p1: int, p2: int) -> VectorClock:
    return VectorClock(PROCS2, initial_values={"P1": p1, "P2": p2})


def _vc3(p1: int, p2: int, p3: int) -> VectorClock:
    return VectorClock(PROCS3, initial_values={"P1": p1, "P2": p2, "P3": p3})


def _two_process_events() -> list[Event]:
    """
    Two-process execution with no messages:
        iota_P1(t=0) -> a1(t=1) -> a2(t=2)  on P1
        iota_P2(t=0) -> b1(t=1) -> b2(t=2)  on P2
    """
    return [
        Event(eid="iota_P1", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0,
              propositions=frozenset({"init_P1"})),
        Event(eid="iota_P2", process="P2", vector_clock=_vc2(0, 1), timestamp=0.0,
              propositions=frozenset({"init_P2"})),
        Event(eid="a1", process="P1", vector_clock=_vc2(2, 0), timestamp=1.0,
              propositions=frozenset({"ready"})),
        Event(eid="b1", process="P2", vector_clock=_vc2(0, 2), timestamp=1.0,
              propositions=frozenset({"waiting"})),
        Event(eid="a2", process="P1", vector_clock=_vc2(3, 0), timestamp=2.0,
              propositions=frozenset({"done"})),
        Event(eid="b2", process="P2", vector_clock=_vc2(0, 3), timestamp=2.0,
              propositions=frozenset({"done"})),
    ]


class TestFrontierCreation:
    """Test Frontier construction."""

    def test_from_mapping(self) -> None:
        """Create frontier from a process-to-event mapping."""
        events = _two_process_events()
        mapping = {"P1": events[0], "P2": events[1]}
        frontier = Frontier.from_mapping(mapping)
        assert len(frontier.events) == 2
        assert events[0] in frontier.events
        assert events[1] in frontier.events

    def test_process_to_event_accessible(self) -> None:
        """process_to_event mapping is accessible."""
        events = _two_process_events()
        mapping = {"P1": events[0], "P2": events[1]}
        frontier = Frontier.from_mapping(mapping)
        assert frontier.process_to_event["P1"] == events[0]
        assert frontier.process_to_event["P2"] == events[1]

    def test_process_to_event_is_immutable(self) -> None:
        """process_to_event mapping cannot be mutated."""
        events = _two_process_events()
        mapping = {"P1": events[0], "P2": events[1]}
        frontier = Frontier.from_mapping(mapping)
        with pytest.raises(TypeError):
            frontier.process_to_event["P1"] = events[2]

    def test_single_process_frontier(self) -> None:
        """Frontier with a single process."""
        events = _two_process_events()
        frontier = Frontier.from_mapping({"P1": events[0]})
        assert len(frontier.events) == 1
        assert frontier.process_to_event["P1"] == events[0]


class TestGlobalState:
    """Test global state (proposition) computation."""

    def test_initial_global_state(self) -> None:
        """Global state from initial events is union of their propositions."""
        events = _two_process_events()
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        state = frontier.global_state()
        assert "init_P1" in state
        assert "init_P2" in state
        assert len(state) == 2

    def test_global_state_union(self) -> None:
        """Global state is the union of propositions from all frontier events."""
        events = _two_process_events()
        frontier = Frontier.from_mapping({"P1": events[2], "P2": events[3]})
        state = frontier.global_state()
        assert "ready" in state
        assert "waiting" in state
        assert len(state) == 2

    def test_global_state_duplicate_propositions(self) -> None:
        """Duplicate propositions across processes are merged."""
        events = _two_process_events()
        # a2 and b2 both have "done"
        frontier = Frontier.from_mapping({"P1": events[4], "P2": events[5]})
        state = frontier.global_state()
        assert "done" in state
        assert len(state) == 1  # Only "done" (deduplicated)

    def test_global_state_empty_propositions(self) -> None:
        """Events with no propositions contribute nothing."""
        ev = Event(eid="x", process="P1", vector_clock=_vc2(1, 0),
                   timestamp=0.0, propositions=frozenset())
        frontier = Frontier.from_mapping({"P1": ev})
        state = frontier.global_state()
        assert len(state) == 0


class TestFrontierSuccessor:
    """Test successor frontier computation."""

    def test_successor_replaces_process_event(self) -> None:
        """Successor replaces the maximal event for the event's process."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        # Add a1 (P1's next event)
        new_frontier = frontier.successor(events[2], po)
        assert new_frontier.process_to_event["P1"] == events[2]
        # P2 unchanged
        assert new_frontier.process_to_event["P2"] == events[1]

    def test_successor_is_new_instance(self) -> None:
        """Successor creates a new Frontier, original unchanged."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        new_frontier = frontier.successor(events[2], po)
        assert new_frontier is not frontier
        assert frontier.process_to_event["P1"] == events[0]

    def test_successor_preserves_other_processes(self) -> None:
        """Successor only changes the event's process entry."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        new_frontier = frontier.successor(events[3], po)  # b1 on P2
        assert new_frontier.process_to_event["P1"] == events[0]
        assert new_frontier.process_to_event["P2"] == events[3]


class TestEventEnablement:
    """Test event enablement from a frontier."""

    def test_initial_successors_enabled(self) -> None:
        """Immediate successors of initial events are enabled."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        assert frontier.is_event_enabled(events[2], po)  # a1
        assert frontier.is_event_enabled(events[3], po)  # b1

    def test_non_successor_not_enabled(self) -> None:
        """Events whose predecessors aren't all in the frontier are not enabled."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        # a2 requires a1 which isn't in the frontier
        assert not frontier.is_event_enabled(events[4], po)

    def test_enabled_after_advancement(self) -> None:
        """After advancing the frontier, new events become enabled."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping({"P1": events[2], "P2": events[1]})
        # a2 should now be enabled (predecessor a1 is in the frontier)
        assert frontier.is_event_enabled(events[4], po)


class TestEnabledEvents:
    """Test the enabled_events method."""

    def test_returns_all_enabled(self) -> None:
        """Returns all events that are enabled from the frontier."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        pending = [events[2], events[3], events[4], events[5]]
        enabled = frontier.enabled_events(pending, po)
        assert events[2] in enabled  # a1
        assert events[3] in enabled  # b1
        assert events[4] not in enabled  # a2 (needs a1)
        assert events[5] not in enabled  # b2 (needs b1)


class TestFrontierEquality:
    """Test equality and hashing."""

    def test_equal_frontiers(self) -> None:
        """Frontiers with same events are equal."""
        events = _two_process_events()
        f1 = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        f2 = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        assert f1 == f2

    def test_different_frontiers_not_equal(self) -> None:
        """Frontiers with different events are not equal."""
        events = _two_process_events()
        f1 = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        f2 = Frontier.from_mapping({"P1": events[2], "P2": events[1]})
        assert f1 != f2

    def test_hash_consistency(self) -> None:
        """Equal frontiers have equal hashes."""
        events = _two_process_events()
        f1 = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        f2 = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        assert hash(f1) == hash(f2)

    def test_usable_in_set(self) -> None:
        """Frontiers can be used in sets."""
        events = _two_process_events()
        f1 = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        f2 = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        f3 = Frontier.from_mapping({"P1": events[2], "P2": events[1]})
        s = {f1, f2, f3}
        assert len(s) == 2  # f1 and f2 are equal


class TestFrontierThreeProcesses:
    """Test Frontier with three processes."""

    def test_three_process_frontier(self) -> None:
        """Frontier works with three processes."""
        events = [
            Event(eid="i1", process="P1", vector_clock=_vc3(1, 0, 0), timestamp=0.0,
                  propositions=frozenset({"init1"})),
            Event(eid="i2", process="P2", vector_clock=_vc3(0, 1, 0), timestamp=0.0,
                  propositions=frozenset({"init2"})),
            Event(eid="i3", process="P3", vector_clock=_vc3(0, 0, 1), timestamp=0.0,
                  propositions=frozenset({"init3"})),
        ]
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1], "P3": events[2]})
        assert len(frontier.events) == 3
        state = frontier.global_state()
        assert "init1" in state
        assert "init2" in state
        assert "init3" in state

    def test_three_process_successor(self) -> None:
        """Successor on three-process frontier updates correct process."""
        events = [
            Event(eid="i1", process="P1", vector_clock=_vc3(1, 0, 0), timestamp=0.0,
                  propositions=frozenset({"init1"})),
            Event(eid="i2", process="P2", vector_clock=_vc3(0, 1, 0), timestamp=0.0,
                  propositions=frozenset({"init2"})),
            Event(eid="i3", process="P3", vector_clock=_vc3(0, 0, 1), timestamp=0.0,
                  propositions=frozenset({"init3"})),
            Event(eid="a1", process="P1", vector_clock=_vc3(2, 0, 0), timestamp=1.0,
                  propositions=frozenset({"ready"})),
        ]
        po = PartialOrder(events, epsilon=float("inf"))
        frontier = Frontier.from_mapping(
            {"P1": events[0], "P2": events[1], "P3": events[2]}
        )
        new_frontier = frontier.successor(events[3], po)
        assert new_frontier.process_to_event["P1"] == events[3]
        assert new_frontier.process_to_event["P2"] == events[1]
        assert new_frontier.process_to_event["P3"] == events[2]


class TestFrontierRepr:
    """Test string representation."""

    def test_repr(self) -> None:
        events = _two_process_events()
        frontier = Frontier.from_mapping({"P1": events[0], "P2": events[1]})
        r = repr(frontier)
        assert "Frontier" in r
        assert "iota_P1" in r
        assert "iota_P2" in r
