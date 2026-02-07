"""
Tests for the Cut class.

Tests cover creation, history-closed validation, event addition,
maximal event computation per process, frontier construction,
and containment checks.
"""

import pytest

from prove.core.cut import Cut
from prove.core.event import Event
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
        iota_P1(t=0) → a1(t=1) → a2(t=2)  on P1
        iota_P2(t=0) → b1(t=1) → b2(t=2)  on P2
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


class TestCutCreation:
    """Test Cut construction."""

    def test_create_with_initial_events(self) -> None:
        """Create a cut containing only initial events."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0], events[1]}, po)
        assert len(cut) == 2
        assert events[0] in cut
        assert events[1] in cut

    def test_create_with_history_closed_set(self) -> None:
        """Create a cut that is history-closed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        # {iota_P1, iota_P2, a1} is history-closed
        cut = Cut({events[0], events[1], events[2]}, po)
        assert len(cut) == 3
        assert cut.is_history_closed()

    def test_events_stored_as_frozenset(self) -> None:
        """Cut events are accessible."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0]}, po)
        assert events[0] in cut


class TestHistoryClosedValidation:
    """Test the is_history_closed property."""

    def test_initial_events_are_history_closed(self) -> None:
        """A cut of initial events (no predecessors) is trivially closed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0], events[1]}, po)
        assert cut.is_history_closed()

    def test_single_initial_event_is_closed(self) -> None:
        """A cut with a single initial event is history-closed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0]}, po)
        assert cut.is_history_closed()

    def test_missing_predecessor_not_closed(self) -> None:
        """A cut missing a predecessor is NOT history-closed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        # {a1} is NOT history-closed because iota_P1 ≺ a1 but iota_P1 ∉ cut
        cut = Cut({events[2]}, po)
        assert not cut.is_history_closed()

    def test_full_execution_is_closed(self) -> None:
        """A cut containing all events is history-closed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut(set(events), po)
        assert cut.is_history_closed()

    def test_partial_but_closed(self) -> None:
        """A partial cut that includes all predecessors is closed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        # {iota_P1, a1, iota_P2} is history-closed
        cut = Cut({events[0], events[2], events[1]}, po)
        assert cut.is_history_closed()


class TestAddEvent:
    """Test adding events to a cut."""

    def test_add_event_returns_new_cut(self) -> None:
        """Adding an event returns a new Cut (immutable pattern)."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut1 = Cut({events[0], events[1]}, po)
        cut2 = cut1.add_event(events[2])
        assert cut2 is not cut1
        assert events[2] in cut2
        assert events[2] not in cut1

    def test_add_preserves_existing_events(self) -> None:
        """New cut contains all old events plus the new one."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut1 = Cut({events[0], events[1]}, po)
        cut2 = cut1.add_event(events[2])
        assert len(cut2) == 3
        assert events[0] in cut2
        assert events[1] in cut2
        assert events[2] in cut2

    def test_add_already_contained_event(self) -> None:
        """Adding an already-contained event doesn't duplicate it."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0], events[1]}, po)
        cut2 = cut.add_event(events[0])
        assert len(cut2) == 2


class TestContains:
    """Test containment checks."""

    def test_contains_present_event(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0]}, po)
        assert cut.contains(events[0])
        assert events[0] in cut

    def test_not_contains_absent_event(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0]}, po)
        assert not cut.contains(events[1])
        assert events[1] not in cut


class TestMaximalEventForProcess:
    """Test finding the maximal event per process."""

    def test_single_event_per_process(self) -> None:
        """With one event per process, it is the maximal."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0], events[1]}, po)
        assert cut.maximal_event_for_process("P1") == events[0]
        assert cut.maximal_event_for_process("P2") == events[1]

    def test_multiple_events_per_process(self) -> None:
        """With multiple events on a process, the latest is maximal."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0], events[1], events[2], events[3]}, po)
        assert cut.maximal_event_for_process("P1") == events[2]  # a1
        assert cut.maximal_event_for_process("P2") == events[3]  # b1

    def test_process_not_in_cut(self) -> None:
        """Returns None for a process with no events in the cut."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0]}, po)
        assert cut.maximal_event_for_process("P2") is None


class TestFrontier:
    """Test frontier computation from a cut."""

    def test_frontier_initial_events(self) -> None:
        """Frontier of initial cut contains both initial events."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0], events[1]}, po)
        frontier = cut.frontier(PROCS2)
        assert events[0] in frontier.events
        assert events[1] in frontier.events
        assert len(frontier.events) == 2

    def test_frontier_larger_cut(self) -> None:
        """Frontier contains only the maximal event per process."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0], events[1], events[2], events[3]}, po)
        frontier = cut.frontier(PROCS2)
        # Maximal for P1 is a1, for P2 is b1
        assert frontier.process_to_event["P1"] == events[2]
        assert frontier.process_to_event["P2"] == events[3]

    def test_frontier_size_equals_processes(self) -> None:
        """Frontier has exactly one event per process."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut(set(events), po)
        frontier = cut.frontier(PROCS2)
        assert len(frontier.events) == 2


class TestCutThreeProcesses:
    """Test Cut with three processes."""

    def test_three_process_cut(self) -> None:
        """Cut and frontier work correctly with three processes."""
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
        cut = Cut({events[0], events[1], events[2], events[3]}, po)
        assert cut.is_history_closed()
        frontier = cut.frontier(PROCS3)
        assert frontier.process_to_event["P1"] == events[3]  # a1
        assert frontier.process_to_event["P2"] == events[1]  # i2
        assert frontier.process_to_event["P3"] == events[2]  # i3


class TestCutLen:
    """Test __len__."""

    def test_len(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        cut = Cut({events[0]}, po)
        assert len(cut) == 1
        cut2 = cut.add_event(events[1])
        assert len(cut2) == 2
