"""
Tests for the PartialOrder class.

Tests cover complete partial order computation combining vector clock
ordering and epsilon-based timestamp ordering, transitive closure,
concurrent/independent event detection, predecessor/successor queries,
topological sorting, and enabled event computation.
"""

import pytest

from prove.core.event import Event
from prove.core.partial_order import PartialOrder
from prove.core.vector_clock import VectorClock

PROCS2 = frozenset({"P1", "P2"})
PROCS3 = frozenset({"P1", "P2", "P3"})


def _vc2(p1: int, p2: int) -> VectorClock:
    return VectorClock(PROCS2, initial_values={"P1": p1, "P2": p2})


def _vc3(p1: int, p2: int, p3: int) -> VectorClock:
    return VectorClock(PROCS3, initial_values={"P1": p1, "P2": p2, "P3": p3})


# -------------------------------------------------------------------- #
# Reusable event sets
# -------------------------------------------------------------------- #

def _two_process_events() -> list[Event]:
    """
    Two-process linear execution (P1 and P2, no messages):
        iota_P1(t=0) → a1(t=1) → a2(t=2)  on P1
        iota_P2(t=0) → b1(t=1) → b2(t=2)  on P2
    Vector clocks are independent (no message exchange).
    """
    return [
        Event(eid="iota_P1", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0,
              propositions=frozenset({"init_P1"})),
        Event(eid="iota_P2", process="P2", vector_clock=_vc2(0, 1), timestamp=0.0,
              propositions=frozenset({"init_P2"})),
        Event(eid="a1", process="P1", vector_clock=_vc2(2, 0), timestamp=1.0),
        Event(eid="b1", process="P2", vector_clock=_vc2(0, 2), timestamp=1.0),
        Event(eid="a2", process="P1", vector_clock=_vc2(3, 0), timestamp=2.0),
        Event(eid="b2", process="P2", vector_clock=_vc2(0, 3), timestamp=2.0),
    ]


def _message_events() -> list[Event]:
    """
    Two processes with a message from P1 to P2:
        iota_P1(t=0) → send(t=1)          on P1
        iota_P2(t=0) → recv(t=2) → b2(t=3) on P2
    The send→recv creates causal ordering: send ≺ recv.
    """
    return [
        Event(eid="iota_P1", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0),
        Event(eid="iota_P2", process="P2", vector_clock=_vc2(0, 1), timestamp=0.0),
        Event(eid="send", process="P1", vector_clock=_vc2(2, 0), timestamp=1.0,
              event_type="send", target_process="P2"),
        Event(eid="recv", process="P2", vector_clock=_vc2(2, 2), timestamp=2.0,
              event_type="receive", source_process="P1"),
        Event(eid="b2", process="P2", vector_clock=_vc2(2, 3), timestamp=3.0),
    ]


class TestPartialOrderVCOnly:
    """Test partial order from vector clocks only (epsilon=inf)."""

    def test_same_process_ordered(self) -> None:
        """Events on the same process are totally ordered by VC."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        iota_p1, _, a1, _, a2, _ = events
        assert po.is_before(iota_p1, a1)
        assert po.is_before(a1, a2)
        assert po.is_before(iota_p1, a2)  # transitive

    def test_cross_process_concurrent(self) -> None:
        """Events on different processes without messages are concurrent."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        _, _, a1, b1, _, _ = events
        assert po.are_concurrent(a1, b1)

    def test_message_creates_ordering(self) -> None:
        """A message send→recv creates causal ordering."""
        events = _message_events()
        po = PartialOrder(events, epsilon=float("inf"))
        send = events[2]
        recv = events[3]
        assert po.is_before(send, recv)

    def test_transitive_through_message(self) -> None:
        """Ordering is transitive through messages: iota_P1 ≺ recv."""
        events = _message_events()
        po = PartialOrder(events, epsilon=float("inf"))
        iota_p1 = events[0]
        recv = events[3]
        assert po.is_before(iota_p1, recv)


class TestPartialOrderEpsilon:
    """Test epsilon-based ordering additions."""

    def test_epsilon_adds_ordering(self) -> None:
        """Events with time diff > epsilon become ordered."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=0.5)
        iota_p1, _, a1, _, _, _ = events
        # a1 is at t=1.0, iota_P2 is at t=0.0, diff=1.0 > 0.5
        iota_p2 = events[1]
        assert po.is_before(iota_p2, a1)

    def test_zero_epsilon_orders_everything(self) -> None:
        """With epsilon=0, any time difference creates ordering."""
        events = [
            Event(eid="e1", process="P1", vector_clock=_vc2(1, 0), timestamp=1.0),
            Event(eid="e2", process="P2", vector_clock=_vc2(0, 1), timestamp=2.0),
        ]
        po = PartialOrder(events, epsilon=0.0)
        assert po.is_before(events[0], events[1])

    def test_same_timestamp_not_ordered_by_epsilon(self) -> None:
        """Events with same timestamp are not epsilon-ordered."""
        events = [
            Event(eid="e1", process="P1", vector_clock=_vc2(1, 0), timestamp=1.0),
            Event(eid="e2", process="P2", vector_clock=_vc2(0, 1), timestamp=1.0),
        ]
        po = PartialOrder(events, epsilon=0.0)
        assert po.are_concurrent(events[0], events[1])

    def test_large_epsilon_no_extra_ordering(self) -> None:
        """With large epsilon, no epsilon-based ordering is added."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=1_000_000.0)
        a1 = events[2]
        b1 = events[3]
        assert po.are_concurrent(a1, b1)


class TestTransitiveClosure:
    """Test transitive closure computation."""

    def test_chain_of_three(self) -> None:
        """a ≺ b and b ≺ c implies a ≺ c."""
        events = [
            Event(eid="a", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0),
            Event(eid="b", process="P1", vector_clock=_vc2(2, 0), timestamp=1.0),
            Event(eid="c", process="P1", vector_clock=_vc2(3, 0), timestamp=2.0),
        ]
        po = PartialOrder(events, epsilon=float("inf"))
        assert po.is_before(events[0], events[2])

    def test_epsilon_transitivity(self) -> None:
        """
        a ≺ b (by VC) and b ≺ c (by epsilon) implies a ≺ c (transitive).
        """
        events = [
            Event(eid="a", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0),
            Event(eid="b", process="P1", vector_clock=_vc2(2, 0), timestamp=1.0),
            Event(eid="c", process="P2", vector_clock=_vc2(0, 1), timestamp=3.0),
        ]
        po = PartialOrder(events, epsilon=1.0)
        # b(t=1.0) ≺ c(t=3.0) by epsilon (3-1=2 > 1)
        assert po.is_before(events[1], events[2])
        # a ≺ b ≺ c by transitive closure
        assert po.is_before(events[0], events[2])


class TestQueryMethods:
    """Test is_after, are_independent, and related queries."""

    def test_is_after(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        assert po.is_after(events[2], events[0])  # a1 after iota_P1
        assert not po.is_after(events[0], events[2])

    def test_are_independent(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        assert po.are_independent(events[0], events[1])  # iota_P1, iota_P2
        assert not po.are_independent(events[0], events[2])  # same process


class TestPredecessorsSuccessors:
    """Test predecessor and successor queries."""

    def test_predecessors(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        a2 = events[4]
        preds = po.predecessors(a2)
        # a2 predecessors on P1: iota_P1, a1
        assert events[0] in preds  # iota_P1
        assert events[2] in preds  # a1
        # P2 events are not predecessors (concurrent)
        assert events[1] not in preds
        assert events[3] not in preds

    def test_immediate_predecessors(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        a2 = events[4]
        imm_preds = po.immediate_predecessors(a2)
        # Only a1 is the immediate predecessor of a2 (iota_P1 ≺ a1 ≺ a2)
        assert events[2] in imm_preds  # a1
        assert events[0] not in imm_preds  # iota_P1 is not immediate

    def test_successors(self) -> None:
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        iota_p1 = events[0]
        succs = po.successors(iota_p1)
        assert events[2] in succs  # a1
        assert events[4] in succs  # a2

    def test_no_predecessors_for_initial(self) -> None:
        """Initial events have no predecessors."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        assert len(po.predecessors(events[0])) == 0
        assert len(po.predecessors(events[1])) == 0


class TestTopologicalSort:
    """Test topological sort (linearization)."""

    def test_respects_order(self) -> None:
        """Topological sort respects the partial order."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        topo = po.topological_sort()
        assert len(topo) == len(events)
        # For each ordered pair, the predecessor must come first
        for i, ei in enumerate(topo):
            for ej in topo[i + 1:]:
                assert not po.is_before(ej, ei), f"{ej.eid} should not precede {ei.eid}"

    def test_respects_epsilon_ordering(self) -> None:
        """Topological sort respects epsilon-based orderings."""
        events = [
            Event(eid="e1", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0),
            Event(eid="e2", process="P2", vector_clock=_vc2(0, 1), timestamp=5.0),
        ]
        po = PartialOrder(events, epsilon=1.0)
        topo = po.topological_sort()
        assert topo[0].eid == "e1"
        assert topo[1].eid == "e2"

    def test_valid_linearization_check(self) -> None:
        """is_valid_linearization validates ordering."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        topo = po.topological_sort()
        assert po.is_valid_linearization(topo)

    def test_invalid_linearization(self) -> None:
        """Reversed order is not a valid linearization."""
        events = [
            Event(eid="a", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0),
            Event(eid="b", process="P1", vector_clock=_vc2(2, 0), timestamp=1.0),
        ]
        po = PartialOrder(events, epsilon=float("inf"))
        assert not po.is_valid_linearization([events[1], events[0]])


class TestMinimalEvents:
    """Test get_minimal_events."""

    def test_initial_events_are_minimal(self) -> None:
        """Initial events (no predecessors) are minimal."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        minimals = po.get_minimal_events()
        assert events[0] in minimals  # iota_P1
        assert events[1] in minimals  # iota_P2
        assert len(minimals) == 2

    def test_with_epsilon_fewer_minimals(self) -> None:
        """Epsilon ordering may reduce the set of minimal events."""
        events = [
            Event(eid="e1", process="P1", vector_clock=_vc2(1, 0), timestamp=0.0),
            Event(eid="e2", process="P2", vector_clock=_vc2(0, 1), timestamp=5.0),
        ]
        po = PartialOrder(events, epsilon=1.0)
        minimals = po.get_minimal_events()
        assert events[0] in minimals
        assert events[1] not in minimals  # e1 ≺ e2 by epsilon


class TestEnabledEvents:
    """Test get_enabled_events."""

    def test_initial_events_enabled_from_empty(self) -> None:
        """Minimal events are enabled when nothing is processed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        enabled = po.get_enabled_events(set())
        assert events[0] in enabled
        assert events[1] in enabled

    def test_successor_enabled_after_predecessor(self) -> None:
        """An event becomes enabled after its predecessors are processed."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        processed = {events[0]}  # iota_P1 processed
        enabled = po.get_enabled_events(processed)
        assert events[2] in enabled  # a1 is now enabled
        assert events[0] not in enabled  # already processed

    def test_not_enabled_if_predecessor_missing(self) -> None:
        """Events with unprocessed predecessors are not enabled."""
        events = _two_process_events()
        po = PartialOrder(events, epsilon=float("inf"))
        enabled = po.get_enabled_events(set())
        assert events[4] not in enabled  # a2 needs a1 and iota_P1


class TestThreeProcesses:
    """Test with three processes including message passing."""

    def test_three_process_ordering(self) -> None:
        """Three processes with messages create proper orderings."""
        events = [
            Event(eid="i1", process="P1", vector_clock=_vc3(1, 0, 0), timestamp=0.0),
            Event(eid="i2", process="P2", vector_clock=_vc3(0, 1, 0), timestamp=0.0),
            Event(eid="i3", process="P3", vector_clock=_vc3(0, 0, 1), timestamp=0.0),
            # P1 sends to P2
            Event(eid="s12", process="P1", vector_clock=_vc3(2, 0, 0), timestamp=1.0,
                  event_type="send", target_process="P2"),
            # P2 receives from P1
            Event(eid="r12", process="P2", vector_clock=_vc3(2, 2, 0), timestamp=2.0,
                  event_type="receive", source_process="P1"),
            # P3 is independent of P1-P2 exchange
            Event(eid="c1", process="P3", vector_clock=_vc3(0, 0, 2), timestamp=1.5),
        ]
        po = PartialOrder(events, epsilon=float("inf"))
        # s12 ≺ r12 (message causality)
        assert po.is_before(events[3], events[4])
        # c1 is concurrent with s12 and r12
        assert po.are_concurrent(events[5], events[3])
        assert po.are_concurrent(events[5], events[4])
