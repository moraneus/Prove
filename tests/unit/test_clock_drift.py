"""
Tests for the ClockDriftHandler.

Tests cover epsilon-based temporal ordering between events, including
ordering determination, definite-before checks, and partitioning of
events into (before, after, concurrent) sets relative to a reference.
"""

import pytest

from prove.core.clock_drift import ClockDriftHandler
from prove.core.event import Event
from prove.core.vector_clock import VectorClock

PROCS = frozenset({"P1", "P2", "P3"})


def _vc(p1: int = 0, p2: int = 0, p3: int = 0) -> VectorClock:
    return VectorClock(PROCS, initial_values={"P1": p1, "P2": p2, "P3": p3})


def _evt(eid: str, process: str, ts: float) -> Event:
    """Create a minimal event for clock drift tests."""
    vals = {"p1": 0, "p2": 0, "p3": 0}
    vals[process.lower()] = 1
    return Event(eid=eid, process=process, vector_clock=_vc(**vals), timestamp=ts)


class TestTimeOrders:
    """Test the time_orders tristate method."""

    def test_e1_before_e2(self) -> None:
        """Returns True when t(e2) - t(e1) > epsilon."""
        h = ClockDriftHandler(epsilon=2.0)
        e1 = _evt("e1", "P1", 1.0)
        e2 = _evt("e2", "P2", 4.0)
        assert h.time_orders(e1, e2) is True

    def test_e2_before_e1(self) -> None:
        """Returns False when t(e1) - t(e2) > epsilon."""
        h = ClockDriftHandler(epsilon=2.0)
        e1 = _evt("e1", "P1", 5.0)
        e2 = _evt("e2", "P2", 1.0)
        assert h.time_orders(e1, e2) is False

    def test_within_epsilon_returns_none(self) -> None:
        """Returns None when both directions are within epsilon."""
        h = ClockDriftHandler(epsilon=2.0)
        e1 = _evt("e1", "P1", 1.0)
        e2 = _evt("e2", "P2", 2.5)
        assert h.time_orders(e1, e2) is None

    def test_exactly_at_epsilon_returns_none(self) -> None:
        """At the epsilon boundary (diff == epsilon), cannot determine order."""
        h = ClockDriftHandler(epsilon=2.0)
        e1 = _evt("e1", "P1", 1.0)
        e2 = _evt("e2", "P2", 3.0)
        assert h.time_orders(e1, e2) is None

    def test_same_timestamp_returns_none(self) -> None:
        """Events with identical timestamps cannot be time-ordered."""
        h = ClockDriftHandler(epsilon=2.0)
        e1 = _evt("e1", "P1", 5.0)
        e2 = _evt("e2", "P2", 5.0)
        assert h.time_orders(e1, e2) is None


class TestIsDefinitelyBefore:
    """Test the is_definitely_before convenience method."""

    def test_definitely_before(self) -> None:
        h = ClockDriftHandler(epsilon=1.0)
        e1 = _evt("e1", "P1", 1.0)
        e2 = _evt("e2", "P2", 3.0)
        assert h.is_definitely_before(e1, e2)

    def test_not_definitely_before(self) -> None:
        h = ClockDriftHandler(epsilon=5.0)
        e1 = _evt("e1", "P1", 1.0)
        e2 = _evt("e2", "P2", 3.0)
        assert not h.is_definitely_before(e1, e2)

    def test_reverse_direction(self) -> None:
        h = ClockDriftHandler(epsilon=1.0)
        e1 = _evt("e1", "P1", 5.0)
        e2 = _evt("e2", "P2", 1.0)
        assert not h.is_definitely_before(e1, e2)
        assert h.is_definitely_before(e2, e1)


class TestZeroEpsilon:
    """Test with epsilon=0 (most restrictive)."""

    def test_any_positive_diff_orders(self) -> None:
        """With ε=0, any positive timestamp difference creates an ordering."""
        h = ClockDriftHandler(epsilon=0.0)
        e1 = _evt("e1", "P1", 1.0)
        e2 = _evt("e2", "P2", 1.001)
        assert h.time_orders(e1, e2) is True

    def test_same_timestamp_still_none(self) -> None:
        """Even with ε=0, identical timestamps are not orderable."""
        h = ClockDriftHandler(epsilon=0.0)
        e1 = _evt("e1", "P1", 5.0)
        e2 = _evt("e2", "P2", 5.0)
        assert h.time_orders(e1, e2) is None


class TestLargeEpsilon:
    """Test with large epsilon (least restrictive)."""

    def test_large_epsilon_no_ordering(self) -> None:
        """With very large ε, even large time differences don't order events."""
        h = ClockDriftHandler(epsilon=1_000_000.0)
        e1 = _evt("e1", "P1", 1.0)
        e2 = _evt("e2", "P2", 999.0)
        assert h.time_orders(e1, e2) is None

    def test_infinite_epsilon(self) -> None:
        """With ε=∞, no time-based ordering is possible."""
        h = ClockDriftHandler(epsilon=float("inf"))
        e1 = _evt("e1", "P1", 0.0)
        e2 = _evt("e2", "P2", 1e12)
        assert h.time_orders(e1, e2) is None


class TestFilterConcurrentEvents:
    """Test partitioning events relative to a reference."""

    def test_partition_mixed(self) -> None:
        """Correctly partitions events into before, after, and concurrent."""
        h = ClockDriftHandler(epsilon=2.0)
        ref = _evt("ref", "P1", 5.0)
        e_before = _evt("eb", "P2", 1.0)  # 5.0 - 1.0 = 4.0 > 2.0 → before
        e_after = _evt("ea", "P3", 9.0)  # 9.0 - 5.0 = 4.0 > 2.0 → after
        e_conc = _evt("ec", "P2", 4.0)  # 5.0 - 4.0 = 1.0 ≤ 2.0 → concurrent

        before, after, concurrent = h.filter_concurrent_events([e_before, e_after, e_conc], ref)
        assert e_before in before
        assert e_after in after
        assert e_conc in concurrent

    def test_all_concurrent(self) -> None:
        """All events within epsilon are concurrent."""
        h = ClockDriftHandler(epsilon=10.0)
        ref = _evt("ref", "P1", 5.0)
        events = [_evt(f"e{i}", "P2", 5.0 + i) for i in range(5)]
        before, after, concurrent = h.filter_concurrent_events(events, ref)
        assert len(before) == 0
        assert len(after) == 0
        assert len(concurrent) == 5

    def test_empty_input(self) -> None:
        """Empty input returns empty partitions."""
        h = ClockDriftHandler(epsilon=2.0)
        ref = _evt("ref", "P1", 5.0)
        before, after, concurrent = h.filter_concurrent_events([], ref)
        assert len(before) == 0
        assert len(after) == 0
        assert len(concurrent) == 0

    def test_all_before(self) -> None:
        """All events definitely before reference."""
        h = ClockDriftHandler(epsilon=0.5)
        ref = _evt("ref", "P1", 10.0)
        events = [_evt(f"e{i}", "P2", float(i)) for i in range(5)]
        before, after, concurrent = h.filter_concurrent_events(events, ref)
        assert len(before) == 5
        assert len(after) == 0
        assert len(concurrent) == 0
