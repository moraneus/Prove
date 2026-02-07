"""
Tests for the Fidge-Mattern vector clock implementation.

Tests cover creation, increment, merge, causal ordering comparisons,
concurrency detection, and string parsing.
"""

import pytest

from prove.core.vector_clock import VectorClock


class TestVectorClockCreation:
    """Test VectorClock initialization."""

    def test_create_with_processes(self) -> None:
        """Create vector clock with process set, all counters start at zero."""
        vc = VectorClock({"P1", "P2", "P3"})
        assert vc.clock["P1"] == 0
        assert vc.clock["P2"] == 0
        assert vc.clock["P3"] == 0

    def test_create_with_initial_values(self) -> None:
        """Create vector clock with explicit initial values."""
        vc = VectorClock({"P1", "P2"}, initial_values={"P1": 5, "P2": 3})
        assert vc.clock["P1"] == 5
        assert vc.clock["P2"] == 3

    def test_create_with_partial_initial_values(self) -> None:
        """Processes not in initial_values default to zero."""
        vc = VectorClock({"P1", "P2", "P3"}, initial_values={"P1": 2})
        assert vc.clock["P1"] == 2
        assert vc.clock["P2"] == 0
        assert vc.clock["P3"] == 0

    def test_processes_attribute(self) -> None:
        """The processes attribute is a frozenset of all process IDs."""
        vc = VectorClock({"P1", "P2"})
        assert vc.processes == frozenset({"P1", "P2"})

    def test_single_process(self) -> None:
        """Vector clock works with a single process."""
        vc = VectorClock({"P1"})
        assert vc.clock["P1"] == 0

    def test_empty_processes_raises(self) -> None:
        """Creating a vector clock with no processes raises ValueError."""
        with pytest.raises(ValueError):
            VectorClock(set())


class TestVectorClockIncrement:
    """Test VectorClock increment operation."""

    def test_increment_returns_new_instance(self) -> None:
        """Increment returns a new VectorClock (immutable)."""
        vc1 = VectorClock({"P1", "P2"})
        vc2 = vc1.increment("P1")
        assert vc2 is not vc1

    def test_increment_single_process(self) -> None:
        """Only the specified process counter is incremented."""
        vc1 = VectorClock({"P1", "P2"})
        vc2 = vc1.increment("P1")
        assert vc2.clock["P1"] == 1
        assert vc2.clock["P2"] == 0

    def test_original_unchanged_after_increment(self) -> None:
        """The original vector clock is not modified (immutable)."""
        vc1 = VectorClock({"P1", "P2"})
        vc1.increment("P1")
        assert vc1.clock["P1"] == 0

    def test_multiple_increments(self) -> None:
        """Multiple increments accumulate correctly."""
        vc = VectorClock({"P1", "P2"})
        vc = vc.increment("P1")
        vc = vc.increment("P1")
        vc = vc.increment("P1")
        assert vc.clock["P1"] == 3
        assert vc.clock["P2"] == 0

    def test_increment_unknown_process_raises(self) -> None:
        """Incrementing a process not in the clock raises ValueError."""
        vc = VectorClock({"P1", "P2"})
        with pytest.raises(ValueError):
            vc.increment("P3")


class TestVectorClockMerge:
    """Test VectorClock merge operation (component-wise maximum)."""

    def test_merge_returns_new_instance(self) -> None:
        """Merge returns a new VectorClock."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P2": 1})
        merged = vc1.merge(vc2)
        assert merged is not vc1
        assert merged is not vc2

    def test_merge_component_wise_maximum(self) -> None:
        """Merge takes the component-wise maximum of both clocks."""
        vc1 = VectorClock({"P1", "P2", "P3"}, initial_values={"P1": 3, "P2": 1, "P3": 0})
        vc2 = VectorClock({"P1", "P2", "P3"}, initial_values={"P1": 1, "P2": 4, "P3": 2})
        merged = vc1.merge(vc2)
        assert merged.clock["P1"] == 3
        assert merged.clock["P2"] == 4
        assert merged.clock["P3"] == 2

    def test_merge_with_identical_clocks(self) -> None:
        """Merging identical clocks produces the same values."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        merged = vc1.merge(vc2)
        assert merged.clock["P1"] == 2
        assert merged.clock["P2"] == 3

    def test_merge_with_zero_clock(self) -> None:
        """Merging with a zero clock returns values from the non-zero clock."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 5, "P2": 3})
        vc2 = VectorClock({"P1", "P2"})
        merged = vc1.merge(vc2)
        assert merged.clock["P1"] == 5
        assert merged.clock["P2"] == 3

    def test_merge_mismatched_processes_raises(self) -> None:
        """Merging clocks with different process sets raises ValueError."""
        vc1 = VectorClock({"P1", "P2"})
        vc2 = VectorClock({"P1", "P3"})
        with pytest.raises(ValueError):
            vc1.merge(vc2)

    def test_originals_unchanged_after_merge(self) -> None:
        """Both original clocks are unmodified after merge."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 3, "P2": 1})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 4})
        vc1.merge(vc2)
        assert vc1.clock["P1"] == 3
        assert vc1.clock["P2"] == 1
        assert vc2.clock["P1"] == 1
        assert vc2.clock["P2"] == 4


class TestVectorClockComparison:
    """Test VectorClock ordering comparisons."""

    def test_le_identical_clocks(self) -> None:
        """A clock is <= itself."""
        vc = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        assert vc <= vc

    def test_le_strictly_less(self) -> None:
        """A clock with all components <= and at least one < is <=."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        assert vc1 <= vc2

    def test_le_not_comparable(self) -> None:
        """Concurrent clocks are not <= each other."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 1})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        assert not (vc1 <= vc2)
        assert not (vc2 <= vc1)

    def test_lt_strictly_before(self) -> None:
        """vc1 < vc2 when all components <= and at least one <."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        assert vc1 < vc2

    def test_lt_not_for_equal(self) -> None:
        """Equal clocks are not strictly less."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        assert not (vc1 < vc2)

    def test_lt_concurrent(self) -> None:
        """Concurrent clocks are not < each other."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 1})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        assert not (vc1 < vc2)
        assert not (vc2 < vc1)

    def test_lt_one_component_equal(self) -> None:
        """vc1 < vc2 when one component is equal and another is less."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 3})
        assert vc1 < vc2


class TestVectorClockConcurrency:
    """Test VectorClock concurrency detection."""

    def test_concurrent_clocks(self) -> None:
        """Clocks are concurrent when neither is causally before the other."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 1})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        assert vc1.is_concurrent_with(vc2)
        assert vc2.is_concurrent_with(vc1)

    def test_not_concurrent_when_ordered(self) -> None:
        """Causally ordered clocks are not concurrent."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 1})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 2})
        assert not vc1.is_concurrent_with(vc2)
        assert not vc2.is_concurrent_with(vc1)

    def test_not_concurrent_with_self(self) -> None:
        """A clock is not concurrent with an equal clock (it's <=)."""
        vc = VectorClock({"P1", "P2"}, initial_values={"P1": 2, "P2": 3})
        assert not vc.is_concurrent_with(vc)

    def test_concurrent_three_processes(self) -> None:
        """Concurrency detection works with three processes."""
        vc1 = VectorClock({"P1", "P2", "P3"}, initial_values={"P1": 2, "P2": 0, "P3": 1})
        vc2 = VectorClock({"P1", "P2", "P3"}, initial_values={"P1": 0, "P2": 2, "P3": 0})
        assert vc1.is_concurrent_with(vc2)


class TestVectorClockFromString:
    """Test VectorClock parsing from string format."""

    def test_parse_basic(self) -> None:
        """Parse a simple vector clock string."""
        vc = VectorClock.from_string("P1:2;P2:1;P3:0", {"P1", "P2", "P3"})
        assert vc.clock["P1"] == 2
        assert vc.clock["P2"] == 1
        assert vc.clock["P3"] == 0

    def test_parse_single_process(self) -> None:
        """Parse vector clock with single process."""
        vc = VectorClock.from_string("P1:5", {"P1"})
        assert vc.clock["P1"] == 5

    def test_parse_zero_values(self) -> None:
        """Parse vector clock where all values are zero."""
        vc = VectorClock.from_string("P1:0;P2:0", {"P1", "P2"})
        assert vc.clock["P1"] == 0
        assert vc.clock["P2"] == 0

    def test_parse_mismatched_processes_raises(self) -> None:
        """Parsing with mismatched process set raises ValueError."""
        with pytest.raises(ValueError):
            VectorClock.from_string("P1:2;P2:1", {"P1", "P2", "P3"})

    def test_parse_invalid_format_raises(self) -> None:
        """Parsing a malformed string raises ValueError."""
        with pytest.raises(ValueError):
            VectorClock.from_string("invalid", {"P1"})


class TestVectorClockEquality:
    """Test VectorClock equality and hashing."""

    def test_equal_clocks(self) -> None:
        """Clocks with same processes and values are equal."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        assert vc1 == vc2

    def test_unequal_clocks(self) -> None:
        """Clocks with different values are not equal."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 3})
        assert vc1 != vc2

    def test_hash_consistency(self) -> None:
        """Equal clocks have equal hashes."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        assert hash(vc1) == hash(vc2)

    def test_usable_in_set(self) -> None:
        """VectorClocks can be used in sets and as dict keys."""
        vc1 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        vc2 = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        assert len({vc1, vc2}) == 1


class TestVectorClockRepr:
    """Test VectorClock string representation."""

    def test_repr_contains_values(self) -> None:
        """String representation includes clock values."""
        vc = VectorClock({"P1", "P2"}, initial_values={"P1": 1, "P2": 2})
        s = repr(vc)
        assert "P1" in s
        assert "P2" in s
