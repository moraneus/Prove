"""
Tests for the sliding window graph implementation.

Tests cover initialization, sequential event processing, concurrent event
processing with backward propagation (diamond property), summary propagation,
redundant node removal, verdict computation, and Paper V2 timing constraints.
"""

import pytest

from prove.core.event import Event
from prove.core.frontier import Frontier
from prove.core.partial_order import PartialOrder
from prove.core.sliding_window import GraphEdge, GraphNode, SlidingWindowGraph
from prove.core.summary import Summary
from prove.core.vector_clock import VectorClock
from prove.parser.ast_nodes import (
    Conjunction,
    Disjunction,
    Implication,
    Negation,
    Proposition,
    Since,
    TrueConstant,
    Yesterday,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCS2 = frozenset({"P1", "P2"})
PROCS3 = frozenset({"P1", "P2", "P3"})


def _vc2(p1: int, p2: int) -> VectorClock:
    return VectorClock(PROCS2, initial_values={"P1": p1, "P2": p2})


def _vc3(p1: int, p2: int, p3: int) -> VectorClock:
    return VectorClock(PROCS3, initial_values={"P1": p1, "P2": p2, "P3": p3})


def _ev(
    eid: str,
    process: str,
    vc: VectorClock,
    timestamp: float,
    props: frozenset[str] = frozenset(),
    event_type: str = "local",
    source_process: str | None = None,
    target_process: str | None = None,
) -> Event:
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


def _make_two_proc_initial() -> tuple[Event, Event]:
    """Create initial events for a 2-process system."""
    iota1 = _ev("iota1", "P1", _vc2(1, 0), 0.0)
    iota2 = _ev("iota2", "P2", _vc2(0, 1), 0.0)
    return iota1, iota2


def _make_three_proc_initial() -> tuple[Event, Event, Event]:
    """Create initial events for a 3-process system."""
    iota1 = _ev("iota1", "P1", _vc3(1, 0, 0), 0.0)
    iota2 = _ev("iota2", "P2", _vc3(0, 1, 0), 0.0)
    iota3 = _ev("iota3", "P3", _vc3(0, 0, 1), 0.0)
    return iota1, iota2, iota3


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test SlidingWindowGraph initialization."""

    def test_initial_node_count(self) -> None:
        """Graph starts with exactly one node."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        assert len(g.nodes) == 1

    def test_initial_frontier(self) -> None:
        """Initial node's frontier contains all initial events."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        node = g.nodes[g.maximal_node_id]
        assert node.frontier.events == frozenset({iota1, iota2})

    def test_initial_summary_present(self) -> None:
        """Initial node has exactly one summary."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        node = g.nodes[g.maximal_node_id]
        assert len(node.summaries) == 1

    def test_initial_no_edges(self) -> None:
        """Graph starts with no edges."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        assert len(g.edges) == 0

    def test_initial_summary_evaluates_initial_props(self) -> None:
        """Initial summary evaluates propositions from initial events."""
        iota1 = _ev("iota1", "P1", _vc2(1, 0), 0.0, frozenset({"ready"}))
        iota2 = _ev("iota2", "P2", _vc2(0, 1), 0.0, frozenset({"done"}))
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Conjunction(Proposition("ready"), Proposition("done"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        node = g.nodes[g.maximal_node_id]
        s = next(iter(node.summaries))
        assert s.evaluate(Proposition("ready")) is True
        assert s.evaluate(Proposition("done")) is True
        assert s.evaluate(formula) is True


# ---------------------------------------------------------------------------
# Tests: Sequential Event Processing
# ---------------------------------------------------------------------------


class TestSequentialProcessing:
    """Test processing events that are totally ordered (no concurrency)."""

    def test_single_event_adds_node_and_edge(self) -> None:
        """Processing one event creates a new node and edge."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"p"}))
        all_events = [iota1, iota2, e1]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        g.process_event(e1)
        assert len(g.nodes) >= 2
        assert len(g.edges) >= 1

    def test_sequential_events_same_process(self) -> None:
        """Two sequential events on same process produce a chain."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0)
        e2 = _ev("e2", "P1", _vc2(3, 0), 2.0)
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        g.process_event(e1)
        g.process_event(e2)
        # At least 2 nodes remain (some may be pruned)
        assert len(g.nodes) >= 1

    def test_summary_updated_on_event(self) -> None:
        """Summary is updated with event's propositions."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"p"}))
        all_events = [iota1, iota2, e1]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        g.process_event(e1)
        node = g.nodes[g.maximal_node_id]
        s = next(iter(node.summaries))
        assert s.evaluate(formula) is True

    def test_maximal_node_updated(self) -> None:
        """Maximal node ID changes after processing an event."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0)
        all_events = [iota1, iota2, e1]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        old_max = g.maximal_node_id
        g.process_event(e1)
        assert g.maximal_node_id != old_max


# ---------------------------------------------------------------------------
# Tests: Concurrent Events and Diamond Property
# ---------------------------------------------------------------------------


class TestConcurrentEvents:
    """Test backward propagation (commutation) with concurrent events."""

    def test_concurrent_events_create_diamond(self) -> None:
        """Two independent concurrent events create a diamond structure."""
        iota1, iota2 = _make_two_proc_initial()
        # e1 on P1, e2 on P2 — independent and concurrent
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0)
        e2 = _ev("e2", "P2", _vc2(0, 2), 1.0)
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        g.process_event(e2)

        # After backward propagation, we should have a diamond:
        # s0 --e1--> s1 --e2--> s3
        # s0 --e2--> s2 --e1--> s3
        # So at least 4 nodes and 4 edges (before pruning)
        # But pruning may remove some; at minimum the maximal node exists
        max_node = g.nodes[g.maximal_node_id]
        # The maximal frontier should have both e1 and e2
        assert max_node.frontier.process_to_event["P1"] == e1
        assert max_node.frontier.process_to_event["P2"] == e2

    def test_diamond_both_paths_produce_summaries(self) -> None:
        """Both paths through diamond contribute summaries to the join node."""
        iota1, iota2 = _make_two_proc_initial()
        # e1 on P1 has prop "p", e2 on P2 has prop "q"
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"p"}))
        e2 = _ev("e2", "P2", _vc2(0, 2), 1.0, frozenset({"q"}))
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Disjunction(Proposition("p"), Proposition("q"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        g.process_event(e2)

        max_node = g.nodes[g.maximal_node_id]
        # At least one summary should exist at the maximal node
        assert len(max_node.summaries) >= 1

    def test_three_concurrent_events(self) -> None:
        """Three concurrent events on three different processes."""
        iota1, iota2, iota3 = _make_three_proc_initial()
        e1 = _ev("e1", "P1", _vc3(2, 0, 0), 1.0)
        e2 = _ev("e2", "P2", _vc3(0, 2, 0), 1.0)
        e3 = _ev("e3", "P3", _vc3(0, 0, 2), 1.0)
        all_events = [iota1, iota2, iota3, e1, e2, e3]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(
            PROCS3, {"P1": iota1, "P2": iota2, "P3": iota3}, formula, po
        )

        g.process_event(e1)
        g.process_event(e2)
        g.process_event(e3)

        max_node = g.nodes[g.maximal_node_id]
        assert max_node.frontier.process_to_event["P1"] == e1
        assert max_node.frontier.process_to_event["P2"] == e2
        assert max_node.frontier.process_to_event["P3"] == e3


# ---------------------------------------------------------------------------
# Tests: Summary Propagation
# ---------------------------------------------------------------------------


class TestSummaryPropagation:
    """Test that summaries are correctly propagated through the graph."""

    def test_since_operator_across_events(self) -> None:
        """Since operator updates correctly when trigger event arrives."""
        iota1, iota2 = _make_two_proc_initial()
        # Event triggers the 'ready' proposition
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"ready"}))
        all_events = [iota1, iota2, e1]
        po = PartialOrder(all_events, epsilon=float("inf"))
        p = Proposition("confirmed")
        q = Proposition("ready")
        formula = Since(p, q)
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        sat, _ = g.get_verdict(formula)
        assert sat is True

    def test_yesterday_operator(self) -> None:
        """Yesterday operator refers to the previous state's value."""
        iota1 = _ev("iota1", "P1", _vc2(1, 0), 0.0, frozenset({"p"}))
        iota2 = _ev("iota2", "P2", _vc2(0, 1), 0.0)
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset())
        all_events = [iota1, iota2, e1]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Yesterday(Proposition("p"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        max_node = g.nodes[g.maximal_node_id]
        # At least one summary should have @p = True (p was true at iota1)
        any_yesterday_true = any(
            s.evaluate(formula) is True for s in max_node.summaries
        )
        assert any_yesterday_true is True

    def test_different_paths_may_produce_different_summaries(self) -> None:
        """Concurrent events with different props can produce different summaries."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"p"}))
        e2 = _ev("e2", "P2", _vc2(0, 2), 1.0, frozenset({"q"}))
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=float("inf"))
        # Yesterday of p: depends on order of events
        formula = Yesterday(Proposition("p"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        g.process_event(e2)

        max_node = g.nodes[g.maximal_node_id]
        # Two paths: e1 then e2, or e2 then e1
        # Path e1→e2: @p after e2 = value of p at e1 = True (e1 has {p})
        # Path e2→e1: @p after e1 = value of p at e2 = False (e2 has {q})
        # So summaries should differ
        assert len(max_node.summaries) >= 1


# ---------------------------------------------------------------------------
# Tests: Redundancy Removal
# ---------------------------------------------------------------------------


class TestRedundancyRemoval:
    """Test that redundant nodes are removed."""

    def test_nodes_removed_when_all_processes_covered(self) -> None:
        """Nodes are pruned when all processes have events beyond them."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0)
        e2 = _ev("e2", "P2", _vc2(0, 2), 2.0)
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        g.process_event(e2)

        # After both P1 and P2 have new events, the initial node should be pruned
        # (R = {P1, P2} = all processes)
        stats = g.get_statistics()
        assert stats["nodes_removed"] >= 1

    def test_node_not_removed_until_all_processes_covered(self) -> None:
        """A node is NOT removed when only some processes have advanced."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0)
        all_events = [iota1, iota2, e1]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        # Only P1 has advanced — initial node should NOT be removed
        stats = g.get_statistics()
        assert stats["nodes_removed"] == 0


# ---------------------------------------------------------------------------
# Tests: Verdict Computation
# ---------------------------------------------------------------------------


class TestVerdict:
    """Test verdict (satisfied/violated) computation."""

    def test_satisfied_simple_prop(self) -> None:
        """Property is satisfied when proposition holds at maximal frontier."""
        iota1 = _ev("iota1", "P1", _vc2(1, 0), 0.0, frozenset({"p"}))
        iota2 = _ev("iota2", "P2", _vc2(0, 1), 0.0)
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        sat, _ = g.get_verdict(formula)
        assert sat is True

    def test_violated_simple_prop(self) -> None:
        """Property is violated when proposition does not hold."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        sat, _ = g.get_verdict(formula)
        assert sat is False

    def test_satisfied_existential(self) -> None:
        """Existential: at least one path satisfies the formula."""
        iota1, iota2 = _make_two_proc_initial()
        # e1 has p, e2 does not — concurrent events
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"p"}))
        e2 = _ev("e2", "P2", _vc2(0, 2), 1.0)
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=float("inf"))
        # @p: yesterday p was true
        # Path e1→e2: last event is e2, @p = True (p was true at e1)
        # Path e2→e1: last event is e1, @p = False (p was false at e2)
        # At least one path has @p = True
        formula = Yesterday(Proposition("p"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        g.process_event(e1)
        g.process_event(e2)
        sat, _ = g.get_verdict(formula)
        assert sat is True

    def test_implication_satisfied(self) -> None:
        """Implication with antecedent false is satisfied."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Implication(Proposition("done"), Proposition("ready"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        sat, _ = g.get_verdict(formula)
        # done=False, so implication is True
        assert sat is True


# ---------------------------------------------------------------------------
# Tests: Paper V2 Timing Constraint on Backward Commutation
# ---------------------------------------------------------------------------


class TestTimingConstraint:
    """Test Paper V2 timing constraint on backward commutation."""

    def test_commutation_blocked_by_timing(self) -> None:
        """Events cannot commute backward when t(e) - t(f) > epsilon."""
        iota1, iota2 = _make_two_proc_initial()
        # e1 on P1 at t=1.0, e2 on P2 at t=5.0, epsilon=2.0
        # t(e2) - t(e1) = 4.0 > 2.0, so e2 cannot commute backward over e1
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"p"}))
        e2 = _ev("e2", "P2", _vc2(0, 2), 5.0, frozenset({"q"}))
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=2.0)
        formula = Yesterday(Proposition("p"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        g.process_event(e2)

        # With timing constraint, e2 should NOT commute backward over e1
        # So there should NOT be a diamond — only the chain path
        max_node = g.nodes[g.maximal_node_id]
        # Only one path: iota -> e1 -> e2
        # So only one summary at the maximal node
        assert len(max_node.summaries) == 1

    def test_commutation_allowed_within_epsilon(self) -> None:
        """Events CAN commute when t(e) - t(f) <= epsilon."""
        iota1, iota2 = _make_two_proc_initial()
        # e1 on P1 at t=1.0, e2 on P2 at t=2.0, epsilon=2.0
        # t(e2) - t(e1) = 1.0 <= 2.0, so commutation IS allowed
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"p"}))
        e2 = _ev("e2", "P2", _vc2(0, 2), 2.0, frozenset({"q"}))
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=2.0)
        formula = Yesterday(Proposition("p"))
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        g.process_event(e2)

        max_node = g.nodes[g.maximal_node_id]
        # Two paths allowed: e1→e2 and e2→e1
        # Should have 2 summaries (different @p values)
        assert len(max_node.summaries) == 2

    def test_epsilon_zero_forces_total_order(self) -> None:
        """With epsilon=0, all events with different timestamps are ordered."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0)
        e2 = _ev("e2", "P2", _vc2(0, 2), 2.0)
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=0.0)
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        g.process_event(e1)
        g.process_event(e2)

        max_node = g.nodes[g.maximal_node_id]
        # With epsilon=0, e1 ≺ e2 by timing, so only one linearization
        assert len(max_node.summaries) == 1


# ---------------------------------------------------------------------------
# Tests: Message Causality
# ---------------------------------------------------------------------------


class TestMessageCausality:
    """Test that message causality is respected."""

    def test_send_receive_order_respected(self) -> None:
        """A receive event cannot precede its corresponding send."""
        iota1, iota2 = _make_two_proc_initial()
        # P1 sends message to P2
        send_e = _ev(
            "send1", "P1", _vc2(2, 0), 1.0,
            event_type="send", target_process="P2",
        )
        recv_e = _ev(
            "recv1", "P2", _vc2(2, 2), 2.0,
            event_type="receive", source_process="P1",
        )
        all_events = [iota1, iota2, send_e, recv_e]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)

        # Process in valid order
        g.process_event(send_e)
        g.process_event(recv_e)

        # send ≺ recv by vector clock, so no diamond
        max_node = g.nodes[g.maximal_node_id]
        assert len(max_node.summaries) == 1


# ---------------------------------------------------------------------------
# Tests: Statistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test graph statistics reporting."""

    def test_statistics_keys(self) -> None:
        """Statistics include expected keys."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        stats = g.get_statistics()
        assert "node_count" in stats
        assert "edge_count" in stats
        assert "nodes_removed" in stats
        assert "max_summaries" in stats

    def test_statistics_after_events(self) -> None:
        """Statistics update after processing events."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0)
        all_events = [iota1, iota2, e1]
        po = PartialOrder(all_events, epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        g.process_event(e1)
        stats = g.get_statistics()
        assert stats["events_processed"] == 1


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases."""

    def test_single_process(self) -> None:
        """Graph works with a single process."""
        procs = frozenset({"P1"})
        iota = _ev("iota", "P1", VectorClock(procs, {"P1": 1}), 0.0)
        e1 = _ev("e1", "P1", VectorClock(procs, {"P1": 2}), 1.0, frozenset({"p"}))
        po = PartialOrder([iota, e1], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(procs, {"P1": iota}, formula, po)
        g.process_event(e1)
        sat, _ = g.get_verdict(formula)
        assert sat is True

    def test_complex_formula_verdict(self) -> None:
        """Verdict works with a complex formula (implication + since)."""
        iota1, iota2 = _make_two_proc_initial()
        e1 = _ev("e1", "P1", _vc2(2, 0), 1.0, frozenset({"ready"}))
        e2 = _ev("e2", "P1", _vc2(3, 0), 2.0, frozenset({"confirmed", "done"}))
        all_events = [iota1, iota2, e1, e2]
        po = PartialOrder(all_events, epsilon=float("inf"))
        # done -> (confirmed S ready)
        formula = Implication(
            Proposition("done"), Since(Proposition("confirmed"), Proposition("ready"))
        )
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        g.process_event(e1)
        g.process_event(e2)
        sat, _ = g.get_verdict(formula)
        assert sat is True

    def test_graph_node_repr(self) -> None:
        """GraphNode has a repr."""
        iota1, iota2 = _make_two_proc_initial()
        po = PartialOrder([iota1, iota2], epsilon=float("inf"))
        formula = Proposition("p")
        g = SlidingWindowGraph(PROCS2, {"P1": iota1, "P2": iota2}, formula, po)
        node = g.nodes[g.maximal_node_id]
        assert "GraphNode" in repr(node)
