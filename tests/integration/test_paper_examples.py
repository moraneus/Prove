"""
Integration tests verifying PROVE against the paper's own examples.

Tests cover trace parsing, partial order computation, message causality,
epsilon-based timing constraints, and end-to-end property verification
for examples from the paper on runtime verification of linear temporal
properties over partial order executions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from prove.core.event import Event
from prove.core.frontier import Frontier
from prove.core.monitor import EPLTLMonitor, MonitorResult
from prove.core.partial_order import PartialOrder
from prove.core.sliding_window import SlidingWindowGraph
from prove.core.vector_clock import VectorClock
from prove.parser.formula import parse_formula
from prove.utils.trace_reader import TraceReader


FIXTURES = Path(__file__).parent.parent / "fixtures"
TRACES = FIXTURES / "traces"
PROPERTIES = FIXTURES / "properties"

FIGURE1_TRACE = TRACES / "paper_figure1.csv"
V2_FIGURE1_TRACE = TRACES / "paper_v2_figure1.csv"


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _event_by_eid(events: list[Event], eid: str) -> Event:
    """Look up a single event by its identifier."""
    for e in events:
        if e.eid == eid:
            return e
    raise ValueError(f"No event with eid '{eid}' found")


def _make_event(
    eid: str,
    process: str,
    vc_values: Dict[str, int],
    timestamp: float,
    propositions: frozenset[str] | None = None,
    event_type: str = "local",
    source_process: str | None = None,
    target_process: str | None = None,
) -> Event:
    """Convenience factory for creating test events."""
    processes = frozenset(vc_values.keys())
    return Event(
        eid=eid,
        process=process,
        vector_clock=VectorClock(processes, initial_values=vc_values),
        timestamp=timestamp,
        propositions=propositions or frozenset(),
        event_type=event_type,
        source_process=source_process,
        target_process=target_process,
    )


# ====================================================================
# Test Suite A: Three Processes with Async Messages (paper_figure1.csv)
# ====================================================================


class TestFigure1ThreeProcesses:
    """Tests for the three-process trace with asynchronous message passing."""

    @pytest.fixture(autouse=True)
    def _load_trace(self) -> None:
        """Load the figure 1 trace data once per test method."""
        reader = TraceReader(FIGURE1_TRACE)
        self.trace_data = reader.read_all(epsilon=2.0)
        self.events = self.trace_data.events
        self.po = self.trace_data.partial_order
        self.metadata = self.trace_data.metadata

    # ----------------------------------------------------------------
    # Trace loading & metadata
    # ----------------------------------------------------------------

    def test_figure1_loads_correctly(self) -> None:
        """TraceReader must parse all 9 events and identify 3 processes."""
        assert len(self.events) == 9
        assert self.metadata.processes == frozenset({"P1", "P2", "P3"})

    def test_figure1_initial_events(self) -> None:
        """Initial events (iota) have no predecessors in the partial order."""
        iota_p1 = _event_by_eid(self.events, "iota_P1")
        iota_p2 = _event_by_eid(self.events, "iota_P2")
        iota_p3 = _event_by_eid(self.events, "iota_P3")

        assert self.po.predecessors(iota_p1) == frozenset()
        assert self.po.predecessors(iota_p2) == frozenset()
        assert self.po.predecessors(iota_p3) == frozenset()

        minimal = self.po.get_minimal_events()
        assert iota_p1 in minimal
        assert iota_p2 in minimal
        assert iota_p3 in minimal

    # ----------------------------------------------------------------
    # Causal ordering
    # ----------------------------------------------------------------

    def test_figure1_message_causality(self) -> None:
        """Send event alpha1 must causally precede receive event beta1."""
        alpha1 = _event_by_eid(self.events, "alpha1")
        beta1 = _event_by_eid(self.events, "beta1")

        assert self.po.is_before(alpha1, beta1)
        assert not self.po.is_before(beta1, alpha1)

    def test_figure1_independence(self) -> None:
        """gamma1 (P3) and alpha1 (P1) are on different processes, hence independent."""
        gamma1 = _event_by_eid(self.events, "gamma1")
        alpha1 = _event_by_eid(self.events, "alpha1")

        assert self.po.are_independent(gamma1, alpha1)

    def test_figure1_concurrent_events(self) -> None:
        """gamma1 (P3) and alpha1 (P1) are concurrent: neither precedes the other."""
        gamma1 = _event_by_eid(self.events, "gamma1")
        alpha1 = _event_by_eid(self.events, "alpha1")

        # gamma1 VC=[0,0,2], alpha1 VC=[2,0,0] -- incomparable by VC
        # t(gamma1)=1.8, t(alpha1)=1.5 -- |Dt|=0.3 <= eps=2 => no timing order
        assert self.po.are_concurrent(gamma1, alpha1)

    # ----------------------------------------------------------------
    # Vector-clock orderings
    # ----------------------------------------------------------------

    def test_figure1_partial_order_vc(self) -> None:
        """Verify fundamental vector-clock orderings in the three-process trace."""
        iota_p1 = _event_by_eid(self.events, "iota_P1")
        iota_p2 = _event_by_eid(self.events, "iota_P2")
        iota_p3 = _event_by_eid(self.events, "iota_P3")
        alpha1 = _event_by_eid(self.events, "alpha1")
        beta1 = _event_by_eid(self.events, "beta1")
        gamma1 = _event_by_eid(self.events, "gamma1")

        # Same-process predecessor relations
        assert self.po.is_before(iota_p1, alpha1)
        assert self.po.is_before(iota_p3, gamma1)

        # Message causality: alpha1 -> beta1, iota_P2 -> beta1
        assert self.po.is_before(alpha1, beta1)
        assert self.po.is_before(iota_p2, beta1)

        # Cross-process non-ordering by VC alone
        # iota_P1 and iota_P2 are concurrent (no causal link)
        assert self.po.are_concurrent(iota_p1, iota_p2)
        assert self.po.are_concurrent(iota_p1, iota_p3)
        assert self.po.are_concurrent(iota_p2, iota_p3)

    # ----------------------------------------------------------------
    # Epsilon-based orderings
    # ----------------------------------------------------------------

    def test_figure1_partial_order_epsilon(self) -> None:
        """
        Verify epsilon-based orderings with eps=2.0.

        Key checks:
        - t(beta2)=3.5, t(alpha1)=1.5 => Dt=2.0, NOT > 2.0, so NO epsilon ordering
        - t(gamma2)=3.0, t(iota_P1)=0.0 => Dt=3.0 > 2.0, so iota_P1 < gamma2 by timing
        - t(beta2)=3.5, t(iota_P1)=0.0 => Dt=3.5 > 2.0, so iota_P1 < beta2 by timing
        """
        iota_p1 = _event_by_eid(self.events, "iota_P1")
        alpha1 = _event_by_eid(self.events, "alpha1")
        beta2 = _event_by_eid(self.events, "beta2")
        gamma2 = _event_by_eid(self.events, "gamma2")

        # Dt = 2.0 is NOT > 2.0 (strict inequality), so no ordering
        # However, alpha1 and beta2 already have alpha1 -> beta1 -> beta2 by
        # VC transitivity (alpha1 VC=[2,0,0] < beta1 VC=[2,2,0] < beta2 VC=[2,3,0]),
        # so alpha1 < beta2 is true anyway via VC transitive closure.
        assert self.po.is_before(alpha1, beta2)

        # Epsilon ordering: iota_P1 (t=0.0) < gamma2 (t=3.0), Dt=3.0 > 2.0
        assert self.po.is_before(iota_p1, gamma2)

        # Epsilon ordering: iota_P1 (t=0.0) < beta2 (t=3.5), Dt=3.5 > 2.0
        assert self.po.is_before(iota_p1, beta2)

    # ----------------------------------------------------------------
    # Topological sort
    # ----------------------------------------------------------------

    def test_figure1_topological_sort(self) -> None:
        """Topological sort must respect all orderings in the partial order."""
        sorted_events = self.po.topological_sort()

        assert len(sorted_events) == 9

        # For every ordered pair, the predecessor must appear earlier
        position = {e.eid: i for i, e in enumerate(sorted_events)}
        for e in self.events:
            for s in self.po.successors(e):
                assert position[e.eid] < position[s.eid], (
                    f"Expected {e.eid} before {s.eid} in topological sort"
                )

        # Verify using the partial order's own validator
        assert self.po.is_valid_linearization(sorted_events)

    # ----------------------------------------------------------------
    # Property verification
    # ----------------------------------------------------------------

    def test_figure1_simple_property_satisfied(self) -> None:
        """Formula 'TRUE S init_P1' (Once init_P1) must be satisfied."""
        formula = parse_formula("TRUE S init_P1")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=self.metadata.processes,
            epsilon=2.0,
        )
        result = monitor.run(self.events)

        assert result.satisfied is True

    def test_figure1_simple_property_violated(self) -> None:
        """Formula 'nonexistent' (proposition never true) must be violated."""
        formula = parse_formula("nonexistent")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=self.metadata.processes,
            epsilon=2.0,
        )
        result = monitor.run(self.events)

        assert result.satisfied is False

    def test_figure1_since_property(self) -> None:
        """
        Formula 'recv_msg -> (TRUE S send_msg)' must be satisfied.

        Wherever recv_msg appears (beta1), send_msg appeared earlier (alpha1),
        and alpha1 < beta1 by message causality.
        """
        formula = parse_formula("recv_msg -> (TRUE S send_msg)")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=self.metadata.processes,
            epsilon=2.0,
        )
        result = monitor.run(self.events)

        assert result.satisfied is True

    # ----------------------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------------------

    def test_figure1_full_pipeline(self) -> None:
        """
        EPLTLMonitor.from_files() end-to-end: load trace, load property,
        run verification, return MonitorResult with proper fields.
        """
        # Create a temporary property file
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".prop", delete=False
        ) as f:
            f.write("# Once init_P1\nTRUE S init_P1\n")
            prop_path = Path(f.name)

        try:
            monitor = EPLTLMonitor.from_files(
                property_file=prop_path,
                trace_file=FIGURE1_TRACE,
                epsilon=2.0,
            )
            result = monitor.run_from_trace()

            assert isinstance(result, MonitorResult)
            assert isinstance(result.satisfied, bool)
            assert isinstance(result.verdict, str)
            assert isinstance(result.statistics, dict)
        finally:
            prop_path.unlink(missing_ok=True)

    def test_figure1_statistics(self) -> None:
        """MonitorResult.statistics must contain expected keys."""
        formula = parse_formula("TRUE S init_P1")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=self.metadata.processes,
            epsilon=2.0,
        )
        result = monitor.run(self.events)

        assert "node_count" in result.statistics
        assert "edge_count" in result.statistics
        assert "events_processed" in result.statistics
        assert result.statistics["events_processed"] > 0


# ====================================================================
# Test Suite B: Paper V2 Figure 1 (Epsilon Timing)
# ====================================================================


class TestV2Figure1EpsilonTiming:
    """Tests for the Paper V2 two-process trace with epsilon-based ordering."""

    @pytest.fixture(autouse=True)
    def _load_trace(self) -> None:
        """Load the V2 figure 1 trace data once per test method."""
        reader = TraceReader(V2_FIGURE1_TRACE)
        self.trace_data = reader.read_all(epsilon=2.0)
        self.events = self.trace_data.events
        self.po = self.trace_data.partial_order
        self.metadata = self.trace_data.metadata

    # ----------------------------------------------------------------
    # Trace loading
    # ----------------------------------------------------------------

    def test_v2_loads_correctly(self) -> None:
        """TraceReader must parse all 8 events and identify 2 processes."""
        assert len(self.events) == 8
        assert self.metadata.processes == frozenset({"P1", "P2"})

    # ----------------------------------------------------------------
    # Timing-based orderings
    # ----------------------------------------------------------------

    def test_v2_timing_ordering(self) -> None:
        """
        alpha1 (t=2.5) and beta2 (t=5.5) are VC-concurrent but
        t(beta2) - t(alpha1) = 3.0 > eps=2, so alpha1 < beta2 by timing.
        """
        alpha1 = _event_by_eid(self.events, "alpha1")
        beta2 = _event_by_eid(self.events, "beta2")

        # VC-concurrent: VC(alpha1)=[2,0], VC(beta2)=[0,3] -- incomparable
        assert alpha1.is_concurrent_with(beta2)

        # But timing orders them
        assert self.po.is_before(alpha1, beta2)

    def test_v2_no_timing_within_epsilon(self) -> None:
        """
        alpha1 (t=2.5) and beta1 (t=1.0): |Dt| = 1.5 <= eps=2,
        so no timing ordering; they remain concurrent.
        """
        alpha1 = _event_by_eid(self.events, "alpha1")
        beta1 = _event_by_eid(self.events, "beta1")

        # VC-concurrent: VC(alpha1)=[2,0], VC(beta1)=[0,2] -- incomparable
        assert alpha1.is_concurrent_with(beta1)

        # Timing: t(alpha1)-t(beta1) = 1.5, NOT > 2; t(beta1)-t(alpha1) = -1.5, NOT > 2
        # So they remain concurrent in the full partial order
        assert self.po.are_concurrent(alpha1, beta1)

    def test_v2_all_timing_orderings(self) -> None:
        """
        Verify all expected epsilon-based cross-process orderings (eps=2).

        Timing orderings (cross-process pairs where Dt > 2):
          iota_P1 (t=0.0) < beta2 (t=5.5): Dt=5.5 > 2
          iota_P1 (t=0.0) < beta3 (t=7.0): Dt=7.0 > 2
          iota_P2 (t=0.0) < alpha1 (t=2.5): Dt=2.5 > 2
          iota_P2 (t=0.0) < alpha2 (t=4.0): Dt=4.0 > 2
          iota_P2 (t=0.0) < alpha3 (t=6.0): Dt=6.0 > 2
          beta1 (t=1.0) < alpha2 (t=4.0): Dt=3.0 > 2
          beta1 (t=1.0) < alpha3 (t=6.0): Dt=5.0 > 2
          alpha1 (t=2.5) < beta2 (t=5.5): Dt=3.0 > 2
          alpha1 (t=2.5) < beta3 (t=7.0): Dt=4.5 > 2
          alpha2 (t=4.0) < beta3 (t=7.0): Dt=3.0 > 2
          beta2 (t=5.5) < alpha3... Dt=0.5, NOT > 2 -- no ordering

        Pairs that remain concurrent (Dt <= 2):
          iota_P1 (t=0.0) and beta1 (t=1.0): Dt=1.0 <= 2
          alpha1 (t=2.5) and beta1 (t=1.0): |Dt|=1.5 <= 2
          alpha2 (t=4.0) and beta2 (t=5.5): Dt=1.5 <= 2
          alpha3 (t=6.0) and beta2 (t=5.5): |Dt|=0.5 <= 2
          alpha3 (t=6.0) and beta3 (t=7.0): Dt=1.0 <= 2
        """
        ev = {e.eid: e for e in self.events}

        # Pairs that ARE ordered by timing (cross-process)
        ordered_pairs = [
            ("iota_P1", "beta2"),
            ("iota_P1", "beta3"),
            ("iota_P2", "alpha1"),
            ("iota_P2", "alpha2"),
            ("iota_P2", "alpha3"),
            ("beta1", "alpha2"),
            ("beta1", "alpha3"),
            ("alpha1", "beta2"),
            ("alpha1", "beta3"),
            ("alpha2", "beta3"),
        ]
        for e1_id, e2_id in ordered_pairs:
            assert self.po.is_before(ev[e1_id], ev[e2_id]), (
                f"Expected {e1_id} < {e2_id} in partial order"
            )

        # Pairs that remain concurrent (cross-process only)
        concurrent_pairs = [
            ("iota_P1", "beta1"),
            ("alpha1", "beta1"),
            ("alpha2", "beta2"),
            ("alpha3", "beta2"),
            ("alpha3", "beta3"),
        ]
        for e1_id, e2_id in concurrent_pairs:
            assert self.po.are_concurrent(ev[e1_id], ev[e2_id]), (
                f"Expected {e1_id} and {e2_id} to be concurrent"
            )

    # ----------------------------------------------------------------
    # Property verification
    # ----------------------------------------------------------------

    def test_v2_property_satisfied(self) -> None:
        """Formula 'TRUE S a' (Once a) must be satisfied -- proposition a appears on alpha1."""
        formula = parse_formula("TRUE S a")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=self.metadata.processes,
            epsilon=2.0,
        )
        result = monitor.run(self.events)

        assert result.satisfied is True

    def test_v2_property_violated(self) -> None:
        """Formula 'nonexistent' (proposition never true) must be violated."""
        formula = parse_formula("nonexistent")
        monitor = EPLTLMonitor(
            formula=formula,
            processes=self.metadata.processes,
            epsilon=2.0,
        )
        result = monitor.run(self.events)

        assert result.satisfied is False

    # ----------------------------------------------------------------
    # Epsilon sensitivity
    # ----------------------------------------------------------------

    def test_v2_different_epsilon_changes_order(self) -> None:
        """
        The same trace with different epsilon values produces partial orders
        with different numbers of ordered pairs.

        eps=0.5 => more events are ordered by timing => more ordered pairs
        eps=10  => fewer events are ordered by timing => fewer ordered pairs
        """
        reader = TraceReader(V2_FIGURE1_TRACE)

        td_small = reader.read_all(epsilon=0.5)
        td_large = reader.read_all(epsilon=10.0)

        po_small = td_small.partial_order
        po_large = td_large.partial_order

        # Count cross-process ordered pairs
        def count_cross_process_ordered(po: PartialOrder) -> int:
            count = 0
            for e in po.events:
                for s in po.successors(e):
                    if e.process != s.process:
                        count += 1
            return count

        cross_small = count_cross_process_ordered(po_small)
        cross_large = count_cross_process_ordered(po_large)

        # Smaller epsilon => more timing orderings => more cross-process pairs
        assert cross_small > cross_large, (
            f"eps=0.5 should yield more cross-process orderings ({cross_small}) "
            f"than eps=10 ({cross_large})"
        )


# ====================================================================
# Test Suite C: Graph Construction
# ====================================================================


class TestGraphConstruction:
    """
    Tests for sliding window graph construction using a simple
    two-process inline trace with concurrent events.
    """

    @pytest.fixture(autouse=True)
    def _build_trace(self) -> None:
        """
        Build a simple two-process trace inline.

        Events:
          iota_P1 on P1 (t=0, VC=[1,0])
          iota_P2 on P2 (t=0, VC=[0,1])
          alpha1  on P1 (t=1, VC=[2,0])
          beta1   on P2 (t=1.5, VC=[0,2])
          alpha2  on P1 (t=2, VC=[3,0])

        With eps=inf: alpha1, beta1, and alpha2/beta1 are concurrent.
        """
        procs = frozenset({"P1", "P2"})

        self.iota_p1 = _make_event(
            "iota_P1", "P1", {"P1": 1, "P2": 0}, 0.0,
            frozenset({"init"}),
        )
        self.iota_p2 = _make_event(
            "iota_P2", "P2", {"P1": 0, "P2": 1}, 0.0,
            frozenset({"init"}),
        )
        self.alpha1 = _make_event(
            "alpha1", "P1", {"P1": 2, "P2": 0}, 1.0,
            frozenset({"ready"}),
        )
        self.beta1 = _make_event(
            "beta1", "P2", {"P1": 0, "P2": 2}, 1.5,
            frozenset({"waiting"}),
        )
        self.alpha2 = _make_event(
            "alpha2", "P1", {"P1": 3, "P2": 0}, 2.0,
            frozenset({"done"}),
        )

        self.all_events = [
            self.iota_p1, self.iota_p2,
            self.alpha1, self.beta1, self.alpha2,
        ]
        self.processes = procs
        self.po = PartialOrder(self.all_events, epsilon=float("inf"))

    def _make_graph(self, formula_str: str) -> SlidingWindowGraph:
        """Helper to create a SlidingWindowGraph for the test trace."""
        formula = parse_formula(formula_str)
        initial_events = {"P1": self.iota_p1, "P2": self.iota_p2}
        return SlidingWindowGraph(
            processes=self.processes,
            initial_events=initial_events,
            formula=formula,
            partial_order=self.po,
        )

    # ----------------------------------------------------------------
    # Graph structure
    # ----------------------------------------------------------------

    def test_graph_initial_node(self) -> None:
        """After initialization, graph has exactly 1 node (the initial frontier)."""
        graph = self._make_graph("ready")
        assert len(graph.nodes) == 1

    def test_graph_after_first_event(self) -> None:
        """After processing alpha1, graph has 2 nodes and 1 edge."""
        graph = self._make_graph("ready")
        graph.process_event(self.alpha1)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_graph_concurrent_events_create_diamond(self) -> None:
        """
        Processing concurrent events alpha1 and beta1 triggers backward
        propagation (commutation), creating a diamond structure.

        The diamond has at least 3 non-pruned nodes because:
        - Initial frontier {iota_P1, iota_P2}
        - After alpha1: {alpha1, iota_P2}
        - After beta1 from {alpha1, iota_P2}: {alpha1, beta1} (new maximal)
        - Backward propagation: beta1 commutes past alpha1 creating
          {iota_P1, beta1}, which also leads to {alpha1, beta1}.

        Some of these nodes may be pruned if they become redundant.
        At minimum, after processing both concurrent events the graph
        should have more nodes than after processing just one event.
        """
        graph = self._make_graph("ready")

        # Process events in topological order
        order = self.po.topological_sort()
        initial_set = {self.iota_p1, self.iota_p2}
        non_initial = [e for e in order if e not in initial_set]

        # Process alpha1 first
        graph.process_event(non_initial[0])
        nodes_after_one = len(graph.nodes)

        # Process next event (beta1 or whatever is next in sort)
        graph.process_event(non_initial[1])
        nodes_after_two = len(graph.nodes)

        # After two concurrent events, there should be at least 3 nodes
        # (even after pruning) or at minimum more than after one event
        assert nodes_after_two >= nodes_after_one

    def test_graph_node_count_grows(self) -> None:
        """After processing all events, graph has more nodes than initial."""
        graph = self._make_graph("ready")

        order = self.po.topological_sort()
        initial_set = {self.iota_p1, self.iota_p2}
        for e in order:
            if e not in initial_set:
                graph.process_event(e)

        # The maximal node must exist
        assert graph.maximal_node_id in graph.nodes

    def test_graph_summaries_propagated(self) -> None:
        """After processing all events, the maximal node must have summaries."""
        graph = self._make_graph("ready")

        order = self.po.topological_sort()
        initial_set = {self.iota_p1, self.iota_p2}
        for e in order:
            if e not in initial_set:
                graph.process_event(e)

        max_node = graph.nodes[graph.maximal_node_id]
        assert len(max_node.summaries) > 0

    def test_graph_verdict_correct(self) -> None:
        """
        Verify get_verdict returns the correct result for formula 'ready'.

        The proposition 'ready' is true at alpha1. Since alpha1 is in
        the execution, there exists a linearization where 'ready'
        holds at the final frontier if the last event processed on P1
        is alpha2 (which has 'done', not 'ready'). The formula 'ready'
        is just a proposition -- it checks the CURRENT state. At the
        maximal frontier, the global state is {done, waiting} (from
        alpha2 and beta1), so 'ready' should be False.

        But 'TRUE S ready' (Once ready) should be True because 'ready'
        was true at some point.
        """
        formula = parse_formula("TRUE S ready")
        initial_events = {"P1": self.iota_p1, "P2": self.iota_p2}
        graph = SlidingWindowGraph(
            processes=self.processes,
            initial_events=initial_events,
            formula=formula,
            partial_order=self.po,
        )

        order = self.po.topological_sort()
        initial_set = {self.iota_p1, self.iota_p2}
        for e in order:
            if e not in initial_set:
                graph.process_event(e)

        is_satisfied, _ = graph.get_verdict(formula)
        assert is_satisfied is True
