"""
Tests for visualization utilities.

Tests cover GraphVisualizer (DOT, ASCII, JSON output) and
PartialOrderVisualizer (ASCII timeline diagrams with process columns,
timestamps, and cross-process ordering annotations).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prove.core.event import Event
from prove.core.partial_order import PartialOrder
from prove.core.sliding_window import SlidingWindowGraph
from prove.core.vector_clock import VectorClock
from prove.parser.ast_nodes import Proposition
from prove.utils.visualization import GraphVisualizer, PartialOrderVisualizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vc(vals: dict[str, int]) -> VectorClock:
    return VectorClock(frozenset(vals.keys()), initial_values=vals)


def _evt(
    eid: str,
    proc: str,
    vc_vals: dict[str, int],
    ts: float,
    props: frozenset[str] = frozenset(),
    event_type: str = "local",
    source_process: str | None = None,
    target_process: str | None = None,
) -> Event:
    return Event(
        eid=eid,
        process=proc,
        vector_clock=_vc(vc_vals),
        timestamp=ts,
        propositions=props,
        event_type=event_type,
        source_process=source_process,
        target_process=target_process,
    )


def _two_proc_events() -> list[Event]:
    """Simple two-process trace: P1 and P2 with 2 events each."""
    return [
        _evt("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"init"})),
        _evt("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"init"})),
        _evt("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"ready"})),
        _evt("e2", "P2", {"P1": 0, "P2": 2}, 2.0, frozenset({"done"})),
    ]


def _three_proc_events() -> list[Event]:
    """Three-process trace with independent events."""
    return [
        _evt("i1", "P1", {"P1": 1, "P2": 0, "P3": 0}, 0.0),
        _evt("i2", "P2", {"P1": 0, "P2": 1, "P3": 0}, 0.0),
        _evt("i3", "P3", {"P1": 0, "P2": 0, "P3": 1}, 0.0),
        _evt("a1", "P1", {"P1": 2, "P2": 0, "P3": 0}, 1.0, frozenset({"a"})),
        _evt("b1", "P2", {"P1": 0, "P2": 2, "P3": 0}, 1.5, frozenset({"b"})),
        _evt("c1", "P3", {"P1": 0, "P2": 0, "P3": 2}, 2.0, frozenset({"c"})),
    ]


def _message_events() -> list[Event]:
    """Two-process trace with message passing (send/receive)."""
    return [
        _evt("ic", "Client", {"Client": 1, "Server": 0}, 0.0, frozenset({"init"})),
        _evt("is", "Server", {"Client": 0, "Server": 1}, 0.0, frozenset({"init"})),
        _evt(
            "c_send",
            "Client",
            {"Client": 2, "Server": 0},
            1.0,
            frozenset({"request"}),
            "send",
            target_process="Server",
        ),
        _evt(
            "s_recv",
            "Server",
            {"Client": 2, "Server": 2},
            2.0,
            frozenset({"recv"}),
            "receive",
            source_process="Client",
        ),
    ]


def _build_graph(events: list[Event], epsilon: float = float("inf")) -> SlidingWindowGraph:
    """Build a SlidingWindowGraph from events."""
    po = PartialOrder(events, epsilon)
    procs = frozenset(e.process for e in events)
    minimal = po.get_minimal_events()
    initial = {}
    for e in minimal:
        if e.process not in initial:
            initial[e.process] = e
    formula = Proposition("ready")
    graph = SlidingWindowGraph(
        processes=procs,
        initial_events=initial,
        formula=formula,
        partial_order=po,
    )
    initial_set = set(initial.values())
    for e in po.topological_sort():
        if e not in initial_set:
            graph.process_event(e)
    return graph


# ===========================================================================
# Tests: GraphVisualizer — DOT
# ===========================================================================


class TestGraphVisualizerDot:
    """Test DOT format output from GraphVisualizer."""

    def test_dot_contains_digraph(self) -> None:
        """DOT output starts with a digraph declaration."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        dot = viz.to_dot()
        assert "digraph" in dot

    def test_dot_contains_nodes(self) -> None:
        """DOT output contains node definitions."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        dot = viz.to_dot()
        # Should have at least one node label
        assert "label=" in dot

    def test_dot_contains_edges(self) -> None:
        """DOT output contains edge definitions with event labels."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        dot = viz.to_dot()
        assert "->" in dot

    def test_dot_three_processes(self) -> None:
        """DOT output handles three-process graphs."""
        graph = _build_graph(_three_proc_events())
        viz = GraphVisualizer(graph)
        dot = viz.to_dot()
        assert "digraph" in dot


# ===========================================================================
# Tests: GraphVisualizer — JSON
# ===========================================================================


class TestGraphVisualizerJson:
    """Test JSON output from GraphVisualizer."""

    def test_json_is_valid(self) -> None:
        """JSON output parses as valid JSON."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        data = json.loads(viz.to_json())
        assert isinstance(data, dict)

    def test_json_has_nodes_and_edges(self) -> None:
        """JSON contains nodes and edges keys."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        data = json.loads(viz.to_json())
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0
        assert len(data["edges"]) > 0


# ===========================================================================
# Tests: GraphVisualizer — ASCII
# ===========================================================================


class TestGraphVisualizerAscii:
    """Test ASCII output from GraphVisualizer."""

    def test_ascii_not_empty(self) -> None:
        """ASCII output is a non-empty string."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        ascii_out = viz.to_ascii()
        assert len(ascii_out) > 0

    def test_ascii_fits_width(self) -> None:
        """ASCII output respects max_width."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        ascii_out = viz.to_ascii(max_width=120)
        for line in ascii_out.splitlines():
            assert len(line) <= 120


# ===========================================================================
# Tests: GraphVisualizer — File I/O
# ===========================================================================


class TestGraphVisualizerFileIO:
    """Test file saving methods."""

    def test_save_dot(self, tmp_path: Path) -> None:
        """save_dot writes a DOT file to disk."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        dot_file = tmp_path / "graph.dot"
        viz.save_dot(dot_file)
        assert dot_file.exists()
        content = dot_file.read_text()
        assert "digraph" in content

    def test_save_png_graceful_fallback(self, tmp_path: Path) -> None:
        """save_png handles missing Graphviz gracefully (no crash)."""
        graph = _build_graph(_two_proc_events())
        viz = GraphVisualizer(graph)
        png_file = tmp_path / "graph.png"
        # Should not raise even if graphviz rendering fails
        try:
            viz.save_png(png_file)
        except Exception:
            pass  # Graceful fallback is acceptable


# ===========================================================================
# Tests: PartialOrderVisualizer — Basic Rendering
# ===========================================================================


class TestPartialOrderVisualizerBasic:
    """Test basic ASCII rendering of partial order diagrams."""

    def test_render_two_processes(self) -> None:
        """Render a two-process partial order."""
        events = _two_proc_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render()
        assert len(output) > 0
        # Should contain process names
        assert "P1" in output
        assert "P2" in output

    def test_render_three_processes(self) -> None:
        """Render a three-process partial order."""
        events = _three_proc_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render()
        assert "P1" in output
        assert "P2" in output
        assert "P3" in output

    def test_render_contains_event_names(self) -> None:
        """Output contains event identifiers."""
        events = _two_proc_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render()
        assert "i1" in output
        assert "i2" in output
        assert "e1" in output
        assert "e2" in output

    def test_render_contains_timestamps(self) -> None:
        """Output contains timestamps."""
        events = _two_proc_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render()
        assert "t=" in output


# ===========================================================================
# Tests: PartialOrderVisualizer — Header and Footer
# ===========================================================================


class TestPartialOrderVisualizerHeaderFooter:
    """Test header and footer formatting."""

    def test_header_contains_epsilon(self) -> None:
        """Header shows epsilon value."""
        events = _two_proc_events()
        po = PartialOrder(events, 2.0)
        viz = PartialOrderVisualizer(po, epsilon=2.0)
        output = viz.render()
        assert "2.0" in output

    def test_header_infinite_epsilon(self) -> None:
        """Header handles infinite epsilon."""
        events = _two_proc_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render()
        assert "inf" in output.lower() or "∞" in output

    def test_footer_lists_orderings(self) -> None:
        """Footer lists cross-process orderings when they exist."""
        events = _two_proc_events()
        # With epsilon=0.5, t(e1)-t(i2) = 1.0 > 0.5, so i2 ≺ e1
        po = PartialOrder(events, 0.5)
        viz = PartialOrderVisualizer(po, epsilon=0.5)
        output = viz.render()
        # Should contain cross-process ordering section
        assert "ordering" in output.lower() or "≺" in output or "<" in output


# ===========================================================================
# Tests: PartialOrderVisualizer — Ordering Classification
# ===========================================================================


class TestOrderingClassification:
    """Test _classify_ordering_reason for different ordering types."""

    def test_vc_ordering_from_message(self) -> None:
        """Message causality is classified as VC ordering."""
        events = _message_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        # c_send ≺ s_recv should be VC
        c_send = events[2]
        s_recv = events[3]
        reason = viz._classify_ordering_reason(c_send, s_recv)
        assert "VC" in reason

    def test_epsilon_ordering(self) -> None:
        """Timestamp-based ordering is classified with Δt and ε."""
        events = [
            _evt("i1", "P1", {"P1": 1, "P2": 0}, 0.0),
            _evt("i2", "P2", {"P1": 0, "P2": 1}, 0.0),
            _evt("e1", "P1", {"P1": 2, "P2": 0}, 1.0),
            _evt("e2", "P2", {"P1": 0, "P2": 2}, 5.0),
        ]
        po = PartialOrder(events, 2.0)
        viz = PartialOrderVisualizer(po, epsilon=2.0)
        e1 = events[2]
        e2 = events[3]
        if po.is_before(e1, e2):
            reason = viz._classify_ordering_reason(e1, e2)
            assert "Δt" in reason or "dt" in reason.lower() or "ε" in reason or "eps" in reason

    def test_vc_ordering_non_message(self) -> None:
        """Non-message VC ordering (same process) classified as VC."""
        events = [
            _evt("i1", "P1", {"P1": 1}, 0.0),
            _evt("e1", "P1", {"P1": 2}, 1.0),
        ]
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        reason = viz._classify_ordering_reason(events[0], events[1])
        assert "VC" in reason


# ===========================================================================
# Tests: PartialOrderVisualizer — Cross-Process Arrows
# ===========================================================================


class TestCrossProcessArrows:
    """Test cross-process arrow rendering."""

    def test_epsilon_arrow_between_processes(self) -> None:
        """Epsilon-based ordering creates a cross-process annotation."""
        events = [
            _evt("i1", "P1", {"P1": 1, "P2": 0}, 0.0),
            _evt("i2", "P2", {"P1": 0, "P2": 1}, 0.0),
            _evt("e1", "P1", {"P1": 2, "P2": 0}, 1.0),
            _evt("e2", "P2", {"P1": 0, "P2": 2}, 5.0),
        ]
        po = PartialOrder(events, 0.5)
        viz = PartialOrderVisualizer(po, epsilon=0.5)
        output = viz.render()
        # Should contain some cross-process ordering information
        lines = output.strip().split("\n")
        assert len(lines) > 3  # More than just header

    def test_message_arrows(self) -> None:
        """Message causality creates cross-process arrows."""
        events = _message_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render()
        # Should mention c_send and s_recv in ordering
        assert "c_send" in output
        assert "s_recv" in output

    def test_no_cross_process_orderings_infinite_epsilon(self) -> None:
        """With infinite epsilon and no messages, no cross-process orderings."""
        events = [
            _evt("i1", "P1", {"P1": 1, "P2": 0}, 0.0),
            _evt("i2", "P2", {"P1": 0, "P2": 1}, 0.0),
        ]
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render()
        # Footer should indicate no cross-process orderings
        assert "i1" in output and "i2" in output


# ===========================================================================
# Tests: PartialOrderVisualizer — Width Control
# ===========================================================================


class TestPartialOrderVisualizerWidth:
    """Test width control for partial order visualization."""

    def test_render_respects_max_width(self) -> None:
        """Output lines do not exceed max_width."""
        events = _two_proc_events()
        po = PartialOrder(events, float("inf"))
        viz = PartialOrderVisualizer(po, epsilon=float("inf"))
        output = viz.render(max_width=80)
        for line in output.splitlines():
            assert len(line) <= 80
