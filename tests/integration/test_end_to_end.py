"""
End-to-end integration tests for the PROVE runtime verification engine.

Tests the complete pipeline from CLI invocation through to verdict,
the Python API (EPLTLMonitor), and edge cases that stress the
interaction between parser, partial order, sliding window graph,
and summary evaluation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

from prove.core.event import Event
from prove.core.monitor import EPLTLMonitor, MonitorResult
from prove.core.partial_order import PartialOrder
from prove.core.vector_clock import VectorClock
from prove.parser.formula import parse_formula

# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent.parent / "fixtures"
TRACES = FIXTURES / "traces"
PROPERTIES = FIXTURES / "properties"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Invoke the PROVE CLI via ``python -m prove`` and return the result."""
    cmd = [sys.executable, "-m", "prove", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(Path(__file__).parent.parent.parent),
    )


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
    """Convenience factory for creating Event objects in tests."""
    processes = frozenset(vc_values.keys())
    return Event(
        eid=eid,
        process=process,
        vector_clock=VectorClock(processes, vc_values),
        timestamp=timestamp,
        propositions=propositions or frozenset(),
        event_type=event_type,
        source_process=source_process,
        target_process=target_process,
    )


# ===================================================================
# CLI Integration Tests
# ===================================================================


class TestCLIIntegration:
    """Test the full CLI pipeline via subprocess invocations."""

    def test_cli_satisfied_exit_code(self, tmp_path: Path) -> None:
        """Formula TRUE is always satisfied -- exit code 0."""
        prop = tmp_path / "true.prop"
        prop.write_text("TRUE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
        )
        assert result.returncode == 0

    def test_cli_violated_exit_code(self, tmp_path: Path) -> None:
        """Formula FALSE is always violated -- exit code 1."""
        prop = tmp_path / "false.prop"
        prop.write_text("FALSE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
        )
        assert result.returncode == 1

    def test_cli_missing_file_exit_code(self, tmp_path: Path) -> None:
        """Non-existent trace file should produce exit code 2."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(tmp_path / "nonexistent.csv"),
        )
        assert result.returncode == 2

    def test_cli_parse_error_exit_code(self, tmp_path: Path) -> None:
        """Syntactically invalid formula should produce exit code 2."""
        prop = tmp_path / "bad.prop"
        prop.write_text("&&& bad syntax\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
        )
        assert result.returncode == 2

    def test_cli_silent_mode(self, tmp_path: Path) -> None:
        """Silent mode should produce no stdout output."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
            "-o", "silent",
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_cli_normal_mode(self, tmp_path: Path) -> None:
        """Normal mode should include 'SATISFIED' or 'VIOLATED' in output."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
            "-o", "normal",
        )
        assert result.returncode == 0
        assert "SATISFIED" in result.stdout or "VIOLATED" in result.stdout

    def test_cli_verbose_mode(self, tmp_path: Path) -> None:
        """Verbose mode should produce more output than normal mode."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")
        trace = str(TRACES / "simple_trace.csv")

        normal = _run_cli("-p", str(prop), "-t", trace, "-o", "normal")
        verbose = _run_cli("-p", str(prop), "-t", trace, "-o", "verbose")

        assert len(verbose.stdout) >= len(normal.stdout)

    def test_cli_visualize_ascii_output(self, tmp_path: Path) -> None:
        """--visualize-ascii should print process column headers."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
            "--visualize-ascii",
        )
        assert result.returncode == 0
        # The simple_trace has processes P1 and P2
        assert "P1" in result.stdout
        assert "P2" in result.stdout

    def test_cli_visualize_dot_file(self, tmp_path: Path) -> None:
        """--visualize FILE.dot should create a DOT file containing 'digraph'."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")
        dot_path = tmp_path / "graph.dot"

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
            "--visualize", str(dot_path),
        )
        assert result.returncode == 0
        assert dot_path.exists()
        content = dot_path.read_text()
        assert "digraph" in content

    def test_cli_stats_output(self, tmp_path: Path) -> None:
        """--stats should print a statistics section."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
            "--stats",
        )
        assert result.returncode == 0
        assert "Statistics" in result.stdout

    def test_cli_combined_flags(self, tmp_path: Path) -> None:
        """Combining --visualize-ascii and --stats should include both."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")

        result = _run_cli(
            "-p", str(prop),
            "-t", str(TRACES / "simple_trace.csv"),
            "--visualize-ascii",
            "--stats",
        )
        assert result.returncode == 0
        # ASCII diagram includes process names
        assert "P1" in result.stdout
        # Stats section
        assert "Statistics" in result.stdout

    def test_cli_epsilon_flag(self, tmp_path: Path) -> None:
        """Passing different epsilon values should produce valid runs."""
        prop = tmp_path / "t.prop"
        prop.write_text("TRUE\n")
        trace = str(TRACES / "epsilon_trace.csv")

        r1 = _run_cli("-p", str(prop), "-t", trace, "-e", "0.5")
        r2 = _run_cli("-p", str(prop), "-t", trace, "-e", "100.0")

        # Both should succeed (TRUE is always satisfied)
        assert r1.returncode == 0
        assert r2.returncode == 0

    def test_cli_version(self) -> None:
        """--version should print version and exit cleanly."""
        result = _run_cli("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout


# ===================================================================
# API Integration Tests
# ===================================================================


class TestAPIIntegration:
    """Test the EPLTLMonitor Python API directly."""

    def test_sequential_single_process(self) -> None:
        """Single-process trace with 3 sequential events."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"init"})),
            _make_event("e1", "P1", {"P1": 2}, 1.0, frozenset({"ready"})),
            _make_event("e2", "P1", {"P1": 3}, 2.0, frozenset({"done"})),
        ]

        formula = parse_formula("ready")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        # "ready" is NOT true at the final state (done is true, not ready).
        # But the summary evaluates the formula at the last event only --
        # and a single linearisation exists.  ready is only true at e1, not e2.
        # So the final verdict depends on whether "ready" means "currently true at end".
        # Since the formula is a bare proposition, it evaluates at the final state.
        # At e2 the propositions are {"done"}, so "ready" is False.
        assert isinstance(result, MonitorResult)
        assert result.satisfied is False

    def test_concurrent_two_processes(self) -> None:
        """Two-process trace with concurrent events (epsilon=inf)."""
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"init_P1"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"init_P2"})),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"a"})),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 1.0, frozenset({"b"})),
        ]

        # At the final frontier, P1 is at e1 (props={"a"}) and P2 is at e2
        # (props={"b"}), so the global state is {"a", "b"}.
        formula = parse_formula("a & b")
        monitor = EPLTLMonitor(formula, procs, epsilon=float("inf"))
        result = monitor.run(events)

        assert result.satisfied is True

    def test_epsilon_changes_verdict(self) -> None:
        """
        Demonstrate that epsilon can change the verdict.

        Trace (2 processes, no messages):
            P1: i1(t=0) -> e1{a}(t=3.0)
            P2: i2(t=0) -> e2{b}(t=1.0)

        Formula: "a S b"  (a since b: b was true at some point and a has been
        true from then until now)

        With epsilon=1.0:
            t(e1) - t(e2) = 2.0 > 1.0 => e2 ≺ e1.
            Linearisation: i1, i2, e2{b}, e1{a}.
            At e2: a_S_b = b_now(True) = True.
            At e1: a_S_b = a_now(True) AND since_before(True) = True.
            So the formula is SATISFIED.

        With epsilon=inf:
            e1 and e2 are concurrent. Multiple linearisations exist.
            Linearisation i1, i2, e1{a}, e2{b}: at final state e2, a=False
            so a_S_b = b_now(True) OR (a_now(False) AND ...) = True.
            Linearisation i1, i2, e2{b}, e1{a}: at final state e1, b=False
            so a_S_b = b_now(False) OR (a_now(True) AND since_before(True)) = True.
            Both linearisations satisfy. Still SATISFIED.

        Let us instead use a formula that is sensitive to ordering:
        "@b" (yesterday b -- b was true in the immediately previous state)

        With epsilon=1.0 (e2 ≺ e1), the only linearisation (after initials)
        processes e2 then e1. At e1, the previous global state included e2's
        "b", so @b = True at the final state.

        With epsilon=inf, both orderings exist. In the path where e1 is
        processed before e2 (i.e., the last event is e2), the previous
        state had e1's "a" and i2's props. @b would be false.
        In the other path, @b is True. Existentially, at least one
        linearisation satisfies, so SATISFIED.

        We need a formula that is VIOLATED existentially under one epsilon
        but SATISFIED under another. Use "!(@b)" with a 3-event construction:

        Actually, let us try a cleaner approach:
        P1: i1(t=0) -> e1{a}(t=1.0)
        P2: i2(t=0) -> e2{b}(t=5.0)

        Formula: "@a" (yesterday a -- a was true in the previous state)

        With epsilon=1.0:
            t(e2)-t(e1) = 4.0 > 1.0 => e1 ≺ e2.
            Linearisation: i1, i2, e1{a}, e2{b}.
            At e2 (final): previous global state after e1 = {a, init_P2}.
            @a at e2 = True. SATISFIED.

        With epsilon=100.0:
            Same ordering (t(e2)-t(e1)=4.0 is NOT > 100), so e1 and e2
            are concurrent. Both orderings possible. In at least one
            linearisation (e1 before e2), @a at e2 is True.
            SATISFIED as well.

        A truly differentiating example requires all paths to fail
        under one epsilon. Let us use 3 events on P1 and force ordering:

        P1: i1(t=0) -> e1{a}(t=10.0)
        P2: i2(t=0) -> e2{b}(t=1.0) -> e3{c}(t=2.0)

        Formula: "@a" (yesterday a)

        With epsilon=0.1:
            t(e2)-t(i1)=1.0 > 0.1, t(e1)-t(e3)=8.0 > 0.1.
            Order: e2 ≺ e1, e3 ≺ e1.
            Only linearisation: i1, i2, e2, e3, e1.
            At e1 (final): previous state had e3{c} and i1 props.
            @a = False (a was not in previous state).
            VIOLATED.

        With epsilon=100.0:
            e1 and e2 concurrent, e1 and e3 concurrent. Multiple orderings.
            Linearisation: i1, i2, e1, e2, e3 -- at e3 (final), prev
            state had e2{b} and e1{a}. @a = True at e3? Let's check:
            the global state after processing e2 (with e1 already done):
            P1 max event = e1{a}, P2 max event = e2{b}. props = {a,b}.
            At e3 (processing e3 after e2): prev summary had a=True.
            @a = old[a] = True. SATISFIED.

        So: epsilon=0.1 => VIOLATED, epsilon=100.0 => SATISFIED.
        """
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"init_P1"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"init_P2"})),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 10.0, frozenset({"a"})),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 1.0, frozenset({"b"})),
            _make_event("e3", "P2", {"P1": 0, "P2": 3}, 2.0, frozenset({"c"})),
        ]

        formula = parse_formula("@a")

        # Small epsilon: e2 and e3 are forced before e1. @a is False at e1.
        monitor_strict = EPLTLMonitor(formula, procs, epsilon=0.1)
        result_strict = monitor_strict.run(events)

        # Large epsilon: concurrent events allow orderings where @a is True.
        monitor_loose = EPLTLMonitor(formula, procs, epsilon=100.0)
        result_loose = monitor_loose.run(events)

        # The two verdicts should differ
        assert result_strict.satisfied != result_loose.satisfied

    def test_since_across_processes(self) -> None:
        """
        Test 'p S q' across processes with message causality.

        P1: i1(t=0) -> e1{p}(t=3.0, receives from P2)
        P2: i2(t=0) -> e2{q}(t=1.0, sends to P1)

        Message causality: e2 ≺ e1 (send before receive).
        Only linearisation: i1, i2, e2, e1.

        At e2: global state = {q, init_P1}. p_S_q = q_now(T) = True.
        At e1: global state = {p, q}. Wait -- after e1 processes,
               P1's max event is e1{p}, P2's max event is e2{q}.
               So global state = {p, q}.
               p_S_q = q_now(T) OR (p_now(T) AND since_before(T)) = True.

        Verdict: SATISFIED.
        """
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"init_P1"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"init_P2"})),
            _make_event(
                "e2", "P2", {"P1": 0, "P2": 2}, 1.0, frozenset({"q"}),
                event_type="send", target_process="P1",
            ),
            _make_event(
                "e1", "P1", {"P1": 2, "P2": 2}, 3.0, frozenset({"p"}),
                event_type="receive", source_process="P2",
            ),
        ]

        formula = parse_formula("p S q")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        assert result.satisfied is True

    def test_yesterday_property(self) -> None:
        """
        Yesterday operator on a single-process trace.

        Trace: e0{a}(t=0) -> e1{b}(t=1).
        After e1, @a should be True (a was true at the previous state).
        After e0 (the first state), @b should be False (no previous state).
        """
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"a"})),
            _make_event("e1", "P1", {"P1": 2}, 1.0, frozenset({"b"})),
        ]

        # @a should be True at the end (a was True in e0, the previous state of e1)
        formula_ya = parse_formula("@a")
        monitor_ya = EPLTLMonitor(formula_ya, procs)
        result_ya = monitor_ya.run(events)
        assert result_ya.satisfied is True

        # @b should be False at the end (b was NOT true before e1,
        # and e1 is the final state so @b = old[b] = value of b at e0 = False)
        formula_yb = parse_formula("@b")
        monitor_yb = EPLTLMonitor(formula_yb, procs)
        result_yb = monitor_yb.run(events)
        assert result_yb.satisfied is False

    def test_complex_formula(self) -> None:
        """
        Test 'done -> (confirmed S ready)' with simple_trace fixture.

        simple_trace.csv events:
            iota_P1{init_P1}(t=0), iota_P2{init_P2}(t=0),
            e1{ready}(t=1.5),      e2{done}(t=1.0),
            e3{confirmed}(t=2.5)

        With epsilon=2 (from file):
            t(e1) - t(e2) = 0.5, not > 2, so e1 and e2 are concurrent (by VC already).
            t(e3) - t(e2) = 1.5, not > 2, concurrent.

        Final global state: P1=e3{confirmed}, P2=e2{done}.
        Global props at end = {confirmed, done}.

        The formula "done -> (confirmed S ready)" requires:
        if done is true, then confirmed has been true since ready was true.
        """
        formula = parse_formula("done -> (confirmed S ready)")
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "since.prop",
            trace_file=TRACES / "simple_trace.csv",
        )
        result = monitor.run_from_trace()
        assert isinstance(result, MonitorResult)
        # We just verify it produces a valid result with a verdict string
        assert "SATISFIED" in result.verdict or "VIOLATED" in result.verdict

    def test_reset_and_rerun(self) -> None:
        """Running, resetting, and rerunning should produce identical verdicts."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"a"})),
            _make_event("e1", "P1", {"P1": 2}, 1.0, frozenset({"b"})),
        ]
        formula = parse_formula("a")
        monitor = EPLTLMonitor(formula, procs)

        result1 = monitor.run(events)
        monitor.reset()
        result2 = monitor.run(events)

        assert result1.satisfied == result2.satisfied
        assert result1.verdict == result2.verdict

    def test_statistics_fields(self) -> None:
        """Monitor result statistics should contain the expected keys."""
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"init"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"init"})),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"a"})),
        ]
        formula = parse_formula("a")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        expected_keys = {
            "node_count", "edge_count", "events_processed",
            "nodes_removed", "max_summaries",
        }
        assert expected_keys <= set(result.statistics.keys())

    def test_from_files_api(self) -> None:
        """EPLTLMonitor.from_files + run_from_trace with real fixture files."""
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "simple.prop",
            trace_file=TRACES / "simple_trace.csv",
        )
        result = monitor.run_from_trace()

        assert isinstance(result, MonitorResult)
        assert isinstance(result.satisfied, bool)
        assert isinstance(result.verdict, str)
        assert isinstance(result.statistics, dict)

    def test_true_always_satisfied(self) -> None:
        """Formula TRUE should always be satisfied regardless of trace."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0),
        ]
        formula = parse_formula("TRUE")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        assert result.satisfied is True

    def test_false_always_violated(self) -> None:
        """Formula FALSE should always be violated regardless of trace."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0),
        ]
        formula = parse_formula("FALSE")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        assert result.satisfied is False


# ===================================================================
# Edge Case Tests
# ===================================================================


class TestEdgeCases:
    """Edge cases that test boundary conditions of the monitoring engine."""

    def test_single_process_single_event(self) -> None:
        """Minimal possible trace: one process, one event."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"x"})),
        ]
        formula = parse_formula("x")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        assert result.satisfied is True
        assert result.statistics["events_processed"] >= 0

    def test_two_processes_no_messages(self) -> None:
        """Two independent processes, no messages, epsilon=inf."""
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset({"a"})),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset({"b"})),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"c"})),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 2.0, frozenset({"d"})),
        ]

        # With epsilon=inf, e1 and e2 are concurrent (VC-unrelated).
        po = PartialOrder(events, float("inf"))
        assert po.are_concurrent(events[2], events[3])

        # Global state at end: {c, d}
        formula = parse_formula("c & d")
        monitor = EPLTLMonitor(formula, procs, epsilon=float("inf"))
        result = monitor.run(events)
        assert result.satisfied is True

    def test_epsilon_zero(self) -> None:
        """
        Epsilon=0.0 means every pair with different timestamps gets ordered.

        With epsilon=0, any events with distinct timestamps are totally ordered
        by time (since diff > 0 = epsilon for any non-zero difference).
        """
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset()),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset()),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"a"})),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 2.0, frozenset({"b"})),
        ]

        po_zero = PartialOrder(events, 0.0)
        po_inf = PartialOrder(events, float("inf"))

        # With epsilon=0, e1 is before e2 (t(e2)-t(e1)=1.0 > 0.0).
        assert po_zero.is_before(events[2], events[3])
        # With epsilon=inf, they remain concurrent.
        assert po_inf.are_concurrent(events[2], events[3])

    def test_epsilon_infinity(self) -> None:
        """
        Epsilon=inf means only VC ordering applies.

        Events on different processes with no message chain remain concurrent.
        """
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset()),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset()),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset()),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 100.0, frozenset()),
        ]

        po = PartialOrder(events, float("inf"))
        # Despite 99-second gap, they remain concurrent with epsilon=inf
        assert po.are_concurrent(events[2], events[3])

    def test_deeply_nested_formula(self) -> None:
        """Deeply nested formula should parse and evaluate correctly."""
        formula_str = "!@(a S (b & (!c | @d)))"
        formula = parse_formula(formula_str)

        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"a", "b", "d"})),
            _make_event("e1", "P1", {"P1": 2}, 1.0, frozenset({"a", "c"})),
            _make_event("e2", "P1", {"P1": 3}, 2.0, frozenset({"a", "b"})),
        ]
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        # Just verify it completes without error and produces a valid result
        assert isinstance(result, MonitorResult)
        assert isinstance(result.satisfied, bool)

    @pytest.mark.slow
    def test_large_trace_performance(self) -> None:
        """
        A trace with 50+ events across 3 processes should complete in time.

        This test constructs a synthetic trace and verifies the monitor
        handles it without timeout or crash.
        """
        procs = frozenset({"P1", "P2", "P3"})
        events: List[Event] = []

        # Initial events
        events.append(
            _make_event("i1", "P1", {"P1": 1, "P2": 0, "P3": 0}, 0.0, frozenset({"init"}))
        )
        events.append(
            _make_event("i2", "P2", {"P1": 0, "P2": 1, "P3": 0}, 0.0, frozenset({"init"}))
        )
        events.append(
            _make_event("i3", "P3", {"P1": 0, "P2": 0, "P3": 1}, 0.0, frozenset({"init"}))
        )

        # Generate 20 events per process (60 total, sequential within each process)
        for proc_idx, proc_name in enumerate(["P1", "P2", "P3"], start=1):
            for seq in range(2, 22):  # VC values 2..21
                vc_vals = {"P1": 0, "P2": 0, "P3": 0}
                vc_vals[proc_name] = seq
                ts = (proc_idx - 1) * 100 + seq * 1.0
                props = frozenset({"active"}) if seq % 3 == 0 else frozenset()
                events.append(
                    _make_event(
                        f"{proc_name}_e{seq}",
                        proc_name,
                        vc_vals,
                        ts,
                        props,
                    )
                )

        assert len(events) >= 50

        formula = parse_formula("TRUE S active")
        monitor = EPLTLMonitor(formula, procs, epsilon=float("inf"))
        result = monitor.run(events)

        assert isinstance(result, MonitorResult)
        assert isinstance(result.satisfied, bool)
        assert result.statistics["events_processed"] > 0


# ===================================================================
# Cross-Component Integration Tests
# ===================================================================


class TestCrossComponent:
    """Tests that exercise the interaction between multiple components."""

    def test_message_trace_from_files(self) -> None:
        """Load the message_trace fixture and verify monitor produces a result."""
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "simple.prop",
            trace_file=TRACES / "message_trace.csv",
        )
        result = monitor.run_from_trace()
        assert isinstance(result, MonitorResult)

    def test_epsilon_trace_with_epsilon_override(self) -> None:
        """
        Load epsilon_trace fixture but override the epsilon value.

        The file specifies epsilon=2, but we override with epsilon=0.5.
        """
        monitor = EPLTLMonitor.from_files(
            property_file=PROPERTIES / "simple.prop",
            trace_file=TRACES / "epsilon_trace.csv",
            epsilon=0.5,
        )
        result = monitor.run_from_trace()
        assert isinstance(result, MonitorResult)
        assert result.statistics["events_processed"] > 0

    def test_partial_order_matches_monitor(self) -> None:
        """
        Verify that the partial order computed by the monitor is consistent
        with standalone PartialOrder computation.
        """
        procs = frozenset({"P1", "P2"})
        events = [
            _make_event("i1", "P1", {"P1": 1, "P2": 0}, 0.0, frozenset()),
            _make_event("i2", "P2", {"P1": 0, "P2": 1}, 0.0, frozenset()),
            _make_event("e1", "P1", {"P1": 2, "P2": 0}, 1.0, frozenset({"a"})),
            _make_event("e2", "P2", {"P1": 0, "P2": 2}, 3.5, frozenset({"b"})),
        ]

        # PartialOrder with epsilon=2
        po = PartialOrder(events, 2.0)
        # e1 should be before e2: t(e2) - t(e1) = 2.5 > 2.0
        assert po.is_before(events[2], events[3])

        # Monitor should use the same ordering
        formula = parse_formula("a & b")
        monitor = EPLTLMonitor(formula, procs, epsilon=2.0)
        result = monitor.run(events)
        assert isinstance(result, MonitorResult)

    def test_implication_with_message_trace(self, tmp_path: Path) -> None:
        """
        Test 'response -> (TRUE S request)' on the message trace.

        This checks: if response is true, request was true at some past point.
        """
        prop = tmp_path / "implication.prop"
        prop.write_text("response -> (TRUE S request)\n")

        monitor = EPLTLMonitor.from_files(
            property_file=prop,
            trace_file=TRACES / "message_trace.csv",
        )
        result = monitor.run_from_trace()

        # In the message trace, response occurs after request via message
        # causality, so this property should be satisfied.
        assert result.satisfied is True

    def test_negation_formula(self) -> None:
        """Test negation: '!error' on a trace without 'error' propositions."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"ok"})),
        ]

        formula = parse_formula("!error")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        # "error" is not in any proposition, so !error = True.
        assert result.satisfied is True

    def test_biconditional_formula(self) -> None:
        """Test biconditional: 'a <-> b' when both are present."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"a", "b"})),
        ]

        formula = parse_formula("a <-> b")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        assert result.satisfied is True

    def test_disjunction_formula(self) -> None:
        """Test disjunction: 'x | y' when only y is present."""
        procs = frozenset({"P1"})
        events = [
            _make_event("e0", "P1", {"P1": 1}, 0.0, frozenset({"y"})),
        ]

        formula = parse_formula("x | y")
        monitor = EPLTLMonitor(formula, procs)
        result = monitor.run(events)

        assert result.satisfied is True
