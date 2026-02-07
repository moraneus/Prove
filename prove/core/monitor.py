"""
Main monitor orchestration for EPLTL verification.

Coordinates event ordering, sliding window graph construction,
EPLTL summary computation, and verdict reporting for verifying
temporal properties over partial order executions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from prove.core.event import Event
from prove.core.partial_order import PartialOrder
from prove.core.sliding_window import SlidingWindowGraph
from prove.parser.ast_nodes import Formula
from prove.parser.formula import parse_formula
from prove.utils.logger import LogLevel, MonitorLogger
from prove.utils.trace_reader import TraceReader


@dataclass
class MonitorResult:
    """
    Result of monitoring a trace against a property.

    Attributes:
        satisfied: Whether the property is satisfied.
        verdict: Human-readable verdict string.
        witness: Optional witness path (list of events).
        statistics: Dictionary of monitoring statistics.
    """

    satisfied: bool
    verdict: str
    witness: Optional[List[Event]]
    statistics: Dict[str, Any]


class EPLTLMonitor:
    """
    Main monitor for EPLTL properties over partial order executions.

    Orchestrates:
    1. Computing the complete partial order (offline)
    2. Identifying initial events and processing order
    3. Sliding window graph construction
    4. EPLTL summary computation
    5. Verdict reporting

    Attributes:
        formula: The EPLTL formula to verify.
        processes: Set of process IDs.
        epsilon: Maximum clock skew.
        logger: Logger for output.
    """

    def __init__(
        self,
        formula: Formula,
        processes: Iterable[str],
        epsilon: float = float("inf"),
        logger: Optional[MonitorLogger] = None,
    ) -> None:
        """
        Initialize the monitor.

        Args:
            formula: The EPLTL formula to verify.
            processes: Set of process IDs in the system.
            epsilon: Maximum clock skew (default: infinity).
            logger: Optional logger for debug output.
        """
        self.formula: Formula = formula
        self.processes: frozenset[str] = frozenset(processes)
        self.epsilon: float = epsilon
        self.logger: MonitorLogger = logger or MonitorLogger(LogLevel.SILENT)

        # Will be initialized when events are provided
        self._graph: Optional[SlidingWindowGraph] = None
        self._trace_data: Optional[object] = None

        # For from_files: store loaded trace data
        self._loaded_events: Optional[List[Event]] = None
        self._loaded_partial_order: Optional[PartialOrder] = None

    def run(self, events: Iterable[Event]) -> MonitorResult:
        """
        Run monitor on a sequence of events.

        Computes the complete partial order, initializes the sliding
        window graph, processes events in topological order, and
        returns the final verdict.

        Args:
            events: All events in the execution.

        Returns:
            MonitorResult with verdict and statistics.
        """
        event_list = list(events)
        if not event_list:
            return MonitorResult(
                satisfied=False,
                verdict="VIOLATED: No events to process",
                witness=None,
                statistics={"events_processed": 0},
            )

        po = PartialOrder(event_list, self.epsilon)
        return self._execute(event_list, po)

    def finalize(self) -> MonitorResult:
        """
        Finalize monitoring and return result.

        Call this after all events have been processed.

        Returns:
            MonitorResult with verdict and statistics.
        """
        if self._graph is None:
            return MonitorResult(
                satisfied=False,
                verdict="VIOLATED: Monitor not initialized",
                witness=None,
                statistics={"events_processed": 0},
            )

        is_satisfied, witness = self._graph.get_verdict(self.formula)
        stats = self._graph.get_statistics()

        # Log the maximal frontier
        max_node = self._graph.nodes[self._graph.maximal_node_id]
        frontier_map = {
            e.process: e.eid
            for e in max_node.frontier.events
        }
        self.logger.frontier_info(frontier_map)

        if is_satisfied:
            verdict = "SATISFIED: Property holds for at least one linearization"
            self.logger.verdict_satisfied()
        else:
            verdict = "VIOLATED: Property does not hold for any linearization"
            self.logger.verdict_violated()

        self.logger.statistics(stats)

        return MonitorResult(
            satisfied=is_satisfied,
            verdict=verdict,
            witness=witness,
            statistics=stats,
        )

    def reset(self) -> None:
        """Reset monitor to initial state."""
        self._graph = None

    @classmethod
    def from_files(
        cls,
        property_file: Path,
        trace_file: Path,
        epsilon: Optional[float] = None,
        logger: Optional[MonitorLogger] = None,
    ) -> "EPLTLMonitor":
        """
        Create monitor from property and trace files.

        Args:
            property_file: Path to EPLTL formula file (.prop).
            trace_file: Path to trace CSV file.
            epsilon: Override epsilon (if not provided, uses file directive).
            logger: Optional logger.

        Returns:
            Configured EPLTLMonitor ready to run.
        """
        # Parse property
        prop_text = property_file.read_text().strip()
        # Remove comment lines
        formula_lines = [
            line for line in prop_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        formula_str = " ".join(formula_lines)
        formula = parse_formula(formula_str)

        # Read trace
        reader = TraceReader(trace_file)
        trace_data = reader.read_all(epsilon=epsilon)

        eps = trace_data.metadata.epsilon if trace_data.metadata.epsilon is not None else float("inf")
        if epsilon is not None:
            eps = epsilon

        monitor = cls(
            formula=formula,
            processes=trace_data.metadata.processes,
            epsilon=eps,
            logger=logger,
        )
        monitor._loaded_events = trace_data.events
        monitor._loaded_partial_order = trace_data.partial_order

        return monitor

    def run_from_trace(self) -> MonitorResult:
        """
        Run monitor using previously loaded trace data (from from_files).

        Returns:
            MonitorResult with verdict and statistics.
        """
        if self._loaded_events is None:
            raise RuntimeError(
                "No trace loaded. Use from_files() to create monitor."
            )
        return self._run_with_partial_order(
            self._loaded_events, self._loaded_partial_order
        )

    def _run_with_partial_order(
        self,
        events: List[Event],
        partial_order: Optional[PartialOrder],
    ) -> MonitorResult:
        """
        Run monitor with a pre-computed partial order.

        Args:
            events: All events.
            partial_order: Pre-computed partial order (or None to compute).

        Returns:
            MonitorResult.
        """
        if not events:
            return MonitorResult(
                satisfied=False,
                verdict="VIOLATED: No events to process",
                witness=None,
                statistics={"events_processed": 0},
            )

        po = partial_order or PartialOrder(events, self.epsilon)
        return self._execute(events, po)

    def _execute(
        self,
        events: List[Event],
        po: PartialOrder,
    ) -> MonitorResult:
        """
        Core execution: log, build graph, process events, finalize.

        Args:
            events: All events.
            po: Complete partial order.

        Returns:
            MonitorResult.
        """
        procs = sorted(self.processes)
        eps_str = "inf" if self.epsilon == float("inf") else str(self.epsilon)
        self.logger.info(
            f"Loaded {len(events)} events from {len(procs)} processes"
        )
        self.logger.info(f"Processes: {', '.join(procs)}")
        self.logger.info(f"Epsilon: {eps_str}")
        self.logger.info(f"Verifying formula: {self.formula}")

        # Identify initial events
        initial_events = self._find_initial_events(events, po)

        # Initialize graph
        self._graph = SlidingWindowGraph(
            processes=self.processes,
            initial_events=initial_events,
            formula=self.formula,
            partial_order=po,
        )

        # Log initial events
        for event in initial_events.values():
            self.logger.event_info(
                event.eid, event.process, event.propositions,
            )

        # Process non-initial events in topological order
        processing_order = po.topological_sort()
        initial_set = set(initial_events.values())
        non_initial = [e for e in processing_order if e not in initial_set]

        for event in non_initial:
            self.logger.event_info(
                event.eid, event.process, event.propositions,
            )
            self.logger.event_processed(
                event.eid, len(self._graph.nodes)
            )
            self._graph.process_event(event)

        return self.finalize()

    def _find_initial_events(
        self, events: List[Event], po: PartialOrder,
    ) -> Dict[str, Event]:
        """
        Find the initial event for each process.

        Initial events are minimal in the partial order (no predecessors).
        Each process must have exactly one initial event.

        Args:
            events: All events.
            po: The partial order.

        Returns:
            Dictionary mapping process ID to its initial event.
        """
        minimal = po.get_minimal_events()
        initial: Dict[str, Event] = {}

        for event in minimal:
            if event.process not in initial:
                initial[event.process] = event

        # If some processes don't have minimal events found,
        # use the first event for each process
        for event in events:
            if event.process not in initial:
                initial[event.process] = event

        return initial
