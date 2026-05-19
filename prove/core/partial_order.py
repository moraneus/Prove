"""
Complete partial order computation over events.

Combines vector clock ordering and epsilon-based timestamp ordering.
The partial order is queried on-demand using direct VC and timestamp
checks, avoiding O(N²) precomputation of the transitive closure.

Key insight: for any events e1, e2, e1 ≺ e2 in the transitive closure
iff VC(e1) < VC(e2) OR t(e2) - t(e1) > epsilon. This holds because
timestamps are monotonic along all ordering edges (both VC and epsilon),
so any transitive chain results in either a direct VC or epsilon ordering.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, Iterable, List, Set

from prove.core.clock_drift import ClockDriftHandler
from prove.core.event import Event


class PartialOrder:
    """
    Complete partial order ``≺`` over a set of events.

    Combines two sources of ordering:

    1. **Vector clock ordering**: ``e ≺ f`` when ``VC(e) < VC(f)``
    2. **Epsilon-based ordering**: ``e ≺ f`` when ``t(f) − t(e) > ε``

    Order queries use direct VC and timestamp checks (O(k) per query
    where k is the number of processes), eliminating the need for
    O(N²) precomputation.

    Attributes:
        events: Tuple of all events (in insertion order).
        epsilon: Maximum clock skew parameter.
    """

    def __init__(self, events: Iterable[Event], epsilon: float) -> None:
        """
        Initialize the partial order.

        Args:
            events: All events in the execution.
            epsilon: Maximum clock skew between processes.
        """
        self._events: tuple[Event, ...] = tuple(events)
        self.epsilon: float = epsilon
        self._drift = ClockDriftHandler(epsilon)

        # Build direct successor edges for topological sort (sparse DAG)
        self._direct_succ: Dict[Event, Set[Event]] = defaultdict(set)
        self._build_direct_edges()

    # ------------------------------------------------------------------ #
    # Internal computation
    # ------------------------------------------------------------------ #

    def _build_direct_edges(self) -> None:
        """Build sparse direct edges for topological sort.

        Intra-process edges: consecutive events on the same process.
        Cross-process edges: only the minimal necessary edges from VC
        ordering and epsilon ordering.
        """
        # Group events by process, sort by their own VC component
        by_process: Dict[str, list[Event]] = defaultdict(list)
        for e in self._events:
            by_process[e.process].append(e)

        for proc, evts in by_process.items():
            evts.sort(key=lambda e: e.vector_clock._clock[proc])
            for i in range(len(evts) - 1):
                self._direct_succ[evts[i]].add(evts[i + 1])

        # Cross-process VC edges (message causality):
        # For receive events, add edge from the matching send predecessor.
        # We detect this via VC: if event f on process q has VC[p] > 0,
        # find the event on process p with VC[p] == f.VC[p] as a predecessor.
        event_by_proc_seq: Dict[str, Dict[int, Event]] = defaultdict(dict)
        for e in self._events:
            proc = e.process
            seq = e.vector_clock._clock[proc]
            event_by_proc_seq[proc][seq] = e

        for e in self._events:
            proc_e = e.process
            for proc, seq in e.vector_clock._clock.items():
                if proc == proc_e or seq == 0:
                    continue
                pred = event_by_proc_seq.get(proc, {}).get(seq)
                if pred is not None and pred is not e:
                    # pred on process `proc` with VC[proc]=seq is a predecessor of e
                    # But only add as direct edge if pred ≺ e
                    if self.is_before(pred, e):
                        self._direct_succ[pred].add(e)

        # Epsilon edges: for each pair of process groups, find cross-process
        # orderings from timestamps. Sort all events by timestamp and use
        # a sweep to find epsilon edges between different processes.
        if self.epsilon != float("inf"):
            sorted_events = sorted(self._events, key=lambda e: e.timestamp)
            n = len(sorted_events)
            for i in range(n):
                ei = sorted_events[i]
                for j in range(i + 1, n):
                    ej = sorted_events[j]
                    if ej.timestamp - ei.timestamp <= self.epsilon:
                        continue
                    # ei ≺ ej by epsilon; add direct edge if they're
                    # on different processes and not already VC-ordered
                    if ei.process != ej.process and not ei.causally_before(ej):
                        self._direct_succ[ei].add(ej)
                    break  # All further events have even larger timestamp gap

    # ------------------------------------------------------------------ #
    # Order queries
    # ------------------------------------------------------------------ #

    def is_before(self, e1: Event, e2: Event) -> bool:
        """True when ``e1 ≺ e2`` in the complete partial order.

        Checks direct VC ordering or epsilon-based timestamp ordering.
        Due to timestamp monotonicity along causal chains, this correctly
        captures the transitive closure without precomputation.
        """
        if e1 is e2:
            return False
        return e1.causally_before(e2) or self._drift.is_definitely_before(e1, e2)

    def is_after(self, e1: Event, e2: Event) -> bool:
        """True when ``e1 ≻ e2`` (i.e. ``e2 ≺ e1``)."""
        return self.is_before(e2, e1)

    def are_concurrent(self, e1: Event, e2: Event) -> bool:
        """True when neither ``e1 ≺ e2`` nor ``e2 ≺ e1``."""
        return not self.is_before(e1, e2) and not self.is_before(e2, e1)

    def are_independent(self, e1: Event, e2: Event) -> bool:
        """True when ``e1`` and ``e2`` belong to different processes."""
        return e1.is_independent_of(e2)

    # ------------------------------------------------------------------ #
    # Predecessor / successor queries
    # ------------------------------------------------------------------ #

    def predecessors(self, event: Event) -> FrozenSet[Event]:
        """Return all events ``e`` where ``e ≺ event``."""
        return frozenset(e for e in self._events if e is not event and self.is_before(e, event))

    def immediate_predecessors(self, event: Event) -> FrozenSet[Event]:
        """
        Return events ``e`` where ``e ≺ event`` and no ``f`` exists with
        ``e ≺ f ≺ event``.
        """
        preds = self.predecessors(event)
        immediate: set[Event] = set()
        for p in preds:
            if not any(
                self.is_before(p, other) and other in preds for other in preds if other is not p
            ):
                immediate.add(p)
        return frozenset(immediate)

    def successors(self, event: Event) -> FrozenSet[Event]:
        """Return all events ``e`` where ``event ≺ e``."""
        return frozenset(e for e in self._events if e is not event and self.is_before(event, e))

    # ------------------------------------------------------------------ #
    # Linearisation
    # ------------------------------------------------------------------ #

    def topological_sort(self) -> List[Event]:
        """
        Return a valid linearisation of the partial order.

        Uses Kahn's algorithm over the sparse direct edge graph.
        Ties are broken by timestamp then by event ID for determinism.
        """
        in_degree: Dict[Event, int] = {e: 0 for e in self._events}
        for e in self._events:
            for s in self._direct_succ.get(e, set()):
                in_degree[s] += 1

        ready = sorted(
            [e for e in self._events if in_degree[e] == 0],
            key=lambda ev: (ev.timestamp, ev.eid),
        )
        result: List[Event] = []

        while ready:
            current = ready.pop(0)
            result.append(current)
            for s in sorted(
                self._direct_succ.get(current, set()),
                key=lambda ev: (ev.timestamp, ev.eid),
            ):
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    ready.append(s)
                    ready.sort(key=lambda ev: (ev.timestamp, ev.eid))

        return result

    def is_valid_linearization(self, sequence: List[Event]) -> bool:
        """
        Check whether *sequence* is a valid linearisation.

        A sequence is valid if for every pair ``(i, j)`` with ``i < j``,
        ``sequence[j]`` does not precede ``sequence[i]`` in the partial order.
        """
        position = {e: i for i, e in enumerate(sequence)}
        for i, e in enumerate(sequence):
            for j in range(i + 1, len(sequence)):
                if self.is_before(sequence[j], e):
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Minimal / enabled events
    # ------------------------------------------------------------------ #

    def get_minimal_events(self) -> FrozenSet[Event]:
        """Return events with no predecessors (initial events)."""
        minimals: list[Event] = []
        for e in self._events:
            is_minimal = True
            for f in self._events:
                if f is not e and self.is_before(f, e):
                    is_minimal = False
                    break
            if is_minimal:
                minimals.append(e)
        return frozenset(minimals)

    def get_enabled_events(self, processed: Set[Event]) -> Set[Event]:
        """
        Return events that can be processed next.

        An event is enabled if all its predecessors have been processed
        and it has not been processed itself.
        """
        enabled: Set[Event] = set()
        for e in self._events:
            if e in processed:
                continue
            all_preds_done = True
            for f in self._events:
                if f is not e and f not in processed and self.is_before(f, e):
                    all_preds_done = False
                    break
            if all_preds_done:
                enabled.add(e)
        return enabled

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def events(self) -> tuple[Event, ...]:
        """All events in the partial order."""
        return self._events

    def __repr__(self) -> str:
        return f"PartialOrder({len(self._events)} events, ε={self.epsilon})"
