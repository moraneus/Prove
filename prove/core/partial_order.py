"""
Complete partial order computation over events.

Combines vector clock ordering and epsilon-based timestamp ordering,
then computes the transitive closure to produce the full partial order
relation used by the sliding window algorithm.

The partial order is computed **once** at initialisation (offline mode)
and then queried during graph construction and summary propagation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, FrozenSet, Iterable, List, Set

from prove.core.clock_drift import ClockDriftHandler
from prove.core.event import Event


class PartialOrder:
    """
    Complete partial order ``≺`` over a set of events.

    Combines two sources of ordering:

    1. **Vector clock ordering**: ``e ≺ f`` when ``VC(e) < VC(f)``
    2. **Epsilon-based ordering**: ``e ≺ f`` when ``t(f) − t(e) > ε``

    After collecting all direct orderings, the transitive closure is
    computed so that ``is_before`` answers in O(1).

    Attributes:
        events: Tuple of all events (in insertion order).
        epsilon: Maximum clock skew parameter.
    """

    def __init__(self, events: Iterable[Event], epsilon: float) -> None:
        """
        Compute the complete partial order.

        Args:
            events: All events in the execution.
            epsilon: Maximum clock skew between processes.
        """
        self._events: tuple[Event, ...] = tuple(events)
        self.epsilon: float = epsilon
        self._drift = ClockDriftHandler(epsilon)

        # Forward adjacency: event → set of direct successors
        self._direct_succ: Dict[Event, Set[Event]] = defaultdict(set)
        # The full transitive closure: event → all events it precedes
        self._succ_closure: Dict[Event, Set[Event]] = {}
        # Reverse closure: event → all events that precede it
        self._pred_closure: Dict[Event, Set[Event]] = {}

        self._compute()

    # ------------------------------------------------------------------ #
    # Internal computation
    # ------------------------------------------------------------------ #

    def _compute(self) -> None:
        """Build the complete partial order with transitive closure."""
        events = self._events

        # 1. Collect direct orderings from VC and epsilon
        for i, ei in enumerate(events):
            for j, ej in enumerate(events):
                if i == j:
                    continue
                if ei.causally_before(ej):
                    self._direct_succ[ei].add(ej)
                elif self._drift.is_definitely_before(ei, ej):
                    self._direct_succ[ei].add(ej)

        # 2. Compute transitive closure via BFS from each node
        for e in events:
            reachable: Set[Event] = set()
            queue = deque(self._direct_succ.get(e, set()))
            while queue:
                curr = queue.popleft()
                if curr in reachable:
                    continue
                reachable.add(curr)
                queue.extend(self._direct_succ.get(curr, set()))
            self._succ_closure[e] = reachable

        # 3. Build reverse closure
        for e in events:
            self._pred_closure[e] = set()
        for e in events:
            for s in self._succ_closure[e]:
                self._pred_closure[s].add(e)

    # ------------------------------------------------------------------ #
    # Order queries
    # ------------------------------------------------------------------ #

    def is_before(self, e1: Event, e2: Event) -> bool:
        """True when ``e1 ≺ e2`` in the complete partial order."""
        return e2 in self._succ_closure.get(e1, set())

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
        return frozenset(self._pred_closure.get(event, set()))

    def immediate_predecessors(self, event: Event) -> FrozenSet[Event]:
        """
        Return events ``e`` where ``e ≺ event`` and no ``f`` exists with
        ``e ≺ f ≺ event``.
        """
        preds = self._pred_closure.get(event, set())
        immediate: Set[Event] = set()
        for p in preds:
            # p is immediate if no other predecessor of event is a successor of p
            if not any(
                other in preds and other != p
                for other in self._succ_closure.get(p, set())
            ):
                immediate.add(p)
        return frozenset(immediate)

    def successors(self, event: Event) -> FrozenSet[Event]:
        """Return all events ``e`` where ``event ≺ e``."""
        return frozenset(self._succ_closure.get(event, set()))

    # ------------------------------------------------------------------ #
    # Linearisation
    # ------------------------------------------------------------------ #

    def topological_sort(self) -> List[Event]:
        """
        Return a valid linearisation of the partial order (Kahn's algorithm).

        The linearisation respects all orderings: if ``e ≺ f`` then
        ``e`` appears before ``f`` in the result.

        For concurrent events, ties are broken by timestamp then by event ID
        for determinism.
        """
        # In-degree map
        in_degree: Dict[Event, int] = {e: 0 for e in self._events}
        for e in self._events:
            for s in self._direct_succ.get(e, set()):
                in_degree[s] += 1

        # Start with zero in-degree events, sorted for determinism
        ready = sorted(
            [e for e in self._events if in_degree[e] == 0],
            key=lambda ev: (ev.timestamp, ev.eid),
        )
        result: List[Event] = []

        while ready:
            # Pick the first ready event (stable sort by timestamp/eid)
            current = ready.pop(0)
            result.append(current)
            for s in sorted(
                self._direct_succ.get(current, set()),
                key=lambda ev: (ev.timestamp, ev.eid),
            ):
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    # Insert in sorted position
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
        for e in self._events:
            for s in self._succ_closure.get(e, set()):
                if position.get(e, -1) >= position.get(s, -1):
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Minimal / enabled events
    # ------------------------------------------------------------------ #

    def get_minimal_events(self) -> FrozenSet[Event]:
        """Return events with no predecessors (initial events)."""
        return frozenset(
            e for e in self._events if not self._pred_closure.get(e, set())
        )

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
            preds = self._pred_closure.get(e, set())
            if preds <= processed:
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
