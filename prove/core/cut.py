"""
Cut representation for consistent global states.

A cut is a non-empty, history-closed subset of events that represents
a consistent global state of the distributed system. History-closed
means: if f is in the cut and e precedes f, then e is also in the cut.

The cut uses a PartialOrder to determine the ordering relation,
which includes both vector clock and epsilon-based ordering.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional

from prove.core.event import Event
from prove.core.frontier import Frontier
from prove.core.partial_order import PartialOrder


class Cut:
    """
    A cut is a non-empty, history-closed subset of events.

    History-closed means: if f in Cut and e < f, then e in Cut.
    A cut represents a consistent global state of the system.

    Attributes:
        _events: Frozenset of events in the cut.
        _partial_order: Reference to the complete partial order.
    """

    __slots__ = ("_events", "_partial_order")

    def __init__(
        self, events: Iterable[Event], partial_order: PartialOrder
    ) -> None:
        """
        Initialize cut with given events and partial order.

        Args:
            events: The events forming this cut.
            partial_order: The complete partial order over all events.
        """
        self._events: frozenset[Event] = frozenset(events)
        self._partial_order: PartialOrder = partial_order

    def is_history_closed(self) -> bool:
        """
        Verify the cut is history-closed w.r.t. the partial order.

        A cut is history-closed if for every event f in the cut,
        all predecessors of f are also in the cut.

        Returns:
            True if the cut is history-closed.
        """
        for event in self._events:
            preds = self._partial_order.predecessors(event)
            if not preds <= self._events:
                return False
        return True

    def add_event(self, event: Event) -> Cut:
        """
        Return a new cut with the given event added.

        This follows the immutable pattern: the original cut is unchanged.

        Args:
            event: The event to add.

        Returns:
            A new Cut containing all existing events plus the new one.
        """
        return Cut(self._events | {event}, self._partial_order)

    def contains(self, event: Event) -> bool:
        """
        Check if event is in this cut.

        Args:
            event: The event to check.

        Returns:
            True if the event is in the cut.
        """
        return event in self._events

    def maximal_event_for_process(self, process: str) -> Optional[Event]:
        """
        Return the maximal event for the given process in this cut.

        The maximal event is the one with no successors (within the cut)
        on the same process.

        Args:
            process: The process identifier.

        Returns:
            The maximal event for the process, or None if the process
            has no events in this cut.
        """
        process_events = [e for e in self._events if e.process == process]
        if not process_events:
            return None

        # The maximal event is the one that no other event in the cut succeeds
        for candidate in process_events:
            is_maximal = True
            for other in process_events:
                if other is not candidate and self._partial_order.is_before(
                    candidate, other
                ):
                    is_maximal = False
                    break
            if is_maximal:
                return candidate

        # Should not reach here if events are properly ordered
        return process_events[0]  # pragma: no cover

    def frontier(self, processes: FrozenSet[str]) -> Frontier:
        """
        Compute the frontier of this cut.

        The frontier maps each process to its maximal event in the cut.

        Args:
            processes: The set of all process identifiers.

        Returns:
            A Frontier containing the maximal event per process.
        """
        mapping: dict[str, Event] = {}
        for process in processes:
            maximal = self.maximal_event_for_process(process)
            if maximal is not None:
                mapping[process] = maximal
        return Frontier.from_mapping(mapping)

    def __contains__(self, event: object) -> bool:
        """Support 'event in cut' syntax."""
        return event in self._events

    def __len__(self) -> int:
        """Number of events in the cut."""
        return len(self._events)

    def __repr__(self) -> str:
        eids = sorted(e.eid for e in self._events)
        return f"Cut({{{', '.join(eids)}}})"
