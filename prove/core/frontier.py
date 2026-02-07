"""
Frontier representation for compact global states.

A frontier maps each process to its maximal event in a cut, providing
a compact representation of a global state. Two cuts with the same
frontier represent the same global state.

The frontier always contains exactly one event per process (|F| = |P|),
as required by the Paper V2 invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, FrozenSet, Iterable, Optional, Set

from prove.core.event import Event


@dataclass(frozen=True)
class Frontier:
    """
    A frontier is the set of maximal events per process in a cut.

    The frontier provides a compact representation of a global state.
    Two cuts with the same frontier represent the same global state.

    Attributes:
        events: Frozenset of maximal events (one per process).
        process_to_event: Immutable mapping from process to its maximal event.
    """

    events: FrozenSet[Event]
    process_to_event: MappingProxyType

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Event]) -> Frontier:
        """
        Construct frontier from a process-to-event mapping.

        Args:
            mapping: Dictionary mapping each process to its maximal event.

        Returns:
            A new Frontier instance.
        """
        events = frozenset(mapping.values())
        return cls(
            events=events,
            process_to_event=MappingProxyType(dict(mapping)),
        )

    def global_state(self) -> FrozenSet[str]:
        """
        Return the set of propositions true in this global state.

        The global interpretation is L(F) = union of L(e) for each e in F.
        """
        props: Set[str] = set()
        for event in self.events:
            props.update(event.propositions)
        return frozenset(props)

    def successor(self, event: Event, partial_order: "PartialOrder") -> Frontier:  # noqa: F821
        """
        Compute the successor frontier after executing event.

        The new frontier replaces the maximal event on the event's process
        and removes any events that are now dominated by the new event.

        Args:
            event: The event to add. Must be enabled from this frontier.
            partial_order: The complete partial order.

        Returns:
            New Frontier with the event incorporated.
        """
        new_mapping: Dict[str, Event] = dict(self.process_to_event)
        new_mapping[event.process] = event
        return Frontier.from_mapping(new_mapping)

    def is_event_enabled(
        self,
        event: Event,
        partial_order: "PartialOrder",  # noqa: F821
    ) -> bool:
        """
        Check if event can be executed from this frontier.

        An event is enabled if all its predecessors (in the complete partial
        order) are contained in the cut that this frontier represents.

        Args:
            event: The event to check.
            partial_order: The complete partial order.

        Returns:
            True if the event is enabled from this frontier.
        """
        preds = partial_order.predecessors(event)
        # Build the set of events in the cut: all events up to and including
        # each frontier event, per process
        frontier_event = self.process_to_event.get(event.process)
        if frontier_event is not None:
            # The event's process already has a maximal event in the frontier.
            # The new event must be a direct successor of that maximal event.
            if not partial_order.is_before(frontier_event, event) and frontier_event != event:
                return False

        # All predecessors must be reachable from the frontier events
        for pred in preds:
            proc = pred.process
            f_event = self.process_to_event.get(proc)
            if f_event is None:
                return False
            # pred must be <= f_event in the partial order
            if pred != f_event and not partial_order.is_before(pred, f_event):
                if not (f_event == pred):
                    return False
        return True

    def enabled_events(
        self,
        pending_events: Iterable[Event],
        partial_order: "PartialOrder",  # noqa: F821
    ) -> Set[Event]:
        """
        Return all events enabled from this frontier.

        Args:
            pending_events: Candidate events to check.
            partial_order: The complete partial order.

        Returns:
            Set of enabled events.
        """
        return {e for e in pending_events if self.is_event_enabled(e, partial_order)}

    def __eq__(self, other: object) -> bool:
        """Check equality based on the process-to-event mapping."""
        if not isinstance(other, Frontier):
            return NotImplemented
        return self.events == other.events

    def __hash__(self) -> int:
        """Hash based on the events frozenset."""
        return hash(self.events)

    def __repr__(self) -> str:
        mapping_str = ", ".join(f"{p}: {e.eid}" for p, e in sorted(self.process_to_event.items()))
        return f"Frontier({{{mapping_str}}})"
