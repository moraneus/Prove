"""
Frontier representation for compact global states.

A frontier maps each process to its maximal event in a cut, providing
a compact representation of a global state. Two cuts with the same
frontier represent the same global state.

The frontier always contains exactly one event per process (|F| = |P|).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, FrozenSet, Iterable, Set

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
        Paper V2 Section 4 enablement predicate (the parts checkable from
        the frontier alone — Conditions 1 and 3).

        An event ``e`` is enabled from frontier ``F`` when:

        - **Condition 1 (history-closure):** every predecessor of ``e``
          in the complete partial order ``≺`` is contained in the cut
          represented by ``F``. Verified by :meth:`_covers_predecessors`.

        - **Condition 3 (timing):** no event ``e' ∉ F`` such that
          ``t(e) - t(e') > ε``. Subsumed by Condition 1 here because
          the complete partial order already includes ε-based ordering
          (see :class:`PartialOrder`).

        **Condition 2 (receive enablement,** ``Γ(sender, receiver) > 0``
        **for receive events)** is *not* checked here — the message-queue
        state lives in :class:`SlidingWindowGraph`, which performs that
        check before calling into the frontier.

        Args:
            event: The event to check.
            partial_order: The complete partial order.

        Returns:
            True if Conditions 1 and 3 hold for ``event`` from this frontier.
        """
        return self._covers_predecessors(event, partial_order)

    def _covers_predecessors(
        self,
        event: Event,
        partial_order: "PartialOrder",  # noqa: F821
    ) -> bool:
        """Paper V2 Condition 1: every predecessor of *event* is in the cut.

        For each process ``p``, the latest event on ``p`` that is a
        predecessor of ``event`` must be at-or-before this frontier's
        event for ``p``.
        """
        # Special case for event's own process: the frontier entry must
        # be a strict predecessor of *event* in the partial order.
        own = self.process_to_event.get(event.process)
        if own is not None and own != event:
            if not partial_order.is_before(own, event):
                return False
        # Generic case: every predecessor (on any process) must be
        # at-or-before its process's frontier event.
        for pred in partial_order.predecessors(event):
            f_event = self.process_to_event.get(pred.process)
            if f_event is None:
                return False
            if pred == f_event:
                continue
            if not partial_order.is_before(pred, f_event):
                return False
        return True

    def is_event_enabled_fast(self, event: Event) -> bool:
        """O(k) variant of :meth:`is_event_enabled` for topological processing.

        When this frontier is the *maximal* one and ``event`` is the next
        event in a valid topological linearization of the complete partial
        order, paper V2 Conditions 1 and 3 are equivalent to a constant-
        time-per-process vector-clock check:

        - For ``p == pr(event)``: ``VC(F(p))[p] == VC(event)[p] - 1``
          (frontier is the immediate predecessor of ``event`` on its process).
        - For ``p != pr(event)``: ``VC(F(p))[p] >= VC(event)[p]``
          (frontier has caught up to ``event``'s view of process ``p``).

        ε-based predecessor coverage (Condition 3) is implied because
        topological order already places every ε-predecessor of ``event``
        in the cut. This avoids the O(N) :meth:`PartialOrder.predecessors`
        enumeration in :meth:`is_event_enabled`, keeping graph construction
        linear in the trace length.

        Args:
            event: The event to check. Must be the next event in a
                topological linearization for the result to be meaningful.

        Returns:
            True if Conditions 1 and 3 hold under the topological-order
            precondition.
        """
        event_vc = event.vector_clock._clock
        for proc, f_event in self.process_to_event.items():
            f_vc = f_event.vector_clock._clock
            if proc == event.process:
                if f_vc.get(proc, 0) != event_vc.get(proc, 0) - 1:
                    return False
            else:
                if f_vc.get(proc, 0) < event_vc.get(proc, 0):
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
