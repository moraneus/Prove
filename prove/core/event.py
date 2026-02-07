"""
Event representation for partial order executions.

Each event belongs to exactly one process (asynchronous message-passing
model per the paper), carries a Fidge-Mattern vector clock for causal
ordering, a global timestamp for epsilon-based clock-drift ordering,
and a set of propositions that hold after the event executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from prove.core.vector_clock import VectorClock

_VALID_EVENT_TYPES = frozenset({"local", "send", "receive"})


@dataclass(frozen=True)
class Event:
    """
    Immutable representation of an event in a partial order execution.

    In the asynchronous model, each event belongs to exactly one process
    (``pr: E → P``).  Message exchanges are modelled as separate send
    and receive events on different processes.

    Attributes:
        eid: Unique event identifier.
        process: The single process this event belongs to.
        vector_clock: Fidge-Mattern vector clock at time of the event.
        timestamp: Global timestamp (required for epsilon-based ordering).
        propositions: Propositions true after the event executes.
        event_type: One of ``'local'``, ``'send'``, ``'receive'``.
        source_process: For receive events, the sending process.
        target_process: For send events, the receiving process.
    """

    eid: str
    process: str
    vector_clock: VectorClock
    timestamp: float
    propositions: FrozenSet[str] = field(default_factory=frozenset)
    event_type: str = "local"
    source_process: Optional[str] = None
    target_process: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate event type and associated fields."""
        if self.event_type not in _VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {sorted(_VALID_EVENT_TYPES)}, "
                f"got '{self.event_type}'"
            )
        if self.event_type == "send" and self.target_process is None:
            raise ValueError("Send events require target_process to be set")
        if self.event_type == "receive" and self.source_process is None:
            raise ValueError("Receive events require source_process to be set")

    # ------------------------------------------------------------------ #
    # Type checks
    # ------------------------------------------------------------------ #

    def is_send(self) -> bool:
        """True if this is a message-send event."""
        return self.event_type == "send"

    def is_receive(self) -> bool:
        """True if this is a message-receive event."""
        return self.event_type == "receive"

    def is_local(self) -> bool:
        """True if this is a local (non-message) event."""
        return self.event_type == "local"

    # ------------------------------------------------------------------ #
    # Causal ordering (vector clock based)
    # ------------------------------------------------------------------ #

    def causally_before(self, other: Event) -> bool:
        """
        True when ``self ≺ other`` according to vector clocks.

        Strictly: ``VC(self) < VC(other)``.
        """
        return self.vector_clock < other.vector_clock

    def is_concurrent_with(self, other: Event) -> bool:
        """
        True when neither event causally precedes the other (by vector
        clock only — does not consider epsilon-based time ordering).
        """
        return self.vector_clock.is_concurrent_with(other.vector_clock)

    # ------------------------------------------------------------------ #
    # Independence
    # ------------------------------------------------------------------ #

    def is_independent_of(self, other: Event) -> bool:
        """
        True when the two events belong to different processes.

        In the asynchronous model ``pr(e) ≠ pr(f)`` means the events
        are independent (they do not share a process).
        """
        return self.process != other.process

    # ------------------------------------------------------------------ #
    # Time-based ordering
    # ------------------------------------------------------------------ #

    def time_before(self, other: Event, epsilon: float) -> bool:
        """
        True when ``other.timestamp - self.timestamp > epsilon``.

        This means that, given bounded clock skew *epsilon*, ``self``
        definitely happened before ``other`` in real time.
        """
        return (other.timestamp - self.timestamp) > epsilon

    def ordered_before(self, other: Event, epsilon: float) -> bool:
        """
        True when ``self`` precedes ``other`` by vector clock **or** by
        time (epsilon-bounded clock drift).
        """
        return self.causally_before(other) or self.time_before(other, epsilon)
