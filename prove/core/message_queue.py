"""
Message queue handler for asynchronous message passing.

Tracks pending (sent but not yet received) messages between each pair
of processes, enforcing FIFO ordering and enabling receive-event
enablement checks.

For each ordered pair ``(sender, receiver)``, the counter
``Γ(sender, receiver)`` records the number of messages in transit.
A receive event for ``(sender, receiver)`` is only enabled when
``Γ(sender, receiver) > 0``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


class MessageQueueHandler:
    """
    Immutable tracker for pending asynchronous messages between process pairs.

    FIFO ordering is assumed: messages between a given pair of processes
    are received in the order they were sent.

    All mutation methods (``send``, ``receive``) return **new** instances,
    leaving the original handler unchanged.

    Attributes:
        queues: Internal mapping from ``(sender, receiver)`` to the
            number of pending messages.
    """

    __slots__ = ("_queues", "_processes")

    def __init__(self, processes: Iterable[str]) -> None:
        """
        Initialise with all process pairs having zero pending messages.

        Args:
            processes: Set of process identifiers in the system.
        """
        self._processes: frozenset[str] = frozenset(processes)
        self._queues: Dict[Tuple[str, str], int] = {
            (s, r): 0 for s in self._processes for r in self._processes
        }

    # Private constructor for immutable copies
    @classmethod
    def _from_raw(
        cls,
        processes: frozenset[str],
        queues: Dict[Tuple[str, str], int],
    ) -> MessageQueueHandler:
        obj = object.__new__(cls)
        obj._processes = processes
        obj._queues = queues
        return obj

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #

    def send(self, sender: str, receiver: str) -> MessageQueueHandler:
        """
        Record a message sent from *sender* to *receiver*.

        Increments ``Γ(sender, receiver)`` by 1.

        Returns:
            A **new** MessageQueueHandler with the updated count.
        """
        new_queues = dict(self._queues)
        new_queues[(sender, receiver)] = new_queues.get((sender, receiver), 0) + 1
        return MessageQueueHandler._from_raw(self._processes, new_queues)

    def receive(self, sender: str, receiver: str) -> MessageQueueHandler:
        """
        Record a message received by *receiver* from *sender*.

        Decrements ``Γ(sender, receiver)`` by 1.

        Raises:
            ValueError: If there is no pending message from *sender* to
                *receiver* (``Γ(sender, receiver) == 0``).

        Returns:
            A **new** MessageQueueHandler with the updated count.
        """
        current = self._queues.get((sender, receiver), 0)
        if current <= 0:
            raise ValueError(
                f"No pending message from '{sender}' to '{receiver}' " f"(Γ = {current})"
            )
        new_queues = dict(self._queues)
        new_queues[(sender, receiver)] = current - 1
        return MessageQueueHandler._from_raw(self._processes, new_queues)

    def can_receive(self, sender: str, receiver: str) -> bool:
        """
        True when *receiver* can receive a message from *sender*.

        Equivalent to ``Γ(sender, receiver) > 0``.
        """
        return self._queues.get((sender, receiver), 0) > 0

    def pending_count(self, sender: str, receiver: str) -> int:
        """Return the number of pending messages from *sender* to *receiver*."""
        return self._queues.get((sender, receiver), 0)

    def copy(self) -> MessageQueueHandler:
        """Return an independent copy of this handler."""
        return MessageQueueHandler._from_raw(self._processes, dict(self._queues))

    def __repr__(self) -> str:
        non_zero = {k: v for k, v in self._queues.items() if v > 0}
        return f"MessageQueueHandler({non_zero})"
