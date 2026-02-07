"""
Clock drift handler for epsilon-based temporal ordering.

When the timestamp difference between two events exceeds the maximum
clock skew (epsilon), a definitive causal ordering can be established
even without a direct causal connection via messages:

    If  t(f) − t(e) > ε   then  e ≺ f

This module provides the logic for determining and applying these
additional orderings during the offline partial-order computation.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

from prove.core.event import Event


class ClockDriftHandler:
    """
    Determines temporal ordering between events based on bounded clock skew.

    Given two events whose local clocks may differ by at most *epsilon*,
    if the observed timestamp gap exceeds *epsilon* we can definitively
    order them.

    Attributes:
        epsilon: Maximum clock skew between any two processes.
    """

    __slots__ = ("epsilon",)

    def __init__(self, epsilon: float) -> None:
        """
        Initialise with the maximum clock skew.

        Args:
            epsilon: Upper bound on the drift between any pair of
                process-local clocks.
        """
        self.epsilon: float = epsilon

    # ------------------------------------------------------------------ #
    # Ordering queries
    # ------------------------------------------------------------------ #

    def time_orders(self, e1: Event, e2: Event) -> Optional[bool]:
        """
        Determine if the timestamp gap imposes a definitive ordering.

        Returns:
            ``True``  if ``e1 ≺ e2`` (``t(e2) − t(e1) > ε``).
            ``False`` if ``e2 ≺ e1`` (``t(e1) − t(e2) > ε``).
            ``None``  if the gap is within *epsilon* in both directions.
        """
        diff = e2.timestamp - e1.timestamp
        if diff > self.epsilon:
            return True
        if -diff > self.epsilon:
            return False
        return None

    def is_definitely_before(self, e1: Event, e2: Event) -> bool:
        """
        True when ``e1`` is definitively before ``e2`` based on timestamps.

        Equivalent to ``t(e2) − t(e1) > epsilon``.
        """
        return (e2.timestamp - e1.timestamp) > self.epsilon

    # ------------------------------------------------------------------ #
    # Bulk partitioning
    # ------------------------------------------------------------------ #

    def filter_concurrent_events(
        self,
        events: Iterable[Event],
        reference: Event,
    ) -> Tuple[Set[Event], Set[Event], Set[Event]]:
        """
        Partition *events* relative to *reference* based on timestamps.

        Returns:
            A triple ``(definitely_before, definitely_after, possibly_concurrent)``
            where each event is placed according to whether it is
            definitively before the reference, definitively after, or
            within the epsilon window (possibly concurrent).
        """
        before: Set[Event] = set()
        after: Set[Event] = set()
        concurrent: Set[Event] = set()

        for e in events:
            order = self.time_orders(e, reference)
            if order is True:
                # e ≺ reference  (e is before reference)
                before.add(e)
            elif order is False:
                # reference ≺ e  (e is after reference)
                after.add(e)
            else:
                concurrent.add(e)

        return before, after, concurrent
