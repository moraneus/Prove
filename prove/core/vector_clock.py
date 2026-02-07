"""
Fidge-Mattern vector clock implementation.

Vector clocks enable determining causal relationships between events
in a distributed system by maintaining a vector of logical timestamps,
one per process. The key property is:

    e ≺ f  ⟺  VC(e) < VC(f)

where < denotes the strict componentwise ordering.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


class VectorClock:
    """
    Fidge-Mattern vector clock for causal ordering of distributed events.

    A vector clock maintains a mapping from process IDs to non-negative
    integer timestamps.  All mutation methods return *new* instances,
    keeping each clock value-immutable once constructed.

    Attributes:
        clock: Read-only mapping from process ID to its logical timestamp.
        processes: Frozenset of all process IDs tracked by this clock.
    """

    __slots__ = ("_clock", "_processes")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        processes: Iterable[str],
        initial_values: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialise a vector clock for *processes*.

        Args:
            processes: Non-empty collection of process identifiers.
            initial_values: Optional mapping of process IDs to starting
                timestamps.  Processes absent from this mapping default
                to zero.

        Raises:
            ValueError: If *processes* is empty.
        """
        procs = frozenset(processes)
        if not procs:
            raise ValueError("VectorClock requires at least one process")

        vals = initial_values or {}
        self._clock: Dict[str, int] = {p: vals.get(p, 0) for p in procs}
        self._processes: frozenset[str] = procs

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def clock(self) -> Dict[str, int]:
        """Return a *copy* of the internal clock mapping."""
        return dict(self._clock)

    @property
    def processes(self) -> frozenset[str]:
        """Return the frozenset of tracked process IDs."""
        return self._processes

    # ------------------------------------------------------------------ #
    # Clock operations
    # ------------------------------------------------------------------ #

    def increment(self, process: str) -> VectorClock:
        """
        Return a **new** VectorClock with *process*'s timestamp incremented by 1.

        This models a local event (or the local component of a send /
        receive event) on the given process.

        Raises:
            ValueError: If *process* is not tracked by this clock.
        """
        if process not in self._processes:
            raise ValueError(f"Unknown process '{process}' (known: {sorted(self._processes)})")

        new_vals = dict(self._clock)
        new_vals[process] += 1
        return VectorClock(self._processes, initial_values=new_vals)

    def merge(self, other: VectorClock) -> VectorClock:
        """
        Return a **new** VectorClock with the component-wise maximum.

        Used when a process receives a message: the receiver merges the
        piggybacked clock from the sender with its own.

        Raises:
            ValueError: If the two clocks track different process sets.
        """
        if self._processes != other._processes:
            raise ValueError(
                f"Cannot merge clocks with different process sets: "
                f"{sorted(self._processes)} vs {sorted(other._processes)}"
            )

        merged = {p: max(self._clock[p], other._clock[p]) for p in self._processes}
        return VectorClock(self._processes, initial_values=merged)

    # ------------------------------------------------------------------ #
    # Ordering
    # ------------------------------------------------------------------ #

    def __le__(self, other: object) -> bool:
        """
        True when every component of *self* is ≤ the corresponding
        component of *other*  (``self ≤ other``).
        """
        if not isinstance(other, VectorClock):
            return NotImplemented
        return all(self._clock[p] <= other._clock[p] for p in self._processes)

    def __lt__(self, other: object) -> bool:
        """
        Strict causal ordering: ``self < other`` iff ``self ≤ other``
        and they are not equal (at least one component is strictly less).
        """
        if not isinstance(other, VectorClock):
            return NotImplemented
        return (self <= other) and any(self._clock[p] < other._clock[p] for p in self._processes)

    def is_concurrent_with(self, other: VectorClock) -> bool:
        """
        True when neither clock causally precedes the other.

        Two events are concurrent iff their vector clocks are
        incomparable under the componentwise ≤ ordering.
        """
        return not (self <= other) and not (other <= self)

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    @classmethod
    def from_string(cls, s: str, processes: Iterable[str]) -> VectorClock:
        """
        Parse a vector clock from the format ``"P1:2;P2:1;P3:0"``.

        Args:
            s: Semicolon-separated ``Process:Count`` pairs.
            processes: Expected set of process IDs.

        Raises:
            ValueError: If the string is malformed or the parsed
                processes do not match *processes*.
        """
        procs = frozenset(processes)
        vals: Dict[str, int] = {}
        try:
            for token in s.split(";"):
                proc_id, count_str = token.split(":")
                vals[proc_id.strip()] = int(count_str.strip())
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Malformed vector clock string: '{s}'") from exc

        if frozenset(vals.keys()) != procs:
            raise ValueError(
                f"Processes in string {sorted(vals.keys())} "
                f"do not match expected {sorted(procs)}"
            )

        return cls(procs, initial_values=vals)

    # ------------------------------------------------------------------ #
    # Equality / hashing / repr
    # ------------------------------------------------------------------ #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return NotImplemented
        return self._processes == other._processes and self._clock == other._clock

    def __hash__(self) -> int:
        return hash((self._processes, tuple(sorted(self._clock.items()))))

    def __repr__(self) -> str:
        entries = ", ".join(f"{p}:{self._clock[p]}" for p in sorted(self._processes))
        return f"VectorClock({entries})"
