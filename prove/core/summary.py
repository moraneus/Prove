"""
EPLTL summary implementation for incremental formula evaluation.

A summary maps each subformula to its current truth value, enabling
efficient runtime verification without re-traversing the entire
execution history. This is the key data structure from the
Havelund-Rosu approach to past-time LTL monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from prove.core.event import Event
from prove.parser.ast_nodes import (
    Biconditional,
    Conjunction,
    Disjunction,
    FalseConstant,
    Formula,
    Implication,
    Negation,
    Proposition,
    Since,
    TrueConstant,
    Yesterday,
)


@dataclass(frozen=True)
class Summary:
    """
    Maps each subformula of an EPLTL formula to its current truth value.

    Summaries enable incremental evaluation: given the previous summary
    and a new event (or global state), the next summary can be computed
    without revisiting earlier events. Multiple summaries per graph node
    represent different possible execution paths.

    Attributes:
        values: Immutable mapping from Formula to bool.
    """

    values: tuple[tuple[Formula, bool], ...]

    @classmethod
    def initial(cls, formula: Formula) -> Summary:
        """
        Create the initial summary before any events are processed.

        Initial truth values:
            - TRUE  -> True
            - FALSE -> False
            - All other subformulas -> False

        Then recompute compound formulas bottom-up so they are
        consistent with their subformulas' initial values.

        Args:
            formula: The top-level EPLTL formula.

        Returns:
            A new Summary with initial truth values.
        """
        subs = _topological_sort(formula)
        vals: dict[Formula, bool] = {}

        for sub in subs:
            if isinstance(sub, TrueConstant):
                vals[sub] = True
            elif isinstance(sub, FalseConstant):
                vals[sub] = False
            elif isinstance(sub, Proposition):
                vals[sub] = False
            elif isinstance(sub, Negation):
                vals[sub] = not vals[sub.operand]
            elif isinstance(sub, Yesterday):
                vals[sub] = False
            elif isinstance(sub, Conjunction):
                vals[sub] = vals[sub.left] and vals[sub.right]
            elif isinstance(sub, Disjunction):
                vals[sub] = vals[sub.left] or vals[sub.right]
            elif isinstance(sub, Implication):
                vals[sub] = (not vals[sub.left]) or vals[sub.right]
            elif isinstance(sub, Biconditional):
                vals[sub] = vals[sub.left] == vals[sub.right]
            elif isinstance(sub, Since):
                vals[sub] = False
            else:  # pragma: no cover
                vals[sub] = False

        return cls(values=tuple(sorted(vals.items(), key=id)))

    def update(self, event: Event, formula: Formula) -> Summary:
        """
        Compute updated summary after observing a new event.

        Implements the summary update rules (Havelund-Rosu):

        Constants:
            TRUE  -> True,  FALSE -> False

        Propositions:
            p -> (p in event.propositions)

        Boolean (use NEW values, computed bottom-up):
            !phi      -> not new[phi]
            phi & psi -> new[phi] and new[psi]
            phi | psi -> new[phi] or new[psi]
            phi -> psi -> (not new[phi]) or new[psi]
            phi <-> psi -> new[phi] == new[psi]

        Temporal (mix of old and new):
            @phi      -> old[phi]     (value BEFORE this update)
            phi S psi -> new[psi] or (new[phi] and old[phi S psi])

        Args:
            event: The newly observed event.
            formula: The top-level formula (for subformula enumeration).

        Returns:
            A new Summary with updated truth values.
        """
        old = dict(self.values)
        new: dict[Formula, bool] = {}

        for sub in _topological_sort(formula):
            if isinstance(sub, TrueConstant):
                new[sub] = True
            elif isinstance(sub, FalseConstant):
                new[sub] = False
            elif isinstance(sub, Proposition):
                new[sub] = sub.name in event.propositions
            elif isinstance(sub, Negation):
                new[sub] = not new[sub.operand]
            elif isinstance(sub, Yesterday):
                # Use OLD value of the operand (before this update)
                new[sub] = old.get(sub.operand, False)
            elif isinstance(sub, Conjunction):
                new[sub] = new[sub.left] and new[sub.right]
            elif isinstance(sub, Disjunction):
                new[sub] = new[sub.left] or new[sub.right]
            elif isinstance(sub, Implication):
                new[sub] = (not new[sub.left]) or new[sub.right]
            elif isinstance(sub, Biconditional):
                new[sub] = new[sub.left] == new[sub.right]
            elif isinstance(sub, Since):
                # psi holds now OR (phi holds now AND phi S psi held before)
                psi_now = new[sub.right]
                phi_now = new[sub.left]
                since_before = old.get(sub, False)
                new[sub] = psi_now or (phi_now and since_before)
            else:  # pragma: no cover
                new[sub] = False

        return Summary(values=tuple(sorted(new.items(), key=id)))

    def evaluate(self, formula: Formula) -> bool:
        """
        Return the truth value of a formula in this summary.

        Args:
            formula: A subformula whose value is stored in this summary.

        Returns:
            The boolean truth value.

        Raises:
            KeyError: If the formula is not in this summary.
        """
        for f, v in self.values:
            if f == formula:
                return v
        raise KeyError(f"Formula {formula} not in summary")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Summary):
            return NotImplemented
        return self._as_dict() == other._as_dict()

    def __hash__(self) -> int:
        # Hash based on sorted (formula_str, value) pairs for stability
        return hash(tuple(
            (str(f), v) for f, v in sorted(self.values, key=lambda x: str(x[0]))
        ))

    def _as_dict(self) -> dict[str, bool]:
        """Convert to a string-keyed dict for comparison."""
        return {str(f): v for f, v in self.values}

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{f}: {v}" for f, v in sorted(self.values, key=lambda x: str(x[0]))
        )
        return f"Summary({{{entries}}})"


def _topological_sort(formula: Formula) -> list[Formula]:
    """
    Return subformulas in bottom-up order (leaves first, root last).

    This ensures that when computing a compound formula's value,
    all of its operands' values have already been computed.
    """
    result: list[Formula] = []
    visited: set[int] = set()

    def visit(f: Formula) -> None:
        fid = id(f)
        if fid in visited:
            return
        visited.add(fid)

        # Visit children first (bottom-up)
        if isinstance(f, Negation):
            visit(f.operand)
        elif isinstance(f, Yesterday):
            visit(f.operand)
        elif isinstance(f, (Conjunction, Disjunction, Implication,
                            Biconditional, Since)):
            visit(f.left)
            visit(f.right)

        result.append(f)

    visit(formula)
    return result
