"""
Abstract syntax tree node definitions for EPLTL formulas.

Defines immutable, hashable AST nodes for all EPLTL operators:
constants (TRUE, FALSE), propositions, unary operators (negation,
yesterday), binary operators (conjunction, disjunction, implication,
biconditional, since), and derived operators (once, historically).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import FrozenSet


class Formula(ABC):
    """
    Base class for all EPLTL formula nodes.

    All formula nodes are immutable and support equality comparison
    and hashing for use in sets and dictionaries.
    """

    @abstractmethod
    def subformulas(self) -> FrozenSet[Formula]:
        """Return set of all subformulas including self."""

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of formula."""

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check equality with another formula."""

    @abstractmethod
    def __hash__(self) -> int:
        """Return hash for use in sets and dicts."""

    def __repr__(self) -> str:
        return str(self)


# === Constants ===


class TrueConstant(Formula):
    """Represents the constant TRUE."""

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self})

    def __str__(self) -> str:
        return "TRUE"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrueConstant):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(("TrueConstant",))


class FalseConstant(Formula):
    """Represents the constant FALSE."""

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self})

    def __str__(self) -> str:
        return "FALSE"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FalseConstant):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(("FalseConstant",))


# === Atomic ===


class Proposition(Formula):
    """
    Represents an atomic proposition.

    Attributes:
        name: The proposition identifier (e.g., "ready", "p1", "is_valid").
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self})

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Proposition):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(("Proposition", self.name))


# === Unary Operators ===


class Negation(Formula):
    """
    Represents !phi (negation).

    Attributes:
        operand: The formula being negated.
    """

    __slots__ = ("operand",)

    def __init__(self, operand: Formula) -> None:
        self.operand = operand

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self}) | self.operand.subformulas()

    def __str__(self) -> str:
        return f"!{self.operand}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Negation):
            return NotImplemented
        return self.operand == other.operand

    def __hash__(self) -> int:
        return hash(("Negation", self.operand))


class Yesterday(Formula):
    """
    Represents @phi (yesterday/previous operator).

    @phi is true at position i if phi was true at position i-1.
    At position 1 (the first state), @phi is always false.

    Attributes:
        operand: The formula to evaluate in the previous state.
    """

    __slots__ = ("operand",)

    def __init__(self, operand: Formula) -> None:
        self.operand = operand

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self}) | self.operand.subformulas()

    def __str__(self) -> str:
        return f"@{self.operand}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Yesterday):
            return NotImplemented
        return self.operand == other.operand

    def __hash__(self) -> int:
        return hash(("Yesterday", self.operand))


# === Binary Operator Base ===


class _BinaryOp(Formula):
    """Base class for binary operators (not part of public API)."""

    __slots__ = ("left", "right")

    _op_symbol: str = ""

    def __init__(self, left: Formula, right: Formula) -> None:
        self.left = left
        self.right = right

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self}) | self.left.subformulas() | self.right.subformulas()

    def __str__(self) -> str:
        return f"({self.left} {self._op_symbol} {self.right})"

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.left, self.right))


# === Binary Operators ===


class Conjunction(_BinaryOp):
    """
    Represents phi & psi (conjunction/AND).

    Attributes:
        left: Left operand.
        right: Right operand.
    """

    _op_symbol = "&"


class Disjunction(_BinaryOp):
    """
    Represents phi | psi (disjunction/OR).

    Attributes:
        left: Left operand.
        right: Right operand.
    """

    _op_symbol = "|"


class Implication(_BinaryOp):
    """
    Represents phi -> psi (implication).

    Equivalent to: !phi | psi

    Attributes:
        left: Antecedent (the "if" part).
        right: Consequent (the "then" part).
    """

    _op_symbol = "->"


class Biconditional(_BinaryOp):
    """
    Represents phi <-> psi (biconditional/iff).

    Equivalent to: (phi -> psi) & (psi -> phi)

    Attributes:
        left: Left operand.
        right: Right operand.
    """

    _op_symbol = "<->"


class Since(_BinaryOp):
    """
    Represents phi S psi (since operator).

    phi S psi is true at position i if there exists a position j <= i
    where psi was true, and phi has been true at all positions from
    j+1 to i (inclusive).

    Attributes:
        left: The formula that must hold since psi (phi in "phi S psi").
        right: The formula that triggered the since (psi in "phi S psi").
    """

    _op_symbol = "S"


# === Derived Operators ===


class Once(Formula):
    """
    Represents O phi (once/eventually in past).

    Equivalent to: TRUE S phi

    O phi is true if phi was true at some point in the past (including now).

    Attributes:
        operand: The formula that must have been true at some point.
    """

    __slots__ = ("operand", "_expanded")

    def __init__(self, operand: Formula) -> None:
        self.operand = operand
        # Store expanded form: TRUE S operand
        self._expanded = Since(TrueConstant(), operand)

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self}) | self._expanded.subformulas()

    def __str__(self) -> str:
        return str(self._expanded)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Once):
            return NotImplemented
        return self.operand == other.operand

    def __hash__(self) -> int:
        return hash(("Once", self.operand))


class Historically(Formula):
    """
    Represents H phi (historically/always in past).

    Equivalent to: !(TRUE S !phi) or !O(!phi)

    H phi is true if phi has been true at all points in the past (including now).

    Attributes:
        operand: The formula that must have always been true.
    """

    __slots__ = ("operand", "_expanded")

    def __init__(self, operand: Formula) -> None:
        self.operand = operand
        # Store expanded form: !(TRUE S !operand)
        self._expanded = Negation(Since(TrueConstant(), Negation(operand)))

    def subformulas(self) -> FrozenSet[Formula]:
        return frozenset({self}) | self._expanded.subformulas()

    def __str__(self) -> str:
        return str(self._expanded)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Historically):
            return NotImplemented
        return self.operand == other.operand

    def __hash__(self) -> int:
        return hash(("Historically", self.operand))
