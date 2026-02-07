"""
Formula utilities for EPLTL.

Provides convenience functions for parsing, inspecting, and
manipulating EPLTL formulas: subformula extraction, proposition
listing, canonical string conversion, and basic simplification.
"""

from __future__ import annotations

from typing import FrozenSet

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
from prove.parser.grammar import EPLTLParser


_parser = EPLTLParser()


def parse_formula(text: str) -> Formula:
    """
    Parse a formula string into an AST.

    Args:
        text: The EPLTL formula string.

    Returns:
        The root Formula node of the AST.

    Raises:
        ParseError: If the formula is syntactically invalid.
    """
    return _parser.parse(text)


def subformulas(formula: Formula) -> FrozenSet[Formula]:
    """
    Return all subformulas of the given formula, including itself.

    Args:
        formula: The formula to extract subformulas from.

    Returns:
        A frozenset of all subformulas.
    """
    return formula.subformulas()


def propositions(formula: Formula) -> FrozenSet[str]:
    """
    Return all proposition names appearing in the formula.

    Args:
        formula: The formula to inspect.

    Returns:
        A frozenset of proposition name strings.
    """
    return frozenset(
        sub.name for sub in formula.subformulas() if isinstance(sub, Proposition)
    )


def to_string(formula: Formula) -> str:
    """
    Convert a formula to its canonical string representation.

    Args:
        formula: The formula to convert.

    Returns:
        The canonical string representation.
    """
    return str(formula)


def simplify(formula: Formula) -> Formula:
    """
    Apply basic simplifications to a formula.

    Simplification rules applied recursively (bottom-up):
        - !!phi        → phi              (double negation elimination)
        - !TRUE        → FALSE
        - !FALSE       → TRUE
        - TRUE & phi   → phi              (conjunction identity)
        - phi & TRUE   → phi
        - FALSE & phi  → FALSE            (conjunction annihilation)
        - phi & FALSE  → FALSE
        - TRUE | phi   → TRUE             (disjunction annihilation)
        - phi | TRUE   → TRUE
        - FALSE | phi  → phi              (disjunction identity)
        - phi | FALSE  → phi

    Args:
        formula: The formula to simplify.

    Returns:
        A simplified formula (may be the same object if no simplification applies).
    """
    return _simplify(formula)


def _simplify(f: Formula) -> Formula:
    """Recursively simplify a formula bottom-up."""
    if isinstance(f, (TrueConstant, FalseConstant, Proposition)):
        return f

    if isinstance(f, Negation):
        operand = _simplify(f.operand)
        # !TRUE → FALSE
        if isinstance(operand, TrueConstant):
            return FalseConstant()
        # !FALSE → TRUE
        if isinstance(operand, FalseConstant):
            return TrueConstant()
        # !!phi → phi
        if isinstance(operand, Negation):
            return operand.operand
        return Negation(operand)

    if isinstance(f, Yesterday):
        return Yesterday(_simplify(f.operand))

    if isinstance(f, Conjunction):
        left = _simplify(f.left)
        right = _simplify(f.right)
        # TRUE & phi → phi
        if isinstance(left, TrueConstant):
            return right
        # phi & TRUE → phi
        if isinstance(right, TrueConstant):
            return left
        # FALSE & phi → FALSE
        if isinstance(left, FalseConstant):
            return FalseConstant()
        # phi & FALSE → FALSE
        if isinstance(right, FalseConstant):
            return FalseConstant()
        return Conjunction(left, right)

    if isinstance(f, Disjunction):
        left = _simplify(f.left)
        right = _simplify(f.right)
        # TRUE | phi → TRUE
        if isinstance(left, TrueConstant):
            return TrueConstant()
        # phi | TRUE → TRUE
        if isinstance(right, TrueConstant):
            return TrueConstant()
        # FALSE | phi → phi
        if isinstance(left, FalseConstant):
            return right
        # phi | FALSE → phi
        if isinstance(right, FalseConstant):
            return left
        return Disjunction(left, right)

    if isinstance(f, Implication):
        return Implication(_simplify(f.left), _simplify(f.right))

    if isinstance(f, Biconditional):
        return Biconditional(_simplify(f.left), _simplify(f.right))

    if isinstance(f, Since):
        return Since(_simplify(f.left), _simplify(f.right))

    return f  # pragma: no cover
