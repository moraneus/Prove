"""
Parser for EPLTL formulas.

Implements a grammar with proper precedence and associativity rules
to parse EPLTL formula strings into an abstract syntax tree (AST).
"""

from __future__ import annotations

import sly

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
from prove.parser.lexer import EPLTLLexer


class ParseError(Exception):
    """Exception raised for parsing errors."""

    pass


class _SLYParser(sly.Parser):
    """
    SLY-based parser for EPLTL formulas.

    Precedence (lowest to highest):
        1. <->  (biconditional, left-to-right)
        2. ->   (implication, right-to-left)
        3. |    (disjunction, left-to-right)
        4. &    (conjunction, left-to-right)
        5. S    (since, right-to-left)
        6. ! @  (unary, right-to-left)
    """

    tokens = EPLTLLexer.tokens

    precedence = (
        ("left", IFF),
        ("right", IMPLIES),
        ("left", OR),
        ("left", AND),
        ("right", SINCE),
        ("right", NOT, YESTERDAY),
    )

    # --- Atomic formulas ---

    @_("PROP")
    def formula(self, p):
        return Proposition(p.PROP)

    @_("TRUE")
    def formula(self, p):
        return TrueConstant()

    @_("FALSE")
    def formula(self, p):
        return FalseConstant()

    # --- Unary operators ---

    @_("NOT formula")
    def formula(self, p):
        return Negation(p.formula)

    @_("YESTERDAY formula")
    def formula(self, p):
        return Yesterday(p.formula)

    # --- Binary operators ---

    @_("formula AND formula")
    def formula(self, p):
        return Conjunction(p.formula0, p.formula1)

    @_("formula OR formula")
    def formula(self, p):
        return Disjunction(p.formula0, p.formula1)

    @_("formula IMPLIES formula")
    def formula(self, p):
        return Implication(p.formula0, p.formula1)

    @_("formula IFF formula")
    def formula(self, p):
        return Biconditional(p.formula0, p.formula1)

    @_("formula SINCE formula")
    def formula(self, p):
        return Since(p.formula0, p.formula1)

    # --- Parentheses ---

    @_("LPAREN formula RPAREN")
    def formula(self, p):
        return p.formula

    def error(self, token):
        if token:
            raise ParseError(
                f"Syntax error at '{token.value}' " f"(type: {token.type}, index: {token.index})"
            )
        raise ParseError("Syntax error: unexpected end of formula")


class EPLTLParser:
    """
    Parser for EPLTL formulas.

    Wraps the SLY-based parser with a clean public interface.
    Converts formula strings into AST nodes.
    """

    def __init__(self) -> None:
        self._lexer = EPLTLLexer()
        self._parser = _SLYParser()

    def parse(self, text: str) -> Formula:
        """
        Parse a formula string into an AST.

        Args:
            text: The formula string to parse.

        Returns:
            The root Formula node of the AST.

        Raises:
            ParseError: If the formula is syntactically invalid.
        """
        text = text.strip()
        if not text:
            raise ParseError("Syntax error: empty formula")

        result = self._parser.parse(self._lexer.tokenize(text))
        if result is None:
            raise ParseError("Syntax error: could not parse formula")
        return result
