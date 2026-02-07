"""
Lexical analyzer for EPLTL formulas.

Tokenizes EPLTL formula strings into a stream of tokens (propositions,
operators, constants, delimiters) that can be consumed by the parser.
"""

from __future__ import annotations

import sly


class LexerError(Exception):
    """Exception raised for lexical analysis errors."""
    pass


class EPLTLLexer(sly.Lexer):
    """
    Lexical analyzer for EPLTL formulas.

    Converts a formula string into a stream of tokens.

    Token Types:
        TRUE, FALSE     - Boolean constants
        PROP            - Proposition identifiers
        NOT, YESTERDAY  - Unary operators
        AND, OR, IMPLIES, IFF, SINCE - Binary operators
        LPAREN, RPAREN  - Delimiters
    """

    tokens = {
        TRUE, FALSE,
        PROP,
        NOT, YESTERDAY,
        AND, OR, IMPLIES, IFF, SINCE,
        LPAREN, RPAREN,
    }

    # Ignored characters
    ignore = " \t"

    # Ignore newlines
    @_(r"\n+")
    def ignore_newline(self, t):
        self.lineno += len(t.value)

    # Ignore comments (# to end of line)
    ignore_comment = r"\#[^\n]*"

    # Multi-character operators (order matters — longer patterns first)

    # <-> must come before ->
    IFF = r"<->|↔"

    # -> must come before single-char matches
    IMPLIES = r"->|→"

    # && must come before &
    AND = r"&&|∧"

    # || must come before |
    OR = r"\|\||∨"

    # Single-character operators and delimiters
    NOT = r"!"
    YESTERDAY = r"@"
    LPAREN = r"\("
    RPAREN = r"\)"

    # Single & and | (after multi-char versions are matched)
    @_(r"&")
    def AND_SINGLE(self, t):
        t.type = "AND"
        return t

    @_(r"\|")
    def OR_SINGLE(self, t):
        t.type = "OR"
        return t

    # Unicode negation
    @_(r"¬")
    def NOT_UNICODE(self, t):
        t.type = "NOT"
        return t

    # Identifiers and keywords
    # This pattern matches all word-like tokens including keywords.
    # Keywords are disambiguated by checking the matched value.
    @_(r"[a-zA-Z_][a-zA-Z0-9_'\.]*")
    def PROP(self, t):
        # Keywords that map to specific token types.
        # Only exact matches are keywords; longer words are propositions.
        keywords = {
            "TRUE": "TRUE",
            "true": "TRUE",
            "FALSE": "FALSE",
            "false": "FALSE",
            "not": "NOT",
            "and": "AND",
            "or": "OR",
            "implies": "IMPLIES",
            "iff": "IFF",
            "since": "SINCE",
            "prev": "YESTERDAY",
            "previous": "YESTERDAY",
            # Single-char keywords: 'S' = SINCE, 'Y' = YESTERDAY
            "S": "SINCE",
            "Y": "YESTERDAY",
        }
        t.type = keywords.get(t.value, "PROP")
        return t

    def error(self, t):
        """Handle invalid characters."""
        raise LexerError(
            f"Invalid character '{t.value[0]}' at index {self.index}"
        )
