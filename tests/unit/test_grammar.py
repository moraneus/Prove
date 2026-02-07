"""
Tests for the EPLTL grammar/parser.

Tests cover parsing of atomic formulas, unary and binary operators,
operator precedence, associativity, parentheses, complex formulas,
and error handling.
"""

import pytest

from prove.parser.ast_nodes import (
    Biconditional,
    Conjunction,
    Disjunction,
    FalseConstant,
    Implication,
    Negation,
    Proposition,
    Since,
    TrueConstant,
    Yesterday,
)
from prove.parser.grammar import EPLTLParser, ParseError


@pytest.fixture
def parser() -> EPLTLParser:
    """Return a fresh parser instance."""
    return EPLTLParser()


def _parse(parser: EPLTLParser, text: str):
    """Helper: parse text and return AST."""
    return parser.parse(text)


class TestAtomicFormulas:
    """Test parsing of atomic formulas (propositions and constants)."""

    def test_simple_proposition(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "ready")
        assert isinstance(result, Proposition)
        assert result.name == "ready"

    def test_proposition_with_underscore(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "is_valid")
        assert isinstance(result, Proposition)
        assert result.name == "is_valid"

    def test_proposition_with_dot(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "state.active")
        assert isinstance(result, Proposition)
        assert result.name == "state.active"

    def test_proposition_with_apostrophe(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "x'")
        assert isinstance(result, Proposition)
        assert result.name == "x'"

    def test_true_constant(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "TRUE")
        assert isinstance(result, TrueConstant)

    def test_true_lowercase(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "true")
        assert isinstance(result, TrueConstant)

    def test_false_constant(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "FALSE")
        assert isinstance(result, FalseConstant)

    def test_false_lowercase(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "false")
        assert isinstance(result, FalseConstant)


class TestUnaryOperators:
    """Test parsing of unary operators."""

    def test_negation_bang(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "!ready")
        assert isinstance(result, Negation)
        assert isinstance(result.operand, Proposition)
        assert result.operand.name == "ready"

    def test_negation_not_keyword(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "not ready")
        assert isinstance(result, Negation)

    def test_negation_unicode(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "\u00acready")
        assert isinstance(result, Negation)

    def test_yesterday_at(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "@ready")
        assert isinstance(result, Yesterday)
        assert isinstance(result.operand, Proposition)

    def test_yesterday_y(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "Y ready")
        assert isinstance(result, Yesterday)

    def test_yesterday_prev(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "prev ready")
        assert isinstance(result, Yesterday)

    def test_double_negation(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "!!p")
        assert isinstance(result, Negation)
        assert isinstance(result.operand, Negation)
        assert isinstance(result.operand.operand, Proposition)

    def test_negation_of_yesterday(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "!@p")
        assert isinstance(result, Negation)
        assert isinstance(result.operand, Yesterday)

    def test_yesterday_of_negation(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "@!p")
        assert isinstance(result, Yesterday)
        assert isinstance(result.operand, Negation)

    def test_nested_yesterday(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "@@p")
        assert isinstance(result, Yesterday)
        assert isinstance(result.operand, Yesterday)


class TestBinaryOperators:
    """Test parsing of binary operators."""

    def test_conjunction(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a & b")
        assert isinstance(result, Conjunction)
        assert isinstance(result.left, Proposition)
        assert isinstance(result.right, Proposition)
        assert result.left.name == "a"
        assert result.right.name == "b"

    def test_conjunction_double_amp(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a && b")
        assert isinstance(result, Conjunction)

    def test_conjunction_keyword(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a and b")
        assert isinstance(result, Conjunction)

    def test_disjunction(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a | b")
        assert isinstance(result, Disjunction)

    def test_disjunction_double_pipe(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a || b")
        assert isinstance(result, Disjunction)

    def test_disjunction_keyword(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a or b")
        assert isinstance(result, Disjunction)

    def test_implication(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a -> b")
        assert isinstance(result, Implication)

    def test_implication_keyword(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a implies b")
        assert isinstance(result, Implication)

    def test_biconditional(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a <-> b")
        assert isinstance(result, Biconditional)

    def test_biconditional_keyword(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a iff b")
        assert isinstance(result, Biconditional)

    def test_since(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a S b")
        assert isinstance(result, Since)
        assert result.left.name == "a"
        assert result.right.name == "b"

    def test_since_keyword(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a since b")
        assert isinstance(result, Since)


class TestPrecedence:
    """Test operator precedence (lowest to highest: <->, ->, |, &, S, !/@ )."""

    def test_and_binds_tighter_than_or(self, parser: EPLTLParser) -> None:
        """a & b | c  →  (a & b) | c"""
        result = _parse(parser, "a & b | c")
        assert isinstance(result, Disjunction)
        assert isinstance(result.left, Conjunction)
        assert isinstance(result.right, Proposition)

    def test_or_left_of_and(self, parser: EPLTLParser) -> None:
        """a | b & c  →  a | (b & c)"""
        result = _parse(parser, "a | b & c")
        assert isinstance(result, Disjunction)
        assert isinstance(result.left, Proposition)
        assert isinstance(result.right, Conjunction)

    def test_and_binds_tighter_than_implies(self, parser: EPLTLParser) -> None:
        """a & b -> c  →  (a & b) -> c"""
        result = _parse(parser, "a & b -> c")
        assert isinstance(result, Implication)
        assert isinstance(result.left, Conjunction)

    def test_or_binds_tighter_than_implies(self, parser: EPLTLParser) -> None:
        """a | b -> c  →  (a | b) -> c"""
        result = _parse(parser, "a | b -> c")
        assert isinstance(result, Implication)
        assert isinstance(result.left, Disjunction)

    def test_implies_binds_tighter_than_iff(self, parser: EPLTLParser) -> None:
        """a -> b <-> c  →  (a -> b) <-> c"""
        result = _parse(parser, "a -> b <-> c")
        assert isinstance(result, Biconditional)
        assert isinstance(result.left, Implication)

    def test_negation_binds_tightest(self, parser: EPLTLParser) -> None:
        """!a & b  →  (!a) & b"""
        result = _parse(parser, "!a & b")
        assert isinstance(result, Conjunction)
        assert isinstance(result.left, Negation)
        assert isinstance(result.right, Proposition)

    def test_since_binds_tighter_than_and(self, parser: EPLTLParser) -> None:
        """a S b & c  →  (a S b) & c"""
        result = _parse(parser, "a S b & c")
        assert isinstance(result, Conjunction)
        assert isinstance(result.left, Since)
        assert isinstance(result.right, Proposition)

    def test_negation_binds_tighter_than_since(self, parser: EPLTLParser) -> None:
        """!a S b  →  (!a) S b"""
        result = _parse(parser, "!a S b")
        assert isinstance(result, Since)
        assert isinstance(result.left, Negation)
        assert isinstance(result.right, Proposition)

    def test_yesterday_binds_tighter_than_since(self, parser: EPLTLParser) -> None:
        """@a S b  →  (@a) S b"""
        result = _parse(parser, "@a S b")
        assert isinstance(result, Since)
        assert isinstance(result.left, Yesterday)


class TestAssociativity:
    """Test operator associativity."""

    def test_conjunction_left_associative(self, parser: EPLTLParser) -> None:
        """a & b & c  →  (a & b) & c"""
        result = _parse(parser, "a & b & c")
        assert isinstance(result, Conjunction)
        assert isinstance(result.left, Conjunction)
        assert isinstance(result.right, Proposition)
        assert result.right.name == "c"

    def test_disjunction_left_associative(self, parser: EPLTLParser) -> None:
        """a | b | c  →  (a | b) | c"""
        result = _parse(parser, "a | b | c")
        assert isinstance(result, Disjunction)
        assert isinstance(result.left, Disjunction)
        assert isinstance(result.right, Proposition)

    def test_implication_right_associative(self, parser: EPLTLParser) -> None:
        """a -> b -> c  →  a -> (b -> c)"""
        result = _parse(parser, "a -> b -> c")
        assert isinstance(result, Implication)
        assert isinstance(result.left, Proposition)
        assert isinstance(result.right, Implication)

    def test_biconditional_left_associative(self, parser: EPLTLParser) -> None:
        """a <-> b <-> c  →  (a <-> b) <-> c"""
        result = _parse(parser, "a <-> b <-> c")
        assert isinstance(result, Biconditional)
        assert isinstance(result.left, Biconditional)
        assert isinstance(result.right, Proposition)

    def test_since_right_associative(self, parser: EPLTLParser) -> None:
        """a S b S c  →  a S (b S c)"""
        result = _parse(parser, "a S b S c")
        assert isinstance(result, Since)
        assert isinstance(result.left, Proposition)
        assert isinstance(result.right, Since)


class TestParentheses:
    """Test parenthesized expressions."""

    def test_simple_parens(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "(a)")
        assert isinstance(result, Proposition)
        assert result.name == "a"

    def test_double_parens(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "((a))")
        assert isinstance(result, Proposition)

    def test_parens_override_precedence(self, parser: EPLTLParser) -> None:
        """a & (b | c)  →  Conjunction(a, Disjunction(b, c))"""
        result = _parse(parser, "a & (b | c)")
        assert isinstance(result, Conjunction)
        assert isinstance(result.right, Disjunction)

    def test_parens_around_binary(self, parser: EPLTLParser) -> None:
        """(a & b) | c  →  Disjunction(Conjunction(a, b), c)"""
        result = _parse(parser, "(a & b) | c")
        assert isinstance(result, Disjunction)
        assert isinstance(result.left, Conjunction)

    def test_parens_around_implication(self, parser: EPLTLParser) -> None:
        """(a -> b) -> c"""
        result = _parse(parser, "(a -> b) -> c")
        assert isinstance(result, Implication)
        assert isinstance(result.left, Implication)
        # Without parens, a -> b -> c would be a -> (b -> c)
        assert result.left.left.name == "a"

    def test_negation_of_parenthesized(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "!(a & b)")
        assert isinstance(result, Negation)
        assert isinstance(result.operand, Conjunction)

    def test_yesterday_of_parenthesized(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "@(a S b)")
        assert isinstance(result, Yesterday)
        assert isinstance(result.operand, Since)


class TestComplexFormulas:
    """Test parsing of complex, nested formulas."""

    def test_done_implies_confirmed_since_ready(self, parser: EPLTLParser) -> None:
        """done -> (confirmed S ready)"""
        result = _parse(parser, "done -> (confirmed S ready)")
        assert isinstance(result, Implication)
        assert isinstance(result.left, Proposition)
        assert result.left.name == "done"
        assert isinstance(result.right, Since)
        assert result.right.left.name == "confirmed"
        assert result.right.right.name == "ready"

    def test_once_pattern(self, parser: EPLTLParser) -> None:
        """TRUE S request  (Once request)"""
        result = _parse(parser, "TRUE S request")
        assert isinstance(result, Since)
        assert isinstance(result.left, TrueConstant)
        assert isinstance(result.right, Proposition)

    def test_historically_pattern(self, parser: EPLTLParser) -> None:
        """!(TRUE S !valid)  (Historically valid)"""
        result = _parse(parser, "!(TRUE S !valid)")
        assert isinstance(result, Negation)
        inner = result.operand
        assert isinstance(inner, Since)
        assert isinstance(inner.left, TrueConstant)
        assert isinstance(inner.right, Negation)
        assert isinstance(inner.right.operand, Proposition)

    def test_biconditional_compound(self, parser: EPLTLParser) -> None:
        """(a & b) <-> (c | d)"""
        result = _parse(parser, "(a & b) <-> (c | d)")
        assert isinstance(result, Biconditional)
        assert isinstance(result.left, Conjunction)
        assert isinstance(result.right, Disjunction)

    def test_negation_yesterday_since(self, parser: EPLTLParser) -> None:
        """!@(a S b)"""
        result = _parse(parser, "!@(a S b)")
        assert isinstance(result, Negation)
        assert isinstance(result.operand, Yesterday)
        assert isinstance(result.operand.operand, Since)

    def test_complex_safety(self, parser: EPLTLParser) -> None:
        """(done & confirmed) -> ((valid S ready) & @initialized)"""
        result = _parse(parser, "(done & confirmed) -> ((valid S ready) & @initialized)")
        assert isinstance(result, Implication)
        assert isinstance(result.left, Conjunction)
        assert isinstance(result.right, Conjunction)
        rhs = result.right
        assert isinstance(rhs.left, Since)
        assert isinstance(rhs.right, Yesterday)

    def test_response_implies_once_request(self, parser: EPLTLParser) -> None:
        """response -> (TRUE S request)"""
        result = _parse(parser, "response -> (TRUE S request)")
        assert isinstance(result, Implication)
        assert isinstance(result.right, Since)
        assert isinstance(result.right.left, TrueConstant)

    def test_mutual_exclusion(self, parser: EPLTLParser) -> None:
        """!(cs1 & cs2)"""
        result = _parse(parser, "!(cs1 & cs2)")
        assert isinstance(result, Negation)
        assert isinstance(result.operand, Conjunction)

    def test_error_not_preceded_by_valid(self, parser: EPLTLParser) -> None:
        """error -> !(@valid)"""
        result = _parse(parser, "error -> !(@valid)")
        assert isinstance(result, Implication)
        neg = result.right
        assert isinstance(neg, Negation)
        assert isinstance(neg.operand, Yesterday)


class TestRoundTrip:
    """Test that parsing produces ASTs with correct string representations."""

    def test_proposition_str(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "ready")
        assert str(result) == "ready"

    def test_conjunction_str(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a & b")
        assert str(result) == "(a & b)"

    def test_implication_str(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "a -> b")
        assert str(result) == "(a -> b)"

    def test_negation_str(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "!p")
        assert str(result) == "!p"

    def test_complex_str(self, parser: EPLTLParser) -> None:
        result = _parse(parser, "done -> (confirmed S ready)")
        assert str(result) == "(done -> (confirmed S ready))"


class TestErrorHandling:
    """Test parser error handling."""

    def test_empty_input(self, parser: EPLTLParser) -> None:
        with pytest.raises(ParseError):
            _parse(parser, "")

    def test_unclosed_paren(self, parser: EPLTLParser) -> None:
        with pytest.raises(ParseError):
            _parse(parser, "(a")

    def test_extra_rparen(self, parser: EPLTLParser) -> None:
        with pytest.raises(ParseError):
            _parse(parser, "a)")

    def test_missing_operand_binary(self, parser: EPLTLParser) -> None:
        with pytest.raises(ParseError):
            _parse(parser, "a &")

    def test_missing_left_operand(self, parser: EPLTLParser) -> None:
        with pytest.raises(ParseError):
            _parse(parser, "& b")

    def test_double_binary_operator(self, parser: EPLTLParser) -> None:
        with pytest.raises(ParseError):
            _parse(parser, "a & & b")

    def test_invalid_character(self, parser: EPLTLParser) -> None:
        """Invalid character in input raises LexerError (which is also acceptable)."""
        with pytest.raises(Exception):
            _parse(parser, "a $ b")

    def test_only_operator(self, parser: EPLTLParser) -> None:
        with pytest.raises(ParseError):
            _parse(parser, "->")
