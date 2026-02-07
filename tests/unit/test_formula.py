"""
Tests for the formula utility functions.

Tests cover parsing, subformula extraction, proposition listing,
canonical string conversion, and basic simplification.
"""

import pytest

from prove.parser.ast_nodes import (
    Biconditional,
    Conjunction,
    Disjunction,
    FalseConstant,
    Formula,
    Implication,
    Negation,
    Once,
    Proposition,
    Since,
    TrueConstant,
    Yesterday,
)
from prove.parser.formula import (
    parse_formula,
    propositions,
    simplify,
    subformulas,
    to_string,
)
from prove.parser.grammar import ParseError


class TestParseFormula:
    """Test the parse_formula convenience function."""

    def test_simple_prop(self) -> None:
        result = parse_formula("ready")
        assert isinstance(result, Proposition)
        assert result.name == "ready"

    def test_conjunction(self) -> None:
        result = parse_formula("a & b")
        assert isinstance(result, Conjunction)

    def test_complex(self) -> None:
        result = parse_formula("done -> (confirmed S ready)")
        assert isinstance(result, Implication)
        assert isinstance(result.right, Since)

    def test_empty_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_formula("")

    def test_invalid_raises(self) -> None:
        with pytest.raises(Exception):
            parse_formula("a $ b")

    def test_with_comments(self) -> None:
        """Comments in the formula string are ignored."""
        result = parse_formula("ready # this is ready")
        assert isinstance(result, Proposition)

    def test_whitespace_stripped(self) -> None:
        result = parse_formula("  ready  ")
        assert isinstance(result, Proposition)


class TestSubformulas:
    """Test subformula extraction."""

    def test_proposition(self) -> None:
        f = Proposition("p")
        subs = subformulas(f)
        assert subs == frozenset({f})

    def test_negation(self) -> None:
        p = Proposition("p")
        n = Negation(p)
        subs = subformulas(n)
        assert n in subs
        assert p in subs
        assert len(subs) == 2

    def test_conjunction(self) -> None:
        p = Proposition("p")
        q = Proposition("q")
        c = Conjunction(p, q)
        subs = subformulas(c)
        assert len(subs) == 3

    def test_nested(self) -> None:
        """Implication(Conjunction(p, q), r) has 5 subformulas."""
        p = Proposition("p")
        q = Proposition("q")
        r = Proposition("r")
        f = Implication(Conjunction(p, q), r)
        subs = subformulas(f)
        assert len(subs) == 5
        assert p in subs
        assert q in subs
        assert r in subs

    def test_deduplication(self) -> None:
        """Same proposition appearing twice is deduplicated."""
        p = Proposition("p")
        f = Conjunction(p, Negation(p))
        subs = subformulas(f)
        assert len(subs) == 3  # p, !p, (p & !p)

    def test_constants(self) -> None:
        t = TrueConstant()
        subs = subformulas(t)
        assert len(subs) == 1
        assert t in subs

    def test_since(self) -> None:
        p = Proposition("p")
        q = Proposition("q")
        s = Since(p, q)
        subs = subformulas(s)
        assert len(subs) == 3

    def test_yesterday(self) -> None:
        p = Proposition("p")
        y = Yesterday(p)
        subs = subformulas(y)
        assert len(subs) == 2


class TestPropositions:
    """Test proposition name extraction."""

    def test_single_prop(self) -> None:
        f = Proposition("ready")
        assert propositions(f) == frozenset({"ready"})

    def test_conjunction(self) -> None:
        f = Conjunction(Proposition("p"), Proposition("q"))
        assert propositions(f) == frozenset({"p", "q"})

    def test_duplicates_merged(self) -> None:
        p = Proposition("p")
        f = Conjunction(p, Negation(p))
        assert propositions(f) == frozenset({"p"})

    def test_no_propositions_in_constants(self) -> None:
        f = TrueConstant()
        assert propositions(f) == frozenset()

    def test_complex(self) -> None:
        f = Implication(
            Conjunction(Proposition("done"), Proposition("ready")),
            Since(Proposition("valid"), Proposition("init")),
        )
        assert propositions(f) == frozenset({"done", "ready", "valid", "init"})

    def test_with_yesterday(self) -> None:
        f = Yesterday(Proposition("p"))
        assert propositions(f) == frozenset({"p"})

    def test_true_s_request(self) -> None:
        """TRUE S request has only 'request' as a proposition."""
        f = Since(TrueConstant(), Proposition("request"))
        assert propositions(f) == frozenset({"request"})


class TestToString:
    """Test canonical string conversion."""

    def test_proposition(self) -> None:
        assert to_string(Proposition("ready")) == "ready"

    def test_true(self) -> None:
        assert to_string(TrueConstant()) == "TRUE"

    def test_false(self) -> None:
        assert to_string(FalseConstant()) == "FALSE"

    def test_negation(self) -> None:
        assert to_string(Negation(Proposition("p"))) == "!p"

    def test_yesterday(self) -> None:
        assert to_string(Yesterday(Proposition("p"))) == "@p"

    def test_conjunction(self) -> None:
        assert to_string(Conjunction(Proposition("a"), Proposition("b"))) == "(a & b)"

    def test_disjunction(self) -> None:
        assert to_string(Disjunction(Proposition("a"), Proposition("b"))) == "(a | b)"

    def test_implication(self) -> None:
        assert to_string(Implication(Proposition("a"), Proposition("b"))) == "(a -> b)"

    def test_biconditional(self) -> None:
        f = Biconditional(Proposition("a"), Proposition("b"))
        assert to_string(f) == "(a <-> b)"

    def test_since(self) -> None:
        assert to_string(Since(Proposition("a"), Proposition("b"))) == "(a S b)"

    def test_complex(self) -> None:
        f = Implication(
            Proposition("done"),
            Since(Proposition("confirmed"), Proposition("ready")),
        )
        assert to_string(f) == "(done -> (confirmed S ready))"


class TestSimplify:
    """Test formula simplification."""

    def test_double_negation(self) -> None:
        """!!p simplifies to p."""
        f = Negation(Negation(Proposition("p")))
        result = simplify(f)
        assert isinstance(result, Proposition)
        assert result.name == "p"

    def test_triple_negation(self) -> None:
        """!!!p simplifies to !p."""
        f = Negation(Negation(Negation(Proposition("p"))))
        result = simplify(f)
        assert isinstance(result, Negation)
        assert isinstance(result.operand, Proposition)

    def test_conjunction_with_true(self) -> None:
        """TRUE & p simplifies to p."""
        f = Conjunction(TrueConstant(), Proposition("p"))
        result = simplify(f)
        assert isinstance(result, Proposition)
        assert result.name == "p"

    def test_conjunction_with_true_right(self) -> None:
        """p & TRUE simplifies to p."""
        f = Conjunction(Proposition("p"), TrueConstant())
        result = simplify(f)
        assert isinstance(result, Proposition)
        assert result.name == "p"

    def test_conjunction_with_false(self) -> None:
        """FALSE & p simplifies to FALSE."""
        f = Conjunction(FalseConstant(), Proposition("p"))
        result = simplify(f)
        assert isinstance(result, FalseConstant)

    def test_conjunction_with_false_right(self) -> None:
        """p & FALSE simplifies to FALSE."""
        f = Conjunction(Proposition("p"), FalseConstant())
        result = simplify(f)
        assert isinstance(result, FalseConstant)

    def test_disjunction_with_true(self) -> None:
        """TRUE | p simplifies to TRUE."""
        f = Disjunction(TrueConstant(), Proposition("p"))
        result = simplify(f)
        assert isinstance(result, TrueConstant)

    def test_disjunction_with_false(self) -> None:
        """FALSE | p simplifies to p."""
        f = Disjunction(FalseConstant(), Proposition("p"))
        result = simplify(f)
        assert isinstance(result, Proposition)
        assert result.name == "p"

    def test_disjunction_with_false_right(self) -> None:
        """p | FALSE simplifies to p."""
        f = Disjunction(Proposition("p"), FalseConstant())
        result = simplify(f)
        assert isinstance(result, Proposition)

    def test_negation_of_true(self) -> None:
        """!TRUE simplifies to FALSE."""
        f = Negation(TrueConstant())
        result = simplify(f)
        assert isinstance(result, FalseConstant)

    def test_negation_of_false(self) -> None:
        """!FALSE simplifies to TRUE."""
        f = Negation(FalseConstant())
        result = simplify(f)
        assert isinstance(result, TrueConstant)

    def test_no_simplification_needed(self) -> None:
        """Already simple formula is returned unchanged."""
        f = Conjunction(Proposition("p"), Proposition("q"))
        result = simplify(f)
        assert result == f

    def test_recursive_simplification(self) -> None:
        """Simplify applies recursively."""
        # !(!(p & TRUE)) → !(!(p)) → p
        inner = Conjunction(Proposition("p"), TrueConstant())
        f = Negation(Negation(inner))
        result = simplify(f)
        assert isinstance(result, Proposition)
        assert result.name == "p"

    def test_proposition_unchanged(self) -> None:
        f = Proposition("p")
        result = simplify(f)
        assert result == f

    def test_constant_unchanged(self) -> None:
        assert simplify(TrueConstant()) == TrueConstant()
        assert simplify(FalseConstant()) == FalseConstant()


class TestParseAndStringRoundTrip:
    """Test that parsing then stringifying produces consistent results."""

    @pytest.mark.parametrize(
        "formula_str,expected_str",
        [
            ("ready", "ready"),
            ("!p", "!p"),
            ("@p", "@p"),
            ("a & b", "(a & b)"),
            ("a | b", "(a | b)"),
            ("a -> b", "(a -> b)"),
            ("a <-> b", "(a <-> b)"),
            ("a S b", "(a S b)"),
            ("TRUE", "TRUE"),
            ("FALSE", "FALSE"),
        ],
    )
    def test_parse_to_string(self, formula_str: str, expected_str: str) -> None:
        result = parse_formula(formula_str)
        assert to_string(result) == expected_str
