"""
Tests for the AST node definitions.

Tests cover creation, subformula computation, string representation,
equality, hashing, and derived operators for all EPLTL formula types.
"""

import pytest

from prove.parser.ast_nodes import (
    Biconditional,
    Conjunction,
    Disjunction,
    FalseConstant,
    Formula,
    Historically,
    Implication,
    Negation,
    Once,
    Proposition,
    Since,
    TrueConstant,
    Yesterday,
)


class TestConstants:
    """Test TrueConstant and FalseConstant."""

    def test_true_constant(self) -> None:
        t = TrueConstant()
        assert str(t) == "TRUE"

    def test_false_constant(self) -> None:
        f = FalseConstant()
        assert str(f) == "FALSE"

    def test_true_subformulas(self) -> None:
        t = TrueConstant()
        subs = t.subformulas()
        assert t in subs
        assert len(subs) == 1

    def test_false_subformulas(self) -> None:
        f = FalseConstant()
        subs = f.subformulas()
        assert f in subs
        assert len(subs) == 1

    def test_true_equality(self) -> None:
        assert TrueConstant() == TrueConstant()

    def test_false_equality(self) -> None:
        assert FalseConstant() == FalseConstant()

    def test_true_not_equal_false(self) -> None:
        assert TrueConstant() != FalseConstant()

    def test_true_hash(self) -> None:
        assert hash(TrueConstant()) == hash(TrueConstant())

    def test_false_hash(self) -> None:
        assert hash(FalseConstant()) == hash(FalseConstant())


class TestProposition:
    """Test Proposition node."""

    def test_create_simple(self) -> None:
        p = Proposition("ready")
        assert p.name == "ready"

    def test_str(self) -> None:
        p = Proposition("ready")
        assert str(p) == "ready"

    def test_proposition_with_underscore(self) -> None:
        p = Proposition("is_valid")
        assert p.name == "is_valid"
        assert str(p) == "is_valid"

    def test_proposition_with_apostrophe(self) -> None:
        p = Proposition("x'")
        assert p.name == "x'"
        assert str(p) == "x'"

    def test_proposition_with_dot(self) -> None:
        p = Proposition("state.active")
        assert p.name == "state.active"

    def test_subformulas(self) -> None:
        p = Proposition("ready")
        subs = p.subformulas()
        assert p in subs
        assert len(subs) == 1

    def test_equality(self) -> None:
        assert Proposition("ready") == Proposition("ready")

    def test_inequality(self) -> None:
        assert Proposition("ready") != Proposition("done")

    def test_hash(self) -> None:
        assert hash(Proposition("ready")) == hash(Proposition("ready"))

    def test_usable_in_set(self) -> None:
        s = {Proposition("ready"), Proposition("ready"), Proposition("done")}
        assert len(s) == 2

    def test_not_equal_to_non_proposition(self) -> None:
        assert Proposition("ready") != TrueConstant()
        assert Proposition("TRUE") != TrueConstant()


class TestNegation:
    """Test Negation node."""

    def test_create(self) -> None:
        n = Negation(Proposition("p"))
        assert isinstance(n.operand, Proposition)

    def test_str(self) -> None:
        n = Negation(Proposition("p"))
        assert str(n) == "!p"

    def test_double_negation_str(self) -> None:
        n = Negation(Negation(Proposition("p")))
        assert str(n) == "!!p"

    def test_subformulas(self) -> None:
        p = Proposition("p")
        n = Negation(p)
        subs = n.subformulas()
        assert n in subs
        assert p in subs
        assert len(subs) == 2

    def test_equality(self) -> None:
        assert Negation(Proposition("p")) == Negation(Proposition("p"))

    def test_inequality(self) -> None:
        assert Negation(Proposition("p")) != Negation(Proposition("q"))

    def test_hash(self) -> None:
        assert hash(Negation(Proposition("p"))) == hash(Negation(Proposition("p")))

    def test_negation_of_compound(self) -> None:
        n = Negation(Conjunction(Proposition("p"), Proposition("q")))
        assert str(n) == "!(p & q)"


class TestYesterday:
    """Test Yesterday node."""

    def test_create(self) -> None:
        y = Yesterday(Proposition("p"))
        assert isinstance(y.operand, Proposition)

    def test_str(self) -> None:
        y = Yesterday(Proposition("p"))
        assert str(y) == "@p"

    def test_subformulas(self) -> None:
        p = Proposition("p")
        y = Yesterday(p)
        subs = y.subformulas()
        assert y in subs
        assert p in subs
        assert len(subs) == 2

    def test_equality(self) -> None:
        assert Yesterday(Proposition("p")) == Yesterday(Proposition("p"))

    def test_hash(self) -> None:
        assert hash(Yesterday(Proposition("p"))) == hash(Yesterday(Proposition("p")))

    def test_nested_yesterday(self) -> None:
        y = Yesterday(Yesterday(Proposition("p")))
        assert str(y) == "@@p"


class TestConjunction:
    """Test Conjunction node."""

    def test_create(self) -> None:
        c = Conjunction(Proposition("p"), Proposition("q"))
        assert isinstance(c.left, Proposition)
        assert isinstance(c.right, Proposition)

    def test_str(self) -> None:
        c = Conjunction(Proposition("p"), Proposition("q"))
        assert str(c) == "(p & q)"

    def test_subformulas(self) -> None:
        p = Proposition("p")
        q = Proposition("q")
        c = Conjunction(p, q)
        subs = c.subformulas()
        assert c in subs
        assert p in subs
        assert q in subs
        assert len(subs) == 3

    def test_equality(self) -> None:
        c1 = Conjunction(Proposition("p"), Proposition("q"))
        c2 = Conjunction(Proposition("p"), Proposition("q"))
        assert c1 == c2

    def test_inequality_different_operands(self) -> None:
        c1 = Conjunction(Proposition("p"), Proposition("q"))
        c2 = Conjunction(Proposition("q"), Proposition("p"))
        assert c1 != c2

    def test_hash(self) -> None:
        c1 = Conjunction(Proposition("p"), Proposition("q"))
        c2 = Conjunction(Proposition("p"), Proposition("q"))
        assert hash(c1) == hash(c2)


class TestDisjunction:
    """Test Disjunction node."""

    def test_create_and_str(self) -> None:
        d = Disjunction(Proposition("p"), Proposition("q"))
        assert str(d) == "(p | q)"

    def test_subformulas(self) -> None:
        p = Proposition("p")
        q = Proposition("q")
        d = Disjunction(p, q)
        subs = d.subformulas()
        assert d in subs
        assert p in subs
        assert q in subs
        assert len(subs) == 3

    def test_equality(self) -> None:
        d1 = Disjunction(Proposition("p"), Proposition("q"))
        d2 = Disjunction(Proposition("p"), Proposition("q"))
        assert d1 == d2


class TestImplication:
    """Test Implication node."""

    def test_create_and_str(self) -> None:
        i = Implication(Proposition("p"), Proposition("q"))
        assert str(i) == "(p -> q)"

    def test_subformulas(self) -> None:
        p = Proposition("p")
        q = Proposition("q")
        i = Implication(p, q)
        subs = i.subformulas()
        assert i in subs
        assert p in subs
        assert q in subs
        assert len(subs) == 3

    def test_equality(self) -> None:
        i1 = Implication(Proposition("p"), Proposition("q"))
        i2 = Implication(Proposition("p"), Proposition("q"))
        assert i1 == i2


class TestBiconditional:
    """Test Biconditional node."""

    def test_create_and_str(self) -> None:
        b = Biconditional(Proposition("p"), Proposition("q"))
        assert str(b) == "(p <-> q)"

    def test_subformulas(self) -> None:
        p = Proposition("p")
        q = Proposition("q")
        b = Biconditional(p, q)
        subs = b.subformulas()
        assert b in subs
        assert p in subs
        assert q in subs
        assert len(subs) == 3

    def test_equality(self) -> None:
        b1 = Biconditional(Proposition("p"), Proposition("q"))
        b2 = Biconditional(Proposition("p"), Proposition("q"))
        assert b1 == b2


class TestSince:
    """Test Since node."""

    def test_create(self) -> None:
        s = Since(Proposition("p"), Proposition("q"))
        assert isinstance(s.left, Proposition)
        assert isinstance(s.right, Proposition)

    def test_str(self) -> None:
        s = Since(Proposition("p"), Proposition("q"))
        assert str(s) == "(p S q)"

    def test_subformulas(self) -> None:
        p = Proposition("p")
        q = Proposition("q")
        s = Since(p, q)
        subs = s.subformulas()
        assert s in subs
        assert p in subs
        assert q in subs
        assert len(subs) == 3

    def test_equality(self) -> None:
        s1 = Since(Proposition("p"), Proposition("q"))
        s2 = Since(Proposition("p"), Proposition("q"))
        assert s1 == s2

    def test_hash(self) -> None:
        s1 = Since(Proposition("p"), Proposition("q"))
        s2 = Since(Proposition("p"), Proposition("q"))
        assert hash(s1) == hash(s2)


class TestOnce:
    """Test Once (derived operator)."""

    def test_create(self) -> None:
        o = Once(Proposition("p"))
        assert isinstance(o.operand, Proposition)

    def test_str(self) -> None:
        o = Once(Proposition("p"))
        assert str(o) == "(TRUE S p)"

    def test_subformulas(self) -> None:
        """Once(p) expands to Since(TRUE, p), so subformulas include TRUE."""
        p = Proposition("p")
        o = Once(p)
        subs = o.subformulas()
        assert o in subs
        assert p in subs
        # TRUE constant should be in subformulas
        has_true = any(isinstance(s, TrueConstant) for s in subs)
        assert has_true

    def test_equality(self) -> None:
        assert Once(Proposition("p")) == Once(Proposition("p"))


class TestHistorically:
    """Test Historically (derived operator)."""

    def test_create(self) -> None:
        h = Historically(Proposition("p"))
        assert isinstance(h.operand, Proposition)

    def test_str(self) -> None:
        h = Historically(Proposition("p"))
        # Historically = !(TRUE S !p)
        assert str(h) == "!(TRUE S !p)"

    def test_subformulas(self) -> None:
        """Historically(p) = !(TRUE S !p), subformulas include all parts."""
        p = Proposition("p")
        h = Historically(p)
        subs = h.subformulas()
        assert h in subs
        assert p in subs

    def test_equality(self) -> None:
        assert Historically(Proposition("p")) == Historically(Proposition("p"))


class TestComplexFormulas:
    """Test complex nested formulas."""

    def test_done_implies_confirmed_since_ready(self) -> None:
        """done -> (confirmed S ready)"""
        f = Implication(Proposition("done"), Since(Proposition("confirmed"), Proposition("ready")))
        assert str(f) == "(done -> (confirmed S ready))"

    def test_nested_subformulas(self) -> None:
        """Subformulas of a nested formula contain all parts."""
        p = Proposition("p")
        q = Proposition("q")
        r = Proposition("r")
        f = Implication(Conjunction(p, q), r)
        subs = f.subformulas()
        assert f in subs
        assert Conjunction(p, q) in subs
        assert p in subs
        assert q in subs
        assert r in subs
        assert len(subs) == 5

    def test_deeply_nested(self) -> None:
        """Deep nesting works correctly."""
        f = Negation(Yesterday(Since(Proposition("a"), Proposition("b"))))
        assert str(f) == "!@(a S b)"
        subs = f.subformulas()
        assert len(subs) == 5  # !, @, S, a, b

    def test_shared_subformulas_deduplicated(self) -> None:
        """When the same proposition appears multiple times, subformulas dedup."""
        p = Proposition("p")
        f = Conjunction(p, Negation(p))
        subs = f.subformulas()
        # p, !p, (p & !p) = 3 unique subformulas
        assert len(subs) == 3

    def test_complex_formula_usable_as_dict_key(self) -> None:
        """Complex formulas can be used as dictionary keys."""
        f1 = Implication(Proposition("p"), Proposition("q"))
        f2 = Implication(Proposition("p"), Proposition("q"))
        d = {f1: "value"}
        assert d[f2] == "value"

    def test_negation_of_conjunction_str(self) -> None:
        """Negation of binary uses parentheses."""
        f = Negation(Conjunction(Proposition("a"), Proposition("b")))
        assert str(f) == "!(a & b)"

    def test_yesterday_of_since_str(self) -> None:
        """Yesterday of binary uses parentheses."""
        f = Yesterday(Since(Proposition("a"), Proposition("b")))
        assert str(f) == "@(a S b)"
