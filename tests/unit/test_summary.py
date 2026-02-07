"""
Tests for the EPLTL summary implementation.

Tests cover initial summary creation, proposition evaluation, boolean
operator updates, temporal operator updates (Yesterday, Since),
multi-event sequences, and equality/hashing.
"""

import pytest

from prove.core.event import Event
from prove.core.summary import Summary
from prove.core.vector_clock import VectorClock
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

PROCS = frozenset({"P1", "P2"})


def _vc(p1: int, p2: int) -> VectorClock:
    return VectorClock(PROCS, initial_values={"P1": p1, "P2": p2})


def _event(
    eid: str,
    process: str,
    p1: int,
    p2: int,
    timestamp: float,
    props: frozenset[str] = frozenset(),
) -> Event:
    return Event(
        eid=eid,
        process=process,
        vector_clock=_vc(p1, p2),
        timestamp=timestamp,
        propositions=props,
    )


class TestInitialSummary:
    """Test initial summary creation."""

    def test_initial_simple_prop(self) -> None:
        """Initial summary for a proposition is False."""
        p = Proposition("ready")
        s = Summary.initial(p)
        assert s.evaluate(p) is False

    def test_initial_true_constant(self) -> None:
        """TRUE constant is True in initial summary."""
        t = TrueConstant()
        s = Summary.initial(t)
        assert s.evaluate(t) is True

    def test_initial_false_constant(self) -> None:
        """FALSE constant is False in initial summary."""
        f = FalseConstant()
        s = Summary.initial(f)
        assert s.evaluate(f) is False

    def test_initial_negation(self) -> None:
        """!p initial: p=False, so !p=True."""
        p = Proposition("p")
        neg = Negation(p)
        s = Summary.initial(neg)
        assert s.evaluate(p) is False
        assert s.evaluate(neg) is True

    def test_initial_conjunction(self) -> None:
        """p & q initial: both False, so conjunction is False."""
        p = Proposition("p")
        q = Proposition("q")
        conj = Conjunction(p, q)
        s = Summary.initial(conj)
        assert s.evaluate(conj) is False

    def test_initial_since(self) -> None:
        """p S q initial: both False, so since is False."""
        p = Proposition("p")
        q = Proposition("q")
        since = Since(p, q)
        s = Summary.initial(since)
        assert s.evaluate(since) is False

    def test_initial_yesterday(self) -> None:
        """@p initial: False (no previous state)."""
        p = Proposition("p")
        y = Yesterday(p)
        s = Summary.initial(y)
        assert s.evaluate(y) is False

    def test_initial_complex(self) -> None:
        """done -> (confirmed S ready): done=False, so implication=True."""
        done = Proposition("done")
        confirmed = Proposition("confirmed")
        ready = Proposition("ready")
        formula = Implication(done, Since(confirmed, ready))
        s = Summary.initial(formula)
        # done=False, so implication is True (False -> anything = True)
        assert s.evaluate(formula) is True

    def test_initial_all_subformulas_present(self) -> None:
        """Initial summary has entries for every subformula."""
        p = Proposition("p")
        q = Proposition("q")
        formula = Conjunction(p, Negation(q))
        s = Summary.initial(formula)
        for sub in formula.subformulas():
            # Should not raise
            s.evaluate(sub)


class TestPropositionUpdate:
    """Test proposition evaluation after event updates."""

    def test_event_with_proposition(self) -> None:
        """Event containing proposition makes it True."""
        p = Proposition("ready")
        s = Summary.initial(p)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"ready"}))
        s2 = s.update(e, p)
        assert s2.evaluate(p) is True

    def test_event_without_proposition(self) -> None:
        """Event not containing proposition makes it False."""
        p = Proposition("ready")
        s = Summary.initial(p)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"other"}))
        s2 = s.update(e, p)
        assert s2.evaluate(p) is False

    def test_proposition_changes_between_events(self) -> None:
        """Proposition can change between events."""
        p = Proposition("ready")
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"ready"}))
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset())
        s = Summary.initial(p)
        s1 = s.update(e1, p)
        assert s1.evaluate(p) is True
        s2 = s1.update(e2, p)
        assert s2.evaluate(p) is False


class TestBooleanOperators:
    """Test boolean operator updates."""

    def test_negation_true(self) -> None:
        """!p when p is True => False."""
        p = Proposition("p")
        neg = Negation(p)
        s = Summary.initial(neg)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s2 = s.update(e, neg)
        assert s2.evaluate(p) is True
        assert s2.evaluate(neg) is False

    def test_negation_false(self) -> None:
        """!p when p is False => True."""
        p = Proposition("p")
        neg = Negation(p)
        s = Summary.initial(neg)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset())
        s2 = s.update(e, neg)
        assert s2.evaluate(p) is False
        assert s2.evaluate(neg) is True

    def test_conjunction_true_true(self) -> None:
        """p & q when both True => True."""
        p = Proposition("p")
        q = Proposition("q")
        conj = Conjunction(p, q)
        s = Summary.initial(conj)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p", "q"}))
        s2 = s.update(e, conj)
        assert s2.evaluate(conj) is True

    def test_conjunction_true_false(self) -> None:
        """p & q when p=True, q=False => False."""
        p = Proposition("p")
        q = Proposition("q")
        conj = Conjunction(p, q)
        s = Summary.initial(conj)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s2 = s.update(e, conj)
        assert s2.evaluate(conj) is False

    def test_disjunction_false_true(self) -> None:
        """p | q when p=False, q=True => True."""
        p = Proposition("p")
        q = Proposition("q")
        disj = Disjunction(p, q)
        s = Summary.initial(disj)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"q"}))
        s2 = s.update(e, disj)
        assert s2.evaluate(disj) is True

    def test_implication_true_false(self) -> None:
        """p -> q when p=True, q=False => False."""
        p = Proposition("p")
        q = Proposition("q")
        impl = Implication(p, q)
        s = Summary.initial(impl)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s2 = s.update(e, impl)
        assert s2.evaluate(impl) is False

    def test_implication_false_false(self) -> None:
        """p -> q when p=False, q=False => True."""
        p = Proposition("p")
        q = Proposition("q")
        impl = Implication(p, q)
        s = Summary.initial(impl)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset())
        s2 = s.update(e, impl)
        assert s2.evaluate(impl) is True

    def test_biconditional_same_values(self) -> None:
        """p <-> q when both True => True."""
        p = Proposition("p")
        q = Proposition("q")
        bic = Biconditional(p, q)
        s = Summary.initial(bic)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p", "q"}))
        s2 = s.update(e, bic)
        assert s2.evaluate(bic) is True

    def test_biconditional_different_values(self) -> None:
        """p <-> q when p=True, q=False => False."""
        p = Proposition("p")
        q = Proposition("q")
        bic = Biconditional(p, q)
        s = Summary.initial(bic)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s2 = s.update(e, bic)
        assert s2.evaluate(bic) is False


class TestYesterdayOperator:
    """Test Yesterday (previous) operator."""

    def test_yesterday_first_event_false(self) -> None:
        """@p after first event update is False (initial value of p was False)."""
        p = Proposition("p")
        y = Yesterday(p)
        s = Summary.initial(y)
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s2 = s.update(e1, y)
        # Yesterday uses OLD value of p, which was False (initial)
        assert s2.evaluate(y) is False

    def test_yesterday_second_event(self) -> None:
        """@p after second event reflects p's value at first event."""
        p = Proposition("p")
        y = Yesterday(p)
        s = Summary.initial(y)
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s1 = s.update(e1, y)
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset())
        s2 = s1.update(e2, y)
        # @p should be True (p was True at first event)
        assert s2.evaluate(y) is True
        # p itself is now False
        assert s2.evaluate(p) is False

    def test_yesterday_tracks_old_value(self) -> None:
        """@p correctly tracks the old value through multiple updates."""
        p = Proposition("p")
        y = Yesterday(p)
        s = Summary.initial(y)

        # Event 1: p=True
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s1 = s.update(e1, y)
        assert s1.evaluate(y) is False  # old p was False (initial)
        assert s1.evaluate(p) is True

        # Event 2: p=False
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset())
        s2 = s1.update(e2, y)
        assert s2.evaluate(y) is True  # old p was True (from e1)
        assert s2.evaluate(p) is False

        # Event 3: p=False
        e3 = _event("e3", "P1", 3, 0, 2.0, frozenset())
        s3 = s2.update(e3, y)
        assert s3.evaluate(y) is False  # old p was False (from e2)


class TestSinceOperator:
    """Test Since temporal operator."""

    def test_since_trigger_true(self) -> None:
        """p S q: when q becomes True, since becomes True."""
        p = Proposition("p")
        q = Proposition("q")
        since = Since(p, q)
        s = Summary.initial(since)
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"q"}))
        s1 = s.update(e1, since)
        assert s1.evaluate(since) is True

    def test_since_maintained(self) -> None:
        """p S q: True stays True if p holds continuously after q was true."""
        p = Proposition("p")
        q = Proposition("q")
        since = Since(p, q)
        s = Summary.initial(since)

        # q becomes True
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"q", "p"}))
        s1 = s.update(e1, since)
        assert s1.evaluate(since) is True

        # p stays True (q can be False now)
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset({"p"}))
        s2 = s1.update(e2, since)
        assert s2.evaluate(since) is True

    def test_since_broken(self) -> None:
        """p S q: becomes False when p stops holding (and q not re-triggered)."""
        p = Proposition("p")
        q = Proposition("q")
        since = Since(p, q)
        s = Summary.initial(since)

        # q becomes True
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"q", "p"}))
        s1 = s.update(e1, since)
        assert s1.evaluate(since) is True

        # p becomes False, q also False
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset())
        s2 = s1.update(e2, since)
        assert s2.evaluate(since) is False

    def test_since_retrigger(self) -> None:
        """p S q: can become True again if q becomes True again."""
        p = Proposition("p")
        q = Proposition("q")
        since = Since(p, q)
        s = Summary.initial(since)

        # q True
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"q"}))
        s1 = s.update(e1, since)
        assert s1.evaluate(since) is True

        # Both False — since broken
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset())
        s2 = s1.update(e2, since)
        assert s2.evaluate(since) is False

        # q True again — since re-established
        e3 = _event("e3", "P1", 3, 0, 2.0, frozenset({"q"}))
        s3 = s2.update(e3, since)
        assert s3.evaluate(since) is True

    def test_once_pattern(self) -> None:
        """TRUE S p: once p was true, always true thereafter."""
        p = Proposition("p")
        t = TrueConstant()
        once = Since(t, p)
        s = Summary.initial(once)

        # p is False
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset())
        s1 = s.update(e1, once)
        assert s1.evaluate(once) is False

        # p becomes True
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset({"p"}))
        s2 = s1.update(e2, once)
        assert s2.evaluate(once) is True

        # p is False again, but once is still True (TRUE S p stays True
        # because TRUE holds now AND (TRUE S p) held before)
        e3 = _event("e3", "P1", 3, 0, 2.0, frozenset())
        s3 = s2.update(e3, once)
        assert s3.evaluate(once) is True


class TestMultiEventSequence:
    """Test summary updates over a sequence of events."""

    def test_three_event_sequence(self) -> None:
        """Verify correct evaluation over a 3-event sequence."""
        p = Proposition("p")
        q = Proposition("q")
        formula = Conjunction(p, q)
        s = Summary.initial(formula)

        # Event 1: p=True, q=False → p&q = False
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s1 = s.update(e1, formula)
        assert s1.evaluate(formula) is False

        # Event 2: p=True, q=True → p&q = True
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset({"p", "q"}))
        s2 = s1.update(e2, formula)
        assert s2.evaluate(formula) is True

        # Event 3: p=False, q=True → p&q = False
        e3 = _event("e3", "P1", 3, 0, 2.0, frozenset({"q"}))
        s3 = s2.update(e3, formula)
        assert s3.evaluate(formula) is False

    def test_complex_formula_sequence(self) -> None:
        """done -> (confirmed S ready) over multiple events."""
        done = Proposition("done")
        confirmed = Proposition("confirmed")
        ready = Proposition("ready")
        formula = Implication(done, Since(confirmed, ready))
        since = Since(confirmed, ready)

        s = Summary.initial(formula)

        # Event 1: ready=True → since becomes True, formula = True
        e1 = _event("e1", "P1", 1, 0, 0.0, frozenset({"ready", "confirmed"}))
        s1 = s.update(e1, formula)
        assert s1.evaluate(since) is True
        assert s1.evaluate(formula) is True

        # Event 2: confirmed=True, done=True → since still True, formula = True
        e2 = _event("e2", "P1", 2, 0, 1.0, frozenset({"confirmed", "done"}))
        s2 = s1.update(e2, formula)
        assert s2.evaluate(since) is True
        assert s2.evaluate(formula) is True

        # Event 3: done=True, confirmed=False, ready=False → since breaks, formula = False
        e3 = _event("e3", "P1", 3, 0, 2.0, frozenset({"done"}))
        s3 = s2.update(e3, formula)
        assert s3.evaluate(since) is False
        assert s3.evaluate(formula) is False


class TestSummaryWithGlobalState:
    """Test summary updates using frontier global state (union of propositions)."""

    def test_update_with_global_state_event(self) -> None:
        """Summary update works when event propositions represent global state."""
        p = Proposition("p")
        q = Proposition("q")
        formula = Disjunction(p, q)
        s = Summary.initial(formula)

        # Simulate global state: create an event with the union of propositions
        e = _event("gs", "P1", 1, 0, 0.0, frozenset({"q"}))
        s1 = s.update(e, formula)
        assert s1.evaluate(p) is False
        assert s1.evaluate(q) is True
        assert s1.evaluate(formula) is True


class TestSummaryEquality:
    """Test equality and hashing."""

    def test_same_summaries_equal(self) -> None:
        p = Proposition("p")
        s1 = Summary.initial(p)
        s2 = Summary.initial(p)
        assert s1 == s2

    def test_same_summaries_same_hash(self) -> None:
        p = Proposition("p")
        s1 = Summary.initial(p)
        s2 = Summary.initial(p)
        assert hash(s1) == hash(s2)

    def test_different_summaries_not_equal(self) -> None:
        p = Proposition("p")
        s1 = Summary.initial(p)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s2 = s1.update(e, p)
        assert s1 != s2

    def test_usable_in_set(self) -> None:
        """Summaries can be stored in sets for deduplication."""
        p = Proposition("p")
        s1 = Summary.initial(p)
        s2 = Summary.initial(p)
        s3 = s1.update(_event("e1", "P1", 1, 0, 0.0, frozenset({"p"})), p)
        result = {s1, s2, s3}
        assert len(result) == 2  # s1 == s2


class TestSummaryImmutability:
    """Test that summary updates return new instances."""

    def test_update_returns_new_instance(self) -> None:
        p = Proposition("p")
        s = Summary.initial(p)
        e = _event("e1", "P1", 1, 0, 0.0, frozenset({"p"}))
        s2 = s.update(e, p)
        assert s is not s2
        # Original unchanged
        assert s.evaluate(p) is False
        assert s2.evaluate(p) is True
