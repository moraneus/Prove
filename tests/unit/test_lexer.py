"""
Tests for the EPLTL lexical analyzer.

Tests cover tokenization of propositions, constants, all operator variants,
delimiters, whitespace handling, comments, error handling, and the
critical distinction between 'S' (Since operator) and S-starting propositions.
"""

import pytest

from prove.parser.lexer import EPLTLLexer, LexerError


@pytest.fixture
def lexer() -> EPLTLLexer:
    """Return a fresh lexer instance."""
    return EPLTLLexer()


def _tokens(lexer: EPLTLLexer, text: str) -> list[tuple[str, str]]:
    """Helper: return list of (type, value) pairs from tokenizing text."""
    return [(tok.type, tok.value) for tok in lexer.tokenize(text)]


def _types(lexer: EPLTLLexer, text: str) -> list[str]:
    """Helper: return list of token types from tokenizing text."""
    return [tok.type for tok in lexer.tokenize(text)]


class TestPropositions:
    """Test proposition tokenization."""

    def test_simple_proposition(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "ready")
        assert toks == [("PROP", "ready")]

    def test_single_letter(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "p")
        assert toks == [("PROP", "p")]

    def test_proposition_with_underscore(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "is_valid")
        assert toks == [("PROP", "is_valid")]

    def test_proposition_starting_with_underscore(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "_private")
        assert toks == [("PROP", "_private")]

    def test_proposition_with_digits(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "p1")
        assert toks == [("PROP", "p1")]

    def test_proposition_with_apostrophe(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "x'")
        assert toks == [("PROP", "x'")]

    def test_proposition_with_dot(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "state.active")
        assert toks == [("PROP", "state.active")]

    def test_proposition_with_mixed_chars(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "P_done.flag'")
        assert toks == [("PROP", "P_done.flag'")]

    def test_multiple_propositions(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "ready done waiting")
        assert types == ["PROP", "PROP", "PROP"]


class TestConstants:
    """Test TRUE and FALSE constant tokenization."""

    def test_true_upper(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "TRUE")
        assert toks == [("TRUE", "TRUE")]

    def test_true_lower(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "true")
        assert toks == [("TRUE", "true")]

    def test_false_upper(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "FALSE")
        assert toks == [("FALSE", "FALSE")]

    def test_false_lower(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "false")
        assert toks == [("FALSE", "false")]


class TestNegationOperator:
    """Test negation operator variants."""

    def test_bang(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "!p")
        assert types == ["NOT", "PROP"]

    def test_not_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "not p")
        assert types == ["NOT", "PROP"]

    def test_unicode_negation(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "\u00acp")
        assert types == ["NOT", "PROP"]


class TestYesterdayOperator:
    """Test yesterday operator variants."""

    def test_at_sign(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "@p")
        assert types == ["YESTERDAY", "PROP"]

    def test_y_keyword(self, lexer: EPLTLLexer) -> None:
        # Standalone 'Y' is YESTERDAY
        types = _types(lexer, "Y p")
        assert types == ["YESTERDAY", "PROP"]

    def test_prev_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "prev p")
        assert types == ["YESTERDAY", "PROP"]

    def test_previous_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "previous p")
        assert types == ["YESTERDAY", "PROP"]


class TestConjunctionOperator:
    """Test conjunction operator variants."""

    def test_single_amp(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a & b")
        assert types == ["PROP", "AND", "PROP"]

    def test_double_amp(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a && b")
        assert types == ["PROP", "AND", "PROP"]

    def test_and_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a and b")
        assert types == ["PROP", "AND", "PROP"]

    def test_unicode_and(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a \u2227 b")
        assert types == ["PROP", "AND", "PROP"]


class TestDisjunctionOperator:
    """Test disjunction operator variants."""

    def test_single_pipe(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a | b")
        assert types == ["PROP", "OR", "PROP"]

    def test_double_pipe(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a || b")
        assert types == ["PROP", "OR", "PROP"]

    def test_or_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a or b")
        assert types == ["PROP", "OR", "PROP"]

    def test_unicode_or(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a \u2228 b")
        assert types == ["PROP", "OR", "PROP"]


class TestImplicationOperator:
    """Test implication operator variants."""

    def test_arrow(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a -> b")
        assert types == ["PROP", "IMPLIES", "PROP"]

    def test_implies_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a implies b")
        assert types == ["PROP", "IMPLIES", "PROP"]

    def test_unicode_implies(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a \u2192 b")
        assert types == ["PROP", "IMPLIES", "PROP"]


class TestBiconditionalOperator:
    """Test biconditional operator variants."""

    def test_double_arrow(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a <-> b")
        assert types == ["PROP", "IFF", "PROP"]

    def test_iff_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a iff b")
        assert types == ["PROP", "IFF", "PROP"]

    def test_unicode_iff(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a \u2194 b")
        assert types == ["PROP", "IFF", "PROP"]


class TestSinceOperator:
    """Test Since operator and S-starting proposition disambiguation."""

    def test_standalone_s(self, lexer: EPLTLLexer) -> None:
        """Standalone 'S' between propositions is SINCE."""
        types = _types(lexer, "a S b")
        assert types == ["PROP", "SINCE", "PROP"]

    def test_since_keyword(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "a since b")
        assert types == ["PROP", "SINCE", "PROP"]

    def test_s_starting_prop(self, lexer: EPLTLLexer) -> None:
        """Propositions starting with S (like 'Start') are PROP, not SINCE."""
        toks = _tokens(lexer, "Start")
        assert toks == [("PROP", "Start")]

    def test_state_prop(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "State")
        assert toks == [("PROP", "State")]

    def test_s_prop_lowercase_after(self, lexer: EPLTLLexer) -> None:
        """'Sa' should be a proposition, not SINCE + PROP."""
        toks = _tokens(lexer, "Sa")
        assert toks == [("PROP", "Sa")]

    def test_s_with_underscore(self, lexer: EPLTLLexer) -> None:
        """'S_flag' should be a proposition."""
        toks = _tokens(lexer, "S_flag")
        assert toks == [("PROP", "S_flag")]

    def test_since_in_formula(self, lexer: EPLTLLexer) -> None:
        """S in a real formula context."""
        types = _types(lexer, "valid S started")
        assert types == ["PROP", "SINCE", "PROP"]


class TestDelimiters:
    """Test parentheses tokenization."""

    def test_lparen(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "(")
        assert types == ["LPAREN"]

    def test_rparen(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, ")")
        assert types == ["RPAREN"]

    def test_nested_parens(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "((a))")
        assert types == ["LPAREN", "LPAREN", "PROP", "RPAREN", "RPAREN"]


class TestWhitespace:
    """Test whitespace handling."""

    def test_spaces_between_tokens(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "a  &  b")
        assert len(toks) == 3

    def test_tabs(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "a\t&\tb")
        assert len(toks) == 3

    def test_newlines(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "a\n&\nb")
        assert len(toks) == 3

    def test_no_spaces(self, lexer: EPLTLLexer) -> None:
        """Operators like ! and @ don't need surrounding spaces."""
        types = _types(lexer, "!p")
        assert types == ["NOT", "PROP"]


class TestComments:
    """Test comment handling."""

    def test_full_line_comment(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "# this is a comment")
        assert toks == []

    def test_comment_after_formula(self, lexer: EPLTLLexer) -> None:
        toks = _tokens(lexer, "a & b # conjunction")
        assert len(toks) == 3


class TestComplexExpressions:
    """Test tokenization of complex formula strings."""

    def test_implication_formula(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "done -> (confirmed S ready)")
        assert types == ["PROP", "IMPLIES", "LPAREN", "PROP", "SINCE", "PROP", "RPAREN"]

    def test_negation_of_conjunction(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "!(a & b)")
        assert types == ["NOT", "LPAREN", "PROP", "AND", "PROP", "RPAREN"]

    def test_yesterday_nested(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "@(a S b)")
        assert types == ["YESTERDAY", "LPAREN", "PROP", "SINCE", "PROP", "RPAREN"]

    def test_double_negation(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "!!p")
        assert types == ["NOT", "NOT", "PROP"]

    def test_complex_safety_property(self, lexer: EPLTLLexer) -> None:
        types = _types(lexer, "(done & confirmed) -> ((valid S ready) & @initialized)")
        expected = [
            "LPAREN",
            "PROP",
            "AND",
            "PROP",
            "RPAREN",
            "IMPLIES",
            "LPAREN",
            "LPAREN",
            "PROP",
            "SINCE",
            "PROP",
            "RPAREN",
            "AND",
            "YESTERDAY",
            "PROP",
            "RPAREN",
        ]
        assert types == expected

    def test_biconditional_with_once(self, lexer: EPLTLLexer) -> None:
        """TRUE S request represents 'Once request'."""
        types = _types(lexer, "locked <-> (TRUE S acquire)")
        assert types == ["PROP", "IFF", "LPAREN", "TRUE", "SINCE", "PROP", "RPAREN"]


class TestKeywordVsProposition:
    """Test that keywords are distinguished from propositions correctly."""

    def test_true_is_constant(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "TRUE") == ["TRUE"]

    def test_trueness_is_prop(self, lexer: EPLTLLexer) -> None:
        """Words starting with 'true' but longer are propositions."""
        assert _tokens(lexer, "trueness") == [("PROP", "trueness")]

    def test_false_is_constant(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "FALSE") == ["FALSE"]

    def test_falsehood_is_prop(self, lexer: EPLTLLexer) -> None:
        assert _tokens(lexer, "falsehood") == [("PROP", "falsehood")]

    def test_not_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "not p") == ["NOT", "PROP"]

    def test_nothing_is_prop(self, lexer: EPLTLLexer) -> None:
        assert _tokens(lexer, "nothing") == [("PROP", "nothing")]

    def test_and_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "a and b") == ["PROP", "AND", "PROP"]

    def test_android_is_prop(self, lexer: EPLTLLexer) -> None:
        assert _tokens(lexer, "android") == [("PROP", "android")]

    def test_or_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "a or b") == ["PROP", "OR", "PROP"]

    def test_order_is_prop(self, lexer: EPLTLLexer) -> None:
        assert _tokens(lexer, "order") == [("PROP", "order")]

    def test_implies_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "a implies b") == ["PROP", "IMPLIES", "PROP"]

    def test_iff_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "a iff b") == ["PROP", "IFF", "PROP"]

    def test_since_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "a since b") == ["PROP", "SINCE", "PROP"]

    def test_prev_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "prev p") == ["YESTERDAY", "PROP"]

    def test_previous_is_operator(self, lexer: EPLTLLexer) -> None:
        assert _types(lexer, "previous p") == ["YESTERDAY", "PROP"]

    def test_y_is_yesterday(self, lexer: EPLTLLexer) -> None:
        """Standalone 'Y' is YESTERDAY operator."""
        assert _types(lexer, "Y p") == ["YESTERDAY", "PROP"]

    def test_yes_is_prop(self, lexer: EPLTLLexer) -> None:
        """'Yes' (more than one char starting with Y) is a proposition."""
        assert _tokens(lexer, "Yes") == [("PROP", "Yes")]


class TestErrorHandling:
    """Test lexer error handling."""

    def test_invalid_char_dollar(self, lexer: EPLTLLexer) -> None:
        with pytest.raises(LexerError):
            list(lexer.tokenize("a $ b"))

    def test_invalid_char_semicolon(self, lexer: EPLTLLexer) -> None:
        with pytest.raises(LexerError):
            list(lexer.tokenize("a ; b"))

    def test_invalid_char_at_start(self, lexer: EPLTLLexer) -> None:
        with pytest.raises(LexerError):
            list(lexer.tokenize("$invalid"))

    def test_empty_input(self, lexer: EPLTLLexer) -> None:
        """Empty input produces no tokens (not an error)."""
        toks = _tokens(lexer, "")
        assert toks == []

    def test_only_whitespace(self, lexer: EPLTLLexer) -> None:
        """Whitespace-only input produces no tokens."""
        toks = _tokens(lexer, "   \t  \n  ")
        assert toks == []
