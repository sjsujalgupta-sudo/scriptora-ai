"""Project context and vocabulary."""

from __future__ import annotations

import pytest

from scriptora.models.context import make_lookup_pattern, normalize_term
from scriptora.services.context_service import DEFAULT_VOCABULARY, ContextService


def test_seeded_with_demo_vocabulary(context: ContextService):
    assert context.terms == list(DEFAULT_VOCABULARY)


def test_fastapi_is_not_seeded(context: ContextService):
    """The primary demo teaches FastAPI by voice, so it must start unknown."""
    assert "FastAPI" not in context.terms
    assert context.context.has_term("fastapi") is False


def test_add_term_reports_newly_added(context: ContextService):
    term, added = context.add_term("FastAPI")

    assert term == "FastAPI"
    assert added is True
    assert "FastAPI" in context.terms


def test_adding_an_already_seeded_term_is_not_new(context: ContextService):
    term, added = context.add_term("AssemblyAI")

    assert term == "AssemblyAI"
    assert added is False


def test_add_new_term_is_added(context: ContextService):
    _term, added = context.add_term("Reflex")

    assert added is True
    assert "Reflex" in context.terms


def test_add_term_is_idempotent(context: ContextService):
    context.add_term("Reflex")
    context.add_term("reflex")
    context.add_term("  REFLEX  ")

    assert context.terms.count("Reflex") == 1


def test_add_term_normalizes_whitespace(context: ContextService):
    context.add_term("  Assembly   AI  ")

    assert context.terms.count("Assembly AI") == 1


def test_add_term_rejects_empty(context: ContextService):
    with pytest.raises(ValueError):
        context.add_term("   ")


def test_add_term_rejects_overlong(context: ContextService):
    with pytest.raises(ValueError):
        context.add_term("x" * 200)


def test_add_term_rejects_unusable_characters(context: ContextService):
    """A term with no alphanumeric content cannot be looked up."""
    with pytest.raises(ValueError):
        context.add_term("+++")


def test_has_term_is_case_insensitive(context: ContextService):
    assert context.context.has_term("assemblyai") is True
    assert context.context.has_term("ASSEMBLYAI") is True
    assert context.context.has_term("FASTA") is False


def test_entry_matches_loose_spelling(context: ContextService):
    # "fast API" is how speech-to-text actually emits "FastAPI".
    context.add_term("FastAPI")
    assert context.context.entry("fast API").canonical == "FastAPI"


def test_remove_term(context: ContextService):
    assert context.remove_term("Atlas") is True
    assert "Atlas" not in context.terms
    assert context.remove_term("Atlas") is False


def test_restore_terms_canonicalises_spelling(context: ContextService):
    context.add_term("FastAPI")
    assert (
        context.restore_terms("Today we are building with fast API and assembly AI.")
        == "Today we are building with FastAPI and AssemblyAI."
    )


def test_restore_terms_is_idempotent(context: ContextService):
    context.add_term("FastAPI")
    once = context.restore_terms("using fast api")
    assert context.restore_terms(once) == "using FastAPI"


def test_restore_terms_leaves_unrelated_text_alone(context: ContextService):
    text = "nothing to see here"
    assert context.restore_terms(text) == text


def test_restore_terms_handles_empty(context: ContextService):
    assert context.restore_terms("") == ""


def test_vocabulary_in_text(context: ContextService):
    context.add_term("FastAPI")
    hits = context.vocabulary_in_text("we use FastAPI and Atlas here")

    assert set(hits) == {"FastAPI", "Atlas"}
    assert "Python" not in hits


def test_keyterms_are_returned_as_a_list(context: ContextService):
    """AssemblyAI's keyterms_prompt is typed list[str]; a string breaks the session."""
    keyterms = context.keyterms()

    assert isinstance(keyterms, list)
    assert all(isinstance(term, str) for term in keyterms)
    assert "AssemblyAI" in keyterms


def test_lookup_pattern_tolerates_whitespace_variation():
    pattern = make_lookup_pattern("FastAPI")
    assert pattern.search("using fast API")
    assert pattern.search("using  fast   api")
    assert not pattern.search("using fastboard")


def test_normalize_term_collapses_spaces():
    assert normalize_term("  a   b  ") == "a b"


# --------------------------------------------------------- remember parsing
@pytest.mark.parametrize(
    "command,expected",
    [
        ("Remember FastAPI as a technical term.", "FastAPI"),
        ("remember Atlas", "Atlas"),
        ("Remember to use AssemblyAI.", "AssemblyAI"),
        ("Add Atlas to the vocabulary", "Atlas"),
        ("Please remember Kubernetes as a tool", "Kubernetes"),
    ],
)
def test_parse_remember_command(context: ContextService, command, expected):
    assert context.parse_remember_command(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "Correct the last subtitle.",
        "Change fast API to FastAPI.",
        "What is the weather today",
    ],
)
def test_parse_remember_command_rejects_non_remember(context: ContextService, command):
    assert context.parse_remember_command(command) is None
