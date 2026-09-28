"""Spoken command parsing.

The command set is intentionally small. These tests pin down both what it
accepts and - just as importantly - what it refuses.
"""

from __future__ import annotations

import pytest

from scriptora.services.command_service import CommandKind, parse_command


@pytest.mark.parametrize(
    "phrase",
    [
        "Correct the last subtitle.",
        "correct the last subtitle",
        "Fix the last subtitle",
        "correct last subtitle",
        "Fix the current subtitle",
        "Correct the latest caption",
        "Correct the last line please.",
        "Fix the most recent subtitle",
    ],
)
def test_correct_last_variants(phrase):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.CORRECT_LAST
    assert parsed.is_supported


@pytest.mark.parametrize(
    "phrase",
    [
        "Fix the previous subtitle.",
        "Correct the previous line",
        "fix previous subtitle",
        "Correct the prior caption",
    ],
)
def test_correct_previous_variants(phrase):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.CORRECT_PREVIOUS


@pytest.mark.parametrize(
    "phrase,find,replace",
    [
        ("Change fast API to FastAPI.", "fast API", "FastAPI"),
        ("Replace Alice with Atlas.", "Alice", "Atlas"),
        ("replace postgres with postgresql", "postgres", "postgresql"),
        ("Swap Python for AssemblyAI", "Python", "AssemblyAI"),
        ("Change the name Dave to David", "the name Dave", "David"),
    ],
)
def test_replace_variants(phrase, find, replace):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.REPLACE
    assert parsed.find == find
    assert parsed.replace == replace


@pytest.mark.parametrize(
    "phrase",
    [
        "Remember FastAPI as a technical term.",
        "remember Atlas",
        "Add Atlas to the vocabulary",
    ],
)
def test_remember_variants(phrase):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.REMEMBER
    assert parsed.find


def test_remember_strips_trailing_as_clause():
    parsed = parse_command("Remember FastAPI as a technical term.")
    assert parsed.find == "FastAPI"


@pytest.mark.parametrize(
    "phrase",
    [
        "",
        "   ",
        "What is the weather like today",
        "Um, so anyway, yeah",
    ],
)
def test_unrecognised_commands_are_unsupported_not_guessed(phrase):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.UNSUPPORTED
    assert parsed.is_supported is False
    # A helpful hint is always provided.
    assert parsed.reason


@pytest.mark.parametrize(
    "phrase",
    [
        "Delete the last subtitle",
        "Undo that correction",
        "Translate this into Spanish",
        "Export the transcript",
    ],
)
def test_out_of_scope_operations_are_reported_honestly(phrase):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.UNSUPPORTED
    assert "not supported" in parsed.reason.lower()


def test_ordinary_dictation_is_not_treated_as_a_command():
    """Speech must not be hijacked by the command parser."""
    parsed = parse_command("I need to remember to lock the door before I leave")

    # "remember" appears, but this is dictation, not a vocabulary command.
    assert parsed.kind is not CommandKind.REMEMBER or parsed.find != ""


def test_replace_missing_target_is_unsupported():
    parsed = parse_command("change it to something")
    assert parsed.kind is CommandKind.UNSUPPORTED


def test_parse_never_raises_on_junk():
    for junk in [None, "\x00", "%%%", "a" * 5000]:
        parsed = parse_command(junk)  # type: ignore[arg-type]
        assert parsed.kind is CommandKind.UNSUPPORTED
