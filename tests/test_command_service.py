"""Spoken command parsing.

The command set is intentionally small. These tests pin down both what it
accepts and - just as importantly - what it refuses.
"""

from __future__ import annotations

import pytest

from scriptora.services.command_service import (
    CommandKind,
    CorrectionTarget,
    parse_command,
)
from scriptora.services.context_service import ContextService


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
    "phrase,term",
    [
        ("Add Kubernete to the vocabulary", "Kubernete"),
        # Case is the speaker's to choose; the tail strip must not alter it.
        ("add kubernete to the vocabulary", "kubernete"),
        ("Add Kubernete to the vocabulary.", "Kubernete"),
        ("Add Atlas to the context", "Atlas"),
        ("Add Redis to the project", "Redis"),
        ("Remember Kubernete", "Kubernete"),
    ],
)
def test_remember_strips_the_phrase_tail_not_the_term(phrase, term):
    """The 'to the vocabulary' tail is phrasing, never part of the term.

    This is the demo's vocabulary step, and a leaked tail would be pushed
    straight to AssemblyAI as a keyterm.
    """
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.REMEMBER
    assert parsed.find == term


@pytest.mark.parametrize(
    "phrase",
    [
        "Add Kubernete to the vocabulary",
        "Remember FastAPI as a technical term.",
        "Add Atlas to the context",
        "Remember Kubernete",
    ],
)
def test_parser_agrees_with_the_context_service(phrase):
    """`command.find` is the fallback when the context parser misses.

    If the two disagree the user silently gets a malformed term, so pin them
    together for every phrasing the demo and docs advertise.
    """
    assert parse_command(phrase).find == ContextService().parse_remember_command(phrase)


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


# ====================================================== natural set-text
#
# "Change <which line> to <text>" is the feature the demo hangs on, so the
# accepted phrasings are pinned exhaustively. Every case here must resolve to
# the *same* target the user meant; a new synonym that silently resolves to
# nothing is worse than a rejection.
@pytest.mark.parametrize(
    "phrase,target,ordinal",
    [
        ('Change this to "To deployed."', CorrectionTarget.THIS, None),
        ("Change this sentence to To deployed.", CorrectionTarget.THIS, None),
        ("Change that to To deployed.", CorrectionTarget.THIS, None),
        ("Change the last sentence to To deployed.", CorrectionTarget.LAST, None),
        ("Change this subtitle to To deployed.", CorrectionTarget.THIS, None),
        ("Change the last subtitle to To deployed.", CorrectionTarget.LAST, None),
        ("Change the last line to To deployed.", CorrectionTarget.LAST, None),
        ("Change the last caption to To deployed.", CorrectionTarget.LAST, None),
        ("Change the last one to To deployed.", CorrectionTarget.LAST, None),
        ("Change the last but one to To deployed.", CorrectionTarget.PREVIOUS, None),
        ("Change the previous sentence to To deployed.", CorrectionTarget.PREVIOUS, None),
        ("Change the previous subtitle to To deployed.", CorrectionTarget.PREVIOUS, None),
        ("Change the prior line to To deployed.", CorrectionTarget.PREVIOUS, None),
        ("Change the preceding caption to To deployed.", CorrectionTarget.PREVIOUS, None),
        ("Change sentence 3 to To deployed.", CorrectionTarget.ORDINAL, 3),
        ("Change subtitle 3 to To deployed.", CorrectionTarget.ORDINAL, 3),
        ("Change line 12 to To deployed.", CorrectionTarget.ORDINAL, 12),
        ("Change the 3rd sentence to To deployed.", CorrectionTarget.ORDINAL, 3),
        ("Change the 3rd subtitle to To deployed.", CorrectionTarget.ORDINAL, 3),
        ("Change the 22nd line to To deployed.", CorrectionTarget.ORDINAL, 22),
        ("Change the first sentence to To deployed.", CorrectionTarget.ORDINAL, 1),
        ("Change the second sentence to To deployed.", CorrectionTarget.ORDINAL, 2),
        ("Change the third sentence to To deployed.", CorrectionTarget.ORDINAL, 3),
        ("Change the tenth sentence to To deployed.", CorrectionTarget.ORDINAL, 10),
        ("Change the twentieth line to To deployed.", CorrectionTarget.ORDINAL, 20),
        ("Replace the last sentence with To deployed.", CorrectionTarget.LAST, None),
        ("Replace sentence 3 with To deployed.", CorrectionTarget.ORDINAL, 3),
        ("Rewrite the third sentence to To deployed.", CorrectionTarget.ORDINAL, 3),
        ("Make the last sentence To deployed.", CorrectionTarget.LAST, None),
        ("Set the last subtitle to To deployed.", CorrectionTarget.LAST, None),
        ("Change the last sentence from To developed to To deployed.", CorrectionTarget.LAST, None),
        ("please change the last sentence to To deployed.", CorrectionTarget.LAST, None),
        ("Change the last sentence to To deployed", CorrectionTarget.LAST, None),
    ],
)
def test_natural_set_text_target_phrasings(phrase, target, ordinal):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.SET_TEXT, parsed.reason
    assert parsed.target is target
    assert parsed.ordinal == ordinal
    assert parsed.replace


def test_a_quoted_replacement_is_preserved_exactly():
    """A typed command is precise, so quotes must not become part of the text."""
    parsed = parse_command('Change the last sentence to "To deployed."')

    assert parsed.kind is CommandKind.SET_TEXT
    assert parsed.replace == "To deployed."


def test_a_dictated_replacement_without_quotes_still_parses():
    """Speech-to-text drops quotation marks, so quotes cannot be required."""
    parsed = parse_command("Change the last sentence to To deployed.")

    assert parsed.kind is CommandKind.SET_TEXT
    assert parsed.replace == "To deployed."


def test_a_replacement_containing_the_word_to_survives_intact():
    """The worst case for greedy-vs-lazy matching on the connector keyword.

    "To deployed." begins with the connector word "to", so a lazy replacement
    group would capture an empty string and blank the line.
    """
    parsed = parse_command("Change the last sentence to To deployed.")

    assert parsed.replace == "To deployed."


def test_a_replacement_containing_quotes_inside_is_kept():
    parsed = parse_command('Change the last sentence to He said "to be or not to be".')

    assert parsed.replace == 'He said "to be or not to be".'


def test_trailing_filler_is_still_removed_from_a_replacement():
    """Verbatim matching must not reintroduce the noise _strip_filler removes.

    The full stop goes with the ", please." rather than with the sentence: the
    filler pattern owns the punctuation that terminates it, so what is left is
    the requested wording. Without quotes that is the most a speaker could ask
    for, and a typed user who wants the stop can still quote it.
    """
    parsed = parse_command("Change the last sentence to To deployed, please.")

    assert parsed.replace == "To deployed"


def test_a_set_text_command_ending_in_a_question_mark_keeps_it():
    parsed = parse_command("Change the last sentence to Is it deployed?")

    assert parsed.replace == "Is it deployed?"


def test_set_text_reason_names_the_target():
    """The reason line is shown to the user, so it must be human-readable."""
    assert parse_command("Change sentence 3 to To deployed.").describe_target() == "sentence 3"
    assert parse_command("Change the last sentence to X.").describe_target() == "the last sentence"
    assert (
        parse_command("Change the previous sentence to X.").describe_target()
        == "the previous sentence"
    )
    assert parse_command("Change this to X.").describe_target() == "this sentence"


@pytest.mark.parametrize(
    "phrase",
    [
        "Change this to X",
        "Change the last sentence to X",
        "Change the previous sentence to X",
        "Change sentence 3 to X",
        "Change the 3rd line to X",
    ],
)
def test_the_reason_never_leaks_an_internal_enum_name(phrase):
    """The activity log is user-facing.

    An earlier version read "Setting ordinal to ..." because the reason was
    built from `target.value` rather than the human phrasing, which put an
    internal identifier in front of whoever was watching the demo.
    """
    reason = parse_command(phrase).reason

    assert "ordinal" not in reason.lower()
    assert "target" not in reason.lower()
    assert reason.startswith("Setting ")


def test_a_reason_does_not_double_the_replacement_punctuation():
    """The replacement is quoted verbatim, so its full stop is its own."""
    assert parse_command('Change this to "To deployed."').reason == (
        'Setting this sentence to "To deployed."'
    )
    # ...but a replacement with no terminal punctuation still reads as a sentence.
    assert parse_command("Change this to To deployed").reason.endswith('deployed".')


@pytest.mark.parametrize(
    "phrase",
    [
        # A target phrase we do not know. Refusing is the point: guessing would
        # rewrite whichever line happened to be last.
        "Change the second-to-last thing to X",
        "Change the sentence before that to X",
        "Change the sentence after this to X",
        "Change the wrong one to X",
    ],
)
def test_an_unrecognised_target_is_refused_not_guessed(phrase):
    parsed = parse_command(phrase)
    assert parsed.kind is CommandKind.UNSUPPORTED
    assert "which line" in parsed.reason.lower()


def test_a_set_text_command_with_no_replacement_is_refused():
    """A truncated dictation must not blank out a transcript line."""
    assert parse_command("Change the last sentence to").kind is CommandKind.UNSUPPORTED
    assert parse_command("Replace the last sentence with").kind is CommandKind.UNSUPPORTED


def test_an_absurdly_long_replacement_is_refused():
    """A subtitle line is not a paragraph, and neither is a demo command."""
    parsed = parse_command("Change the last sentence to " + "word " * 200)
    assert parsed.kind is CommandKind.UNSUPPORTED


@pytest.mark.parametrize(
    "phrase,kind",
    [
        ("Change fast API to FastAPI", CommandKind.REPLACE),
        ("Replace Alice with Atlas", CommandKind.REPLACE),
        ("Swap Python for AssemblyAI", CommandKind.REPLACE),
        ("Correct the last subtitle", CommandKind.CORRECT_LAST),
        ("Fix the previous subtitle", CommandKind.CORRECT_PREVIOUS),
        ("Remember FastAPI as a technical term", CommandKind.REMEMBER),
    ],
)
def test_the_new_verb_cannot_shadow_the_existing_commands(phrase, kind):
    """SET_TEXT is greedy, so it must not swallow the commands that came first.

    "Change fast API to FastAPI" is the flagship demo command; if the new
    pattern had been checked first without a closed target set it would have
    parsed as a whole-line rewrite of the last sentence and quietly broken it.
    """
    assert parse_command(phrase).kind is kind


def test_needs_model_is_true_only_for_correction_questions():
    """One source of truth for "will this call the gateway".

    The session uses it to decide whether to show a pending state, so if it
    disagreed with the engine the UI would either spin forever or hide real
    latency.
    """
    from scriptora.services.command_service import needs_model

    assert needs_model(parse_command("Correct the last subtitle")) is True
    assert needs_model(parse_command("Fix the previous subtitle")) is True
    # These carry their own answer, so a gateway call could not change them.
    assert needs_model(parse_command("Change fast API to FastAPI")) is False
    assert needs_model(parse_command("Change sentence 3 to To deployed.")) is False
    assert needs_model(parse_command("Remember FastAPI")) is False


# ==================================================== natural "sentence" support
#
# A live rehearsal said "Correct the last sentence." and got silence. The target
# grammar knew the word "sentence" and the correction rules had their own,
# older noun list that did not, so the phrase parsed as UNSUPPORTED and the user
# was told nothing. These pin the two vocabularies to one source.


@pytest.mark.parametrize(
    "phrase",
    [
        "Correct the last sentence.",
        "Correct the last subtitle.",
        "Fix the last sentence.",
        "Fix the last subtitle.",
    ],
)
def test_sentence_and_subtitle_are_interchangeable_for_the_last_line(phrase):
    command = parse_command(phrase)
    assert command.kind is CommandKind.CORRECT_LAST
    assert command.target is CorrectionTarget.LAST


@pytest.mark.parametrize(
    "phrase",
    [
        "Fix the previous sentence.",
        "Fix the previous subtitle.",
        "Correct the previous sentence.",
        "Correct the previous subtitle.",
    ],
)
def test_sentence_and_subtitle_are_interchangeable_for_the_previous_line(phrase):
    command = parse_command(phrase)
    # Which line it names is carried by the *kind*; the resolver branches on
    # that and looks up `previous_original()`.
    assert command.kind is CommandKind.CORRECT_PREVIOUS


@pytest.mark.parametrize(
    "noun",
    ["sentence", "subtitle", "line", "caption"],
)
def test_every_line_noun_resolves_to_the_same_last_target(noun):
    command = parse_command(f"Change the last {noun} to To deployed.")
    assert command.kind is CommandKind.SET_TEXT
    assert command.target is CorrectionTarget.LAST
    assert command.replace == "To deployed."


def test_a_correction_question_named_in_a_sentence_still_extracts_its_term():
    """The synonym must not cost us the answer the user dictated with it."""
    command = parse_command("Correct the last sentence. It's FastAPI.")

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.find == "FastAPI"


# ================================================ transcription punctuation
#
# AssemblyAI punctuated a dictated command as "change the last sentence to, we
# deployed it", and the comma silently invalidated an otherwise perfect command.


@pytest.mark.parametrize(
    "phrase",
    [
        "Change the last sentence to We deployed it.",
        "Change the last sentence to, We deployed it.",
        "Change the last sentence to: We deployed it.",
        "Change the last subtitle to We deployed it.",
        "Change the last sentence with We deployed it.",
        "Change the last sentence with, We deployed it.",
    ],
)
def test_pause_punctuation_after_the_delimiter_is_ignored(phrase):
    command = parse_command(phrase)
    assert command.kind is CommandKind.SET_TEXT
    assert command.target is CorrectionTarget.LAST
    assert command.replace == "We deployed it."


def test_punctuation_inside_the_replacement_is_never_stripped():
    """Only the delimiter's own marks go; the speaker's own stay."""
    command = parse_command('Change the last sentence to: "Hello, world."')

    assert command.kind is CommandKind.SET_TEXT
    assert command.replace == "Hello, world."


def test_an_unquoted_replacement_keeps_its_internal_punctuation():
    command = parse_command("Change the last sentence to, Hello, world.")

    assert command.kind is CommandKind.SET_TEXT
    assert command.replace == "Hello, world."


def test_a_spelled_out_delimiter_with_a_comma_resolves_like_the_typed_form():
    """Spoken and typed must converge on the same target and replacement.

    A second spoken-only grammar would be the place for these two to drift.
    """
    spoken = parse_command("Change the last sentence to, we deployed it to Qwen clusters.")
    typed = parse_command('Change the last sentence to "We deployed it to Qwen clusters."')

    assert spoken.kind is typed.kind is CommandKind.SET_TEXT
    assert spoken.target is typed.target
    assert spoken.ordinal == typed.ordinal
    # Case aside, the dictated and typed forms name the same new line.
    assert spoken.replace.casefold() == typed.replace.casefold()


def test_the_from_to_form_also_tolerates_a_delimiter_comma():
    command = parse_command("Change the last sentence from Quen to, Qwen clusters.")

    assert command.kind is CommandKind.SET_TEXT
    assert command.replace == "Qwen clusters."


def test_a_rewrite_with_no_replacement_is_still_refused():
    """The looser delimiter must not let an empty replacement through."""
    command = parse_command("Change the last sentence to")

    assert command.is_supported is False


def test_the_new_punctuation_tolerance_cannot_shadow_a_find_replace():
    """`Change X to Y` must not be captured as a target-based rewrite."""
    command = parse_command("Change fast API to FastAPI")

    assert command.kind is CommandKind.REPLACE
    assert command.find == "fast API"
    assert command.replace == "FastAPI"


# ==================== correct <term> in <target> ==========================
# The user pointing at one wrong word, rather than asking for a whole line to
# be revisited. Nothing here is specific to any particular mishearing, so the
# arbitrary nonsense word in most of these cases is the point.


@pytest.mark.parametrize(
    "phrase",
    [
        "Correct Symfpony in the last sentence.",
        "Fix Symfpony in the last subtitle.",
        "Correct the word Symfpony in the last sentence.",
        "Fix the term Symfpony in the last subtitle.",
        "Correct the phrase Symfpony in the last line.",
    ],
)
def test_naming_a_word_in_the_last_line_captures_the_term(phrase):
    command = parse_command(phrase)

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.target is CorrectionTarget.LAST
    assert command.find == "Symfpony"


@pytest.mark.parametrize("noun", ["sentence", "subtitle", "line", "caption", "transcript"])
def test_every_line_noun_the_project_already_uses_is_accepted(noun):
    command = parse_command(f"Correct Symfpony in the last {noun}.")

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.find == "Symfpony"


@pytest.mark.parametrize(
    "phrase",
    [
        "Correct Symfpony in the previous sentence.",
        "Fix Symfpony in the previous subtitle.",
        "Correct Symfpony in the prior line.",
        "Fix Symfpony in the preceding caption.",
    ],
)
def test_naming_a_word_in_the_previous_line_keeps_previous_semantics(phrase):
    command = parse_command(phrase)

    assert command.kind is CommandKind.CORRECT_PREVIOUS
    assert command.target is CorrectionTarget.PREVIOUS
    assert command.find == "Symfpony"


@pytest.mark.parametrize(
    "phrase,ordinal",
    [
        ("Correct Symfpony in sentence 3.", 3),
        ("Fix Symfpony in sentence 3.", 3),
        ("Correct Symfpony in the 3rd sentence.", 3),
        ("Fix the word Symfpony in the 22nd line.", 22),
    ],
)
def test_an_ordinal_target_is_resolved_not_defaulted_to_the_last_line(phrase, ordinal):
    command = parse_command(phrase)

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.target is CorrectionTarget.ORDINAL
    assert command.ordinal == ordinal
    assert command.find == "Symfpony"


@pytest.mark.parametrize(
    "phrase",
    [
        "Correct Symfpony in the last sentence",
        "correct symfpony in the last sentence.",
        "CORRECT SYMFPONY IN THE LAST SENTENCE.",
        "Correct Symfpony in the last sentence!",
        "Correct Symfpony in the last sentence, please.",
        "Please correct Symfpony in the last sentence.",
        "Please fix Symfpony in the last sentence, thanks.",
    ],
)
def test_punctuation_case_and_politeness_variants_parse_identically(phrase):
    command = parse_command(phrase)

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.find.casefold() == "symfpony"


def test_a_multi_word_term_is_kept_whole():
    command = parse_command("Correct Kubernetes Cluster in the last sentence.")

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.find == "Kubernetes Cluster"


def test_a_trailing_named_answer_does_not_displace_the_pointed_at_word():
    """The user can still say the answer; the pointer stays the primary term."""
    command = parse_command("Correct Symfpony in the last sentence. It should be Symphony.")

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.find == "Symfpony"


def test_the_named_form_does_not_shadow_the_existing_verified_commands():
    """The new grammar is more specific, so every older form must still win."""
    assert parse_command("Correct the last subtitle.").kind is CommandKind.CORRECT_LAST
    assert parse_command("Correct the last sentence.").kind is CommandKind.CORRECT_LAST
    assert parse_command("Fix the previous subtitle.").kind is CommandKind.CORRECT_PREVIOUS

    rewrite = parse_command("Change the last sentence to we deployed it.")
    assert rewrite.kind is CommandKind.SET_TEXT
    assert rewrite.replace == "we deployed it."


def test_the_named_form_never_invents_a_replacement():
    """Naming the mistake is not naming the answer; the model still decides."""
    command = parse_command("Correct Symfpony in the last sentence.")

    assert command.find == "Symfpony"
    assert command.replace is None


# ------------------------------------------------------------- negative cases


def test_ordinary_prose_mentioning_the_phrase_is_not_a_command():
    """A past-tense report of having corrected something is not an instruction."""
    command = parse_command("I corrected Symfpony yesterday in the last sentence of my report.")

    assert command.kind is CommandKind.UNSUPPORTED
    assert command.is_supported is False


def test_prose_that_does_not_begin_with_the_verb_is_not_a_command():
    command = parse_command("I need to fix the error in the last line of the report.")

    assert command.is_supported is False


def test_a_vague_term_yields_no_find_so_nothing_is_invented():
    """`something` cannot identify a word, so the command carries no pointer."""
    command = parse_command("Correct something in the last sentence.")

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.find is None


def test_the_word_marker_alone_is_not_treated_as_the_term():
    """A word named but forgotten must not pin the correction to the word "word"."""
    command = parse_command("Correct the word in the last sentence.")

    assert command.kind is CommandKind.CORRECT_LAST
    assert command.find is None


def test_an_empty_term_is_not_a_command():
    command = parse_command("Correct in the last sentence.")

    assert command.is_supported is False


def test_an_oversized_term_is_not_a_command():
    """A paragraph is someone explaining, not naming a word."""
    command = parse_command(
        "Correct the entire paragraph that I dictated earlier today in the last sentence."
    )

    assert command.is_supported is False


def test_an_unresolvable_target_is_refused_rather_than_guessed():
    """Defaulting to the last line here would rewrite the wrong subtitle."""
    command = parse_command("Correct Symfpony in the sentence after this.")

    # Refused outright: the target grammar has no "after this", so the command
    # must not quietly become a correction of the newest line.
    assert command.is_supported is False
    assert command.kind is CommandKind.UNSUPPORTED
    assert command.replace is None


def test_a_target_followed_by_a_preposition_is_not_read_as_a_target():
    command = parse_command("Correct Symfpony in the last sentence of my report.")

    assert command.is_supported is False


def test_the_grammar_is_generic_over_the_word_being_corrected():
    """No mishearing is special-cased: the term is simply whatever was named."""
    for word in ("Wobble", "Kubernete", "Quen", "Zorblax", "AT&T"):
        command = parse_command(f"Correct {word} in the last sentence.")

        assert command.kind is CommandKind.CORRECT_LAST
        assert command.find == word
