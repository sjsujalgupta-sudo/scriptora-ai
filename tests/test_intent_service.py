"""Repair interpreter: planning, validation and the Stage-B cue gate."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from scriptora.config import Settings
from scriptora.models.intent import IntentKind, IntentResult, IntentTarget
from scriptora.services.command_service import CommandKind, CorrectionTarget
from scriptora.services.context_service import ContextService
from scriptora.services.intent_service import (
    SYSTEM_PROMPT_INTENT,
    IntentInterpreter,
    command_from_intent,
    looks_like_repair,
)
from scriptora.services.subtitle_service import SubtitleService

FAKE_KEY = "test-key-not-a-credential"


def build(intent: dict | None = None):
    settings = Settings(assemblyai_api_key=FAKE_KEY)
    subtitles = SubtitleService()
    context = ContextService()
    return settings, subtitles, context, intent


def _intent_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": __import__("json").dumps(payload)},
                    "finish_reason": "stop",
                }
            ]
        },
    )


# ================================================================ Stage-B cue


@pytest.mark.parametrize(
    "true_text",
    [
        "That's supposed to be Symphony.",
        "I meant Symphony.",
        "No, I said Symphony.",
        "Actually, it should be Symphony.",
        "The last one should say Symphony.",
        "Use Symphony instead.",
        "I actually said Symphony.",
        "That was meant to be Symphony.",
        "No, it's Symphony.",
        "What I meant was Symphony.",
    ],
)
def test_looks_like_repair(true_text: str):
    assert looks_like_repair(true_text), true_text


@pytest.mark.parametrize(
    "false_text",
    [
        "I need to remember to lock the door before I leave.",
        "We should change the configuration before shipping.",
        "We should replace the battery tomorrow.",
        "Please fix the slides before Friday.",
        "Please replace the battery before tomorrow morning so we are not late.",
        "We should probably be careful here.",
    ],
)
def test_looks_like_repair_rejects_ordinary_speech(false_text: str):
    """Prose that carries none of the repair discourse markers never cues."""
    assert not looks_like_repair(false_text), false_text


@pytest.mark.parametrize(
    "cued_but_ordinary",
    [
        "I meant to call John yesterday.",
        "I actually said the meeting starts at five.",
        "Actually, the quarterly review is scheduled for the afternoon.",
    ],
)
def test_looks_like_repair_cues_these_but_the_interpreter_must_reject_them(cued_but_ordinary: str):
    """These carry a marker, so they reach the interpreter - whose transcript
    check is what classifies them as ordinary speech, not the cue gate."""
    assert looks_like_repair(cued_but_ordinary)


def test_looks_like_repair_ignores_long_dictation():
    """A long sentence mentioning a repair word must not trigger the gateway."""
    text = (
        "I just wanted to say that the meeting should be moved, actually, and "
        "that we should be very careful about the whole rollout plan next week."
    )
    assert not looks_like_repair(text)


# ================================================================ validation


def test_intent_forces_enum_values():
    with pytest.raises(ValidationError):
        IntentResult.model_validate({"intent": "flip_the_table", "confidence": 0.9})


def test_intent_coerces_and_bounds_fields():
    result = IntentResult.model_validate(
        {
            "intent": "  Replace_Text ",
            "target": "FIND_TEXT",
            "find": "  Senzani ",
            "replacement": "  Symphony ",
            "confidence": "0.95",
            "reason": "user clarifies",
        }
    )
    assert result.intent is IntentKind.REPLACE_TEXT
    assert result.target is IntentTarget.FIND_TEXT
    assert result.find == "Senzani"
    assert result.replacement == "Symphony"
    assert result.confidence == 0.95
    assert result.is_repair


def test_intent_clamps_confidence():
    high = IntentResult.model_validate({"intent": "none", "confidence": 1.7})
    low = IntentResult.model_validate({"intent": "none", "confidence": -0.4})
    garbage = IntentResult.model_validate({"intent": "none", "confidence": "maybe"})
    missing = IntentResult.model_validate({"intent": "none"})
    assert (high.confidence, low.confidence, garbage.confidence, missing.confidence) == (
        1.0,
        0.0,
        0.0,
        0.0,
    )


def test_intent_drops_overlong_find_and_replacement():
    result = IntentResult.model_validate(
        {
            "intent": "replace_text",
            "find": "x" * 300,
            "replacement": "y" * 900,
            "reason": "z" * 900,
        }
    )
    assert result.find is None
    assert result.replacement is None
    assert len(result.reason) == 300
    assert result.is_repair


def test_intent_rejects_nonsensical_ordinal():
    result = IntentResult.model_validate(
        {"intent": "correct_subtitle", "target": "ordinal", "ordinal": 0}
    )
    assert result.ordinal is None


# ================================================================ mapping


def test_replace_intent_maps_to_a_replace_command():
    result = IntentResult.model_validate(
        {
            "intent": "replace_text",
            "target": "find_text",
            "find": "Senzani",
            "replacement": "Symphony",
            "confidence": 0.9,
            "reason": "user clarifies",
        }
    )
    command = command_from_intent(result, "No, I said Symphony.")
    assert command is not None
    assert command.kind is CommandKind.REPLACE
    assert command.find == "Senzani"
    assert command.replace == "Symphony"


def test_replace_intent_without_find_asks_which_word():
    result = IntentResult.model_validate(
        {
            "intent": "replace_text",
            "target": "last",
            "find": None,
            "replacement": "Symphony",
            "confidence": 0.9,
        }
    )
    command = command_from_intent(result, "The last one should say Symphony.")
    assert command is not None
    assert command.kind is CommandKind.UNSUPPORTED
    assert "which word" in command.reason


def test_replace_intent_without_replacement_is_dropped():
    result = IntentResult.model_validate(
        {"intent": "replace_text", "target": "last", "find": "Senzani", "replacement": None}
    )
    assert command_from_intent(result, "No, I said ...") is None


def test_correct_subtitle_intent_targets_previous():
    result = IntentResult.model_validate(
        {"intent": "correct_subtitle", "target": "previous", "find": None, "confidence": 0.9}
    )
    command = command_from_intent(result, "Fix the previous line.")
    assert command is not None
    assert command.kind is CommandKind.CORRECT_PREVIOUS
    assert command.target is CorrectionTarget.PREVIOUS


def test_correct_subtitle_intent_targets_ordinal():
    result = IntentResult.model_validate(
        {
            "intent": "correct_subtitle",
            "target": "ordinal",
            "ordinal": 3,
            "find": "Senjani",
            "confidence": 0.9,
        }
    )
    command = command_from_intent(result, "Correct the word in sentence 3.")
    assert command is not None
    assert command.kind is CommandKind.CORRECT_LAST
    assert command.target is CorrectionTarget.ORDINAL
    assert command.ordinal == 3
    assert command.find == "Senjani"


def test_correct_subtitle_rejects_vague_find():
    result = IntentResult.model_validate(
        {"intent": "correct_subtitle", "target": "last", "find": "the word", "confidence": 0.9}
    )
    command = command_from_intent(result, "Correct the word in the last sentence.")
    assert command is not None
    assert command.find is None


def test_remember_intent_maps_to_a_remember_command():
    result = IntentResult.model_validate(
        {"intent": "remember_term", "target": None, "find": "Qwen", "confidence": 0.9}
    )
    command = command_from_intent(result, "Remember Qwen.")
    assert command is not None
    assert command.kind is CommandKind.REMEMBER
    assert command.find == "Qwen"


def test_no_intent_maps_to_no_command():
    result = IntentResult.model_validate({"intent": "none", "confidence": 0.8})
    assert command_from_intent(result, "ordinary words") is None


# ================================================================ interpreter


async def test_interpret_parses_a_valid_plan():
    settings, subtitles, context, _ = build()
    sub = subtitles.add("Can you type Senzani?")
    sub.finalize()

    with patch(
        "httpx.AsyncClient.post",
        return_value=_intent_response(
            {
                "intent": "replace_text",
                "target": "find_text",
                "find": "Senzani",
                "replacement": "Symphony",
                "confidence": 0.9,
                "reason": "user clarifies",
            }
        ),
    ):
        result = await IntentInterpreter(settings).interpret(
            "No, I said Symphony.", subtitles, context
        )

    assert result is not None
    assert result.intent is IntentKind.REPLACE_TEXT
    assert result.find == "Senzani"


async def test_interpret_returns_none_on_unreachable_gateway():
    settings, subtitles, context, _ = build()
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("no route")):
        result = await IntentInterpreter(settings).interpret(
            "No, I said Symphony.", subtitles, context
        )
    assert result is None


async def test_interpret_returns_none_on_non_200():
    settings, subtitles, context, _ = build()
    with patch("httpx.AsyncClient.post", return_value=httpx.Response(500, text="boom")):
        result = await IntentInterpreter(settings).interpret(
            "No, I said Symphony.", subtitles, context
        )
    assert result is None


async def test_interpret_returns_none_on_unparseable_output():
    settings, subtitles, context, _ = build()
    with patch(
        "httpx.AsyncClient.post",
        return_value=_intent_response("Sure, happy to help, but I have no idea."),
    ):
        result = await IntentInterpreter(settings).interpret(
            "No, I said Symphony.", subtitles, context
        )
    assert result is None


async def test_interpret_returns_none_on_schema_violation():
    settings, subtitles, context, _ = build()
    with patch(
        "httpx.AsyncClient.post",
        return_value=_intent_response({"intent": "rewrite_everything", "confidence": 0.99}),
    ):
        result = await IntentInterpreter(settings).interpret(
            "No, I said Symphony.", subtitles, context
        )
    assert result is None


async def test_interpret_sends_transcript_and_vocabulary():
    settings, subtitles, context, _ = build()
    subtitles.add("The agenda is a draft.").finalize()
    subtitles.add("Can you type Senzani?").finalize()
    subtitles.add("Moving on.").finalize()
    context.add_term("Symphony")

    captured: dict = {}

    class RecordingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            captured.update(kwargs["json"])
            return _intent_response({"intent": "none", "confidence": 0.1})

    with patch("httpx.AsyncClient", RecordingClient):
        await IntentInterpreter(settings).interpret("Moving on.", subtitles, context)

    assert captured["temperature"] == 0
    assert captured["messages"][0]["role"] == "system"
    payload = json.loads(captured["messages"][1]["content"])
    # The turn being interpreted ("Moving on.") is already the trailing subtitle
    # by the time Stage B runs; it must not be served back as context, or the
    # interpreter would see the utterance repeat the "current subtitle" and
    # classify nothing as a repair.
    assert payload["utterance"] == "Moving on."
    assert payload["current_subtitle"] == "Can you type Senzani?"
    assert payload["previous_subtitles"] == ["The agenda is a draft."]
    assert payload["project_vocabulary"] == [
        "AssemblyAI",
        "Python",
        "Atlas",
        "PostgreSQL",
        "Symphony",
    ]


def test_the_prompt_does_not_hard_code_demo_terms():
    for token in ("Senzani", "Symphony", "QEMU", "Qwen", "Symfpony", "Quen"):
        assert token not in SYSTEM_PROMPT_INTENT
