"""Repair interpretation (JOB 1): understand ordinary correction language.

This is deliberately separate from the correction engine. The user's spec
asks for a hybrid: the LLM only ever *understands* a spoken repair and returns
a validated plan; the deterministic engine still *executes* it. Nothing in this
module touches state. `command_from_intent` maps a plan onto the same
`ParsedCommand` vocabulary the parser already produces, so the session's single
executor (`_run_command`) applies it with every existing safety mechanism.

Two further hard limits:

* `looks_like_repair` is the cheap gate before the LLM, so ordinary dictation
  ("We should replace the battery tomorrow.") never costs a round trip. It
  looks for repair *discourse* markers, not verbs: "change/replace/fix X to Y"
  is already handled deterministically by the Stage-A gate, and in prose those
  verbs mean nothing.
* The interpreter never invents a target. It is told to put in `find` only
  words that appear in the transcript it was given, and the session re-checks
  that before running anything.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from ..config import LLM_GATEWAY_URL, Settings
from ..models.correction import extract_json_object
from ..models.intent import IntentKind, IntentResult, IntentTarget
from .command_service import CommandKind, CorrectionTarget, ParsedCommand
from .context_service import ContextService
from .subtitle_service import SubtitleService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage-B gate: does this turn even look like a spoken repair?
# ---------------------------------------------------------------------------

# Repairs are terse. Anything longer than this is explanation or dictation, and
# short-circuiting here avoids an LLM round trip on ordinary speech.
_MAX_INTENT_WORDS = 16

# Repair *discourse* markers. Deliberately not the command verbs: those are the
# Stage-A gate's domain, and "We should change the configuration before
# shipping" is prose that must not pay for a gateway call. The marker is only a
# candidate - the interpreter still has to confirm, against the transcript, or
# the turn stays in the transcript.
_REPAIR_CUE = re.compile(
    r"\b(?:"
    r"that(?:'s|\s+is)\s+(?:supposed|meant|not)\s+to\s+be"
    r"|it(?:'s|\s+is)\s+(?:supposed|meant)\s+to\s+be"
    r"|was\s+(?:supposed|meant)\s+to\s+be"
    r"|should(?:n'?t)?\s+(?:be|say|have\s+been)"
    r"|what(?:\s+i)?\s+meant\s+was"
    r"|i\s+meant"
    r"|i\s+didn'?t\s+say"
    r"|no\s*,\s*(?:i\s+(?:said|meant)|it'?s|that'?s|actually)"
    r"|i\s+actually\s+said"
    r"|actually\b"
    r"|instead\s+of"
    r"|use\b(?:\s+\w+){0,5}\s+instead"
    r"|the\s+(?:last|previous|second)\s+one\b"
    r"|it'?s\s+(?:supposed|meant)\s+to\s+be"
    r"|still\s+should\s+be"
    r")\b",
    re.IGNORECASE,
)


def looks_like_repair(text: str) -> bool:
    """Whether `text` is worth asking the interpreter about.

    True only for a short utterance carrying a repair discourse marker. The
    interpreter still has to tie it to the transcript; this just keeps the LLM
    out of ordinary speech.
    """
    lowered = text.strip().lower()
    return bool(_REPAIR_CUE.search(lowered)) and len(lowered.split()) <= _MAX_INTENT_WORDS


# ---------------------------------------------------------------------------
# The interpreter
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_INTENT = (
    "You are the repair interpreter for Scriptora, a subtitle agent. The user "
    "speaks, the agent shows live subtitles, and the user sometimes corrects a "
    "word out loud. You decide whether the latest spoken turn is such a repair "
    "and what it asks for. You never edit state yourself - you return a plan.\n\n"
    "Output ONLY one JSON object, no prose, no markdown fences:\n"
    '{"intent": "none" | "correct_subtitle" | "replace_text" | "remember_term",\n'
    ' "target": "last" | "current" | "previous" | "ordinal" | "find_text" | null,\n'
    ' "ordinal": integer | null,\n'
    ' "find": string | null,\n'
    ' "replacement": string | null,\n'
    ' "confidence": number,\n'
    ' "reason": string}\n\n'
    "Decide between the intents (X = a wrong word that appears in a subtitle "
    "you were given, Y = the word the user says it should be):\n"
    '- "replace_text": the user is pointing at a wrong word or phrase in an '
    'existing subtitle and says what it should be. Example phrasings: "No, I '
    'said Y.", "That\'s supposed to be Y.", "I meant Y.", "Use Y instead.", '
    '"The last one should say Y.", "Change X to Y.", '
    '"Correct X - Y."\n'
    '  * find: X, exactly as it appears in a subtitle you were given. "No, I '
    'said Y." against a subtitle containing a wrong word X means '
    "find=X, replacement=Y. If the wrong word cannot be identified from the "
    "current_subtitle or previous_subtitles, return intent none instead of "
    "guessing.\n"
    '  * target: "find_text" when the repair names the wrong word (the common '
    'case); otherwise "last"/"current"/"previous"/"ordinal" only when '
    "the user explicitly names a line.\n"
    "  * replacement: Y, the word or phrase the user says it should be.\n"
    '- "correct_subtitle": the user asks to fix a subtitle without stating the '
    'answer, e.g. "Correct the last subtitle.", or names only the wrong word '
    "and a line. "
    "Target: last/current/previous/ordinal.\n"
    '- "remember_term": the user asks to remember or register a term; put the '
    "term in find.\n"
    '- "none": ordinary speech, or anything that does not clearly refer to the '
    "subtitle transcript you were given. Reject these examples with intent "
    'none: "I meant to call John yesterday.", "We should replace the battery '
    'tomorrow.", "I actually said the meeting starts at five.", "Please fix the '
    'slides before Friday." An utterance is a repair only if it refers to the '
    "content of the subtitles, usually by repeating a word that appears in "
    "one.\n\n"
    "Rules:\n"
    "- The replacement must be text the user actually spoke. Never invent a "
    "find: it must appear in the subtitles you were given, or be a clear "
    "near-spelling of a word that does.\n"
    "- Confidence is how sure you are, 0 to 1. When unsure whether this is a "
    'repair at all, set intent to "none". For a clear repair use a confidence '
    "of at least 0.7; never return a confident guess.\n"
    "- Do not output subtitle ids or anything outside the schema.\n"
    "- find and replacement are short phrases, copied verbatim from the turn "
    "or a subtitle; do not rephrase or summarise them."
)


class IntentInterpreter:
    """LLM-backed interpreter that turns a spoken repair into a validated plan."""

    name = "intent"

    def __init__(self, settings: Settings, *, timeout: float = 25.0) -> None:
        self._settings = settings
        self._timeout = timeout

    async def interpret(
        self,
        utterance: str,
        subtitles: SubtitleService,
        context: ContextService,
    ) -> IntentResult | None:
        """Classify a turn, returning None on any problem (never raises).

        None collapses to "leave the turn in the transcript" in the session, so
        an unreachable gateway, a non-200, unparseable output or a schema
        violation all cost the user nothing but the turn staying as content.
        """
        body = {
            "model": self._settings.llm_model,
            "max_tokens": 250,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_INTENT},
                {
                    "role": "user",
                    "content": json.dumps(self._payload(utterance, subtitles, context)),
                },
            ],
        }
        try:
            raw = await self._complete(body)
            return IntentResult.model_validate(extract_json_object(raw))
        except Exception as exc:
            logger.info(
                "Repair interpretation unavailable (%s); leaving turn as content",
                type(exc).__name__,
            )
            return None

    @staticmethod
    def _payload(utterance: str, subtitles: SubtitleService, context: ContextService) -> dict:
        originals = subtitles.originals()
        # The turn being interpreted is already the trailing subtitle by the
        # time Stage B runs - the session finalizes it before asking. It must
        # not be handed back as context, or the interpreter would see the
        # utterance repeating the "current subtitle" and classify nothing as a
        # repair. The subtitles under repair are the ones that came before it.
        if originals and originals[-1].text.strip() == utterance.strip():
            originals = originals[:-1]
        current = originals[-1].text if originals else None
        previous = [item.text for item in reversed(originals[:-1])][:2]
        return {
            "utterance": utterance,
            "current_subtitle": current,
            "previous_subtitles": previous,
            "project_vocabulary": context.terms,
        }

    async def _complete(self, body: dict) -> str:
        headers = {"authorization": self._settings.require_api_key()}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                LLM_GATEWAY_URL, headers=headers, json=body, timeout=self._timeout
            )
        if response.status_code != 200:
            raise RuntimeError(f"LLM gateway returned HTTP {response.status_code}")
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM gateway response had no message content") from exc


# ---------------------------------------------------------------------------
# Plan -> command
# ---------------------------------------------------------------------------

# A vetted `find` for the correction paths: short, and must carry letters the
# transcript lookup can actually match. Mirrors the parser's own vetting so the
# interpreter cannot slip a garbage find into a model correction.
_MAX_CORRECTION_FIND = 40
_VAGUE_FIND = frozenset(
    {"it", "that", "this", "them", "those", "these", "they", "that word", "the word"}
)


def _correction_find(find: str | None) -> str | None:
    if not find:
        return None
    cleaned = find.strip().strip(" .?!,")
    if not cleaned or len(cleaned) > _MAX_CORRECTION_FIND:
        return None
    if cleaned.casefold() in _VAGUE_FIND:
        return None
    if not any(char.isalnum() for char in cleaned):
        return None
    return cleaned


def command_from_intent(result: IntentResult, raw: str) -> ParsedCommand | None:
    """Map a validated repair plan onto the parser's command vocabulary.

    Returns None when the plan cannot become a command (an empty plan, or a
    `replace_text` with no replacement) - the session then leaves the turn as
    content. An ambiguous but confident plan becomes UNSUPPORTED so the user is
    told specifically what is missing instead of hearing nothing.
    """
    if not result.is_repair:
        return None

    if result.intent is IntentKind.REPLACE_TEXT:
        replacement = (result.replacement or "").strip()
        if not replacement:
            return None
        find = (result.find or "").strip()
        if not find:
            return ParsedCommand(
                CommandKind.UNSUPPORTED,
                raw,
                reason=(f'I can change it to "{replacement}", but which word should I change?'),
            )
        return ParsedCommand(
            CommandKind.REPLACE,
            raw,
            find=find,
            replace=replacement,
            reason=result.reason or f'Repairing "{raw}": replace "{find}" with "{replacement}".',
        )

    if result.intent is IntentKind.CORRECT_SUBTITLE:
        if result.target is IntentTarget.ORDINAL and result.ordinal is not None:
            return ParsedCommand(
                CommandKind.CORRECT_LAST,
                raw,
                find=_correction_find(result.find),
                target=CorrectionTarget.ORDINAL,
                ordinal=result.ordinal,
                reason=result.reason or f"Fixing sentence {result.ordinal}.",
            )
        if result.target is IntentTarget.PREVIOUS:
            return ParsedCommand(
                CommandKind.CORRECT_PREVIOUS,
                raw,
                find=_correction_find(result.find),
                target=CorrectionTarget.PREVIOUS,
                reason=result.reason or "Fixing the previous subtitle.",
            )
        return ParsedCommand(
            CommandKind.CORRECT_LAST,
            raw,
            find=_correction_find(result.find),
            target=CorrectionTarget.LAST,
            reason=result.reason or "Fixing the last subtitle.",
        )

    if result.intent is IntentKind.REMEMBER_TERM:
        term = (result.find or result.replacement or "").strip()
        if not term:
            return ParsedCommand(
                CommandKind.UNSUPPORTED,
                raw,
                reason="Which term should I remember?",
            )
        return ParsedCommand(
            CommandKind.REMEMBER,
            raw,
            find=term,
            reason=result.reason or f'Remembering "{term}".',
        )

    return None
