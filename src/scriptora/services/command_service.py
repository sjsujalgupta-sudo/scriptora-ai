"""Spoken-command parsing.

Deliberately a small, closed set of intents. The brief is explicit that
reliable commands beat broad ones, so this matches a handful of phrasings
rather than pretending to understand arbitrary natural language. Anything
unrecognised returns `UNSUPPORTED` and is reported honestly in the UI instead
of being guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..models.context import normalize_term


class CommandKind(str, Enum):
    CORRECT_LAST = "correct_last"
    CORRECT_PREVIOUS = "correct_previous"
    REPLACE = "replace"
    REMEMBER = "remember"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ParsedCommand:
    kind: CommandKind
    raw: str
    find: str | None = None
    replace: str | None = None
    reason: str = ""

    @property
    def is_supported(self) -> bool:
        return self.kind is not CommandKind.UNSUPPORTED


# Trailing politeness/filler that speech-to-text appends and that carries no
# intent. Stripped before matching so "Correct the last subtitle, please."
# parses identically to "Correct the last subtitle."
_FILLER = re.compile(r"[,\s]*(?:please|now|okay|ok|thanks|thank you)\s*[.!?]*\s*$", re.IGNORECASE)

_LAST = re.compile(
    r"^(?:correct|fix)(?:\s+(?:the|that|this))?\s+"
    r"(?:last|current|latest|newest|most\s+recent)\s+"
    r"(?:subtitle|caption|line|transcript|one|text)\b",
    re.IGNORECASE,
)

_PREVIOUS = re.compile(
    r"^(?:correct|fix|redo|revisit)\s+(?:the\s+)?"
    r"(?:previous|prior|preceding|above|last\s+one\s+but\s+one)\s+"
    r"(?:subtitle|caption|line|transcript|one|text)\b",
    re.IGNORECASE,
)

# The user often names the right answer straight after the instruction:
# "Correct the last subtitle. It's FastAPI." / "...it should be PostgreSQL".
# Without this the term was thrown away and the correction had nothing to work
# from, which made the primary demo unable to fix itself.
_NAMED_TERM = re.compile(
    r"(?:it'?s|it\s+is|it\s+should\s+be|should\s+be|should\s+say|"
    r"make\s+it|that\s+should\s+be|meant)\s+"
    r"(?P<term>[^,.;!?]+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# "change X to Y" / "replace X with Y" / "swap X for Y"
_REPLACE_WITH = re.compile(
    r"^(?:please\s+)?(?:change|replace|swap|substitute)\s+"
    r"(?P<find>.+?)\s+"
    r"(?:to|with|for|into)\s+"
    r"(?P<replace>.+)$",
    re.IGNORECASE,
)

_REMEMBER = re.compile(
    r"^(?:please\s+)?(?:remember|add|include)\s+"
    r"(?P<term>.+?)(?:\s+as\s+.*)?$",
    re.IGNORECASE,
)

# Phrases that look like commands but are ambiguous or out of scope. Caught
# before the generic handlers so we can explain rather than misfire.
_UNSUPPORTED_HINTS = re.compile(
    r"\b(?:delete|remove|undo|translate|export|download|"
    r"who\s+(?:is|said)|split|merge|translate)\b",
    re.IGNORECASE,
)


# A "find" of a bare pronoun or filler can never identify anything to replace,
# so such a command is reported as unsupported rather than silently no-op'ing.
_VAGUE_FIND = frozenset(
    {
        "it",
        "that",
        "this",
        "them",
        "those",
        "these",
        "they",
        "he",
        "she",
        "the word",
        "the name",
        "something",
        "anything",
        "stuff",
        "that word",
    }
)


def _strip_filler(text: str) -> str:
    previous = None
    current = text.strip()
    # Loop because "..., okay, please" leaves more than one pass.
    while previous != current:
        previous = current
        current = _FILLER.sub("", current).strip()
    return current


# Longest phrase we will accept as "the term the user dictated". A dictated
# technical term is a word or two, not a sentence; anything longer is someone
# explaining the correction, not naming the answer.
_MAX_NAMED_TERM = 40
_MAX_NAMED_TERM_WORDS = 3


def _named_term(text: str) -> str | None:
    """Pull the answer out of "...correct the last subtitle, it's FastAPI".

    Returns None when the user named nothing, so a bare "Correct the last
    subtitle." still falls back to vocabulary and the LLM.
    """
    match = _NAMED_TERM.search(text)
    if not match:
        return None

    term = normalize_term(match.group("term")).strip(" .?!,")
    if not term or len(term) > _MAX_NAMED_TERM:
        return None
    if term.casefold() in _VAGUE_FIND:
        return None
    # "it should be the name of the framework" is someone explaining, not
    # naming an answer.
    if len(term.split()) > _MAX_NAMED_TERM_WORDS:
        return None
    # Must contain something a lookup pattern can actually match.
    if not any(char.isalnum() for char in term):
        return None
    return term


def parse_command(command: str) -> ParsedCommand:
    """Parse a spoken command into a `ParsedCommand`.

    Never raises - unparseable input is reported as `UNSUPPORTED`.
    """
    raw = (command or "").strip()
    if not raw:
        return ParsedCommand(CommandKind.UNSUPPORTED, raw, reason="Empty command.")

    text = _strip_filler(normalize_term(raw)).rstrip(".?!, ")

    if _UNSUPPORTED_HINTS.search(text):
        return ParsedCommand(
            CommandKind.UNSUPPORTED,
            raw,
            reason="That operation is not supported by Scriptora yet.",
        )

    if _PREVIOUS.search(text):
        return ParsedCommand(
            CommandKind.CORRECT_PREVIOUS,
            raw,
            find=_named_term(text),
            reason="Fixing the previous subtitle.",
        )

    if _LAST.search(text):
        return ParsedCommand(
            CommandKind.CORRECT_LAST,
            raw,
            find=_named_term(text),
            reason="Fixing the last subtitle.",
        )

    match = _REPLACE_WITH.match(text)
    if match:
        find = normalize_term(match.group("find")).strip(" .?!,")
        replace = normalize_term(match.group("replace")).strip(" .?!,")
        if find and replace and find.casefold() not in _VAGUE_FIND:
            return ParsedCommand(
                CommandKind.REPLACE,
                raw,
                find=find,
                replace=replace,
                reason=f'Replacing "{find}" with "{replace}".',
            )
        return ParsedCommand(
            CommandKind.UNSUPPORTED,
            raw,
            reason='Tell me exactly what to replace with what, e.g. "Change fast API to FastAPI".',
        )

    match = _REMEMBER.match(text)
    if match:
        term = normalize_term(match.group("term")).strip(" .?!,")
        term = re.sub(
            r"\s+as\s+(?:a\s+|an\s+|the\s+)?\w+.*$", "", term, flags=re.IGNORECASE
        ).strip()
        if term:
            return ParsedCommand(
                CommandKind.REMEMBER,
                raw,
                find=term,
                reason=f'Remembering "{term}".',
            )

    return ParsedCommand(
        CommandKind.UNSUPPORTED,
        raw,
        reason=(
            'Not a supported command. Try "Correct the last subtitle", '
            '"Fix the previous subtitle", "Change X to Y", or '
            '"Remember X as a technical term".'
        ),
    )
