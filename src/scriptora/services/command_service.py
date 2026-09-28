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
    SET_TEXT = "set_text"
    REMEMBER = "remember"
    UNSUPPORTED = "unsupported"


class CorrectionTarget(str, Enum):
    """Which subtitle a correction refers to.

    Kept separate from `CommandKind` because two different intents share the
    same target vocabulary: "correct the last subtitle" asks the model what the
    line should say, while "change the last sentence to X" states the answer
    outright. Only the first needs the LLM.
    """

    THIS = "this"
    LAST = "last"
    PREVIOUS = "previous"
    ORDINAL = "ordinal"


@dataclass(frozen=True)
class ParsedCommand:
    kind: CommandKind
    raw: str
    find: str | None = None
    replace: str | None = None
    reason: str = ""
    # Defaults to LAST so commands that predate target references keep working.
    target: CorrectionTarget = CorrectionTarget.LAST
    # 1-based position, only meaningful when `target` is ORDINAL.
    ordinal: int | None = None

    @property
    def is_supported(self) -> bool:
        return self.kind is not CommandKind.UNSUPPORTED

    def describe_target(self) -> str:
        """Human phrasing of the target, for messages and the activity log."""
        if self.target is CorrectionTarget.ORDINAL and self.ordinal:
            return f"sentence {self.ordinal}"
        return {
            CorrectionTarget.THIS: "this sentence",
            CorrectionTarget.LAST: "the last sentence",
            CorrectionTarget.PREVIOUS: "the previous sentence",
            CorrectionTarget.ORDINAL: "that sentence",
        }[self.target]


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

# The "to the vocabulary/context/project" tail is part of the phrasing, not the
# term. Without stripping it here, "Add Kubernete to the vocabulary" would carry
# a bogus term through to `command.find` and on into AssemblyAI keyterms.
_REMEMBER = re.compile(
    r"^(?:please\s+)?(?:remember|add|include)\s+"
    r"(?P<term>.+?)"
    r"(?:\s+as\s+.*|\s+to\s+the\s+(?:vocabulary|context|project)\b.*)?$",
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


# A dictated technical term is a word or two, not a sentence; anything longer is
# someone explaining the correction, not naming the answer.
_MAX_NAMED_TERM = 40
_MAX_NAMED_TERM_WORDS = 3

# The nouns a person uses for a transcript line. All of them mean the same thing
# to the user, so they all resolve to the same target.
_LINE_NOUN = r"(?:sentence|subtitle|line|caption|transcript)"

# "third", "3rd", "3" all name position 3. Spoken commands rarely use digits,
# but a typed one usually does, and both must land on the same line.
_WORD_ORDINALS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    "eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|"
    "seventeenth|eighteenth|nineteenth|twentieth"
)
_WORD_ORDINAL_VALUES = {
    word: index for index, word in enumerate(_WORD_ORDINALS.split("|"), start=1)
}

# The target half of "change <target> to <text>". Deliberately a closed set: a
# looser pattern would let arbitrary prose mutate the transcript, and the
# requirement is that a bad target is refused rather than guessed at.
_TARGET = (
    # "this" / "that" / "this sentence"
    rf"(?:this|that)(?:\s+{_LINE_NOUN})?"
    # "the last sentence", "the last line"
    rf"|the\s+last\s+(?:{_LINE_NOUN}|one)"
    # "the previous sentence", "the last but one", "the second to last" - all
    # mean second-from-the-end, so they share one target.
    rf"|the\s+(?:previous|prior|preceding|earlier|second\s+to\s+last)\s+(?:{_LINE_NOUN}|one)"
    rf"|the\s+last\s+but\s+one"
    # "sentence 3", "subtitle 3"
    rf"|(?:{_LINE_NOUN})\s+(?P<digit>\d{{1,3}})"
    # "the 3rd sentence", "the 22nd line"
    rf"|the\s+(?P<suffix>\d{{1,3}})(?:st|nd|rd|th)\s+(?:{_LINE_NOUN}|one)"
    # "the third sentence", "the second line"
    rf"|the\s+(?P<word>{_WORD_ORDINALS})\s+(?:{_LINE_NOUN}|one)"
)

# "change the last sentence to X", "replace sentence 3 with X", "set this to X".
#
# The trailing group is the whole replacement and is captured greedily, so a
# sentence containing "to" ("to deployed") survives intact. The verb deliberately
# excludes "correct"/"fix": those ask the model *what* the line should be and are
# handled by the vocabulary-driven rules above.
_SET_TEXT = re.compile(
    r"^(?:please\s+)?(?:change|replace|rewrite|make|set|put)\s+"
    rf"(?P<target>{_TARGET})\s+"
    r"(?:to|with|into|as|for)\s+"
    r"(?P<replacement>.+)$",
    re.IGNORECASE,
)

# "change the last sentence from X to Y" - the two-step form. Only the
# replacement is used; the "from" side is advisory context, not an instruction.
_SET_TEXT_FROM = re.compile(
    r"^(?:please\s+)?(?:change|replace|rewrite|make|set|put)\s+"
    rf"(?P<target>{_TARGET})\s+"
    r"from\s+.+?\s+to\s+"
    r"(?P<replacement>.+)$",
    re.IGNORECASE,
)

# A replacement the user quoted is authoritative: keep it byte for byte.
_QUOTED = re.compile(r'^\s*[""\u201c\u2018\'](?P<inner>.+?)[""\u201d\u2019\']\s*$')

# Longest replacement accepted as "the new line". A dictated sentence is short;
# anything longer is a paragraph, which a subtitle line is not.
_MAX_REPLACEMENT = 240

# Words that mean "the user is pointing at a transcript line". Used only to
# decide whether an unrecognised command deserves a targeting error rather than
# the generic "not a command" one - never to resolve a target.
_REFERENCES_A_LINE = re.compile(
    rf"\b(?:{_LINE_NOUN}s?|one|this|that|previous|prior|preceding|earlier|latest|last|next)\b",
    re.IGNORECASE,
)


def _clean_replacement(text: str) -> str:
    """Strip wrapping quotes but never touch the text inside them.

    Speech-to-text drops quotation marks, so an unquoted replacement has to work
    just as well; a quoted one is preserved exactly, punctuation included.
    """
    candidate = text.strip()
    quoted = _QUOTED.match(candidate)
    if quoted:
        candidate = quoted.group("inner").strip()
    # Only a trailing sentence-final mark is added by dictation noise; interior
    # punctuation is the speaker's and is left alone.
    return candidate.strip()


def _parse_target(text: str) -> tuple[CorrectionTarget, int | None] | None:
    """Map a target phrase to (target, ordinal), or None if unrecognised."""
    text = text.strip().lower()
    if text in ("this", "that") or re.fullmatch(rf"(?:this|that)\s+{_LINE_NOUN}", text):
        return CorrectionTarget.THIS, None
    if re.fullmatch(rf"the\s+last\s+(?:{_LINE_NOUN}|one)", text):
        return CorrectionTarget.LAST, None
    if re.fullmatch(
        rf"the\s+(?:previous|prior|preceding|earlier|second\s+to\s+last)"
        rf"\s+(?:{_LINE_NOUN}|one)",
        text,
    ) or re.fullmatch(r"the\s+last\s+but\s+one", text):
        return CorrectionTarget.PREVIOUS, None

    digits = re.fullmatch(rf"(?:{_LINE_NOUN})\s+(\d{{1,3}})", text)
    if digits:
        return CorrectionTarget.ORDINAL, int(digits.group(1))

    suffix = re.fullmatch(rf"the\s+(\d{{1,3}})(?:st|nd|rd|th)\s+(?:{_LINE_NOUN}|one)", text)
    if suffix:
        return CorrectionTarget.ORDINAL, int(suffix.group(1))

    word = re.fullmatch(rf"the\s+({_WORD_ORDINALS})\s+(?:{_LINE_NOUN}|one)", text)
    if word:
        return CorrectionTarget.ORDINAL, _WORD_ORDINAL_VALUES[word.group(1)]

    return None


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


def needs_model(command: ParsedCommand) -> bool:
    """Whether this command can only be answered by asking the model.

    Single source of truth, because both the correction engine (which decides
    whether to call the gateway) and the session (which decides whether to show
    a pending state) need the same answer, and a disagreement between them
    would mean either a pointless round trip or a spinner that never resolves.
    """
    return command.kind in (CommandKind.CORRECT_LAST, CommandKind.CORRECT_PREVIOUS)


def parse_command(command: str) -> ParsedCommand:
    """Parse a spoken command into a `ParsedCommand`.

    Never raises - unparseable input is reported as `UNSUPPORTED`.
    """
    raw = (command or "").strip()
    if not raw:
        return ParsedCommand(CommandKind.UNSUPPORTED, raw, reason="Empty command.")

    text = _strip_filler(normalize_term(raw)).rstrip(".?!, ")

    # The replacement is the user's own final wording, so it is matched against
    # a copy that kept its terminal punctuation. `text` strips it on purpose
    # for the rest of the grammar, but that would silently turn a requested
    # "To deployed." into "To deployed" and make an otherwise no-op rewrite look
    # like a real change.
    verbatim = _strip_filler(raw)

    if _UNSUPPORTED_HINTS.search(text):
        return ParsedCommand(
            CommandKind.UNSUPPORTED,
            raw,
            reason="That operation is not supported by Scriptora yet.",
        )

    # "change the 3rd sentence to X" is matched before the "correct the last
    # subtitle" rules because it is strictly more specific: it requires both a
    # known target phrase and a replacement, so a bare "fix the last subtitle"
    # still falls through to the rule below. Checked first so the more precise
    # intent always wins, regardless of verb.
    for pattern in (_SET_TEXT_FROM, _SET_TEXT):
        match = pattern.match(verbatim)
        if not match:
            continue
        resolved = _parse_target(match.group("target"))
        if resolved is None:
            # A verb-led sentence we do not understand. Refuse rather than guess:
            # guessing here would overwrite the wrong line.
            return ParsedCommand(
                CommandKind.UNSUPPORTED,
                raw,
                reason=(
                    "I could not tell which line to change. Try "
                    '"change the last sentence to ..." or "change sentence 3 to ...".'
                ),
            )
        target, ordinal = resolved
        replacement = _clean_replacement(match.group("replacement"))
        if not replacement or len(replacement) > _MAX_REPLACEMENT:
            return ParsedCommand(
                CommandKind.UNSUPPORTED,
                raw,
                reason='Tell me the new text, e.g. "change the last sentence to To deployed."',
            )
        # The replacement is quoted verbatim, punctuation and all, so a trailing
        # full stop of its own must not be doubled up here. This string is shown
        # to the user in the activity log.
        stop = "" if replacement[-1] in ".?!" else "."
        # Phrased for the user, not from the enum - "Setting ordinal to ..."
        # would leak an internal name into the activity log.
        phrasing = ParsedCommand(
            CommandKind.SET_TEXT, raw, target=target, ordinal=ordinal
        ).describe_target()
        return ParsedCommand(
            CommandKind.SET_TEXT,
            raw,
            replace=replacement,
            target=target,
            ordinal=ordinal,
            reason=f'Setting {phrasing} to "{replacement}"{stop}',
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
        # `_VAGUE_FIND` blocks the pronouns; `_REFERENCES_A_LINE` blocks the
        # "change <line> to X" phrasings whose target we could not resolve, so
        # a command that names a line never degrades into a find/replace against
        # a coincidental substring of some other line.
        if (
            find
            and replace
            and find.casefold() not in _VAGUE_FIND
            and not _REFERENCES_A_LINE.search(find)
        ):
            return ParsedCommand(
                CommandKind.REPLACE,
                raw,
                find=find,
                replace=replace,
                reason=f'Replacing "{find}" with "{replace}".',
            )
        if _REFERENCES_A_LINE.search(find):
            # It really was aiming at a line, we just could not name which one.
            return ParsedCommand(
                CommandKind.UNSUPPORTED,
                raw,
                reason=(
                    "I could not tell which line to change. Try "
                    '"change the last sentence to ...", "change the previous sentence '
                    'to ...", or "change sentence 3 to ...".'
                ),
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
            '"Change the last sentence to ...", "Change sentence 3 to ...", '
            '"Fix the previous subtitle", "Change X to Y", or '
            '"Remember X as a technical term".'
        ),
    )
