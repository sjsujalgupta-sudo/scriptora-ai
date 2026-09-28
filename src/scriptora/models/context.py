"""Project context / vocabulary model.

The vocabulary is the "context" half of context-aware correction. It is
injected into the correction engine and, importantly, is also pushed to
AssemblyAI as `keyterms_prompt` so future turns are transcribed correctly
from the start.
"""

from __future__ import annotations

import hashlib
import re
import time

from pydantic import BaseModel, Field

# Terms are used in a case-insensitive regex, so normalise away characters
# that would otherwise be regex metacharacters (e.g. "FastAPI++", "C++").
_TERM_SAFE = re.compile(r"[^0-9A-Za-z ]+")


def _term_id(canonical: str) -> str:
    """Stable, deterministic id for a term.

    Deliberately not `hash()`, which is salted per process (PYTHONHASHSEED)
    and would change ids between runs.
    """
    digest = hashlib.sha1(canonical.casefold().encode("utf-8")).hexdigest()[:8]
    return f"vocab_{digest}"


def normalize_term(term: str) -> str:
    """Collapse whitespace and trim a vocabulary term."""
    return re.sub(r"\s+", " ", term).strip()


def make_lookup_pattern(term: str) -> re.Pattern[str]:
    """Case-insensitive pattern that finds a vocabulary term however STT split it.

    Speech-to-text writes "FastAPI" as "fast API" or "fast  api" far more often
    than it gets it right, so a plain pattern is not enough - `\bFastAPI\b`
    never matches "fast API". Instead the term is matched with optional
    whitespace between every character:

        "FastAPI"  ->  \\bF\\s*a\\s*s\\s*t\\s*A\\s*P\\s*I\\b

    The leading `\\b` and trailing `\\b` stop the term matching inside a longer
    word, so "Atlas" does not match "Atlases".

    Known tradeoff: because a single inserted space is enough to match, a
    vocabulary term that is also a common phrase ("Atlas" vs "at last") can
    match unintended text. This is why `restore_terms` is only ever applied to
    a subtitle the user explicitly asked to correct, never to incoming audio.
    """
    canonical = normalize_term(term)
    if not canonical:
        raise ValueError(f"vocabulary term is empty: {term!r}")

    # Drop anything outside the alphabet so it cannot break the pattern, then
    # drop existing whitespace so the optional-`\s*` join below is the only
    # thing permitting a split.
    cleaned = re.sub(r"\s+", "", _TERM_SAFE.sub("", canonical))
    if not cleaned:
        raise ValueError(f"vocabulary term has no usable characters: {term!r}")

    body = r"\s*".join(re.escape(char) for char in cleaned)
    try:
        return re.compile(r"\b" + body + r"\b", re.IGNORECASE)
    except re.error as exc:  # pragma: no cover - defensive
        raise ValueError(f"could not build a lookup pattern for {term!r}") from exc


class VocabularyEntry(BaseModel):
    """A single remembered term."""

    id: str
    canonical: str
    source: str = "seed"  # "seed" | "remember" | "manual"
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))

    @property
    def display(self) -> str:
        return self.canonical


class ProjectContext(BaseModel):
    """Project-level context handed to the correction engine."""

    project_name: str = "Scriptora Demo"
    vocabulary: list[VocabularyEntry] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    # ---------------------------------------------------------------- queries
    @property
    def terms(self) -> list[str]:
        """Canonical vocabulary terms, in insertion order."""
        return [entry.canonical for entry in self.vocabulary]

    def has_term(self, term: str) -> bool:
        target = normalize_term(term).casefold()
        return any(entry.canonical.casefold() == target for entry in self.vocabulary)

    def find_term(self, term: str) -> VocabularyEntry | None:
        target = normalize_term(term).casefold()
        for entry in self.vocabulary:
            if entry.canonical.casefold() == target:
                return entry
        return None

    def entry(self, term: str) -> VocabularyEntry | None:
        """Loose lookup: matches 'fast api' against canonical 'FastAPI'."""
        try:
            pattern = make_lookup_pattern(term)
        except ValueError:
            return None
        for entry in self.vocabulary:
            # Apply the input's pattern to the canonical spelling, so a loose
            # transcription of a term finds its canonical entry.
            if pattern.fullmatch(entry.canonical):
                return entry
        return None

    # --------------------------------------------------------------- mutation
    def add_term(self, term: str, *, source: str = "remember") -> VocabularyEntry:
        """Add a term. Idempotent: re-adding an existing term returns it.

        Raises ValueError on empty or unusable input.
        """
        canonical = normalize_term(term)
        if not canonical:
            raise ValueError("vocabulary term must not be empty")
        if len(canonical) > 80:
            raise ValueError("vocabulary term is too long (max 80 characters)")
        # Validate the term can actually be used in a lookup regex.
        make_lookup_pattern(canonical)

        existing = self.find_term(canonical)
        if existing is not None:
            return existing

        entry = VocabularyEntry(
            id=_term_id(canonical),
            canonical=canonical,
            source=source,
        )
        self.vocabulary.append(entry)
        return entry

    def remove_term(self, term: str) -> bool:
        existing = self.find_term(term)
        if existing is None:
            return False
        self.vocabulary = [e for e in self.vocabulary if e.id != existing.id]
        return True

    def as_keyterms(self, limit: int = 40) -> list[str]:
        """Vocabulary rendered for AssemblyAI's `keyterms_prompt`.

        Order is preserved (seed terms first) so the demo's initial context
        stays visible.
        """
        return self.terms[:limit]
