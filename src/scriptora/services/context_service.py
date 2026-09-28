"""Project context service.

Owns the vocabulary and knows how to render it two ways:

* as `keyterms_prompt` for AssemblyAI, which makes the *next* turn
  transcribe the term correctly in the first place;
* as vocabulary-aware text for the correction engine, which repairs turns
  that were already transcribed.
"""

from __future__ import annotations

import re

from ..models.context import ProjectContext, make_lookup_pattern, normalize_term

# Seeded so the demo has real context before the user says anything.
#
# FastAPI is deliberately NOT seeded. It is the term the primary demo teaches
# with "Remember FastAPI as a technical term", and seeding it made both halves
# of that demo lie: AssemblyAI already received it as a keyterm and transcribed
# it correctly, so "Correct the last subtitle" had nothing to fix and
# "Remember FastAPI" reported the term was already there. Leaving it out makes
# the correction visible and the memory command actually change the count.
DEFAULT_VOCABULARY = ("AssemblyAI", "Python", "Atlas", "PostgreSQL")


class ContextService:
    def __init__(self, *, project_name: str = "Scriptora Demo") -> None:
        self._context = ProjectContext(project_name=project_name)
        for term in DEFAULT_VOCABULARY:
            self._context.add_term(term, source="seed")

    @property
    def context(self) -> ProjectContext:
        return self._context

    @property
    def terms(self) -> list[str]:
        return self._context.terms

    def add_term(self, term: str, *, source: str = "remember") -> tuple[str, bool]:
        """Add a term. Returns (term, added).

        `added` is False when the term was already present, which lets the
        activity log say "already known" instead of pretending it was new.
        """
        already = self._context.has_term(term)
        entry = self._context.add_term(term, source=source)
        return entry.canonical, not already

    def remove_term(self, term: str) -> bool:
        return self._context.remove_term(term)

    def keyterms(self) -> list[str]:
        """Vocabulary rendered for AssemblyAI's `keyterms_prompt`.

        The SDK types this field as `list[str]`; passing a joined string makes
        AssemblyAI reject the whole session with error 3006.
        """
        return self._context.as_keyterms()

    def restore_terms(self, text: str) -> str:
        """Rewrite `text` so known terms use their canonical spelling.

        Deterministic and offline - this is the fallback correction path and
        also a safety net applied after LLM corrections when a term was missed.
        """
        if not text:
            return text
        result = text
        for term in self._context.terms:
            try:
                pattern = make_lookup_pattern(term)
            except (ValueError, re.error):
                continue
            # Only replace when it differs from canonical, to avoid churn.
            if pattern.search(result) and term not in result:
                result = pattern.sub(term, result)
        return result

    def vocabulary_in_text(self, text: str) -> list[str]:
        """Which vocabulary terms occur in `text` (canonical spelling)."""
        hits: list[str] = []
        for term in self._context.terms:
            try:
                pattern = make_lookup_pattern(term)
            except (ValueError, re.error):
                continue
            if pattern.search(text):
                hits.append(term)
        return hits

    def parse_remember_command(self, command: str) -> str | None:
        """Extract a term from a spoken "remember ..." command.

        Deliberately narrow. Voice commands must be reliable, so this matches
        the handful of phrasings the demo uses rather than attempting general
        natural language.

            "Remember FastAPI as a technical term."
            "Remember Atlas."
            "Please remember to use AssemblyAI."
            "Add FastAPI to the vocabulary."

        Returns the normalized term, or None if nothing usable was found.
        """
        text = normalize_term(command)
        if not text:
            return None

        # Leading politeness is common in speech and carries no intent.
        text = re.sub(r"^(?:please|okay|ok|hey)\s+", "", text, flags=re.IGNORECASE)

        patterns = (
            # "remember X as <a/an/the> <kind>" - drop the trailing description
            r"^remember\s+(?P<term>.+?)\s+as\s+(?:a\s+|an\s+|the\s+)?\w+",
            # "remember to use/say/call X"
            r"^remember\s+to\s+(?:use|say|call|spells?|pronounce)\s+(?P<term>.+)$",
            r"^(?:add|include)\s+(?P<term>.+?)\s+to\s+the\s+(?:vocabulary|context|project)",
            r"^remember\s+(?P<term>.+)$",
        )
        for raw in patterns:
            match = re.search(raw, text, re.IGNORECASE)
            if not match:
                continue
            term = normalize_term(match.group("term"))
            # Drop trailing filler the speaker may have added.
            term = re.sub(
                r"\s+(?:please|okay|ok|thanks|as well)\.?$", "", term, flags=re.IGNORECASE
            )
            term = term.strip(" .?!,")
            if term and len(term) <= 80:
                return term
        return None
