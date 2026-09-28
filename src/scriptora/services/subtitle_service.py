"""In-memory subtitle store for one listening session.

Kept deliberately small: an ordered list plus id lookup. Persistence is out of
scope for the MVP - each browser session starts with a clean transcript.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..models.subtitle import Subtitle, SubtitleStatus


class SubtitleNotFound(LookupError):
    """Raised when a correction targets a subtitle id that does not exist."""

    def __init__(self, subtitle_id: str) -> None:
        super().__init__(f"No subtitle with id {subtitle_id!r}")
        self.subtitle_id = subtitle_id


class SubtitleService:
    """Ordered subtitle collection with partial/final semantics."""

    def __init__(self, max_items: int = 200) -> None:
        self._items: list[Subtitle] = []
        self._max_items = max_items

    # ------------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Subtitle]:
        return iter(self._items)

    @property
    def items(self) -> list[Subtitle]:
        return list(self._items)

    def known_ids(self) -> set[str]:
        return {item.id for item in self._items}

    def get(self, subtitle_id: str) -> Subtitle:
        for item in self._items:
            if item.id == subtitle_id:
                return item
        raise SubtitleNotFound(subtitle_id)

    def find(self, subtitle_id: str) -> Subtitle | None:
        for item in self._items:
            if item.id == subtitle_id:
                return item
        return None

    def last(self) -> Subtitle | None:
        return self._items[-1] if self._items else None

    def previous(self, before_id: str | None = None) -> Subtitle | None:
        """The subtitle preceding `before_id`, or the last-but-one by default.

        Skips partials, because "fix the previous subtitle" should not resolve
        to an utterance still being revised.
        """
        if before_id is not None:
            for index, item in enumerate(self._items):
                if item.id == before_id:
                    target_index = index
                    break
            else:
                raise SubtitleNotFound(before_id)
        else:
            # Exclude the most recent line: "previous" means the one before it.
            target_index = len(self._items) - 1

        for item in reversed(self._items[:target_index]):
            if not item.is_provisional:
                return item
        return None

    def finalized(self) -> list[Subtitle]:
        return [item for item in self._items if not item.is_provisional]

    def originals(self) -> list[Subtitle]:
        """Finalized lines that are *not* corrections of another line.

        Every "which line?" question - this, last, previous, the third sentence -
        must be answered against this list. Corrections are interleaved in
        `_items` so the UI can render them under their original, which means
        counting `_items` directly would let a correction answer "the third
        sentence" and shift every later ordinal.
        """
        return [item for item in self._items if not item.is_provisional and not item.is_correction]

    def last_original(self) -> Subtitle | None:
        lines = self.originals()
        return lines[-1] if lines else None

    def previous_original(self) -> Subtitle | None:
        """The line before the most recent original."""
        lines = self.originals()
        return lines[-2] if len(lines) >= 2 else None

    def original_at(self, ordinal: int) -> Subtitle | None:
        """The `ordinal`-th original line, 1-based. None when out of range."""
        if ordinal < 1:
            return None
        lines = self.originals()
        return lines[ordinal - 1] if ordinal <= len(lines) else None

    def corrections_of(self, subtitle_id: str) -> list[Subtitle]:
        """Corrections attached to one line, in the order they were made."""
        return [item for item in self._items if item.corrects_id == subtitle_id]

    def recent(self, limit: int = 3) -> list[str]:
        """Most recent finalized texts, newest first - LLM context window."""
        return [item.text for item in reversed(self.finalized())][:limit]

    # -------------------------------------------------------------- mutation
    def add(
        self,
        text: str,
        *,
        turn_order: int | None = None,
        status: SubtitleStatus = SubtitleStatus.PARTIAL,
    ) -> Subtitle:
        subtitle = Subtitle(text=text, turn_order=turn_order, status=status)
        if status is not SubtitleStatus.PARTIAL and subtitle.end_time is None:
            subtitle.end_time = subtitle.start_time
        self._items.append(subtitle)
        self._trim()
        return subtitle

    def upsert_partial(
        self,
        text: str,
        *,
        turn_order: int | None = None,
        confidence: float | None = None,
    ) -> Subtitle:
        """Insert or update the in-progress turn.

        AssemblyAI re-sends the whole running turn on every `Turn` event, so
        the correct behaviour is to update the newest partial in place rather
        than append a new line. This is what keeps the live view from filling
        up with duplicates.
        """
        for item in reversed(self._items):
            if item.status is SubtitleStatus.PARTIAL:
                item.text = text
                if turn_order is not None:
                    item.turn_order = turn_order
                if confidence is not None:
                    item.confidence = confidence
                return item
        return self.add(text, turn_order=turn_order)

    def finalize(
        self,
        text: str,
        *,
        turn_order: int | None = None,
        end_time: int | None = None,
    ) -> Subtitle | None:
        """Commit the in-progress turn, or append a new one.

        Returns None when `text` is empty, so blank transcripts never create
        empty subtitle lines.
        """
        if not text or not text.strip():
            return None

        for item in reversed(self._items):
            if item.status is SubtitleStatus.PARTIAL:
                item.finalize(text, end_time=end_time)
                if turn_order is not None:
                    item.turn_order = turn_order
                return item
        # No turn in progress, so this is a complete utterance in its own right.
        return self.add(text, turn_order=turn_order, status=SubtitleStatus.FINAL)

    def correct(self, subtitle_id: str, replacement_text: str) -> Subtitle:
        """Apply a validated correction. Raises on unknown id or empty text."""
        subtitle = self.get(subtitle_id)
        subtitle.apply_correction(replacement_text)
        return subtitle

    def correct_as_child(self, subtitle_id: str, replacement_text: str) -> Subtitle:
        """Record a correction as a new line directly after its original.

        The original's `text` is never touched: what AssemblyAI heard stays in
        the transcript, and the corrected wording is a separate child that
        references it. That is what lets the UI show both, and what keeps
        "change the 3rd sentence" addressing the same line it did before any
        correction existed.

        A partial line is still corrected in place, because it is a live
        utterance with nothing settled to preserve and inserting a sibling would
        leave a duplicate behind.
        """
        replacement_text = (replacement_text or "").strip()
        if not replacement_text:
            raise ValueError("replacement_text must not be empty")

        original = self.get(subtitle_id)
        if original.is_provisional:
            original.apply_correction(replacement_text)
            return original

        child = Subtitle(
            text=replacement_text,
            status=SubtitleStatus.CORRECTED,
            turn_order=original.turn_order,
            # The comparison panel reads before/after; carrying the original
            # text on the child keeps that working without another lookup.
            raw_text=original.text,
            corrects_id=original.id,
        )
        child.end_time = original.end_time or child.start_time
        # Insert after this line's existing corrections rather than straight
        # after the original. Inserting at the original's index every time
        # would put the newest correction first, so the transcript and
        # corrections_of() would list the edits in reverse.
        insert_at = self._items.index(original) + 1
        while insert_at < len(self._items) and self._items[insert_at].corrects_id == original.id:
            insert_at += 1
        self._items.insert(insert_at, child)
        self._trim()
        return child

    def remove(self, subtitle_id: str) -> None:
        """Drop a subtitle.

        Used when a finalized turn was a spoken command: the instruction
        should not also appear in the transcript.
        """
        self._items = [item for item in self._items if item.id != subtitle_id]

    def clear(self) -> None:
        self._items.clear()

    def _trim(self) -> None:
        if len(self._items) > self._max_items:
            del self._items[: len(self._items) - self._max_items]
