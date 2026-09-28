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
