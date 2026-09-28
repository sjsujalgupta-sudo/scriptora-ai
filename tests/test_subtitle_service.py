"""Subtitle state: creation, partial/final transitions, correction, lookups."""

from __future__ import annotations

import pytest

from scriptora.models.subtitle import Subtitle, SubtitleStatus
from scriptora.services.subtitle_service import SubtitleNotFound, SubtitleService


def test_add_creates_partial_subtitle_with_id(subtitles: SubtitleService):
    sub = subtitles.add("hello world")

    assert sub.text == "hello world"
    assert sub.status is SubtitleStatus.PARTIAL
    assert sub.id
    assert len(subtitles) == 1


def test_ids_are_unique(subtitles: SubtitleService):
    ids = {subtitles.add(f"line {i}").id for i in range(20)}
    assert len(ids) == 20


def test_upsert_partial_updates_in_place_instead_of_duplicating():
    store = SubtitleService()
    store.upsert_partial("today we are")
    store.upsert_partial("today we are building")
    store.upsert_partial("today we are building the backend")

    # A live turn is re-sent whole on every event; it must stay one line.
    assert len(store) == 1
    assert store.last().text == "today we are building the backend"


def test_finalize_promotes_the_partial_in_place():
    store = SubtitleService()
    store.upsert_partial("using fast API")
    sub = store.finalize("Using fast API.")

    assert len(store) == 1
    assert sub.status is SubtitleStatus.FINAL
    assert sub.text == "Using fast API."
    assert sub.end_time is not None


def test_finalize_appends_when_there_is_no_partial():
    store = SubtitleService()
    sub = store.finalize("a finalized turn with no prior partial")

    assert len(store) == 1
    assert sub.status is SubtitleStatus.FINAL


def test_finalize_ignores_empty_transcripts():
    """Empty AssemblyAI turns must never create blank subtitle lines."""
    store = SubtitleService()

    assert store.finalize("") is None
    assert store.finalize("   ") is None
    assert len(store) == 0


def test_finalize_records_raw_text_when_text_changes():
    store = SubtitleService()
    store.upsert_partial("hello")
    sub = store.finalize("Hello there.")

    assert sub.text == "Hello there."
    assert sub.raw_text == "hello"


def test_correction_preserves_original_transcript():
    store = SubtitleService()
    sub = store.add("Today we are building the backend using fast API.")
    sub.finalize()

    store.correct(sub.id, "Today we are building the backend using FastAPI.")

    assert store.get(sub.id).status is SubtitleStatus.CORRECTED
    assert store.get(sub.id).text.endswith("using FastAPI.")
    # The raw transcript must survive for the RAW vs CORRECTED comparison.
    assert store.get(sub.id).raw_text == "Today we are building the backend using fast API."
    assert store.get(sub.id).was_corrected is True


def test_correction_rejects_empty_replacement():
    store = SubtitleService()
    sub = store.add("some text")

    with pytest.raises(ValueError):
        store.correct(sub.id, "   ")


def test_correction_of_unknown_id_raises():
    store = SubtitleService()
    store.add("some text")

    with pytest.raises(SubtitleNotFound):
        store.correct("does-not-exist", "new text")


def test_get_unknown_id_raises(subtitles: SubtitleService):
    with pytest.raises(SubtitleNotFound) as exc:
        subtitles.get("nope")
    assert exc.value.subtitle_id == "nope"


def test_find_returns_none_for_unknown_id(subtitles: SubtitleService):
    assert subtitles.find("nope") is None


def test_previous_skips_partials():
    store = SubtitleService()
    store.finalize("first line")
    store.upsert_partial("still speaking")

    # "Fix the previous subtitle" must not target an in-progress utterance.
    assert store.previous().text == "first line"


def test_previous_of_last_returns_the_line_before_it():
    store = SubtitleService()
    store.finalize("first line")
    store.finalize("second line")
    store.finalize("third line")

    assert store.previous().text == "second line"
    assert store.previous(before_id=store.last().id).text == "second line"


def test_previous_returns_none_when_only_one_line():
    store = SubtitleService()
    store.finalize("only line")

    assert store.previous() is None


def test_recent_returns_newest_first():
    store = SubtitleService()
    store.finalize("one")
    store.finalize("two")
    store.finalize("three")

    assert store.recent(2) == ["three", "two"]


def test_remove_drops_a_subtitle():
    store = SubtitleService()
    sub = store.add("a command, not a subtitle")

    store.remove(sub.id)

    assert len(store) == 0


def test_max_items_trims_oldest():
    store = SubtitleService(max_items=3)
    for i in range(5):
        store.finalize(f"line {i}")

    assert len(store) == 3
    assert [s.text for s in store] == ["line 2", "line 3", "line 4"]


def test_subtitle_defaults_are_sane():
    sub = Subtitle(text="hi")
    assert sub.is_provisional is True
    assert sub.was_corrected is False
    assert sub.start_time > 0
