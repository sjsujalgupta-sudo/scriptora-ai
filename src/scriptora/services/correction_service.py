"""Correction engine.

Two interchangeable backends behind one result type:

* `LLMCorrector`   - AssemblyAI's LLM Gateway, authenticated with the same
                     AssemblyAI key. Reads the subtitle, its neighbours, the
                     vocabulary and the command, and returns JSON.
* `RuleCorrector`  - deterministic, offline, no network. Handles replace
                     edits and vocabulary re-canonicalisation.

The LLM's reply never reaches application state directly: it is parsed,
validated by Pydantic, re-checked against the ids that actually exist, and
discarded in favour of the rule corrector if any of that fails. A weak or
confused model therefore costs demo quality, not correctness.
"""

from __future__ import annotations

import json
import logging

import httpx

from ..config import LLM_GATEWAY_URL, Settings
from ..models.context import make_lookup_pattern
from ..models.correction import (
    CorrectionAction,
    CorrectionOutcome,
    CorrectionResult,
    SubtitleCorrection,
    VocabularyAddition,
    extract_json_object,
)
from ..models.subtitle import Subtitle
from .command_service import CommandKind, ParsedCommand
from .context_service import ContextService
from .subtitle_service import SubtitleNotFound, SubtitleService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Scriptora, a subtitle correction engine for live speech-to-text.\n"
    "You output ONLY one JSON object. No prose, no explanation, no markdown fences.\n\n"
    "Required JSON shape:\n"
    '{"action": "correct_subtitle" | "add_vocabulary" | "no_action",\n'
    ' "target_subtitle_id": string | null,\n'
    ' "replacement_text": string | null,\n'
    ' "vocabulary_term": string | null,\n'
    ' "reason": string}\n\n'
    "Rules:\n"
    "1. Set target_subtitle_id to the id of the subtitle you were asked to fix. "
    "Use only an id present in the input. Never invent an id.\n"
    "2. If the command is a replace ('change X to Y'), apply exactly that edit.\n"
    "3. Otherwise, correct only clear speech-to-text errors: wrong casing or "
    "spacing of a term in project_vocabulary, and obvious misrecognitions. "
    "Use the exact spelling from project_vocabulary.\n"
    "4. Preserve the speaker's words, order and punctuation. Never summarise, "
    "translate, expand abbreviations or add commentary.\n"
    "5. If nothing needs changing, return action 'no_action'.\n"
    "6. Keep replacement_text under 500 characters."
)


class RuleCorrector:
    """Deterministic corrector. Always available, never fails."""

    name = "rules"

    def correct_subtitle(
        self, subtitle: Subtitle, command: ParsedCommand, context: ContextService
    ) -> str:
        """Return the corrected text for `subtitle`."""
        text = subtitle.text

        if command.kind is CommandKind.REPLACE and command.find and command.replace:
            try:
                pattern = make_lookup_pattern(command.find)
            except ValueError:
                pattern = None
            if pattern is not None:
                text = pattern.sub(command.replace, text)
            else:
                # Term had no usable characters; fall back to a literal swap.
                text = re_escape_replace(text, command.find, command.replace)

        elif command.kind in (CommandKind.CORRECT_LAST, CommandKind.CORRECT_PREVIOUS):
            # "Correct the last subtitle, it's FastAPI" - the user named the
            # answer, so canonicalise any loose spelling of it. This is what
            # makes the primary demo work even before FastAPI is remembered.
            hint = command.find
            if hint:
                try:
                    pattern = make_lookup_pattern(hint)
                except ValueError:
                    pattern = None
                if pattern is not None:
                    text = pattern.sub(hint, text)
                else:
                    text = re_escape_replace(text, hint, hint)

        # Always finish by canonicalising any vocabulary term that is present
        # but misspelled/cased wrongly.
        text = context.restore_terms(text)
        return text.strip()

    def vocabulary_term(self, command: ParsedCommand, context: ContextService) -> str | None:
        return context.parse_remember_command(command.raw) or command.find


def re_escape_replace(text: str, find: str, replace: str) -> str:
    """Case-insensitive literal replace, for terms with no regex-safe form."""
    import re as _re

    return _re.sub(_re.escape(find), replace, text, flags=_re.IGNORECASE)


class LLMCorrector:
    """Corrector backed by AssemblyAI's OpenAI-compatible LLM Gateway."""

    name = "llm"

    def __init__(self, settings: Settings, *, timeout: float = 25.0) -> None:
        self._settings = settings
        self._timeout = timeout

    def _payload(
        self,
        subtitle: Subtitle,
        command: ParsedCommand,
        context: ContextService,
        subtitles: SubtitleService,
    ) -> dict:
        target_id = self._target_id(subtitle)
        return {
            "current_subtitle": {"id": subtitle.id, "text": subtitle.text},
            "previous_subtitles": subtitles.recent(3),
            "all_subtitle_ids": sorted(subtitles.known_ids()),
            "project_vocabulary": context.terms,
            "user_command": command.raw,
            "instructions": {
                "explicit_edit": (
                    {"find": command.find, "replace": command.replace}
                    if command.kind is CommandKind.REPLACE
                    else None
                ),
                "suggested_target_subtitle_id": target_id,
            },
        }

    @staticmethod
    def _target_id(subtitle: Subtitle) -> str | None:
        """Name the target subtitle for the model, which cannot resolve it.

        "Last" and "previous" are resolved by the app in
        `CorrectionService._resolve_target` before we get here, so this must
        NOT walk the list again - doing that once sent the model the line
        before the intended target. It is only offered as a hint; an id the
        model invents is rejected rather than trusted.
        """
        return subtitle.id

    async def correct_subtitle(
        self,
        subtitle: Subtitle,
        command: ParsedCommand,
        context: ContextService,
        subtitles: SubtitleService,
    ) -> SubtitleCorrection:
        body = {
            "model": self._settings.llm_model,
            "max_tokens": 400,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(self._payload(subtitle, command, context, subtitles)),
                },
            ],
        }
        raw = await self._complete(body)
        parsed = SubtitleCorrection.model_validate(extract_json_object(raw))
        return parsed

    async def vocabulary_term(
        self, command: ParsedCommand, context: ContextService
    ) -> VocabularyAddition:
        body = {
            "model": self._settings.llm_model,
            "max_tokens": 200,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "project_vocabulary": context.terms,
                            "user_command": command.raw,
                            "task": "Extract the term the user asked to remember.",
                        }
                    ),
                },
            ],
        }
        raw = await self._complete(body)
        return VocabularyAddition.model_validate(extract_json_object(raw))

    async def _complete(self, body: dict) -> str:
        headers = {"authorization": self._settings.require_api_key()}
        try:
            # A client per call rather than a shared one: corrections are rare,
            # user-initiated events, so the connection setup is not worth
            # optimising, and it leaves no client to leak, close or rebind when
            # several sessions are running side by side. The previous
            # `httpx.post` was synchronous, so a slow gateway blocked the event
            # loop - and with it the audio stream to AssemblyAI - for as long as
            # the request took.
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    LLM_GATEWAY_URL, headers=headers, json=body, timeout=self._timeout
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"LLM gateway unreachable: {type(exc).__name__}") from exc

        if response.status_code != 200:
            # Never echo the body verbatim - it can contain account metadata.
            raise RuntimeError(f"LLM gateway returned HTTP {response.status_code}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM gateway response had no message content") from exc


class CorrectionService:
    """Routes a parsed command to the right subtitle and applies the result."""

    def __init__(
        self,
        settings: Settings,
        subtitles: SubtitleService,
        context: ContextService,
    ) -> None:
        self._settings = settings
        self._subtitles = subtitles
        self._context = context
        self._rules = RuleCorrector()
        self._llm = LLMCorrector(settings) if settings.llm_enabled else None

    @property
    def active_backend(self) -> str:
        return "llm" if self._llm is not None else "rules"

    @property
    def _static_backend(self) -> str:
        """Backend label for results the LLM never contributed to.

        Reporting `active_backend` here would claim the model was used for
        vocabulary edits and for rejections, which is a false attribution in
        the UI's history log.
        """
        return "rules"

    # ------------------------------------------------------------------ api
    def remember(self, command: ParsedCommand) -> CorrectionResult:
        """Add a term to the vocabulary (the "Remember ..." command)."""
        if command.kind is not CommandKind.REMEMBER:
            return CorrectionResult.failure(
                CorrectionOutcome.UNSUPPORTED,
                "That is not a remember command.",
                backend=self._static_backend,
            )

        term = self._context.parse_remember_command(command.raw) or command.find
        if not term:
            return CorrectionResult.failure(
                CorrectionOutcome.UNSUPPORTED,
                "I could not tell which term to remember.",
                backend=self._static_backend,
                reason=command.reason,
            )

        try:
            canonical, added = self._context.add_term(term, source="remember")
        except ValueError as exc:
            return CorrectionResult.failure(
                CorrectionOutcome.INVALID, str(exc), backend=self._static_backend
            )

        message = (
            f'Remembered "{canonical}" - it is now in the project vocabulary.'
            if added
            else f'"{canonical}" was already in the vocabulary.'
        )
        return CorrectionResult(
            outcome=CorrectionOutcome.APPLIED,
            message=message,
            backend="context",
            action=CorrectionAction.ADD_VOCABULARY,
            vocabulary_term=canonical,
        )

    async def correct(self, command: ParsedCommand) -> CorrectionResult:
        """Apply a correction command.

        Async because it may consult the LLM gateway. Callers on the async
        session path must await it, so a slow gateway suspends only this
        command instead of blocking the event loop.
        """
        if command.kind is CommandKind.REMEMBER:
            return self.remember(command)

        if command.kind is CommandKind.UNSUPPORTED:
            return CorrectionResult.failure(
                CorrectionOutcome.UNSUPPORTED,
                command.reason or "Unsupported command.",
                backend=self._static_backend,
            )

        target = self._resolve_target(command)
        if target is None:
            return CorrectionResult.failure(
                CorrectionOutcome.INVALID,
                "There are no subtitles to correct yet. Speak first, then correct.",
                backend=self._static_backend,
                reason=command.reason,
            )

        # 1. Deterministic baseline. Always correct, always safe.
        fallback = self._rules.correct_subtitle(target, command, self._context)
        backend = self._rules.name
        reason = command.reason

        # 2. Try the LLM, but only for commands where it could change the
        #    outcome, and adopt its answer only if it validates. `backend` is
        #    upgraded to "llm" only when the model actually produced the text
        #    that gets applied, so the UI never misattributes a result.
        #
        #    A literal "change X to Y" is skipped entirely: the deterministic
        #    path already performs it exactly, and the model is instructed to
        #    apply exactly that edit too, so its answer could only ever be
        #    discarded. The gateway call is now non-blocking, so this saves a
        #    pointless round trip and its latency rather than protecting the
        #    event loop - but it is still the most common command in the demo,
        #    and a request that cannot change the result is pure latency.
        if self._llm is not None and command.kind is not CommandKind.REPLACE:
            proposal = await self._try_llm(target, command)

            if proposal is not None and proposal.is_valid_for(self._subtitles.known_ids()):
                proposed_text = proposal.replacement_text.strip()

                if proposed_text != target.text.strip():
                    # Normalise vocabulary on top of the model's own answer.
                    fallback = self._context.restore_terms(proposed_text)
                    backend = self._llm.name
                    reason = proposal.reason or command.reason
            elif proposal is None:
                reason = f"{command.reason} (AI unavailable, applied rules)"

        if fallback.strip() == target.text.strip():
            return CorrectionResult(
                outcome=CorrectionOutcome.NO_ACTION,
                message="No correction was needed.",
                backend=backend,
                subtitle_id=target.id,
                before=target.text,
                after=target.text,
                reason=reason,
            )

        before = target.text
        try:
            self._subtitles.correct(target.id, fallback)
        except (SubtitleNotFound, ValueError) as exc:
            return CorrectionResult.failure(
                CorrectionOutcome.ERROR,
                f"Could not apply the correction: {exc}",
                backend=backend,
            )

        return CorrectionResult(
            outcome=CorrectionOutcome.APPLIED,
            message="Subtitle corrected.",
            backend=backend,
            action=CorrectionAction.CORRECT_SUBTITLE,
            subtitle_id=target.id,
            before=before,
            after=fallback,
            reason=reason,
        )

    async def _try_llm(self, target: Subtitle, command: ParsedCommand) -> SubtitleCorrection | None:
        """Ask the LLM for a correction, returning None on any problem.

        Never raises: an unreachable gateway, a non-200, unparseable JSON, a
        schema violation or an invented subtitle id all collapse to None so the
        caller keeps the deterministic result.
        """
        assert self._llm is not None
        try:
            return await self._llm.correct_subtitle(target, command, self._context, self._subtitles)
        except Exception as exc:
            logger.info("LLM correction unavailable (%s); using rules", type(exc).__name__)
            return None

    def _resolve_target(self, command: ParsedCommand) -> Subtitle | None:
        if command.kind is CommandKind.CORRECT_PREVIOUS:
            return self._subtitles.previous()
        return self._subtitles.last()
