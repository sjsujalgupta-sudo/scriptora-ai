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
from .command_service import CommandKind, CorrectionTarget, ParsedCommand, needs_model
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
    "What to correct, in priority order:\n"
    "1. If the command is a replace ('change X to Y'), apply exactly that edit.\n"
    "2. If a word from project_vocabulary appears with wrong casing or spacing, "
    "use the exact spelling from project_vocabulary. Example: 'fast API' becomes "
    "'FastAPI'.\n"
    "3. If a single word is a clear mishearing of a well-known name, product or "
    "technical term, and you are confident of the correct spelling, fix it even "
    "when it is NOT in project_vocabulary. Examples: 'Quen' becomes 'Qwen', "
    "'kubernetties' becomes 'Kubernetes'.\n\n"
    "Rules:\n"
    "- user_named_term is the exact word or phrase the user pointed at. When the "
    "command was 'correct X in the last sentence', X is the word they believe is "
    "wrong: inspect it first. When the command ended '... it's X', X is the text "
    "they believe is right: use that spelling. Either way, never change a named "
    "term that already appears correctly.\n"
    "- Set target_subtitle_id to the id of the subtitle you were asked to fix. "
    "Use only an id present in the input. Never invent an id.\n"
    "- Change as few words as possible. Repair only the misheard word and leave "
    "every other character identical, including punctuation and capitalisation.\n"
    "- Never summarise, translate, expand abbreviations or add commentary.\n"
    "- A word that is merely unfamiliar is NOT an error. If you cannot say what "
    "it should be, leave it exactly as spoken. Never invent a replacement word.\n"
    "- If nothing needs changing, return action 'no_action' with "
    "replacement_text null.\n"
    "- Keep replacement_text under 500 characters."
)


class RuleCorrector:
    """Deterministic corrector. Always available, never fails."""

    name = "rules"

    def correct_subtitle(
        self, subtitle: Subtitle, command: ParsedCommand, context: ContextService
    ) -> str:
        """Return the corrected text for `subtitle`."""
        text = subtitle.text

        if command.kind is CommandKind.SET_TEXT and command.replace:
            # The user stated the replacement outright, so there is nothing to
            # infer. Deliberately not passed through the lookup pattern: "to
            # deployed." must land verbatim, punctuation included.
            return command.replace.strip()

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
                # "Correct Symfpony in the last sentence." names the word the
                # user believes is wrong. Passing it explicitly spares the model
                # from re-parsing the utterance, and stays a hint: it may still
                # answer no_action when the word turns out to be correct.
                "user_named_term": command.find
                if command.kind in (CommandKind.CORRECT_LAST, CommandKind.CORRECT_PREVIOUS)
                else None,
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
    def resolve(self, command: ParsedCommand) -> Subtitle | None:
        """Which line this command refers to, without changing anything.

        Exposed so the session can reject a second correction aimed at a line
        that is already waiting on the model, using the same resolution rules
        the engine would use.
        """
        return self._resolve_target(command)

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
            # Distinguish "nothing spoken yet" from "you asked for a line that
            # does not exist" - the second is a targeting mistake, and telling
            # the user to speak first would be actively misleading.
            if command.target is CorrectionTarget.ORDINAL:
                available = len(self._subtitles.originals())
                if available == 0:
                    detail = "There are no spoken lines yet. Speak first, then correct."
                else:
                    detail = (
                        f"There {'is' if available == 1 else 'are'} {available} spoken "
                        f"{'line' if available == 1 else 'lines'} so far."
                    )
                return CorrectionResult.failure(
                    CorrectionOutcome.INVALID,
                    f"There is no sentence {command.ordinal}. {detail}",
                    backend=self._static_backend,
                    reason=command.reason,
                )
            if command.target is CorrectionTarget.PREVIOUS and self._subtitles.last_original():
                return CorrectionResult.failure(
                    CorrectionOutcome.INVALID,
                    "There is no previous subtitle yet - this is the first one.",
                    backend=self._static_backend,
                    reason=command.reason,
                )
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
        #    Two kinds are skipped entirely. A literal "change X to Y" is
        #    already performed exactly by the deterministic path, and a
        #    "change <line> to <text>" carries the answer in the command, so
        #    neither can be changed by a model. Skipping them is what keeps
        #    the common correction instant: the gateway round trip is the only
        #    thing in this method that costs the user visible time.
        consult_model = needs_model(command)
        if self._llm is not None and consult_model:
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
            else:
                # The model declined - usually `no_action`. It normally explains
                # itself, and that explanation is the only thing that tells the
                # user *why* a subtitle did not change, so it is surfaced rather
                # than dropped on the floor behind a bare "No correction was
                # needed."
                reason = proposal.reason or command.reason

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
            # The original is left untouched; the corrected wording becomes a
            # child line directly after it so the transcript still shows what
            # AssemblyAI actually heard.
            child = self._subtitles.correct_as_child(target.id, fallback)
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
            corrected_subtitle_id=child.id,
            before=before,
            after=child.text,
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
        """Resolve the line a command refers to.

        Resolved against `originals()`, never the raw list, so an inserted
        correction can never be addressed as "the last sentence" or shift the
        meaning of "the third sentence".
        """
        if command.kind is CommandKind.CORRECT_PREVIOUS:
            return self._subtitles.previous_original()
        if command.kind is CommandKind.SET_TEXT and command.target is CorrectionTarget.ORDINAL:
            return self._subtitles.original_at(command.ordinal or 0)
        if command.target is CorrectionTarget.PREVIOUS:
            return self._subtitles.previous_original()
        if command.target is CorrectionTarget.ORDINAL:
            return self._subtitles.original_at(command.ordinal or 0)
        # THIS and LAST both mean the newest line; the distinction exists for the
        # user's benefit, not to select a different subtitle.
        return self._subtitles.last_original()
