# Demo script

A 90-second demo that runs on the real AssemblyAI API, plus an honest account
of which parts to trust on stage.

## Setup before you present

```bash
pip install -e ".[dev]"
copy .env.example .env      # add your ASSEMBLYAI_API_KEY
pytest                     # 145 passing, no network
python -m scriptora.main
```

Open <http://127.0.0.1:8000>, check the **status** line reads Idle, that the
sidebar shows **API key** as configured and the **Corrector** you expect, then
press **Start Listening** and grant microphone access when prompted.

## The demo (90 seconds)

Pick a project term the model does **not** already know. `Kubernete` is ideal:
the model normalises it to `Kubernetes` unless your vocabulary says otherwise.
Every step below was verified against the live API.

**1. Speak a sentence using the term.**

> "it runs on Kubernete clusters"

The transcript shows `Kubernetes`. Nothing is wrong yet — that is the point. The
model has no idea `Kubernete` is what you meant.

**2. Correct it by voice — use the explicit form.**

> "Change Kubernetes to Kubernete."

The line changes to `Kubernete`, and the **RAW / CORRECTED** panel shows the
before and after. The activity log names the backend that did the work.

> Use the `Change X to Y` phrasing. The named form — *"correct the last
> subtitle, it's Kubernete"* — **does not** do this, because `Kubernete` is not
> a loose-spelling match of `Kubernetes` (it is a prefix of it, and the lookup
> pattern is word-anchored). Verified: that phrasing returns `no_action`, even
> with the LLM backend. The named form only helps when the term differs from
> what was heard by *spacing or case* — "fast API" → `FastAPI`, "quill base" →
> `Quillbase`.

**3. Remember the term.**

> "Add Kubernete to the vocabulary"

The project context panel gains an entry, and the activity log confirms
`AssemblyAI keyterms updated with "Kubernete"`. That log line is the important
one: the term went into the **live** session, not just a list.

> Prefer this phrasing over *"Remember Kubernete as a technical term"*.
> AssemblyAI sometimes transcribes the word inside the command itself
> ("Remember Kubernetes…"), and Scriptora can only remember the spelling it
> heard. If the context panel shows the wrong term, correct it there — or add
> it with the **Add term** box, which never goes through speech.

**4. Say it again.**

> "it runs on Kubernete clusters"

This time it stays `Kubernete`, because AssemblyAI now has it in
`keyterms_prompt`. Verified: the same audio transcribed as `Kubernetes`
before the term was added and as `Kubernete` after. You have just changed
what the model hears.

**5. Optional: show a command it will not fake.**

> "delete the last subtitle"

Scriptora answers *"That operation is not supported by Scriptora yet."* It does
not guess.

## Backup if the mic misbehaves

Live rooms are loud and browsers are nervous about permissions. Every UI
control has an equivalent typed action — the same code path, no audio:

- **Command** field, then **Run** — sends the identical command the voice path
  would.
- **Add term** in the project context panel — the same `add_term` action the
  spoken *"Add X to the vocabulary"* produces, and it never goes through
  speech, so it cannot be misheard.
- **Stop** — tears the session down cleanly, terminates the AssemblyAI session,
  and stops billing.

## Commands

| Say | Effect |
|---|---|
| "Change X to Y." | Literal replacement. **The most reliable correction.** |
| "Correct the last subtitle." | Fix the newest line using project vocabulary. |
| "Correct the last subtitle, it's X." | Fix it using X as the spelling — only when X differs by spacing or case. |
| "Correct the previous subtitle." | Fix the line before the last. |
| "Add X to the vocabulary" | Add X to project context and push it to AssemblyAI. |
| "Remember X as a technical term." | Same, but the word may be misheard inside the command. |
| Anything else | Reported honestly as unsupported. |

## What was actually verified

Checked against the live API, not assumed:

- Partial and final turns stream from `universal-3-5-pro`.
- Frame sizes and format are accepted (errors 3006 / 3007 gone).
- `Change Kubernetes to Kubernete` applies via the deterministic corrector.
- A spoken command is detected, executed, and **withdrawn** from the transcript.
- Adding a term reaches the open session (`set_params` awaited and confirmed).
- `keyterms_prompt` measurably changes transcription: the same audio
  transcribed as `Kubernetes` before the term was added and as `Kubernete`
  after.
- The named form *"correct the last subtitle, it's Kubernete"* is a **no-op**,
  with either backend. Use `Change X to Y`.
- Unsupported commands are reported, not guessed.

## What not to claim on stage

- **Do not stage a deliberate mis-transcription.** This model is very good. On
  short technical phrases it usually normalises spelling rather than getting it
  wrong, so a demo that waits for an organic error can stall.
- **Do not rely on the named correction form for a term the model rewrote.**
  `Kubernete` is a prefix of `Kubernetes`, so word-anchored matching will not
  substitute it. Use `Change X to Y`, which is verified to work.
- **Do not claim a correction changes what AssemblyAI hears.** A correction
  updates Scriptora's subtitle state and the UI. Only `keyterms_prompt` changes
  AssemblyAI's transcription, and that requires adding the term to the
  vocabulary (step 3).
- **Do not claim the AI did the work when the rules did.** The backend label is
  honest for a reason: if the LLM output was rejected, it says `rules`. That is
  the system working, not a gap to hide — the deterministic path is what makes
  the demo reliable.
