# Demo script

A five-minute demo that runs on the real AssemblyAI API, plus an honest
account of which parts to trust on stage.

## Setup before you present

```bash
pip install -e ".[dev]"
copy .env.example .env      # add your ASSEMBLYAI_API_KEY
pytest                     # 145 passing, no network
python -m scriptora.main
```

Open <http://127.0.0.1:8000>, check the header shows **AssemblyAI connected**
and the backend line, then grant microphone access when prompted.

## The demo (90 seconds)

Pick a project term the model does **not** already know. `Kubernete` is ideal:
the model normalises it to `Kubernetes` unless your vocabulary says otherwise.

**1. Speak a sentence using the term.**

> "we're deploying on Kubernete clusters tonight"

The transcript shows `Kubernetes`. Nothing is wrong yet — that is the point.
The model has no idea `Kubernete` is what you meant.

**2. Correct it by voice.**

> "correct the last subtitle, it's Kubernete"

The line changes to `Kubernete`. The **RAW / CORRECTED** panel shows the
before and after, and the activity log names the backend that did the work.

**3. Remember the term.**

> "remember Kubernete as a technical term"

The project context panel gains an entry, and the activity log confirms
`AssemblyAI keyterms updated with "Kubernete"`. That log line is the important
one: the term went into the **live** session, not just a list.

**4. Say it again.**

> "we're deploying on Kubernete clusters tonight"

This time it stays `Kubernete`, because AssemblyAI now has it in
`keyterms_prompt`. You have just changed what the model hears.

**5. Optional: show a command it will not fake.**

> "delete the last subtitle"

Scriptora answers *"That operation is not supported by Scriptora yet."* It
does not guess. If you have a real mistake in a line, the **raw text is
preserved** and shown next to the correction.

## Backup if the mic misbehaves

Live rooms are loud and browsers are nervous about permissions. Every UI
control has an equivalent typed action — the same code path, no audio:

- **Command** field, then **Run** — sends the identical command the voice path
  would.
- **Add term** in the context panel — the same `add_term` action the spoken
  *"remember …"* produces.
- **Mic off / switch input** — stops cleanly, terminates the AssemblyAI
  session, and closes billing.

## Commands

| Say | Effect |
|---|---|
| "Correct the last subtitle." | Fix the newest line. |
| "Correct the last subtitle, it's X." | Fix it, using X as the correct spelling. |
| "Correct the previous subtitle." | Fix the line before the last. |
| "Change X to Y." | Literal replacement. |
| "Remember X as a technical term." | Add X to project context and push it to AssemblyAI. |
| Anything else | Reported honestly as unsupported. |

## What was actually verified

Checked against the live API, not assumed:

- Partial and final turns stream from `universal-3-5-pro`.
- Frame sizes and format are accepted (errors 3006 / 3007 gone).
- A spoken command is detected, executed, and **withdrawn** from the transcript.
- `Remember …` adds a term and `set_params` genuinely reaches the open session.
- `keyterms_prompt` measurably changes transcription: `Kubernete` is preserved
  once it is a keyterm and normalised to `Kubernetes` when it is not.
- Unsupported commands are reported, not guessed.

## What not to claim on stage

- **Do not stage a deliberate mis-transcription.** This model is very good. On
  short technical phrases it usually normalises spelling rather than getting it
  wrong, so a demo that waits for an organic error can stall. Use a
  project-specific term and correct it *deliberately* via "Change … to …" if
  you want a guaranteed visible edit.
- **Do not claim the AI did the work when the rules did.** The backend label is
  honest for a reason: if the LLM output was rejected, it says `rules`. That is
  the system working, not a gap to hide — the deterministic path is what makes
  the demo reliable.
- **Do not demo remember-before-correct on a term the model rewrites.** If you
  say "Remember Kubernete" and the transcript reads "Remember Kubernetes",
  Scriptora stores `Kubernetes`. Correct the line first, then remember it.
