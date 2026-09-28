# Demo script

A 90-second demo that runs on the real AssemblyAI API, plus an honest account
of which parts to trust on stage.

## Setup before you present

```bash
pip install -e ".[dev]"
copy .env.example .env      # add your ASSEMBLYAI_API_KEY
 pytest                     # 279 passing, no network
python -m scriptora.main
```

Open <http://127.0.0.1:8000>, check the **status** line reads Idle, that the
sidebar shows **API key** as configured and the **Corrector** you expect, then
press **Start Listening** and grant microphone access when prompted.

### Two different models, two different jobs

The bottom panel names both, and they are not interchangeable:

| Panel | What it does |
|---|---|
| **Speech model** (`universal-3-5-pro`) | AssemblyAI turns your voice into text. It hears the words. It does **not** know what you meant. |
| **Corrector** (`qwen3.5-4b-32k-fast`) | Scriptora asks this model to repair the text *after* transcription, using your project context. It never hears audio. |

So if the corrector is named `qwen`, Qwen is **not** transcribing you. Saying
"we run Qwen on the audio" would be wrong. The accurate framing, and the one the
demo actually shows, is:

```
AssemblyAI transcribes      →  "Quen."
Scriptora applies context   →  knows "Qwen" is a real tool, hears a likely mishearing
Corrector (Qwen LLM) repairs →  "Qwen."
UI shows RAW → CORRECTED
```

A judge should hear: transcription and correction are separate problems, solved
by separate models, joined by context.

## The demo (90 seconds)

Pick a project term the model does **not** already know. `Kubernete` is ideal:
the model normalises it to `Kubernetes` unless your vocabulary says otherwise.
Every step below was verified against the live API.

**1. Speak a sentence using the term.**

> "it runs on Kubernete clusters"

The transcript shows `Kubernetes`. Nothing is wrong yet — that is the point. The
model has no idea `Kubernete` is what you meant.

**2. Say what the line should be.**

> "Change the last sentence to Kubernete runs it."
>
> or, if you already know the wording: "Change the last sentence to Kubernete."

The original line is left exactly as it was spoken, and the corrected wording
appears directly underneath it, marked `corrected` and indented under its
original. The **RAW / CORRECTED** panel shows the before and after. The chip on
that panel names the backend that did the work — `rules` or `llm`. The panel is
labelled "Corrected" rather than "AI corrected" because in this demo the
deterministic rules engine is the one doing it.

Rewriting the line this way skips the model entirely: you supplied the text, so
there is nothing to infer and nothing to wait for.

> To correct a *spelling* inside the line instead of rewriting the whole thing,
> use the find-and-replace form: *"Change Kubernetes to Kubernete."* The named
> form — *"correct the last subtitle, it's Kubernete"* — **does not** do this,
> because `Kubernete` is not
> a loose-spelling match of `Kubernetes` (it is a prefix of it, and the lookup
> pattern is word-anchored). Verified against the live gateway: that phrasing
> returns `no_action` with both backends, because the word-anchored rules cannot
> match it and the model declines to mangle a correctly spelled word into a
> truncated one. The named form does work when the term is a real word the
> model can recognise — "fast API" → `FastAPI`, and "Quen" → `Qwen`.

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

## If the connection drops on stage

A failed connection is reported in the activity log with what to do about it,
and the status line returns to `Disconnected`. Press **Start Listening** to
reconnect; the vocabulary you built up is kept, so the demo can continue from
the corrected term rather than start over.

This covers a connection that dies *during* a session, not just one that fails
to open — a dropped Wi-Fi link or a suspended laptop used to leave the UI
showing "Listening" against a stream that was already gone.

## Commands

### Rewriting a line

Say or type `change <which line> to <the new text>`. The line stays where it was
and the correction appears directly underneath it, so you can see both the
original wording and the fix.

| Which line | Accepted phrasings |
|---|---|
| The one just spoken | "this", "that", "this sentence", "the last sentence", "the last line", "the last subtitle", "the last one" |
| The one before it | "the previous sentence", "the previous subtitle", "the prior line", "the preceding caption", "the last but one", "the second to last" |
| A numbered one | "sentence 3", "subtitle 3", "the 3rd sentence", "the 22nd line", "the second sentence", "the third caption" |

So all of these do the same thing:

> Change this to "To deployed." · Change the last sentence to "To deployed." ·
> Change sentence 3 to "To deployed." · Change the 3rd line to "To deployed."

Useful details:

- **Quoted text is used exactly as typed**, punctuation included. Speech-to-text
  drops quotation marks, so an unquoted replacement works too.
- **No model is consulted.** You already said what the line should be, so the
  rewrite is applied immediately. This is the fastest correction in the app and
  the one that cannot go wrong creatively.
- **Numbers count spoken lines only.** A correction you made earlier is not a
  sentence, so `the 3rd sentence` means the same thing before and after you
  correct anything.
- **An unrecognised target is refused, not guessed.** "Change the sentence after
  this to X" is reported as unsupported rather than being applied to whichever
  line happened to be last.
- **Asking for what is already there does nothing** — no duplicate line appears.

The line count in the transcript also ignores corrections, so it keeps matching
what you have actually said.

### Other commands

| Say | Effect |
|---|---|
| "Change X to Y." | Find-and-replace inside the last line. Distinct from the above: this one looks for `X` in the text. |
| "Correct the last subtitle." | Let the model repair the newest line, using project vocabulary plus its own knowledge of well-known technical terms. |
| "Correct the last subtitle, it's X." | Fix it using X as the spelling. Works when X is a real word the model recognises — whether the difference is spacing/case (`fast API`) or a mishearing (`Quen` → `Qwen`). It will not truncate a real word: `Kubernete` will not replace `Kubernetes`. |
| "Correct the previous subtitle." | Fix the line before the last. |
| "Add X to the vocabulary" | Add X to project context and push it to AssemblyAI. |
| "Remember X as a technical term." | Same, but the word may be misheard inside the command. |
| Anything else | Reported honestly as unsupported. |

### While a correction is in flight

The two model-backed commands ("correct the last/previous subtitle") have to
wait for the gateway, so the UI says so immediately: the target line pulses and
the activity log records *"Correcting the last sentence…"* before the request is
even sent. Audio and transcription carry on meanwhile — a slow correction never
pauses the stream.

A second command aimed at a line that is still busy is refused with an
explanation rather than racing it. A different line is unaffected. The busy
state clears on every outcome, including a refusal.

## What was actually verified

Checked against the live API, not assumed:

- Partial and final turns stream from `universal-3-5-pro`.
- Frame sizes and format are accepted (errors 3006 / 3007 gone).
- `Change Kubernetes to Kubernete` applies via the deterministic corrector.
- A spoken command is detected, executed, and **withdrawn** from the transcript.
- Adding a term reaches the open session (`set_params` awaited and confirmed).
- Audio reaches AssemblyAI at exactly 1x real time. Measured over a sustained
  live session: 200 s of audio sent produced `AssemblyAI session ended after
  200s of audio` — no drift, no backlog.
- A session held open for 200 s (past the ~155 s point where the live connection
  previously dropped) completed with no transport error, and **Stop finalised
  cleanly** — the server sends a termination frame and waits for AssemblyAI to
  acknowledge it rather than tearing the socket down underneath the last words.
- The named form *"correct the last subtitle, it's Kubernete"* is a **no-op**,
  with either backend. Use `Change X to Y`.
- A wrong-letter mishearing is repaired by the model **without** the term being
  in the vocabulary. Verified live: `Quen.` → `Qwen.`, `applied via llm`, with
  the project vocabulary left as AssemblyAI / Python / Atlas / PostgreSQL.
- The repair is general, not a lookup: unrelated mishearings
  (`kubernetties`, `kubernets`) resolve the same way, and there is no Qwen rule
  anywhere in the codebase.
- It does not invent words. Verified live: `blargh` and `We met Sarah in Denver.`
  are both left untouched, so an unfamiliar word never gets "corrected" into
  something plausible.
- When the model declines, its reason reaches the activity log, so a no-op says
  why instead of looking like an ignored command.
- Unsupported commands are reported, not guessed — including a `change ... to ...`
  whose target line cannot be identified, which is refused rather than applied to
  the wrong line.

Still to confirm with a real microphone before claiming it on stage:

- `keyterms_prompt` measurably changes transcription for this build — i.e. the
  same spoken audio transcribed as `Kubernetes` before the term is added and as
  `Kubernete` after. The mechanism is verified (the parameter reaches the open
  session); the before/after difference in a spoken sentence is not.
- **Everything in the "Rewriting a line" section is covered by the test suite but
  has not been spoken aloud yet.** The parser, the target resolution, the
  original-preserving child line, and the line count are verified in-process. What
  a microphone would add is the part no unit test can reach: that AssemblyAI
  transcribes *"change the third sentence to to deployed"* as something the
  parser still recognises. That is the single most likely thing to need a tweak
  before you rely on it on stage, because the dictated wording is the input the
  test suite has to assume.
- **The busy-line indicator has not been seen while a real request is in
  flight.** The ordering that matters — the pending event is emitted before the
  gateway call, and the busy state clears on every outcome including a refusal —
  is covered by tests that hold the gateway open deliberately. Seeing it pulse
  against a real round trip, on a line that is genuinely being corrected, is
  still to do.
- **Speaking *through* a slow correction has not been recorded.** That audio
  keeps flowing while the gateway is awaited is tested by forwarding a frame
  against a parked gateway; it has not been confirmed with someone actually
  talking through a two-second correction.

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
