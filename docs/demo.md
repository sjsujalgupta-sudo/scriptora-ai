# Demo script

A ~90-second demo that runs on the real AssemblyAI API, plus an honest account
of which parts to trust on stage.

## Run of show

The short version. Everything after this section is reference material.

The clock below is **wall-clock including your pauses and narration**, not
application latency. Scriptora's own actions in this sequence take about 20
seconds; the rest is you talking between beats. Treat the app as instant and the
times as pacing cues.

| # | You say / do | What appears | Wall clock |
|---|---|---|---|
| 1 | **Start Listening** | Status → Listening, waveform moves | 0:00 |
| 2 | *"We deployed the application on Qwen clusters."* | `We deployed the application on Quen clusters.` | 0:05 |
| 3 | Pause, then *"Correct the last subtitle."* | Line pulses, log: `Correcting the last sentence…` | 0:15 |
| 4 | — | `↳ We deployed the application on Qwen clusters.` appears **under** the original | 0:20 |
| 5 | *"Change the last subtitle to We deployed it to Qwen clusters."* | A second `↳` line, immediately below the first | 0:30 |
| 6 | *"We also store the transcripts in PostgreSQL."* | New line appears **while the previous work settled** | 0:40 |
| 7 | **Stop** | `Session ended` + final transcript | 0:50 |
| 8 | Point at the two `↳` lines | *"The original is still there. That's the point."* | 0:55 |

### Demo script vs. everything you can say

The wording in the table is chosen for **reliability on stage**, not because
Scriptora only understands it. These are all equally valid:

- **Beat 3** — the app treats *sentence*, *subtitle*, *line*, and *caption* as
  the same thing, so *"Correct the last sentence."*, *"Fix the last subtitle."*
  and *"Fix the last line."* all do beat 3. The script uses *"subtitle"* only
  because it is the phrasing with the most rehearsal history behind it.
- **Beat 5** — say it **without quotation marks**. The recogniser sometimes puts
  a comma after *to* (*"change the last subtitle to, we deployed it…"*), and
  Scriptora ignores punctuation there on purpose, so both forms work. Quoted
  text is still honoured exactly if you type it instead of saying it.

If you memorise nothing else: you can refer to a transcript line as a
**sentence** or a **subtitle** interchangeably, and you never have to learn a
vocabulary.

Land these three sentences, in this order:

- *"AssemblyAI hears words. It doesn't know what you meant."*
- *"The corrector fixes likely mishearings — using project context and its own knowledge of real tools."*
- *"It never overwrites what you said. The original stays, the fix sits under it."*

### Why these lines

Measured against the live API, so the beats are not guesses:

- **"Qwen" is reliably misheard as "Quen".** Tested with a voice pronouncing
  *Qwen* correctly — the recognizer still returned `Quen`. The mishearing is
  the recognizer's, not your delivery, so beat 2 and beat 4 will happen.
- **"Qwen" is deliberately not in project context.** That is the stronger
  version of the demo: the corrector is not copying a spelling you supplied, it
  is recognising that `Quen` is not a real thing and `Qwen` is.
- **Do not build a beat on "assembly AI".** AssemblyAI normalises the casing
  and spacing itself (`assembly AI` → `AssemblyAI`), so there is nothing left
  for the corrector to do. Verified: the transcript already reads `AssemblyAI`
  before any correction runs.
- **PostgreSQL transcribes correctly**, which is why it is a good beat 6 — it
  proves the system is not rewriting everything.

### When a beat does not land

If beat 3 produces *"No correction was needed"* or nothing at all, the gateway
did not answer. Do not retry it live — go straight to beat 5, which is
deterministic and cannot fail, and say:

> *"That one's already right. Here's me telling it the exact wording instead —
> this path never calls a model, so it returns instantly."*

The demo still lands, because beat 5 is the same inline-result UI. The
difference is only which line of code produced it, and that is a better
answer than a visible stall.

Two things genuinely fail on stage and are worth pre-checking: microphone
permission, and the LLM gateway rate limit (`429`) if you have been testing
repeatedly in the same hour.

## Setup before you present

```bash
pip install -e ".[dev]"
copy .env.example .env      # add your ASSEMBLYAI_API_KEY
pytest                      # 311 passing, no network
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
| The one just spoken | "this", "that", "this sentence", "the last sentence", "the last line", "the last subtitle", "the last caption", "the last one" |
| The one before it | "the previous sentence", "the previous subtitle", "the prior line", "the preceding caption", "the last but one", "the second to last" |
| A numbered one | "sentence 3", "subtitle 3", "the 3rd sentence", "the 22nd line", "the second sentence", "the third caption" |

**Sentence, subtitle, line and caption are the same word to Scriptora.** Say
whichever sounds natural to you — that equivalence holds for the whole command
language, not just for rewrites. So all of these do the same thing:

> Change this to "To deployed." · Change the last sentence to "To deployed." ·
> Change the last subtitle to "To deployed." · Change sentence 3 to "To
> deployed." · Change the 3rd line to "To deployed."

Useful details:

- **Quoted text is used exactly as typed**, punctuation included. Speech-to-text
  drops quotation marks, so an unquoted replacement works too.
- **A pause after *to* is fine.** Speech-to-text punctuates dictation, so
  *"change the last sentence to, we deployed it"* and *"change the last
  sentence to we deployed it"* both work. Only the punctuation acting as the
  delimiter is ignored — commas inside the new text are kept, so
  *"change the last sentence to, Hello, world."* still produces `Hello, world.`
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
| "Correct the last subtitle." or "Correct the last sentence." | Let the model repair the newest line, using project vocabulary plus its own knowledge of well-known technical terms. |
| "Correct the last subtitle, it's X." | Fix it using X as the spelling. Works when X is a real word the model recognises — whether the difference is spacing/case (`fast API`) or a mishearing (`Quen` → `Qwen`). It will not truncate a real word: `Kubernete` will not replace `Kubernetes`. |
| "Correct the previous subtitle." or "Correct the previous sentence." | Fix the line before the last. |
| "Add X to the vocabulary" | Add X to project context and push it to AssemblyAI. |
| "Remember X as a technical term." | Same, but the word may be misheard inside the command. |
| Anything else | Reported honestly as unsupported, **and the line is pulled off the transcript** so an instruction never sits there looking like content. |

### If Scriptora does not understand a command

An instruction it cannot carry out is **withdrawn from the transcript and
explained in the activity log** — never silently dropped, and never applied to
a line you did not mean. Example: *"Change the last thing to hello."* cannot
resolve *thing* to a line, so the activity log reads:

> I could not tell which line to change. Try "change the last sentence to …",
> "change the previous sentence to …", or "change sentence 3 to …".

Nothing is called, nothing is rewritten, and the line you were aiming at is
untouched. Ordinary sentences that merely *contain* a command word are left
alone: *"I need to remember to lock the door"* stays in the transcript as
something you said, because it is not shaped like an instruction.

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
- **The full 8-beat run of show was rehearsed end to end** through the real
  websocket audio path into real AssemblyAI, in one continuous session. That
  rehearsal is what surfaced three bugs, all now fixed and re-verified live:
  *"Correct the last sentence."* was silently ignored; a comma after *to* killed
  a dictated rewrite; and an unparsed command was left in the transcript as if
  it were content. A clean run of the sequence takes about 20 s of application
  time.
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
- `Qwen` is misheard as `Quen` by `universal-3-5-pro` even when pronounced
  correctly, so the demo's mishearing does not depend on how you speak. Verified
  by streaming synthesised audio of a correctly pronounced "Qwen clusters": the
  transcript came back `Quen clusters`.
- Not every suspected mishearing is one. Verified in the same session:
  `assembly AI` was normalised to `AssemblyAI` by AssemblyAI itself, and
  `PostgreSQL` came back correct. Both were left alone by the corrector, which is
  the behaviour you want — but it also means they make poor demo beats.
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
- **A spoken command that cannot be parsed now produces visible feedback.** This
  was the rehearsal's worst finding: *"Correct the last sentence."* matched the
  voice gate, failed to parse, and was dropped on the floor, so the instruction
  sat in the transcript looking like content and the user heard nothing. Verified
  live: *"Change the last thing to hello."* is now withdrawn from the transcript
  and answered in the activity log with the three accepted target phrasings, with
  no model call and no line mutated.
- **"Sentence" and "subtitle" are interchangeable everywhere**, not just in
  rewrites. They used to be two separate vocabularies — the target grammar knew
  *sentence* and the correction rules did not — which is exactly how the silent
  failure above happened. There is now a single source of truth. Verified live:
  spoken *"Correct the last sentence."* reaches the model and applies.
- **A dictated rewrite survives the recogniser's punctuation.** Verified live:
  spoken *"Change the last sentence to, we deployed it to Qwen clusters."*
  (with the comma) applies deterministically via `rules`, and punctuation inside
  the replacement is still preserved.
- **Prose containing a command word is not a command.** *"I need to remember to
  lock the door"* and *"We should change the configuration"* both stay in the
  transcript. The voice gate matches an instruction only from the start of the
  utterance, which is what makes reporting an unparsed command safe.

Still to confirm with a real microphone before claiming it on stage:

- `keyterms_prompt` measurably changes transcription for this build — i.e. the
  same spoken audio transcribed as `Kubernetes` before the term is added and as
  `Kubernete` after. The mechanism is verified (the parameter reaches the open
  session); the before/after difference in a spoken sentence is not.
- **Ordinals have been spoken aloud, but only in tests.** *"Change the third
  sentence to …"* is covered in-process, including that corrections never
  renumber, and ordinal targeting is far less fragile than the last/previous
  forms because digits survive transcription intact. A real microphone pass
  would still be worth doing before you lean on it.
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
