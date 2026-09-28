# Architecture

How Scriptora is put together, and why the awkward parts are the way they are.

## Shape

```
Browser                          Server                          AssemblyAI
-------                          ------                          ---------
getUserMedia                       
  │                                
  ▼                                
AudioContext (16 kHz target)        
  │                                
  ▼                                
ScriptProcessorNode
  │  Float32 → Int16 PCM            
  ▼                                
frame slicer
  │  3200 B = 100 ms frames
  ▼                                
WebSocket /ws/audio  ──binary──▶  ScriptoraSession
                                       │
                                       │  queue + normalising generator
                                       ▼
                                  AssemblyAIRealtimeService
                                       │            │
                          websocket ───┘            └── set_params
                                       ▼                    (live keyterms)
                            AsyncRealTimeTranscriber
                                       │
                    partial/final turns ┘
                                       ▼
                            _handle_final → is it a command?
                                                │
                              ┌─────────────────┴──────────────────┐
                              ▼                                    ▼
                     RuleCorrector (always)          LLMCorrector (validated)
                              └─────────────────┬──────────────────┘
                                                ▼
                                        SubtitleService (in-memory)
                                                ▼
                                        ServerEvent stream ──▶ browser
```

## Layout

```
src/scriptora/
  config.py                 Settings; reads env/.env
  main.py                   app factory, static mounting, entrypoint
  models/
    subtitle.py             Subtitle lifecycle: partial → final → corrected
    context.py              ProjectContext, vocabulary entries, term matching
    correction.py           Validated AI response boundary (Pydantic)
    events.py               ServerEvent envelope + EventType
  services/
    subtitle_service.py     In-memory ordered store
    context_service.py      Vocabulary + "remember ..." parsing
    command_service.py      Narrow spoken-command grammar
    correction_service.py   Rules, LLM Gateway, and the adoption decision
    assemblyai_service.py   Realtime v3 wrapper
    session.py              Per-WebSocket orchestration
  api/routes.py             /api/health, /ws/audio
  ui/                       Jinja template, app.js, styles.css
```

Each service takes its dependencies through the constructor and returns plain
values. There is no module-level mutable state, so a test can build a session,
a corrector, or a context in isolation with no patching of globals.

## Design decisions

### One WebSocket for everything

A single `/ws/audio` connection carries audio up and events down, rather than a
socket for audio plus a separate channel for subtitles. Ordering matters: a
correction must arrive after the subtitle it corrects, and a withdrawal must
arrive after the line it removes. One stream makes that ordering a property of
the transport instead of something to coordinate.

Messages in: `hello`, `start`, `audio` (binary), `stop`, `command`, `add_term`,
`remove_term`, `ping`.
Messages out: `hello`, `session_started`, `status`, `subtitle`,
`subtitle_removed`, `correction`, `context`, `activity`, `session_ended`,
`error`.

### The audio path is deliberately dumb

The browser converts to 16 kHz mono signed 16-bit and packs 100 ms (3200-byte)
frames. The server re-chunks into `frame_bytes` anyway, in
`ScriptoraSession._frame_generator`.

That is not redundancy for its own sake. AssemblyAI rejects audio outside
50–1000 ms with error **3007**, and WebSocket framing jitter will eventually
produce a short final frame. Normalising on the server means a whole class of
intermittent, hard-to-reproduce production failures cannot happen at all. The
trailing short frame is dropped rather than sent, with a debug log.

### `_frame_generator` instead of per-frame sends

`AsyncRealTimeTranscriber.stream(data)` is **one-shot**: it accepts a single
`bytes`, `Iterable`, or `AsyncIterable`, drains it, and returns. It is also a
coroutine.

So the session exposes an async generator and the SDK consumes it once, as a
background task. The task ends when `stop()` puts a `None` sentinel on the
queue. Per-chunk `stream()` calls are not the API, and calling the coroutine
without `await` silently discards every frame while appearing to work.

### Rules run first, and are labelled honestly

`CorrectionService.correct` always computes a deterministic answer. The LLM is
consulted second, and its answer is adopted only if it:

1. arrives as parseable JSON (fences and surrounding prose are tolerated),
2. validates against the Pydantic model,
3. names a `target_subtitle_id` that actually exists,
4. differs from the text already there.

If any of those fail, the rules' result is used and `backend` stays `"rules"`.
If the model produced the text that got applied, `backend` becomes `"llm"`.

`backend` is never `active_backend` on a path the model did not touch.
Vocabulary edits report `"context"` and rejected commands report `"rules"`,
because claiming the AI did the work would make the activity log a lie.

#### Literal edits skip the model entirely

A `change X to Y` is a fully specified edit, and the deterministic path already
performs it exactly — the system prompt tells the model to apply "exactly that
edit" too, so there is no outcome the model could produce that the rules do not
already have. The gateway is therefore not called for `REPLACE` at all.

This was originally implemented the other way round: the model was asked, and
its answer was then discarded in favour of the rules, with a log line
("Keeping literal replace over LLM rewrite") explaining the discard. That cost
an HTTP round-trip to learn nothing.

`REPLACE` is the command the demo script uses in step 2, so the most-used
command was paying for an answer that could not be applied. It now costs no
network call at all; `tests/test_correction_service.py` asserts the call count
is zero, so the regression cannot come back quietly.

#### The gateway is never called from the event loop

`_complete` used to be a blocking `httpx.post` invoked from the async session,
so it held the event loop — and therefore audio streaming to AssemblyAI — for
as long as the request took. Against the 25 s timeout, a slow or unreachable
gateway froze transcription for the whole session.

It is now `httpx.AsyncClient`, and the whole chain is awaited:
`LLMCorrector._complete` → `correct_subtitle` → `CorrectionService._try_llm` →
`CorrectionService.correct` → the session. A pending request suspends only that
correction; the audio pump and the websocket keep running. The timeout, the
error handling and the fallback are unchanged — a gateway failure still
collapses to the deterministic result labelled `rules`.

A client is created per call rather than shared. Corrections are rare,
user-initiated events, so the connection setup is not worth optimising, and it
avoids a long-lived pool that would have to be closed on session teardown.

The behavioural test cannot detect a *reintroduced* blocking call: a thread
stuck inside one cannot be woken by any asyncio construct, so the test would
hang rather than fail. Two structural guards cover that case instead — the
coroutine identity of each frame in the chain, and an AST check that no
`httpx.post` call remains in the module.

### The target subtitle is resolved in the app, not the model

"Last" and "previous" are positional. The model is given the resolved subtitle
and an id it may echo, and an id it invents is rejected. `_target_id` returns
`subtitle.id` and does *not* walk the list again — resolving "previous" twice
silently targets the wrong line.

### Vocabulary is pushed into the live AssemblyAI session

Remembering a term is not just bookkeeping. `set_params` is called on the
open session so `keyterms_prompt` includes the term, and AssemblyAI uses it for
subsequent transcription. This is the difference between fixing text after the
fact and changing what gets heard.

`set_params` is a coroutine on the async client. Not awaiting it sent nothing
while still returning `True`, so the activity log claimed a success that never
happened.

`keyterms_prompt` is typed `list[str]`. A comma-joined string is rejected with
error **3006** and kills the session.

### Commands are narrowed on purpose

The grammar is a short list of anchored patterns, not free-form intent
classification. "Delete the last subtitle" is recognised well enough to be
answered honestly with *"That operation is not supported yet"* instead of
being silently ignored or, worse, guessed at.

`_maybe_run_command` only treats a turn as a command if the parser recognises
it as a *supported* one. Dictation like "I need to remember to lock the door"
therefore stays a subtitle.

A recognised command is an instruction, not content, so it is removed from the
transcript. Because the `subtitle` event for that turn has already been sent by
then, an explicit `subtitle_removed` event follows — otherwise the browser
keeps rendering a line the server has deleted.

### The named answer in a correction

"Correct the last subtitle. It's FastAPI." carries the answer. The parser
extracts it (`_named_term`) and the rules canonicalise loose spellings of it.
This means a correction works *before* the term has been remembered, which is
the order a user actually tries it in. Extraction is bounded by length, word
count, and a vague-pronoun list, so "it should be the name of the framework"
is rejected rather than treated as a term.

The substitution is word-anchored, so it only repairs differences of spacing
and case — `fast API` → `FastAPI`, `quill base` → `Quillbase`. It cannot rewrite
a *different* word: `Kubernete` is a prefix of `Kubernetes`, so the anchor
blocks the match and the result is `no_action`. For a genuinely different
spelling, `Change X to Y` is the command that works.

### Audio in the browser

`AudioContext({ sampleRate: 16000 })` and `AudioWorkletNode`, with
`echoCancellation`, `noiseSuppression` and `autoGainControl` enabled — those
matter for subtitle quality, not just comfort. Float samples are converted to
Int16 PCM in the worklet and `ScriptProcessorNode` (still the widest-supported
place to observe the stream) batches them into 3200-byte frames.

### No database

A session is per WebSocket and lives in memory. A hackathon demo does not need
persistence, and in-memory state makes the whole request path easy to reason
about and to test.

## What live testing against AssemblyAI found

Four bugs passed 130 unit tests and were caught only by talking to the real
API. They are now regression-tested in `tests/test_assemblyai_service.py`:

| Symptom | Cause |
|---|---|
| Session dies instantly, error 3006 | `keyterms_prompt` given a `str` instead of `list[str]`. |
| `RuntimeWarning: coroutine ... never awaited`, zero transcripts | `stream()` not awaited, and gated on `_connected` which is still `False` before `Begin` arrives. |
| `RuntimeWarning: ... never awaited` on keyterm update, but reported success | `set_params` not awaited. |
| Dropped trailing frame | Short final frame sent straight to AssemblyAI (error 3007). |

The lesson generalises: the stub accepted a `str` the SDK rejects and a
non-awaited coroutine that looks like a successful call. The AssemblyAI wrapper
now has tests that assert the exact SDK contract — parameter types, that
`stream` is a coroutine that drains its iterable, and that `set_params` is
awaited — rather than tests that only assert our own calls happened.

## Honest limitations

- **A strong model rarely mis-transcribes short phrases.** With
  `universal-3-5-pro`, "Postgres Sequel" comes back as "PostgreSQL" and "fast
  API" as "FastAPI". A demo built on catching an organic error is not
  reliable. The reliable, honest demo is a *project-specific* term the model
  would otherwise normalise — `Kubernete` → `Kubernetes` — or a term with a
  word split (`Quillbase` → `quill base`).
- **Scriptora can only remember the spelling it heard.** If you say "Remember
  Kubernete" and the transcript says "Remember Kubernetes", Scriptora stores
  `Kubernetes`. Use "Change … to …" first when you need to fix a term's
  spelling, then remember it.
- **The LLM backend is best-effort and slow to answer.** A slow gateway delays
  the correction by up to the 25 s timeout, but it no longer blocks anything
  else: the request is awaited, so audio keeps streaming to AssemblyAI while it
  is outstanding. The deterministic path is unaffected, which is why it runs
  first, and a gateway failure still falls back to it. Commands that can be
  answered without the model — literal edits — never wait at all.
- **The command grammar is narrow by design**, so novel phrasings fall through
  to "unsupported" instead of being guessed at.
