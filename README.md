# Scriptora

Context-aware, voice-controlled AI subtitle agent powered by
[AssemblyAI](https://assemblyai.com) real-time speech-to-text.

Speak into your microphone and subtitles appear live. Correct a line by talking
about it. Teach Scriptora a project term once, and it remembers it — both when
correcting text and when transcribing your next sentence.

```
you  ->  "it runs on Kubernete clusters"
ui   ->  1. it runs on Kubernetes clusters
you  ->  "Change Kubernetes to Kubernete."
ui   ->  1. it runs on Kubernete clusters               [corrected]
you  ->  "Add Kubernete to the vocabulary"
ui   ->  project context: AssemblyAI, Python, Atlas, PostgreSQL, Kubernete
                              AssemblyAI keyterms updated with "Kubernete"
you  ->  "it runs on Kubernete clusters"
ui   ->  2. it runs on Kubernete clusters               [heard correctly now]
```

The last line is the point: the term reached AssemblyAI's live
`keyterms_prompt`, so the model now *hears* your project the way you say it
instead of normalising it to a word it already knows.

## Why this exists

Speech-to-text is accurate, but it has no idea what *your* project calls things.
It will happily rewrite `Kubernete` to `Kubernetes` because that is a word it
knows, and it will not learn that you meant otherwise. Scriptora adds the part
that is missing: a project vocabulary that you build by speaking, and a spoken
interface for fixing lines without touching the keyboard.

## Features

| | |
|---|---|
| **Live subtitles** | Real-time partial and final turns from AssemblyAI `universal-3-5-pro`. |
| **Spoken commands** | Correct, replace, or remember — narrow grammar, no wake word. |
| **Two correction backends** | Deterministic rules that never fail, plus an LLM for real edits. |
| **Project vocabulary** | Remembered terms feed AssemblyAI `keyterms_prompt` **live**, mid-session. |
| **Before / after** | Every correction shows the raw text next to the corrected text. |
| **Honest reporting** | If nothing needed fixing, it says so. It never claims the AI did work the rules did. |

## Quick start

Requires Python 3.11+ (developed on 3.12).

```bash
git clone https://github.com/sjsujalgupta-sudo/scriptora-ai.git
cd scriptora-ai

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -e ".[dev]"

copy .env.example .env          # then paste your key into .env
```

Get a key from the [AssemblyAI dashboard](https://dashboard.assemblyai.com),
then:

```bash
python -m scriptora.main
# or, once installed:  scriptora
```

Open <http://127.0.0.1:8000>, press **Start Listening**, allow microphone access, and talk.

### Configuration

Everything is optional except the key.

| Variable | Default | Purpose |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | — | **Required.** Used for both STT and the LLM Gateway. |
| `SCRIPTORA_HOST` | `127.0.0.1` | Bind address. |
| `SCRIPTORA_PORT` | `8000` | Port. |
| `SCRIPTORA_SPEECH_MODEL` | `universal-3-5-pro` | AssemblyAI realtime model. |
| `SCRIPTORA_STREAM_MODE` | `balanced` | `min_latency` / `balanced` / `max_accuracy`. Anything else falls back to `balanced`. |
| `SCRIPTORA_CORRECTOR_BACKEND` | `auto` | `auto`, `llm`, or `rules`. |
| `SCRIPTORA_LLM_MODEL` | `qwen3.5-4b-32k-fast` | Gateway model. |
| `SCRIPTORA_FRAME_DURATION_MS` | `100` | Audio frame size sent to AssemblyAI. Clamped to 50-1000, which is what the API accepts. |

The key is read from the environment or a `.env` file, which is git-ignored.
It is never sent to the browser; `/api/health` only reports whether one is set.

## How it works

```
mic ──▶ ScriptProcessorNode ──▶ 16 kHz mono s16le ──▶ 100 ms frames (3200 B)
                                                     │
                                           WebSocket (binary)
                                                     ▼
                                         FastAPI  /ws/audio
                                                     │
                                      per-connection ScriptoraSession
                                                     ▼
                            AssemblyAI Realtime v3  (universal-3-5-pro)
                                          │              │
                        final turn ───────┘              └──── keyterms_prompt
                                          ▼                              ▲
                        is this a spoken command?                        │
                             │                        "Remember X ..." ───┘
                    ┌────────┴────────┐
                    ▼                 ▼
          deterministic rules    LLM Gateway (validates, then adopts)
                    └────────┬────────┘
                             ▼
                   corrected subtitle + event stream ──▶ browser
```

- **Rules first.** A deterministic corrector always runs and always produces a
  safe answer. The LLM is only consulted when it could change the outcome, and
  is adopted only when its output parses, validates against the known subtitle
  IDs, and is actually different. Otherwise the rules win and the result is
  labelled `rules`, not `llm`.
- **A literal edit never calls the model.** "Change X to Y" is already
  implemented exactly by the deterministic path, so the gateway is skipped
  entirely and the edit is applied without a network round trip.
- **The model is consulted off the event loop.** LLM-backed corrections use
  `httpx.AsyncClient`, so a slow or unreachable gateway suspends only the
  pending correction. Audio keeps streaming to AssemblyAI while the request is
  outstanding, and a gateway failure falls back to the rules as before.
- **Vocabulary is pushed to AssemblyAI.** Remembering a term calls
  `set_params` on the *live* session, so the very next sentence is transcribed
  with the term in `keyterms_prompt`. This is the part that changes
  transcription rather than just fixing text after the fact.
- **Commands never become subtitles.** A recognised command is withdrawn from
  the transcript with an explicit `subtitle_removed` event.

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/demo.md`](docs/demo.md) for a demo script and an honest account of what
does and does not work.

## Development

```bash
 pytest                 # 279 tests, no network access
ruff check .
ruff format .
```

The test suite never calls AssemblyAI. Live behaviour was verified separately
against the real API; see `docs/architecture.md` for what that found.

## Tech stack

Python 3.11+ · FastAPI · AssemblyAI Realtime v3 · Pydantic v2 · Jinja2 ·
vanilla JS + ScriptProcessorNode · WebSocket binary frames

## Licence

MIT
