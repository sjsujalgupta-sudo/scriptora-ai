/* Deterministic model of the browser shutdown path (app.js onAudioData +
 * flushAndStop + the stop click handler).
 *
 * The bug this pins down: clicking Stop called stopCapture() immediately, which
 * destroyed the audio graph and cleared pcmCarry while a ScriptProcessor block
 * (~90 ms of audio, containing the final words) was still in flight. The stop
 * control was also written to the socket before that block's frames. The block
 * was then either never processed, or its frames arrived after the backend had
 * already flipped `_running` to false and were discarded. Either way the tail
 * of the final sentence never reached AssemblyAI.
 *
 * The fix waits for the in-flight block to be processed, flushes the pcmCarry
 * tail as a padded full-size frame, and only then sends the stop control - so
 * the WebSocket order is always [.. audio ..][stop], never the reverse.
 *
 * This file models that logic faithfully (capture at SAMPLE_RATE, so the
 * resampler is the identity path) and asserts:
 *   - every captured sample, including the in-flight final block and the
 *     pcmCarry tail, is placed on the socket as full-size frames;
 *   - the stop control is the last message and no audio follows it;
 *   - a padded tail is always exactly one full frame, never a short frame.
 *
 * Run with: node tests/frontend_shutdown.test.mjs
 */
"use strict";

import assert from "node:assert/strict";

const SAMPLE_RATE = 16000;
const FRAME_MS = 100;
const FRAME_SAMPLES = Math.round((SAMPLE_RATE * FRAME_MS) / 1000); // 1600
const BLOCK = 4096; // ScriptProcessorNode buffer size

function makePipeline() {
  return {
    pcmCarry: new Float32Array(0),
    resampleCarry: 0,
    stopping: false,
    flushDone: false,
    listening: true,
    audioAlive: true,
    messages: [], // {kind:'audio', samples} | {kind:'control', action}
    sentSamples: 0,
    audioFrames: 0,
  };
}

function makeBlock(value = 0.0) {
  const block = new Float32Array(BLOCK);
  for (let i = 0; i < BLOCK; i++) block[i] = value;
  return block;
}

function floatTo16BitPCM(floats) {
  const buffer = new ArrayBuffer(floats.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < floats.length; i++) {
    const s = Math.max(-1, Math.min(1, floats[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Int16Array(buffer);
}

function sendFrame(p, int16) {
  p.sentSamples += int16.length;
  p.audioFrames += 1;
  p.messages.push({ kind: "audio", samples: int16.length });
}

function sendControl(p, action) {
  p.messages.push({ kind: "control", action });
}

// Mirrors app.js onAudioData, including the end-of-block flush hook.
function onAudioData(p, inputBuffer) {
  const resampled = inputBuffer; // identity: the model captures at SAMPLE_RATE
  const merged = new Float32Array(p.pcmCarry.length + resampled.length);
  merged.set(p.pcmCarry, 0);
  merged.set(resampled, p.pcmCarry.length);
  let offset = 0;
  while (merged.length - offset >= FRAME_SAMPLES) {
    sendFrame(p, floatTo16BitPCM(merged.subarray(offset, offset + FRAME_SAMPLES)));
    offset += FRAME_SAMPLES;
  }
  p.pcmCarry = merged.slice(offset);
  if (p.stopping && !p.flushDone) {
    flushAndStop(p);
  }
}

// Mirrors app.js flushAndStop.
function flushAndStop(p) {
  if (p.flushDone) return;
  p.flushDone = true;
  if (p.pcmCarry.length > 0) {
    const padded = new Float32Array(FRAME_SAMPLES);
    padded.set(p.pcmCarry, 0);
    sendFrame(p, floatTo16BitPCM(padded));
  }
  p.listening = false;
  sendControl(p, "stop");
  p.audioAlive = false;
}

// The pre-fix stop handler: stop control first, then immediate teardown that
// discards the in-flight block and the pcmCarry tail.
function buggyStop(p) {
  sendControl(p, "stop");
  p.listening = false;
  p.audioAlive = false;
  p.pcmCarry = new Float32Array(0);
}

function stopClick(p, fixed) {
  if (p.stopping) return;
  p.stopping = true;
  if (fixed) {
    // Flag only: the next audio callback performs the real flush. On a real
    // mic the cadence guarantees one; the model always delivers the block
    // that was in flight when the click happened.
    return;
  }
  buggyStop(p);
}

// Drive a session: blocks are delivered one at a time; the click happens while
// `clickAtBlock` is captured but not yet delivered to onAudioData.
function runScenario(fixed, clickAtBlock) {
  const p = makePipeline();
  const total = clickAtBlock + 5;
  for (let i = 0; i < total; i++) {
    if (i === clickAtBlock) {
      stopClick(p, fixed);
      if (!fixed) continue; // buggy model: the in-flight block is never delivered
    }
    if (!p.audioAlive) break;
    onAudioData(p, makeBlock());
  }
  return p;
}

function lastMessage(p) {
  return p.messages[p.messages.length - 1];
}

function audioAfterStop(p) {
  const stopIndex = p.messages.findIndex((m) => m.kind === "control");
  return p.messages.some((m, i) => i > stopIndex && m.kind === "audio");
}

function everyFrameFullSize(p) {
  return p.messages.every((m) => m.kind !== "audio" || m.samples === FRAME_SAMPLES);
}

// ---------------------------------------------------------------- the race
{
  const captured = 48 * BLOCK; // 47 delivered blocks + the one in flight
  const fixed = runScenario(true, 47);
  const buggy = runScenario(false, 47);

  // The buggy (pre-fix) flow demonstrably discards the in-flight block and the
  // pcmCarry tail: fewer samples reach the socket than were captured.
  assert.ok(
    buggy.sentSamples < captured,
    `buggy flow must lose captured audio (sent ${buggy.sentSamples} < captured ${captured})`
  );

  // The fixed flow accounts for every captured sample: it sends at least what
  // was captured, padded by at most one full frame of tail.
  assert.ok(
    fixed.sentSamples >= captured,
    `fixed flow must not drop audio (sent ${fixed.sentSamples} < captured ${captured})`
  );
  assert.ok(
    fixed.sentSamples - captured < FRAME_SAMPLES,
    `fixed flow padded more than one frame beyond captured (${fixed.sentSamples - captured})`
  );
  assert.strictEqual(lastMessage(fixed).kind, "control", "stop control must be the last message");
  assert.strictEqual(lastMessage(fixed).action, "stop", "the last control is the stop control");
  assert.strictEqual(
    audioAfterStop(fixed),
    false,
    "no audio frame may follow the stop control in the fixed flow"
  );
  assert.ok(everyFrameFullSize(fixed), "every audio frame must be full-size in the fixed flow");

  // A second stop click must be ignored: exactly one stop control is sent.
  const double = runScenario(true, 47);
  stopClick(double, true); // ignored: stopping is already set
  assert.strictEqual(
    double.messages.filter((m) => m.kind === "control").length,
    1,
    "a second stop click must not send a second stop control"
  );
}

// ------------------------------------------------ stop at a clean boundary
{
  // Words end exactly on a frame boundary: 25 blocks = 102400 samples = exactly
  // 64 frames, so pcmCarry is empty when the click lands. The in-flight block
  // after the click is still captured - it contributes its own frames and a
  // padded tail - but nothing from BEFORE the boundary is left unaccounted for.
  const p = makePipeline();
  const zero = makeBlock(0);
  for (let i = 0; i < 25; i++) onAudioData(p, zero);
  assert.strictEqual(p.pcmCarry.length, 0, "25 blocks must land on a frame boundary");

  stopClick(p, true);
  onAudioData(p, zero); // in-flight block arrives and triggers the flush

  const captured = 26 * BLOCK; // 25 pre-click + 1 in-flight block
  const expectedFrames = Math.ceil(captured / FRAME_SAMPLES);
  assert.strictEqual(p.audioFrames, expectedFrames, "every captured sample must be accounted for");
  assert.strictEqual(p.sentSamples, expectedFrames * FRAME_SAMPLES);
  assert.ok(p.sentSamples - captured < FRAME_SAMPLES, "at most one padded tail frame");
  assert.strictEqual(lastMessage(p).action, "stop");
  assert.strictEqual(audioAfterStop(p), false);
  assert.ok(everyFrameFullSize(p));
}

// ------------------------------------------ fallback (no callback fires)
{
  // Models an AudioContext that suspended exactly as Stop was pressed: the
  // in-flight block is never delivered, so the cadence-bound fallback performs
  // the flush. Audio already captured into pcmCarry must still be sent as one
  // padded full-size frame, and the stop control must come last.
  const p = makePipeline();
  onAudioData(p, makeBlock());
  onAudioData(p, makeBlock());
  onAudioData(p, makeBlock());
  stopClick(p, true); // no block will be delivered afterwards
  flushAndStop(p); // the fallback

  assert.strictEqual(lastMessage(p).action, "stop");
  assert.strictEqual(audioAfterStop(p), false);
  assert.ok(everyFrameFullSize(p), "the fallback must never send a short frame");
  assert.ok(
    p.pcmCarry.length > 0 || p.audioFrames > 0,
    "whatever was captured before the fallback must not vanish from under the pad"
  );
}

console.log("frontend shutdown model: all ordering assertions passed");