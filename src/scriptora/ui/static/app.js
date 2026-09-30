/* Scriptora browser client.
 *
 * Responsibilities:
 *   1. capture microphone audio
 *   2. resample to mono 16-bit PCM at AssemblyAI's required rate
 *   3. slice it into frames inside AssemblyAI's accepted 50-1000 ms window
 *   4. stream those frames as binary WebSocket messages
 *   5. render the JSON events coming back
 *
 * Mic capture cannot be done in Python in a browser, so this file is the one
 * piece of JavaScript in the project. Everything it emits is consumed by
 * Python services.
 */

(function () {
  "use strict";

  var CONFIG = window.SCRIPTORA_CONFIG || {};
  var SAMPLE_RATE = CONFIG.sample_rate || 16000;
  var FRAME_MS = CONFIG.frame_duration_ms || 100;
  var FRAME_SAMPLES = Math.round((SAMPLE_RATE * FRAME_MS) / 1000);
  var FRAME_BYTES = FRAME_SAMPLES * 2;

  // ---------------------------------------------------------------- elements
  var el = {
    listen: document.getElementById("btn-listen"),
    stop: document.getElementById("btn-stop"),
    statusDot: document.getElementById("status-dot"),
    statusText: document.getElementById("status-text"),
    transcript: document.getElementById("transcript"),
    transcriptEmpty: document.getElementById("transcript-empty"),
    subtitleCount: document.getElementById("subtitle-count"),
    compare: document.getElementById("compare"),
    compareRaw: document.getElementById("compare-raw"),
    compareFixed: document.getElementById("compare-fixed"),
    compareBackend: document.getElementById("compare-backend"),
    vocab: document.getElementById("vocab"),
    termCount: document.getElementById("term-count"),
    vocabForm: document.getElementById("vocab-form"),
    vocabInput: document.getElementById("vocab-input"),
    activity: document.getElementById("activity"),
    clearLog: document.getElementById("btn-clear-log"),
    commandInput: document.getElementById("command-input"),
    commandBtn: document.getElementById("btn-command"),
    waveform: document.getElementById("waveform-wrap"),
    levelFill: document.getElementById("level-fill"),
    metaCorrector: document.getElementById("meta-corrector"),
    metaKey: document.getElementById("meta-key"),
  };

  // ------------------------------------------------------------------- state
  var ws = null;
  var audio = null;          // { ctx, stream, source, processor }
  var listening = false;
  var connected = false;
  var seedCount = 0;         // how many vocabulary chips came from the seed set
  var seenSubtitleIds = {};
  // Ids of lines that are corrections of another line, tracked so the count can
  // exclude them without re-deriving the parent/child relationship each time.
  var correctionIds = {};
  // Lines with a correction in flight, so a busy line can be shown as busy.
  var pendingCorrections = {};
  var logItems = 0;

  // Resampling state. `carry` holds fractional samples across processor calls
  // so nothing is lost between 128-sample AudioWorklet-sized blocks.
  var resampleCarry = 0;
  var pcmCarry = new Float32Array(0);

  // Shutdown sequencing. Stop is a controlled flush: the audio block that is
  // in flight inside the audio graph must be placed on the socket before the
  // stop control, so the server drains it and AssemblyAI can finish the final
  // turn. Destroying the graph in the click handler was discarding that block,
  // which is how the last spoken sentence disappeared from the transcript.
  var stopping = false;
  var flushDone = false;
  var stopFallbackTimer = null;

  // =====================================================================
  // Logging
  // =====================================================================
  function logActivity(message, kind) {
    if (el.activity.firstElementChild &&
        el.activity.firstElementChild.classList.contains("activity-muted") &&
        logItems === 0) {
      el.activity.innerHTML = "";
    }
    logItems += 1;

    var li = document.createElement("li");
    li.className = "activity-item" + (kind ? " activity-" + kind : "");

    var time = document.createElement("span");
    time.className = "activity-time";
    time.textContent = new Date().toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });

    li.appendChild(time);
    li.appendChild(document.createTextNode(message));
    el.activity.appendChild(li);
    el.activity.scrollTop = el.activity.scrollHeight;

    while (el.activity.childElementCount > 120) {
      el.activity.removeChild(el.activity.firstElementChild);
    }
  }

  function setStatus(state, text) {
    el.statusDot.setAttribute("data-state", state);
    el.statusText.textContent = text;
  }

  function enableControls(on) {
    el.listen.disabled = on || !CONFIG.api_key_configured;
    el.stop.disabled = !on;
    el.commandInput.disabled = !on;
    el.commandBtn.disabled = !on;
    el.vocabInput.disabled = !on;
    el.vocabForm.querySelector("button").disabled = !on;
  }

  // =====================================================================
  // Vocabulary panel
  // =====================================================================
  function renderVocabulary(terms) {
    el.vocab.innerHTML = "";
    terms.forEach(function (term, index) {
      var li = document.createElement("li");
      li.textContent = term;
      if (index < seedCount) li.className = "vocab-seed";
      el.vocab.appendChild(li);
    });
    el.termCount.textContent = String(terms.length);
  }

  // =====================================================================
  // Transcript
  // =====================================================================
  function removeSubtitle(sub) {
    if (!sub) return;
    var node = document.getElementById("sub-" + sub.id);
    if (node && node.parentNode) node.parentNode.removeChild(node);
    delete seenSubtitleIds[sub.id];
    delete correctionIds[sub.id];
    delete pendingCorrections[sub.id];
    updateCount();
  }

  function markPending(subtitleId, on) {
    if (!subtitleId) return;
    if (on) {
      pendingCorrections[subtitleId] = true;
    } else {
      delete pendingCorrections[subtitleId];
    }
    var node = document.getElementById("sub-" + subtitleId);
    // The node may not exist yet on the very first correction after a line
    // lands; the correction event that follows renders it without the spinner,
    // which is correct - by then the work is already finished.
    if (node) node.classList.toggle("subtitle-busy", !!on);
  }

  function renderSubtitle(sub, isPartial) {
    if (!sub) return;
    if (el.transcriptEmpty && el.transcriptEmpty.parentNode) {
      el.transcriptEmpty.parentNode.removeChild(el.transcriptEmpty);
      el.transcriptEmpty = null;
    }

    var existing = document.getElementById("sub-" + sub.id);
    var div = existing || document.createElement("div");
    if (!existing) {
      div.id = "sub-" + sub.id;
      seenSubtitleIds[sub.id] = true;
      if (sub.corrects_id) {
        correctionIds[sub.id] = true;
        // Recorded on the node so later insertions can find the end of this
        // line's existing corrections.
        div.dataset.correctsId = sub.corrects_id;
      }
    }

    var cls = "subtitle subtitle-" + (sub.status || (isPartial ? "partial" : "final"));
    // A correction is a child of another line, so it is visually attached to
    // it rather than sitting in the transcript as though it were a new one.
    if (sub.corrects_id) cls += " subtitle-correction";
    div.className = cls;
    div.innerHTML = "";

    var p = document.createElement("p");
    p.className = "subtitle-text";
    p.textContent = sub.text;
    div.appendChild(p);

    var foot = document.createElement("div");
    foot.className = "subtitle-foot";

    var badge = document.createElement("span");
    badge.className = "subtitle-badge";
    badge.textContent = sub.status || (isPartial ? "partial" : "final");
    foot.appendChild(badge);

    if (sub.confidence != null) {
      var conf = document.createElement("span");
      conf.textContent = "conf " + Number(sub.confidence).toFixed(2);
      foot.appendChild(conf);
    }
    if (sub.was_corrected || (sub.raw_text && sub.raw_text !== sub.text)) {
      var edited = document.createElement("span");
      edited.textContent = "corrected";
      foot.appendChild(edited);
    }
    div.appendChild(foot);

    if (!existing) {
      // A correction belongs directly beneath the line it fixes. Its reference
      // names the original, so this needs no server-side ordering.
      var anchor = sub.corrects_id && document.getElementById("sub-" + sub.corrects_id);
      if (anchor && anchor.parentNode) {
        // Step over the corrections this line already has before inserting.
        // Inserting at anchor.nextSibling every time would put each new edit
        // above the last, so repeated fixes would read newest-first and drift
        // away from the line they belong to.
        var ref = anchor.nextSibling;
        while (ref && ref.dataset && ref.dataset.correctsId === sub.corrects_id) {
          ref = ref.nextSibling;
        }
        anchor.parentNode.insertBefore(div, ref);
      } else {
        el.transcript.appendChild(div);
      }
    }
    el.transcript.scrollTop = el.transcript.scrollHeight;

    updateCount();
  }

  function updateCount() {
    // Corrections are not sentences the user spoke. Counting them would make
    // the number climb on every edit, and would no longer match the "sentence
    // 3" an ordinal command resolves to on the server.
    var n = Object.keys(seenSubtitleIds).filter(function (id) {
      return !correctionIds[id];
    }).length;
    el.subtitleCount.textContent = n === 1 ? "1 line" : n + " lines";
  }

  function showComparison(result) {
    if (!result || !result.before || !result.after) return;
    if (result.before === result.after) {
      el.compare.hidden = true;
      return;
    }
    el.compareRaw.textContent = result.before;
    el.compareFixed.textContent = result.after;
    el.compareBackend.textContent = result.backend || "";
    el.compare.hidden = false;
  }

  // =====================================================================
  // Audio capture
  // =====================================================================

  /* Linear-interpolating resampler from the device rate down to SAMPLE_RATE.
   *
   * Scriptora relies on this being correct: a mismatched sample rate does not
   * produce an error from AssemblyAI, it produces garbled text. */
  function resampleToTarget(input, inputRate) {
    if (inputRate === SAMPLE_RATE) return input;

    var ratio = inputRate / SAMPLE_RATE;
    var outLength = Math.floor((input.length + resampleCarry) / ratio);
    if (outLength <= 0) {
      resampleCarry += input.length;
      return new Float32Array(0);
    }

    var out = new Float32Array(outLength);
    var outIndex = 0;
    var pos = -resampleCarry;   // negative position reads the previous block

    for (var i = 0; i < outLength; i++) {
      var exact = pos + i * ratio;
      var idx = Math.floor(exact);
      var frac = exact - idx;

      var s0 = idx >= 0 ? (input[idx] || 0) : 0;
      var s1 = idx + 1 < input.length ? input[idx + 1] : s0;
      out[outIndex++] = s0 + (s1 - s0) * frac;
    }

    var consumed = outLength * ratio;
    resampleCarry = input.length + resampleCarry - consumed;
    if (resampleCarry < 0) resampleCarry = 0;
    return out;
  }

  function floatTo16BitPCM(floats) {
    var buffer = new ArrayBuffer(floats.length * 2);
    var view = new DataView(buffer);
    for (var i = 0; i < floats.length; i++) {
      var s = Math.max(-1, Math.min(1, floats[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Int16Array(buffer);
  }

  function sendFrame(int16) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(int16.buffer);
  }

  function onAudioData(inputBuffer) {
    // Level meter, from the raw (pre-resample) signal.
    var peak = 0;
    for (var i = 0; i < inputBuffer.length; i++) {
      var v = Math.abs(inputBuffer[i]);
      if (v > peak) peak = v;
    }
    el.levelFill.style.width = Math.min(100, peak * 190) + "%";

    var resampled = resampleToTarget(inputBuffer, audio.ctx.sampleRate);
    if (resampled.length === 0) return;

    // Stitch onto the carry, then emit whole FRAME_SAMPLES-sized frames.
    var merged;
    if (pcmCarry.length === 0) {
      merged = resampled;
    } else {
      merged = new Float32Array(pcmCarry.length + resampled.length);
      merged.set(pcmCarry, 0);
      merged.set(resampled, pcmCarry.length);
    }

    var offset = 0;
    while (merged.length - offset >= FRAME_SAMPLES) {
      sendFrame(floatTo16BitPCM(merged.subarray(offset, offset + FRAME_SAMPLES)));
      offset += FRAME_SAMPLES;
    }
    pcmCarry = merged.slice(offset);

    // A Stop requested while a block was in flight is honoured here, at the
    // block boundary: this is the last captured block, so send any complete
    // frames it produced, flush the pcmCarry tail, and only then send the stop
    // control. That ordering is the fix - the stop control must never overtake
    // audio that the microphone has already captured.
    if (stopping && !flushDone) {
      flushAndStop();
    }
  }

  async function startCapture() {
    if (audio) return;

    // These constraints matter for subtitle quality: without echo cancellation
    // and noise suppression the transcript picks up room noise and playback.
    var stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    var Ctx = window.AudioContext || window.webkitAudioContext;
    var ctx = new Ctx();
    if (ctx.state === "suspended") await ctx.resume();

    var source = ctx.createMediaStreamSource(stream);

    // ScriptProcessorNode is deprecated but is the only node available without
    // shipping a separate AudioWorklet module, and it is supported in every
    // current browser. Reliability matters more than modernity for the demo.
    var processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = function (event) {
      try {
        onAudioData(event.inputBuffer.getChannelData(0));
      } catch (err) {
        console.error("Audio processing error", err);
      }
    };

    source.connect(processor);
    // Route to a muted gain node: ScriptProcessorNode only runs when connected
    // to a destination, but we must not echo the mic to the speakers.
    var mute = ctx.createGain();
    mute.gain.value = 0;
    processor.connect(mute);
    mute.connect(ctx.destination);

    audio = { ctx: ctx, stream: stream, source: source, processor: processor, mute: mute };
  }

  function stopCapture() {
    if (!audio) return;
    try { audio.source.disconnect(); } catch (e) { /* already gone */ }
    try { audio.processor.disconnect(); } catch (e) { /* already gone */ }
    try { audio.mute.disconnect(); } catch (e) { /* already gone */ }
    audio.stream.getTracks().forEach(function (t) { t.stop(); });
    audio.ctx.close().catch(function () { /* ignore */ });
    audio = null;
    pcmCarry = new Float32Array(0);
    resampleCarry = 0;
    el.levelFill.style.width = "0%";
  }

  // Finish a stop at a clean audio boundary. Every complete frame produced so
  // far is already on the socket (onAudioData sent each full frame as soon as
  // it was stitched); the only remaining audio is the pcmCarry tail, which is
  // under one frame and would otherwise be silently dropped. Pad it to a
  // full-size frame - never send a short one, AssemblyAI rejects frames under
  // 50 ms - then send the stop control AFTER the audio, and only then release
  // the capture pipeline. The WebSocket stays open so the server's final
  // subtitle and session_ended events still arrive.
  function flushAndStop() {
    if (flushDone) return;
    flushDone = true;
    if (stopFallbackTimer !== null) {
      clearTimeout(stopFallbackTimer);
      stopFallbackTimer = null;
    }
    if (pcmCarry.length > 0) {
      var padded = new Float32Array(FRAME_SAMPLES);
      padded.set(pcmCarry, 0);
      sendFrame(floatTo16BitPCM(padded));
    }
    listening = false;
    sendControl("stop");
    setStatus("idle", "Stopping...");
    stopCapture();
  }

  // =====================================================================
  // WebSocket
  // =====================================================================
  function openSocket() {
    return new Promise(function (resolve, reject) {
      var proto = location.protocol === "https:" ? "wss:" : "ws:";
      var socket = new WebSocket(proto + "//" + location.host + "/ws/audio");
      socket.binaryType = "arraybuffer";

      var settled = false;
      socket.onopen = function () {
        connected = true;
        logActivity("Connected to Scriptora.", "ok");
        resolve(socket);
      };
      socket.onerror = function () {
        if (!settled) { settled = true; reject(new Error("WebSocket connection failed")); }
      };
      socket.onclose = function () {
        connected = false;
        if (listening) {
          setStatus("error", "Disconnected");
          logActivity("Connection to Scriptora was lost.", "err");
        }
        teardown();
      };
      socket.onmessage = function (event) {
        var msg;
        try {
          msg = JSON.parse(event.data);
        } catch (e) {
          return;
        }
        handleEvent(msg);
      };
    });
  }

  function sendControl(action, extra) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    var payload = { action: action };
    if (extra) Object.keys(extra).forEach(function (k) { payload[k] = extra[k]; });
    ws.send(JSON.stringify(payload));
  }

  function handleEvent(msg) {
    switch (msg.type) {
      case "hello":
        seedCount = (msg.payload && msg.payload.vocabulary) ? msg.payload.vocabulary.length : 0;
        if (msg.payload && msg.payload.vocabulary) renderVocabulary(msg.payload.vocabulary);
        break;

      case "session_started":
        logActivity("Session started. Speaking now...");
        setStatus("listening", "Connecting...");
        break;

      case "status":
        setStatus(listening ? "listening" : "idle", msg.message || "");
        break;

      case "subtitle":
        if (msg.subtitle) renderSubtitle(msg.subtitle, msg.payload && msg.payload.partial);
        break;

      case "subtitle_removed":
        // A spoken command reaches the transcript before we recognise it as a
        // command, so the server withdraws the line afterwards. Without this the
        // command would sit in the transcript for good.
        removeSubtitle(msg.subtitle);
        break;

      case "context":
        if (msg.payload && msg.payload.vocabulary) renderVocabulary(msg.payload.vocabulary);
        break;

      case "correction_pending":
        // The gateway call is the slow part of a correction. Showing the line as
        // busy immediately is what stops the UI looking frozen while the user
        // keeps talking.
        markPending(msg.payload && msg.payload.subtitle_id, true);
        logActivity(msg.message, null);
        break;

      case "correction": {
        var r = msg.payload || {};
        // Always clear the busy state, whatever the outcome: a refusal or a
        // no-action is still the end of the wait.
        markPending(r.subtitle_id, false);
        if (msg.subtitle) renderSubtitle(msg.subtitle, false);
        if (r.outcome === "applied") {
          logActivity(msg.message + (r.reason ? " - " + r.reason : ""), "ok");
          showComparison(r);
        } else if (r.outcome === "no_action") {
          // Show the reason too. "No correction was needed" on its own reads
          // like the subtitle is correct, when the usual cause is that the
          // model had nothing to correct *towards*.
          logActivity(msg.message + (r.reason ? " - " + r.reason : ""), null);
        } else {
          logActivity(msg.message, "err");
        }
        break;
      }

      case "activity":
        logActivity(msg.message, null);
        break;

      case "error":
        logActivity(msg.message, "err");
        setStatus("error", "Error");
        if (msg.payload && msg.payload.fatal) {
          teardown();
          closeSocket();
        }
        break;

      case "session_ended":
        logActivity(msg.message, null);
        setStatus("idle", "Stopped");
        // The final subtitle (if any) arrived before this event, so tearing the
        // capture down and closing the socket can no longer lose transcript.
        teardown();
        closeSocket();
        break;
    }
  }

  // =====================================================================
  // Lifecycle
  // =====================================================================
  // Close the socket only once the server's final events have arrived (or the
  // session failed and there is nothing left to receive). Closing early is what
  // would race the final subtitle; closing late just leaks an idle connection
  // on the server until the next Start.
  function closeSocket() {
    if (!ws) return;
    var socket = ws;
    ws = null;
    try {
      if (socket.readyState === WebSocket.OPEN) socket.close(1000, "session closed");
    } catch (e) { /* already closing */ }
  }

  function teardown() {
    listening = false;
    stopCapture();
    el.waveform.hidden = true;
    el.listen.classList.remove("btn-live");
    el.listen.querySelector(".btn-label").textContent = "Start Listening";
    enableControls(false);
  }

  el.listen.addEventListener("click", async function () {
    el.listen.disabled = true;
    setStatus("connecting", "Requesting microphone...");

    try {
      await startCapture();
    } catch (err) {
      setStatus("error", "Microphone blocked");
      logActivity(
        "Could not access the microphone. Allow microphone permission for this " +
        "page, then press Start Listening again.",
        "err"
      );
      enableControls(false);
      return;
    }

    try {
      ws = await openSocket();
    } catch (err) {
      stopCapture();
      setStatus("error", "Server unreachable");
      logActivity(
        "Could not reach the Scriptora server. Is it still running?", "err"
      );
      enableControls(false);
      return;
    }

    listening = true;
    stopping = false;
    flushDone = false;
    if (stopFallbackTimer !== null) {
      clearTimeout(stopFallbackTimer);
      stopFallbackTimer = null;
    }
    el.waveform.hidden = false;
    el.listen.classList.add("btn-live");
    el.listen.querySelector(".btn-label").textContent = "Listening...";
    enableControls(true);
    setStatus("connecting", "Connecting to AssemblyAI...");
    sendControl("start");
  });

  el.stop.addEventListener("click", function () {
    if (stopping) return;
    stopping = true;
    if (!audio) {
      // Nothing was ever captured, or capture already failed: there is no
      // audio to flush. If a server session is live, end it; otherwise just
      // restore the UI.
      if (!listening) { teardown(); return; }
      flushAndStop();
      return;
    }
    // A ScriptProcessor block (~90 ms) is usually in flight when the button is
    // clicked. Rather than destroying the graph immediately - which drops that
    // block and the captured tail - let the next audio callback perform the
    // flush at the block boundary. The fallback only fires if the AudioContext
    // suspended at the exact moment Stop was pressed, and is bounded by the
    // block cadence, not a stall: a suspended context is the one case where a
    // final block can no longer be delivered at all.
    stopFallbackTimer = setTimeout(flushAndStop, 600);
  });

  el.clearLog.addEventListener("click", function () {
    el.activity.innerHTML = "";
    logItems = 0;
    el.activity.innerHTML =
      '<li class="activity-item activity-muted">Waiting...</li>';
  });

  function runCommand() {
    var text = el.commandInput.value.trim();
    if (!text) return;
    sendControl("command", { text: text });
    el.commandInput.value = "";
  }

  el.commandBtn.addEventListener("click", runCommand);
  el.commandInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") runCommand();
  });

  el.vocabForm.addEventListener("submit", function (e) {
    e.preventDefault();
    var term = el.vocabInput.value.trim();
    if (!term) return;
    sendControl("add_term", { term: term });
    el.vocabInput.value = "";
  });

  window.addEventListener("beforeunload", function () {
    if (listening && ws) sendControl("stop");
  });

  // =====================================================================
  // Boot
  // =====================================================================
  el.metaKey.textContent = CONFIG.api_key_configured ? "configured" : "missing";
  el.metaKey.className = CONFIG.api_key_configured ? "meta-ok" : "meta-err";

  el.metaCorrector.textContent = CONFIG.llm_model
    ? CONFIG.llm_model + " (LLM Gateway)"
    : "rules (offline)";
  if (CONFIG.llm_model) el.metaCorrector.className = "meta-ok";

  enableControls(false);
  setStatus("idle", CONFIG.api_key_configured ? "Ready" : "Key required");
  logActivity("Scriptora ready. Press Start Listening.");
})();
